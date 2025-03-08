from src import llm, utils, config
from functools import partial
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, PreTrainedModel
from peft import LoraConfig
from trl import CPOConfig, ModelConfig, ScriptArguments, get_peft_config, setup_chat_format
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import os

from mcpo import MCPOTrainer

# parameters
num_train_epochs = 5
learning_rate = 2e-4
max_completion_length = 10
max_prompt_length = 2048
max_length = max_prompt_length + max_completion_length
gradient_accumulation_steps = 1
k_folds = 5  # Number of cross-validation folds

metadata = [
    ["assets/sft-phi-recovery", llm.load_phi_llm, 'ReCOVery', "Phi", 1],
    ["assets/sft-phi-fakehealth", llm.load_phi_llm, 'FakeHealth', "Phi", 1],
]

for model_path, load_callback, dataset_name, model_name, per_device_train_batch_size in metadata:
    for model_loss in ['sigmoid_bco']:
        train_set, test_set = utils.load_datasets(path="dataset", dataset_name=dataset_name)
        prompt_template = config.standardized_prompt_template
        
        def prepare_dataset(df):
            data = []
            for title, news, label in zip(df['title'], df['news'], df['claim']):
                neg_label = 'real' if label == 'fake' else 'fake'
                conversation = {
                    "prompt": [{"role": "system", "content": config.instruction},
                               {"role": "user", "content": prompt_template.replace("{TITLE}", str(title)).replace("{NEWS}", str(news))}],
                    "chosen": [{"role": "assistant", "content": label}],
                    "rejected": [{"role": "assistant", "content": neg_label}],
                }
                data.append(conversation)
            return data
        
        dataset_df = pd.concat([train_set, test_set], ignore_index=True)
        labels = [1 if claim == "fake" else 0 for claim in dataset_df['claim'].tolist()]
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_result_dir = f"results/{dataset_name}-CV-{model_name}"
        
        if not os.path.exists(cv_result_dir):
            os.mkdir(cv_result_dir)
            
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset_df)), labels)):
            tokenizer, model, llm_path = load_callback(llm_path=model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.chat_template is None:
                model, tokenizer = setup_chat_format(model, tokenizer)

            output_dir = f"assets/{dataset_name}-CV-{model_name}/rlhf-{model_name.lower()}-{dataset_name.lower()}-cpo-{model_loss.replace('_', '-')}-fold{fold+1}"
            print(f"\nTraining Fold {fold + 1}/{k_folds}...\n")

            train_data = dataset_df.iloc[train_idx]
            val_data = dataset_df.iloc[val_idx]
            dataset = prepare_dataset(train_data)
            dataset_lst = Dataset.from_list(dataset)
            
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.5,
                target_modules=["q_proj", "v_proj"] if model_name != "Phi" else ['o_proj', 'qkv_proj'],
                bias="none",
                task_type="CAUSAL_LM",
            )
            training_args = CPOConfig(
                output_dir=output_dir,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                max_length=max_length,
                max_completion_length=max_completion_length,
                max_prompt_length=max_prompt_length,
                gradient_checkpointing_kwargs={"use_reentrant": True},
                remove_unused_columns=False,
                loss_type=model_loss
            )
            trainer = MCPOTrainer(
                model,
                args=training_args,
                train_dataset=dataset_lst,
                processing_class=tokenizer,
                peft_config=peft_config,
            )
            trainer.train()
            trainer.save_model(output_dir)

            del trainer
            print("Inferencing on Validation Set:")

            # if dataset_name == 'FakeHealth':
                # prompt_template = config.instruction + '\n' + config.standardized_prompt_template

            tokenizer, model, llm_path = load_callback(llm_path=output_dir)

            def standard_prompting(test, tokenizer, model, prompt_template, batch_size=1, max_new_tokens=10):
                test_data = llm.MISSINFODataset(tokenizer=tokenizer, df=test, prompt_template=prompt_template)
                test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                predicts, labels = llm.make_the_generation(model=model, tokenizer=tokenizer, 
                                                           data_loader=test_dataloader, max_new_tokens=max_new_tokens)
                processed_predicts = utils.output_processor(predicts)
                clf_report = utils.evaluation_report(y_true=labels, y_pred=processed_predicts)
                return predicts, processed_predicts, clf_report

            predicts, processed_predicts, clf_report = standard_prompting(test=val_data, 
                                                                          tokenizer=tokenizer, 
                                                                          model=model, 
                                                                          prompt_template=prompt_template)

            report_dict = {
                "llm": llm_path,
                "clf_report": clf_report,
                "generations": predicts,
                "processed_generations": processed_predicts,
            }
            
            val_data.to_csv(f"{cv_result_dir}/validation-fold{fold+1}.csv")
            train_data.to_csv(f"{cv_result_dir}/train-fold{fold+1}.csv")

            utils.write_json(data=report_dict, path=f"{cv_result_dir}/{model_name}-{dataset_name}-rlhf-cpo-{model_loss.replace('_', '-')}-fold{fold+1}.json")

            del model
            del tokenizer
