from src import llm, utils, config
from functools import partial

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, PreTrainedModel
from peft import LoraConfig
from trl import BCOConfig, BCOTrainer, ModelConfig, ScriptArguments, get_peft_config, setup_chat_format
from torch.utils.data import DataLoader


def embed_prompt(input_ids: torch.LongTensor, attention_mask: torch.LongTensor, model: PreTrainedModel):
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    model_output = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = average_pool(model_output.last_hidden_state, attention_mask)
    return embeddings



# parameters
num_train_epochs = 5
learning_rate = 2e-4
max_completion_length = 10
max_prompt_length = 2048
max_length = max_prompt_length + max_completion_length
gradient_accumulation_steps = 1


metadata = [
    ["assets/sft-qwen-fakehealth", llm.load_qwen_llm, 'FakeHealth', "Qwen", 4],
    ["assets/sft-qwen-recovery", llm.load_qwen_llm, 'ReCOVery', "Qwen", 4],
    
    ["assets/sft-llama-fakehealth", llm.load_llama_llm, 'FakeHealth', "Llama3", 4], 
    ["assets/sft-llama-recovery", llm.load_llama_llm, 'ReCOVery', "Llama3", 4], 

    ["assets/sft-falcon-fakehealth", llm.load_falcon_llm, 'FakeHealth', "Falcon", 2],
    ["assets/sft-falcon-recovery", llm.load_falcon_llm, 'ReCOVery', "Falcon", 2],
    
    ["assets/sft-phi-fakehealth", llm.load_phi_llm, 'FakeHealth', "Phi", 2],
    ["assets/sft-phi-recovery", llm.load_phi_llm, 'ReCOVery', "Phi", 2],
    
]


for model_path, load_callback, dataset_name, model_name, per_device_train_batch_size in metadata:

    output_dir = f"assets/rlhf-{model_name.lower()}-{dataset_name.lower()}-bco"
    train_set, test_set = utils.load_datasets(path="dataset", dataset_name=dataset_name)
    prompt_template = config.standardized_prompt_template
    
    tokenizer, model, llm_path = load_callback(llm_path=model_path)
    _, ref_model, _ = load_callback(llm_path=model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    def prepare_dataset(df):
        data = []
        for title, news, label in zip(df['title'], df['news'], df['claim']):
            positive_conversation = {
                "prompt": [{"role": "system", "content": config.instruction},
                           {"role": "user", "content": prompt_template.replace("{TITLE}", str(title)).replace("{NEWS}", news)}],
                "completion": [{"role": "assistant", "content": label}],
                "label": True
            }

            if label == 'fake':
                neg_label = 'real'
            else:
                neg_label = 'fake'

            negative_conversation = {
                "prompt": [{"role": "system", "content": config.instruction},
                           {"role": "user", "content": prompt_template.replace("{TITLE}", str(title)).replace("{NEWS}", news)}],
                "completion": [{"role": "assistant", "content": neg_label}],
                "label": False
            }
            data.append(positive_conversation)
            data.append(negative_conversation)
        return data

    dataset = prepare_dataset(train_set)
    dataset_lst = Dataset.from_list(dataset)
    
    accelerator = Accelerator()
    embedding_model = AutoModel.from_pretrained(
        "thenlper/gte-base",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    embedding_model = accelerator.prepare_model(embedding_model)
    embedding_tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base", trust_remote_code=True)
    embedding_func = partial(embed_prompt, model=embedding_model)
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.5,
        target_modules=["q_proj", "v_proj"] if model_name != "Phi" else ['o_proj', 'qkv_proj'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = BCOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        # gradient_accumulation_steps = gradient_accumulation_steps,
        # per_gpu_train_batch_size=per_gpu_train_batch_size,
        max_length=max_length,
        max_completion_length = max_completion_length,
        max_prompt_length = max_prompt_length,
        gradient_checkpointing_kwargs = {"use_reentrant": True},
        remove_unused_columns=False,
    )
    trainer = BCOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset_lst,
        processing_class=tokenizer,
        peft_config=peft_config,
        embedding_func=embedding_func,
        embedding_tokenizer=embedding_tokenizer,
    )
    trainer.train()
    
    trainer.save_model(output_dir)
    
    
    del model
    del tokenizer
    del trainer
    
    print("Inferencing:")
    
    tokenizer, model, llm_path = load_callback(llm_path=output_dir)
    def standard_prompting(test, tokenizer, model, prompt_template, batch_size=1, max_new_tokens=10):
        test_data = llm.MISSINFODataset(tokenizer=tokenizer, df=test, prompt_template=prompt_template)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        predicts, labels = llm.make_the_generation(model=model, tokenizer=tokenizer, 
                                                   data_loader=test_dataloader, max_new_tokens=max_new_tokens)
        processed_predicts = utils.output_processor(predicts)
        clf_report = utils.evaluation_report(y_true=labels, y_pred=processed_predicts)
        return predicts, processed_predicts, clf_report

    predicts, processed_predicts, clf_report = standard_prompting(test=test_set, 
                                                                  tokenizer=tokenizer, 
                                                                  model=model, 
                                                                  prompt_template=prompt_template)
    report_dict = {
        "llm": llm_path,
        "clf_report": clf_report,
        "generations": predicts,
        "processed_generations": processed_predicts,
    }

    utils.write_json(data=report_dict, path=f"results/{model_name}-{dataset_name}-rlhf-bco.json")
    
    del model
    del tokenizer

