from src import llm, utils, config

from peft import LoraConfig
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from datasets import Dataset
from torch.utils.data import DataLoader

# parameters
num_train_epochs = 5



metadata = [
    ["assets/sft-llama-fakehealth", llm.load_llama_llm, 'FakeHealth', "Llama3", 4], 
    ["assets/sft-llama-recovery", llm.load_llama_llm, 'ReCOVery', "Llama3", 4], 
    ["assets/sft-phi-fakehealth", llm.load_phi_llm, 'FakeHealth', "Phi", 2],
    ["assets/sft-phi-recovery", llm.load_phi_llm, 'ReCOVery', "Phi", 2],
    ["assets/sft-falcon-fakehealth", llm.load_falcon_llm, 'FakeHealth', "Falcon", 2],
    ["assets/sft-falcon-recovery", llm.load_falcon_llm, 'ReCOVery', "Falcon", 2],
    ["assets/sft-qwen-fakehealth", llm.load_qwen_llm, 'FakeHealth', "Qwen", 4],
    ["assets/sft-qwen-recovery", llm.load_qwen_llm, 'ReCOVery', "Qwen", 4],
]

for finetuning_metadata in metadata:
    output_dir = finetuning_metadata[0]
    tokenizer, model, llm_path = finetuning_metadata[1]()
    dataset_name = finetuning_metadata[2]
    model_name = finetuning_metadata[3]
    batch_size = finetuning_metadata[4]

    train_set, test_set = utils.load_datasets(path="dataset", dataset_name=dataset_name)

    train = Dataset.from_pandas(train_set[['title', 'news', 'claim']])

    peft_params = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules= ["q_proj", "v_proj"] if model_name != "Phi" else ['o_proj', 'qkv_proj'],  
        bias="none",
        task_type="CAUSAL_LM"
    )

    prompt_template = config.standardized_prompt_template
    
    def preprocess_chat_data(examples):
        input_ids = []
        attention_masks = []
        labels = []

        for title, news, label in zip(examples['title'], examples['news'], examples['claim']):
            conversation = [
                {"role": "system", "content": config.instruction},
                {"role": "user", "content": prompt_template.replace("{TITLE}", str(title)).replace("{NEWS}", news)},
                {"role": "assistant", "content": label}
            ]
            # Format the conversation into a string
            formatted_input = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            # Tokenize the input conversation
            tokenized_input = tokenizer(formatted_input, truncation=True, max_length=2048, padding="max_length")
            # Tokenize the label
            tokenized_label = tokenizer(label, truncation=True, max_length=10, padding="max_length")
            # Append results to the respective lists
            input_ids.append(tokenized_input["input_ids"])
            attention_masks.append(tokenized_input["attention_mask"])
            labels.append(tokenized_label["input_ids"])

        # Return all tokenized outputs as a dictionary
        return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}


    preprocess_train = train.map(preprocess_chat_data, batched=True, remove_columns=['title', 'news', 'claim'])

    training_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=batch_size,
        optim="paged_adamw_8bit",
        save_steps=500,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        report_to="tensorboard"
    )

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=preprocess_train,
        peft_config=peft_params,
        # dataset_text_field="input_ids",  # Specify the correct field for text
        # max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_params,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    
    
    print("Inferencing:")
    prompt_template = config.standardized_prompt_template
    
    tokenizer, model, llm_path = finetuning_metadata[1](llm_path=output_dir)
    
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

    utils.write_json(data=report_dict, path=f"results/{model_name}-{dataset_name}-supervised-finetuning.json")




