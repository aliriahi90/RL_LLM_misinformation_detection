from transformers import BertTokenizer, LongformerForSequenceClassification, BertForSequenceClassification, LongformerTokenizer, TrainingArguments, Trainer, BertModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from src import llm, utils, config
import torch

# Parameters
num_train_epochs = 5
batch_size = 8

metadata = [
    ['bert-base-uncased','FakeHealth', 'bert-base', 512, batch_size*4],
    ['bert-base-uncased','ReCOVery',  'bert-base', 512, batch_size*4],
    
    ['dhruvpal/fake-news-bert','FakeHealth', 'fake-news-bert', 512, batch_size*4],
    ['dhruvpal/fake-news-bert','ReCOVery', 'fake-news-bert', 512, batch_size*4],
]

label2id = {'fake':0, 'real': 1}
id2label = {0: 'fake', 1:'real'}


for finetuning_metadata in metadata:
    
    model_id, dataset_name, model_name, max_length, batch_size = finetuning_metadata
    output_path =  f"assets/{model_name}-{dataset_name}"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    
    # Load dataset
    train_set, test_set = utils.load_datasets(path="dataset", dataset_name=dataset_name)

    # Prepare datasets
    train = Dataset.from_pandas(train_set[['title', 'news', 'claim']])
    test = Dataset.from_pandas(test_set[['title', 'news', 'claim']])

    def preprocess_data(examples):
        inputs = tokenizer(
            [f"Title: {str(title)}. News: {str(news)}" for title, news in zip(examples['title'], examples['news'])],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        labels = [label2id[claim.lower()] for claim in examples['claim']]
        inputs['labels'] = labels
        return inputs

    train = train.map(preprocess_data, batched=True, remove_columns=['title', 'news', 'claim'])
    test = test.map(preprocess_data, batched=True, remove_columns=['title', 'news', 'claim'])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        logging_steps=10,
        learning_rate=2e-5,
        save_steps=1000,
        weight_decay=0.01,
        report_to='tensorboard'
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        # tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_path)

    # Evaluation
    predictions = trainer.predict(test)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    labels = [id2label[label] for label in labels]
    preds = [id2label[label] for label in preds]

    # Report results
    from sklearn.metrics import classification_report
    report = classification_report(labels, preds, output_dict=True, digits=4)
    print(report)

    # Save results
    utils.write_json(data=report, path=f"results/{model_name}-{dataset_name}-bert-finetuning.json")
    print(f"Finished finetuning BERT on {dataset_name}")
    
