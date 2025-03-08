from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
import os

_ = load_dotenv(find_dotenv())

class MISSINFODataset(Dataset):
    def __init__(self, tokenizer, df, prompt_template):
        self.title = df['title'].tolist()
        self.news = df['news'].tolist()
        self.labels = df['claim'].tolist()
        self.prompt_template = prompt_template
        self.tokenizer = tokenizer

    def build_message(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant in misinformation detection within healthcare news article. You only output 'Fake' word for fake news and 'Real' word for correct/real news."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idxs):
        inputs, inputs_labels = [], []
        for idx in idxs:
            prompt = self.prompt_template.replace("{TITLE}", str(self.title[idx])).replace("{NEWS}", str(self.news[idx]))
            inputs.append(self.build_message(prompt))
            inputs_labels.append(self.labels[idx])
        return {"inputs":inputs, "labels":inputs_labels}
    

def make_the_generation(model, tokenizer, data_loader, max_new_tokens=512):
    gen_texts, labels = [], []

    for batch in tqdm(data_loader):
        input_data = batch['inputs']
        labels += batch['labels']
        tokenized_input_data = tokenizer(input_data, max_length=4048, return_tensors="pt").to(model.device)
        # tokenized_input_data = tokenizer(input_data, return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **tokenized_input_data,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_input_data.input_ids, generated_ids)]

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        gen_texts += [generated_texts]
    return gen_texts, labels

def load_llm(llm_path):
    tokenizer = AutoTokenizer.from_pretrained(llm_path,  padding_side='left')

    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with 4-bit precision
    model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=quant_config,
                                                 device_map={"": 0})
    return tokenizer, model, llm_path

def load_qwen_llm(llm_path="Qwen/Qwen2.5-0.5B-Instruct"):
    return load_llm(llm_path)

def load_falcon_llm(llm_path = "tiiuae/Falcon3-3B-Instruct"):
    return load_llm(llm_path)

def load_phi_llm(llm_path="microsoft/Phi-3.5-mini-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(llm_path, 
                                                quantization_config=quant_config, 
                                                # attn_implementation="flash_attention_2",
                                                _attn_implementation="eager",
                                                trust_remote_code=True,
                                                device_map={"": 0})
    return tokenizer, model, llm_path

def load_llama_llm(llm_path="meta-llama/Llama-3.2-1B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(llm_path,  padding_side='left', token=os.environ['HUGGINGFACE_ACCESS_TOKEN'])

    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with 4-bit precision
    model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=quant_config,
                                                 device_map={"": 0},  token=os.environ['HUGGINGFACE_ACCESS_TOKEN'])
    return tokenizer, model, llm_path