import argparse
import yaml
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import torch
from rag_util import create_embeddings, build_faiss_index, retrieve_documents, load_data, preprocess_data, build_index_json, retrieve_by_company_name, load_data_json

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_model(model_name, models_config):
    model_details = models_config[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_details['model_name'])

    if model_name == 'bert':
        model = AutoModelForMaskedLM.from_pretrained(model_details['model_name'])
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_details['model_name'], device_map="auto", pad_token_id=tokenizer.eos_token_id)
        tokenizer.pad_token = tokenizer.eos_token
    
    # model.to(device)  # Ensure the model is on the correct device
    

    return tokenizer, model


parser = argparse.ArgumentParser(description="Run financial analysis script with specified pipeline.")
parser.add_argument('--pipeline', type=str, default='financial_news', help='Pipeline name (e.g., "earnings_calls", "financial_news")')

args = parser.parse_args()

config = load_config()
pipeline_config = config['pipelines'].get(args.pipeline, config['pipelines'][config['default_pipeline']])
models_config = config['models']

device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"

# input_file_path = 'news_sample.txt'
input_file_path = 'earnings_call_sample.json'

data = load_data(input_file_path)
documents = preprocess_data(data)

tokenizer, model = setup_model(pipeline_config['model'], models_config)

# embeddings = create_embeddings(model, tokenizer, documents, device)
# index = build_faiss_index(embeddings)

def generate_output(tokenizer, model):
    query = "Should I invest in NVIDIA right now? And Why?"
    # query = "What is the "
    # query_embedding = create_embeddings(model, tokenizer, ["Company: Bausch + Lomb Corporation"], device)
    # doc_indices = retrieve_documents(index, query_embedding)
    # relevant_docs = " ".join(documents[idx] for idx in doc_indices)

    data = load_data_json('earnings_call_sample.json')
    index = build_index_json(data)
    
    results = retrieve_by_company_name(index, "NVIDIA")

    print(results)

    prompt = f"{query} Context: {results}" 

    # print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    # outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
    outputs = model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(text)

generate_output(tokenizer, model)

output_file_path = f'Generations/{args.pipeline}/sample_output.txt'
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(text)