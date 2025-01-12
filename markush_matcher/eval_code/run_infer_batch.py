import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import AutoTokenizer, T5ForConditionalGeneration
from train_group_pred import make_data
import json
from rouge_score import rouge_scorer
import numpy as np
from nltk.translate.meteor_score import meteor_score
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import textdistance
from concurrent.futures import ProcessPoolExecutor
import ast
import multiprocessing
import random
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
data_path = '../data/test.json'
model_path = 'PATH_TO_MODEL'
tokenizer = AutoTokenizer.from_pretrained(model_path)

from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from run_gpt4o import analysis_substituent
from Levenshtein import distance as lev_distance
from run_infer import Metric_calculator

def predict(input_text, model, tokenizer, max_length=512):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)

    device = next(model.parameters()).device
    output_ids = model.generate(input_ids.to(device), max_length=max_length, eos_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

def process_data(input_data):
    texts = make_data(input_data)
    input_text = texts['input_text']
    target_text = texts['target_text']
    model = input_data['model']
    pred_text = predict(input_text, model, tokenizer)
    pred_text = pred_text.replace('{ ', '{')
    pred_text = pred_text.replace('""', '"')
    return pred_text, target_text

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument('--root', type=str, default='batch_match_results/')
    parser.add_argument('--mode', choices=['match_model', 'gpt_4o', 'rdkit'], default='match_model')
    args = parser.parse_args()
    return args
def compute_simi(mol1, mol2):
    mol1 = Chem.MolFromSmiles(mol1)
    mol2 = Chem.MolFromSmiles(mol2)
    if mol1 is None or mol2 is None:
        return 0, 0
    fps1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2,nBits=1024)
    fps2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2,nBits=1024)
    simi = DataStructs.FingerprintSimilarity(fps1, fps2)
    tanimoto_sim = DataStructs.TanimotoSimilarity(fps1, fps2)
    return simi, tanimoto_sim
    
if __name__ == '__main__':
    args = parse_args()
    metrics = Metric_calculator()
    with open(data_path, 'r') as f:
        data = json.load(f)
    predict_list = []
    target_list = []
    idx = 0
    num_workers = 16
    if not os.path.exists(args.root):
        os.makedirs(args.root, exist_ok=True)
    available_gpus = [_ for _ in range(torch.cuda.device_count())]
    print(f'Available GPUs: {available_gpus}', flush=True)
        
    for step_num in range(1, 26):
        step_string = f'{step_num}000'
        print(f'\n\n\n\nProcessing step {step_string}', flush=True)
        model_path = f'PATH_TO_MODEL/checkpoint-{step_string}'
        model_list = []
        for gpu_id in available_gpus:
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(f'cuda:{gpu_id}')
            model.eval()
            model_list.append(model)
        multiprocessing.set_start_method('spawn', force=True)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_input = {executor.submit(process_data, {**input_data, 'model': random.choice(model_list)}): input_data for input_data in data}

            predict_list = []
            target_list = []
            for future in as_completed(future_to_input):
                input_data = future_to_input[future]
                try:
                    result = future.result()
                    if result is not None:
                        pre, target = result
                        if pre is not None and target is not None:
                            predict_list.append(pre)
                            target_list.append(target)
                except Exception as exc:
                    print(f'Error in processing data: {exc}', flush=True)
                    raise exc
        json.dump(predict_list, open(os.path.join(args.root, f'{step_string}-predict.json'), 'w'))
        json.dump(target_list, open(os.path.join(args.root, f'{step_string}-target.json'), 'w'))
    
        res = metrics(target_list, predict_list, use_tokenizer=True, verbose=True)
        with open(os.path.join(args.root, f'{step_string}-metrics.json'), 'w') as f:
            json.dump(res, f, indent=4)
        
        print(f'Finished step {step_string}', flush=True)