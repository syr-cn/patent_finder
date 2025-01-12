import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import AutoTokenizer, T5ForConditionalGeneration
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


def make_data(one_data_dict):
    """
    data_sample = {
        "markush": "*n1c(=O)nc2sc(S(N)(=O)=O)nn2c1=O<sep><a>0:R[7]</a>",
        "smiles": "CCCCC(=O)n1c(=O)nc2sc(S(N)(=O)=O)nn2c1=O",
        "r_group_map": {
            "R[7]": "*C(CCCC)=O"
        }
    }
    """
    markush = one_data_dict["markush"]
    smiles = one_data_dict["smiles"]
    r_group_map = json.dumps(one_data_dict["r_group_map"])
    
    input_text = f"Markush: {markush}\t SMILES: {smiles}\t Predict the values of each substituent in markush structure."
    target_text = r_group_map

    return {'input_text': input_text, 'target_text': target_text}


from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from run_gpt4o import analysis_substituent
from Levenshtein import distance as lev_distance
def levenshtein_similarity(truth: List[str], pred: List[str]) -> List[float]:
    assert len(truth) == len(pred)
    scores: List[float] = [
        textdistance.levenshtein.normalized_similarity(t, p)
        for t, p in zip(truth, pred)
    ]
    return scores

def accuracy_score(score_list, threshold):
    matches = sum(score>=threshold for score in score_list)
    acc = matches / len(score_list)
    return acc

def accuracy_score_any(score_list, threshold):
    is_ok = [all(score >= threshold for score in scores) for scores in score_list]
    matches = sum(is_ok)
    acc = matches / len(score_list)
    return acc

class Metric_calculator:
    def __init__(self, text_trunc_length=1024):
        # self.converter = ReadableConverter(separator=' ; ')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side='right')
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.text_trunc_length = text_trunc_length
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    def tokenize(self, gt_list, pred_list):
        references = []
        hypotheses = []
        
        for gt, out in zip(gt_list, pred_list):
            gt_tokens = self.tokenizer.tokenize(gt)
            out_tokens = self.tokenizer.tokenize(out)

            references.append([gt_tokens])
            hypotheses.append(out_tokens)
        return references, hypotheses
    

    def __call__(self, gt_list, pred_list, use_tokenizer=False, verbose=True):
        gt_dict_list = [json.loads(gt) for gt in gt_list]
        pred_dict_list = []
        for pred in pred_list:
            try:
                pred_dict_list.append(json.loads(pred))
            except:
                print('error when parsing json string: ', pred)
                pred_dict_list.append({})
        gt_list = [gt.strip() for gt in gt_list]
        pred_list = [pred.strip() for pred in pred_list]
        # import pdb; pdb.set_trace()
        if use_tokenizer:
            references, hypotheses = self.tokenize(gt_list, pred_list)
            bleu2, bleu4 = self.bleu(references, hypotheses)
            _meteor_score = self.meteor(references, hypotheses)
        else:
            raise NotImplementedError
            bleu2 = modified_bleu(gt_list, pred_list, bleu_n=2)
            bleu4 = modified_bleu(gt_list, pred_list, bleu_n=4)
            _meteor_score = 0
        
        acc_100, acc_90, acc_75, acc_50, lev_score = self.accuracy(gt_list, pred_list)
        lev_dis_list, lev_dis = self.lev_distance(gt_list, pred_list)

        rouge_1, rouge_2, rouge_l = self.rouge(gt_list, pred_list)
        fp_score_list, tanimoto_score_list, fp_score, tani_score, finger_acc_100, finger_acc_90, finger_acc_75, finger_acc_50 = self.fingerprint_accuracy(gt_dict_list, pred_dict_list)
        validity = self.validity(gt_dict_list, pred_dict_list)
        if verbose:
            print('BLEU-2 score:', bleu2)
            print('BLEU-4 score:', bleu4)
            print('Average Meteor score:', _meteor_score)
            print('rouge1:', rouge_1)
            print('rouge2:', rouge_2)
            print('rougeL:', rouge_l)
            print('Levenshtein similarity:', lev_score)
            print('Levenshtein Distance:', lev_dis)
            print('Accuracy at 100%:', acc_100)
            print('Accuracy at 90%:', acc_90)
            print('Accuracy at 75%:', acc_75)
            print('Accuracy at 50%:', acc_50)
            print('Fingerprint Similarity:', fp_score)
            print('Tanimoto Similarity:', tani_score)
            print('Fingerprint Accuracy at 100%:', finger_acc_100)
            print('Fingerprint Accuracy at 90%:', finger_acc_90)
            print('Fingerprint Accuracy at 75%:', finger_acc_75)
            print('Fingerprint Accuracy at 50%:', finger_acc_50)
            print('Validity:', validity)

            line = ''
            # for score in [bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score, lev_score, acc_100, acc_90, acc_75, acc_50, finger_acc_100, finger_acc_90, finger_acc_75, finger_acc_50, validity]:
            for score in [finger_acc_100, validity, fp_score, lev_score, lev_dis, bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score]:
                line += f'{score:.4f} '
            print(line)
        
        return {
            'exact_match': finger_acc_100,
            'validity': validity,
            'fp_score': fp_score,
            'lev_score': lev_score,
            'lev_dis': lev_dis,
            'bleu2': bleu2,
            'bleu4': bleu4,
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l,
            'meteor_score': _meteor_score,
            'acc_100': acc_100,
            'acc_90': acc_90,
            'acc_75': acc_75,
            'acc_50': acc_50,
            'finger_acc_100': finger_acc_100,
            'finger_acc_90': finger_acc_90,
            'finger_acc_75': finger_acc_75,
            'finger_acc_50': finger_acc_50,
            'lev_dis_list': lev_dis_list,
            'fp_score_list': fp_score_list,
            'tanimoto_score_list': tanimoto_score_list,
        }
    
    def get_result_list(self, gt_list, pred_list, use_tokenizer=False):
        gt_list = [gt.strip() for gt in gt_list]
        pred_list = [pred.strip() for pred in pred_list]

        if use_tokenizer:
            references, hypotheses = self.tokenize(gt_list, pred_list)
            bleu2 = [corpus_bleu([gt], [pred], weights=(.5,.5)) for gt, pred in zip(references, hypotheses)]
            bleu4 = [corpus_bleu([gt], [pred], weights=(.25,.25,.25,.25)) for gt, pred in zip(references, hypotheses)]
            _meteor_score = [meteor_score(gt, out) for gt, out in zip(references, hypotheses)]
        else:
            raise NotImplementedError
            bleu2 = [modified_bleu([gt], [pred], bleu_n=2) for gt, pred in zip(gt_list, pred_list)]
            bleu4 = [modified_bleu([gt], [pred], bleu_n=4) for gt, pred in zip(gt_list, pred_list)]
            _meteor_score = 0
        rouge_1, rouge_2, rouge_l = self.rouge(gt_list, pred_list, return_list=True)
        acc_100, acc_90, acc_75, acc_50, lev_score = self.accuracy(gt_list, pred_list)
        
        return {
            'bleu2': bleu2,
            'bleu4': bleu4,
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l,
            'meteor_score': _meteor_score,
            'lev_score': lev_score,
            'acc_100': acc_100,
            'acc_90': acc_90,
            'acc_75': acc_75,
            'acc_50': acc_50,
        }
    
    def bleu(self, references, hypotheses):
        bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
        bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
        bleu2 *= 100
        bleu4 *= 100
        return bleu2, bleu4
    
    def meteor(self, references, hypotheses):
        meteor_scores = []
        for gt, out in zip(references, hypotheses):
            mscore = meteor_score(gt, out)
            meteor_scores.append(mscore)
        _meteor_score = np.mean(meteor_scores)
        _meteor_score *= 100
        return _meteor_score

    def rouge(self, targets, predictions, return_list=False):
        rouge_scores = []
        for gt, out in zip(targets, predictions):
            rs = self.scorer.score(out, gt)
            rouge_scores.append(rs)

        rouge_1 = [rs['rouge1'].fmeasure for rs in rouge_scores]
        rouge_2 = [rs['rouge2'].fmeasure for rs in rouge_scores]
        rouge_l = [rs['rougeL'].fmeasure for rs in rouge_scores]
        if return_list:
            return rouge_1, rouge_2, rouge_l

        rouge_1 = np.mean(rouge_1) * 100
        rouge_2 = np.mean(rouge_2) * 100
        rouge_l = np.mean(rouge_l) * 100
        return rouge_1, rouge_2, rouge_l

    def accuracy(self, gt_list, pred_list):
        score_list = levenshtein_similarity(gt_list, pred_list)
        acc_100 = 100*accuracy_score(score_list, 1.0)
        acc_90 = 100*accuracy_score(score_list, 0.90)
        acc_75 = 100*accuracy_score(score_list, 0.75)
        acc_50 = 100*accuracy_score(score_list, 0.50)
        lev_score = np.mean(score_list)
        return acc_100, acc_90, acc_75, acc_50, lev_score*100

    def lev_distance(self, gt_list, pred_list):
        score_list = [
            lev_distance(gt, pred) for gt, pred in zip(gt_list, pred_list)
        ]
        lev_dis = np.mean(score_list)
        return score_list, lev_dis

    def fingerprint_accuracy(self, gt_list, pred_list):
        '''
        Note here the gt_list and pred list should be the R substitute groups, rather than the whole sentence
        '''
        finger_sim_list = []
        tanimoto_score_list = []
        
        for gt, pred in zip(gt_list, pred_list):
            sim_scores, tanimoto_scores = self.fingerprint_similarity(gt, pred)
            finger_sim_list.append(sim_scores)
            tanimoto_score_list.append(tanimoto_scores)
        # print(finger_sim_list)

        acc_100 = 100*accuracy_score_any(finger_sim_list, 1.0)
        acc_90 = 100*accuracy_score_any(finger_sim_list, 0.90)
        acc_75 = 100*accuracy_score_any(finger_sim_list, 0.75)
        acc_50 = 100*accuracy_score_any(finger_sim_list, 0.50)
        
        score_list = [(np.mean(scores) if scores else 0) for scores in finger_sim_list]
        tanimoto_score_list = [(np.mean(scores) if scores else 0) for scores in tanimoto_score_list]
        fp_score = np.mean(score_list)
        tani_score = np.mean(tanimoto_score_list)

        return score_list, tanimoto_score_list, fp_score*100, tani_score*100, acc_100, acc_90, acc_75, acc_50

    def fingerprint_similarity(self, gt_dict, pred_dict, threshold=0.5):
        gt_keys = set(gt_dict.keys())
        pred_value_list = [pred_dict.get(key, '') for key in gt_keys]
        gt_value_list = [gt_dict[key] for key in gt_keys]
        
        scores = []
        tanimoto_scores = []
        for pred, gt in zip(pred_value_list, gt_value_list):
            if pred == '':
                sim, tanimoto_sim = 0, 0
            else:
                sim, tanimoto_sim = compute_simi(pred, gt)
            scores.append(sim)
            tanimoto_scores.append(tanimoto_sim)
        return scores, tanimoto_scores

    def validity(self, gt_list, pred_list):
        has_keys = []
        is_valid = []
        pred_value_cache = dict()
        
        for gt, pred in zip(gt_list, pred_list):
            gt_keys = set(gt.keys())
            has_keys.append(all(key in pred for key in gt_keys))
            for value in pred.values():
                if value in pred_value_cache:
                    continue
                else:
                    try:
                        mol = Chem.MolFromSmiles(value)
                    except:
                        mol = None
                    pred_value_cache[value] = mol is not None
            is_valid.append(all(pred_value_cache[value] for value in pred.values()))
        valid_ratio = sum([valid and has_key for valid, has_key in zip(is_valid, has_keys)]) / len(gt_list)
        return valid_ratio*100


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
    parser.add_argument('--root', type=str, default='zty_test_res/')
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
    model_list = []
    if args.mode == 'match_model':
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
                    print(f"Generated an exception: {exc}")
    elif args.mode == 'rdkit':
        from substructre_match import query_mol
        
        for input_data in data:
            target = input_data['r_group_map']
            try:
                prediction = query_mol(input_data['markush'], input_data['smiles'])
                assert isinstance(prediction, dict)
            except:
                prediction = {}
            predict_list.append(json.dumps(prediction))
            target_list.append(json.dumps(target))
    elif args.mode == 'gpt_4o':
        multiprocessing.set_start_method('spawn', force=True)

            # Use ProcessPoolExecutor to process data in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_input = {executor.submit(analysis_substituent, input_data): input_data for input_data in data}

            # Collect results as they are completed
            for future in as_completed(future_to_input):
                input_data = future_to_input[future]
                try:
                    result = future.result()
                    if result is not None:
                        pre, target = result
                        
                        if pre is not None and target is not None:
                            predict_list.append(json.dumps(pre))
                            target_list.append(json.dumps(target))
                except Exception as exc:
                    print(f'Error processing input: {input_data}. Exception: {exc}')
    json.dump(predict_list, open(os.path.join(args.root, 'predict.json'), 'w'))
    json.dump(target_list, open(os.path.join(args.root, 'target.json'), 'w'))
    
    res = metrics(target_list, predict_list, use_tokenizer=True)
    print(res)
    with open(os.path.join(args.root, 'metrics.json'), 'w') as f:
        json.dump(res, f, indent=4)
