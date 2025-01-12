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
from concurrent.futures import ProcessPoolExecutor, as_completed
# data_path = 'data/data-sft-Nov2.json'
data_path = 'data/data-Nov6-top1k.json'
model_path = 'results/pretrain-large-dataNov6-Nov06/final/'
model_path = 'results/pretrain-large-dataNov6-Nov08/final/'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to('cuda')
from typing import List


model.eval()
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
    

    def __call__(self, gt_list, pred_list, use_tokenizer=False):
        gt_list = [gt.strip() for gt in gt_list]
        pred_list = [pred.strip() for pred in pred_list]

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
        rouge_1, rouge_2, rouge_l = self.rouge(gt_list, pred_list)

        print('BLEU-2 score:', bleu2)
        print('BLEU-4 score:', bleu4)
        print('Average Meteor score:', _meteor_score)
        print('rouge1:', rouge_1)
        print('rouge2:', rouge_2)
        print('rougeL:', rouge_l)
        print('Levenshtein similarity:', lev_score)
        print('Accuracy at 100%:', acc_100)
        print('Accuracy at 90%:', acc_90)
        print('Accuracy at 75%:', acc_75)
        print('Accuracy at 50%:', acc_50)


        line = ''
        for score in [bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score, lev_score, acc_100, acc_90, acc_75, acc_50]:
            line += f'{score:.6f} '
        print(line)
        
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
        return acc_100, acc_90, acc_75, acc_50, lev_score

def predict(input_text, model, tokenizer, max_length=512):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)

    output_ids = model.generate(input_ids.to('cuda'), max_length=max_length, eos_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

def process_data(input_data):
    try:
        texts = make_data(input_data)
        input_text = texts['input_text']
        target_text = texts['target_text']
        pred_text = predict(input_text, model, tokenizer)
        return pred_text, target_text
    except Exception as e:
        print(f"Error in processing data: {e}", flush=True)
        return None, None

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument('--root', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    metrics = Metric_calculator()
    with open(data_path, 'r') as f:
        data = json.load(f)
    predict_list = []
    target_list = []
    idx = 0
    num_workers = 2
    multiprocessing.set_start_method('spawn', force=True)
    
    if not os.path.exists(args.root):
        os.makedirs(args.root, exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_input = {executor.submit(process_data, input_data): input_data for input_data in data}

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
    json.dump(predict_list, open(os.path.join(args.root, 'predict.json'), 'w'))
    json.dump(target_list, open(os.path.join(args.root, 'target.json'), 'w'))
    
    res = metrics(target_list, predict_list, use_tokenizer=True)
    print(res)
    with open(os.path.join(args.root, 'metrics.json'), 'w') as f:
        json.dump(res, f)
