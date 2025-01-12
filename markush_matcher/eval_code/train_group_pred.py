from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import argparse
import os
import json
from multiprocessing import cpu_count
import pandas as pd
from functools import partial
from nltk.translate.bleu_score import corpus_bleu

import random
import torch
import numpy as np
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(data, filename):
    print(f'Writing {len(data)} data to {filename}...')
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f'Reading {len(data)} data from {filename}...')
    return data

def compute_metrics(eval_pred, tokenizer):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    predictions = torch.argmax(torch.tensor(predictions[0]), dim=-1)

    predictions_sentences = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    label_ids[label_ids==-100] = 0
    labels_sentences = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    formatted_references = [[label] for label in label_ids]
    bleu_score = corpus_bleu(formatted_references, predictions_sentences)

    return {
        'bleu': bleu_score,
        # 'rouge1': rouge_score['rouge1'].mid.fmeasure,
        # 'rouge2': rouge_score['rouge2'].mid.fmeasure,
        # 'rougeL': rouge_score['rougeL'].mid.fmeasure,
        'predictions': predictions_sentences,
        'references': labels_sentences
    }

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

def main(args):
    data_list = load_json(args.dataset_path)
    data_list = data_list[1000:]
    processed_dataset = process_map(make_data, data_list, max_workers=cpu_count(), mininterval=10, chunksize=1)
    dataset = Dataset.from_list(processed_dataset)
    # dataset = dataset.train_test_split(test_size=0.1)['test']

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.add_tokens(['{', '}', '<sep>', '<a>', '</a>', '<dum>', '<r>', '</r>', '<c>', '</c>'])
    tokenizer.pad_token = tokenizer.eos_token
    model = T5ForConditionalGeneration.from_pretrained(args.base_model_name)
    model.resize_token_embeddings(len(tokenizer))

    def preprocess(examples):
        model_inputs = tokenizer(examples['input_text'], max_length=args.max_text_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target_text'], max_length=args.max_text_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    dataset = dataset.map(preprocess, batched=True)
    max_test_size = 50
    test_size_ratio = min(max_test_size / len(dataset), 0.01)
    train_test_split = dataset.train_test_split(test_size=test_size_ratio)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = {
        'output_dir': args.checkpoint_dir,
        'evaluation_strategy': "steps",
        'eval_steps': 10000,
        'save_strategy': "steps",
        'save_steps': 10000,
        'logging_steps': 20,
        'learning_rate': args.learning_rate,
        'num_train_epochs': args.num_train_epochs,
        'seed': args.random_seed,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'per_device_eval_batch_size': max(args.per_device_train_batch_size//2, 1),
        'load_best_model_at_end': True,
        'metric_for_best_model': "loss",
        'greater_is_better': False,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
    }
    if not args.no_eval:
        del training_args['eval_steps']
        training_args['evaluation_strategy'] = "no"
        training_args['load_best_model_at_end'] = False
        training_args['metric_for_best_model'] = None

    training_args = TrainingArguments(**training_args)

    compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_tokenizer,
    )

    if not args.no_train:
        trainer.train()

        model_save_path = os.path.join(args.checkpoint_dir, "final")
        trainer.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")

        log_save_path = os.path.join(args.checkpoint_dir, "log_history.csv")
        log_history_df = pd.DataFrame(trainer.state.log_history)
        log_history_df.to_csv(log_save_path, index=False)
        print(f"Log history saved to {log_save_path}")

    if not args.no_eval:
        evaluation_results = trainer.evaluate()
        evaluation_results_path = os.path.join(args.checkpoint_dir, "evaluation_results.json")
        with open(evaluation_results_path, "w") as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Evaluation results saved to {evaluation_results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--base_model_name", type=str, default="laituan245/molt5-base")
    parser.add_argument("--no_train", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--dataset_path", type=str, default="none")
    parser.add_argument("--checkpoint_dir", type=str, default="ckpt")
    # model parameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--max_text_length", type=int, default=512)
    args = parser.parse_args()
    print('-'*200)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print('-'*200)
    set_random_seed(args.random_seed)
    main(args)
