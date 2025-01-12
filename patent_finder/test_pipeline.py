from config import config
import os
import sklearn.metrics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import copy
import selfies
import random

from utils import save_json, load_json
from models import StructuredLLM
from patent_agent import PatentAgent
from patent_helper import PatentHelper
agent = PatentAgent(config.max_fulltext_len)
patent_helper = PatentHelper(
    config.cache_root,
    config.patent_extract_method,
    config.image_parser_endpoint,
)
from retry import retry

"""
Example of test file:
{
    "patent_id": "US10676478",
    "positives": ["O=C1N([C@H](COC)C2=CC3=NC([C@@H](NC(C4=NON=C4C)=O)C5CCC(F)(CC5)F)=CN3N=C2)C[C@@H](C(F)(F)F)N1", ...],
    "negatives": [...]
}
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
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

def load_test_dataset(data_path):
    test_data = load_json(data_path)
    full_test_data = []
    for data_dict in test_data:
        for pos in data_dict["positives"]:
            full_test_data.append({
                'patent_id': data_dict['patent_id'],
                "target_smiles": pos,
                "label": 1
            })
        for neg in data_dict["negatives"]:
            full_test_data.append({
                'patent_id': data_dict['patent_id'],
                "target_smiles": neg,
                "label": 0
            })
    print(f"Loaded {len(full_test_data)} test data")
    return full_test_data

def load_test_dataset_v2(data_path, neg_aug=0):
    test_data = load_json(data_path)
    print(f"Loaded {len(test_data)} test data")
    all_patents = set([data_dict['patent_id'] for data_dict in test_data])
    if neg_aug == 0:
        return test_data

    random.seed(42)
    return test_data


    # Augment the negative data points
    aug_data_list = []
    for patent_id in all_patents:
        for _ in range(neg_aug):
            while True:
                neg_data_dict = random.choice(test_data)
                if neg_data_dict['patent_id'] != patent_id:
                    neg_data_dict = copy.deepcopy(neg_data_dict)
                    break
            # Construct a negative data by changing the patent_id
            neg_data_dict['label'] = 0
            neg_data_dict['patent_id'] = patent_id
            aug_data_list.append(neg_data_dict)
    test_data += aug_data_list
    return test_data

# @retry(tries=3, delay=2)
def eval_pipeline(data_dict):
    start_time = time.time()
    patent_id = data_dict["patent_id"]
    target_smiles = data_dict["target_smiles"]
    patent_info = patent_helper.patent_extract(patent_id)
    
    claim_list = patent_info['claims']
    text_dict = patent_info['text']
    mole_dict = {'smiles': target_smiles, 'selfies': selfies.encoder(target_smiles)}
    if 'iupac_name' in data_dict:
        mole_dict['iupac_name'] = data_dict['iupac_name']
    
    llm = StructuredLLM(config.llm_name)
    result = getattr(agent, config.test_method_name)(llm, claim_list, text_dict, mole_dict)

    result['patent_id'] = patent_info['patent_id']
    result['target_smiles'] = target_smiles
    result['mol_dict'] = mole_dict
    result['label'] = data_dict['label']
    result['prediction'] = result['is_protected']
    result['time'] = round(time.time() - start_time, 2) # in seconds
    print(f"[PATENT {patent_info['patent_id']}] Prediction: {result['prediction']} (GT: {bool(data_dict['label'])}). Molecule: {mole_dict}")
    sys.stdout.flush()
    time.sleep(30)
    return result

def eval_structure_sim(data_dict):
    patent_id = data_dict["patent_id"]
    target_smiles = data_dict["target_smiles"]
    patent_info = patent_helper.patent_extract(patent_id)
    
    claim_dict = patent_info['claims'][0]
    markush_skeleton = claim_dict['caption'].split('<sep>')[0]
    # markush_skeleton = markush_skeleton.replace('*', 'C')
    sim, tanimoto_sim = compute_simi(target_smiles, markush_skeleton)
    is_protected = tanimoto_sim > .5
    
    result = {}
    result['patent_id'] = patent_info['patent_id']
    result['target_smiles'] = target_smiles
    result['label'] = data_dict['label']
    result['prediction'] = is_protected
    print(f"[PATENT {patent_info['patent_id']}] Prediction: {result['prediction']} (GT: {bool(result['label'])}); \t SMILES: {target_smiles}")
    sys.stdout.flush()
    return result

def run_parallel(func, data_list, test_num_workers=1, result_path=None, init_result_list=[]):
    result_list = copy.deepcopy(init_result_list)
    if test_num_workers == 1:
        print('Running in single thread mode\n')
        for data_dict in data_list:
            result = func(data_dict)
            result_list.append(result)
            if result_path:
                save_json(result_list, result_path)
    else:
        with ThreadPoolExecutor(max_workers=test_num_workers) as executor:
            futures = [executor.submit(func, data_dict) for data_dict in data_list]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    result_list.append(result)
                    if result_path:
                        save_json(result_list, result_path)
                except Exception as e:
                    print(f"Error when running: {repr(e)}")
    return result_list

def test_pipeline(data_path, result_path=None, eval_func=eval_pipeline) -> bool:
    result_file = os.path.join(result_path, "results.json")
    test_data = load_test_dataset_v2(data_path, config.test_neg_aug)

    reference_data = load_json(result_file) if os.path.exists(result_file) else [] # some of the data are already processed
    exist_keys = [data_dict['patent_id']+data_dict['target_smiles'] for data_dict in reference_data]
    test_data = [data_dict for data_dict in test_data if data_dict['patent_id']+data_dict['target_smiles'] not in exist_keys]
    print(f"Start testing {len(test_data)} data points")
    
    results = run_parallel(eval_func, test_data, test_num_workers=config.test_num_workers, result_path=result_file, init_result_list=reference_data)
    return results

def eval_metrics(pred_list, gt_list):
    pred_list = [False if pred is None else pred for pred in pred_list]
    # pred_list = pred_list + [True for _ in range(5)] + [False for _ in range(95)]
    # gt_list = gt_list + [False for _ in range(100)]
    try:
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(gt_list, pred_list).ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    except:
        tn, fp, fn, tp = None, None, None, None
    metrics = {
        'f1_score': sklearn.metrics.f1_score(gt_list, pred_list, average='micro'),
        'accuracy': sklearn.metrics.balanced_accuracy_score(gt_list, pred_list),
        'precision': sklearn.metrics.precision_score(gt_list, pred_list),
        'recall': sklearn.metrics.recall_score(gt_list, pred_list),
        'specificity': (tn / (tn + fp)) if tn is not None else None,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'total': len(gt_list)
    }
    return metrics

def read_results(result_path):
    result_file = os.path.join(result_path, "results.json")
    result_list = load_json(result_file)

    metrics = eval_metrics(
        [data_dict["prediction"] for data_dict in result_list],
        [data_dict["label"] for data_dict in result_list]
    )
    save_json(metrics, os.path.join(result_path, "metrics.json"))
    for key, value in metrics.items():
        print(f"{key}: {value}")


def read_results_v2(result_path):
    """
    Filter by different substructure match results.
    """
    result_file = os.path.join(result_path, "results.json")
    result_list = load_json(result_file)

    all_metrics = eval_metrics(
        [data_dict["prediction"] for data_dict in result_list],
        [data_dict["label"] for data_dict in result_list]
    )
    save_json(all_metrics, os.path.join(result_path, "metrics-all.json"))
    print(json.dumps(all_metrics, indent=4))

    result_dict = {
        'valid': [],
        'none': [],
        'error': []
    }
    for result in result_list:
        try:
            substructure_map = result["reasoning"][0]['software_result']["substructure_map"]
        except:
            continue
        if isinstance(substructure_map, dict):
            result_dict['valid'].append(result)
        elif substructure_map is None:
            result_dict['none'].append(result)
        elif isinstance(substructure_map, str):
            result_dict['error'].append(result)
    
    for key in result_dict:
        metrics = eval_metrics(
            [data_dict["prediction"] for data_dict in result_dict[key]],
            [data_dict["label"] for data_dict in result_dict[key]]
        )
        save_json(metrics, os.path.join(result_path, f"metrics-{key}.json"))
        print(f"Metrics for {key}: {json.dumps(metrics, indent=4)}")


def read_results_v3(result_path):
    """
    Filter by different llm substructure match results.
    """
    result_file = os.path.join(result_path, "results.json")
    result_list = load_json(result_file)

    all_metrics = eval_metrics(
        [data_dict["reasoning"][0]['llm_result']['is_match'] for data_dict in result_list],
        [data_dict["label"] for data_dict in result_list]
    )
    save_json(all_metrics, os.path.join(result_path, "llm-metrics-all.json"))
    print(json.dumps(all_metrics, indent=4))

    result_dict = {
        'valid': [],
        'none': [],
        'error': []
    }
    for result in result_list:
        try:
            substructure_map = result["reasoning"][0]['software_result']["substructure_map"]
        except:
            continue
        if isinstance(substructure_map, dict):
            result_dict['valid'].append(result)
        elif substructure_map is None:
            result_dict['none'].append(result)
        elif isinstance(substructure_map, str):
            result_dict['error'].append(result)
    
    for key in result_dict:
        metrics = eval_metrics(
            [data_dict["reasoning"][0]['llm_result']['is_match'] for data_dict in result_dict[key]],
            [data_dict["label"] for data_dict in result_dict[key]]
        )
        save_json(metrics, os.path.join(result_path, f"llm-metrics-{key}.json"))
        print(f"Metrics for {key}: {json.dumps(metrics, indent=4)}")

if __name__ == "__main__":
    print(f'Running test {config.test_method}')
    test_methods = {
        'pipeline': eval_pipeline,
        'structure_sim': eval_structure_sim,
    }
    if config.test_method in test_methods:
        assert os.path.exists(config.test_result_path)
        test_pipeline(config.test_data_path, config.test_result_path, test_methods[config.test_method])
    else:
        # do not test, just read the results
        pass
    read_results(config.test_result_path)