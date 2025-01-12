from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

p = Chem.AdjustQueryParameters.NoAdjustments()
p.makeDummiesQueries = True
def check_substructure_match(query_smiles, base_smiles):
    # Create molecule objects from the SMILES strings
    query_mol = Chem.MolFromSmiles(query_smiles)
    hq = Chem.AdjustQueryProperties(query_mol, p)
    base_mol = Chem.MolFromSmiles(base_smiles)
    matches = base_mol.GetSubstructMatches(hq)
    return len(matches)

def count_occurrances(query_smiles, smiles_list):
    count = 0
    for smiles in smiles_list:
        count += check_substructure_match(query_smiles, smiles)
    return count

with open('data/mol_collections/pubchem324k_smiles.txt') as f:
    smiles_list = f.read().splitlines()

import random
random.seed(42)
def do_count(smiles, n=1000):
    sublist = random.sample(smiles_list, n)
    return count_occurrances(smiles, sublist)/n


from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count

with open('data/mol_collections/dums.txt') as f:
    dums = f.read().splitlines()
dums = [d for d in dums if '.' not in d]
# dums = dums[:100]
results = process_map(do_count, dums, max_workers=cpu_count(), chunksize=1)

dum_results = dict(zip(dums, results))
dum_results = {k: v for k, v in sorted(dum_results.items(), key=lambda item: item[1], reverse=True)}

import json
with open('data/mol_collections/dum_counts_raw.json', 'w') as f:
    json.dump(dum_results, f, indent=4)