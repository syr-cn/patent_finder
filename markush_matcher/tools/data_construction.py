import json
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
import random 
import itertools
import re
from rdkit import Chem
# Language model

def frags2mols(allFrags_smi, res_num=10, max_trial=100):
    '''
    '''
    # 使用 itertools.product 生成所有排列组合
    combinations = list(itertools.product(*allFrags_smi))
    res_combine = []
    idx = 0
    # 输出结果
    results=[]
    for num, combination in enumerate(combinations):
        if idx >= res_num or num >= max_trial:
            break
        print(combination, flush=True)
        smi = '.'.join(combination)
        mol = Chem.MolFromSmiles(smi)
        try:
            mol_com = Chem.molzip(mol)
            newsmi = Chem.MolToSmiles(mol_com)
        # print(newsmi)
            results.append(newsmi)
            idx+=1
            res_combine.append(combination)
        except:
            # print(combination)
            continue
    return res_combine, results