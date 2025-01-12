from rdkit import Chem
import json
import math
from collections import Counter
import random
from tools.rdkit_utils.translate import Translator
from tools.rdkit_utils.misc import RingIndex, AtomIndex
import pandas as pd
import re

from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count

def get_group_name(group):
    return str(group)

def concat_smiles(smiles_list):
    '''
    demo input: ['c1cc([*:1])c2c(c1)ccn2[*:2]', '[*:1]C', '[*:2]OC1CC1']
    '''
    smi = '.'.join(smiles_list)
    mol = Chem.MolFromSmiles(smi)
    mol_com = Chem.molzip(mol)
    # newsmi = Chem.MolToSmiles(mol_com, canonical=False)
    newsmi = Chem.MolToSmiles(mol_com)
    return newsmi

class GroupSampler:
    def __init__(self) -> None:
        dum_path = {
            1: 'data/mol_collections/dum_counts_fixed.json',
            2: 'data/mol_collections/dumXYZ_counts_raw.json',
            3: 'data/mol_collections/dum_counts_3.json',
        }
        dum_prob_map = {k: json.load(open(v)) for k, v in dum_path.items()}
        for dum_num, dum_probs in dum_prob_map.items():
            if dum_num != 1:
                continue
            for dum, prob in dum_probs.items():
                # dum_probs[dum] = math.log(prob + 1e-3)
                # dum_probs[dum] = math.sqrt(prob + 1e-3)
                # dum_probs[dum] = prob + 1e-3
                dum_probs[dum] = (prob + 0.01)/(sum(c.isalpha() for c in dum))
            for dum, prob in dum_probs.items():
                dum_probs[dum] = prob / sum(dum_probs.values())
        self.dum_list_dict = {k: list(v.keys()) for k, v in dum_prob_map.items()}
        self.dum_prob_dict = {k: list(v.values()) for k, v in dum_prob_map.items()}
        self.abbrevs_dict = json.load(open('data/mol_collections/abbrevs.json'))

    def do_sample_dum(self, num_dums=1):
        dum_list  = self.dum_list_dict[num_dums]
        dum_probs = self.dum_prob_dict[num_dums]
        sampled_dum = random.choices(dum_list, dum_probs, k=1)[0]
        return sampled_dum

    def get_atom_group_value(self, group, skeleton_mol=None):
        atom_id = group.id
        group_symbol = group.symbol
        if group_symbol in self.abbrevs_dict:
            group_value = self.abbrevs_dict[group_symbol]
            group_value = '*' + group_value
            return group_value
        assert 'CH' not in group_symbol
        assert len([c for c in str(group) if c.isalpha()]) <= 2

        if skeleton_mol is None:
            num_bonds = 1
        else:
            atom = skeleton_mol.GetAtomWithIdx(atom_id)
            num_bonds = len(atom.GetBonds())
        if num_bonds == 1:
            group_value = self.do_sample_dum(num_dums=1)
        else:
            assert NotImplementedError(f'num_bonds={num_bonds} not implemented')
        return group_value

    def get_atom_value(self, bond_types):
        bond_cnt = sum(bond_types)
        if bond_cnt == 2:
            return random.choices(['C', 'N'], [0.8, 0.2])[0]
        elif bond_cnt == 3:
            if len(bond_types) == 3:
                return random.choices(['C', 'N'], [0.8, 0.2])[0]
            else:
                return 'C'
        elif bond_cnt == 4:
            return 'C'
        return 'C'

group_sampler = GroupSampler()

def attach_atoms(skeleton_smarts, atom_list):
    skeleton_mol = Chem.MolFromSmarts(skeleton_smarts)
    group_map = {}
    tmp_smarts = skeleton_smarts.replace('*', '|')
    fragment_list = []
    for group in atom_list:
        group_value = group_sampler.get_atom_group_value(group, skeleton_mol)

        group_symbol = get_group_name(group)
        group_map[group_symbol] = group_value
        
        group_id = len(fragment_list) + 1
        fragment_list.append(group_value.replace('*', f'[*:{group_id}]'))
        tmp_smarts = tmp_smarts.replace('|', f'[*:{group_id}]', 1)
    fragment_list.append(tmp_smarts)
    newsmi = concat_smiles(fragment_list)
    return newsmi, group_map

def attach_atoms_v2(skeleton_smarts, atom_dict):
    skeleton_mol = Chem.MolFromSmarts(skeleton_smarts)
    group_map = {}
    tmp_smarts = skeleton_smarts.replace('*', '|')
    fragment_list = []

    for atom in skeleton_mol.GetAtoms():
        if atom.GetSymbol() != '*':
            continue
        atom_id = atom.GetIdx()
        group = atom_dict[int(atom_id)]
        
        bond_cnt = len(atom.GetNeighbors())
        if bond_cnt == 1:
            group_value = group_sampler.get_atom_group_value(group, skeleton_mol)
            group_id = len(fragment_list) + 1
            fragment_list.append(group_value.replace('*', f'[*:{group_id}]'))
            tmp_smarts = tmp_smarts.replace('|', f'[*:{group_id}]', 1)
        else:
            bond_types = [int(bond.GetBondType()) for bond in atom.GetBonds()]
            isAromatic = any([a.GetIsAromatic() for a in atom.GetNeighbors()])
            group_value = group_sampler.get_atom_value(bond_types)
            if isAromatic:
                group_value = group_value.lower()
            tmp_smarts = tmp_smarts.replace('|', group_value, 1)
            group_value = group_value.upper()

        group_symbol = get_group_name(group)
        group_map[group_symbol] = group_value.replace('*', '')
        
    fragment_list.append(tmp_smarts)
    newsmi = concat_smiles(fragment_list)
    return newsmi, group_map

def add_attachment_on_rings(smiles, ring_indices):
    """
    This function is totally GPT generated code. Not tested yet.
    """
    # Load the SMILES string into an RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    # Find ring information
    ssr = Chem.GetSymmSSSR(mol)  # Get the list of rings in the molecule
    
    # Create an editable molecule
    emol = Chem.EditableMol(mol)
    
    # Track label count for unique group labeling
    label_count = 1
    
    for ring_index in ring_indices:
        # Check if the ring_index is valid
        if ring_index - 1 >= len(ssr):
            raise ValueError("Ring index is out of range.")
        
        # Get the specified ring
        target_ring = ssr[ring_index - 1]
        
        # Find atoms with at least one hydrogen to choose from
        candidates = []
        weights = []
        for atom_idx in target_ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            num_hs = atom.GetTotalNumHs()
            if num_hs > 0:
                candidates.append(atom_idx)
                weights.append(num_hs)
        
        if not candidates:
            raise ValueError("No suitable atoms with hydrogen found in the specified ring.")

        # Randomly select an atom from candidates weighted by the number of hydrogens
        chosen_atom_idx = random.choices(candidates, weights=weights)[0]
        
        # Add a dummy atom with a unique label to the molecule
        star_atom = Chem.Atom(0)  # Dummy atom type 0 (wildcard)
        star_atom.SetAtomMapNum(label_count)
        star_atom_idx = emol.AddAtom(star_atom)
        
        # Add a bond between the chosen atom and the new labeled dummy atom
        emol.AddBond(chosen_atom_idx, star_atom_idx, Chem.BondType.SINGLE)
        
        # Increment label count for the next potential group
        label_count += 1
    
    # Get the modified molecule
    modified_mol = emol.GetMol()
    
    # Generate and return the new SMILES string
    return Chem.MolToSmiles(modified_mol, canonical=False)


def attach_rings(skeleton_smarts, ring_list):
    if len(ring_list) == 0:
        return skeleton_smarts, {}
    ring_indices = [group.id for group in ring_list]
    new_smiles = add_attachment_on_rings(skeleton_smarts, ring_indices)
    fragment_list = [new_smiles]
    group_map = {}
    for idx, group in enumerate(ring_list):
        group_value = group_sampler.get_atom_group_value(group=group, skeleton_mol=None)
        group_symbol = group.symbol if group.script is None else f'{group.symbol}[{group.script}]'
        group_map[group_symbol] = group_value.replace('*', '')
        fragment_list.append(group_value.replace('*', f'[*:{idx+1}]'))
    new_smiles = concat_smiles(fragment_list)
    return new_smiles, group_map


def process_markush(markush_str):
    """
    demo input:
    - '*c1cccc2ccn(*)c12<sep><a>0:R[1]</a><a>9:R[2]</a>'
    - 'Cc1c(C(=O)O)nc2c(C)cccc2c1OC<sep><r>2:R[3]</r>'
    """
    parsed = Translator.parse_caption(markush_str, return_mol=False)
    if parsed is None:
        return None
    skeleton_smarts, groups = parsed
    
    grp_descriptions = Translator.parse_groups(groups)
    atom_list = []
    ring_list = []
    for grp in grp_descriptions:
        if isinstance(grp.id, RingIndex):
            ring_list.append(grp)
        elif isinstance(grp.id, AtomIndex):
            atom_list.append(grp)
    atom_list = sorted(atom_list, key=lambda x: x.id)
    atom_dict = {int(grp.id): grp for grp in atom_list}
    assert len(atom_dict) >= skeleton_smarts.count('*')

    atomed_smiles, atom_group_map = attach_atoms_v2(skeleton_smarts, atom_dict)
    full_smiles, ring_group_map = attach_rings(atomed_smiles, ring_list)
    group_map = {**atom_group_map, **ring_group_map}
    return full_smiles, group_map


def is_flawed_markush(markush_str):
    for term in ['<c>', '<dum>', '.', '|Sg:n|']: # check for invalid terms
        if term in markush_str:
            return 1
    if re.search(r':[a-zA-Z]{4,}', markush_str): # check for too long group name
        return 2
    if len(markush_str.split('<sep>')[0]) < 5: # check for too short skeleton
        return 3
    if len(markush_str.split('<sep>')[1]) < 2: # check for empty group
        return 4
    _, groups = Translator.parse_caption(markush_str, return_mol=False)
    grp_descriptions = Translator.parse_groups(groups)
    grp_strings = [str(grp) for grp in grp_descriptions]
    if len(grp_strings) != len(set(grp_strings)): # check for duplicate groups
        return 5
    return 0

def process_markush_runnable(markush_str):
    try:
        flowed_markush_type = is_flawed_markush(markush_str)
        if flowed_markush_type != 0:
            return flowed_markush_type
        newsmi, group_map = process_markush(markush_str)
        return {
            'markush': markush_str,
            'smiles': newsmi,
            'r_group_map': group_map
        }
    except Exception as e:
        if isinstance(e, AssertionError):
            return 6
        return -1

def main():
    input_file = 'data/markush/all.csv'
    output_file = 'data/all.json'

    df = pd.read_csv(input_file)
    markush_list = df['SMILES'].tolist()
    markush_list = [m.strip() for m in markush_list]
    
    print(f'Number of markush strings: {len(markush_list)}')
    results = process_map(process_markush_runnable, markush_list, max_workers=cpu_count(), mininterval=100, chunksize=1)
    exceptions = [r if isinstance(r, int) else 0 for r in results]
    exceptions_counter = dict(sorted(Counter(exceptions).items(), key=lambda item: item[0]))
    print('exceptions_counter: ', json.dumps(exceptions_counter, indent=4))
    results = [r for r in results if isinstance(r, dict)]
    print(f'Number of processed markush strings: {len(results)}')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    random.seed(42)
    main()