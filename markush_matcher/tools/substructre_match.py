from typing import Dict, Optional, Sequence, Union, List
import os
import sys
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition as rdRGD
from rdkit.Chem.rdchem import Mol, RWMol, Atom, ChiralType, BondType
import json
# from rdkit_utils.translate import Translator, GroupDesc
# from rdkit_utils.misc import AtomIndex, is_valid_index
from .rdkit_utils.translate import Translator, GroupDesc, TextType
from .rdkit_utils.misc import AtomIndex, is_valid_index
from .rdkit_utils.misc import RingIndex 
import copy
import re
from .data_construction import frags2mols
def _get_decom_params():
    param = rdRGD.RGroupDecompositionParameters()
    param.onlyMatchAtRGroups = True
    param.allowNonTerminalRGroups = True
    param.removeAllHydrogenRGroups = False
    param.removeAllHydrogenRGroupsAndLabels = False 
    param.allowMultipleRGroupsOnUnlabelled = True
    return param


def _merge_nonterminal_Rsite(r_mol: Mol, r_atom: Atom) -> Optional[Mol]:
    r_mol = RWMol(r_mol)
    dummy_idx = end_idx = None
    for atom in r_mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber') and atom.GetDegree() == 1:
            dummy_idx = atom.GetIdx()
            end_idx = atom.GetNeighbors()[0].GetIdx()
            break
        
    if dummy_idx is None:
        return
    
    r_mol.RemoveAtom(dummy_idx)
    begin_idx = r_mol.AddAtom(r_atom)
    r_mol.AddBond(begin_idx, end_idx, BondType.SINGLE)  # XXX What if non-single bond?
    
    return r_mol.GetMol()
    
    
class MoleculeQuerier:
    """"""
    def __init__(self, mol, grp_descriptions, **kwds) -> None:
        # parsed = Translator.parse_caption(caption, return_mol=True)
        # if parsed is None:
        #     raise ValueError(f'Incorrect caption: {caption}')
        # mol, groups = parsed
        # grp_descriptions = Translator.parse_groups(groups)
        self._Rgroups = self._collect_Rgroups(mol, grp_descriptions)
        # import pdb; pdb.set_trace()
        self._core = self._prepare_core(mol)
        self._Rsites = self._collect_Rsites(self._core)
    
    def _prepare_core(self, mol: Mol) -> Mol:
        """Remove atom chirality and set R-site index."""
        # TODO Set aromaticity in some rings with multiple R-sites.
        Rsite_idx = 1
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED:
                atom.SetChiralTag(ChiralType.CHI_UNSPECIFIED)
            if atom.GetSymbol() == '*':
                atom.SetProp('molAtomMapNumber', str(Rsite_idx))
                Rsite_idx += 1

        return mol
    def _replace_variable_groups(self, original_string, values_dict):
        """
        用字典中的值替换原字符串中的可变基团。

        :param original_string: 包含可变基团的原字符串
        :param values_dict: 替换可变基团的字典，键为基团的占位符，值为要替换的值
        :return: 替换后的字符串
        """
        # 遍历字典，将每个键值对中的键替换为对应的值
        for group, value in values_dict.items():
            original_string = original_string.replace(group, value)
        
        return original_string
    def _replace_placeholders(self, r_groups=None,caption=None, res_num=3):
        """
        将replacements_dict中的信息替换到input_str中的相应占位符位置
        :param input_str: 包含占位符的原字符串
        :param mapping_dict: 键为数字，占位符位置映射的字典，值为 R1, R2, R3 等
        :param replacements_dict: 每个 R 键对应替换列表的字典
        :return: 替换后的字符串
        """
        from CombineMols.CombineMols import CombineMols
        # import pdb; pdb.set_trace()
        mapping_dict = self._Rgroups
        random_list = ['C', 'CC', 'c1ccccc1', 'c1ccccc1CO', 'c1cccnc1']
        # for ran in random_list:
        #     print(Chem.MolFromSmiles(ran))
        #  ['C', 'CC(F)(F)F', 'CC'], 'R2': ['c1ccccc1', 'c1ccccc1CO', 'c1cccnc1']
        import random
        # import pdb; pdb.set_trace()
        # 遍历 mapping_dict，将每个占位符替换为对应的 R 值中的一项
        idx = 1
        smi_list = []
        smi_list.append([Chem.MolToSmiles(self._core)])
        # import pdb; pdb.set_trace()
        keys = list(r_groups.keys())
        for key in keys:
            old_key = key
            key = key.replace('[', '').replace(']', '')
            r_groups[key] = r_groups.pop(old_key)
        # import pdb; pdb.set_trace()
        for position, r_key in mapping_dict.items():
            try:
                values = r_groups[r_key]
            except KeyError:
                return []
            vs = []
            for num, value in enumerate(values):
                value =  f'[*:{idx}]'+value
                vs.append(value)
            # print(vs)
            smi_list.append(vs)
            idx += 1
        combinations, results = frags2mols(smi_list, res_num)
        idx = 0
        value_dicts = []
        for combination, result in zip(combinations, results):
            if idx > res_num:
                break
            value_dict = {}
            value_dict['mapping'] = {}
            value_dict['markush'] = caption
            value_dict['smiles'] = result
            value_dict['Rgroups'] = mapping_dict
            value_dict['LLM_annotation'] = r_groups
            num = 0
            combination = combination[1:]
            for position, r_key in mapping_dict.items():
                value_dict['mapping'][r_key] = combination[num]
                num += 1
            value_dicts.append(value_dict)
            idx += 1
        return value_dicts
    def _collect_Rsites(self, mol: Mol) -> Dict[str, int]:
        """Gather information of R-sites in core molecule."""
        Rsites = {}
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*' and atom.GetDegree() >= 1:
                try:
                    i = atom.GetProp('molAtomMapNumber')
                except:
                    continue
                Rsites[f'R{i}'] = atom.GetIdx()
                
        return Rsites,
    
    def _collect_Rgroups(
        self, 
        mol: Mol,
        grp_descriptions: List[GroupDesc]
    ) -> Dict[int, str]:
        """Gather information of R-groups parsed from caption."""
        num_atoms = mol.GetNumAtoms()
        num_rings = mol.GetRingInfo().NumRings()
        Rgroups = {}  # key: atom index
        for desc in grp_descriptions:
            if not isinstance(desc.id, AtomIndex):  # TODO consider ring substitutes
                continue
            if not is_valid_index(desc.id, num_atoms, num_rings):
                continue
            if (
                desc.is_dummy 
                or desc.symbol is None
                or mol.GetAtomWithIdx(int(desc.id)).GetSymbol() != '*'
            ):
                continue
            
            Rgroups[int(desc.id)] = str(desc)
        
        return Rgroups
        
    def _postprocess_Rgroups(self, decom: Dict[str, Mol]) -> Dict[str, str]:
        """Finalize decomposed R-groups of one query molecule for validity check."""
        unk_idx = 1
        groups = {}
        import pdb; pdb.set_trace()
        for Rsite, atom_idx in self._Rsites.items():
            if atom_idx in self._Rgroups:
                key = self._Rgroups[atom_idx]
            else:
                key = f'UNK{unk_idx}'
                unk_idx += 1
            smi = self._get_Rsite_smi(atom_idx, Rsite, decom)
            if smi is not None:
                groups[key] = smi
                
        return groups
            
    def _get_Rsite_smi(
        self, 
        atom_idx: int, 
        Rsite: str, 
        decom: Dict[str, Mol]
    ) -> Optional[str]:
        """Get SMILES expression for single R-site."""
        r_atom = self._core.GetAtomWithIdx(atom_idx)
        if r_atom.GetDegree() == 1:
            return self._get_terminal_Rsite_smi(Rsite, decom)
        return self._get_nonterminal_Rsite_smi(Rsite, decom)
    
    def _get_terminal_Rsite_smi(
        self, 
        Rsite: str, 
        decom: Dict[str, Mol]
    ) -> Optional[str]:
        r_mol = decom.get(Rsite)
        if r_mol is None:
            return
        
        dummy_idx = root_atom = num_expl_Hs = None
        for atom in r_mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber') and atom.GetDegree() == 1:
                # NOTE Atom symbol is not `*`.
                dummy_idx = atom.GetIdx()
                root_atom = atom.GetNeighbors()[0]
                break
        
        if dummy_idx is None:
            return

        r_mol = RWMol(r_mol)
        r_mol.RemoveAtom(dummy_idx)
        # NOTE Manually add a hydrogen atom.
        begin_idx = r_mol.AddAtom(Atom(1))
        r_mol.AddBond(begin_idx, root_atom.GetIdx(), BondType.SINGLE)
        r_mol = r_mol.GetMol()
        r_mol = Chem.RemoveHs(r_mol)
        return Chem.MolToSmiles(r_mol, rootedAtAtom=root_atom.GetIdx())
    
    def _get_nonterminal_Rsite_smi(
        self, 
        Rsite: str, 
        decom: Dict[str, Mol]
    ) -> Optional[str]:
        # NOTE The atom at R-site in query molecule will not be part of decomposition.
        # First we find the atom neighboring R-site in decomposed core, which is 
        # directly copied from query molecule.
        d_core = decom['Core']
        Rsite_idx = Rsite[1:]
        r_atom = None
        for atom in d_core.GetAtoms():
            if (
                atom.HasProp('molAtomMapNumber')
                and atom.GetProp('molAtomMapNumber') == Rsite_idx
                and atom.GetDegree() == 1
            ):
                r_atom = atom.GetNeighbors()[0]
                break
        
        if r_atom is None:
            return

        if decom.get(Rsite) is not None:
            r_mol = _merge_nonterminal_Rsite(
                r_mol=decom.get(Rsite), r_atom=Atom(r_atom.GetAtomicNum())
            )
        else:
            r_mol = RWMol()
            r_mol.AddAtom(Atom(r_atom.GetAtomicNum()))
            r_mol = r_mol.GetMol()
    
        if r_mol is None:
            return

        # NOTE Start from the very last atom in both cases.
        return Chem.MolToSmiles(r_mol, rootedAtAtom=(r_mol.GetNumAtoms() - 1))
    
    def query(self, q_mols: Union[str, Sequence[str]]) -> List[Optional[Dict[str, str]]]:
        # import pdb; pdb.set_trace()
        if isinstance(q_mols, str):
            q_mols = [q_mols]
        q_mols = [Chem.MolFromSmiles(q) for q in q_mols]
        matched, unmatched = rdRGD.RGroupDecompose(
            [self._core], q_mols, asSmiles=False, options=_get_decom_params()
        )
        # import pdb; pdb.set_trace()
        res = []
        j = 0  # pointer in `matched`
        for i in range(len(q_mols)):
            if i in unmatched:
                res.append(None)
            else:
                res.append(self._postprocess_Rgroups(decom=matched[j]))
                j += 1
        
        return res    

def substructre_match(claim, target_smiles):
    try:
        match_result = Molecule_query(claim, target_smiles)
        print(f'Structure matching success. Match result: {match_result}')
        assert match_result is not None
        return {
            "is_match": any(match_result),
            "substructure_map": match_result[0]
        }
    except Exception as e:
        print(f'Structure matching failed. {repr(e)}')
        return {
            "is_match": None,
            "substructure_map": 'Error occurred during structure matching.'
        }



def Molecule_query(caption, q_mols):
    '''
    extend the function and consider the ring subsitute group
    input: markush caption: str ,
    the target molecule: list of str
    example: 
    caption = '*C1CCON1C(=O)C1CCN(*)CC1<sep><a>0:R[3]</a><a>12:R[1]</a><r>1:R[2]</r>'
    q_mols = [
        'NC(=O)c1cc(N2CCC(C(=O)N3OCC[C@H]3c3cc(F)cc(F)c3)CC2)ncn1'
    ]
    output: judge result. If not matched, return None, else return the matched group list
    '''
    if not isinstance(q_mols, list):
        q_mols = [q_mols]
    parsed = Translator.parse_caption(caption, return_mol=True)
    if parsed is None:
        raise ValueError(f'Incorrect caption: {caption}')
    mol, groups = parsed
    grp_descriptions = Translator.parse_groups(groups)
    atom_list = []
    ring_list = []
    '''add ring judge. if has ring subsitute like <r>x</r>, 
    make the mol to a group of mol where the ring subsitute is replaced by atom subsitute
    '''
    for grp in grp_descriptions:
        if isinstance(grp.id, RingIndex):
            ring_list.append(grp)
        elif isinstance(grp.id, AtomIndex):
            atom_list.append(grp)
    if len(ring_list) == 0:
        mol_querier = MoleculeQuerier(mol, atom_list)
        res = mol_querier.query(q_mols)
        return res
    else:
        ri = mol.GetRingInfo()
        atominfo = ri.AtomRings()
        smi = Chem.MolToSmiles(mol)
        for ring in ring_list:
            orig_atom_list = copy.deepcopy(atom_list)
            orig_mol = copy.deepcopy(mol)
            ring_id = ring.id
            atom_idx = atominfo[ring_id]
            for idx in atom_idx:
                rw_mol = Chem.RWMol(orig_mol)
                add_idx = rw_mol.AddAtom(Chem.Atom(0))
                rw_mol.AddBond(idx, add_idx, Chem.BondType.SINGLE)
                # mol = rw_mol.GetMol()
                smi =   Chem.MolToSmiles(rw_mol.GetMol(),rootedAtAtom=0, canonical=False)
                try:
                    mol = Chem.RWMol(Chem.MolFromSmiles(smi))
                except:
                    continue
                desc = GroupDesc(id=AtomIndex(idx+1))
                desc.symbol = ring.symbol
                desc.script = ring.script
                desc.prime = ring.prime
                desc.multiple = ring.multiple
                for atom in orig_atom_list:
                    if atom.id > idx:
                        atom.id = AtomIndex(atom.id + 1)
                orig_atom_list.append(desc)
                
                mol_querier = MoleculeQuerier(mol, orig_atom_list)
                res = mol_querier.query(q_mols)
                for result in res:
                    if result is not None:
                        return res
    return None

def mol_generation(caption,r_groups=None):
    '''
    extend MoleculeQuerier and generate molecule with LLM response group
    '''
    parsed = Translator.parse_caption(caption, return_mol=True)
    if parsed is None:
        raise ValueError(f'Incorrect caption: {caption}')
    mol, groups = parsed
    grp_descriptions = Translator.parse_groups(groups)
    atom_list = []
    ring_list = []
    # import pdb; pdb.set_trace()
    '''add ring judge. if has ring subsitute like <r>x</r>, 
    make the mol to a group of mol where the ring subsitute is replaced by atom subsitute
    '''
    for grp in grp_descriptions:
        if isinstance(grp.id, RingIndex):
            ring_list.append(grp)
        elif isinstance(grp.id, AtomIndex):
            atom_list.append(grp)
    if len(ring_list) == 0:
        mol_querier = MoleculeQuerier(mol, atom_list)
        mols = mol_querier._replace_placeholders(r_groups,caption=caption)
        
        return mols
    return []
    # else:
    #     ri = mol.GetRingInfo()
    #     atominfo = ri.AtomRings()
    #     smi = Chem.MolToSmiles(mol)
    #     smi_without_numbers = re.sub(r'\d+', '', smi)
    #     for ring in ring_list:
    #         orig_atom_list = copy.deepcopy(atom_list)
    #         orig_mol = copy.deepcopy(mol)
    #         ring_id = ring.id
    #         atom_idx = atominfo[ring_id]
    #         for idx in atom_idx:
    #             rw_mol = Chem.RWMol(orig_mol)
    #             add_idx = rw_mol.AddAtom(Chem.Atom(0))
    #             rw_mol.AddBond(idx, add_idx, Chem.BondType.SINGLE)
    #             # mol = rw_mol.GetMol()
    #             smi =   Chem.MolToSmiles(rw_mol.GetMol(),rootedAtAtom=0, canonical=False)
    #             try:
    #                 # my_mol = Chem.Mol(Chem.MolFromSmiles(smi))
    #                 # my_mol = Chem.RWMol(Chem.MolFromSmiles(smi))
    #                 my_mol = Chem.MolFromSmiles(smi)
    #                 if my_mol is None:
    #                     continue
    #             except Exception as e:
    #                 print(f"Add position error: {e}")
    #                 continue
    #             desc = GroupDesc(id=AtomIndex(idx+1))
    #             desc.symbol = ring.symbol
    #             desc.script = ring.script
    #             desc.prime = ring.prime
    #             desc.multiple = ring.multiple
    #             for atom in orig_atom_list:
    #                 if atom.id > idx:
    #                     atom.id = AtomIndex(atom.id + 1)
    #             orig_atom_list.append(desc)
                
    #             mol_querier = MoleculeQuerier(my_mol, orig_atom_list)
    #             mols = mol_querier._replace_placeholders(r_groups,caption=caption)
        
    #         return mols
if __name__ == '__main__':
    caption = "*CC[C@@H](NC(=O)C(C)(C)N)C(=O)N1CCC2=NN(*)C(=O)[C@]2(*)C1<sep><a>0:R[2]</a><a>19:R[1]</a><a>23:R[3]</a>"
    # positive_sub = {'R1': ['C', 'CC(F)(F)F', 'CC'], 'R2': ['c1ccccc1', 'c1ccccc1CO', 'c1cccnc1'], 'R3': ['c1ccccc1C', 'c1cccnc1C', 'c1cccnc1C']}
    # negative_sub = {'R1': ['CCC', 'CCCC', 'CCCCC'], 'R2': ['c1ccccc1C', 'c1cccnc1C', 'c1cccnc1C'], 'R3': ['c1cccnc1C', 'c1cccnc1C', 'c1cccnc1C']}
    # mol_generation(caption)
    from data_construction import generate_markush_rgroups
    import time
    
    with open('/root/workspace/data_consturction/filtered_smiles.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    res_list = []
    total_num = 0
    total_cost = 0
    t1 = time.time()
    for i in range(0, len(lines)):
        if total_num > 100:
            break
        # import pdb; pdb.set_trace()
        llm_result = generate_markush_rgroups(lines[i])
        # print(llm_result)
        if 'error' in llm_result:
            total_cost += llm_result['cost']
            continue
        if len(llm_result['r_group_values']) > 0:
            try:
                r_groups = json.loads(llm_result['r_group_values'])
            except Exception as e:
                print(f"Error: {e}")
                print(llm_result['r_group_values'])
                continue
            
        else:
            r_groups = {}
        if len(llm_result['specific_r_group_values']) > 0:
            try:
                specific_r_groups = json.loads(llm_result['specific_r_group_values'])
            except Exception as e:
                print(f"Error: {e}")
                print(llm_result['specific_r_group_values'])
                continue
        else:
            specific_r_groups = {}
        total_cost += llm_result['cost']
        all_keys = set(r_groups.keys()).union(specific_r_groups.keys())
        all_groups = {}
        for key in all_keys:
            if key in r_groups and key in specific_r_groups:
                all_groups[key] = r_groups[key] 
            elif key in r_groups:
                all_groups[key] = r_groups[key]
            elif key in specific_r_groups:
                all_groups[key] = [specific_r_groups[key]]
        # import pdb; pdb.set_trace()
        res = mol_generation(lines[i],all_groups)
        if isinstance(res,list):
            if len(res) > 0:
                res_list.append(res)
                total_num += 1
    import json
    t2 = time.time()
    
    with open('sample_output.json', 'w', encoding='utf-8') as json_file:
        json.dump(res_list, json_file, ensure_ascii=False, indent=4)
    print(f"total time: {t2-t1}")
    print(f"total cost: {total_cost}$")
    # import concurrent.futures
    # import json

    # # 读取 SMILES 文件
    # with open('/root/workspace/data_consturction/filtered_smiles.txt', 'r', encoding='utf-8') as file:
    #     lines = [line.strip() for line in file.readlines()]

    # res_list = []
    # total_cost = 0

    # def process_smile(smile):
    #     llm_result = generate_markush_rgroups(smile)
    #     if 'error' in llm_result:
    #         return [], llm_result['cost']
    #     if len(llm_result['r_group_values']) > 0:
    #         r_groups = json.loads(llm_result['r_group_values'])
    #     else:
    #         r_groups = {}
    #     if len(llm_result['specific_r_group_values']) > 0:
    #         specific_r_groups = json.loads(llm_result['specific_r_group_values'])
    #     else:
    #         specific_r_groups = {}

    #     all_keys = set(r_groups.keys()).union(specific_r_groups.keys())
    #     all_groups = {}
    #     for key in all_keys:
    #         if key in r_groups and key in specific_r_groups:
    #             all_groups[key] = r_groups[key]
    #         elif key in r_groups:
    #             all_groups[key] = r_groups[key]
    #         elif key in specific_r_groups:
    #             all_groups[key] = [specific_r_groups[key]]

    #     res = mol_generation(smile, all_groups)
    #     return res, llm_result['cost']

    # # 定义最大并行线程数和处理总数
    # max_workers = 4
    # total_num = 100
    # import time
    # t1 = time.time()
    # # 使用 ThreadPoolExecutor 并行化
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {executor.submit(process_smile, lines[i]): i for i in range(total_num)}
        
    #     for future in concurrent.futures.as_completed(futures):
    #         res, cost = future.result()
    #         if res:
    #             res_list.append(res)
    #         total_cost += cost

    # # 保存结果到 JSON 文件
    # with open('sample_output.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(res_list, json_file, ensure_ascii=False, indent=4)
    # t2 = time.time()
    # print(f"Time cost: {t2 - t1:.2f}s")
    # print(f"Total cost: {total_cost}$")

    exit()
    '''
    current patent: WO2009117421A2, current markush: ***C(*)(*)c1*:*:*n1*<sep><a>0:R[2]</a><a>1:G[2]</a><a>2:G[1]</a><a>4:R[6]</a><a>5:R[5]</a><a>7:X[3]</a><a>8:X[2]</a><a>9:X[1]</a><a>11:R[1]</a>, current formula: c1(C(F)(F)F)cnc(NNC(=O)c2cccn2-c2cccc(C)c2)c(Cl)c1
骨架识别负例失败
    '''
    # caption = ' *c1[nH]nc(Nc2c*c3*n*c-3*c2)c1*<sep><a>0:Y</a><a>8:Z</a><a>10:X[2]</a><a>12:X[1]</a><a>14:Z</a><a>17:X</a>'
    '''
    current patent: US11773096, current markush: **c1ncc2c(n1)N(*)[C@H](*)C(=O)N2C<sep><a>0:R</a><a>1:R</a><a>9:R[2]</a><a>11:R[1]'</a>, 
    current formula: CC(C)C[C@@H]1C(=O)N(C)c2cnc(Nc3cc(Cl)c(O)c(Cl)c3)nc2N1C1CCCC1
    '''
    caption = '*C1CCON1C(=O)C1CCN(*)CC1<sep><a>0:R[30]</a><a>12:R[1]</a><r>1:R[2]</r>'
    # caption = '*C1CCON1C(=O)C1CCN(*)CC1<sep><a>0:R[3]</a><a>12:R[1]</a><r>1:R[2]</r>'
    # caption = '**c1ncc2c(n1)N(*)[C@H](*)C(=O)N2C<sep><a>0:R</a><a>1:R</a><a>9:R[2]</a><a>11:R[1]</a>'
    
    q_mols = [
        # ' c1cnc2[nH]nc(Nc3ccc4c(C5CCCC5)noc4c3)c2c1',
        'NC(=O)c1cc(N2CCC(C(=O)N3OCC[C@H]3c3cc(F)cc(F)c3)CC2)ncn1'
        # 'CC(C)C[C@@H]1C(=O)N(C)c2cnc(Nc3cc(Cl)c(O)c(Cl)c3)nc2N1C1CCCC1'
    ]
    parsed = Translator.parse_caption(caption, return_mol=True)
    if parsed is None:
        raise ValueError(f'Incorrect caption: {caption}')
    mol, groups = parsed
    import pdb; pdb.set_trace()
    grp_descriptions = Translator.parse_groups(groups)
    atom_list = []
    ring_list = []
    '''add ring judge. if has ring subsitute like <r>x</r>, 
    make the mol to a group of mol where the ring subsitute is replaced by atom subsitute
    '''
    for grp in grp_descriptions:
        if isinstance(grp.id, RingIndex):
            ring_list.append(grp)
        elif isinstance(grp.id, AtomIndex):
            atom_list.append(grp)
    if len(ring_list) == 0:
        mol_querier = MoleculeQuerier(mol, atom_list)
        res = mol_querier.query(q_mols)
        print(res)
    else:
        ri = mol.GetRingInfo()
        atominfo = ri.AtomRings()
        smi = Chem.MolToSmiles(mol)
        smi_without_numbers = re.sub(r'\d+', '', smi)
        for ring in ring_list:
            orig_atom_list = copy.deepcopy(atom_list)
            orig_mol = copy.deepcopy(mol)
            ring_id = ring.id
            atom_idx = atominfo[ring_id]
            for idx in atom_idx:
                rw_mol = Chem.RWMol(orig_mol)
                add_idx = rw_mol.AddAtom(Chem.Atom(0))
                rw_mol.AddBond(idx, add_idx, Chem.BondType.SINGLE)
                # mol = rw_mol.GetMol()
                smi =   Chem.MolToSmiles(rw_mol.GetMol(),rootedAtAtom=0, canonical=False)
                try:
                    # my_mol = Chem.Mol(Chem.MolFromSmiles(smi))
                    # my_mol = Chem.RWMol(Chem.MolFromSmiles(smi))
                    my_mol = Chem.MolFromSmiles(smi)
                    if my_mol is None:
                        continue
                except Exception as e:
                    print(f"Add position error: {e}")
                    continue
                desc = GroupDesc(id=AtomIndex(idx+1))
                desc.symbol = ring.symbol
                desc.script = ring.script
                desc.prime = ring.prime
                desc.multiple = ring.multiple
                for atom in orig_atom_list:
                    if atom.id > idx:
                        atom.id = AtomIndex(atom.id + 1)
                orig_atom_list.append(desc)
                
                mol_querier = MoleculeQuerier(my_mol, orig_atom_list)
                try:
                    # import pdb; pdb.set_trace()
                    res = mol_querier.query(q_mols)
                except Exception as e:
                    print(f"Query error: {e}")
                    continue
                for result in res:
                    if result is not None:
                        print(res)
                        exit(0)

