from .tools.substructure_match import substructure_match
from .tools.substructure_match_nn import substructure_match_nn

from .agents.sketch_extractor import sketch_extractor
from .agents.substitutes_matcher import substitutes_matcher
from .agents.requirements_examinator import requirements_examinator
from .agents.fact_checker import fact_checker
from .agents.manager import manager


def parse_text(blocks):
    claim_dict_list = []
    for block in blocks:
        if block['class'] == 'molecule' and '<sep>' in block['str']:
            smi = block['str']
            smi = smi.replace("\\begin{molecule}", "")
            smi = smi.replace("\\end{molecule}", "")
            smi = smi.replace("\\n", "")
            smi = smi.replace("\n", "")
            claim_dict_list.append({"caption": smi})
    return claim_dict_list

class PatentAgent():
    def __init__(self, max_tokens=96000):
        self.max_tokens = max_tokens

    def filter_claims(self, claim_list):
        filtered_claims = [claim_list[0]]
        return filtered_claims

    def claim_check(self, llm, claim_dict, text_dict, mole_dict):
        software_result = substructure_match(claim_dict["caption"], mole_dict['smiles'])

        if isinstance(software_result['substructure_map'], dict):
            substructure_map = software_result['substructure_map']
            print(f'Software Structure matching success. Match result: {substructure_map}')
        else:
            substructure_map = None
            print(f'Software {software_result["substructure_map"]}.')
        try:
            nn_result = substructure_match_nn(markush=claim_dict["caption"], smiles=mole_dict['smiles'])
            print(f'Neural Network Structure matching success. Match result: {nn_result}')
        except Exception as e:
            nn_result = {'r_group_map': None, 'reasoning': str(e)}
            print('Neural Network Structure matching failed.')

        verified_substructure_map = substitutes_matcher(llm, claim_dict, mole_dict, substructure_map, nn_result['r_group_map'], text_dict['claim'], self.max_tokens)
        verified_substructure_map['software_result'] = software_result
        verified_substructure_map['nn_result'] = nn_result
        requirements_examine_result = requirements_examinator(llm, claim_dict, mole_dict, verified_substructure_map['r_group_matching'], text_dict['claim'], self.max_tokens)
        infringement_report = manager(llm, claim_dict, mole_dict, text_dict, verified_substructure_map, requirements_examine_result, self.max_tokens)

        result = {}
        result['mole_dict'] = mole_dict
        result['claim_dict'] = claim_dict
        result['rgroups_verify_result'] = verified_substructure_map
        result['requirements_examine_result'] = requirements_examine_result
        result['infringement_report'] = infringement_report
        result['is_protected'] = requirements_examine_result['is_protected']
        return result
    
    def extract_text_from_blocks(self, llm, claim_list, blocks, max_blocks=150):
        claim_dict = self.filter_claims(claim_list)[0]
        key_block_result = sketch_extractor(llm, claim_dict, blocks, max_blocks=max_blocks, max_tokens=self.max_tokens)
        key_blocks = blocks[key_block_result['begin_index']:key_block_result['end_index']]
        block_text_dict = {
            'claim': ' '.join([block['str'] for block in key_blocks]),
            'full': ' '.join([block['str'] for block in blocks])
        }
        return block_text_dict, claim_dict

    def claim_check_pdf(self, llm, blocks, mole_dict):
        claim_list = parse_text(blocks)

        block_text_dict, claim_dict = self.extract_text_from_blocks(llm, claim_list, blocks, max_blocks=150)
        result = self.claim_check(llm, claim_dict, block_text_dict, mole_dict)
        answer_block_result = fact_checker(llm, claim_list[0], mole_dict, blocks, result['requirements_examine_result']['is_protected'], result['requirements_examine_result']['reasoning'], max_blocks=10, max_tokens=self.max_tokens)

        result['answer_block_result'] = answer_block_result
        return result