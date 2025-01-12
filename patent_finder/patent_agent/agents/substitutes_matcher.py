from .llm_utils import truncate_string, MARKUSH_STRING_DEFINITION, RGROUP_MAPPING_DEFINITION
from pydantic import BaseModel, Field, validator
import json

# LLM tool

class ReturnFormat(BaseModel):
    reasoning: str = Field(title="Reasoning", description="Explanation of the analysis process and the reasoning behind the final conclusion")
    r_group_matching: str = Field(title="R-Group Matching", description="R-group values from the substructure match result, in JSON format")

    @validator('r_group_matching', pre=True, always=True)
    def ensure_json_string(cls, v):
        if isinstance(v, dict):
            return json.dumps(v)
        else:
            try:
                return str(v)
            except:
                return 'Error in parsing R-group matching result'

system_prompt = """\
You are an expert in chemical patents and molecular representations. Your task is to verify the correctness of the substructure match result between the markush claim and the query molecule.

{MARKUSH_STRING_DEFINITION}

{RGROUP_MAPPING_DEFINITION}

Note that: 
- If you don't think the molecular matches the claim, return `r_group_matching` as an empty dictionary.
- The R-group values are considered correct iff when we replace the R-groups in the Markush structure with the corresponding values, we get the query molecule. Otherwise, the R-group values are incorrect.
- Be very cautious to any potential protection scope of the patent. If this sample is miss classified, it may lead to a serious legal issue.
- The substruture match shoul only be used as reference, since the matching algorithm may not be perfect.
- In your reasoning, compare each R-group definition with the query molecule, and analyze the correctness of the R-group values one by one.

After you analysis, provide a verified R-group mapping result explicitly.\
"""

system_prompt = system_prompt.format(MARKUSH_STRING_DEFINITION=MARKUSH_STRING_DEFINITION, RGROUP_MAPPING_DEFINITION=RGROUP_MAPPING_DEFINITION)

user_prompt = """\
**Markush Structure**: `{markush_string}`

**Query Molecule**:
{mole_string}

**R-Group Mapping Extracted by Chemical Software**:
```json
{r_group_mapping}
```

**R-Group Mapping Extracted by Neural network model**:
```json
{nn_result}
```
"""

def substitutes_matcher(llm, claim_dict, mole_dict, substructure_map, nn_result, claim_text, max_tokens=96000):
    """
    substitutes_matcher Version 2.0: Nov 13 2024
    """
    if isinstance(mole_dict, str):
        mole_dict = {'smiles': mole_dict}
    mole_string = json.dumps(mole_dict, indent=4)
    claim_text = truncate_string(claim_text, max_tokens)
    input_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt.format(
                markush_string=claim_dict['caption'],
                mole_string=mole_string,
                r_group_mapping=json.dumps(substructure_map, indent=4),
                nn_result=nn_result,
            ),
        }
    ]
    llm_result = llm.chat(input_messages, ReturnFormat)
    try:
        llm_result['r_group_matching'] = json.loads(llm_result['r_group_matching'])
    except:
        pass

    return llm_result