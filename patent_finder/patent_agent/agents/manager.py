from .llm_utils import truncate_string, MARKUSH_STRING_DEFINITION, CONFIDENCE_SCORE_DEFINITION
from pydantic import BaseModel, Field
import json

# LLM tool

class ReturnFormat(BaseModel):
    reasoning: str = Field(title="Reasoning", description="Explanation of the analysis process and the reasoning behind the final conclusion")
    is_protected: bool = Field(title="Is Protected", description="Final conclusion on whether the query molecule matches the claims")

system_prompt = """\
You are an expert in chemical patents and molecular representations. You are tasked with determining whether a given molecule is covered under the protection scope of a specific patent.
{MARKUSH_STRING_DEFINITION}

You will be provided with some information about a molecule, and a relavant markush patent.
Some neural network models have already been used to match the molecule with the markush claim, and analyze the R-group substitutions in the query molecule based on the claim requirement text.
Your task is to carefully re-organize these information into a comprehensive infringement report, and provide a definitive conclusion on whether the query molecule is covered under the protection scope of the patent.

Your infringement report must include:
- The structure of the query molecule.
- The structure of the core Markush protected in the patent, and the corresponding R-group definitions.
- The value of each R-group value defined in the Markush claim.
- Whether the substitutions of each R-group in the query molecule align with the conditions specified for patent coverage.
- Whether there are any other requirements in the patent, which the molecule infringes.
- A definitive conclusion on whether the query molecule is covered under the protection scope of the patent.
"""
system_prompt = system_prompt.format(MARKUSH_STRING_DEFINITION=MARKUSH_STRING_DEFINITION, CONFIDENCE_SCORE_DEFINITION=CONFIDENCE_SCORE_DEFINITION)

user_prompt = """\
1. **Markush Claim**: `{markush_string}`

2. **Query Molecule**:
{mole_string}

3. **Claim Requirement Text:**
```
{claim_text}
```

4. **Current Substructure Match Result:**
```json
{structure_match}
```

5. **Current Requirement Examination Result:**
```json
{requirements_examination}
```
"""


def manager(llm, claim_dict, mole_dict, text_dict, verified_substructure_map, requirements_examine_result, max_tokens=96000):
    """
    manager Version 2.0: Nov 13 2024
    """
    if isinstance(mole_dict, str):
        mole_dict = {'smiles': mole_dict}
    mole_string = json.dumps(mole_dict, indent=4)
    claim_text = text_dict['claim']
    claim_text = truncate_string(claim_text, max_tokens)
    
    match_result = verified_substructure_map['r_group_matching']
    requirements_examination = requirements_examine_result
    
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
                claim_text=claim_text,
                structure_match=json.dumps(match_result, indent=4),
                requirements_examination=json.dumps(requirements_examination, indent=4),
            )
        }
    ]
    llm_result = llm.chat(input_messages, ReturnFormat)
    try:
        llm_result['r_group_matching'] = json.loads(llm_result['r_group_matching'])
    except:
        pass

    return llm_result