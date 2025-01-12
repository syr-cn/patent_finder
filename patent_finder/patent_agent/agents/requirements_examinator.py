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

Special Attention Required:
- In your analysis, pay particular attention to the matching of R-group values from the substructure match result with the stipulations outlined in the claim requirement text. It is crucial to meticulously assess whether each R-group and its corresponding substitutions align with the conditions specified for patent coverage. Ensure that this careful examination is clearly reflected in your step-by-step reasoning.
- Though the molecule's skeleton may match the markush, you must verify the value of each R-group value by comparint them against the claim requirement text one by one. Also mention the corresponding definition of each R-group in your analysis.
- Be careful to draw a "is protected" conclusion. If any of the R-group values are not protected by the claim, the molecule is not protected by the patent.
- If you do find out every R-group value is protected by the claim, don't be afraid to draw a "is protected" conclusion. But make sure to provide a clear reasoning for your conclusion.

# Output Expectation:
- **Analysis:** Carefully analyze the R-group substitutions in the query molecule based on the claim requirement text. Ensure that each R-group and its corresponding substitutions align with the conditions specified for patent coverage.
- **Prediction:** Provide a definitive conclusion on whether the query molecule is covered under the protection scope of the patent.
"""
system_prompt = system_prompt.format(MARKUSH_STRING_DEFINITION=MARKUSH_STRING_DEFINITION, CONFIDENCE_SCORE_DEFINITION=CONFIDENCE_SCORE_DEFINITION)

user_prompt = """\
1. **Markush Claim**: `{markush_string}`

2. **Current Substructure Match Result:**
```json
{structure_match}
```

3. **Query Molecule**:
{mole_string}

4. **Claim Requirement Text:**
```
{claim_text}
```
"""


def requirements_examinator(llm, claim_dict, mole_dict, match_result, claim_text, max_tokens=96000):
    """
    requirements_examinator Version 2.0: Nov 13 2024
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
                structure_match=json.dumps(match_result, indent=4),
                claim_text=claim_text,
            )
        }
    ]
    llm_result = llm.chat(input_messages, ReturnFormat)
    try:
        llm_result['r_group_matching'] = json.loads(llm_result['r_group_matching'])
    except:
        pass

    return llm_result