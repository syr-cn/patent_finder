from .llm_utils import truncate_string, MARKUSH_STRING_DEFINITION
from pydantic import BaseModel, Field
from typing import List
import json

class ReturnFormat(BaseModel):
    markush_index: int = Field(title="Markush Index",description="Index of the block that contains the Markush structure OCR result which is most relave.")
    requirement_indices: List[int] = Field(title="Requirement Indices",description="List of indices of the blocks within the patent document that are relevant to the assessment of the query molecule.")
    requirement_details: List[str] = Field(title="Requirement Details",description="Explanation of why each block is relevant to the infringement analysis.")

system_prompt = """\
You are an expert in chemical patent analysis with a focus on molecular structures and patent claims.

# Task Overview:
- Extract which of the provided blocks contain the markush structure that actually protect the target molecule.
- Extract the key sections of the patent that have been directly contributed to the infringement analysis.
- Direct contributions include R-Group definitions infringement, and other relevant claim requirements.
- For each extracted block, explain why is it relevant to the infringement analysis by summarizing the key fact claimed in the block.
- Then length of requirement_indices and requirement_details should be the same. Each element in the list `requirement_details` should correspond to the block index in `requirement_indices`.

# Special Instructions:
- Focus on the detailed claim requirements that specify the allowed variations, substitutions, or conditions for the Markush groups.
- Be as precise as possible in locating the most relevant blocks.
- Try to be concise. You should identify the most relevant {max_blocks} blocks that directly contribute to the infringement analysis.
- Return as few requirement block indices as you can. If one block is enough to provide a clear answer, only one block is needed.
- Sort the block indices by their contribution to the infringement analysis.

{MARKUSH_STRING_DEFINITION}
"""
system_prompt = system_prompt.replace('{MARKUSH_STRING_DEFINITION}', MARKUSH_STRING_DEFINITION)

user_prompt = """\
# Target Molecule:
{target_smiles}

# Patent PDF Blocks:
{block_text}

# Infringement Analysis Results:
Infringement: {input_is_protected}
Analysis: {input_reasoning}
"""


def fact_checker(llm, claim_dict, target_smiles, blocks, input_is_protected, input_reasoning, max_blocks=5, max_tokens=96000):
    """
    fact_checker Version 2.0: Nov 13 2024
    """
    text_blocks = [{
        'page': block['page'],
        'index': idx,
        'str': block['str'],
        'class': block['class'],
    } for idx, block in enumerate(blocks)]
    block_text = json.dumps(text_blocks, indent=2)
    block_text = truncate_string(block_text, max_tokens)

    input_messages = [
        {
            "role": "system",
            "content": system_prompt.format(max_blocks=max_blocks)
        },
        {
            "role": "user",
            "content": user_prompt.format(
                # markush_string=claim_dict['caption'],
                target_smiles=target_smiles,
                block_text=block_text,
                input_is_protected=str(input_is_protected),
                input_reasoning=input_reasoning
            )
        }
    ]
    llm_result = llm.chat(input_messages, ReturnFormat)

    return llm_result