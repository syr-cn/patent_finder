from .llm_utils import truncate_string, MARKUSH_STRING_DEFINITION
from pydantic import BaseModel, Field
import json

# LLM tool

class ReturnFormat(BaseModel):
    reasoning: str = Field(..., description="The reasoning behind the extraction of the key information.")
    begin_index: int = Field(..., description="The index of the block where the key information of the patent starts.")
    end_index: int = Field(..., description="The index of the block where the key information of the patent ends.") 

system_prompt = """\
You are an expert in analyzing chemical patent documents. Your task is to identify the sections of a patent that contain key information about a specific Markush structure and its associated claim requirements.

# Task Overview:
- Read through the patent document provided.
- Identify and mark the beginning and end indices of sections that discuss the Markush structure and its detailed claim requirements.
- These sections are critical as they determine the scope of patent protection for the molecule.

# Special Instructions:
- Pay close attention to descriptions of the molecular structure, especially the parts that detail variations and substitutions allowed under the patent.
- Highlight any discussions related to the legal scope of the patent, including examples, embodiments, and specific conditions mentioned for the Markush groups.
- Do not miss any important information.
- Try to be concise. No more than {max_blocks} blocks after summarization are tolerable for this task.

Input the full patent text below and use the provided Markush structure as a reference to guide your extraction process.

{MARKUSH_STRING_DEFINITION}
"""
system_prompt = system_prompt.replace('{MARKUSH_STRING_DEFINITION}', MARKUSH_STRING_DEFINITION)

user_prompt = """\
# Core Markush Structure:
{markush_string}

# Patent PDF Blocks:
{block_text}

"""


def sketch_extractor(llm, claim_dict, blocks, max_blocks=150, max_tokens=96000):
    """
    sketch_extractor Version 2.0: Nov 13 2024
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
            "content": user_prompt.format(markush_string=claim_dict['caption'], block_text=block_text)
        }
    ]
    llm_result = llm.chat(input_messages, ReturnFormat)

    return llm_result