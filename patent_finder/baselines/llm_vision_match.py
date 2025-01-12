import sys
import os
import json
from langchain_core.messages import HumanMessage, SystemMessage
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from models import get_chain
from utils import truncate_string
# LLM tool

json_schema = {
    "title": "llm_patent_analysis",
    "description": "Analyze the claims section of a patent and determine whether a given molecular formula is protected by the patent",
    "type": "object",
    "properties": {
        "is_protected": {
            "type": "boolean",
            "description": "Final conclusion on whether the query molecule matches the claims"
        },
        "reasoning": {
            "type": "string",
            "description": "Values of each R-Group if the molecule matches the claims",
        }
    },
    "required": ["is_protected", "reasoning"]
}

system_prompt = """\
You are an expert in the field of molecular determination. Next, I will provide you with the full content of a patent, the image of core markush structure of the patent, and a specific molecular formula. Please analyze the claims section of the patent and determine whether this molecular formula is protected by the patent.

In your reasoning:
- briefly recall the molecular structure, and the core markush structure of the patent's claim requirements
- Explain your reasoning step-by-step
"""

user_prompt = """\
Full text of the patent:
```
{full_text}
```

Query SMILES:
`{target_smiles}`
"""

def llm_visual_match_func_update(claim_dict, target_smiles, full_text):
    '''
    1. compare the reference markush structure with the markush image, correct the markush image if necessary
    2. compare the markush image with the target smiles, return the substructure match result
    '''
    reference_markush = claim_dict['caption']
    image_path = claim_dict['path']
    full_text = truncate_string(full_text)
    import base64
    with open(image_path, "rb") as image_file:
    # 读取图片并转换为Base64编码
        base64_encoded_data = base64.b64encode(image_file.read())
    # 转换为字符串
    image_base64 = base64_encoded_data.decode("utf-8")
    content = [
        {"type": "text", "text": f"Here is the content of the patent: {full_text}"},
        { "type": "text", "text": "Here is the markush image that contains the core markush structure of the patent's claim requirements." },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
        },
        {
        "type": "text",
        "text": f"There is a molecule with the SMILES formula '{target_smiles}'. Based on your analysis, \
                provide the judgment you think is more likely to be correct. \
You should provide a boolean value for 'is_protected' and a string value for 'reasoning'.",
        },
    ]
    message = [SystemMessage(content=system_prompt) ,HumanMessage(content=content)]
    chain = get_chain(json_schema, system_prompt, user_prompt, messages=message)
    if isinstance(chain, dict):
        return chain
    # import asyncio
    # from langchain_community.callbacks import get_openai_callback
    # with get_openai_callback() as cb: 
    #     chain.invoke({ 'target_smiles': data_dict['target_smiles'], 'full_text': data_dict['pdf_content'] })
    # total_tokens = cb.total_tokens
    result = chain.invoke({})
    return result