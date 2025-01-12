import sys
import os
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from models import get_chain,get_llm
from utils import truncate_string
from utils import config
# LLM tool
from langchain_core.prompts import ChatPromptTemplate
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

# system_prompt = """\
# You are an expert in the field of molecular determination. Next, I will provide you with the full content of a patent and a specific molecular formula. Please analyze the claims section of the patent and determine whether this molecular formula is protected by the patent.
# """
system_prompt = """\
You are an expert in the field of molecular determination. Next, I will provide you with the full content of a patent, the markush formula that contains the core markush structure of the patent's claim requirements, and a specific molecular formula. Please analyze the claims section of the patent and determine whether this molecular formula is protected by the patent.
**Markush Structure Description Rules**: 
        1. **Basic Format:**
    - The description is in the format: `SMILES<sep>EXTENSION`.
    - `SMILES` is the RDKit-compatible SMILES representation of the molecule, generated using the following command:
        ```
        Chem.MolToSmiles(mol, rootedAtAtom=0, canonical=False, isomericSmiles=True)
        ```
    - `<sep>` is a separator between the SMILES string and the following extension.

    2. **Extension:**
    - The extension provides additional descriptions for the SMILES structure, such as substituent groups, connection points, and repeating groups. The extension is written in XML-style tags.

    3. **Extension Components:**
    - **Substituent Group:** 
        ```
        <a>[ATOM_INDEX]:[GROUP_NAME]</a>
        ```
        - `ATOM_INDEX` is the atomic index (starting from 0) of the substituent.
        - `GROUP_NAME` is the substituent name (e.g., R, X, Y, Z, Ph, Me, OMe, CF3). If it is a Markush substituent, it can use a subscript or superscript, such as R[1] or R[3].
    - **Ring Group:** 
        ```
        <r>[RING_INDEX]:[GROUP_NAME]</r>
        ```
        - `RING_INDEX` is the ring index (starting from 0).
        - `GROUP_NAME` refers to substituents attached to any position on the ring.
    - **Aromatic Ring Group (rare):**
        ```
        <c>[CIRCLE_INDEX]:[CIRCLE_NAME]</c>
        ```
        - `CIRCLE_INDEX` is the aromatic ring's index.
        - `CIRCLE_NAME` refers to the name of the ring.

    4. **Special Tokens:**
    - `<dum>`: Represents a connection point, which is an undefined attachment point for further substituents.

    5. **Repeating Units:**
    - If the structure involves repeating units, it is represented with:
        ```
        ?[n]
        ```
        - `n` is the number of repeating units (can be a range like 1-3 or an exact number).
    - For repeating units in a ring system, the notation is as follows:
        ```
        <r>[RING_INDEX]:[GROUP_NAME]?n
        ```

    6. **Example:**
    ```
    *C(O)c1cc(C(=O)N(*)*)cc(-c2*ccc*2)c1<sep><a>0:CF3</a><a>9:R[3]</a><a>10:R[2]</a><a>14:X</a><a>18:Y</a><r>1:R[1]?1-3</r>
    ```
    This describes a molecule with a core SMILES string and substituent groups attached to specific atom positions, with ring substituents replicated 1 to 3 times.

In your reasoning:
- briefly recall the molecular structure, and the core markush structure of the patent's claim requirements
- Explain your reasoning step-by-step
"""

user_prompt = """\
Here is the content of the patent: {text}
Here is the markush that contains the core markush structure of the patent's claim requirements:{markush}

There is a molecule with the SMILES formula '{target_smiles}'. Based on your analysis, provide the judgment you think is more likely to be correct.
You should provide a boolean value for 'is_protected' and a string value for 'reasoning'.
"""


def llm_direct_analysis_update(
        full_text,
        query_molecule,
        markush
    ):
    full_text = truncate_string(full_text)
    if config.llm_name == 'llama':
        config.llm_name = 'vllm'
    if config.llm_name in ['gpt-o1', 'moonshot']:
        user_prompt_o1 = user_prompt.replace("{text}", full_text).replace("{target_smiles}", query_molecule).replace("{markush}", markush)
        response = get_chain(json_schema, system_prompt, user_prompt_o1)
        # cleaned_response = response.replace("```","").replace('json\n', '').replace('\n', '', 1).replace('\n}', '}')
        return response
    chain = get_chain(json_schema, system_prompt, user_prompt)
    return chain.invoke({
        'text': full_text,
        'target_smiles': query_molecule,
        'markush': markush
    })
    
    
async def llm_direct_analysis_update_async(
        full_text,
        query_molecule,
        markush
    ):
    full_text = truncate_string(full_text)
    if config.llm_name == 'llama':
        config.llm_name = 'vllm'
    if config.llm_name in ['gpt-o1', 'moonshot']:
        user_prompt_o1 = user_prompt.replace("{text}", full_text).replace("{target_smiles}", query_molecule).replace("{markush}", markush)
        response = get_chain(json_schema, system_prompt, user_prompt_o1)
        # cleaned_response = response.replace("```","").replace('json\n', '').replace('\n', '', 1).replace('\n}', '}')
        return response
    chain = get_chain(json_schema, system_prompt, user_prompt)
    return await chain.ainvoke({
        'text': full_text,
        'target_smiles': query_molecule,
        'markush': markush
    })
    
    
    