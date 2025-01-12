import tiktoken
import json
import base64

MARKUSH_STRING_DEFINITION = """\
We define a markush string as: `SMILES<sep>EXTENSION`
where:
- `SMILES` is the basic molecular skeleton
- `EXTENSION` is a list of completions for the R-groups in the SMILES string, such as R-group definitions, substitutions, or modifications.
- The EXTENSION is optional, and is in XML format.
    - <a>[ATOM_INDEX]:[GROUP_NAME]</a> indicates that the atom at index ATOM_INDEX (which is the index of the corresponding asterisk in the SMILES string) in the SMILES string corresponds to the R-group named GROUP_NAME, starting from 0. In this case, the number of asterisks in the SMILES string should be equal to the number of <a> tags in the EXTENSION.
    - <r>[RING_INDEX]:[GROUP_NAME]</r> indicates that the ring at index RING_INDEX, which is the index of the rings in the SMILES string in the order of appearance. For example, the first-occurred ring has an index of 0.
    - <c>[CIRCLE_INDEX]:[CIRCLE_NAME]</c> indicates that the circle at index CIRCLE_INDEX in the SMILES string corresponds to the R-group named CIRCLE_NAME.
    - A special token <dum> is used to indicate a connection point in the SMILES string.
    - GROUP_NAME is the name of the R-group, which can be abbreviation or full name (e.g., R, X, Y, Z, Ph, Me, OMe, CF3, etc.).
    - If there the GROUP_NAME is a markush substituent group and there's subscription, the subscript is appended to the GROUP_NAME with a colon (e.g., R[1], R[2], R[3], etc.).
In the pdf parsing results, the OCR results of markush and molecular SMILES images are wrapped in a \caption tag.
"""

RGROUP_MAPPING_DEFINITION = """\
The R-group mapping is a dictionary that maps the R-group names to the corresponding SMILES strings. The R-group names are the keys, and the SMILES strings are the values. These mappings are extracted during the match between a markush string and a molecule instance.

Example 1:
- Markush: *c1*c(*)c2cc(*)c(*)cc2n1<sep><a>0:R[1]</a><a>2:X</a><a>4:R[a]</a><a>8:R[2]</a><a>10:R[3]</a>
- Molecular SMILES: c1(C(=O)O)cc(O)nc2ccc(C#N)cc12
- R-Group Mapping: {"R1": "O", "X": "C", "Ra": "C(=O)O", "R2": "C#N", "R3": "[H][H]"}

Example 2:
- Markush: *C(C(=*)N(*)*)Sc1nc2*:*:*:*c2c(=O)n1*<sep><a>0:R[2]</a><a>3:A[5]</a><a>5:R[4]</a><a>6:R[3]</a><a>11:A[4]?n</a><a>12:A[3]</a><a>13:A[2]</a><a>14:A[1]</a><a>19:R[1]</a>
- Molecular SMILES: C(C)(Sc1nc2ccccc2c(=O)n1C1CCCC1)C(=O)N(C)c1ccccc1
- R-Group Mapping: "substructure_map": {"R2": "C", "A5": "O", "R4": "c1ccccc1", "R3": "C", "A4": "C", "A3": "C", "A2": "C", "A1": "C", "R1": "C1CCCC1"}

Example 3:
- Markush: *C(=O)C(*)(*)Cc1ccc(C(=O)Oc2ccc(C(=N)N)cc2*)s1<sep><a>0:X</a><a>4:R[1]</a><a>5:R[2]</a><a>23:R[7]</a>
- Molecular SMILES: CC(C)(Cc1ccc(C(=O)Oc2ccc(C(=N)N)cc2F)s1)C(=O)Nc1cc(C(=O)O)cc(C(=O)O)c1
- R-Group Mapping: {"X": "Nc1cc(C(=O)O)cc(C(=O)O)c1", "R1": "C", "R2": "C", "R7": "F"}
"""

CONFIDENCE_SCORE_DEFINITION = """\
# Scoring Criteria:
- **High Confidence (80 to 99.9)**: The evidence is clear, the Markush structure matches precisely with the query molecule, and the claim requirements are unambiguously met.
- **Moderate Confidence (50 to 79)**: There are some ambiguities in the claim requirements or minor uncertainties in the match, but overall, the evidence leans towards infringement.
- **Low Confidence (20 to 49)**: The evidence is contradictory, the Markush structure and the query molecule have only partial overlap, or the claim requirements are broadly interpreted.
- **Very Low Confidence (0 TO 19)**: There is little to no evidence of infringement, with significant mismatches or misinterpretations of the claim requirements.
"""

def truncate_string(text, max_tokens=96000):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded = encoding.encode(text)
    truncated = encoded[:max_tokens]
    truncated_text = encoding.decode(truncated)
    return truncated_text

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def load_local_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
        
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw
def smiles_to_image(smiles, img_size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")
    mol_img = Draw.MolToImage(mol, size=img_size)
    buffered = BytesIO()
    mol_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
def get_image_message(image_data):
    image_message = {
        "type": "image_url",
        "image_url": {
            "url":  f"data:image/jpeg;base64,{image_data}"
        },
    }
    return image_message
"""
Demo of chatting with images:
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is in this image?",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                },
            },
        ]
    }
]
"""