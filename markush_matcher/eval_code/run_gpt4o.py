from langchain_openai import AzureChatOpenAI
modelGPT4o = AzureChatOpenAI(
    model='gpt-4o',
    azure_endpoint=OPENAI_ENDPOINT,
    openai_api_version="2024-03-01-preview",
    openai_api_key=OPENAI_KEY,
    temperature=0.2,
    max_retries=3,
)


json_schema = {
    "title": "substructure_analysis",
    "description": "Analyze the specific substituent values for the given molecule",
    "type": "object",
    "properties": {},
    "required": []
}

system_prompt = """\
You are an expert in molecular chemistry, tasked with analyzing a molecule and identifying the substituent values for specific positions in the molecule. You are provided with two types of inputs:

Markush Structure: This is a generalized molecular structure in which certain positions are marked with placeholders (R-groups) such as R[6], R[7], R[9], etc. These placeholders represent substituent groups in the molecule. Next I will give you the detailed description of the Markush Structure.
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
SMILES Notation: This is a specific representation of the molecule where each atom and bond is explicitly described, including the substituent groups corresponding to the placeholders in the Markush structure.
Goal: Your task is to analyze the given Markush Structure and SMILES Notation and determine the corresponding values of each R-group placeholder. The output should be a dictionary where the keys are the R-group placeholders (e.g., R[6], R[7]), and the values are the actual substituents taken from the SMILES Notation.

"""

user_prompt = """\
Identify the R-group Placeholders: The Markush Structure contains placeholders for R-groups, such as R[6], R[7], R[9]. These placeholders need to be mapped to the corresponding substituent groups found in the SMILES Notation.

Analyze the SMILES Notation: The SMILES Notation describes the actual molecular structure, including the specific substituents attached to the core structure. Use this notation to find the actual chemical groups that correspond to each R-group placeholder.

Generate the R-group Mapping In Sequence: For each R-group placeholder (e.g., R[6], R[7]), find the matching substituent from the SMILES string and generate a mapping. The final output should be a dictionary where the keys are the R-group placeholders and the values are the corresponding substituent values from the SMILES.
Please provide the values of each R-group placeholder based on the SMILES Notation in sequence of the R-group placeholders in the Markush Structure.
Next I will provide you two examples to help you understand the task.
Example 1
Input:
Markush Structure:
*c1ccc(CSc2nnc(NC(=O)Nc3ccc(*)c(*)c3*)s2)cc1<sep><a>0:R[6]</a><a>19:R[7]'</a><a>21:R[9]</a><a>23:R[4]</a>
SMILES Notation:
CCc1ccc(CSc2nnc(NC(=O)Nc3ccc(S(C)(=O)=O)c(C)c3C[C@H](C)O)s2)cc1
Expected Output:
{
  'R[6]': 'CC',
  'R[7]': 'S(C)(=O)=O',
  'R[9]': 'C',
  'R[4]': 'C[C@@H](O)C'
}
Example 2
Input:
Markush Structure:
*COc1*(*):*(*)c(*)c(*)c1NC(=O)C(*)NC(c1*c(*)c(*)c(*)c1)c1c(*)*:*cc1*<sep><a>0:Zyp</a><a>1:?z</a><a>4:R</a><a>5:R[15]</a><a>6:X</a><a>7:R[3]'</a><a>9:R[10]</a><a>11:Hvb</a><a>17:A</a><a>21:G</a><a>23:R[7]</a><a>25:R[5]</a><a>27:R[2]</a><a>31:R[8]</a><a>32:Y[8]</a><a>33:Y[8]</a><a>36:R[22]</a>
SMILES Notation:
[H]c1c(/C=C/CC)cc(C(NC(C#C)C(=O)Nc2c([H])c(OC[C@H](C)OC)c(C=C)c(C=C)c2OCN(C)C)c2c(CC)cccc2[C@@H](O)CO)cc1[C@@H](O)CO
Expected Output:
{
    "Zyp": "N(C)C",
    "R": "C",
    "R[15]": "C=C",
    "X": "C",
    "R[3]": "C=C",
    "R[10]": "OC[C@@H](OC)C",
    "Hvb": "[H]",
    "A": "C#C",
    "G": "C",
    "R[7]": "C=CCC",
    "R[5]": "[H]",
    "R[2]": "[C@H](CO)O",
    "R[8]": "[C@H](CO)O",
    "Y[8]": "C",
    "R[22]": "CC"
}
Now I will provide you with the actual input.
Markush Structure:
{markush}
SMILES Notation:
{smiles}

"""

def analysis_substituent(input_data):
    markush_structure = input_data['markush']
    smiles = input_data['smiles']
    user_prompt_format = user_prompt.replace("{markush}", markush_structure).replace("{smiles}", smiles)
    output_dict = {}
    required_keys = []
    for key, value in input_data['r_group_map'].items():
        output_dict[key] = {
            "type": "string",
            "description": f"The value of the {key} placeholder in the Markush Structure."
        }
        required_keys.append(key)
    json_schema['properties'] = output_dict
    json_schema['required'] = required_keys
    prompt_template = [
        ('system', system_prompt),
        ('user', user_prompt_format)
    ]
    structured_llm = modelGPT4o.with_structured_output(json_schema)
    response = structured_llm.invoke(prompt_template)
    target = input_data['r_group_map']
    return response, target

    