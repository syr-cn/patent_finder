
import requests
import json
def substructure_match_nn(markush: str, smiles: str, endpoint: str=MARKUSH_MATCH_ENDPOINT):
    input_data = {'markush': markush, 'smiles': smiles}
    json_data = json.dumps(input_data)
    response = requests.post(endpoint, data=json_data, headers={'Content-Type': 'application/json'})
    result = response.json()
    return result

if __name__ == '__main__':
    demo_data = {
        "markush": "*N1CC2C(=NN(*)C=2CC1)N(*)*<sep><a>0:R[4]</a><a>7:R[1]</a><a>12:R[3]</a><a>13:R[2]</a>",
        "smiles": "CC(=O)N1CCc2c(c(N3CCc4ccccc43)nn2CCCCCCC)C1"
    }
    result = substructure_match_nn(**demo_data)
    print(result)