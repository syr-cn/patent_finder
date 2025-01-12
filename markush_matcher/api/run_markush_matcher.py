import logging
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]: %(message)s"
)
logger = logging.getLogger(__name__)

import torch
if torch.cuda.is_available():
    print("CUDA is available. GPU can be used.", flush=True)
else:
    print("CUDA is not available. Using CPU.", flush=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, T5ForConditionalGeneration
model_path = 'PATH_TO_MODEL'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model = model.to(device)

def make_data(markush, smiles):
    input_text = f"Markush: {markush}\t SMILES: {smiles}\t Predict the values of each substituent in markush structure."

    return input_text

examples = [
    {
        "markush": "*N1CC2C(=NN(*)C=2CC1)N(*)*<sep><a>0:R[4]</a><a>7:R[1]</a><a>12:R[3]</a><a>13:R[2]</a>",
        "smiles": "CC(=O)N1CCc2c(c(N3CCc4ccccc43)nn2CCCCCCC)C1",
    }
]

def predict(input_text, model, tokenizer, max_length=512):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)

    output_ids = model.generate(input_ids.to(device), max_length=max_length, eos_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_text = output_text.replace('{ ', '{').replace('""', '"')
    return output_text

class RequestFormat(BaseModel):
    smiles: str
    markush: str

class ResponseFormat(BaseModel):
    smiles: str
    markush: str
    r_group_map: str

@app.post("/markush_match", response_model=ResponseFormat)
async def markush_match(req: RequestFormat = Body(RequestFormat, examples=examples)):
    print(f'Request received: {req}')
    try:
        assert req.smiles and req.markush, "Invalid input"
        r_group_map = predict(make_data(req.markush, req.smiles), model, tokenizer)
        response = ResponseFormat(smiles=req.smiles, markush=req.markush, r_group_map=r_group_map)
        print(f'Response: {response}')
        return response
    except Exception as e:
        print(f"Error Occurred: {e}")
        logger.info(f"Error Occurred: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7892, log_level="info")