from transformers import AutoTokenizer, T5ForConditionalGeneration
from train_group_pred import make_data

model_path = 'results/pretrain-large-dataNov6-Nov06/final/'
# model_path = 'results/tune-large-Nov07/final/'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

model.eval()

def predict(input_text, model, tokenizer, max_length=512):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)

    output_ids = model.generate(input_ids, max_length=max_length, eos_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

input_data = {
    "markush": "*N1CC2C(=NN(*)C=2CC1)N(*)*<sep><a>0:R[4]</a><a>7:R[1]</a><a>12:R[3]</a><a>13:R[2]</a>",
    "smiles": "CC(=O)N1CCc2c(c(N3CCc4ccccc43)nn2CCCCCCC)C1",
    'r_group_map': {}
}
input_text = make_data(input_data)['input_text']

predicted_output = predict(input_text, model, tokenizer)
print(predicted_output)