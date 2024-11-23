from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json

app = Flask(__name__)

def load_tutorial_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['tutorial']

def format_tutorial_for_llm(tutorial_data):
    tutorial_text = ""
    for section in tutorial_data:
        tutorial_text += f"**{section['section']}**\n{section['content']}\n\n"
    return tutorial_text

def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

def generate_response(question, tutorial_text, tokenizer, model):
    input_text = f"Here is a Python tutorial content:\n\n{tutorial_text}\n\nQuestion: {question}\nAnswer:"
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    max_input_length = 1024  
    if input_ids.shape[1] > max_input_length:
        input_ids = input_ids[:, -max_input_length:]

    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
    pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask=attention_mask,  
            pad_token_id=pad_token_id,  
            max_new_tokens=100,  
            no_repeat_ngram_size=2,  
            temperature=0.7  
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    response = response.split('Answer:')[-1].strip()
    return response



tutorial_data = load_tutorial_from_json('python_tutorial.json')
tutorial_text = format_tutorial_for_llm(tutorial_data)
tokenizer, model = load_gpt2_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    response = generate_response(user_input, tutorial_text, tokenizer, model)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
