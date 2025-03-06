import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load pre-trained T5 model and tokenizer
model_name = os.environ["MODEL_NAME"]
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/gen', methods=['POST'])
def generate_text():
    input_text = request.json['input_text']
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({'output_text': output_text})

if __name__ == '__main__':
    app.run(debug=True)



