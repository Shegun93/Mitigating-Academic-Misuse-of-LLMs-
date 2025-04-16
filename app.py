from flask import Flask, render_template, request, jsonify
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = Flask(__name__)

# Model Configuration
model_name = "nairs-2b-fine-tuned"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
use_quantization_config = True

# Load Model and Tokenizer
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        quantization_config=quantization_config if use_quantization_config else None,
        low_cpu_mem_usage=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Process Model Output
def process_model_output(output):
    question_match = re.search(r"^(.*?)A:", output, re.DOTALL)
    question = question_match.group(1).strip() if question_match else None

    options_match = re.search(r"(A: .*?)(?=Correct Option:|Explanation:|$)", output, re.DOTALL)
    options = options_match.group(1).strip().split("\n") if options_match else None

    explanation_match = re.search(r"Explanation:(.*)", output, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else None

    correct_option_match = re.search(r"Correct Option: ([A-D])", output)
    correct_option = correct_option_match.group(1) if correct_option_match else None

    if question and options and explanation and correct_option:
        return {
            "question": question,
            "options": options,
            "explanation": explanation,
            "correct_option": correct_option,
        }
    else:
        return {"raw_response": output}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if not prompt.strip():
            return render_template("index.html", error="Please enter a valid prompt.")

        inputs = tokenizer(prompt.strip(), return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=500)
        raw_output = tokenizer.decode(output[0], skip_special_tokens=True)

        processed_output = process_model_output(raw_output)
        return render_template("index.html", result=processed_output)
    
    return render_template("index.html")

@app.route("/check_answer", methods=["POST"])
def check_answer():
    user_answer = request.form.get("user_answer").strip().upper()
    correct_option = request.form.get("correct_option")
    explanation = request.form.get("explanation")

    if user_answer not in ["A", "B", "C", "D"]:
        return jsonify({"status": "error", "message": "Please enter a valid option (A, B, C, or D)."})
    
    if user_answer == correct_option:
        return jsonify({"status": "success", "message": f"Correct! Explanation: {explanation}"})
    else:
        return jsonify({"status": "error", "message": f"Incorrect! Explanation: {explanation}"})

if __name__ == "__main__":
    app.run(debug=True)
