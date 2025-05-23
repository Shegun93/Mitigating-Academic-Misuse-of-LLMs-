import re

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = Flask(__name__)


MODEL_NAME = "/home/shegun93/n_projects/nairs-2e"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)
USE_QUANTIZATION_CONFIG = True


def load_model_and_tokenizer():
    """
    Function that loads the model and the tokenizer to memory
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        quantization_config=quantization_config if USE_QUANTIZATION_CONFIG else None,
        low_cpu_mem_usage=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()


def normalize_options_format(text):
    # Fix any variant like 'D.' or 'C.' to 'D:' or 'C:'
    for letter in ["A", "B", "C", "D"]:
        text = re.sub(rf"{letter}\.", f"{letter}:", text)

    # Ensure each option starts on a new line for reliable splitting
    for letter in ["A:", "B:", "C:", "D:"]:
        text = text.replace(letter, f"\n{letter}")

    # Clean leading/trailing whitespace
    return text.strip()


def process_model_output(output):
    output = normalize_options_format(output)

    question_match = re.search(r"^(.*?)\nA:", output, re.DOTALL)
    Question = question_match.group(1).strip() if question_match else None

    options_match = re.search(
        r"\n(A:.*?)(?=\nAnswer:|\nCorrect Option:|\nExplanation:|$)", output, re.DOTALL
    )
    options_text = options_match.group(1).strip() if options_match else None

    if options_text:
        options_lines = options_text.split("\n")
        Options = {}
        for line in options_lines:
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                Options[key] = value
    else:
        Options = None

    explanation_match = re.search(r"Explanation:(.*)", output, re.DOTALL)
    Explanation = explanation_match.group(1).strip() if explanation_match else None

    correct_option_match = re.search(r"(?:Answer|Correct Option): ([A-D])", output)
    Answer = correct_option_match.group(1) if correct_option_match else None

    if Question and Options and Explanation and Answer:
        return {
            "question": Question,
            "options": Options,
            "explanation": Explanation,
            "answer": Answer,
        }
    return {"raw_response": output}


@app.route("/", methods=["GET", "POST"])
def index():
    """Post queries to Model"""
    if request.method == "POST":
        context = request.form.get("prompt")
        if not context.strip():
            return render_template("index.html", error="Please enter a valid prompt.")

        prompt = f"""You are a helpful assistant that generates science and physics assessment questions.

Example:
Context: When soldiers march across a suspension bridge...
Question: Why are marching soldiers advised to break step on bridges?
Options:
A: To reduce air resistance
B: To prevent resonance
C: To minimize friction
D: To decrease bridge weight
Answer: B
Explanation: Marching soldiers are advised to break step on bridges to prevent resonance. When soldiers march in unison, their rhythmic footsteps can match the bridge's natural frequency. This matching of frequencies can cause the bridge to oscillate with increasing amplitude, potentially leading to structural damage. Breaking step ensures that the periodic force isn't applied at the bridge's natural frequency, preventing dangerous resonance effects.

Now do the same for the context below.

Context: {context}
Question:"""
        # prompt = request.form.get("prompt")
        # if not context.strip():
        #     return render_template("index.html", error="Please enter a valid prompt.")

        inputs = tokenizer(prompt.strip(), return_tensors="pt", truncation=True).to(
            model.device
        )
        output = model.generate(
            **inputs, max_new_tokens=500, do_sample=True, temperature=0.8
        )
        raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_part = raw_output.split("Question:")[-1].strip()

        processed_output = process_model_output(generated_part)
        return render_template("index.html", result=processed_output)

    return render_template("index.html")


@app.route("/check_answer", methods=["POST"])
def check_answer():
    """Get Response back to the users"""

    user_answer = request.form.get("user_answer").strip().upper()
    correct_option = request.form.get("correct_option")
    explanation = request.form.get("explanation")

    if user_answer not in ["A", "B", "C", "D"]:
        return jsonify(
            {
                "status": "error",
                "message": "Please enter a valid option (A, B, C, or D).",
            }
        )

    if user_answer == correct_option:
        return jsonify({"status": "success", "message": f"Correct! "})
    return jsonify({"status": "error", "message": f"Incorrect!"})


if __name__ == "__main__":
    app.run(debug=False)
