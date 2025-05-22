"""
Streamlit version API gateway
"""

import re

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

MODEL_NAME = "/home/shegun93/n_projects/nairs-2e"
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)
USE_QUANTIZATION_CONFIG = True


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    ATTN_IMPLEMENTATION = "flash_attention_2"
else:
    ATTN_IMPLEMENTATION = "sdpa"

st.info(f"Using attention implementation: {ATTN_IMPLEMENTATION}")
st.info(f"Loading model: {MODEL_NAME}")


@st.cache_resource
def load_model_and_tokenizer():
    """
    Function that loads the model and the tokenizer to memory

    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        quantization_config=QUANTIZATION_CONFIG if USE_QUANTIZATION_CONFIG else None,
        low_cpu_mem_usage=True,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to(DEVICE)
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()


def process_model_output(output):
    """
    Post processing functions.
    """
    question_match = re.search(r"^(.*?)A:", output, re.DOTALL)
    question = question_match.group(1).strip() if question_match else None

    options_match = re.search(
        r"(A: .*?)(?=Correct Option:|Explanation:|$)", output, re.DOTALL
    )
    options = options_match.group(1).strip().split("\n") if options_match else None

    explanation_match = re.search(r"Explanation:(.*)", output, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else None

    correct_option_match = re.search(r"Correct Option: ([A-D])", output)
    correct_option = correct_option_match.group(1) if correct_option_match else None

    # If all required fields exist, return structured data
    if question and options and explanation and correct_option:
        return {
            "question": question,
            "options": options,
            "explanation": explanation,
            "correct_option": correct_option,
            "raw_response": output,
        }

    return {"raw_response": output}


st.title("Student Assessment App")

# Step 1: Prompt Input
st.subheader("Enter Your Prompt")
prompt = st.text_area("Type your question or text here", height=150)

if st.button("Generate"):
    if not prompt.strip():
        st.error("Please enter a valid prompt.")
    else:
        with st.spinner("Generating question and options..."):
            # Generate the response using the model
            inputs = tokenizer(prompt.strip(), return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=500)
            raw_output = tokenizer.decode(output[0], skip_special_tokens=True)

            # Process the model output
            processed_output = process_model_output(raw_output)

            if "question" in processed_output and "options" in processed_output:
                st.session_state["question"] = processed_output["question"]
                st.session_state["options"] = processed_output["options"]
                st.session_state["explanation"] = processed_output["explanation"]
                st.session_state["correct_option"] = processed_output["correct_option"]
                st.session_state["answered"] = False
            else:
                st.code(processed_output.get("raw_response", "No response available"))

# Step 2: Display Question and Options
if "question" in st.session_state and not st.session_state.get("answered", False):
    st.subheader(f"Question: {st.session_state['question']}")

    # Display options as buttons
    for option in st.session_state["options"]:
        if st.button(option):
            selected_option = option.strip()
            correct_option = st.session_state["correct_option"]
            explanation = st.session_state["explanation"]

            # Check if the selected option is correct
            if selected_option.startswith(correct_option):
                st.success(f"Correct! Explanation: {explanation}")
            else:
                st.error(f"Incorrect! Explanation: {explanation}")

            # Mark question as answered
            st.session_state["answered"] = True

# Reset the flow after the explanation is shown
if st.session_state.get("answered", False):
    if st.button("Reset for a new question"):
        st.session_state.clear()
        st.experimental_rerun()
