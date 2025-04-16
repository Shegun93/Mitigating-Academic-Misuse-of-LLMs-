import streamlit as st
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure PyTorch does NOT use GPU (in case it was inadvertently using GPU)
torch.cuda.empty_cache()  # Clear any cached GPU memory
torch.backends.cudnn.enabled = False  # Disable CuDNN (ensures CPU usage)

# Model Configuration
device = torch.device("cpu")
model_name = "nairs-model"

# Load Model and Tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model, ensuring it uses the CPU explicitly
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

        # Explicitly move the model to the CPU
        model.to(device)

        # Move all parameters in the model to CPU (checking every component)
        for param in model.parameters():
            param.data = param.data.to(device)

        # Return model and tokenizer
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

tokenizer, model = load_model_and_tokenizer()

# Disable the "Generate" button if the model is not loaded
if tokenizer is None or model is None:
    st.error("Model or tokenizer not loaded. Please check the logs.")
    st.stop()  # Stop further execution

# Function to process model output
def process_model_output(output):
    question_match = re.search(r"^(.*?)A:", output, re.DOTALL)
    question = question_match.group(1).strip() if question_match else None

    options_match = re.search(r"(A: .*?)(?=Correct Option:|Explanation:|$)", output, re.DOTALL)
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
        }
    else:
        # If format is incorrect, return raw response
        return {
            "raw_response": output
        }

# Streamlit Interface
st.title("Nairs Apps")

# Step 1: Prompt Input
st.subheader("Enter Your Prompt")
prompt = st.text_area("Type your question or text here", height=100)

if st.button("Generate"):
    if not prompt.strip():
        st.error("Please enter a valid prompt.")
    else:
        with st.spinner("Generating..."):
            try:
                # Move input tensors explicitly to CPU (for consistency)
                inputs = tokenizer(prompt.strip(), return_tensors="pt").to(device)

                # Ensure the input tensors are correctly on CPU
                for key, value in inputs.items():
                    value = value.to(device)  # Ensure every tensor is on CPU
                    assert value.device == device, f"Input tensor {key} is on {value.device}, expected {device}"

                # Generate text on CPU
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=500)

                # Ensure that output tensor is on the CPU
                assert output.device == device, f"Output tensor is on {output.device}, expected {device}"

                # Decode output (ensure the output tensor is also on CPU)
                raw_output = tokenizer.decode(output[0].cpu(), skip_special_tokens=True)

                # Process the model output
                processed_output = process_model_output(raw_output)

                if "question" in processed_output and "options" in processed_output:
                    st.session_state["question"] = processed_output["question"]
                    st.session_state["options"] = processed_output["options"]
                    st.session_state["explanation"] = processed_output["explanation"]
                    st.session_state["correct_option"] = processed_output["correct_option"]
                else:
                    st.code(processed_output.get("raw_response", "No response available"))
            except Exception as e:
                st.error(f"Error generating response: {e}")

# Step 2: Display Question and Options
if "question" in st.session_state and not st.session_state.get("answered", False):
    st.subheader(f"Question: {st.session_state['question']}")
    
    # Display options
    st.write("**Options:**")
    for option in st.session_state["options"]:
        st.write(option)

    # Input field for the user's answer
    user_answer = st.text_input("Enter your answer (A, B, C, or D):").strip().upper()

    # Submit button to check the answer
    if st.button("Submit"):
        if user_answer in ["A", "B", "C", "D"]:
            correct_option = st.session_state["correct_option"]
            explanation = st.session_state["explanation"]

            # Check if the selected option is correct
            if user_answer == correct_option:
                st.success(f"Correct! Explanation: {explanation}")
            else:
                st.error(f"Incorrect! Explanation: {explanation}")
            
            # Mark question as answered
            st.session_state["answered"] = True
        else:
            st.warning("Please enter a valid option (A, B, C, or D).")

# Reset the flow after the explanation is shown
if st.session_state.get("answered", False):
    if st.button("Reset for a new question"):
        st.session_state.clear()
