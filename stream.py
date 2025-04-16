import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

# Model Configuration
<<<<<<< HEAD

model_name = "nairs-2b-fine-tuned"
=======
model_name = "./nairs-2b-fine-tuned"
>>>>>>> 5fa5f5443c4a988f4f3729d665650aaa69629b79
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
use_quantization_config = False  # Set this to False for CPU

<<<<<<< HEAD
# # Debugging: Check if the model directory exists
# if not os.path.exists(model_name):
#     st.error(f"Model directory not found: {model_name}")
# else:
#     st.info(f"Model directory exists: {model_name}")
#     st.info(f"Files in directory: {os.listdir(model_name)}")
 
# Check for Flash Attention 2 availability
=======
# Disable CUDA if not available
device = "cuda" if torch.cuda.is_available() else "cpu"

>>>>>>> 5fa5f5443c4a988f4f3729d665650aaa69629b79
if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"

st.info(f"Using attention implementation: {attn_implementation}")
st.info(f"Loading model: {model_name}")

# Load Model and Tokenizer
@st.cache_resource
def load_model_and_tokenizer():
<<<<<<< HEAD
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map = "auto",
            quantization_config=quantization_config if use_quantization_config else None,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None
=======
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Adjusting for CPU usage
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,  # Use float32 for CPU
        quantization_config=quantization_config if use_quantization_config else None,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation
    ).to(device)  # Use 'cpu' or 'cuda' based on availability
    return tokenizer, model
>>>>>>> 5fa5f5443c4a988f4f3729d665650aaa69629b79

tokenizer, model = load_model_and_tokenizer()

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
            "raw_response": output
        }
    else:
        # If format is incorrect, return raw response
        return {
            "raw_response": output
        }

# Streamlit Interface
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
            output = model.generate(**inputs)
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
