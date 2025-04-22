## Mitigating Academic Misuse of LLms Through Formative AI systems for Personalized Learning Support
This is a code repository for the title above. This paper proposes an ethical and pedagogically guided use of LLMs to enhance learning without enabling misuse. Drawing techniques such as instruction fine-tuning and structured data modeling, we develop a personalized learning system that assists students in understanding, not solving, academic problems. 
The system works by accepting a narrated problem scenario from the student, generating diagnostic assessment questions to gauge conceptual understanding, and delivering tailored feedback based on the student’s responses. Unlike general-purpose AI assistants, the model is intentionally constrained to avoid solution generation, instead focusing on formative assessment and guided explanation. The paper is currently undergoing review. 
## Overview
The model was fine-tuned using:

- LoRA (Low-Rank Adaptation) for efficient training

- 4-bit quantization to reduce memory requirements

- Physics question dataset in JSON format

- Structured Supervised fine-tuning
## Project Folder Structure
```
.
├── App
│   ├── app.py
│   ├── streamlit.py
│   └── stream.py
├── Data
│   ├── Physics_questions.json
│   └── test_data.json
├── Dockerfile
├── Epochs_2
│   └── events.out.tfevents.1744709703.shegun93-DQ67SW.25349.0
├── Evaluation.ipynb
├── nairs-fine-tunned.ipynb
├── nairs-test.ipynb
├── __pycache__
│   ├── app2.cpython-39.pyc
│   ├── flask.cpython-39.pyc
│   └── streamlit.cpython-39.pyc
├── README.md
├── requirements.txt
└── templates
    └── index.html
```
## Usage
Clone the repo
```
git clone https://github.com/Shegun93/nairs_paper.git
cd nairs_paper
```
install requirements
```
pip install -r requirements.txt
```

## Key capabilities:

- Generates multiple-choice physics questions

- Provides detailed explanations for correct answers

- Assesses student understanding of physics concepts

- Handles various physics topics (mechanics, thermodynamics, electromagnetism, etc.)
