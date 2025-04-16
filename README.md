## Llama-2 Physics Question Generator
This repository contains a fine-tuned Llama-2 model (7B parameters) that generates physics assessment questions, options, correct answers, and explanations based on given problem descriptions.
## Overview

# The model was fine-tuned using:

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

## Key capabilities:

- Generates multiple-choice physics questions

- Provides detailed explanations for correct answers

- Assesses student understanding of physics concepts

- Handles various physics topics (mechanics, thermodynamics, electromagnetism, etc.)
