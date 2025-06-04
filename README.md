## 🧠 Mitigating Academic Misuse of LLMs Through Formative AI Systems for Personalized Learning Support
This repository supports the research project focused on building an ethical, personalized learning assistant using Large Language Models (LLMs). The goal is to help students understand physics concepts, not simply generate answers, thereby preserving academic integrity and promoting deep learning

## 🔧 What This System Does
- Accepts narrated student prompts describing conceptual struggles
- Generates diagnostic multiple-choice questions to assess understanding
- Evaluates student responses to identify conceptual gaps
- Provides guided explanations, not direct answers
- Supports a wide range of physics topics: mechanics, thermodynamics, electromagnetism, optics, etc.
## 📐 System Design
The model is trained to take structured input and produce diagnostic reasoning. Here's how it works:

![System Design](./system_design.png)

## 📁 Dataset Structure
Each entry includes:

A narrated input text

A multiple-choice question

Four answer options

Correct option

An explanatory rationale

Example:

![Dataset Structure](./dataset_sample.png)

## 🧠 Model Inference
Upon inference

![Result_Inference](./Result_Inference.png)

## ⚙️ Model & Training
The LLM was fine-tuned using:

LoRA (Low-Rank Adaptation) for efficient parameter tuning

4-bit quantization to reduce memory load

Supervised instruction fine-tuning on structured university-level physics questions

Physics dataset curated from Federal University of Technology Akure (FUTA) materials

### 📊 Evaluation Results

| Prompting Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|------------------|------------|------------|------------|
| **Zero-Shot**    | 0.0451     | 0.0196     | 0.0411     |
| **Few-Shot**     | 0.0828     | 0.0322     | 0.0604     |


## Project Folder Structure
```
.
├── App
├── bfg.jar
├── compile.ipynb
├── Data
├── dataset_sample.png
├── Dockerfile
├── Epochs_2
├── Epochs_5
├── Evaluation.ipynb
├── Images
├── inf.png
├── nairs-2d
├── nairs-2e
├── nairs-fine-tunned.ipynb
├── nairs_merged
├── nairs-sample-4
├── nairs-test.ipynb
├── __pycache__
├── README.md
├── requirements.txt
├── Result_Inference.png
└── system_design.png
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
