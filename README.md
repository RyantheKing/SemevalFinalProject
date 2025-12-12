# SemEval 2026 Task 10 Experiments
## Ryan King

## The Goal
The goal of my experimentation for this project was to test the usefulness of embedding models other than BERT in the conspiracy analysis task. I started with nomic-embed-text, but the current implementation uses the qwen3-embedding model to create embeddings which are then classified using a variety of basic classifiers.

## Prerequisites
Setup for the repository is detailed in the README.md file located in the semeval26_task10_starter_pack folder. \
You will rehydrate the training and dev test data and run the training to build the embeddings and train the model. \
For this code to work, ollama must be running on your machine, and the qwen3-embedding model must be available. (If `ollama list` shows `qwen3-embedding:latest` but the code still errors, try running `ollama run qwen3-embedding test` in your terminal (to get ollama to initialize the model), and then run `train_binary.py` again).

## Experiments/Changes to try
