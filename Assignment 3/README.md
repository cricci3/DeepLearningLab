# Headline News Generator with LSTM

A PyTorch-based language model utilizing Long Short-Term Memory (LSTM) networks to generate political news headlines. The model is trained on the Hugging Face [heegyu/news-category-dataset](https://huggingface.co/datasets/heegyu/news-category-dataset), specifically focusing on headlines from the POLITICS category.

## Features
- LSTM-based language model with an embedding layer.
- Two training implementations:
  - Standard training
  - Truncated Backpropagation Through Time (TBPTT)
- Two text generation strategies:
  - Random sampling with top-k
  - Greedy (argmax) sampling

## Model Overview
The model leverages an LSTM architecture with the following components:
- **Word Embeddings**: Converts word indices into dense vector representations.
- **LSTM Layers**: Captures temporal dependencies in text sequences.
- **Fully Connected Layer**: Maps LSTM outputs to vocabulary space for word prediction.
- **Dropout**: Applied for regularization to prevent overfitting.

## Usage
Train the model using the provided dataset and generate realistic political headlines based on learned patterns. The model can be fine-tuned with different hyperparameters such as embedding dimensions, hidden size, and number of LSTM layers.

For more details on implementation, refer to the project repository.