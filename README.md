# GPT Language Model in PyTorch

This repository contains an implementation of a transformer-based GPT language model using PyTorch. The model is designed to generate text by predicting the next token in a sequence, leveraging multi-head self-attention and feed-forward networks in a stacked transformer structure.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Components](#model-components)
- [Training](#training)
- [Acknowledgments](#acknowledgments)

## Project Overview
This project implements a simple GPT-like language model using PyTorch. The model is trained on a text corpus and can generate new text sequences by sampling from the learned probability distributions of the token sequences. Key features of this implementation include:
- Multi-head self-attention mechanism
- Layer normalization and dropout for stability and regularization
- Transformer blocks consisting of self-attention and feed-forward layers

## Requirements
- Python 3.x
- PyTorch (with MPS support for macOS if available)
- Install other dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Prepare the Data**: Save the text data in a file named `input.txt`. This file will be read to create a vocabulary and train the model.
2. **Run the Training Script**:
    ```bash
    python Nano-GPT.py
    ```

3. **Generate Text**: The model can generate new text samples after training. Run the following code in the script to generate new text:
    ```python
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    ```

## Model Components
- **MultiHeadAttention**: Multi-head self-attention mechanism for contextual representation.
- **FeedForward**: Fully connected feed-forward network.
- **GPTLanguageModel**: Main model class combining embedding, attention, and feed-forward layers in transformer blocks.

## Training
- **Hyperparameters**:
  - `batch_size`: Number of samples per batch
  - `block_size`: Sequence length to process at once
  - `learning_rate`: Learning rate for the optimizer
  - `n_embd`: Embedding dimension
  - `n_head`: Number of attention heads
  - `n_layer`: Number of transformer blocks
  - `dropout`: Dropout probability

- **Optimizer**: Uses AdamW for training, with gradients updated every batch.

- **Evaluation**: Calculates training and validation loss every 500 iterations to monitor model performance.

## Acknowledgments
This project is inspired by transformer-based language models and follows the architecture outlined in the original GPT paper.
