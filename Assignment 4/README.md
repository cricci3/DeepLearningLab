# Heuristic for TSP Using Transformers

This project explores the application of Transformer models to solve the Traveling Salesman Problem (TSP), a well-known NP-hard optimization challenge. Traditionally tackled using combinatorial optimization and heuristic approaches, this work investigates the feasibility of leveraging deep learning models, specifically Transformers, to develop an efficient heuristic for TSP.

## Overview
- **Problem Definition**: The TSP requires finding the shortest route that visits all given cities exactly once and returns to the starting point.
- **Transformer-Based Approach**: Originally designed for sequence-to-sequence tasks in NLP, Transformers have shown promise in structured data applications. This project encodes TSP instances into a format suitable for a Transformer-based model.
- **Training Strategy**: The model is trained using gradient accumulation to optimize performance and stability.
- **Evaluation**: The heuristic model’s performance is compared against traditional approaches, analyzing its efficiency and accuracy.

This work aims to bridge the gap between deep learning and combinatorial optimization by experimenting with modern architectures for solving classical problems.
