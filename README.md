# Knowledge Graph Validator Tool 🌐🔍

Welcome to the Knowledge Graph Validator Tool, a cutting-edge utility designed to enhance the accuracy and reliability of Knowledge Graphs (KGs) and Knowledge Bases (KBs) using an advanced Language Model (LM) as the annotator agent. 

This tool leverages the power of Large Language Models (LLMs) to validate knowledge graphs with or without additional context, ensuring your data is not only accurate but also rich and interconnected.

## Features ✨

Our Knowledge Graph Validator offers a variety of validation modes to suit your needs:

- **No-Context Validation** 🧠: Validate your KG using the LLM's extensive background knowledge, perfect for quick checks or when external context is unavailable.
  
- **Textual Context Validation** 📚: Incorporate a collection of documents to provide context during validation, for a deeper, more nuanced analysis.
  
- **Reference KG Validation** 🌐: Use LangChain to retrieve the most likely reference properties from another KG, allowing for a comparison-based validation approach.

## Getting Started 🚀

To get started with the Knowledge Graph Validator Tool, follow these simple steps:

### Prerequisites

You'll need an OpenAI API key.

### Installation

```
conda create kg_validator python=3.10 -y
conda activate kg_validator
pip install git+https://github.com/jackboyla/knowledge-base-builder
```


# How It Works 🧐
The Knowledge Graph Validator leverages advanced NLP techniques and the vast knowledge embedded in LLMs to validate and enrich your KGs:

No-Context Mode: Directly queries the LLM with entities and relationships from your KG, validating against the LLM's internal knowledge base.

Textual Context Mode: Enhances validation by including relevant documents, providing a richer context for more accurate and nuanced validation.

Reference KG Mode: Compares your KG against a selected reference KG, identifying discrepancies and validating relationship mappings.


# Contributing 🤝
We welcome contributions! Whether it's submitting bugs, requesting features, or contributing code, feel free to open an issue or pull request on our GitHub repository.


# License 📄
This tool is made available under the MIT License. Feel free to use, modify, and distribute it as per the license terms.

# Support 💖
If you encounter any issues or have questions, please file an issue on GitHub. Your feedback helps make this tool better for everyone!

Happy Validating! 🎉

