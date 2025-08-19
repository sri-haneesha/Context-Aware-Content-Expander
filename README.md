# Context-Aware-Content-Expander
This project explores automatic text expansion using fine-tuned language models. The system takes a short, concise input (e.g., a headline, abstract, or note) and expands it into a more detailed, context-aware version. Two specialized models were developed:
Factual Expander → generates informative, objective expansions (trained on Wikipedia + CNN data).
Creative Expander → produces imaginative, story-like outputs (trained on TinyStories + Reddit-style data).

**Features**

Fine-tuned FLAN-T5 / DistilGPT2 with LoRA adapters for lightweight training.
Interactive demo for real-time content expansion.
Support for both factual and creative text generation.
Training & evaluation implemented with Hugging Face Transformers.

**Dataset**

Factual Dataset: Wikipedia + CNN articles.
Creative Dataset: TinyStories + Reddit-like text.
Collected concise → detailed text pairs for supervised fine-tuning.

**Tech Stack**

Python
Hugging Face Transformers
LoRA-based fine-tuning
PyTorch
Jupyter Notebooks for experimentation

**Results**

The factual model produces concise, knowledge-grounded expansions suitable for academic/educational text.
The creative model generates richer, story-like narratives.
Both models outperform base pretrained models in coherence and relevance (qualitative evaluation).

**Sample I/O**

Input: "AI is transforming healthcare."

Factual Expansion: "AI is transforming healthcare by improving diagnostic accuracy, enabling predictive analytics, and supporting personalized treatment strategies."

Creative Expansion: "AI is transforming healthcare, with machines acting as silent assistants to doctors, uncovering hidden patterns in patient histories like detectives of the digital age."

**Future Work**

Quantitative evaluation using BLEU, ROUGE, and BERTScore.
Multi-lingual expansion support.
Integration with web app/Chrome extension for real-world usage.

**References**

Hugging Face Transformers Documentation
LoRA: Low-Rank Adaptation of LLMs
Datasets: Wikipedia Dumps
, CNN/DailyMail
, TinyStories
