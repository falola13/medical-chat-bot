# End-to-End Medical Chatbot using Llama 2

A medical Q&A chatbot that runs Llama 2 locally and answers questions grounded in your own medical documents.

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate a conda environment

```bash
conda create -n mchatbot python=3.10 -y
conda activate mchatbot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the model

**Option A: Run the download script (recommended)**

```bash
python model/download_model.py
```

This downloads the model and saves it to the `model/` folder automatically.

**Option B: Download it manually**

Download `llama-2-7b-chat.ggmlv3.q4_0.bin` from Hugging Face:

https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

Then place it in a `model/` folder at the root of the project:

