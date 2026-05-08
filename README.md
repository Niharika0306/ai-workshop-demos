# ai-workshop-demos

## Prerequisites

Before running the app, make sure you have Ollama installed and set up:

```bash
ollama pull granite-embedding:278m
ollama serve
```

## Setup and Run

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. When done, deactivate the virtual environment:

```bash
deactivate