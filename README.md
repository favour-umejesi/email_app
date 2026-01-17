# Email App

This project is an AI-powered email helper application built as part of the AI.Accelerate Bootcamp. It uses Streamlit for the UI and supports various email-related tasks.

## Project Structure
- `datasets/`: Contains `.jsonl` datasets for training/testing email actions.
- `app.py`: The main Streamlit application.
- `generate.py`: Script for generating data or processing.
- `prompts.yaml`: Configuration for AI prompts.

---

## Getting Started

### **1. Clone the repository**
```bash
git clone https://github.com/favour-umejesi/email_app.git
cd email_app
```

### **2. Set up a Virtual Environment**

> Recommended: Python 3.9+

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```
When activated, your terminal prompt should display `(venv)`.

### **3. Install Required Dependencies**

With the virtual environment activated, install the project dependencies:

```bash
pip install -r requirements.txt
```

### **4. Start the app**
To run the app locally:
```bash
streamlit run app.py
```
This will open the app in your default browser at `http://localhost:8501`.

---
Welcome to Email App!
