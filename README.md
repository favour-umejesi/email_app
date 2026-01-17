# AI Email Editor

An intelligent email enhancement tool that uses large language models to refine, transform, and evaluate email content. This application provides a streamlined interface for editing emails with specific goals like shortening, lengthening, or adjusting the tone, complete with an AI-powered judging system to ensure quality.

## Features

- **Email Transformation**: 
  - **Shorten**: Condense long emails while preserving key information.
  - **Lengthen**: Expand brief drafts into more detailed correspondence.
  - **Tone Adjustment**: Change the tone of your email to Professional, Friendly, or Sympathetic.
- **Selective Editing**: Transform only a specific portion of an email by pasting it into the tool.
- **AI Judging System**: Evaluate edited emails across six critical metrics:
  - Faithfulness
  - Completeness
  - Conciseness
  - Grammar & Clarity
  - Tone & Style Consistency
  - URL Preservation
- **Model Flexibility**: Choose between different LLMs for both generation and evaluation (e.g., GPT-4o-mini, GPT-4.1).

## Project Structure

- `app.py`: The main Streamlit application providing the user interface.
- `generate.py`: Core logic for AI generation and judging.
- `datasets/`: Pre-loaded email datasets for testing different transformation tasks.
- `prompts.yaml`: Configuration file for AI prompts and evaluation criteria.
- `requirements.txt`: List of Python dependencies.

## Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/favour-umejesi/email_app.git
cd email_app
```

### **2. Set up a Virtual Environment**
It is recommended to use Python 3.9+.

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

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Environment Variables**
Ensure you have your OpenAI API key configured (if required by `generate.py`):
```bash
# Create a .env file or export the variable
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

1. **Launch the application**:
   ```bash
   streamlit run app.py
   ```
2. **Select a Dataset**: Choose between "shorten", "lengthen", or "tone" from the sidebar.
3. **Select an Email**: Pick an email ID to load existing content.
4. **Edit and Transform**:
   - Modify the content in the text area if desired.
   - Use the "Transform selected portion only" checkbox if you only want to edit a specific part.
   - Click the transformation button (e.g., "Shorten Email").
5. **Evaluate**: Click the "Judge Response" button to see the AI evaluation of the changes.

---
Developed by [Favour Umejesi](https://github.com/favour-umejesi)
