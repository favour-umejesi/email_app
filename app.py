import streamlit as st
import json
from generate import GenerateEmail

# --- CONFIG ---
st.set_page_config(page_title="AI Email Editor", layout="wide")

# --- HELPER FUNCTION ---
def load_emails(dataset_name):
    """Load emails from a JSONL file"""
    filepath = f"datasets/{dataset_name}.jsonl"
    emails = []
    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                emails.append(json.loads(line))
    return emails

# --- UI HEADER ---
st.title("AI Email Editing Tool")
st.write("Select a dataset and email ID, then use AI to refine it.")

# --- MODEL SELECTORS ---
st.sidebar.markdown("### Generation")
selected_model = st.sidebar.selectbox(
    "Generator Model",
    options=["gpt-4o-mini", "gpt-4.1"]
)

st.sidebar.markdown("### Evaluation")
judge_model = st.sidebar.selectbox(
    "Judge Model",
    options=["gpt-4.1", "gpt-4o-mini"]
)

# --- STEP 1: DATASET SELECTOR ---
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    options=["shorten", "lengthen", "tone"]
)

# --- STEP 2: LOAD DATASET ---
emails = load_emails(dataset_choice)

# --- STEP 3: EMAIL ID SELECTOR ---
# Get all IDs from the loaded emails
email_ids = [email["id"] for email in emails]

# Create dropdown to select an ID
selected_id = st.sidebar.selectbox(
    "Select Email ID",
    options=email_ids
)

# Find the selected email from the list
selected_email = next(email for email in emails if email["id"] == selected_id)

# --- STEP 4: TONE SELECTOR (only if "tone" dataset is selected) ---
selected_tone = None  # default value

if "tone" in dataset_choice:
    selected_tone = st.sidebar.selectbox(
        "Select Tone",
        options=["professional", "friendly", "sympathetic"]
    )

# --- STEP 5: DISPLAY SELECTED EMAIL ---
st.markdown(f"### Email ID: `{selected_id}`")
st.markdown(f"**From:** {selected_email['sender']}")
st.markdown(f"**Subject:** {selected_email['subject']}")

# Editable text area for the email content
email_content = st.text_area(
    "Email Content",
    value=selected_email["content"],
    height=250
)

# --- STEP 5.5: SELECTED TEXT OPTION ---
use_selected_text = st.checkbox("Transform selected portion only")

selected_text = None
if use_selected_text:
    selected_text = st.text_area(
        "Paste the text portion to transform",
        height=100,
        help="Copy and paste the exact portion of the email you want to transform"
    )
    if selected_text and selected_text not in email_content:
        st.warning("Selected text not found in the email. Make sure it matches exactly.")

# --- STEP 6: GENERATE BUTTON ---
# Set button label based on dataset
if "shorten" in dataset_choice:
    button_label = "Shorten Email"
elif "lengthen" in dataset_choice:
    button_label = "Lengthen Email"
else:
    button_label = f"Change Tone to {selected_tone.capitalize()}"

# Initialize session state for edited email
if "edited_email" not in st.session_state:
    st.session_state.edited_email = None

# Generate button
if st.button(button_label):
    with st.spinner("Generating..."):
        # Initialize the generator with selected model
        generator = GenerateEmail(model=selected_model)
        
        # Determine the action
        action = dataset_choice
        
        # Determine what text to transform
        if use_selected_text and selected_text and selected_text in email_content:
            # Transform only the selected portion
            transformed_portion = generator.generate(
                action=action,
                email_content=selected_text,
                tone=selected_tone
            )
            # Replace the selected portion in the original email
            st.session_state.edited_email = email_content.replace(selected_text, transformed_portion)
            st.session_state.transformed_portion = selected_text  # Store for reference
        else:
            # Transform the entire email
            st.session_state.edited_email = generator.generate(
                action=action,
                email_content=email_content,
                tone=selected_tone
            )
            st.session_state.transformed_portion = None

# Display edited email if it exists
if st.session_state.edited_email:
    st.markdown("### Edited Email:")
    st.write(st.session_state.edited_email)
    
    # --- JUDGE BUTTON (separate) ---
    st.markdown("---")
    if st.button(f"Judge Response ({judge_model})"):
        with st.spinner(f"Evaluating with {judge_model}..."):
            judge = GenerateEmail(model=judge_model)
            metrics = judge.judge(
                original_email=email_content,
                edited_email=st.session_state.edited_email
            )
        
        # Display all 6 metrics
        st.markdown("### Judge Metrics")
        
        # Row 1: Faithfulness, Completeness, Conciseness
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Faithfulness**")
            st.metric("Score", f"{metrics['faithfulness']['score']}/10")
            st.info(metrics["faithfulness"]["explanation"])
        
        with col2:
            st.markdown("**Completeness**")
            st.metric("Score", f"{metrics['completeness']['score']}/10")
            st.info(metrics["completeness"]["explanation"])
        
        with col3:
            st.markdown("**Conciseness**")
            st.metric("Score", f"{metrics['conciseness']['score']}/10")
            st.info(metrics["conciseness"]["explanation"])
        
        # Row 2: Grammar/Clarity, Tone Consistency, URL Preservation
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown("**Grammar & Clarity**")
            st.metric("Score", f"{metrics['grammar_clarity']['score']}/10")
            st.info(metrics["grammar_clarity"]["explanation"])
        
        with col5:
            st.markdown("**Tone & Style Consistency**")
            st.metric("Score", f"{metrics['tone_consistency']['score']}/10")
            st.info(metrics["tone_consistency"]["explanation"])
        
        with col6:
            st.markdown("**URL Preservation**")
            st.metric("Score", f"{metrics['url_preservation']['score']}/10")
            st.info(metrics["url_preservation"]["explanation"])