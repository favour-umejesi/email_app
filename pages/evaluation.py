import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from generate import GenerateEmail

# --- CONFIG ---
st.set_page_config(page_title="Evaluation Dashboard", layout="wide")

# --- HELPER FUNCTIONS ---
def load_emails(dataset_name, filter_type=None):
    """Load emails from a JSONL file, optionally filtering by type"""
    filepath = f"datasets/{dataset_name}.jsonl"
    emails = []
    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                email = json.loads(line)
                # Filter by type if specified
                if filter_type is None or email.get('type', 'original') == filter_type:
                    emails.append(email)
    return emails

def run_evaluation(datasets, generator_models, judge_model, sample_size, selected_tone, filter_type, progress_bar):
    """Run batch evaluation across datasets and models"""
    results = []
    
    # Calculate total iterations for progress
    total_iterations = 0
    for dataset in datasets:
        emails = load_emails(dataset, filter_type)
        num_emails = min(sample_size, len(emails))
        total_iterations += num_emails * len(generator_models)
    
    if total_iterations == 0:
        return pd.DataFrame()
    
    current_iteration = 0
    
    for dataset in datasets:
        emails = load_emails(dataset, filter_type)
        emails_to_process = emails[:sample_size]  # Limit sample size
        
        # Determine action from dataset name
        action = dataset
        tone = selected_tone if action == "tone" else None
        
        for email in emails_to_process:
            email_type = email.get('type', 'original')
            
            for gen_model in generator_models:
                # Generate edited email with timing
                generator = GenerateEmail(model=gen_model)
                try:
                    start_time = time.time()
                    edited_email = generator.generate(
                        action=action,
                        email_content=email["content"],
                        tone=tone
                    )
                    generation_time = round(time.time() - start_time, 2)
                except Exception as e:
                    edited_email = f"Error: {str(e)}"
                    continue
                
                # Judge the edited email with fixed judge model (gpt-4.1)
                judge = GenerateEmail(model=judge_model)
                try:
                    metrics = judge.judge(
                        original_email=email["content"],
                        edited_email=edited_email
                    )
                    
                    # Extract scores
                    result = {
                        "dataset": dataset,
                        "email_type": email_type,
                        "action": action,
                        "tone": tone if tone else "N/A",
                        "email_id": email["id"],
                        "generator_model": gen_model,
                        "judge_model": judge_model,
                        "response_time_sec": generation_time,
                        "original_content": email["content"],
                        "generated_content": edited_email,
                        "faithfulness": metrics.get("faithfulness", {}).get("score", "N/A"),
                        "completeness": metrics.get("completeness", {}).get("score", "N/A"),
                        "conciseness": metrics.get("conciseness", {}).get("score", "N/A"),
                        "grammar_clarity": metrics.get("grammar_clarity", {}).get("score", "N/A"),
                        "tone_consistency": metrics.get("tone_consistency", {}).get("score", "N/A"),
                        "url_preservation": metrics.get("url_preservation", {}).get("score", "N/A"),
                    }
                    results.append(result)
                except Exception as e:
                    st.warning(f"Error judging email {email['id']}: {str(e)}")
                
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
    
    return pd.DataFrame(results)

# --- UI ---
st.title("Evaluation Dashboard")
st.write("Compare **gpt-4o-mini** vs **gpt-4.1** generator models")

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### Dataset Settings")

# Dataset selection (no more _synthetic options)
all_datasets = ["shorten", "lengthen", "tone"]
selected_datasets = st.sidebar.multiselect(
    "Select Datasets",
    options=all_datasets,
    default=["shorten"]
)

# Filter by email type (original vs synthetic)
filter_type = st.sidebar.radio(
    "Email Type Filter",
    options=["all", "original", "synthetic"],
    help="Filter emails by type - 'all' includes both original and synthetic"
)
filter_type = None if filter_type == "all" else filter_type

# Tone selector - only show if tone dataset is selected
selected_tone = "professional"  # default
has_tone_dataset = "tone" in selected_datasets
if has_tone_dataset:
    selected_tone = st.sidebar.selectbox(
        "Select Tone",
        options=["professional", "friendly", "sympathetic"],
        help="Tone for tone transformation evaluation"
    )

# Sample size - NUMBER INPUT instead of slider
st.sidebar.markdown("### Sample Size")
sample_size = st.sidebar.number_input(
    "Number of Emails to Evaluate",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
    help="Enter the number of emails to evaluate from each dataset"
)

# Generator model selection (main focus)
st.sidebar.markdown("### Generator Models")
st.sidebar.write("Compare these models:")
generator_models = st.sidebar.multiselect(
    "Select Generator Models",
    options=["gpt-4o-mini", "gpt-4.1"],
    default=["gpt-4o-mini", "gpt-4.1"]
)

# Judge model - default to gpt-4.1
st.sidebar.markdown("### Judge Model")
judge_model = st.sidebar.selectbox(
    "Judge Model",
    options=["gpt-4.1", "gpt-4o-mini"],
    index=0,
    help="gpt-4.1 is recommended as the primary judge"
)

# --- MAIN CONTENT ---
st.markdown("---")

# Show dataset info
if selected_datasets:
    st.markdown("### Dataset Overview")
    cols = st.columns(len(selected_datasets))
    for i, dataset in enumerate(selected_datasets):
        with cols[i]:
            all_emails = load_emails(dataset)
            original_count = len([e for e in all_emails if e.get('type', 'original') == 'original'])
            synthetic_count = len([e for e in all_emails if e.get('type') == 'synthetic'])
            
            st.metric(f"{dataset.title()}", f"{len(all_emails)} emails")
            st.caption(f"{original_count} original | {synthetic_count} synthetic")

st.markdown("---")

# Run evaluation button
if st.button("Run Batch Evaluation", type="primary"):
    if not selected_datasets or not generator_models:
        st.error("Please select at least one dataset and generator model.")
    else:
        filter_info = f" ({filter_type} only)" if filter_type else " (all types)"
        tone_info = f", tone: {selected_tone}" if has_tone_dataset else ""
        st.info(f"Evaluating {sample_size} emails from {len(selected_datasets)} datasets{filter_info}{tone_info}")
        st.info(f"Comparing generators: {', '.join(generator_models)} | Judge: {judge_model}")
        
        progress_bar = st.progress(0)
        
        with st.spinner("Evaluating... This may take a few minutes."):
            results_df = run_evaluation(
                selected_datasets,
                generator_models,
                judge_model,
                sample_size,
                selected_tone,
                filter_type,
                progress_bar
            )
        
        if not results_df.empty:
            st.session_state["evaluation_results"] = results_df
            st.success(f"Evaluation complete! {len(results_df)} results collected.")
        else:
            st.warning("No results collected. Check if the selected filter has matching emails.")

# --- DISPLAY RESULTS ---
if "evaluation_results" in st.session_state and not st.session_state["evaluation_results"].empty:
    results_df = st.session_state["evaluation_results"]
    
    st.markdown("## Results Overview")
    
    # Convert scores to numeric, handling "N/A"
    metric_cols = ["faithfulness", "completeness", "conciseness", "grammar_clarity", "tone_consistency", "url_preservation"]
    for col in metric_cols:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    
    # Show raw data with per-model averages and overall average
    with st.expander("View Raw Results"):
        # Create a copy for display
        display_df = results_df.copy()
        
        avg_rows = []
        
        # Calculate average for each generator model
        for model in display_df["generator_model"].unique():
            model_data = display_df[display_df["generator_model"] == model]
            model_avg = {col: "" for col in display_df.columns}
            model_avg["dataset"] = f"AVG ({model})"
            model_avg["email_id"] = "-"
            model_avg["generator_model"] = model
            model_avg["judge_model"] = "-"
            model_avg["email_type"] = "-"
            model_avg["action"] = "-"
            model_avg["tone"] = "-"
            model_avg["response_time_sec"] = round(model_data["response_time_sec"].mean(), 2)
            
            for col in metric_cols:
                model_avg[col] = round(model_data[col].mean(), 2)
            
            avg_rows.append(model_avg)
        
        # Calculate overall average
        overall_avg = {col: "" for col in display_df.columns}
        overall_avg["dataset"] = "OVERALL AVERAGE"
        overall_avg["email_id"] = "-"
        overall_avg["generator_model"] = "-"
        overall_avg["judge_model"] = "-"
        overall_avg["email_type"] = "-"
        overall_avg["action"] = "-"
        overall_avg["tone"] = "-"
        overall_avg["response_time_sec"] = round(display_df["response_time_sec"].mean(), 2)
        
        for col in metric_cols:
            overall_avg[col] = round(display_df[col].mean(), 2)
        
        avg_rows.append(overall_avg)
        
        # Append all average rows
        avg_df = pd.DataFrame(avg_rows)
        display_df = pd.concat([display_df, avg_df], ignore_index=True)
        
        # Hide content columns from raw data display
        cols_to_show = [c for c in display_df.columns if c not in ['original_content', 'generated_content']]
        st.dataframe(display_df[cols_to_show], use_container_width=True)
    
    # --- MODEL RESPONSES SECTION ---
    with st.expander("View Model Responses (Generated Emails)"):
        st.write("Compare original emails with model-generated transformations")
        
        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            filter_model = st.selectbox(
                "Filter by Generator Model",
                options=["All"] + list(results_df["generator_model"].unique()),
                key="response_filter_model"
            )
        with col_filter2:
            num_to_show = st.number_input(
                "Number of responses to display",
                min_value=1,
                max_value=len(results_df),
                value=min(5, len(results_df)),
                key="num_responses"
            )
        
        # Filter dataframe
        filtered_df = results_df.copy()
        if filter_model != "All":
            filtered_df = filtered_df[filtered_df["generator_model"] == filter_model]
        
        # Display responses
        for idx, row in filtered_df.head(num_to_show).iterrows():
            with st.container():
                st.markdown(f"**Email ID {row['email_id']}** | Model: `{row['generator_model']}` | Action: `{row['action']}` | Avg Score: `{row[metric_cols].mean():.1f}/10`")
                
                col_orig, col_gen = st.columns(2)
                with col_orig:
                    st.markdown("**Original:**")
                    st.text_area(
                        "Original", 
                        value=row['original_content'][:500] + "..." if len(str(row['original_content'])) > 500 else row['original_content'],
                        height=150,
                        key=f"orig_{idx}_{row['generator_model']}",
                        disabled=True
                    )
                with col_gen:
                    st.markdown("**Generated:**")
                    st.text_area(
                        "Generated",
                        value=row['generated_content'][:500] + "..." if len(str(row['generated_content'])) > 500 else row['generated_content'],
                        height=150,
                        key=f"gen_{idx}_{row['generator_model']}",
                        disabled=True
                    )
                st.markdown("---")
    
    st.markdown("---")
    
    # --- SUMMARY TABLE (Report Format) ---
    st.markdown("### Model Comparison Summary")
    
    # Build summary table
    summary_data = []
    for model in results_df["generator_model"].unique():
        model_data = results_df[results_df["generator_model"] == model]
        row = {"Model": model}
        for col in metric_cols:
            row[col.replace("_", " ").title()] = round(model_data[col].mean(), 2)
        row["Avg Time (sec)"] = round(model_data["response_time_sec"].mean(), 2)
        row["Overall Avg"] = round(model_data[metric_cols].mean().mean(), 2)
        summary_data.append(row)
    
    # Add overall row
    overall_row = {"Model": "OVERALL"}
    for col in metric_cols:
        overall_row[col.replace("_", " ").title()] = round(results_df[col].mean(), 2)
    overall_row["Avg Time (sec)"] = round(results_df["response_time_sec"].mean(), 2)
    overall_row["Overall Avg"] = round(results_df[metric_cols].mean().mean(), 2)
    summary_data.append(overall_row)
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # --- MAIN CHART: Generator Model Comparison ---
    st.markdown("### Generator Model Comparison")
    st.write("**Primary Goal**: Which model generates better email transformations?")
    
    gen_comparison = results_df.groupby("generator_model")[metric_cols].mean().reset_index()
    gen_melted = gen_comparison.melt(id_vars=["generator_model"], var_name="Metric", value_name="Score")
    
    fig1 = px.bar(
        gen_melted,
        x="Metric",
        y="Score",
        color="generator_model",
        barmode="group",
        title="Generator Model Performance Comparison",
        color_discrete_sequence=["#636EFA", "#EF553B"]
    )
    fig1.update_layout(yaxis_range=[0, 10], height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Overall scores comparison
    col1, col2 = st.columns(2)
    for i, (model, group) in enumerate(results_df.groupby("generator_model")):
        avg_score = group[metric_cols].mean().mean()
        with col1 if i == 0 else col2:
            st.metric(f"{model} Average", f"{avg_score:.2f}/10")
    
    st.markdown("---")
    
    # --- CHART: Radar comparison ---
    st.markdown("### Radar Chart: Model Profile Comparison")
    
    gen_radar = results_df.groupby("generator_model")[metric_cols].mean()
    
    fig_radar = go.Figure()
    colors = ["#636EFA", "#EF553B"]
    for i, (model, row) in enumerate(gen_radar.iterrows()):
        fig_radar.add_trace(go.Scatterpolar(
            r=row.values.tolist() + [row.values[0]],
            theta=metric_cols + [metric_cols[0]],
            fill='toself',
            name=model,
            line_color=colors[i % len(colors)],
            opacity=0.6
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="Generator Model Metrics Comparison",
        height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("---")
    
    # --- CHART: Original vs Synthetic (if both exist) ---
    if len(results_df['email_type'].unique()) > 1:
        st.markdown("### Original vs Synthetic Dataset Comparison")
        st.write("How do models perform on original vs synthetic (edge case) data?")
        
        type_comparison = results_df.groupby("email_type")[metric_cols].mean().reset_index()
        type_melted = type_comparison.melt(id_vars=["email_type"], var_name="Metric", value_name="Score")
        
        fig3 = px.bar(
            type_melted,
            x="Metric",
            y="Score",
            color="email_type",
            barmode="group",
            title="Original vs Synthetic Email Performance",
            color_discrete_sequence=["#19D3F3", "#FF6692"]
        )
        fig3.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("---")
    
    # --- CHART: Heatmap ---
    st.markdown("### Heatmap: Generator x Metric")
    
    heatmap_data = results_df.groupby("generator_model")[metric_cols].mean()
    
    fig_heatmap = px.imshow(
        heatmap_data.values,
        x=metric_cols,
        y=heatmap_data.index.tolist(),
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Score Heatmap by Generator Model",
        labels=dict(x="Metric", y="Generator Model", color="Score")
    )
    fig_heatmap.update_layout(height=300)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # --- RESPONSE TIME COMPARISON ---
    st.markdown("### Response Time Comparison")
    
    time_comparison = results_df.groupby("generator_model")["response_time_sec"].agg(['mean', 'min', 'max']).reset_index()
    time_comparison.columns = ["Model", "Avg (sec)", "Min (sec)", "Max (sec)"]
    
    col_t1, col_t2 = st.columns(2)
    for i, row in time_comparison.iterrows():
        with col_t1 if i == 0 else col_t2:
            st.metric(f"{row['Model']} Avg Time", f"{row['Avg (sec)']:.2f}s")
            st.caption(f"Range: {row['Min (sec)']:.2f}s - {row['Max (sec)']:.2f}s")
    
    st.markdown("---")
    
    # --- SUMMARY ---
    st.markdown("### Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg = results_df[metric_cols].mean().mean()
        st.metric("Overall Average", f"{avg:.2f}/10")
    
    with col2:
        best_gen = results_df.groupby("generator_model")[metric_cols].mean().mean(axis=1).idxmax()
        st.metric("Best Generator", best_gen)
    
    with col3:
        best_metric = results_df[metric_cols].mean().idxmax()
        st.metric("Strongest Metric", best_metric.replace("_", " ").title())
    
    with col4:
        worst_metric = results_df[metric_cols].mean().idxmin()
        st.metric("Needs Improvement", worst_metric.replace("_", " ").title())
    
    # Download
    st.markdown("---")
    st.markdown("### Export Results")
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="evaluation_results.csv",
        mime="text/csv"
    )

else:
    st.info("Configure settings in the sidebar and click 'Run Batch Evaluation' to start.")
    
    st.markdown("### What will be evaluated:")
    st.markdown("""
    - **Primary Goal**: Compare **gpt-4o-mini** vs **gpt-4.1** as generators
    - **Judge Model**: gpt-4.1 (recommended)
    - **Datasets**: shorten, lengthen, tone (includes both original and synthetic emails)
    - **Metrics**: Faithfulness, Completeness, Conciseness, Grammar, Tone, URL Preservation
    """)
