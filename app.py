import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize the VADER sentiment engine
analyzer = SentimentIntensityAnalyzer()

# Page configuration
st.set_page_config(page_title="AI Sentiment Pro", page_icon="🧠")

st.title("Advanced Sentiment Analyzer")
st.write("This model accurately interprets nuances and negations (e.g., 'don't feel good').")

# Single text analysis section
message = st.text_area("Text to analyze (English):")

if st.button("Run Analysis"):
    if message:
        # VADER returns a dictionary of scores (pos, neg, neu, compound)
        vs = analyzer.polarity_scores(message)
        score = vs["compound"]  # Global score between -1 and 1

        # Display logic based on score
        if score >= 0.05:
            st.success(f"Positive (score: {score:.2f})")
        elif score <= -0.05:
            st.error(f"Negative (score: {score:.2f})")
        else:
            st.warning(f"Neutral (score: {score:.2f})")

        # Technical details for a professional look
        with st.expander("View AI Technical Details"):
            st.write(f"Negativity: {vs['neg']}")
            st.write(f"Neutrality: {vs['neu']}")
            st.write(f"Positivity: {vs['pos']}")
    else:
        st.info("Please enter a sentence!")

# Group analysis section
st.divider()
st.subheader("📊 Custom Group Analysis")

# Text area for multiple phrases separated by commas or new lines
input_groupe = st.text_area(
    "Enter multiple phrases to compare (separate by comma or new line):",
    placeholder="Ex: I love it, I hate it, Not bad..."
)

if st.button("Run Group Analysis"):
    if input_groupe:
        # Clean the input to create a proper list
        # Handle both new lines and commas
        lines = input_groupe.replace('\n', ',').split(',')
        user_phrases = [p.strip() for p in lines if p.strip()]

        results = []
        for p in user_phrases:
            score = analyzer.polarity_scores(p)['compound']
            results.append({"Text": p, "Score": score})

        # Convert results to a DataFrame
        df = pd.DataFrame(results)

        # Visual representation
        st.bar_chart(df.set_index("Text"))
        st.table(df)
    else:
        st.warning("Please add at least one phrase for the simulation.")