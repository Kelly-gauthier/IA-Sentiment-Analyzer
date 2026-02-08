import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize the VADER sentiment engine
analyzer = SentimentIntensityAnalyzer()

# Page configuration
st.set_page_config(page_title="AI Sentiment Pro", page_icon="🧠")

st.title("Advanced Sentiment Analyzer")
st.write("This model accurately interprets nuances and negations (e.g., 'don't feel good').")

message = st.text_area("Text a analyser (Anglais):")

if st.button("Lancer l'analyse"):
    if message:
        #VADER donne un dictionnaire de scores (pos, neg, neu, compound)
        vs = analyzer.polarity_scores(message)
        score = vs["compound"] # le score global entre -1 et 1

        #logique d'affichage
        if score >= 0.05:
            st.success(f"Positif (score: {score:.2f})")
        elif score <= -0.05:
            st.error(f"negatif (score: {score:.2f})")
        else:
            st.warning(f"Neutre (score: {score:.2f})")

            #Details techniques pour faire pro
            with st.expander("Voir les details de L'IA"):
                st.write(f"Negativite: {vs['neg']}")
                st.write(f"Neutralite: {vs['neu']}")
                st.write(f"Positivite: {vs['pos']}")
    else:
        st.info("ecris un phrase !")

import pandas as pd # Pour gerer les donnees sous forme de tableau

# ... (garde ton code precedent au dessus)
st.divider()
st.subheader("📊 Analyse de groupe personnalisée")

# Zone pour saisir plusieurs phrases séparées par une virgule ou un retour à la ligne
input_groupe = st.text_area(
    "Entrez plusieurs phrases à comparer (séparez-les par une virgule ou une ligne) :",
    placeholder="Ex: I love it, I hate it, Not bad..."
)

if st.button("Lancer l'analyse groupée"):
    if input_groupe:
        # On sépare le texte pour créer une liste propre
        # On gère les virgules et les sauts de ligne
        lignes = input_groupe.replace('\n', ',').split(',')
        phrases_utilisateurs = [p.strip() for p in lignes if p.strip()]

        resultats = []
        for p in phrases_utilisateurs:
            score = analyzer.polarity_scores(p)['compound']
            resultats.append({"Texte": p, "Score": score})

        df = pd.DataFrame(resultats)

        # Affichage visuel
        st.bar_chart(df.set_index("Texte"))
        st.table(df)
    else:
        st.warning("Ajoutez au moins une phrase pour la simulation.")