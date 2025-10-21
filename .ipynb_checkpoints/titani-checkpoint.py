# ========================================
# TP2 TITANIC – MACHINE LEARNING INTERACTIF AVEC STREAMLIT
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# CONFIGURATION DE LA PAGE
# -------------------------------
st.set_page_config(page_title="TP2 Titanic ML", layout="wide")
st.title("🚢 Machine Learning sur le Titanic Dataset")
st.markdown("### INF 365 – Ingénierie de Donnée")

st.sidebar.header("⚙️ Paramètres de l'application")

# -------------------------------
# 1️⃣ CHARGEMENT DU FICHIER
# -------------------------------
uploaded_file = st.sidebar.file_uploader("📂 Importer le fichier Titanic (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Fichier chargé avec succès !")

    # Aperçu du dataset
    st.subheader("📋 Aperçu des données")
    st.dataframe(df.head())

    # Informations générales
    st.subheader("ℹ️ Informations générales")
    st.write("**Nombre de lignes et colonnes :**", df.shape)
    st.write("**Valeurs manquantes :**")
    st.dataframe(df.isnull().sum())

    # -------------------------------
    # 2️⃣ PRÉTRAITEMENT
    # -------------------------------
    st.subheader("🧹 Nettoyage et Prétraitement des Données")

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    # Imputation des valeurs manquantes
    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')

    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    # Encodage
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    # Normalisation
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    st.success("✅ Données nettoyées, encodées et normalisées avec succès !")

    # -------------------------------
    # 3️⃣ SÉPARATION TRAIN / TEST
    # -------------------------------
    if "Survived" not in df.columns:
        st.error("❌ La colonne 'Survived' (variable cible) est manquante dans le dataset.")
    else:
        # X = df.drop("Survived", axis=1)
        # y = df["Survived"]
        X = df.drop("Survived", axis=1)
        y = df["Survived"]

# 🔧 Correction : forcer les étiquettes à être de type entier (classification)
        y = y.round().astype(int)


        test_size = st.sidebar.slider("📊 Taille de l’ensemble de test (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )

        st.write(f"**Taille de l’ensemble d’entraînement :** {X_train.shape}")
        st.write(f"**Taille de l’ensemble de test :** {X_test.shape}")

        # -------------------------------
        # 4️⃣ CHOIX DU MODÈLE
        # -------------------------------
        st.sidebar.subheader("🧠 Choix du modèle")
        model_choice = st.sidebar.selectbox(
            "Sélectionnez un algorithme :",
            ["KNN", "SVM", "Random Forest", "Régression Logistique"]
        )

        if st.sidebar.button("🚀 Entraîner le modèle"):
            if model_choice == "KNN":
                model = KNeighborsClassifier()
            elif model_choice == "SVM":
                model = SVC()
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
            else:
                model = LogisticRegression(max_iter=1000)

            # Entraînement
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            st.subheader("📈 Résultats du Modèle")
            st.write(f"**Exactitude (Accuracy)** : {acc:.4f}")
            st.text("📊 Rapport de classification :")
            st.text(classification_report(y_test, y_pred))

            # Matrice de confusion graphique
            st.subheader("🔍 Matrice de Confusion")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
            st.pyplot(fig)

        # -------------------------------
        # 5️⃣ OPTIMISATION DES HYPERPARAMÈTRES
        # -------------------------------
        st.sidebar.subheader("🎯 Optimisation GridSearchCV")

        if st.sidebar.button("🔎 Lancer l’optimisation"):
            if model_choice == "KNN":
                param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
                grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
            elif model_choice == "Random Forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
                grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
            else:
                st.warning("⚠️ L’optimisation est uniquement disponible pour KNN et Random Forest.")
                grid = None

            if grid is not None:
                with st.spinner("⏳ Recherche des meilleurs paramètres..."):
                    grid.fit(X_train, y_train)
                st.success("✅ Optimisation terminée !")
                st.write("**Meilleurs paramètres :**", grid.best_params_)
                st.write("**Meilleure précision (cross-validation)** :", grid.best_score_)

else:
    st.info("📥 Veuillez importer un fichier Titanic CSV pour commencer.")
