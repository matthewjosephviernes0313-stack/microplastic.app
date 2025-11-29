import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone

st.set_page_config(page_title="Microplastic Risk Dashboard", page_icon="🧪", layout="wide")
st.title("🧪 Microplastic Risk Analysis — Enhanced Interactive Dashboard")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
tabs = [
    "1. Upload & Preview",
    "2. Data Preprocessing",
    "3. Modeling & Performance",
    "4. Visualizations"
]
selected_tab = st.sidebar.radio("Go to step:", tabs)

# Session state for data
if "df" not in st.session_state:
    st.session_state.df = None
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False

num_cols = ["MP_Count_per_L", "Risk_Score", "Microplastic_Size_mm_midpoint", "Density_midpoint"]
cat_cols = ["Location", "Shape", "Polymer_Type", "pH", "Salinity", "Industrial_Activity",
            "Population_Density", "Risk_Type", "Risk_Level", "Author"]

# -----------------------------
# 1. Upload & Preview
# -----------------------------
if selected_tab == tabs[0]:
    st.header("Step 1: Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel Dataset", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='latin1')
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("✅ Dataset uploaded successfully! Preview below:")
            st.subheader("Dataset Preview (First 10 Rows)")
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("""
            <details>
            <summary style='font-weight:bold'>Show full uploaded dataset</summary>
            """, unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown("</details>", unsafe_allow_html=True)
            st.info("Proceed to Data Preprocessing for cleaning and transformation.")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()
    elif st.session_state.df is not None:
        st.success("Dataset previously uploaded.")
        st.subheader("Dataset Preview (First 10 Rows)")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        st.markdown("""
        <details>
        <summary style='font-weight:bold'>Show full uploaded dataset</summary>
        """, unsafe_allow_html=True)
        st.dataframe(st.session_state.df, use_container_width=True)
        st.markdown("</details>", unsafe_allow_html=True)
        st.info("You may continue to Data Preprocessing.")

# -----------------------------
# 2. Data Preprocessing
# -----------------------------
elif selected_tab == tabs[1]:
    st.header("Step 2: Data Preprocessing")
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ Please upload a dataset in Step 1 first.")
        st.stop()

    df_prep = df.copy()
    # Outlier handling & skewness
    outlier_report = []
    for col in num_cols:
        if col in df_prep.columns:
            orig_stats = df_prep[col].describe()
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')
            nan_count = df_prep[col].isna().sum()
            if df_prep[col].notna().sum() > 0:
                Q1 = df_prep[col].quantile(0.25)
                Q3 = df_prep[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                num_clipped = ((df_prep[col] < lower) | (df_prep[col] > upper)).sum()
                df_prep[col] = np.clip(df_prep[col], lower, upper)
                outlier_report.append(f"- **{col}**: {num_clipped} outliers clipped. {nan_count} NaNs.")
            skew = df_prep[col].skew()
            if df_prep[col].notna().sum() > 0 and skew > 1:
                safe_log = df_prep[col] > -1
                df_prep.loc[safe_log, col] = np.log1p(df_prep.loc[safe_log, col])
                outlier_report.append(f"- **{col}**: Log transformation applied (skew={skew:.2f}).")
    # Encoding
    for col in cat_cols:
        if col in df_prep.columns:
            df_prep[col] = LabelEncoder().fit_transform(df_prep[col].astype(str))
    # Scaling
    scaler = StandardScaler()
    for col in num_cols:
        if col in df_prep.columns and df_prep[col].notna().sum() > 0:
            df_prep[col] = scaler.fit_transform(df_prep[[col]])

    st.session_state.df = df_prep
    st.session_state.preprocessed = True

    st.success("✅ Data preprocessing complete!")
    st.subheader("Preprocessed Dataset (First 10 Rows)")
    st.dataframe(df_prep.head(10), use_container_width=True)
    st.markdown("""
    <details>
    <summary style='font-weight:bold'>Show full preprocessed dataset</summary>
    """, unsafe_allow_html=True)
    st.dataframe(df_prep, use_container_width=True)
    st.markdown("</details>", unsafe_allow_html=True)

    with st.expander("Preprocessing Details & Report", expanded=False):
        st.markdown("**Outlier & Skewness Report:**")
        for report in outlier_report:
            st.markdown(report)
        st.markdown("""
        - **Categorical Encoding:** All categorical columns transformed with LabelEncoder.
        - **Scaling:** All numerical columns standardized using StandardScaler.
        """)

    with st.expander("Compare basic statistics before and after preprocessing", expanded=False):
        num_cols_present = [col for col in num_cols if col in df.columns]
        st.write("Original statistics (first numeric columns):")
        if num_cols_present:
            st.dataframe(df[num_cols_present].describe().T)
        else:
            st.warning("No valid numeric columns found for statistics in the uploaded dataset.")
        num_cols_prep_present = [col for col in num_cols if col in df_prep.columns]
        st.write("After preprocessing:")
        if num_cols_prep_present:
            st.dataframe(df_prep[num_cols_prep_present].describe().T)
        else:
            st.warning("No valid numeric columns found for statistics in the preprocessed dataset.")

# -----------------------------
# 3. Modeling & Performance (updated to give each model its own window/tab)
# -----------------------------
elif selected_tab == tabs[2]:
    st.header("Step 3: Modeling & Performance")
    df = st.session_state.df
    if df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ Please preprocess the data first.")
        st.stop()
    if "Risk_Type" not in df.columns or "Risk_Level" not in df.columns:
        st.warning("⚠️ Required columns for modeling not found in data.")
        st.stop()

    X = df.drop(columns=["Risk_Type", "Risk_Level"], errors="ignore")
    y_type = df['Risk_Type']
    y_level = df['Risk_Level']
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    y_type = y_type.fillna(0)
    y_level = y_level.fillna(0)

    model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    model_objs = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    st.markdown(
        "Each model below has its own window with metrics for Risk Type and Risk Level, cross-validation, and short interpretation."
    )

    # consistent train/test split for fair comparison across models
    X_train, X_test, y_train_type, y_test_type = train_test_split(X, y_type, test_size=0.2, random_state=42)
    _, _, y_train_level, y_test_level = train_test_split(X, y_level, test_size=0.2, random_state=42)

    # Create individual tabs/windows for each model so each has its own view
    model_tabs = st.tabs(model_names)
    for idx, name in enumerate(model_names):
        with model_tabs[idx]:
            st.subheader(f"{name} — Detailed Results")
            mod = model_objs[name]

            # Fit and evaluate for Risk Type
            try:
                mod_type = clone(mod)
                mod_type.fit(X_train, y_train_type)
                preds_type = mod_type.predict(X_test)
                perf_type = {
                    "Accuracy": accuracy_score(y_test_type, preds_type),
                    "Precision": precision_score(y_test_type, preds_type, average='weighted', zero_division=0),
                    "Recall": recall_score(y_test_type, preds_type, average='weighted', zero_division=0),
                    "F1-Score": f1_score(y_test_type, preds_type, average='weighted', zero_division=0)
                }
                st.write("Risk Type Classification Metrics")
                st.dataframe(pd.DataFrame(perf_type, index=[name]).T, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to evaluate Risk Type for {name}: {e}")

            # Fit and evaluate for Risk Level
            try:
                mod_level = clone(mod)
                mod_level.fit(X_train, y_train_level)
                preds_level = mod_level.predict(X_test)
                perf_level = {
                    "Accuracy": accuracy_score(y_test_level, preds_level),
                    "Precision": precision_score(y_test_level, preds_level, average='weighted', zero_division=0),
                    "Recall": recall_score(y_test_level, preds_level, average='weighted', zero_division=0),
                    "F1-Score": f1_score(y_test_level, preds_level, average='weighted', zero_division=0)
                }
                st.write("Risk Level Classification Metrics")
                st.dataframe(pd.DataFrame(perf_level, index=[name]).T, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to evaluate Risk Level for {name}: {e}")

            # K-Fold Cross Validation (Risk Type)
            try:
                st.subheader("K-Fold Cross Validation Accuracy (Risk Type)")
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(clone(mod), X, y_type, cv=kf, scoring='accuracy')
                st.bar_chart(cv_scores)
                st.info("The bar chart displays cross-validation fold accuracies for this model.")
            except Exception as e:
                st.error(f"Cross-validation failed for {name}: {e}")

            st.markdown("---")
            st.info(
                "Interpretation hints: Compare Accuracy and F1-Score to see whether the model balances overall correctness with per-class performance. "
                "Precision/Recall trade-offs indicate which types/levels are favored or missed."
            )

    # Model Metric Comparison (All Models) - summary across models for Risk Type
    st.subheader("Model Metric Comparison (Risk Type) — Summary")
    metrics_dict = {}
    for name, mod in model_objs.items():
        try:
            mod_tmp = clone(mod)
            mod_tmp.fit(X_train, y_train_type)
            pred = mod_tmp.predict(X_test)
            metrics_dict[name] = [
                accuracy_score(y_test_type, pred),
                precision_score(y_test_type, pred, average='weighted', zero_division=0),
                recall_score(y_test_type, pred, average='weighted', zero_division=0),
                f1_score(y_test_type, pred, average='weighted', zero_division=0)
            ]
        except Exception as e:
            metrics_dict[name] = [np.nan, np.nan, np.nan, np.nan]
            st.warning(f"Could not compute metrics for {name}: {e}")

    perf_df = pd.DataFrame(metrics_dict, index=["Accuracy", "Precision", "Recall", "F1-Score"])
    fig, ax = plt.subplots(figsize=(10, 5))
    perf_df.plot(kind='bar', ax=ax)
    ax.set_title("Model Comparison on Test Set (Risk Type)")
    st.pyplot(fig)
    st.info("This bar chart visually compares model performance across key metrics for Risk Type classification.")

    # -----------------------------
    # Visualizations BELOW modeling
    # -----------------------------
    st.header("Step 4: Visualizations & Data Interpretations")
    vis_options = [
        "Risk Score Distribution",
        "Risk Score vs MP_Count_per_L",
        "Risk Score by Risk Level"
    ]
    selected_vis = st.sidebar.selectbox("Choose a visualization:", vis_options, index=0)

    if selected_vis == vis_options[0]:
        st.subheader("Risk Score Distribution")
        if 'Risk_Score' in df.columns and df['Risk_Score'].notna().sum() > 0:
            fig, ax = plt.subplots()
            sns.histplot(df['Risk_Score'].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
            st.info("""
            **Interpretation:**  
            The histogram shows the distribution of overall risk scores assigned to the water samples. 
            Peaks in this graph indicate the most common risk levels present in the dataset after preprocessing. 
            A right-skewed plot would suggest many samples have low risk, while a more uniform or left-skewed shape would suggest higher risk is more prevalent.
            """)
        else:
            st.warning("Risk_Score column not found or empty.")

    elif selected_vis == vis_options[1]:
        st.subheader("Risk Score vs MP Count per Liter")
        if (
            'Risk_Score' in df.columns
            and 'MP_Count_per_L' in df.columns
            and df['Risk_Score'].notna().sum() > 0
            and df['MP_Count_per_L'].notna().sum() > 0
        ):
            fig, ax = plt.subplots()
            ax.scatter(df['Risk_Score'], df['MP_Count_per_L'])
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("MP Count per L")
            st.pyplot(fig)
            st.info("""
            **Interpretation:**  
            This scatter plot relates microplastic count per liter to the assigned risk score.
            A positive correlation suggests that areas with higher microplastic concentrations tend to have higher calculated risk scores.
            Outliers may represent samples where risk assessment doesn't strictly follow microplastic levels, possibly due to other environmental factors.
            """)
        else:
            st.warning("Required columns not found or empty.")

    elif selected_vis == vis_options[2]:
        st.subheader("Risk Score by Risk Level")
        if (
            'Risk_Level' in df.columns
            and 'Risk_Score' in df.columns
            and df['Risk_Score'].notna().sum() > 0
        ):
            fig, ax = plt.subplots()
            sns.boxplot(x=df['Risk_Level'], y=df['Risk_Score'], ax=ax)
            st.pyplot(fig)
            st.info("""
            **Interpretation:**  
            This boxplot visualizes how risk scores vary across different categorical risk levels.
            It showcases the central tendency (median) and spread for each risk group.
            If there is good separation between levels, it means the risk scoring system discriminates well between categories.
            Overlaps or outliers may suggest inconsistencies or areas needing further analysis.
            """)
        else:
            st.warning("Required columns not found or empty.")

# -----------------------------
# 4. Visualizations (standalone tab)
# -----------------------------
elif selected_tab == tabs[3]:
    st.header("Step 4: Visualizations & Data Interpretations")
    df = st.session_state.df
    if df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ Please preprocess the data first.")
        st.stop()

    vis_options = [
        "Risk Score Distribution",
        "Risk Score vs MP_Count_per_L",
        "Risk Score by Risk Level"
    ]
    selected_vis = st.sidebar.selectbox("Choose a visualization:", vis_options, index=0)

    if selected_vis == vis_options[0]:
        st.subheader("Risk Score Distribution")
        if 'Risk_Score' in df.columns and df['Risk_Score'].notna().sum() > 0:
            fig, ax = plt.subplots()
            sns.histplot(df['Risk_Score'].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
            st.info("""
            **Interpretation:**  
            The histogram shows the distribution of overall risk scores assigned to the water samples. 
            Peaks in this graph indicate the most common risk levels present in the dataset after preprocessing. 
            A right-skewed plot would suggest many samples have low risk, while a more uniform or left-skewed shape would suggest higher risk is more prevalent.
            """)
        else:
            st.warning("Risk_Score column not found or empty.")

    elif selected_vis == vis_options[1]:
        st.subheader("Risk Score vs MP Count per Liter")
        if (
            'Risk_Score' in df.columns
            and 'MP_Count_per_L' in df.columns
            and df['Risk_Score'].notna().sum() > 0
            and df['MP_Count_per_L'].notna().sum() > 0
        ):
            fig, ax = plt.subplots()
            ax.scatter(df['Risk_Score'], df['MP_Count_per_L'])
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("MP Count per L")
            st.pyplot(fig)
            st.info("""
            **Interpretation:**  
            This scatter plot relates microplastic count per liter to the assigned risk score.
            A positive correlation suggests that areas with higher microplastic concentrations tend to have higher calculated risk scores.
            Outliers may represent samples where risk assessment doesn't strictly follow microplastic levels, possibly due to other environmental factors.
            """)
        else:
            st.warning("Required columns not found or empty.")

    elif selected_vis == vis_options[2]:
        st.subheader("Risk Score by Risk Level")
        if (
            'Risk_Level' in df.columns
            and 'Risk_Score' in df.columns
            and df['Risk_Score'].notna().sum() > 0
        ):
            fig, ax = plt.subplots()
            sns.boxplot(x=df['Risk_Level'], y=df['Risk_Score'], ax=ax)
            st.pyplot(fig)
            st.info("""
            **Interpretation:**  
            This boxplot visualizes how risk scores vary across different categorical risk levels.
            It showcases the central tendency (median) and spread for each risk group.
            If there is good separation between levels, it means the risk scoring system discriminates well between categories.
            Overlaps or outliers may suggest inconsistencies or areas needing further analysis.
            """)
        else:
            st.warning("Required columns not found or empty.")
