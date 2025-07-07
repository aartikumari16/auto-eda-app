# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io, json, zipfile
# from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

# Page Setup
st.set_page_config(page_title="Auto EDA Dashboard", page_icon="📊", layout="wide")

# Sidebar
# Sidebar
st.sidebar.image("eda_banner.png", use_container_width=True)   # ✅ fixed
st.sidebar.markdown("## 🚀 Auto EDA Dashboard")
st.sidebar.markdown("A simple yet powerful tool to explore your CSV datasets instantly.")
st.sidebar.markdown("**Created by:** Aarti")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/aartikumari16)")



# App Title
st.markdown(
    """
    <style>
    .title {
        font-size:40px !important;
        font-weight: 700;
        color: #2e7efb;
    }
    .subtitle {
        font-size:18px;
        color: #555;
    }
    .footer {
        font-size:14px;
        color: #aaa;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<h1 class="title">📊 Auto EDA Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your CSV file and explore your data with one click!</p>', unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Shape
    st.markdown("---")
    st.subheader("📐 Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Dtypes
    st.markdown("---")
    st.subheader("🔢 Column Data Types")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    # Info
    st.markdown("---")
    st.subheader("ℹ️ Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # Head
    st.markdown("---")
    st.subheader("👀 First 5 Rows")
    st.dataframe(df.head())

    # Summary
    st.markdown("---")
    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe())

    # Nulls
    st.markdown("---")
    st.subheader("❓ Missing Values")
    st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"}))

    # Duplicates
    st.markdown("---")
    st.subheader("🔁 Duplicate Records")
    st.write(f"Total duplicate rows: **{df.duplicated().sum()}**")

    # 📌 Correlation Heatmap (upper triangle only)
    st.markdown("---")
    st.subheader("📌 Correlation Heatmap")

    num_df = df.select_dtypes(include=np.number)

    if not num_df.empty:
        # Compute correlations
        corr = num_df.corr()

        # Mask the lower triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Auto‑scale the figure size: wider datasets → larger heatmap
        n_cols = corr.shape[1]
        fig_size = min(0.5 * n_cols + 4, 20)   # cap at 20×20 so it never gets enormous
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Plot heatmap
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75},
            ax=ax
        )
        ax.set_title("Upper‑Triangle Correlation Matrix", pad=12)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation heatmap.")


    # Histogram + Boxplot
    st.markdown("---")
    st.subheader("📈 Histogram & Boxplot")
    for col in num_df.columns:
        st.markdown(f"**🔹 {col}**")
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=axs[0], color='skyblue')
        axs[0].set_title('Histogram')
        sns.boxplot(x=df[col], ax=axs[1], color='orange')
        axs[1].set_title('Boxplot')
        st.pyplot(fig)

    # 🧼 Missing‑Value Treatment
    st.markdown("---")
    st.subheader("🧼 Missing Value Treatment")

    missing_cols = df.columns[df.isnull().any()]

    if len(missing_cols) == 0:
        st.success("🎉 No missing values found in the dataset!")
    else:
        # ▼––– everything below runs ONLY if we actually have nulls –––▼
        selected_missing_col = st.selectbox(
            "Select a column with missing values",
            missing_cols,
            key="missing_col_selector"
    )
        if selected_missing_col:           # guard
            col_data = df[selected_missing_col]
            st.write(f"🔎 Missing: {col_data.isnull().sum()} values in `{selected_missing_col}`")

        # Pick an imputation method
            if np.issubdtype(col_data.dtype, np.number):
                method = st.radio(
                    "Choose how to handle the missing values:",
                    ["Fill with Mean", "Fill with Mode", "Drop Rows with Nulls"],
                    key="missing_method"
                )
            else:
                method = st.radio(
                    "Choose how to handle the missing values:",
                    ["Fill with Mode", "Drop Rows with Nulls"],
                    key="missing_method"
                )

        # Apply button
            if st.button("✅ Apply Treatment", key="apply_missing"):
                if method == "Fill with Mean":
                    df[selected_missing_col].fillna(col_data.mean(), inplace=True)
                    st.success("Filled with column mean.")
                elif method == "Fill with Mode":
                    df[selected_missing_col].fillna(col_data.mode()[0], inplace=True)
                    st.success("Filled with column mode.")
                elif method == "Drop Rows with Nulls":
                    df.dropna(subset=[selected_missing_col], inplace=True)
                    st.success("Rows containing nulls in this column have been dropped.")

                # Show updated count
                st.write(
                    f"🧼 Updated missing count: "
                    f"{df[selected_missing_col].isnull().sum()}"
                )

    # 📌 Custom Column Summary Tool
    st.markdown("---")
    st.subheader("🔍 Column Summary Tool")

    selected_col = st.selectbox("Select a column to explore in detail", df.columns)

    if selected_col:
        col_data = df[selected_col]
        st.write(f"**🧠 Data Type:** `{col_data.dtype}`")
        st.write(f"**🕳️ Missing Values:** {col_data.isnull().sum()} / {len(col_data)} ({round(col_data.isnull().mean()*100, 2)}%)")
        st.write(f"**🔢 Unique Values:** {col_data.nunique()}")

    if col_data.dtype == 'object' or col_data.dtype.name == 'category':
        st.write("📊 **Top Value Counts:**")
        st.dataframe(col_data.value_counts().head(10).reset_index().rename(columns={'index': 'Value', selected_col: 'Count'}))

    if np.issubdtype(col_data.dtype, np.number):
        st.write("📈 **Distribution & Boxplot:**")
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(col_data.dropna(), kde=True, ax=axs[0], color='mediumseagreen')
        axs[0].set_title('Histogram')
        sns.boxplot(x=col_data, ax=axs[1], color='coral')
        axs[1].set_title('Boxplot')
        st.pyplot(fig)

    

    # st.markdown("---")
    # st.subheader("📄 Download EDA Insights Report")




# if st.button("📊  Generate EDA Report"):
#     # 1️⃣  Build the report and save it to disk (HTML)
#     profile = ProfileReport(df, title="📌 Auto‑EDA Report", explorative=True)
#     profile.to_file("auto_eda_report.html")

#     # 2️⃣  (Optional) Tiny text summary — handy for quick copy‑paste
#     summary = io.StringIO()
#     summary.write("Auto EDA Quick Stats\n")
#     summary.write(f"Rows: {len(df)}   |   Columns: {df.shape[1]}\n")
#     summary.write("\nMissing‑value overview:\n")
#     summary.write(df.isnull().sum().to_string())
#     summary.seek(0)
#     with open("eda_quick_summary.txt", "w", encoding="utf‑8") as f:
#         f.write(summary.read())

#     # 3️⃣  Bundle both files into an in‑memory ZIP
#     zip_buffer = io.BytesIO()
#     with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
#         zf.write("auto_eda_report.html")
#         zf.write("eda_quick_summary.txt")
#     zip_buffer.seek(0)

#     # 4️⃣  Download button
#     st.success("✅  Report generated!  Download below:")
#     st.download_button(
#         label="⬇️  Download Auto EDA Bundle",
#         data=zip_buffer,
#         file_name="auto_eda_report.zip",
#         mime="application/zip"
#     )  

    # Footer
    st.markdown(
        '<div class="footer">Made with ❤️ using Streamlit · © 2025 Aarti</div>',
        unsafe_allow_html=True
    )
else:
    st.info("Please upload a CSV file to begin your analysis.")
