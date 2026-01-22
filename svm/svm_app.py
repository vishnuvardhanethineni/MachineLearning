import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Loan Prediction using SVM", layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------- TITLE CARD ----------------
st.markdown("""
<div class="card">
    <h1>SVM Loan Prediction Application</h1>
    <p>Predict whether a loan will be approved using a <b>Support Vector Machine</b>.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD & PREPROCESS DATA ----------------
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

df['Credit_History'].ffill(inplace=True)
df['Dependents'].ffill(inplace=True)
df['Self_Employed'].ffill(inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

df.dropna(subset=['Married', 'Gender'], inplace=True)
df.drop('Loan_ID', axis=1, inplace=True)

df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# ---------------- DATASET CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head(10), width="stretch")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MODEL TRAINING ----------------
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ---------------- MODEL PERFORMANCE CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h1>Model Performance</h1>", unsafe_allow_html=True)
st.markdown(f"<p><b>Accuracy:</b> {acc*100:.2f}%</p>", unsafe_allow_html=True)

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Rejected", "Approved"],
    cmap="Blues",
    ax=ax
)
ax.set_title("Confusion Matrix")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HYPERSPACE VISUALIZATION ----------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h1>SVM Hyperspace Visualization (PCA)</h1>", unsafe_allow_html=True)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("SVM Decision Space (2D Projection)")
st.pyplot(plt.gcf())

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- USER PREDICTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Make a Prediction")

Gender = st.selectbox('Gender', ('Male', 'Female'))
Married = st.selectbox('Married', ('Yes', 'No'))
Dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'))
Education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
Self_Employed = st.selectbox('Self Employed', ('Yes', 'No'))

ApplicantIncome = st.number_input('Applicant Income', min_value=0)
CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
LoanAmount = st.number_input('Loan Amount', min_value=0)

Loan_Amount_Term = st.selectbox(
    'Loan Amount Term', (12, 36, 60, 84, 120, 180, 240, 300, 360)
)

Credit_History = st.selectbox('Credit History', (0.0, 1.0))
Property_Area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))

input_df = pd.DataFrame([{
    'Gender': 1 if Gender == 'Male' else 0,
    'Married': 1 if Married == 'Yes' else 0,
    'Dependents': 3 if Dependents == '3+' else int(Dependents),
    'Education': 1 if Education == 'Graduate' else 0,
    'Self_Employed': 1 if Self_Employed == 'Yes' else 0,
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Property_Area': 2 if Property_Area == 'Urban' else 1 if Property_Area == 'Semiurban' else 0
}])

scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

if st.button("Predict Loan Status"):
    result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"
    st.markdown(f'<div class="prediction-box">{result}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
