import streamlit as st
import matplotlib as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
#page config#
st.set_page_config(page_title="Multiple Linear Regression App", layout="centered")
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")
# st.title("Linear Regression Web Application")
st.markdown("""
    <div class="card">
            <h1>Multple Regression Web Application</h1>
            <p>Predict <b>Tip Amount</b> based on <b>Total Bill and Size</b> using Multiple Linear Regression.</p>
            </div>""", unsafe_allow_html=True)
# Load dataset
def load_data():
    return sns.load_dataset('tips')
df=load_data()
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[["total_bill","size", "tip"]].head())
st.markdown('</div>', unsafe_allow_html=True)
#prepare data#
x,y=df[['total_bill',"size"]],df['tip']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)  
#train model#
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
#evaluate model#
mse=mean_squared_error(y_test,y_pred)
mae=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
st.markdown(f"""<div class="card">
    <h1>Model Evaluation Metrics</h1>
     <p>
    - **Mean Squared Error (MSE):** {mse:.2f}</br>
    - **Mean Absolute Error (MAE):** {mae:.2f}</br>
    - **Root Mean Squared Error (RMSE):** {rmse:.2f}</br>
    - **R² Score:** {r2:.2f}
</p>
</div>""", unsafe_allow_html=True)
#Visualizations#
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h1>Model Visualization</h1>', unsafe_allow_html=True)
plt.title("Total Bill vs Tip Amount with Multiple Linear Regression Line")
plt.xlabel("Total Bill")
plt.ylabel("Tip Amount")
plt.scatter(df["total_bill"], y, color='blue', label='Actual Tips')
plt.plot(df["total_bill"], model.predict(scaler.transform(df[["total_bill", "size"]])), color='red', linewidth=2, label='Predicted Tips')
plt.legend()
st.pyplot(plt.gcf())
st.markdown('</div>', unsafe_allow_html=True)
st.markdown(f"""<div class="card">
            <h1>Model Coefficients</h1>
            <p>
            <b>Intercept for total_bill:</b> {model.intercept_:.2f} <br>
            <b>Coefficient for Total Bill:</b> {model.coef_[0]:.2f}</br>
            <b>Coefficient for Size:</b> {model.coef_[1]:.2f}
            </p>
            </div>""", unsafe_allow_html=True)
st.subheader("Make Your Own Predictions")
total_bill_input = st.number_input("Enter Total Bill Amount:", min_value=0.0, step=0.5)
size_input = st.number_input("Enter Size of the Party:", min_value=1, step=1)
if st.button("Predict Tip"):
    scaled_input = scaler.transform([[total_bill_input, size_input]])
    predicted_tip = model.predict(scaled_input)[0]
    st.markdown(f'<div class="prediction-box">Predicted Tip Amount: {predicted_tip:.2f}</div>', unsafe_allow_html=True)