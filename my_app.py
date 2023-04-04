import streamlit as st
from PIL import Image
import pandas as pd      
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from sklearn.model_selection import cross_validate, cross_val_score
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500)
from sklearn.linear_model import LinearRegression
pd.options.display.float_format = '{:.3f}'.format
from yellowbrick.regressor import PredictionError
from yellowbrick.features import RadViz
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.features import RadViz
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from scipy.stats import skew

from sklearn.model_selection import cross_validate, cross_val_score
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500)
from sklearn.linear_model import Lasso, LassoCV





st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 550px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

st.sidebar.title("Car Price Prediction")
html_temp= """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;tect-align:center;">Streamlit ML Cloud App</hp">
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)


age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3,4))
hp=st.sidebar.slider("What is the hp of your car:", 40,200,step=1)
km=st.sidebar.slider("What is the km of your car:",0,200000,step=500)
gearing_type=st.sidebar.radio("Select gear type",('Automatic','Manual','Semi-automatic'))
car_model=st.sidebar.selectbox("Select model of your car",('A1','A3','Astra','Corsa','Insignia','Clio',"Duster","Espace"))

columns=["km","hp","age","model_A1","model_A3","model_Astra","model_Clio","model_Corsa","model_Duster","model_Espace","model_Insignia","Gearing_Automatic","Gearing_Manual","Gearing_Semi-automatic"]


model_name=st.selectbox("Select your model:",("Random Forest","Decision Tree","Lasso"))

if model_name=="Random Forest":
    model=pickle.load(open("last_model_RF","rb"))
    st.success("You selected {} model".format(model_name))
elif model_name=="Decision Tree":
    model=pickle.load(open("last_model_Dt","rb"))
    st.success("You selected {} model".format(model_name))
elif model_name=="Lasso":
    model=pickle.load(open("last_model_Lasso","rb"))
    st.success("You selected {} model".format(model_name))
ASK=st.selectbox("Would You like to select from the table?",("No","Yes"))

if ASK=="NO":
    st.stop()

elif ASK=="Yes":
    st.header("Real infos exsit below")
    st.success("You can now select a row from table")
    st.table(pd.read_csv("df1.csv"))
    z=st.selectbox("Please select index number:",(0,1,2,3,4))
    st.success("You selected {}. index from table".format(z))
    r=pd.read_csv("df1.csv").loc[[z]]
    t=pd.get_dummies(pd.read_csv("df1.csv").loc[[z]]).reindex(columns=columns,fill_value=0)
    if st.button("Press to predict"):
        prediction=model.predict(t)
        st.success("The estimated price of your car is {} Euro ".format(int(prediction[0])))
        #o=100-abs((float(prediction)-float(r.price.values))/(float(prediction))/100)
        #st.success("The Model is working with % {} Success".format(o))

my_dict={
    "age" : age,
    "hp":hp,
    "km":km,
    "model":car_model,
    "Gearing":gearing_type
}
x=pd.DataFrame.from_dict([my_dict])
st.header("The Configuration of your car is below")
st.table(x)


x=pd.get_dummies(x).reindex(columns=columns,fill_value=0)

st.subheader("Press predict if configuration is okay")

print(x)

if st.button("Predict"):
    prediction=model.predict(x)
    st.success("The estimated price of your car is {} Euro ".format(int(prediction[0])))




