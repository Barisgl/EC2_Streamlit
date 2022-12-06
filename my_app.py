import streamlit as st
from PIL import Image
import pandas as pd      
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from sklearn.model_selection import cross_validate, cross_val_score
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = (10,6)
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
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500)
from sklearn.linear_model import Lasso, LassoCV

#def train_val(model, X_train, y_train, X_test, y_test):
    
    #y_pred = model.predict(X_test)
   # y_train_pred = model.predict(X_train) # overfitting var mı yok mu kıyaslamak için
    
    #scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    #"mae" : mean_absolute_error(y_train, y_train_pred),
    #"mse" : mean_squared_error(y_train, y_train_pred),                          
    #"rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    #"test": {"R2" : r2_score(y_test, y_pred),
   #"mae" : mean_absolute_error(y_test, y_pred),
    #"mse" : mean_squared_error(y_test, y_pred),
    #"rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    
    #return pd.DataFrame(scores)


###df=pd.read_csv("last_data.csv")
#X=df.drop(columns=["price"])
#y=df.price
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
# Random
#random_final= RandomForestRegressor(n_estimators= 500,max_depth=None,min_samples_leaf=2,min_samples_split=8,max_features=2)
#random_final.fit(X,y)

#Decision
#Decision_T= DecisionTreeRegressor(max_depth=None,min_samples_leaf=2,min_samples_split=8,splitter='random')
#Decision_T.fit(X,y)

#Lasso
#scaler = MinMaxScaler()
#scaler.fit(X_train)
#X_train_scaled = scaler.transform(X_train)
#X_test_scaled = scaler.transform(X_test)
#alpha_space = np.linspace(0.01, 100, 100)
#lasso_model = Lasso(random_state=42)

#param_grid = {'alpha':alpha_space}

#lasso_grid_model = GridSearchCV(estimator=lasso_model,
 #                         param_grid=param_grid,
  #                        scoring='neg_root_mean_squared_error',
   #                       cv=10,
    #                      n_jobs = -1)
#lasso_grid_model.fit(X_train_scaled,y_train)
#lasso_grid_model.fit(X,y)

#pickle.dump(random_final,open("last_model_RF","wb"))
#pickle.dump(Decision_T,open("last_model_Dt","wb"))
#pickle.dump(lasso_grid_model,open("last_model_Lasso","wb"))





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
img=Image.open("x.png")
st.sidebar.image(img,caption="R2",width=500)


columns=["km","hp_Kw","age","make_model_A1","make_model_A3","make_model_Corsa","make_model_Astra",

"make_model_Insignia","make_model_Clio","make_model_Duster","make_model_Espace",
"Gearing_Type_Automatic","Gearing_Type_Manual","Gearing_Type_Semi-automatic"]


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
    "hp_Kw":hp,
    "km":km,
    "make_model":car_model,
    "Gearing_Type":gearing_type
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




