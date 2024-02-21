import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
final_df=pd.read_excel('Train_Breast_Cancer.xlsx')
final_df['Breast-quad'].replace({'?':'left_low'},inplace=True)
final_df['Node-caps'].replace({'?':'no'},inplace=True)
final_x= final_df.iloc[:,:-1]
final_y= final_df.iloc[:,-1]
encoder=LabelEncoder()
final_y=encoder.fit_transform(final_y)
x_train,x_test,y_train,y_test=train_test_split(final_x,final_y,test_size=0.25,random_state=0,stratify=final_y)
final_trf=ColumnTransformer([
    ('final_encoder',OrdinalEncoder(),[0,1,2,3,4,5,7,8])
],remainder='passthrough')
final_model=LogisticRegression()
final_pipe=Pipeline([
    ('final_trf',final_trf),
    ('final_model',final_model)
])
final_pipe.fit(x_train,y_train)



st.title('Radiotherapy Prediction in Breast Cancer')

Class = st.selectbox('Class',('no-recurrence-events', 'recurrence-events'))
Age = st.selectbox('Age',('30-39' ,'40-49', '60-69', '50-59', '70-79', '20-29'))
Menopause = st.selectbox('Menopause',('premeno', 'ge40' ,'lt40'))
Tumor_size = st.selectbox('Tumor-size',('30-34' ,'20-24', '15-19' ,
                                        '0-4', '25-29' ,'50-54' ,'10-14',
                                        '40-44', '35-39','5-9' ,'45-49'))
Inv_nodes = st.selectbox('Inv-nodes',('0-2', '6-8', '3-5', '15-17' ,'9-11' ,'12-14' ,'24-26'))
Node_caps = st.selectbox('Node-caps',('no' ,'yes'))
Deg_malig = st.number_input('Deg-malig',min_value=1,max_value=3)
Breast = st.selectbox('Breast',('left' ,'right'))
Breast_quad = st.selectbox('Breast-quad',('left_low', 'right_up', 'left_up', 'right_low' ,'central'))

user_input=pd.DataFrame(data=np.array([Class,Age,Menopause,Tumor_size,Inv_nodes,
                                       Node_caps, Deg_malig,Breast,Breast_quad ]).reshape(1,9),columns=final_x.columns)
st.write(user_input)
Irradiate=final_pipe.predict(user_input)
if st.button("Predict Radiotherapy"):
    Irradiate=final_pipe.predict(user_input)
    if Irradiate == 1:
        st.write('the patient has to take radiotherapy')
    else:
        st.write('Radiotherapy is not needed')