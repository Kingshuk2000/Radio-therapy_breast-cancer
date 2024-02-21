import streamlit as st
import pandas as pd
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
final_y_pred=final_pipe.predict(x_test)


import pickle
pickle.dump(final_pipe,open('breast-cancer.pk1','wb'))
loaded_model=pickle.load(open('breast-cancer.pk1','rb'))
result=loaded_model.score(x_test,y_test)
print(result)