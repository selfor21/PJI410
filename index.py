from flask import Flask , render_template, request
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import pandas as pd
from pandas import json_normalize
import dash_bootstrap_components as dbc
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

app = Flask(__name__)

df = pd.read_csv('HR.csv') 
X = df.drop('Attrition', axis = 1)
y = df['Attrition']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100, max_depth=3)
clf.fit(X_train,y_train)


y_pred=clf.predict(X_test)
acuracia_modelo = metrics.accuracy_score(y_test, y_pred)
print("Acuracia do Modelo:", round(acuracia_modelo,4))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/dash", methods=["GET", "POST"])
def dash(): 
    print(request.form.get('attributes'))
    if request.form.get('attributes') :    
        col_chosen = request.form.get('attributes')
    else :
       col_chosen ="Age"      
    fig = px.histogram(df, x=col_chosen, y='Attrition', histfunc='count', color="Attrition")
    fig.write_html("./templates/teste.html")
    return render_template("dash.html")

@app.route("/teste")
def teste(): 
    return render_template("teste.html")

@app.route('/send', methods=['POST'])
def prediction():
    data = request.json   
    df3 = json_normalize(data) 
    y3 =clf.predict(df3)    
    
    if y3 == 0:
        return('0')
    else:
        return('1')
            

   # Run the app
if __name__ == '__main__':
    app.run(debug=True)