#PROJETO INTEGRADOR IV - PJI 41- DRP CAMPINAS - GRUPO 18
#ANÁLISE DE DADOS
#MACHINE LEARNING
#PEOPLE ANALYTICS

from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#LEITURA BASE DADOS
df = pd.read_csv('HR_02.csv')


#MACHINE LEARNING - ALGORITMO RANDOM FOREST CLASSIFIER
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

df3 = pd.read_csv('HR_03.csv')
X3 = df3.drop('Attrition', axis = 1)
y3 =clf.predict(X3)

print('O resultado da sua consulta é (0 = Não / 1 = Sim ', y3)


# Initialize the app - incorporate css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = html.Div([
    html.Div(className='row', children='Desligamentos por Atributos', style={'textAlign':'center', 'color': 'blue', 'fontSize': 42}),
    html.Hr(),

    dcc.Dropdown(options=["Idade", "US$ / Dia", "Distância de Casa","US$ / Hora","Nível do Cargo","Função","Estado Conjugal",
            "Recebimento Mensal", "US$ / Mes","Qtde Empresas Trabalhadas","Hora Extra","Opções de Ações","Tempo de Trabalho",
            "Tempo de Empresa", "Tempo mesma função","Tempo com Gestor Atual"], value='Idade', id='controls-and-radio-item'),


    #dcc.RadioItems(options=["Idade", "US$ / Dia", "Distância de Casa","US$ / Hora","Nível do Cargo","Função","Estado Conjugal",
     #       "Recebimento Mensal", "US$ / Mes","Qtde Empresas Trabalhadas","Hora Extra","Opções de Ações","Tempo de Trabalho",
      #      "Tempo de Empresa", "Tempo mesma função","Tempo com Gestor Atual"], value='Idade', id='controls-and-radio-item'),

    #dash_table.DataTable(data=df.to_dict('records'), page_size=6),

    dcc.Graph(figure={}, id='controls-and-graph')
])

# Add controls to build the interaction
@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Input(component_id='controls-and-radio-item', component_property='value')
)
def update_graph(col_chosen):
    #fig = px.histogram(df, x='Attrition', y=col_chosen, histfunc='count')
    fig = px.histogram(df, x=col_chosen, y='Attrition', histfunc='count', color="Attrition")
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
