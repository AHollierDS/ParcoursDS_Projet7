"""
dashboard.py

This script generate an interpretability dashboard for explaining why a customer
was granted the loan he/she applied for.
"""


import pandas as pd
import numpy as np
import shap

from joblib import load

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


def generate(thres=0.5):
    """
    Build and display the dashboard.
    
    params:
        thres:
            Threshold risk value above which a customer's loan is denied.
            
    returns:
        a web application displaying the interpretability dashboard
    """
    
    app = dash.Dash()
    
    # Load data
    df_decision = load_decisions(thres=thres)

    # Dashboard layout
    app.layout = html.Div(children=[
        
        # Dash title
        html.H1(children='Decision-making dashboard'),
        html.Div(children='Select a customer ID to explain why granted/denied the loan'),
        
        # Customer selection and decision
        html.Label('Customer selection :'),
        dcc.Dropdown(
            id='customer_selection',
            options=df_decision['option'].tolist()
        ),
        
        html.Label('Final decision :'),
        html.Div(id='customer_decision')
    ])
    
    @app.callback(
        Output(component_id='customer_decision', component_property='children'),
        [Input(component_id='customer_selection', component_property='value')]
    )
    def update_decision_outputs(customer_id):
        
        decision = df_decision[df_decision['SK_ID_CURR']==customer_id]['LOAN'].values[0]
        risk = df_decision[df_decision['SK_ID_CURR']==customer_id]['TARGET'].values[0]
        
        decision = 'granted' if decision else 'denied'
        output = 'Estimated risk = {:.1%} - Loan is {}'.format(risk, decision)
        
        return output
    
    # Run the dashboard
    app.run_server()

    
def load_decisions(thres):
    """
    Load submissions made on the test set and prepare data for dashboard.
    """
    
    df_decision = pd.read_csv('../02_classification/submission.csv')
    df_decision['LOAN'] = df_decision['TARGET']<thres
    df_decision['option'] = df_decision['SK_ID_CURR'].apply(
        lambda x : {'label': str(x), 'value':x})
    
    return df_decision
    