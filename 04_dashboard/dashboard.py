"""
dashboard.py

This script generate an interpretability dashboard for explaining why a customer
was granted the loan he/she applied for.
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import shap

from joblib import load


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
            options=df_decision['option'].tolist()
        )
    ])
    
    # Run the dashboard
    app.run_server()

    
def load_decisions(thres):
    """
    Load submissions made on the test set and prepare data for dashboard.
    """
    
    df_decision = pd.read_csv('../02_classification/submission.csv')
    df_decision['LOAN'] = df_decision['TARGET']<thres
    df_decision['option'] = df_decision['SK_ID_CURR'].apply(
        lambda x : {'label': str(x), 'value':str(x)})
    
    return df_decision
    