'''
dashboard.py

This script generate an interpretability dashboard for explaining why a customer
was granted the loan he/she applied for.
'''

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import shap

from joblib import load


def generate():
    
    app = dash.Dash()

    # Dashboard layout
    app.layout = html.Div(children=[
        html.H1(children='Decision-making dashboard'),
        html.Div(children='Select a customer ID to explain why granted/denied the loan')
    ])
    
    app.run_server()
