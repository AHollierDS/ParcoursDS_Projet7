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
import plotly.express as px


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
        html.Div(
            [html.Div(
                [html.Img(src='https://user.oc-static.com/upload/2019/02/25/15510866018677_logo%20projet%20fintech.png')],
                className="one-third column"),
            html.Div(
                [html.H1(children='Decision-making dashboard')],
                className='one-half column'),
            ],className='row flex-display'),
        
        html.Div(children='Select a customer ID to explain why granted/denied the loan'),
        
        # Customer selection and decision
        html.H2('Customer selection :'),
        dcc.Dropdown(
            id='customer_selection',
            options=df_decision['option'].tolist()
        ),
        
        html.Label('Final decision :'),
        html.Div(id='customer_decision'),
        
        # Customer position vs customers panel
        html.H2(children='Customer position in customer panel'),
        dcc.Graph(id='panel', figure=plot_panel(thres))
    ])
   

    # Callbacks and component updates
    
    # Decision about selected customet
    @app.callback(
        Output(component_id='customer_decision', component_property='children'),
        Input(component_id='customer_selection', component_property='value')
    )
    def update_decision_outputs(customer_id):
        
        decision = df_decision[df_decision['SK_ID_CURR']==customer_id]['LOAN'].values[0]
        risk = df_decision[df_decision['SK_ID_CURR']==customer_id]['TARGET'].values[0]
        
        decision = 'granted' if decision else 'denied'
        output = 'Estimated risk = {:.1%} - Loan is {}'.format(risk, decision)
        
        return output
    
    # Panel figure
    @app.callback(
         Output(component_id='panel', component_property='figure'),
         Input(component_id='customer_selection', component_property='value')
    )
    def update_panel(customer_id):
        fig = plot_panel(thres)
        
        cust_target = df_decision[df_decision['SK_ID_CURR']==customer_id]['TARGET'].values[0]
        heights = np.histogram(df_decision['TARGET'], bins=np.arange(0,1,0.01))[0]
        heights = heights/heights.sum()
        cust_height = 100*heights[int(cust_target//0.01)]

        fig.add_shape(type='rect',x0=cust_target//0.01/100, 
                  x1=cust_target//0.01/100 + 0.01, 
                  y0=0, y1=cust_height, fillcolor='yellow')
        
        return fig
    
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
    
def plot_panel(thres):
    
    # Plot customer risk distribution
    df_decision = load_decisions(thres=thres)
    fig = px.histogram(df_decision, x='TARGET', nbins=100, histnorm='percent',
                  title='Distribution of estimated risk on a representative panel',
                  labels={'TARGET':'Estimated risk'})
    fig.update_layout(yaxis_title='% of customers', xaxis_tickformat = ',.0%')
    
    # Display threshold
    fig.add_shape(type='line', x0=thres, x1=thres, y0=0, y1=15, 
                  line_color='red', line_dash='dot')
    fig.add_annotation(text='Maximum allowed risk ({:.1%})'.format(thres),
                       x=thres, y=15)
    
    return fig