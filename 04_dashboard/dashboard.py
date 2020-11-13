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
import plotly.graph_objs as go


def generate(thres=0.5, n_sample = 1000):
    """
    Build and display the dashboard.
    
    params:
        thres:
            Threshold risk value above which a customer's loan is denied.
        n_sample : 
            Number of customers to include in the dashboard
            
    returns:
        a web application displaying the interpretability dashboard
    """
    
    app = dash.Dash()
    
    # Load data
    df_decision = load_decisions(thres=thres)
    #df_cust = load_customer_data(n_sample=n_sample)
    #df_shap = load_shap_values(n_sample=n_sample)

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
        dcc.Graph(id='panel', figure=plot_panel(thres)),
        
        # Waterfall plot
        html.H2(children='Waterfall for selected customer'),
        dcc.Graph(id='waterfall')
        
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
    
    
    # Waterfall plot
    @app.callback(
         Output(component_id='waterfall', component_property='figure'),
         Input(component_id='customer_selection', component_property='value')
    )
    
    def update_waterfall(customer_id):
        fig = plot_waterfall(customer_id, thres=thres)
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


def load_customer_data(n_sample=None):
    """
    Load dataset containing customer information (after feature engineering).
    Dataset is restricted to the n_sample-th first customers.
    """
    
    df_cust = pd.read_pickle('../data/data_outputs/feature_engineered/cleaned_dataset_test.pkl')
    df_cust.index = df_cust['SK_ID_CURR']
    l_drop = ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index', 'source']
    feats = [f for f in df_cust.columns if f not in l_drop]
    df_cust=df_cust[feats]
    
    if n_sample != None :
        df_cust = df_cust.iloc[:n_sample]
    
    return df_cust


def load_shap_values(n_sample=None):
    """
    Load dataset containing shap values for each customer.
    Dataset is restricted to the n_sample-th first customers.
    """
    
    df_shap = pd.read_csv('../03_interpretability/test_shap_values.csv')
    df_shap.index=df_shap['SK_ID_CURR']
    df_shap = df_shap.drop(columns = ['SK_ID_CURR'])
    
    if n_sample != None :
        df_shap = df_shap.iloc[:n_sample]
    
    return df_shap

    
def plot_panel(thres):
    """
    Display estimated risk on a representative panel of customers.
    """
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


def find_base_value(thres):
    """
    """
    # Find ID of last granted and first denied
    df_decision = load_decisions(thres)
    df_decision.sort_values(by='TARGET', inplace=True)
    last_granted = df_decision[df_decision['LOAN']]['SK_ID_CURR'].tail(1).values[0]
    first_denied = df_decision[~df_decision['LOAN']]['SK_ID_CURR'].head(1).values[0]
    
    print(last_granted)
    print(first_denied)
    
    # Calculate respective total SHAP values
    df_shap = load_shap_values()
    last_shap = df_shap.loc[last_granted].sum()
    first_shap = df_shap.loc[first_denied].sum()
    
    print(last_shap)
    print(first_shap)
    
    # Base value
    base = -(last_shap+first_shap)/2
    return base


def plot_waterfall(customer_id, thres):
    """
    """
    # Load data
    df_shap = load_shap_values()   
    
    # Set data for waterfall
    df_waterfall = pd.DataFrame(df_shap.loc[customer_id])
    df_waterfall.columns = ['values']
    df_waterfall['abs']=df_waterfall['values'].apply('abs')
    df_waterfall.sort_values(by='abs', inplace=True)
    
    # Aggregate shap values not in top 20
    df_top=df_waterfall.tail(20)
    df_others = pd.DataFrame(df_waterfall.iloc[:-20].sum(axis=0)).T
    df_others.index = ['others']
    df_waterfall = df_others.append(df_top)
    
    base_value = find_base_value(thres)
    
    # Plot waterfall
    fig = go.Figure(
        go.Waterfall(
            base = base_value,
            orientation = 'h',
            y=df_waterfall.index,
            x=df_waterfall['values']),
        layout = go.Layout(height=600, width=800)
    )
    
    # Add base value and final result
    
    fig.add_shape(type='line', x0=base_value, x1=base_value, y0=-1, y1=1)
    fig.add_annotation(text='Base value', x=base_value, y=0)
    
    final_value = df_waterfall['values'].sum() + base_value
    fig.add_shape(type='line', x0=final_value, x1=final_value, y0=20, y1=21)
    fig.add_annotation(text='score = {:.3}'.format(final_value), 
                       x=final_value, y=21)
    
    # Threshold line
    fig.add_shape(type='line', x0=0, x1=0, y0=-1, y1=21, line_color='red', line_dash='dot')
    
    return fig

    