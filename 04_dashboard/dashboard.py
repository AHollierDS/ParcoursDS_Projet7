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

    # Load data
    df_decision = load_decisions(thres=thres)
    logo = 'https://user.oc-static.com/upload/2019/02/25/15510866018677_'+\
        'logo%20projet%20fintech.png'
    headers_list = ['Criteria name', "Customer's value", "Impact"]
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    # Dashboard layout
    app = dash.Dash(external_stylesheets=external_stylesheets)
    
    app.layout = html.Div(children=[
        
        # Dash title
        html.Div(
            [html.Div(
                [html.Img(src=logo)],
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
        dcc.Graph(id='waterfall'),
        
        # Top criteria for selected customer
        html.H2(children='Most important criteria for selected customer'),
        html.Table(id='table_customer')
        
    ])

    # Callbacks and component updates
    
    # Decision about selected customet
    @app.callback(
        Output(component_id='customer_decision', component_property='children'),
        Input(component_id='customer_selection', component_property='value')
    )
    
    def update_decision_outputs(customer_id):
        """
        Display loan decision and the evaluated risk for a given customer
        """
        # Classification
        decision = df_decision[df_decision['SK_ID_CURR']==customer_id]['LOAN'].values[0]
        decision = 'granted' if decision else 'denied'
        
        # Risk estimation
        risk = df_decision[df_decision['SK_ID_CURR']==customer_id]['TARGET'].values[0]
        
        # Output
        output = 'Estimated risk = {:.1%} - Loan is {}'.format(risk, decision)
        return output
    
    
    # Panel figure
    @app.callback(
         Output(component_id='panel', component_property='figure'),
         Input(component_id='customer_selection', component_property='value')
    )
    
    def update_panel(customer_id):
        """
        Highlight customer's bin in the evaluated risk distribution 
        """
        # Plot risk distribution
        fig = plot_panel(thres)
        
        # Find customer's bin
        cust_target = df_decision[df_decision['SK_ID_CURR']==customer_id]['TARGET'].values[0]
        heights = np.histogram(df_decision['TARGET'], bins=np.arange(0,1,0.01))[0]
        heights = heights/heights.sum()
        cust_height = 100*heights[int(cust_target//0.01)]

        # Highlight customer's bin
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
        """
        Display waterfall to explain loan decision for a given customer.
        """
        fig = plot_waterfall(customer_id, thres=thres)
        return fig
    
    
    # Criteria tables
    @app.callback(
        Output(component_id='table_customer', component_property='children'),
        Input(component_id='customer_selection', component_property='value')
    )
    
    def update_customer_table(customer_id):
        """
        """
        children = generate_customer_table(customer_id)
        return children
    
    
    # Run the dashboard
    app.run_server()

    
def load_decisions(thres):
    """
    Load submissions made on the test set and prepare data for dashboard.
    """
    # Load dataset and set decision with the threshold
    df_decision = pd.read_csv('../02_classification/submission.csv')
    df_decision['LOAN'] = df_decision['TARGET']<thres
    
    # Dict for customer selection
    df_decision['option'] = df_decision['SK_ID_CURR'].apply(
        lambda x : {'label': str(x), 'value':x})
    
    return df_decision


def load_customer_data(n_sample=None):
    """
    Load dataset containing customer information (after feature engineering).
    Dataset is restricted to the n_sample-th first customers.
    """
    # Load data
    df_cust=pd.read_pickle(
        '../data/data_outputs/feature_engineered/cleaned_dataset_test.pkl')

    df_cust.index = df_cust['SK_ID_CURR']
    
    # Select features and sample dataset
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
    # Load data
    df_shap = pd.read_csv('../03_interpretability/test_shap_values.csv')
    df_shap.index=df_shap['SK_ID_CURR']
    df_shap = df_shap.drop(columns = ['SK_ID_CURR'])
    
    # Sample dataset
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
    Calculate shapley base value based on the threshold.
    """
    # Find ID of last granted and first denied
    df_decision = load_decisions(thres)
    df_decision.sort_values(by='TARGET', inplace=True)
    last_granted = df_decision[df_decision['LOAN']]['SK_ID_CURR'].tail(1).values[0]
    first_denied = df_decision[~df_decision['LOAN']]['SK_ID_CURR'].head(1).values[0]
    
    # Calculate respective total SHAP values
    df_shap = load_shap_values()
    last_shap = df_shap.loc[last_granted].sum()
    first_shap = df_shap.loc[first_denied].sum()
    
    # Base value
    base = -(last_shap+first_shap)/2
    return base


def plot_waterfall(customer_id, thres):
    """
    Calculate waterfall based on shapley values.
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
    
    # Add base value and final result on the plot
    fig.add_shape(type='line', x0=base_value, x1=base_value, y0=-1, y1=1)
    fig.add_annotation(text='Base value', x=base_value, y=0)
    
    final_value = df_waterfall['values'].sum() + base_value
    fig.add_shape(type='line', x0=final_value, x1=final_value, y0=20, y1=21)
    fig.add_annotation(text='score = {:.3}'.format(final_value), 
                       x=final_value, y=21)
    
    # Threshold line
    fig.add_shape(type='line', x0=0, x1=0, y0=-1, y1=21, 
                  line_color='red', line_dash='dot')
    
    return fig


def generate_customer_table(customer_id):
    """
    """
    # Retrieve shap values for selected customer 
    df_shap = load_shap_values()
    df_1 = df_shap.loc[[customer_id]].T
    df_1.columns=['impact']

    # Retrieve criteria values for selected customer
    df_cust = load_customer_data()
    df_2 = df_cust.loc[[customer_id]].T
    df_2.columns=['customer values']

    # Merge and sort by impact
    df_table = df_1.merge(df_2, left_index=True, right_index=True)
    df_table['abs']=df_1['impact'].apply('abs')
    df_table['criteria'] = df_table.index
    df_table.sort_values(by='abs', ascending=False, inplace=True)
    df_table = df_table[['criteria', 'customer values', 'impact']].head(15)
    
    child = [
        html.Thead(
            html.Tr([html.Th(col) for col in df_table.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df_table.iloc[i][col]) for col in df_table.columns
            ]) for i in range(len(df_table))
        ])
    ]
    
    return child 
    
        