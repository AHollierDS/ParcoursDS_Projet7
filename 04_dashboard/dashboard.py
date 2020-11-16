"""
dashboard.py

This script generate an interpretability dashboard for explaining why a customer
was granted the loan he/she applied for.
"""

import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_functions

from dash.dependencies import Input, Output


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
    df_decision = dash_functions.load_decisions(thres=thres)
    df_crit=dash_functions.load_criteria_descriptions()
    
    logo = 'https://user.oc-static.com/upload/2019/02/25/15510866018677_'+\
        'logo%20projet%20fintech.png'
    
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    # Dashboard layout
    app = dash.Dash(external_stylesheets=external_stylesheets)
    
    app.layout = html.Div(children=[
        
        # Dash title
        html.Div(
            [html.Div(
                [html.Img(src=logo, width=218, height=200)],
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
        dcc.Graph(id='panel', figure=dash_functions.plot_panel(thres)),
        
        # Waterfall plot
        html.H2(children='Waterfall for selected customer'),
        dcc.Graph(id='waterfall'),
        
        # Top criteria for selected customer
        html.H2(children='Most important criteria'),
        html.Div(id='top_tables'),
        
        # Criteria selection and description
        html.H2(children='Criteria description'),
        dcc.Dropdown(
            id='criteria_selection',
            options=df_crit['options'].tolist()
        ),
        
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
        fig = dash_functions.plot_panel(thres)
        
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
        fig = dash_functions.plot_waterfall(customer_id, thres=thres)
        return fig
    
    
    # Criteria tables
    @app.callback(
        Output(component_id='top_tables', component_property='children'),
        Input(component_id='customer_selection', component_property='value')
    )
    
    def update_top_tables(customer_id):
        """
        """
        children = dash_functions.generate_top_tables(customer_id)
        #children = [html.H3('test'), html.H4('test aussi')]
        return children
    
    
    # Run the dashboard
    app.run_server()
