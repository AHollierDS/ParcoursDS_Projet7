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


def generate(thres=0.5, n_sample=10000):
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
    df_decision=dash_functions.load_decisions(thres=thres, n_sample=n_sample)
    df_crit=dash_functions.load_criteria_descriptions()
    df_cust=dash_functions.load_customer_data(n_sample=n_sample)
    df_shap=dash_functions.load_shap_values(n_sample=n_sample)
    
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
        dcc.Graph(id='panel', figure=dash_functions.plot_panel(df_decision, thres)),
        
        # Waterfall plot
        html.H2(children='Waterfall for selected customer'),
        dcc.Graph(id='waterfall'),
        
        # Top criteria for selected customer
        html.H2(children='Most important criteria'),
        html.Div(id='top_tables'),
        
        # Criteria selection and description
        html.H2(children='Criteria description'),
        html.Div(
            children=[
                html.H3(children='Select a criteria'),
                dcc.Dropdown(id='crit_selection',options=df_crit['options'].tolist()),
                html.H3(children='Description :'),    
                html.Div(id='crit_descr') 
            ], 
            className='one-third column'),
        
        # Shap vs value scatter plot
        html.Div(
            children=[
                html.H3(id='scatter_title'),
                dcc.Graph(id='scatter_plot')
            ],
            className='one-half column')
       
        
    ])

    # Callbacks and component updates
    
    # Callback when new customer is selected
    @app.callback(
        [Output(component_id='customer_decision', component_property='children'),
         Output(component_id='panel', component_property='figure'),
         Output(component_id='waterfall', component_property='figure'),
         Output(component_id='top_tables', component_property='children')],
        Input(component_id='customer_selection', component_property='value')
    )
    
    def update_customer(customer_id):
        """
        Update decision, position in panel, waterfall and top 15 criteria
        when a customer is selected in dropdown.
        """
        # Update loan decison
        decision = df_decision[df_decision['SK_ID_CURR']==customer_id]['LOAN'].values[0]
        decision = 'granted' if decision else 'denied'
        risk = df_decision[df_decision['SK_ID_CURR']==customer_id]['TARGET'].values[0]
        decision_output = 'Estimated risk = {:.1%} - Loan is {}'.format(risk, decision)
        
        # Update customer panel
        fig_panel = dash_functions.plot_panel(df_decision, thres)
        
        cust_target = df_decision[df_decision['SK_ID_CURR']==customer_id]['TARGET'].values[0]
        heights = np.histogram(df_decision['TARGET'], bins=np.arange(0,1,0.01))[0]
        heights = heights/heights.sum()
        cust_height = 100*heights[int(cust_target//0.01)]
        
        fig_panel.add_shape(type='rect',x0=cust_target//0.01/100, 
                  x1=cust_target//0.01/100 + 0.01, 
                  y0=0, y1=cust_height, fillcolor='yellow')
        
        # Update waterfall
        fig_waterfall = dash_functions.plot_waterfall(df_decision, df_shap, customer_id, thres=thres)
        
        # Update top 15 tables
        children_top = dash_functions.generate_top_tables(df_cust, df_shap, customer_id)
        
        return decision_output, fig_panel, fig_waterfall, children_top
    
    
    # Callbacks with a new criteria selected
    @app.callback(
        [Output(component_id='crit_descr', component_property='children'),
         Output(component_id='scatter_title', component_property='children'),
         Output(component_id='scatter_plot', component_property='figure')],
        [Input(component_id='crit_selection', component_property='value'),
         Input(component_id='customer_selection', component_property='value')]
    )
    
    def update_description(crit, cust=None):
        """
        Plot scatter plot for evolution of impact with change in criteria value.
        """
        output=df_crit[df_crit['Row']==crit]['Description'].values[0]
        title=f'Evolution of impact with {crit} value :'
        fig=dash_functions.plot_shap_scatter(df_cust, df_shap, crit, cust)
        
        return output, title, fig
    
    
    # Run the dashboard
    app.run_server()
