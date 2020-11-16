"""
dash_functions.py

This script contains functions called by dashboard.py to update the dashboard.
"""

import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go


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


def load_criteria_descriptions():
    """
    Load a table containing description of each criteria.
    """
    # Load data
    df_crit = pd.read_csv(
        '../data/data_inputs/HomeCredit_columns_description.csv', 
        encoding ='cp1252', usecols=[2,3])

    df_crit['options']=df_crit['Row'].apply(lambda x: {'label':x, 'value':x})
    
    return df_crit

    
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


def generate_top_tables(customer_id):
    """
    For a given customer id, retrieves the 15 criteria having most impact on loan decision
    """
    # Retrieve shap values for selected customer 
    df_shap = load_shap_values()
    df_1 = df_shap.loc[[customer_id]].T
    df_1.columns=['impact']

    # Retrieve criteria values for selected customer
    df_cust = load_customer_data()
    df_2 = df_cust.loc[[customer_id]].T
    df_2.columns=['customer values']

    # Merge
    df_table = df_1.merge(df_2, left_index=True, right_index=True)
    df_table['abs']=df_1['impact'].apply('abs')
    df_table['criteria'] = df_table.index
    
    # Top 15 table sorted by impact for selected customer
    df_table_c = df_table.sort_values(by='abs', ascending=False)
    df_table_c = df_table_c[['criteria', 'customer values', 'impact']].head(15)
    df_table_c = df_table_c.applymap(lambda x: round(x,3) if pd.api.types.is_number(x) else x)
    
    child_c = [
        html.Div(children=[
            html.H3(children='Top 15 criteria - Selected customer'),
            html.Table([
                html.Thead(html.Tr([html.Th(col) for col in df_table_c.columns])),
                html.Tbody([
                    html.Tr([html.Td(df_table_c.iloc[i][col]) for col in df_table_c.columns
                    ]) for i in range(len(df_table_c))])
                ])
        ], className='one-half column'
    )]
    
    # Top 15 table sorted by mean absolute impact for all customers
    overall_top = df_shap.apply('abs').mean().sort_values(ascending=False).head(15)
    df_overall = df_table.loc[overall_top.index]
    df_overall['mean abs impact'] = overall_top
    df_overall['criteria'] = df_overall.index
    df_overall = df_overall.applymap(lambda x: round(x,3) if pd.api.types.is_number(x) else x)
    df_overall = df_overall[['criteria', 'mean abs impact', 'customer values', 'impact']]

    child_o = [
        html.Div(children=[
            html.H3(children='Top 15 criteria - Overall'),
            html.Table([
                html.Thead(html.Tr([html.Th(col) for col in df_overall.columns])),
                html.Tbody([
                    html.Tr([html.Td(df_overall.iloc[i][col]) for col in df_overall.columns
                    ]) for i in range(len(df_overall))])
            ])
        ], className='one-half column'
    )]
    
    # Append and return children
    children=child_c+child_o
    return children

