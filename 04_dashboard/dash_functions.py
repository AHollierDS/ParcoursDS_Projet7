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


source_path = '../data/data_dashboard/'

def load_decisions(thres, n_sample=None):
    """
    Load submissions made on the test set and prepare data for dashboard.
    
    params:
        thres:
            Threshold risk value above which a customer's loan is denied.
        n_sample : 
            Number of customers to include in the dataset.
            If None, all customers are included.
    
    returns:
        A DataFrame containing estimated risk and loan verdict for each customer.     
    """
    # Load dataset and set decision with the threshold
    file='submission.csv'
    
    df_decision = pd.read_csv(source_path+file)
    df_decision.sort_values(by='SK_ID_CURR', inplace=True)
    df_decision=df_decision.iloc[:n_sample]
    df_decision['LOAN'] = df_decision['TARGET']<thres
    
    # Dict for customer selection
    df_decision['option'] = df_decision['SK_ID_CURR'].apply(
        lambda x : {'label': str(x), 'value':x})
    
    return df_decision


def load_customer_data(n_sample=None):
    """
    Load dataset containing customer information (after feature engineering).
    Dataset is restricted to the n_sample-th first customers.
    
    params:
        n_sample : 
            Number of customers to include in the dataset.
            If None, all customers are included.
    
    returns:
        A DataFrame containing preprocessed data describing all customers.
    """
    # Load data
    file = 'customers_values.csv.gzip'
    df_cust=pd.read_csv(source_path+file, compression='gzip')
    df_cust.index = df_cust['SK_ID_CURR']
    
    # Select features and sample dataset
    l_drop = ['Unnamed: 0', 'TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index', 'source']
    feats = [f for f in df_cust.columns if f not in l_drop]
    df_cust=df_cust[feats]
    
    if n_sample != None :
        df_cust = df_cust.iloc[:n_sample]
    
    return df_cust


def load_shap_values(n_sample=None):
    """
    Load dataset containing shap values for each customer.
    Dataset is restricted to the n_sample-th first customers.
    
    params:
        n_sample : 
            Number of customers to include in the dataset.
            If None, all customers are included.
    
    returns:
        A DataFrame containing shapley values for all criteria per customer.
    """
    # Load data
    file='shap_values.csv.gzip'
    df_shap = pd.read_csv(source_path+file, compression='gzip')
    df_shap.index=df_shap['SK_ID_CURR']
    df_shap = df_shap.drop(columns = ['SK_ID_CURR', 'Unnamed: 0'])
    
    # Sample dataset
    if n_sample != None :
        df_shap = df_shap.iloc[:n_sample]
    
    return df_shap


def load_criteria_descriptions():
    """
    Load a table containing description of each criteria.
    
    returns:
        Filtered HomeCredit columns description provided for the project.
    """
    # Load data
    file='HomeCredit_columns_description.csv'
    df_crit = pd.read_csv(source_path+file, encoding ='cp1252', usecols=[2,3])

    # Drop customer id and target
    l_drop = ['SK_ID_CURR', 'TARGET']
    df_crit=df_crit[~df_crit['Row'].isin(l_drop)]
    
    # Define options for dropdown
    df_crit['options']=df_crit['Row'].apply(lambda x: {'label':x, 'value':x})
    
    return df_crit

    
def plot_panel(df_decision, thres):
    """
    Display estimated risk on a representative panel of customers.
    
    params:
        df_decision:
            A loan decision DataFrame.
        thres:
            Threshold risk value above which a customer's loan is denied.
            
    returns:
        A figure displaying distribution of estimated risk on the panel.
    """
    # Plot customer risk distribution
    fig = px.histogram(df_decision, x='TARGET', nbins=100, histnorm='percent',
                  title='Distribution of estimated risk on a representative panel',
                  labels={'TARGET':'Estimated risk'})
    fig.update_layout(yaxis_title='% of customers', xaxis_tickformat = ',.0%',
                     margin_t=30)
    
    # Display threshold
    fig.add_shape(type='line', x0=thres, x1=thres, y0=0, y1=15, 
                  line_color='red', line_dash='dot')
    fig.add_annotation(text='Maximum allowed risk ({:.0%})'.format(thres),
                       x=thres, y=15)
    
    return fig


def find_base_value(df_decision, df_shap, thres):
    """
    Calculate shapley base value based on the threshold.
    
    params:
        df_decision:
            A loan decision DataFrame.
        df_shap:
            A Shapley values DataFrame.
        thres:
            Threshold risk value above which a customer's loan is denied.
            
    returns:
        Initial score in the waterfall. 
    """
    # Find ID of last granted and first denied
    df_decision.sort_values(by='TARGET', inplace=True)
    last_granted = df_decision[df_decision['LOAN']]['SK_ID_CURR'].tail(1).values[0]
    first_denied = df_decision[~df_decision['LOAN']]['SK_ID_CURR'].head(1).values[0]
    
    # Calculate respective total SHAP values
    last_shap = df_shap.loc[last_granted].sum()
    first_shap = df_shap.loc[first_denied].sum()
    
    # Base value
    base = -(last_shap+first_shap)/2
    return base


def plot_waterfall(df_decision, df_shap, customer_id, n_top, thres):
    """
    Calculate waterfall based on shapley values for a given customer.
    
    params:
        df_decision:
            A loan decision DataFrame.
        df_shap:
            A Shapley values DataFrame.
        customer_id :
            The SK_ID_CURR value of the customer for whom application decision
            will be explained.
        n_top:
            Number of top criteria to display.
        thres:
            Threshold risk value above which a customer's loan is denied.
            
    returns:
        The waterfall figure for selected customer.
        Loan applications with a final score below 0 are denied.
    """ 
    # Set data for waterfall
    df_waterfall = pd.DataFrame(df_shap.loc[customer_id])
    df_waterfall.columns = ['values']
    df_waterfall['abs']=df_waterfall['values'].apply('abs')
    df_waterfall.sort_values(by='abs', inplace=True)
    
    # Aggregate shap values not in top n
    df_top=df_waterfall.tail(n_top)
    df_others = pd.DataFrame(df_waterfall.iloc[:-n_top].sum(axis=0)).T
    df_others.index = [f'others (n={len(df_waterfall.iloc[:-n_top])})']
    df_waterfall = df_others.append(df_top)
    
    base_value = find_base_value(df_decision, df_shap, thres)
    
    # Plot waterfall
    fig = go.Figure(
        go.Waterfall(
            base = base_value,
            orientation = 'h',
            y=df_waterfall.index,
            x=df_waterfall['values']),
        layout = go.Layout(
            height=200+(25*n_top), 
            #width=600,
            xaxis_title='Confidence score', 
            yaxis_title = 'Criteria',
            yaxis_side='right',
            yaxis_tickfont=dict(size=10),
            margin_l=10, margin_r=10, 
            margin_t=30, margin_b=10
        )
    )
    
    # Add base value and final result on the plot
    fig.add_shape(type='line', x0=base_value, x1=base_value, y0=-1, y1=1)
    fig.add_annotation(text='Base value', x=base_value, y=0)
    
    final_value = df_waterfall['values'].sum() + base_value
    fig.add_shape(type='line', x0=final_value, x1=final_value, y0=n_top, y1=n_top+1)
    fig.add_annotation(text='score = {:.3}'.format(final_value), 
                       x=final_value, y=n_top+1)
    
    # Threshold line
    fig.add_shape(type='line', x0=0, x1=0, y0=-1, y1=n_top+1, 
                  line_color='red', line_dash='dot')
    
    return fig


def generate_top_tables(n_top, df_cust, df_shap, customer_id):
    """
    For a given customer id, retrieves the n_top criteria having most impact on loan decision.
    
    params:
        n_top:
            Number of top criteria to display.
        df_cust:
            A customer description DataFrame.
        df_shap:
            A Shapley values DataFrame.
        customer_id :
            The SK_ID_CURR value of the customer for whom main criteria are searched.
    
    returns:
        Two tables of main criteria :
        - Top decisive criteria for selected customer;
        - Top decisive overall criteria compared to the customer values.
    """
    # Retrieve shap values for selected customer 
    df_1 = df_shap.loc[[customer_id]].T
    df_1.columns=['Impact']

    # Retrieve criteria values for selected customer
    df_2 = df_cust.loc[[customer_id]].T
    df_2.columns=['Values']

    # Merge
    df_table = df_1.merge(df_2, left_index=True, right_index=True)
    df_table['abs']=df_1['Impact'].apply('abs')
    df_table['Criteria'] = df_table.index
    df_table['Criteria']=df_table['Criteria'].apply(lambda x:str(x)[:30])
    
    # Top n table sorted by impact for selected customer
    df_table_c = df_table.sort_values(by='abs', ascending=False)
    df_table_c = df_table_c[['Criteria', 'Values', 'Impact']].head(n_top)
    df_table_c = df_table_c.applymap(lambda x: round(x,3) if pd.api.types.is_number(x) else x)
    
    child_c = [
        html.Div(children=[
            html.H3(children=f'Top {n_top} criteria - Customer'),
            html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in df_table_c.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            children=df_table_c.iloc[i][col], 
                            style={'fontSize':10}
                        ) for col in df_table_c.columns
                    ]) for i in range(len(df_table_c))])
                ])
        ], className='five columns'
    )]
    
    # Top table sorted by mean absolute impact for all customers
    overall_top = df_shap.apply('abs').mean().sort_values(ascending=False).head(n_top)
    df_overall = df_table.loc[overall_top.index]
    df_overall['Mean impact'] = overall_top
    df_overall['Criteria'] = df_overall.index
    df_overall['Criteria']=df_overall['Criteria'].apply(lambda x:str(x)[:30])
    df_overall = df_overall.applymap(lambda x: round(x,3) if pd.api.types.is_number(x) else x)
    df_overall = df_overall[['Criteria', 'Mean impact', 'Values', 'Impact']]

    child_o = [
        html.Div(children=[
            html.H3(children=f'Top {n_top} criteria - Overall'),
            html.Table([
                html.Thead(html.Tr([html.Th(col) for col in df_overall.columns])),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            children=df_overall.iloc[i][col],
                            style={'fontSize':10}
                        ) for col in df_overall.columns
                    ]) for i in range(len(df_overall))])
            ], style={"textOverflow":'ellipsis', 'overflow':'hidden', 'maxWidth':0})
        ], className='seven columns'
    )]
    
    # Append and return children
    children=child_c+child_o
    return children


def plot_shap_scatter(df_cust, df_shap, crit, cust):
    """
    Shows evolution of SHAP value depending on selected criteria's value.
    
    params:
        df_cust:
            A customer description DataFrame.
        df_shap:
            A Shapley values DataFrame.
        crit:
            The criteria to we want to plot.
        cust:
            A customer's ID.
            If not None, shows where the selected customer stands on the plot.
            
    returns:
        A partial dependence plot, where x is the criteria value and y the SHAP value.
    """
    # Shap values
    s_shap=df_shap[[crit]].copy()
    s_shap.columns=['shap_value']
    
    # Criteria values
    s_vals = df_cust[[crit]].copy()
    s_vals.columns=['crit_value']
    
    # Join data & visualization
    df_summary = s_shap.join(s_vals)
    fig=px.scatter(df_summary, x='crit_value', y='shap_value', opacity=0.1)
    
    fig.update_layout(
        xaxis_title='Criteria value',  yaxis_title = 'Impact', margin_t=30
    )
    
    # Selected customer data
    if cust is not None:
        # Customer Shap and value for selected criteria
        df_summary['selected_customer'] = df_summary.index==cust
        cust_value=df_summary[df_summary['selected_customer']]['crit_value'].values[0]
        cust_shap=df_summary[df_summary['selected_customer']]['shap_value'].values[0]
        
        # Show selected customer on the plot
        fig.add_shape(type='line', 
              x0=cust_value, x1=cust_value, 
              y0=df_summary['shap_value'].min(), y1=cust_shap,
              line_dash='dot', line_color='red')

        fig.add_shape(type='line', 
                      x0=df_summary['crit_value'].min(), x1=cust_value, 
                      y0=cust_shap, y1=cust_shap,
                      line_dash='dot', line_color='red')

    return fig

    
