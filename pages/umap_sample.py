import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pages import umap_sample10

## umap csv
df = pd.read_csv('umap_appended.csv')

## list of features
features_lst = ['heart_rate', 'respiratory_rate', 'hematocrit', 'creatinine', 'sodium', 'fraction_inspired_oxygen', 'mean_blood_pressure']

## Split into dataframes based on labels (umap)
df_umap_1 = df.loc[df['labels'] == 0]
df_umap_2 = df.loc[df['labels'] == 1]
df_umap_3 = df.loc[df['labels'] == 2]
df_umap_1_describe = df_umap_1.describe()
df_umap_2_describe = df_umap_2.describe()
df_umap_3_describe = df_umap_3.describe()


## sampling function for each label
def sample_plot(df_label, df_describe, n, features, y_label):
    sample = df_label.sample(n)
    new_df = sample.loc[:, sample.columns.to_series().str.contains(features)]
    transpose  = new_df.transpose()
    transpose.columns = ['sample'+'_{}'.format(i) for i in range(len(transpose.columns))]
    ## mean median
    mean_df = df_describe.loc[['mean'], df_describe.columns.to_series().str.contains(features)]
    median_df = df_describe.loc[['50%'], df_describe.columns.to_series().str.contains(features)]
    # var_df = df_describe.loc[['std'], df_describe.columns.to_series().str.contains(features)]
    mean_Transpose = mean_df.transpose()
    median_Transpose = median_df.transpose()    
    # var_Transpose = var_df.transpose()
    ## merge the two dataframes
    mean_median_Transpose = pd.merge(mean_Transpose, median_Transpose, left_index=True, right_index=True)
    
    # mean_median_Transpose = pd.merge(mean_median_Transpose, var_Transpose, left_index=True, right_index=True)
    mean_median_Transpose.columns = ['mean', 'median']

    fig = go.Figure()
    for i in range(len(transpose.columns)):
        fig.add_trace(go.Scatter(x=transpose.index, y=transpose.iloc[:,i], name=transpose.columns[i], line = dict(width=0.5, color="#aaaaaa")))
    fig.add_trace(go.Scatter(x=mean_median_Transpose.index, y=mean_median_Transpose['mean'], name='mean', line=dict(color='firebrick', width=2)))
    fig.add_trace(go.Scatter(x=mean_median_Transpose.index, y=mean_median_Transpose['median'], name='median', line=dict(color='royalblue', width=2)))
    # fig.add_trace(go.Scatter(x=mean_median_Transpose.index, y=mean_median_Transpose['std'], name='std', line=dict(color='green', width=2)))
    fig.update_layout(xaxis_title='Time', yaxis_title=y_label)
    if features=='heart_rate':
        fig.update_layout(yaxis_range=[80,100])
    if features=='respiratory_rate':
        fig.update_layout(yaxis_range=[15,30])
    if features=='hematocrit':
        fig.update_layout(yaxis_range=[28,32])
    if features=='creatinine':
        fig.update_layout(yaxis_range=[0.8,1.8])
    if features=='mean_blood_pressure':
        fig.update_layout(yaxis_range=[70,120])
    if features=='fraction_inspired_oxygen':
        fig.update_layout(yaxis_range=[0.0, 0.8])
    if features=='sodium':
        fig.update_layout(yaxis_range=[135, 145])
    return fig


layout = html.Div([

    dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("UMAP analysis", href="/page-1")),
        dbc.NavItem(dbc.NavLink("TSNE analysis", href="/page-2")),
    ],
    brand="MIMIC Visualization",
    brand_href="/",
    color="primary",
    dark=True,),

    html.Br(),
    
    dbc.Row([html.H1(children='UMAP sample analysis (5 samples)', style={'textAlign': 'center'})], style={'padding': '10px'}),

    dbc.Row([html.H3(children='**Double click the legend to isolate individual plots**', style={'textAlign': 'center'})], style={'padding': '10px'}),



    ## drop down enter the number of samples
    # html.H5(children='Select the features', style={'textAlign': 'center'}),

    # dbc.Col([dcc.Dropdown(['NYC', 'MTL', 'SF'], 'NYC', id='demo-dropdown')]),

    ## drop down enter the number of samples
    html.H5(children='Select the number of samples', style={'textAlign': 'center'}),

    dbc.Col([
    dbc.Button(
            "5", id="example-button-5", className="me-2",  color="warning", href="/umap-sample-5",
        ),
    dbc.Button(
            "10", id="example-button-10", className="me-2",  color="warning", href="/umap-sample-10",
        ),
    dbc.Button(
            "20", id="example-button-20", className="me-2",  color="warning", href="/umap-sample-20",
        ),
    ], width = 12, style = {'textAlign': 'center'}),

    html.Br(),
    
    # dbc.Row(dcc.Slider(id='my-slider', value=10, min=0, max=20, step=5), style={'textAlign': 'center'}),


    dbc.Row
    ([
         dbc.Col([
            
            dbc.Row([

                html.H5('Label 0', style={'textAlign': 'center'}),

                html.Div([dcc.Graph(figure = sample_plot(df_umap_1, df_umap_1_describe, 5, "heart_rate", "heart_rate"))], 
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
               html.Div([dcc.Graph(figure = sample_plot(df_umap_1, df_umap_1_describe, 5, "respiratory_rate", "respiratory_rate"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_1, df_umap_1_describe, 5, "hematocrit", "hematocrit"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_1, df_umap_1_describe, 5, "fraction_inspired_oxygen", "fraction_inspired_oxygen"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_1, df_umap_1_describe, 5, "creatinine", "creatinine"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
               html.Div([dcc.Graph(figure = sample_plot(df_umap_1, df_umap_1_describe, 5, "mean_blood_pressure", "mean_blood_pressure"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_1, df_umap_1_describe, 5, "sodium", "sodium"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),
        ],width=4, sm=4, md=4, lg=4),
        
        
        dbc.Col([
            
            dbc.Row([

                html.H5('Label 1', style={'textAlign': 'center'}),

                html.Div([dcc.Graph(figure = sample_plot(df_umap_2, df_umap_2_describe, 5, "heart_rate", "heart_rate"))], 
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
               html.Div([dcc.Graph(figure = sample_plot(df_umap_2, df_umap_2_describe, 5, "respiratory_rate", "respiratory_rate"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_2, df_umap_2_describe, 5, "hematocrit", "hematocrit"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_2, df_umap_2_describe, 5, "fraction_inspired_oxygen", "fraction_inspired_oxygen"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_2, df_umap_2_describe, 5, "creatinine", "creatinine"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
               html.Div([dcc.Graph(figure = sample_plot(df_umap_2, df_umap_2_describe, 5, "mean_blood_pressure", "mean_blood_pressure"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_2, df_umap_2_describe, 5, "sodium", "sodium"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),
        ],width=4, sm=4, md=4, lg=4),

        dbc.Col([
            
            dbc.Row([

                html.H5('Label 3', style={'textAlign': 'center'}),

                html.Div([dcc.Graph(figure = sample_plot(df_umap_3, df_umap_3_describe, 5, "heart_rate", "heart_rate"))], 
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
               html.Div([dcc.Graph(figure = sample_plot(df_umap_3, df_umap_3_describe, 5, "respiratory_rate", "respiratory_rate"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_3, df_umap_3_describe, 5, "hematocrit", "hematocrit"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_3, df_umap_3_describe, 5, "fraction_inspired_oxygen", "fraction_inspired_oxygen"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_3, df_umap_3_describe, 5, "creatinine", "creatinine"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
               html.Div([dcc.Graph(figure = sample_plot(df_umap_3, df_umap_3_describe, 5, "mean_blood_pressure", "mean_blood_pressure"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df_umap_3, df_umap_3_describe, 5, "sodium", "sodium"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),
        ],width=4, sm=4, md=4, lg=4),
    ]),

])


