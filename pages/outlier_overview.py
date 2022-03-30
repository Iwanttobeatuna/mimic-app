import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

## umap csv
df = pd.read_csv('umap(rm)_appended.csv')

## tsne csv
df_2 = pd.read_csv('tsne(rm)_appended.csv')

## umap original df
umap_df = pd.read_csv('umap_appended.csv')

## tsne original df
tsne_df = pd.read_csv('tsne_appended.csv')


## umap initial plot
fig_2d = px.scatter(
    df, x='umap_0', y='umap_1',
    color=df['labels'], labels={'color': df['labels']},
    hover_data=['icustay_id']
)

## tsne initial plot
fig_tsne = px.scatter(
    df_2, x='tsne_0', y='tsne_1',
    color=df_2['labels'], labels={'color': df_2['labels']},
    hover_data=['icustay_id']
)

fig_compare = px.scatter(
    df_2, x='tsne_0', y='tsne_1',
    color=df['labels'],
    hover_data=['icustay_id'],
    title = "UMAP labels on tsne plots"
)
    
def bar_plot(df):
    s = df['labels'].value_counts() 
    new = pd.DataFrame({'labels':s.index, 'Count':s.values})  

    # plot structure
    fig_bar = px.bar(new,
                x='labels',
                y = 'Count',
                title='Number of patients per label',
                text_auto = True)
    
    return fig_bar


layout = html.Div([
    dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Overview", href="/"),
                dbc.DropdownMenuItem("Overview(Outliers Removed)", href="/outlier-overview"),
            ],
            nav=True,
            in_navbar=True,
            label="Home",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Mean/Median Analysis", href="/page-1"),
                dbc.DropdownMenuItem("Mean/Median Analysis (outliers removed)", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="UMAP analysis",
        ),
        dbc.NavItem(dbc.NavLink("TSNE analysis", href="/page-2")),
        dbc.NavItem(dbc.NavLink("UMAP sampling analysis", href="/page-3")),
    ],
    brand="MIMIC Visualization",
    brand_href="/",
    color="primary",
    dark=True,),

    dbc.Container([

        html.Br(),

        dbc.Row([ 

            dbc.Row([html.H1(children = "UMAP and TSNE(outliers removed) summary", style={'text-align': 'center'}),]),

            dbc.Row([html.H3(children = "UMAP vs TSNE summary", style={'text-align': 'center'}),]),

        dbc.Col([

            html.Br(),

            dbc.Row([
            html.Div([
            dcc.Graph(figure = fig_compare)])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),

        
        dbc.Col([

            html.Br(),

            html.H5(children ="Analysis", style={'text-align': 'center'}),

            html.Br(),

            html.Ul([
                html.Li(children = "For TSNE label 0 (2883 patients), which corresponds to UMAP label 2 (2887 patients), only 1 patient that are in TSNE label 0 that are not present in  UMAP label 2, the patient's ID is: 265601."),
                html.Li(children = "For TSNE label 1 (135 patients), which corresponds to UMAP label 1 (125 patients), only 11 patients that are in TSNE label 1 that are not present in  UMAP label 1, the patient's ID are: 209881, 221594, 221720, 226013, 244588, 250103, 255146, 275725, 276135, 282744, 288246."),
                html.Li(children = "For TSNE label 2 (7166 patients), which corresponds to UMAP label 0 (7172 patients), only 3 patients that are in TSNE label 2 that are not present in  UMAP label 0, the patient's ID are: 220319, 257597, 263398."),
            ])

            
            
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),
    
        ]),

        dbc.Row([ 

        dbc.Col([

            html.Br(),

            dbc.Row([html.H5(children = "UMAP plots", style={'text-align': 'center'}),]),

            dbc.Row([
            html.Div([
            dcc.Graph(
                id='crossfilter-indicator-scatter',
                figure = fig_2d,
            )])]),

            dbc.Row([html.H5(children = "TSNE plots", style={'text-align': 'center'}),]),

            dbc.Row([
            html.Div([
            dcc.Graph(
                id='crossfilter-indicator-scatter',
                figure = fig_tsne,
                hoverData={'points': [{'customdata': [200001]}]}
            )])]),
            
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),

        
        
        dbc.Col([

            html.Br(),

            html.H5(children ="UMAP patient label count", style={'text-align': 'center'}),

            dbc.Row([
            html.Div([
            dcc.Graph(
                figure = bar_plot(df),
            )])]),

            html.H5(children ="TsNE patient label count", style={'text-align': 'center'}),

            dbc.Row([
            html.Div([
            dcc.Graph(
                figure = bar_plot(df_2),
            )])]),
            
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),
    
    ]),


    ]),
])
