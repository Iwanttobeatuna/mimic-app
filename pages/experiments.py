import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from viz import scatter_plot

umap_df = pd.read_csv('umap_appended.csv')

tsne_df = pd.read_csv('tsne_appended.csv')

umap_rm_df = pd.read_csv('umap(rm)_appended.csv')

umap_normal_validated = pd.read_csv('umap(rm)_normal_validated.csv')

umap_valid_validated = pd.read_csv('umap(rm)_valid_validated.csv')

umap_against_normal = pd.read_csv('umap(rm)validate_against_normal.csv')

umap4_normal_validated = pd.read_csv('umap4(rm)normal_validate.csv')

umap4_valid_validated = pd.read_csv('umap4(rm)valid_validate.csv')

umap4_against_valid = pd.read_csv('umap4(rm)validate_against_valid.csv')

umap4_against_normal = pd.read_csv('umap4(rm)validate_against_normal.csv')

tsne_rm_df = pd.read_csv('tsne(rm)_appended.csv')



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
        dbc.NavItem(dbc.NavLink('Experiments', href='/page-4')),
    ],
    brand="MIMIC Visualization",
    brand_href="/",
    color="primary",
    dark=True,),

    dbc.Container([

        html.Br(),

        dbc.Row([ 

            dbc.Row([html.H5(children = "UMAP remove outliers' plot with different standardization Method (7 features)", style={'text-align': 'center'}),]),
            html.P(children="7 features includes: Heart rate, Creatinine, Hematocrit, Fi02, Mean Blood Pressure, Respiratory Rate, Sodium"),

        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap_valid_validated, 'Valid Validated', type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),

        
        dbc.Col([


            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap_normal_validated, 'Normal Validated', type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),
    
        ]),

        dbc.Row([ 

        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap_rm_df, 'Standardized against Valid', type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),

        
        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap_against_normal, 'Standardized against Normal', type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),
    
        ]),

        html.Br(),

        dbc.Row([ 

            dbc.Row([html.H5(children = "UMAP remove outliers' plot with different standardization Method (4 features)", style={'text-align': 'center'}),]),
            html.P(children="4 features includes: Heart rate, Respiratory Rate, Creatinine, Hematocrit"),

        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap4_valid_validated, 'Valid Validated', type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),

        
        dbc.Col([


            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap4_normal_validated, 'Normal Validated', type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),
    
        ]),

        dbc.Row([ 

        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap4_against_valid, 'Standardized against Valid', type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),

        
        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap4_against_normal, 'Standardized against Normal', type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),
    
        ]),

    ]),
])
