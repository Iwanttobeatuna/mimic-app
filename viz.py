import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

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

def extract_df_label(df, label):
    df_label = df.loc[df['label'] == label]
    return df_label

## return df.describe() for each labels
def extract_df_describe_3labels(df, label1, label2, label3):
    df_describe_1 = extract_df_label(df, label1).describe()
    df_describe_2 = extract_df_label(df, label2).describe()
    df_describe_3 = extract_df_label(df, label3).describe()
    return df_describe_1, df_describe_2, df_describe_3

def extract_df_describe_2labels(df, label1, label2):
    df_describe_1 = extract_df_label(df, label1).describe()
    df_describe_2 = extract_df_label(df, label2).describe()
    return df_describe_1, df_describe_2

def find_mean_median(df, features):
    mean_df = df.loc[['mean'], df.columns.to_series().str.contains(features)]
    median_df = df.loc[['50%'], df.columns.to_series().str.contains(features)]
    mean_Transpose = mean_df.transpose()
    median_Transpose = median_df.transpose()
    mean_median_Transpose = pd.merge(mean_Transpose, median_Transpose, left_index=True, right_index=True)
    mean_median_Transpose.columns = ['mean', 'median']
    return mean_median_Transpose

def plot_feature_label_3labels(features, label1, label2, label3, y_label, stat='mean'):
    mean_median_1 = find_mean_median(label1, features)
    mean_median_2 = find_mean_median(label2, features)
    mean_median_3 = find_mean_median(label3, features)
    ## plot mean for different labels
    if stat == 'mean':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_median_1.index, y=mean_median_1['mean'], name = 'label 1' , line=dict(color='firebrick', width=2)))
        fig.add_trace(go.Scatter(x=mean_median_2.index, y=mean_median_2['mean'], name = 'label 2' ,line=dict(color='royalblue', width=2)))
        fig.add_trace(go.Scatter(x=mean_median_3.index, y=mean_median_3['mean'], name = 'label 3' ,line=dict(color='green', width=2)))
        fig.update_layout(xaxis_title='Time', yaxis_title=y_label, title=y_label+' mean')
    ## plot median for different labels
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_median_1.index, y=mean_median_1['median'], name = 'label 1' ,line=dict(color='firebrick', width=2)))
        fig.add_trace(go.Scatter(x=mean_median_2.index, y=mean_median_2['median'], name = 'label 2' ,line=dict(color='royalblue', width=2)))
        fig.add_trace(go.Scatter(x=mean_median_3.index, y=mean_median_3['median'], name = 'label 3' ,line=dict(color='green', width=2)))
        fig.update_layout(xaxis_title='Time', yaxis_title=y_label, title=y_label+' median')
    
    return fig

## sampling function for each label
def sample_plot(df_label, df_describe, n, features, y_label):
    sample = df_label.sample(n)
    new_df = sample.loc[:, sample.columns.to_series().str.contains(features)]
    transpose  = new_df.transpose()
    transpose.columns = ['sample'+'_{}'.format(i) for i in range(len(transpose.columns))]
    ## mean median
    mean_df = df_describe.loc[['mean'], df_describe.columns.to_series().str.contains(features)]
    median_df = df_describe.loc[['50%'], df_describe.columns.to_series().str.contains(features)]
    mean_Transpose = mean_df.transpose()
    median_Transpose = median_df.transpose()
    ## merge the two dataframes
    mean_median_Transpose = pd.merge(mean_Transpose, median_Transpose, left_index=True, right_index=True)
    mean_median_Transpose.columns = ['mean', 'median']

    fig = go.Figure()
    for i in range(len(transpose.columns)):
        fig.add_trace(go.Scatter(x=transpose.index, y=transpose.iloc[:,i], name=transpose.columns[i], line = dict(color="#6a6a6a", width=2)))
    fig.add_trace(go.Scatter(x=mean_median_Transpose.index, y=mean_median_Transpose['mean'], name='mean', line=dict(color='firebrick', width=2)))
    fig.add_trace(go.Scatter(x=mean_median_Transpose.index, y=mean_median_Transpose['median'], name='median', line=dict(color='royalblue', width=2)))
    fig.update_layout(xaxis_title='Time', yaxis_title=y_label)
    return fig

## bar_plot for number of count for each label
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

def scatter_plot(df, title, label_name, type="umap"):
    if type == "umap":
        fig_scatter = px.scatter(df,
                            x=df['umap_0'],
                            y=df['umap_1'],
                            color=label_name,
                            title=title)
    elif type == "tsne":
        fig_scatter = px.scatter(df,
                            x=df['tsne_0'],
                            y=df['tsne_1'],
                            color=label_name,
                            title=title)
    return fig_scatter

def create_confusion_matrix(model1, model2, title):
    confusion_matrix = np.zeros( (len(np.unique(model1)), len(np.unique(model2))))
    print(np.unique(model2))
    for i_idx, i in enumerate(np.unique(model1)):
        for j_idx, j in enumerate(np.unique(model2)):
            confusion_matrix[i_idx, j_idx] = np.sum( (model1 == i) & (model2 == j) )
    # heatmap
    ax = plt.axes()
    heatmap(confusion_matrix, ax = ax)
    ax.set_title(title)
    ax.set_xlabel("Cluster 1")
    ax.set_ylabel("Cluster 2")
    plt.show()

def diff_standardization_layout_7features(method, df_ref):
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
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Continuous: Valid", href="/page-3"),
                dbc.DropdownMenuItem("Discrete: Valid", href="/page-4"),
            ],
            nav=True,
            in_navbar=True,
            label="UMAP Sampling",
        ),
        dbc.NavItem(dbc.NavLink('Experiments', href='/7vv')),
    ],
    brand="MIMIC Visualization",
    brand_href="/",
    color="primary",
    dark=True,),



    dbc.Container([

        html.Br(),

        dbc.Row([ 

            dbc.Row([html.H5(children = "UMAP remove outliers' plot with different standardization Method (7 features)", style={'text-align': 'center'}),]),
            html.P(children="7 features includes: Heart rate, Creatinine, Hematocrit, Fi02, Mean Blood Pressure, Respiratory Rate, Sodium", style={'text-align': 'center'}),
            

            html.H5(children='Select the label color', style={'textAlign': 'center'}),

            html.Br(),
            dbc.Col([
            dbc.Button(
                    "Discrete: Valid", id="example-button-10", className="me-2",  color="warning", href="/7vv",
                ),
            dbc.Button(
                    "Discrete: Normal", id="example-button-5", className="me-2",  color="warning", href="/7nv",
                ),
            dbc.Button(
                    "Continuous: Valid", id="example-button-10", className="me-2",  color="warning", href="/7av",
                ),
            dbc.Button(
                    "Continuous: Normal", id="example-button-20", className="me-2",  color="warning", href="/7an",
                ),
            ], width = 12, style = {'textAlign': 'center'}),

            html.Br(),

            html.H5(children=method, style={'textAlign': 'center'}),

        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap_valid_validated, 'Discrete: Valid', label_name=df_ref['labels'],  type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),

        
        dbc.Col([


            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap_normal_validated, 'Discreate: Normal', label_name=df_ref['labels'], type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),
    
        ]),

        dbc.Row([ 

        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap_rm_df, 'Continuous: Valid', label_name=df_ref['labels'], type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),

        
        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(figure = scatter_plot(umap_against_normal, 'Continuous: Normal', label_name=
            df_ref['labels'],  type="umap"))])]),
        
        ], width= 6, sm = 12, md = 6, lg = 6, xl = 6),
    
        ]),

        
    ]),
    ])

    return layout

def layout_for_one_sampling(df, df_describe, width):
            
    sample_layout = dbc.Col([
            
            dbc.Row([

                html.H5('Label 0', style={'textAlign': 'center'}),

                html.Div([dcc.Graph(figure = sample_plot(df, df_describe, 5, "heart_rate", "heart_rate"))], 
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
               html.Div([dcc.Graph(figure = sample_plot(df, df_describe, 5, "respiratory_rate", "respiratory_rate"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df, df_describe, 5, "hematocrit", "hematocrit"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df, df_describe, 5, "fraction_inspired_oxygen", "fraction_inspired_oxygen"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df, df_describe, 5, "creatinine", "creatinine"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
               html.Div([dcc.Graph(figure = sample_plot(df, df_describe, 5, "mean_blood_pressure", "mean_blood_pressure"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = sample_plot(df, df_describe, 5, "sodium", "sodium"))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),
        ],width=width, sm=width, md=width, lg=width),
    
    return sample_layout


"df1, df2, df3=None"

def sampling_plot(df1, df2, df3=None):


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


    html.Br(),
    
    dbc.Row([html.H1(children='UMAP sample analysis (5 samples)', style={'textAlign': 'center'})], style={'padding': '10px'}),

    dbc.Row([html.H3(children='**Double click the legend to isolate individual plots**', style={'textAlign': 'center'})], style={'padding': '10px'}),

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


    return layout
