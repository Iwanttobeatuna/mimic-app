import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

## umap csv
df = pd.read_csv('umap_appended.csv')

## tsne csv
df_2 = pd.read_csv('tsne_appended.csv')

## Split into dataframes based on labels (umap)
df_umap_1 = df.loc[df['labels'] == 0]
df_umap_2 = df.loc[df['labels'] == 1]
df_umap_3 = df.loc[df['labels'] == 2]
df_umap_1_describe = df_umap_1.describe()
df_umap_2_describe = df_umap_2.describe()
df_umap_3_describe = df_umap_3.describe()

## Split into dataframe based on labels (tsne)
df_tsne_1 = df_2.loc[df_2['labels'] == 0]
df_tsne_2 = df_2.loc[df_2['labels'] == 1]
df_tsne_3 = df_2.loc[df_2['labels'] == 2]
df_tsne_1_describe = df_tsne_1.describe()
df_tsne_2_describe = df_tsne_2.describe()
df_tsne_3_describe = df_tsne_3.describe()


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

def find_mean_median(df, features):
    mean_df = df.loc[['mean'], df.columns.to_series().str.contains(features)]
    median_df = df.loc[['50%'], df.columns.to_series().str.contains(features)]
    mean_Transpose = mean_df.transpose()
    median_Transpose = median_df.transpose()
    mean_median_Transpose = pd.merge(mean_Transpose, median_Transpose, left_index=True, right_index=True)
    mean_median_Transpose.columns = ['mean', 'median']
    return mean_median_Transpose

def plot_feature_label(features, label1, label2, label3, y_label, stat='mean'):
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

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP]) 


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("UMAP analysis", href="/page-1")),
        dbc.NavItem(dbc.NavLink("TSNE analysis", href="/page-2")),
    ],
    brand="MIMIC Visualization",
    brand_href="/",
    color="primary",
    dark=True,),
])

page_1_layout = html.Div([

    dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("UMAP analysis", href="/page-1")),
        dbc.NavItem(dbc.NavLink("TSNE analysis", href="/page-2")),
    ],
    brand="MIMIC Visualization",
    brand_href="/",
    color="primary",
    dark=True,),

    html.Br(),
    
    dbc.Row([html.H1(children='UMAP label analysis', style={'textAlign': 'center'})], style={'padding': '10px'}),

      dbc.Row
    ([
         dbc.Col([
            
            dbc.Row([

                html.H5('Mean', style={'textAlign': 'center'}),

                html.Div([dcc.Graph(figure = plot_feature_label('heart_rate', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'heart_rate', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('respiratory_rate', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'respiratory rate', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('hematocrit', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'hematocrit', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('fraction_inspired_oxygen', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'fraction inspired oxygen', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('creatinine', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'creatinine', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('mean_blood_pressure', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'mean blood pressure', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('sodium', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'sodium', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),
        ],width=6, sm=6, md=6, lg=6),
        
        
        dbc.Col([
            
            dbc.Row([

                html.H5('Median', style={'textAlign': 'center'}),

                html.Div([dcc.Graph(figure = plot_feature_label('heart_rate', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'heart_rate', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('respiratory_rate', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'respiratory rate', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('hematocrit', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'hematocrit', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('fraction_inspired_oxygen', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'fraction inspired oxygen', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('creatinine', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'creatinine', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('mean_blood_pressure', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'mean blood pressure', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('sodium', df_umap_1_describe, df_umap_2_describe, df_umap_3_describe, 'sodium', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),
        ],width=6, sm=6, md=6, lg=6),
    ]),

])



page_2_layout = html.Div([

    dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("UMAP analysis", href="/page-1")),
        dbc.NavItem(dbc.NavLink("TSNE analysis", href="/page-2")),
    ],
    brand="MIMIC Visualization",
    brand_href="/",
    color="primary",
    dark=True,),

    html.Br(),
    
    dbc.Row([html.H1(children='TSNE label analysis', style={'textAlign': 'center'})], style={'padding': '10px'}),

      dbc.Row
    ([
         dbc.Col([
            
            dbc.Row([

                html.H5('Mean', style={'textAlign': 'center'}),

                html.Div([dcc.Graph(figure = plot_feature_label('heart_rate', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'heart_rate', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('respiratory_rate', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'respiratory rate', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('hematocrit', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'hematocrit', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('fraction_inspired_oxygen', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'fraction inspired oxygen', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('creatinine', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'creatinine', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('mean_blood_pressure', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'mean blood pressure', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('sodium', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'sodium', 'mean'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),
        ],width=6, sm=6, md=6, lg=6),
        
        
        dbc.Col([
            
            dbc.Row([

                html.H5('Median', style={'textAlign': 'center'}),

                html.Div([dcc.Graph(figure = plot_feature_label('heart_rate', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'heart_rate', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('respiratory_rate', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'respiratory rate', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('hematocrit', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'hematocrit', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('fraction_inspired_oxygen', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'fraction inspired oxygen', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('creatinine', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'creatinine', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('mean_blood_pressure', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'mean blood pressure', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),

            dbc.Row([

                
                html.Div([dcc.Graph(figure = plot_feature_label('sodium', df_tsne_1_describe, df_tsne_2_describe, df_tsne_3_describe, 'sodium', 'median'))],
                style={'width': '100%', 'display': 'inline-block'})
            ]),
        ],width=6, sm=6, md=6, lg=6),
    ]),

])



# Update the index
@callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

if __name__ == '__main__':
    app.run_server(debug=True)

