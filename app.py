import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pages import tsne, umap, umap_sample, umap_sample10, umap_sample20

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

fig_compare = px.scatter(
    df, x='umap_0', y='umap_1',
    color=df_2['labels'],
    hover_data=['icustay_id'],
    title = "TsNE labels on UMAP plots"
)
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
        fig.add_trace(go.Scatter(x=transpose.index, y=transpose.iloc[:,i], name=transpose.columns[i], line = dict(width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=mean_median_Transpose.index, y=mean_median_Transpose['mean'], name='mean', line=dict(color='firebrick', width=2)))
    fig.add_trace(go.Scatter(x=mean_median_Transpose.index, y=mean_median_Transpose['median'], name='median', line=dict(color='royalblue', width=2)))
    fig.update_layout(xaxis_title='Time', yaxis_title=y_label)
    fig.show()
    
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
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("UMAP analysis", href="/page-1")),
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

    dbc.Row([
                html.H5(children = "Updates", style={'text-align': 'center'}),

                html.P(children="(14 Feb: Updated UMAP and TSNE plots)")
    ]),

    ]),
])


# Update the index
@callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return umap.layout
    elif pathname == '/page-2':
        return tsne.layout
    elif pathname == '/page-3':
        return umap_sample.layout
    elif pathname == '/':
        return index_page
    elif pathname == "/umap-sample-10":
        return umap_sample10.layout
    elif pathname == "/umap-sample-5":
        return umap_sample.layout
    elif pathname == '/umap-sample-20':
        return umap_sample20.layout
    else:
        return "404"
    # You could also return a 404 "URL not found" page here

if __name__ == '__main__':
    app.run_server(debug=True)

