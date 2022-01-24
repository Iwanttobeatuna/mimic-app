import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np

# icustay_id
app = dash.Dash(__name__)
server = app.server

## umap csv
df = pd.read_csv('umap_appended.csv')

## tsne csv
df_2 = pd.read_csv('tsne_appended.csv')

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

## Bootstrap template
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP]) 

app.layout = dbc.Container([

    ## UMAP
    html.Br(),
    
    dbc.Row([html.H1(children='UMAP', style={'textAlign': 'center'})], style={'padding': '20px'}),

    html.Br(),

    dbc.Row
    ([
        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(
                id='crossfilter-indicator-scatter',
                figure = fig_2d,
                hoverData={'points': [{'customdata': [200001]}]}
            )])]),

            dbc.Row([

                dbc.Col([

                    html.H5(children='Mean Blood Pressure', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'mbp-time-series')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ])
            
            ]),
            
            ], width=5, sm=12, md=12, lg=5),
        
        
        dbc.Col([
            
            dbc.Row([

                dbc.Col([

                    html.H5(children='Heart Rate', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'heart-rate-time-series')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),

                dbc.Col([

                    html.H5(children='Respiratory Rate', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'rr-time-series')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),

            ]),

            dbc.Row([

                dbc.Col([

                    html.H5(children='Fraction Inspired Oxygen', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'fio2-time-series')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),

                dbc.Col([

                    html.H5(children='Creatinine', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'creatinine-time-series')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),

            ]),

            dbc.Row([

                dbc.Col([

                    html.H5(children='Sodium', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 's-time-series')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),

                dbc.Col([

                    html.H5(children='Hematocrit', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'hematocrit-time-series')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),
            ]),
            
        ],width=7, sm=12, md=12, lg=7)
    
    ]),

    ## TSNE

    html.Hr(),

    html.Br(),
    
    dbc.Row([html.H1(children='TSNE', style={'textAlign': 'center'})], style={'padding': '20px'}),

    html.Br(),

    dbc.Row
    ([
        dbc.Col([

            dbc.Row([
            html.Div([
            dcc.Graph(
                id='crossfilter-tsne-indicator',
                figure = fig_tsne,
                hoverData={'points': [{'customdata': [200001]}]}
            )])]),

            dbc.Row([

                dbc.Col([

                    html.H5(children='Mean Blood Pressure', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'mbp-tsne-ts')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ])
            
            ]),
            
            ], width=5, sm=12, md=12, lg=5),
        
        
        dbc.Col([
            
            dbc.Row([

                dbc.Col([

                    html.H5(children='Heart Rate', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'hr-tsne-ts')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),

                dbc.Col([

                    html.H5(children='Respiratory Rate', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'rr-tsne-ts')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),

            ]),

            dbc.Row([

                dbc.Col([

                    html.H5(children='Fraction Inspired Oxygen', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'fio2-tsne-ts')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),

                dbc.Col([

                    html.H5(children='Creatinine', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'creatinine-tsne-ts')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),

            ]),

            dbc.Row([

                dbc.Col([

                    html.H5(children='Sodium', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 's-tsne-ts')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),

                dbc.Col([

                    html.H5(children='Hematocrit', style={'textAlign': 'center'}),

                    html.Div([dcc.Graph(id = 'hematocrit-tsne-ts')], 
                    style={'width': '100%', 'display': 'inline-block'})
                ]),
            ]),
            
        ],width=7, sm=12, md=12, lg=7)
    
    ])
])



## UMAP callbacks
@app.callback(
    dash.dependencies.Output('heart-rate-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_heart_rate_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df.loc[df['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("heart_rate")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['heart_rate', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['heart_rate'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[0,200])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Heart rate against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    return fig

@app.callback(
    dash.dependencies.Output('rr-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_r_rate_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df.loc[df['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("respiratory_rate")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['respiratory_rate', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['respiratory_rate'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[0,50])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Respiratory rate against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    return fig

@app.callback(
    dash.dependencies.Output('s-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_sodium_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df.loc[df['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("sodium")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['sodium', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['sodium'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[100,180])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Sodium against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 20})
    return fig

@app.callback(
    dash.dependencies.Output('creatinine-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_sodium_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df.loc[df['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("creatinine")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['creatinine', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['creatinine'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[0,20])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Creatinine against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 20})
    return fig

@app.callback(
    dash.dependencies.Output('fio2-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_sodium_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df.loc[df['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("fraction_inspired_oxygen")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['fio2', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['fio2'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[0,1.5])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'FiO2 against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 20})
    return fig

@app.callback(
    dash.dependencies.Output('hematocrit-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_sodium_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df.loc[df['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("hematocrit")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['hematocrit', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['hematocrit'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[10,50])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Hematocrit against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 20})
    return fig

@app.callback(
    dash.dependencies.Output('mbp-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_sodium_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df.loc[df['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("mean_blood_pressure")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['mbp', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['mbp'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[0,180])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Mean blood pressure against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 20})
    return fig

## TSNE callbacks

@app.callback(
    dash.dependencies.Output('hr-tsne-ts', 'figure'),
    [dash.dependencies.Input('crossfilter-tsne-indicator', 'hoverData')])
def update_heart_rate_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df_2.loc[df_2['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("heart_rate")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['heart_rate', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['heart_rate'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[0,200])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Heart rate against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    return fig

@app.callback(
    dash.dependencies.Output('rr-tsne-ts', 'figure'),
    [dash.dependencies.Input('crossfilter-tsne-indicator', 'hoverData')])
def update_r_rate_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df_2.loc[df_2['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("respiratory_rate")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['respiratory_rate', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['respiratory_rate'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[0,50])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Respiratory rate against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    return fig

@app.callback(
    dash.dependencies.Output('s-tsne-ts', 'figure'),
    [dash.dependencies.Input('crossfilter-tsne-indicator', 'hoverData')])
def update_sodium_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df_2.loc[df_2['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("sodium")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['sodium', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['sodium'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[100,180])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Sodium against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 20})
    return fig

@app.callback(
    dash.dependencies.Output('creatinine-tsne-ts', 'figure'),
    [dash.dependencies.Input('crossfilter-tsne-indicator', 'hoverData')])
def update_sodium_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df_2.loc[df_2['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("creatinine")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['creatinine', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['creatinine'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[0,20])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Creatinine against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 20})
    return fig

@app.callback(
    dash.dependencies.Output('fio2-tsne-ts', 'figure'),
    [dash.dependencies.Input('crossfilter-tsne-indicator', 'hoverData')])
def update_sodium_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df_2.loc[df_2['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("fraction_inspired_oxygen")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['fio2', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['fio2'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[0,1.5])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'FiO2 against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 20})
    return fig

@app.callback(
    dash.dependencies.Output('hematocrit-tsne-ts', 'figure'),
    [dash.dependencies.Input('crossfilter-tsne-indicator', 'hoverData')])
def update_sodium_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df_2.loc[df_2['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("hematocrit")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['hematocrit', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['hematocrit'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[10,50])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Hematocrit against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 20})
    return fig

@app.callback(
    dash.dependencies.Output('mbp-tsne-ts', 'figure'),
    [dash.dependencies.Input('crossfilter-tsne-indicator', 'hoverData')])
def update_sodium_timeseries(hoverData):
    id = hoverData['points'][0]['customdata']
    dff = df_2.loc[df_2['icustay_id'].apply(lambda x: [x] == id)]
    dff = dff.loc[:, dff.columns.to_series().str.contains("mean_blood_pressure")]
    dff_transpose = dff.transpose()
    dff_transpose['time'] = range(0, len(dff_transpose))
    dff_transpose.columns = ['mbp', 'time']
    fig = px.line(dff_transpose, x = dff_transpose['time'], y = dff_transpose['mbp'])
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(yaxis_range=[0,180])
    title = 'Patient id: <b>{}</b><br>{}'.format(id, 'Mean blood pressure against time(in hours)')
    fig.add_annotation(x=0, y=0.89, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=230, margin={'l': 20, 'b': 30, 'r': 10, 't': 20})
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)