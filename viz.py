import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

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

def scatter_plot(df, title, type="umap"):
    if type == "umap":
        fig_scatter = px.scatter(df,
                            x=df['umap_0'],
                            y=df['umap_1'],
                            color=df['labels'],
                            title=title)
    elif type == "tsne":
        fig_scatter = px.scatter(df,
                            x=df['tsne_0'],
                            y=df['tsne_1'],
                            color=df['labels'],
                            title=title)
    return fig_scatter

