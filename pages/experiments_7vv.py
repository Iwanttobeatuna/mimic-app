import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from viz import diff_standardization_layout_7features

umap_rm_df = pd.read_csv('umap(rm)_appended.csv')

umap_normal_validated = pd.read_csv('umap(rm)_normal_validated.csv')

umap_valid_validated = pd.read_csv('umap(rm)_valid_validated.csv')

umap_against_normal = pd.read_csv('umap(rm)validate_against_normal.csv')

methods = ['Valid Validated', 'Normal Validated', 'Against Normal', 'Against Valid']

layout = diff_standardization_layout_7features(methods[0], umap_valid_validated)
