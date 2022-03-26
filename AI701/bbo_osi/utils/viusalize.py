from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import os
from bayesmark.constants import (
    ITER,
    LB_MEAN,
    LB_MED,
    LB_NORMED_MEAN,
    METHOD,
    NORMED_MEAN,
    NORMED_MED,
    OBJECTIVE,
    PERF_BEST,
    PERF_CLIP,
    PERF_MEAN,
    PERF_MED,
    SUGGEST,
    TEST_CASE,
    TRIAL,
    UB_MEAN,
    UB_MED,
    UB_NORMED_MEAN,
)

import logging
from collections import Counter

import numpy as np
import xarray as xr

import bayesmark.constants as cc
import bayesmark.xr_util as xru
from bayesmark.cmd_parse import CmdArgs, agg_parser, parse_args, serializable_dict, unserializable_dict
from bayesmark.constants import ARG_DELIM, EVAL_RESULTS, ITER, METHOD, SUGGEST, TEST_CASE, TIME_RESULTS, TRIAL
from bayesmark.serialize import XRSerializer
from bayesmark.signatures import analyze_signatures
from bayesmark.sklearn_funcs import SklearnModel
from bayesmark.util import str_join_safe

logger = logging.getLogger(__name__)


def summary_plot(DB_ROOT, DBID, objective='_visible_to_opt'):
   
    agg_result, final_score = read_data(DB_ROOT, DBID, objective=objective)
    fig = make_subplots(rows=1, cols=3)

    for c, k in enumerate(["dataset", "metric", "model"]):
        plot_data = agg_result.groupby(k)["mean normed"].mean()
        x = plot_data.index
        y = plot_data
        fig.add_trace(
            go.Bar(x=x, y=y, name=k),
            row=1, col=c+1
        )
    fig.update_layout(height=300, width=800, title_text="Grouped Mean")
    fig.show()
    
def bar_plot(DB_ROOT, DBID, colorby=None,objective='_visible_to_opt'):
    agg_result, final_score = read_data(DB_ROOT, DBID, objective=objective)
    plot_data = agg_result[agg_result.objective==objective].groupby("function")["mean normed"].mean()
    plot_data = plot_data.reset_index()
    x = plot_data.function
    y = plot_data["mean normed"]
    plot_data["model"] = pd.Series(x.apply(lambda x : x.split("_")[0]))
    plot_data["dataset"] = x.apply(lambda x : x.split("_")[1])
    plot_data["metric"] = x.apply(lambda x : x.split("_")[2])
    fig = px.bar(plot_data, x="function", y="mean normed" ,color=colorby)
    fig.show()

def read_data(DB_ROOT, DBID, objective='_visible_to_opt'):
    agg_result, meta = XRSerializer.load_derived(DB_ROOT, DBID, key=cc.PERF_RESULTS)
    summary, meta = XRSerializer.load_derived(DB_ROOT, DBID, key=cc.MEAN_SCORE)
    agg_result = agg_result.to_dataframe()
    agg_result.reset_index(inplace=True)
    agg_result = agg_result[agg_result.objective==objective]
    agg_result["model"] = agg_result.function.apply(lambda x : x.split("_")[0])
    agg_result["dataset"] = agg_result.function.apply(lambda x : x.split("_")[1])
    agg_result["metric"] = agg_result.function.apply(lambda x : x.split("_")[2])
    final_score = summary[PERF_MEAN].sel({cc.OBJECTIVE: objective}, drop=True)[{ITER: -1}]

    return agg_result, final_score