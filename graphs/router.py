from plotly.io import to_html
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from ml import (
    create_gradient_plots,
    create_regression_plots,
    create_etr_plots,
    create_bayesian_optimization_plots,
    create_bayesian_ridge_plots
)

graphs_router = APIRouter(prefix="/graphs", tags=["graphs"])

@graphs_router.get("/regression", response_class=HTMLResponse)
async def plot_regressions_graphs():
    fig = create_regression_plots()
    html_data = to_html(fig, full_html=True)
    return HTMLResponse(content=html_data)

@graphs_router.get("/gradient", response_class=HTMLResponse)
async def plot_gradient_graphs():
    fig = create_gradient_plots()
    html_data = to_html(fig, full_html=True)
    return HTMLResponse(content=html_data)

@graphs_router.get("/extra_trees_regressor", response_class=HTMLResponse)
async def plot_etr_graphs():
    fig = create_etr_plots()
    html_data = to_html(fig, full_html=True)
    return HTMLResponse(content=html_data)

@graphs_router.get("/polynomial_bayes", response_class=HTMLResponse)
async def plot_bayesian_optimization_graphs():
    fig = create_bayesian_optimization_plots()
    html_data = to_html(fig, full_html=True)
    return HTMLResponse(content=html_data)

@graphs_router.get("/bayesian_ridge", response_class=HTMLResponse)
async def plot_bayesian_ridge_graphs():
    fig = create_bayesian_ridge_plots()
    html_data = to_html(fig, full_html=True)
    return HTMLResponse(content=html_data)