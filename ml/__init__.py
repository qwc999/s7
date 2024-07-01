from .extra_trees_reg import create_etr_plots
from .gradient import create_gradient_plots
from .regression import create_regression_plots
from .bayesian_ridge import bayesian_load_or_train_and_plot


__all__ = (
    'create_etr_plots',
    'create_gradient_plots',
    'create_regression_plots',
    'bayesian_load_or_train_and_plot'
)