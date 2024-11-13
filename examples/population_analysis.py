{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Population Size Analysis for GSA\n",
    "\n",
    "This notebook analyzes the impact of different population sizes on GSA performance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('../')\n",
    "\n",
    "from src.visualizations import run_population_analysis, plot_convergence"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Running Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test different population sizes\n",
    "population_sizes = [10, 30, 40, 80, 100]\n",
    "results = run_population_analysis(\n",
    "    population_sizes=population_sizes,\n",
    "    n_trials=5,\n",
    "    fitness_function='rosenbrock'\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualizing Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot convergence for different population sizes\n",
    "fig = plot_convergence(\n",
    "    results,\n",
    "    title='GSA Convergence Analysis - Rosenbrock Function'\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysis Summary\n",
    "\n",
    "The plot shows:\n",
    "1. Convergence behavior for different population sizes\n",
    "2. Standard deviation bands showing consistency of results\n",
    "3. Trade-off between population size and optimization performance"
   ]
  }
 ]
}
