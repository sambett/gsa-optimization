{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GSA Basic Usage Example"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from src.gsa import gravitational_search\n",
    "\n",
    "# Run basic optimization\n",
    "best_position, best_fitness, history = gravitational_search(\n",
    "    n_particles=30,\n",
    "    n_dimensions=30,\n",
    "    fitness_function='sphere'\n",
    ")\n",
    "\n",
    "print(f\"Best fitness found: {best_fitness:.6f}\")"
   ]
  }
 ]
}
