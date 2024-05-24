# CNRS Formation Wandercraft 2024

[Chat room](https://matrix.to/#/#wan24:laas.fr)


## Installation procedure

### Requirements

For this tutorial session, we are going to need:
- Crocoddyl v2 (2.0.2)
- Pinocchio
- For some chapters: Casadi with the Pinocchio bindings. Pinocchio v2.99 is then recommanded.

### With Conda (prefered)

We are going to use the Conda channel prepared for the [Agimus Winter School](https://github.com/agimus-project/winter-school-2023).
All the required packages are available on the [AWS channel](https://anaconda.org/agm-ws-2023/repo).
Conda is to be installed on your machine by following these [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

You can then install a package by typing in your terminal:
```bash
conda create -n wan24 python=3.10
conda activate wan24
conda install -c agm-ws-2023 -c conda-forge mim-solvers
```
The two first lines create a new environment named `wan24` and then activate it.
The third line installs `mim-solvers` using the [AWS channel](https://anaconda.org/agm-ws-2023) where the packages have been compiled (MiM-solvers is the most down-stream package, which then drains Pinocchio, Crocoddyl, etc).

You then need to install the following additional tools via pip:
```bash
pip install tqdm meshcat ipython jupyterlab ipywidgets matplotlib gepetuto`
```

### With PyPI

You can install with PyPI but you then need two environments:
- the first one with the Pinocchio Casadi module, specifically compiled and frozen to a 2023 version
- the second one with the mainstream Crocoddyl and Pinocchio.

This can be done using two virtual environments.

#### Setup the Pinocchio/Casadi environment

1. create an environment:

    `python -m venv .venv-pincasadi`

2. activate it:

    `source .venv-pincasadi/bin/activate`

3. update pip:

    `pip install -U pip`

    4. install dependencies:

    `pip install example-robot-data-jnrh2023 jupyterlab meshcat ipywidgets matplotlib gepetuto`

5. when done with the virtual environment, deactivate:

   `deactivate`


#### Setup the Crocoddyl environment

1. create an environment:

    `python -m venv .venv-croc`

2. activate it:

    `source .venv-croc/bin/activate`

3. update pip:

    `pip install -U pip`

4. install dependencies:

    `pip install cmeel-mim-solvers jupyterlab meshcat ipywidgets matplotlib gepetuto`

5. when done with the virtual environment, deactivate:

   `deactivate`

### Start Jupyter Lab

The tutorials are all in Jupyter Lab. Start them with `PYTHONPATH=. jupyter-lab`


## Tutorial list


