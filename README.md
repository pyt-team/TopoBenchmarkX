# ICML Topological Deep Learning Challenge 2024: Beyond the Graph Domain
Official repository for the Topological Deep Learning Challenge 2024, jointly organized by [TAG-DS](https://www.tagds.com) & PyT-Team and hosted by the [Geometry-grounded Representation Learning and Generative Modeling (GRaM) Workshop](https://gram-workshop.github.io) at ICML 2024.

## Relevant Information
- The deadline is **July 12th, 2024 (Anywhere on Earth)**. Participants are welcome to modify their submissions until this time.
- Please, check out the [main webpage of the challenge](https://pyt-team.github.io/packs/challenge.html) for the full description of the competition (motivation, submission requirements, evaluation, etc.)

## Brief Description
The main purpose of the challenge is to further expand the current scope and impact of Topological Deep Learning (TDL), enabling the exploration of its applicability in new contexts and scenarios. To do so, we propose participants to design and implement lifting mappings between different data structures and topological domains (point-clouds, graphs, hypergraphs, simplicial/cell/combinatorial complexes), potentially bridging the gap between TDL and all kinds of existing datasets.


## General Guidelines
Everyone can participate and participation is free --only principal PyT-Team developers are excluded. It is sufficient to:
- Send a valid Pull Request (i.e. passing all tests) before the deadline.
- Respect Submission Requirements (see below).

Teams are accepted, and there is no restriction on the number of team members. An acceptable Pull Request automatically subscribes a participant/team to the challenge.

We encourage participants to start submitting their Pull Request early on, as this helps addressing potential issues with the code. Moreover, earlier Pull Requests will be given priority consideration in the case of multiple submissions of similar quality implementing the same lifting.

A Pull Request should contain no more than one lifting. However, there is no restriction on the number of submissions (Pull Requests) per participant/team.

## Basic Setup
To develop on your machine, here are some tips.

First, we recommend using Python 3.11.3, which is the python version used to run the unit-tests.

For example, create a conda environment:
   ```bash
   conda create -n topox python=3.11.3
   conda activate topox
   ```

Then:

1. Clone a copy of tmx from source:

   ```bash
   git clone git@github.com:pyt-team/challenge-icml-2024.git
   cd challenge-icml-2024
   ```

2. Install tmx in editable mode:

   ```bash
   pip install -e '.[all]'
   ```
   **Notes:**
   - Requires pip >= 21.3. Refer: [PEP 660](https://peps.python.org/pep-0660/).
   - On Windows, use `pip install -e .[all]` instead (without quotes around `[all]`).

4. Install torch, torch-scatter, torch-sparse with or without CUDA depending on your needs.

      ```bash
      pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
      pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
      pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
      ```

      where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation (`torch.version.cuda`).

5. Ensure that you have a working tmx installation by running the entire test suite with

   ```bash
   pytest
   ```

    In case an error occurs, please first check if all sub-packages ([`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse), [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and [`torch-spline-conv`](https://github.com/rusty1s/pytorch_spline_conv)) are on its latest reported version.

6. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Questions

Feel free to contact us through GitHub issues on this repository, or through the [Geometry and Topology in Machine Learning slack](https://tda-in-ml.slack.com/join/shared_invite/enQtOTIyMTIyNTYxMTM2LTA2YmQyZjVjNjgxZWYzMDUyODY5MjlhMGE3ZTI1MzE4NjI2OTY0MmUyMmQ3NGE0MTNmMzNiMTViMjM2MzE4OTc#/). Alternatively, you can contact us via mail at any of these accounts: guillermo.bernardez@upc.edu, lev.telyatnikov@uniroma1.it.
