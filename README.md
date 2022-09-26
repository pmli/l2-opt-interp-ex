# Code for Numerical Experiments in "A Unifying Framework for Interpolatory $\mathcal{L}_2$-optimal Reduced-order Modeling"

This repository contains code for numerical experiments reported in

> P. Mlinarić, S. Gugercin,
> **A Unifying Framework for Interpolatory $\mathcal{L}_2$-optimal Reduced-order
> Modeling**,
> [*arXiv preprint*](https://arxiv.org/abs/2209.00714),
> 2022

## Installation

To run the examples, at least Python 3.8 is needed
(the code was tested using Python 3.8.12).

The necessary packages are listed in [`requirements.txt`](requirements.txt).
They can be installed in a virtual environment by, e.g.,

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Running the Experiments

The experiments are given as `runme_*.py` scripts.
They can be opened as Jupyter notebooks via
[`jupytext`](https://jupytext.readthedocs.io/en/latest/)
(included when installing via [`requirements.txt`](requirements.txt)).

## Author

Petar Mlinarić:

- affiliation: Virginia Tech
- email: mlinaric@vt.edu
- ORCiD: 0000-0002-9437-7698

## License

The code is published under the MIT license.
See [LICENSE](LICENSE).
