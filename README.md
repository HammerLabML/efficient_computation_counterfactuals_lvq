# Efficient computation of counterfactual explanations of LVQ models

This repository contains the implementation of the methods proposed in the paper [Efficient computation of counterfactual explanations of LVQ models](paper.pdf) by André Artelt and Barbara Hammer.

The methods are implemented in `counterfactuals_lvq.py`. A minimalistic usage example is given in [test.py](test.py).

The default solver is [SCS](https://github.com/cvxgrp/scs). If you want to use a different solver, you have to overwrite the `_solve` method.

## Requirements

- Python3.6
- Packages as listed in `requirements.txt`

## License

MIT license - See [LICENSE.md](LICENSE.md)

## How to cite

You can cite the version on [arXiv](https://arxiv.org/abs/1908.00735).

## Conference paper

A short version of this paper has been accepted as a conference paper at ESANN-2020.