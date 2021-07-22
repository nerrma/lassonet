# Sina's Fork of LassoNet

## Installation

To install:
1. Clone this repo `git clone https://www.github.com/nerrma/lassonet`
2. Pack and install it `tar -czf lassonet.tar.gz lassonet/ && pip install lassonet.tar.gz`
3. Use it by importing the interfaces `from lassonet import LassoNetClassifier, LassoNetRegressor`

LassoNet is based on the work presented in [this paper](https://arxiv.org/abs/1907.12207) ([bibtex here for citation](https://github.com/lasso-net/lassonet/blob/master/citation.bib)).
Here is a [link](https://www.youtube.com/watch?v=bbqpUfxA_OA) to the promo video:

<a href="https://www.youtube.com/watch?v=bbqpUfxA_OA" target="_blank"><img src="https://raw.githubusercontent.com/lasso-net/lassonet/master/docs/images/video_screenshot.png" width="450" alt="Promo Video"/></a>

### Code

We have designed the code to follow scikit-learn's standards to the extent possible (e.g. [linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)).

Our plan is to add more functionality that help users understand the important features in neural networks.

### Website

LassoNet's website is [https://lassonet.ml](https://lassonet.ml). It contains many useful references including the paper, live talks and additional documentation.
