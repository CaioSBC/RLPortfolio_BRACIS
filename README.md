# RLPortfolio: Reinforcement Learning for Financial Portfolio Optimization

This anonymous repository contains the code used in the experiments run in the article *RLPortfolio: Reinforcement Learning for Financial Portfolio Optimization*, which introduces RLPortfolio, a python library that provides the tools necessary to develop, train and test a portfolio optimization agent with reinforcement learning.

### Understanding the repository

This repository is made of two folders:
- **experiment:** contains the small experiment that was conducted in the writing of the article. It is divided as follows:
    - **graphs:** a folder with all the plots created in the experiment.
    - **runs:** a folder with a tensorboard log.
    - **BR_stocks.ipynb:** jupyter notebook with the main code of the experiment.
    - **policy_BR.pt:** saved pytorch module of the trained policy.
- **lib:** contains the library that is being introduced in this article.

### Running the experiment

To run the experiment again, it is necessary to install jupyter and RLportfolio through the following commands:

```bash
pip install jupyter
pip install lib/.
```

Finally, you can run the following command to open the experiment code:

```bash
jupyter notebook experiment/BR_stocks.ipynb
```

It is important to note that, due to random generators in the code, different runs might achieve considerably different results. 