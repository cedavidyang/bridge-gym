# RL Environment for Bridge Management

This repository is associated with the following paper:

> **Citation**
>
> Moayyedi, S.A., Yang, D.Y., 2026. Interpretable Deep Reinforcement Learning for Element-level Bridge Life-cycle Optimization. [arXiv:2604.02528](https://arxiv.org/abs/2604.02528).

## Overview

The package is built on `gymnasium` and its core `Env` base class. It can therefore be directly converted to a `torchrl` environment for RL training. A working demostration is provided in `debug_example_nbe107.py` file.

Two environments related to bridge life-cycle management are provided in this package:

* `example_nbe107`: An element-level environment where the system state is the condition state (CS) vector of a single NBE 107 element. The cost data reflect per-element costs. __This is the environment used in the cited paper__.

* `bridge_nbe107`: Explicitly models a bridge with a defined number of NBE 107 elements and a specific initial state. Since costs are defined per element, it functions similarly to `example_nbe107`. However, the condition state of each element is randomly generated per element based on the transition matrix and an element's current CS. Because the number of elements is finite, the observed system state incorporates sampling uncertainty revolving around this theoretical CS distribution.

## Replicating Study Results

The `custom_example_nbe107.py` file allows you to validate the interpretable life-cycle policies detailed in the cited paper. You can also use this file to experiment with and test your own custom life-cycle policies. 

To do so, simply modify the `action_policy` function within the script as needed.
