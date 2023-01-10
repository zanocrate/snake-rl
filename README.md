# snake-rl

A Snake custom `OpenAI` environment, and a framework to train Deep Q Learning agents to play it using PyTorch.

## Usage

Define your `DQN` network class as a PyTorch `nn.Module` subclass, edit the `train.py` file to use it and define the number of episodes for training. Then run

```shell
python train.py
```