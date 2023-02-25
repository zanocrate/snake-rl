# snake-rl

A Snake custom `OpenAI` environment, and a framework to train Deep Q Learning agents to play it using PyTorch.

## Usage

Define your `DQN` network class as a PyTorch `nn.Module` subclass in the `src/model.py` module, customize your training and environment configurations in `config.json`, and run

```shell
python train.py
```