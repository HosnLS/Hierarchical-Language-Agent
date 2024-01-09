# Hierarchical Language Agent

This is a repository of the paper [LLM-Powered Hierarchical Language Agent for Real-time Human-AI Coordination](https://arxiv.org/abs/2312.15224).

More demonstrations can be seen on the [Project Website](https://sites.google.com/view/overcooked-hla/).

## 1. Install

### 1.1 LLM-API

It is recommended to install the `LLM-API` on a server. And use ssh tunneling to access the server from your local machine.

```bash
# create conda environment
conda create -n llm-api python=3.10
conda activate llm-api

# install dependencies
cd llm-api
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# download model
python download-model.py TheBloke/Llama-2-13B-chat-GPTQ --branch gptq-4bit-32g-actorder_True

# start api on port 5000
python server.py --api --api-blocking-port 5000 --api-streaming-port 5100 --model TheBloke_Llama-2-13B-chat-GPTQ_gptq-4bit-32g-actorder_True --loader exllama_HF --gpu-split 40,0,0,0 --max_seq_len 4096 --alpha_value 2

```

### 1.2 Environment and Agent

The environment and agent can be installed on local machine.

```bash
# create conda environment
conda create -n env-agent python=3.10
conda activate env-agent

# install dependencies: env
cd testbed-cooking
pip install -e .
cd ..

# install dependencies: agent
cd agent
pip install -e .
cd ..
```

## 2. Testbed: Overcooked Game

### 2.1 Introduction

The testbed is based on [gym-cooking (Wu, Sarah A., et al. "Too Many Cooks: Bayesian Inference for Coordinating Multi‐Agent Collaboration.")](https://github.com/rosewang2008/gym-cooking). Some additional features are added to the testbed:

1. **Diverse Ingredients**: We offer three ingredients, i.e., onion, tomato, and lettuce.
2. **Chopping Mechanism**: All ingredients must be chopped on the chop board before they are mixed and put in the pot.
3. **New Dishes**: design four dishes, i.e., Alice/Bob/Cathy/David Soup.
4. **Fire Mechanism**: Leaving the cooked soup in pot too long would overcook the soup and set the pot on fire.
5. **Trash Can**: Dispose of any unwanted ingredients.
6. **Order Timeout**: Soup orders appear randomly and expires after a certain time.
7. **Human-AI Chat Interface**: Human player can send chat messages to the AI agent, and the AI agent can give responses.

### 2.2 Interaction

Human player can play the game with keyboard. Each human player controls the movement of one character. Interacting with items on the map is also done by moving. For example, when closed to a Cut Board with some vegetable on it, try to move towards it repeatedly to chop the vegetable. The keys for movement are as follows:

- player1: `up`/`down`/`left`/`right`
- player2: `w`/`s`/`a`/`d` (only in Human-Human Play)
- chat: `space` (only in Human-AI Play)

The game will pause to wait for human player to type in the chat message.

## 3. Play

### 3.1 Human-Human Play

```bash
python testbed-cooking/gym_cooking/play_test.py --map partition
```

- `map`: can be `ring`, `bottleneck`, `partition` or `quick`.

### 3.2 Human-AI Play

Assume the LLM is running on a server. And its `API` is exposed on port `5000` of local machine.

```bash
# set env variables on Linux
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_ORGANIZATION="YOUR_ORGANIZATION"
export LLAMA_ADDRESS="127.0.0.1:5000"

# set env variables on Windows
$env:OPENAI_API_KEY="YOUR_API_KEY"
$env:OPENAI_ORGANIZATION="YOUR_ORGANIZATION"
$env:LLAMA_ADDRESS="127.0.0.1:5000"

python agent/agent/play_main.py --map ring --agent HLA
```

If you don't have an organization, you can use `OPENAI_ORGANIZATION=""` instead.

- `map`: can be `ring`, `bottleneck`, `partition` or `quick`. In the first three maps, the AI agent is set at a speed of `2.5` fps. In the `quick` map, the AI agent is set at a speed of `3.5` fps.

- `agent`: can be`HLA`,`SMOA`,`FMOA` or `NEA`.`HLA` is Hierarchical Language Agent. `SMOA` is Slow-Mind-Only Agent. `FMOA` is Flow-Mind-Only Agent. `NEA` is No-Executor Agent.

After playing a round, the replay will be saved in `agent/agent/replay` folder.

### 3.3 Human-AI Replay

The replay can be replayed by the following command:

```bash
python agent/agent/replay_main.py --replay ring-HLA-20010101_0101.rep
```

The replay is played in `2X` speed.

## Citation

If you find this repository useful, please cite [our paper](https://arxiv.org/abs/2312.15224):

```
@misc{liu2023llmpowered,
      title={LLM-Powered Hierarchical Language Agent for Real-time Human-AI Coordination}, 
      author={Jijia Liu and Chao Yu and Jiaxuan Gao and Yuqing Xie and Qingmin Liao and Yi Wu and Yu Wang},
      year={2023},
      eprint={2312.15224},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
