# Markov Decision Process – Snakes and Ladders

## 1. Project Description

This repository contains the implementation of a [**Markov Decision Process (MDP)**](https://en.wikipedia.org/wiki/Markov_decision_process) applied to a simplified *Snakes and Ladders* game.
The project was developed in the context of the course [**LINFO2275 – Data Mining and Decision Making**](https://sites.uclouvain.be/archives-portail/cdc2022/en-cours-2022-linfo2275) (UCLouvain, 2024–2025).

The goal was to design and implement the **Value Iteration** algorithm to compute the optimal dice policy, under the constraint of using only **Python and NumPy**, without external reinforcement learning libraries.

### 1.1 Snake & Ladder
The game is composed of 15 squares, with a branching point at square 3 leading to a **slow lane** or a **fast lane**.  The objective is to reach square 15 while choosing dice strategically to minimize the expected number of turns.

![Board layout](assets\snakes_and_ladder.png)

Each dice available offers different risk and movement properties: a safe die (moves 0–1 step, immune to traps and bonuses), a normal die (moves 0–2 steps, 50% chance of triggering traps/bonuses), and a risky die (moves 0–3 steps, traps/bonuses always triggered).

A full problem statement is available in `Projet_Data_Mining_LINFO2275_2024_2025.pdf`.

### 1.2 Markov Decision Processes (MDPs)

The game is modeled as a MDP, a framework for sequential decision-making where outcomes depend only the current state and the chosen action. Since the board configuration and dice probabilities are fully known, this falls under **model-based reinforcement learning**, and the optimal policy can be computed using **dynamic programming techniques**, specifically Value Iteration.

A more detailed explanation of MDPs and their application to this problem is provided in the report (`report.pdf`).

## 2. Repository Structure

* `boardgame.py` – main implementation:

  * `BoardGame` class: environment, transitions, traps, Value Iteration.
  * `markovDecision(layout, circle)`: entry function returning expected costs and optimal dice per state.
* `report.pdf` – project report, including theoretical background, methodology, experiments, and results.
* `Projet_Data_Mining_LINFO2275_2024_2025.pdf` – official project description and requirements.
* `requirements.txt` – dependencies (only NumPy).

## 3. Usage Example

```python
import numpy as np
from boardgame import markovDecision

# Define a board layout (0=normal, 1=restart, 2=-3, 3=bonus, 4=prison)
layout = np.zeros(15, dtype=int) # Layout MUST be 15 tiles
layout[3] = 2
layout[6] = 1
layout[9] = 4
layout[10] = 3  

# Compute optimal strategy
expec, dice = markovDecision(layout, circle=False)

print("Expected costs:", expec)
print("Optimal dice:", dice)
```
