
# Glosten–Milgrom Model (educational simulation)

This repository contains a compact **Monte Carlo simulation** of the Glosten & Milgrom (1985) sequential trade model.
An asset has a binary fundamental value \( V \in \{V_L, V_H\} \). In each period a single order arrives:
with probability \(\mu\) it comes from an **informed trader**, otherwise from an **uninformed** (noise) trader
who buys/sells with 50/50 probability. A competitive market maker updates beliefs via **Bayes' rule**
upon seeing the order direction and sets quotes to break even in expectation.

## Core ideas
- **Belief**: \(\delta_t = \Pr(V = V_L \mid \text{history up to } t)\).
- **Bayesian updates** (used in the code):  
  After a **buy**: \( \delta' = \frac{\delta(1-\mu)}{1 + \mu(1-2\delta)} \)  
  After a **sell**: \( \delta' = \frac{\delta(1+\mu)}{1 - \mu(1-2\delta)} \).
- **Quotes**: for didactic clarity we use a simple quoting rule around the mid (see `calculate_prices`).  
  For the exact GM rule you would set **ask** = \(\mathbb{E}[V \mid \text{Buy}]\) and **bid** = \(\mathbb{E}[V \mid \text{Sell}]\).

## What the script does
1. Simulates many paths and stores: **spread**, **belief** \(\delta_t\), **fundamental**, **ask**, **bid**.
2. Averages across Monte Carlo runs.
3. Plots:
   - Spread dynamics
   - Posterior probability \(\delta_t\)
   - Fundamental vs ask/bid

## Parameters (edit at the top of the script)
- `V_H`, `V_L`: high/low values of the asset (default 101/99).
- `delta`: initial prior \(\Pr(V=V_L)\) (default 0.5).
- `mu`: fraction of informed traders (default 0.2).
- `n_tiks`: number of time steps per simulation (default 1000).
- `n_simulations`: Monte Carlo replications (default 1000).

## Run
```bash
python Glosten_Milgrom_model_02.py
```

## Requirements
```
numpy
matplotlib
```

> Note: This is a **didactic** implementation that keeps your original structure intact and comments in **English**.
  Feel free to replace `calculate_prices` with the exact conditional-expectation quotes if you want a textbook-perfect version.

### Reference
Glosten, L. R., & Milgrom, P. R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*. JFE 14(1), 71–100.
