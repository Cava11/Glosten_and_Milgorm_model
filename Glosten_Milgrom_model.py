
"""
Glosten–Milgrom model (sequential trade microstructure, educational simulation)

This script simulates the classic Glosten & Milgrom (1985) setup with a binary fundamental value
V ∈ {V_L, V_H}. In each period, an order arrives from an informed trader with probability μ
and from an uninformed (noise) trader with probability 1−μ, the latter buying or selling
with 50/50 probability. A competitive market maker sets quotes (bid/ask) to break even
in expectation using Bayes' rule after observing order direction. The posterior belief
about the low state, δ_t = P(V = V_L | history), is updated after each buy/sell.

This code follows your structure and keeps the mechanics intact; comments and plot titles
are in English for GitHub. It also averages results across Monte Carlo replications.

Key quantities:
- δ_t: posterior probability P(V = V_L) after t trades
- μ: fraction of informed traders
- Quotes (simplified rule used here): ask/bid around the mid as a function of μ and δ
- Spread: reported for illustration; shrinks as δ → {0,1}

References:
Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market
with heterogeneously informed traders. Journal of Financial Economics, 14(1), 71–100.
"""
import random
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
V_H = 101   # High value of the asset
V_L = 99    # Low value of the asset
delta = 0.5 # Initial prior P(V = V_L)
mu = 0.2    # Fraction of informed traders (0 <= mu <= 1)

n_tiks = 1000       # Number of time steps ("ticks") in one simulation
n_simulations = 1000  # Monte Carlo replications

# (Optional) reproducibility
random.seed(42)
np.random.seed(42)

def calculate_prices(V_L, V_H, delta, mu):
    """
    Compute (ask, bid) given current belief delta and informed share mu.
    This is a simple linearized quoting rule around the mid-price
    (for didactic purposes). For exact GM pricing, one would set
    PA = E[V | Buy], PB = E[V | Sell] using Bayes' rule.
    """
    PA = (V_L + V_H) / 2 + (V_H - V_L) * mu * delta / 2
    PB = (V_L + V_H) / 2 - (V_H - V_L) * mu * delta / 2
    return PA, PB

def update_delta_buy(delta, mu):
    """
    Posterior update after observing a BUY (Bayes' rule).
    With uninformed buying with prob 1/2, one gets:
    P(L | buy) = [delta * (1 - mu)] / [1 + mu * (1 - 2*delta)].
    """
    return (delta * (1 - mu)) / (1 + mu * (1 - 2 * delta))

def update_delta_sell(delta, mu):
    """
    Posterior update after observing a SELL (Bayes' rule).
    P(L | sell) = [delta * (1 + mu)] / [1 - mu * (1 - 2*delta)].
    """
    return (delta * (1 + mu)) / (1 - mu * (1 - 2 * delta))

def simulate_model(delta, mu, V_L, V_H, n_tiks):
    """
    Run one path of the model for n_tiks periods.
    Returns histories for spread, delta, fundamental value, ask, bid.
    """
    spread_history = []
    delta_history = [delta]
    asset_price_history = []
    ask_history = []
    bid_history = []

    # Initial illustrative spread (not used in pricing step directly)
    spread = mu * (V_H - V_L)

    for _ in range(n_tiks):
        # Draw trader type: informed with prob mu, else uninformed
        trader_type = 'informed' if random.random() < mu else 'uninformed'

        # Draw true fundamental (binary state) from current prior
        true_value = V_L if random.random() < delta else V_H
        asset_price_history.append(true_value)

        # Quote prices
        PA, PB = calculate_prices(V_L, V_H, delta, mu)
        ask_history.append(PA)
        bid_history.append(PB)

        # Informed trader trades in the direction of the true value
        if trader_type == 'informed':
            if true_value == V_H:  # informed BUY
                delta = update_delta_buy(delta, mu)
            else:                  # informed SELL
                delta = update_delta_sell(delta, mu)

        # Illustrative spread dynamics: narrower as beliefs concentrate
        spread = mu * (V_H - V_L) * delta * (1 - delta)
        spread_history.append(spread)
        delta_history.append(delta)

    return spread_history, delta_history, asset_price_history, ask_history, bid_history

# --- Monte Carlo ---
all_spread_histories = []
all_delta_histories = []
all_asset_price_histories = []
all_ask_histories = []
all_bid_histories = []

for _ in range(n_simulations):
    (spread_history, delta_history, asset_price_history,
     ask_history, bid_history) = simulate_model(delta, mu, V_L, V_H, n_tiks)
    all_spread_histories.append(spread_history)
    all_delta_histories.append(delta_history)
    all_asset_price_histories.append(asset_price_history)
    all_ask_histories.append(ask_history)
    all_bid_histories.append(bid_history)

# Averages across simulations
mean_spread = np.mean(all_spread_histories, axis=0)
mean_delta = np.mean(all_delta_histories, axis=0)
mean_asset_price = np.mean(all_asset_price_histories, axis=0)
mean_ask = np.mean(all_ask_histories, axis=0)
mean_bid = np.mean(all_bid_histories, axis=0)

# --- Plots ---
plt.figure(figsize=(12, 10))

# Spread over time
plt.subplot(3, 1, 1)
plt.plot(mean_spread, label="Average spread", linewidth=1.25)
plt.title('Spread dynamics (Monte Carlo)')
plt.xlabel('Time (ticks)')
plt.ylabel('Bid–Ask spread')

# Posterior delta over time
plt.subplot(3, 1, 2)
plt.plot(mean_delta, label="Average delta", linewidth=1.25)
plt.title('Belief about low state δ_t (Monte Carlo)')
plt.xlabel('Time (ticks)')
plt.ylabel('δ_t = P(V = V_L)')

# Fundamental vs. quotes
plt.subplot(3, 1, 3)
plt.plot(mean_asset_price, label="Fundamental (avg)", linewidth=1.25)
plt.plot(mean_ask, label="Ask (avg)", linestyle='--', linewidth=1.0)
plt.plot(mean_bid, label="Bid (avg)", linestyle='--', linewidth=1.0)
plt.title('Fundamental, Ask and Bid (Monte Carlo averages)')
plt.xlabel('Time (ticks)')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.show()
