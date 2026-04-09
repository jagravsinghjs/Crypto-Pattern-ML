Here is the formatted Markdown for your project. You can copy and paste this directly into your `README.md` file.

***

# Crypto-Pattern-ML

## Overview
This project implements a complete machine learning pipeline for algorithmic trading on Bitcoin (BTC/USDT) using historical OHLCV data. 

The objective is not to predict prices directly, but to design a decision system that determines when to buy, sell, or hold, and evaluates whether the strategy outperforms a Buy & Hold benchmark under realistic constraints (including transaction costs).

## Problem Formulation
We model the problem as:
$$p_t = P(r_{t+1} > 0 \mid X_t)$$
$$\pi_t = f(p_t, \sigma_t)$$

Where:
* $r_{t+1}$: next-day return
* $X_t$: feature vector at time $t$
* $p_t$: predicted probability of price increase
* $\sigma_t$: volatility
* $\pi_t$: trading decision

The goal is to maximize:
$$\mathbb{E}[\text{PnL}] \quad \text{subject to transaction cost and risk}$$

## Dataset
* **Asset:** Bitcoin (BTC/USDT)
* **Frequency:** Daily
* **Period:** Jan 2020 – Dec 2023
* **Rows:** 1461
* **Features:** Open, High, Low, Close (OHLC), Volume

## Pipeline
1.  **Raw Data** → Preprocessing (returns, target)
2.  **Feature Engineering**
3.  **Model Training**
4.  **Probability Prediction**
5.  **Trading Strategy** (signal generation)
6.  **Backtesting** (with fees)
7.  **Evaluation** (vs Buy & Hold)

## Feature Engineering

**1. Returns (stationarity)**
$$r_t = \log\left(\frac{P_t}{P_{t-1}}\right)$$

**2. Moving Averages**
$$MA_k = \frac{1}{k} \sum_{i=0}^{k-1} P_{t-i}$$

**3. Momentum**
$$\text{Momentum}_k = P_t - P_{t-k}$$

**4. Volatility**
$$\sigma_t = \text{std}(r_{t-k:t})$$

**5. Lag Features**
$r_{t-1}, r_{t-2}, \dots, r_{t-5}$

**6. Volume Normalization**
$$V' = \frac{V - \mu}{\sigma}$$

## Model
* **Baseline:** Logistic Regression (linear classifier)
* **Notes:** * Predicts probability of upward movement.
    * Limited by a linear decision boundary.

## Trading Strategy
Signals are derived from model probability and volatility. 

**Decision Rule:**
* **High volatility** → stricter thresholds
* **Low volatility** → relaxed thresholds

```python
if v > 0.04:
    if p > 0.75: 
        BUY
    elif p < 0.25: 
        SELL
    else: 
        HOLD
else:
    if p > 0.6: 
        BUY
    elif p < 0.4: 
        SELL
    else: 
        HOLD
```

## Backtesting
* **Assumptions:**
    * Initial capital: ₹1000
    * Full allocation per trade
    * Transaction cost: 0.15%
* **Portfolio Update:**
    $$E_t = \text{cash} + \text{BTC} \cdot P_t$$

## Results

| Metric | Value |
| :--- | :--- |
| **Strategy Final Value** | ₹996.43 |
| **Buy & Hold** | ₹884.03 |

**Interpretation:**
* The strategy reduces losses during bearish periods.
* Logistic Regression provides a weak predictive signal (~48% accuracy).
* Performance is mainly driven by risk control + thresholding.

## Observations
* **Model Accuracy:**
    * Train: ~55.7%
    * Test: ~48.0% (below random baseline)
* **Strategy Behavior:**
    * Mostly generates BUY/HOLD signals.
    * Rare SELL signals due to low confidence.
* **Market Regime Effect:** Outperformance occurs mainly during bearish phases.

## Limitations
* Linear model (underfitting nonlinear market behavior).
* Weak predictive signal.
* No position sizing (all-in/all-out trading).
* No risk metrics incorporated (Sharpe, Max Drawdown).

## Future Improvements
* **Model:** Random Forest, XGBoost (nonlinear).
* **Features:** MA ratios, Rolling mean/std of returns, Technical indicators (RSI, MACD, Bollinger Bands).
* **Strategy:** * Position sizing:
        $$\text{position} = 2(p - 0.5)$$
    * Volatility scaling
    * Stop-loss mechanisms
* **Evaluation:** Sharpe Ratio, Sortino Ratio, Max Drawdown.

## Project Structure
```text
Crypto-Pattern-ML/
├── data/
├── models/
├── notebooks/
└── README.md
```

## Key Takeaway
> This project demonstrates that in financial markets, small predictive edges combined with strong risk management can outperform naive strategies, even when the underlying ML model is weak.

## Setup

```bash
git clone https://github.com/jagravsinghjs/Crypto-Pattern-ML.git
cd Crypto-Pattern-ML

python3 -m venv venv
source venv/bin/activate

pip install pandas numpy scikit-learn xgboost matplotlib
```

## Author
**Jagrav Singh**
*BTech/MTech Student*

## License
MIT License