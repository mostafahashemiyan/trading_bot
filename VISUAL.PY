# equity_curve_plot.py (نسخه بهبودیافته)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

EQUITY_FILE = "equity_curve.csv"

try:
    equity = pd.read_csv(EQUITY_FILE, index_col=0, parse_dates=True).squeeze()
    print(f"Loaded {len(equity)} points from {equity.index[0].date()} → {equity.index[-1].date()}")
except:
    print("No equity file - using sample data")
    dates = pd.date_range("2025-01-01", periods=1000, freq="h")
    equity = pd.Series(np.random.normal(0, 0.5, 1000).cumsum() * 100 + 10000, index=dates)

# محاسبه drawdown
peak = equity.cummax()
drawdown = (equity - peak) / peak * 100

# نمودار
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Equity
ax1.plot(equity.index, equity, color='#1f77b4', lw=1.5, label='Equity Curve')
ax1.set_title('Backtest Equity Curve & Drawdown', fontsize=16, pad=15)
ax1.set_ylabel('Balance (USDT)', fontsize=12)
ax1.grid(True, alpha=0.3, ls='--')
ax1.legend(loc='upper left', fontsize=11)

# Drawdown
ax2.fill_between(drawdown.index, drawdown, 0, where=(drawdown <= 0), color='red', alpha=0.4, label='Drawdown')
ax2.axhline(0, color='black', lw=0.8, ls='--')
ax2.set_ylabel('Drawdown %', fontsize=12)
ax2.grid(True, alpha=0.3, ls='--')
ax2.legend(loc='lower left', fontsize=11)

# آمار روی نمودار
stats_text = f"Final: ${equity.iloc[-1]:,.0f}\nReturn: +{(equity.iloc[-1]/equity.iloc[0]-1)*100:.1f}%\nMax DD: {drawdown.min():.1f}%"
fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.xlabel('Date', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("backtest_performance.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Max Drawdown: {drawdown.min():.2f}%")
print(f"Final Balance: ${equity.iloc[-1]:,.2f}")
print(f"Total Return: +{(equity.iloc[-1]/equity.iloc[0]-1)*100:.2f}%")