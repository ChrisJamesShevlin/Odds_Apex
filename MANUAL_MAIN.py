#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox

def calculate_stake():
    try:
        bankroll = float(bankroll_entry.get())
        edge_pct = float(edge_entry.get()) / 100.0
        odds = float(odds_entry.get())
        frac = float(fraction_var.get()) / 100.0

        if odds <= 1:
            raise ValueError("Odds must be > 1")
        b = odds - 1
        # full Kelly fraction
        f_star = edge_pct / b
        # apply user fraction (e.g. 50% Kelly)
        stake = f_star * bankroll * frac

        result_var.set(f"Recommended Stake: £{stake:.2f}")
    except Exception as e:
        messagebox.showerror("Input Error", str(e))

root = tk.Tk()
root.title("Quick Lay Stake Calculator")

# Bankroll
tk.Label(root, text="Bankroll (£):").grid(row=0, column=0, sticky="e", padx=4, pady=2)
bankroll_entry = tk.Entry(root)
bankroll_entry.grid(row=0, column=1, padx=4, pady=2)

# Edge %
tk.Label(root, text="Edge (%):").grid(row=1, column=0, sticky="e", padx=4, pady=2)
edge_entry = tk.Entry(root)
edge_entry.grid(row=1, column=1, padx=4, pady=2)

# Odds
tk.Label(root, text="Lay Odds:").grid(row=2, column=0, sticky="e", padx=4, pady=2)
odds_entry = tk.Entry(root)
odds_entry.grid(row=2, column=1, padx=4, pady=2)

# Kelly fraction
tk.Label(root, text="Kelly Fraction:").grid(row=3, column=0, sticky="e", padx=4, pady=2)
fraction_var = tk.StringVar(value="50")
fraction_menu = tk.OptionMenu(root, fraction_var, "100", "50", "25", "10")
fraction_menu.grid(row=3, column=1, padx=4, pady=2)

# Calculate button
tk.Button(root, text="Calculate Stake", command=calculate_stake)\
    .grid(row=4, column=0, columnspan=2, pady=6)

# Result display
result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, font=("Helvetica", 12, "bold"))\
    .grid(row=5, column=0, columnspan=2, pady=4)

root.mainloop()
