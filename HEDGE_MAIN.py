import tkinter as tk

def calculate_hedge():
    try:
        name = name_entry.get().strip()
        liability = float(liab_entry.get())
        hedge_pct = float(pct_entry.get()) / 100.0
        back_odds = float(odds_entry.get())

        if back_odds <= 1.01 or hedge_pct < 0 or liability <= 0:
            result_var.set("Check your inputs.")
            return

        stake = (liability * hedge_pct) / (back_odds - 1)
        result_var.set(
            f"Golfer: {name}\n"
            f"Hedge Stake: £{stake:.2f} @ {back_odds}\n"
            f"(Liability: £{liability:.2f}   Hedge: {int(hedge_pct*100)}%)"
        )
    except Exception as e:
        result_var.set("Input error.")

def clear_all():
    name_entry.delete(0, tk.END)
    liab_entry.delete(0, tk.END)
    pct_entry.delete(0, tk.END)
    odds_entry.delete(0, tk.END)
    result_var.set("")

root = tk.Tk()
root.title("Golf Hedge Calculator")

tk.Label(root, text="Golfer Name:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
name_entry = tk.Entry(root, width=20); name_entry.grid(row=0, column=1, padx=4, pady=2)

tk.Label(root, text="Current Liability (£):").grid(row=1, column=0, sticky="e", padx=4, pady=2)
liab_entry = tk.Entry(root, width=10); liab_entry.grid(row=1, column=1, padx=4, pady=2)

tk.Label(root, text="Hedge % (e.g. 30):").grid(row=2, column=0, sticky="e", padx=4, pady=2)
pct_entry = tk.Entry(root, width=10); pct_entry.grid(row=2, column=1, padx=4, pady=2)

tk.Label(root, text="Back Odds:").grid(row=3, column=0, sticky="e", padx=4, pady=2)
odds_entry = tk.Entry(root, width=10); odds_entry.grid(row=3, column=1, padx=4, pady=2)

tk.Button(root, text="Calculate Hedge Stake", command=calculate_hedge)\
    .grid(row=4, column=0, pady=8, padx=4)
tk.Button(root, text="Clear", command=clear_all)\
    .grid(row=4, column=1, pady=8, padx=4)

# Output window styling
output_frame = tk.Frame(root, bg="#eef2fa", bd=2, relief="groove")
output_frame.grid(row=5, column=0, columnspan=2, padx=6, pady=(10,8), sticky="ew")
result_var = tk.StringVar()
tk.Label(output_frame, textvariable=result_var, fg="#0a4da4", bg="#eef2fa",
         font=("Arial", 12, "bold"), justify="left", anchor="w", padx=8, pady=10)\
    .pack(fill="both", expand=True)

root.mainloop()
