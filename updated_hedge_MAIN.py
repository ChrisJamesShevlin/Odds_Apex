#!/usr/bin/env python3
# Odds Apex — Golf (Lay-Only)   v2025‑07‑24

import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import math

# ───────────────────── CONFIG ───────────────────── #
p1, s1 = 0.0064, 20
p2, s2 = 0.10,   60
a, b   = 0.10, 22        # flattened slope + shifted midpoint

P_FLOOR          = 0.02
MAX_FAIR         = 50.0
SB_SCALE         = 0.35
BLEND_MODEL      = 0.25   # 25% heuristic, 75% simulation
FAIR_BLEND_WT    = 0.65
TOTAL_HOLES      = 72
COURSE_FIT_SCALE = 100
commission       = 0.02
KELLY_FRACTION   = 0.50

FIELD_CONTENDERS = {"weak": 30, "average": 70, "strong": 120}
EDGE_FLOOR_LAY   = 3.0

golfer_rows: list[dict] = []

# ─────────────────── MONTE‑CARLO ─────────────────── #
def simulate_win_prob(shots_diff, holes_left, sg_expect_round,
                      contenders=70, sims=5000, rnd_sd=2.4):
    md = -(sg_expect_round * holes_left / 18.0)
    sd = rnd_sd * math.sqrt(holes_left / 18.0)
    you = shots_diff + np.random.normal(md, sd, size=sims)
    oth = np.random.normal(0, sd, size=(sims, contenders-1))
    return np.mean(you[:, None] <= oth.min(axis=1))


# ────────────────── MAIN CALCULATION ────────────────── #
def calculate_score():
    try:
        name            = name_entry.get().strip()
        xwins           = float(xwins_entry.get())
        total_shots     = float(total_shots_entry.get())
        putt            = float(putt_entry.get())
        T2G             = float(t2g_entry.get())
        sg_true         = float(sg_true_entry.get())
        sg_expected_pre = float(sg_expected_entry.get())
        course_fit      = float(course_fit_entry.get())
        ranking         = float(ranking_entry.get())
        live_odds       = float(live_odds_entry.get())
        leaderboard_pos = float(leaderboard_pos_entry.get())
        finishes        = [float(e.get()) for e in finish_entries]

        sg_off_tee      = float(sg_off_tee_entry.get())
        sg_approach     = float(sg_approach_entry.get())
        sg_putting      = float(sg_putting_entry.get())
        sg_around_green = float(around_green_entry.get())

        holes_left      = int(holes_left_entry.get())
        quality_key     = quality_var.get().lower().strip()
        shots_diff      = float(shots_diff_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Enter valid numbers in every field.")
        return

    # basic stats
    n_contenders = FIELD_CONTENDERS.get(quality_key, 70)
    sg_diff      = sg_true - sg_expected_pre
    avg_last5    = sum(finishes) / len(finishes)
    pressure     = sg_diff * 15

    # ───── SG‑imbalance penalty ────── #
    t2g_score = sg_off_tee + sg_approach + sg_around_green
    imbalance_penalty = max(0.0, sg_putting - t2g_score) * 0.5
    # ────────────────────────────────── #

    # raw score build
    score = (
        50
        + xwins
        + 0.5 * (total_shots + putt + T2G)
        + pressure
        + course_fit * COURSE_FIT_SCALE
        - 0.5 * ranking
        - 0.3 * leaderboard_pos
        - 0.5 * avg_last5
        + 0.5 * (sg_off_tee + sg_approach + sg_putting + sg_around_green)
    )
    score -= (shots_diff / math.sqrt(max(holes_left,1))) * SB_SCALE
    score -= imbalance_penalty
    score = float(np.clip(score, 0, 100))

    # heuristic win‑prob
    p_model = max(1 / (1 + math.exp(-a * (score - b))), P_FLOOR)

    # simulate
    holes_played = TOTAL_HOLES - holes_left
    sg_remaining = sg_expected_pre
    if holes_played:
        sg_so_far = sg_off_tee + sg_approach + sg_putting + sg_around_green
        rate_h    = min(sg_so_far / holes_played, 0.12)
        sg_remaining = 0.7 * sg_expected_pre + 0.3 * rate_h * holes_left

    p_sim = simulate_win_prob(shots_diff, holes_left, sg_remaining, n_contenders)

    # final blend
    p_final = BLEND_MODEL * p_model + (1 - BLEND_MODEL) * p_sim
    p_final = min(max(p_final, P_FLOOR), 0.25)
    
        # … after p_final = … blend … 

    # simple fragility penalty
    total_sg = abs(sg_off_tee) + abs(sg_approach) + abs(sg_putting) + abs(sg_around_green)
    putt_share = abs(sg_putting) / max(total_sg, 1e-8)
    # only penalise when putting is more than 30% of total SG
    if putt_share > 0.30:
        # penalty scales from 0→20% as putting_share goes 30%→100%
        fragility = (putt_share - 0.30) / 0.70 * 0.20  
        p_final = max(P_FLOOR, p_final * (1 - fragility))

    # now floor/ceiling again
    p_final = min(max(p_final, P_FLOOR), 0.25)


    # edge & fair odds
    p_mkt   = 1 / live_odds
    edge    = p_final - p_mkt
    fair_m  = min(1 / p_final, MAX_FAIR)
    fair_bl = FAIR_BLEND_WT * fair_m + (1 - FAIR_BLEND_WT) * live_odds
    ev_lay  = (1 - p_final) - p_final * (live_odds - 1)

    # store & print
    new_row = dict(
        Name=name, p_model=p_final, Market_p=p_mkt, Edge=edge,
        FairOdds=fair_bl, LiveOdds=live_odds, EV=ev_lay,
        holes_left=holes_left
    )
    if golfer_rows and golfer_rows[-1]["Name"] == name:
        golfer_rows[-1] = new_row
    else:
        golfer_rows.append(new_row)

    print(f"{name:12s} | Model:{p_final*100:6.2f}%  "
          f"Market:{p_mkt*100:6.2f}%  Edge:{edge*100:+6.2f}%  "
          f"Imb:{imbalance_penalty:.2f}  Score:{score:.2f}")


# ────────────────── HEDGE PERCENT FUNCTION ────────────────── #
def recommended_hedge_pct(odds, holes_left, direction="lay"):
    """
    Return the % of current red (liability) to hedge, based on odds bracket and round.
    """
    # End of R1: 54+ holes left (target: 15–30x)
    if holes_left >= 48:
        if 15.0 <= odds <= 30.0:
            return 0.60
        elif odds > 30.0:
            return 0.20
        else:
            return 0.80
    # End of R2: 25–47 holes left (target: 8–15x)
    elif 24 < holes_left < 48:
        if 8.0 <= odds <= 15.0:
            return 0.60
        elif odds > 15.0:
            return 0.25
        else:
            return 0.80
    # End of R3: 10–24 holes left (target: 4.5–8x)
    elif 9 < holes_left <= 24:
        if 4.5 <= odds <= 8.0:
            return 0.60
        elif odds > 8.0:
            return 0.30
        else:
            return 0.85
    # Final 9 holes
    else:
        if odds < 4.5:
            return 0.90
        else:
            return 0.30

def calc_hedge_back_stake(current_red, hedge_pct, back_odds):
    """
    Returns the back stake needed to hedge a % of your *current red* at the given odds.
    No minimum stake, no rounding.
    """
    if back_odds <= 1.01 or hedge_pct <= 0 or current_red <= 0:
        return 0.0
    stake = (current_red * hedge_pct) / (back_odds - 1)
    return stake

# Usage example:
# For Knapp: red = 4.49, hedge % = 0.60, odds = 21.0
knapp_stake = calc_hedge_back_stake(4.49, 0.60, 21.0)
# For Stevens: red = 1.65, hedge % = 0.30, odds = 19.0
stevens_stake = calc_hedge_back_stake(1.65, 0.30, 19.0)

print(f"Back Knapp: £{knapp_stake:.2f} at 21.0")
print(f"Back Stevens: £{stevens_stake:.2f} at 19.0")



# ────────────────── COMPARISON POP‑UP ────────────────── #
def compare_golfers():
    if not golfer_rows:
        messagebox.showinfo("Compare", "No golfers entered yet.")
        return
    try:
        bankroll = float(bankroll_entry.get())
        assert bankroll > 0
    except:
        messagebox.showerror("Input", "Enter a positive bankroll.")
        return

    df = pd.DataFrame(golfer_rows)

    def _lay_liab(r):
        if -r.Edge*100 < EDGE_FLOOR_LAY: return 0.0
        return max(
            0.0,
            KELLY_FRACTION * bankroll * (r.Market_p - r.p_model) / (r.LiveOdds - 1)
        )

    df["LayLiability"]       = df.apply(_lay_liab, axis=1)
    df["LayStake"]           = np.where(df.LayLiability>0,
                                        df.LayLiability/(df.LiveOdds-1), 0.0)
    df["LayHedgePct"]        = df.apply(lambda r:
        recommended_hedge_pct(r.LiveOdds, r.holes_left, "lay") if r.LayLiability>0 else 0,
        axis=1
    )
    df["LayHedgeStake"]      = df.LayLiability * df.LayHedgePct
    df["LayHedgePctDisplay"] = df.LayHedgePct.apply(lambda x: f"{int(x*100)}%")
    df["RankProb"]           = df.p_model.rank(ascending=False, method="min")
    df["RankEdge"]           = df.Edge.rank(ascending=False, method="min")
    df.sort_values("Edge", inplace=True, ascending=False)

    # columns + headers
    cols = [
        "Name","p_model","Market_p","Edge","FairOdds","LiveOdds",
        "LayStake","LayLiability","LayHedgeStake","LayHedgePctDisplay",
        "RankProb","RankEdge"
    ]
    headers = {
        "p_model":"Model %","Market_p":"Mkt %","Edge":"Edge %",
        "FairOdds":"Fair","LiveOdds":"Odds",
        "LayStake":"Lay Stake £","LayLiability":"Lay Liab £",
        "LayHedgeStake":"Lay Hedge £","LayHedgePctDisplay":"Lay Hedge %",
        "RankProb":"RankProb","RankEdge":"RankEdge"
    }

    # build Treeview
    top = tk.Toplevel(root); top.title("Odds Apex – Golf (Lay-Only)")
    tree = ttk.Treeview(top, columns=cols, show="headings", height=min(len(df)+1,30))
    vsb  = ttk.Scrollbar(top, orient="vertical",   command=tree.yview)
    hsb  = ttk.Scrollbar(top, orient="horizontal", command=tree.xview)
    tree.configure(yscroll=vsb.set, xscroll=hsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    top.grid_rowconfigure(0, weight=1); top.grid_columnconfigure(0, weight=1)

    # headings
    for c in cols:
        tree.heading(c, text=headers.get(c, c))
        tree.column(c,
            width=140 if c in ("Name","LayStake","LayLiability","LayHedgeStake") else 118,
            anchor="center"
        )

    # rows
    for _, r in df.iterrows():
        tree.insert("", tk.END, values=[
            r["Name"],
            f"{r.p_model*100:.2f}%",
            f"{r.Market_p*100:.2f}%",
            f"{r.Edge*100:+.2f}%",
            f"{r.FairOdds:.2f}",
            f"{r.LiveOdds:.2f}",
            f"£{r.LayStake:.2f}",
            f"£{r.LayLiability:.2f}",
            f"£{r.LayHedgeStake:.2f}",
            r.LayHedgePctDisplay,
            int(r.RankProb),
            int(r.RankEdge),
        ])

def clear_list():
    golfer_rows.clear()
    messagebox.showinfo("Reset", "Stored list cleared.")

# ───────────────────── GUI ───────────────────── #
root = tk.Tk()
root.title("Odds Apex – Golf (Lay-Only)")

fields = [
    "Golfer Name","Expected Wins (xwins)","Total Shots Gained","Putt",
    "T2G","SG True","SG Expected","Course Fit","Current Ranking",
    "Live Odds","Leaderboard Position","Shots +/-"
]
entries = {}
for i, lbl in enumerate(fields):
    tk.Label(root, text=lbl).grid(row=i, column=0, sticky="e", padx=4, pady=2)
    e = tk.Entry(root); e.grid(row=i, column=1, padx=4, pady=2)
    entries[lbl] = e

(
    name_entry, xwins_entry, total_shots_entry, putt_entry, t2g_entry,
    sg_true_entry, sg_expected_entry, course_fit_entry, ranking_entry,
    live_odds_entry, leaderboard_pos_entry, shots_diff_entry
) = [entries[l] for l in fields]

# Last‑5 finishes
last5_row = len(fields)
tk.Label(root, text="Last 5 Finishes").grid(row=last5_row, column=0, sticky="e")
finish_frame = tk.Frame(root); finish_frame.grid(row=last5_row, column=1)
finish_entries = []
for j in range(5):
    e = tk.Entry(finish_frame, width=4); e.grid(row=0, column=j, padx=2)
    finish_entries.append(e)

# SG inputs
sg_inputs = ["SG Off Tee","SG Approach","SG Putting","SG Around the Green"]
for k, lbl in enumerate(sg_inputs, start=last5_row+1):
    tk.Label(root, text=lbl).grid(row=k, column=0, sticky="e", padx=4, pady=2)
    e = tk.Entry(root); e.grid(row=k, column=1, padx=4, pady=2)
    if lbl=="SG Off Tee":        sg_off_tee_entry = e
    elif lbl=="SG Approach":     sg_approach_entry = e
    elif lbl=="SG Putting":      sg_putting_entry = e
    else:                        around_green_entry = e

new_row = last5_row + 1 + len(sg_inputs)
tk.Label(root, text="Holes Remaining").grid(row=new_row, column=0, sticky="e")
holes_left_entry = tk.Entry(root); holes_left_entry.grid(row=new_row, column=1)
tk.Label(root, text="Contenders").grid(row=new_row+1, column=0, sticky="e")
n_contenders_entry = tk.Entry(root); n_contenders_entry.grid(row=new_row+1, column=1)
tk.Label(root, text="Field Quality").grid(row=new_row+2, column=0, sticky="e")
quality_var = tk.StringVar(root); quality_var.set("average")
tk.OptionMenu(root, quality_var, "weak","average","strong").grid(row=new_row+2, column=1)
tk.Label(root, text="Bankroll (£)").grid(row=new_row+3, column=0, sticky="e")
bankroll_entry = tk.Entry(root); bankroll_entry.grid(row=new_row+3, column=1)

tk.Button(root, text="Calculate Score & EV", command=calculate_score)\
  .grid(row=new_row+4, column=0, columnspan=2, pady=6)
tk.Button(root, text="Compare Entered Golfers", command=compare_golfers)\
  .grid(row=new_row+5, column=0, columnspan=2, pady=4)
tk.Button(root, text="Clear List", command=clear_list)\
  .grid(row=new_row+6, column=0, columnspan=2, pady=2)

root.mainloop()
