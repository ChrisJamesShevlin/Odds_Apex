import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import math

# ------------------------  CONFIG / GLOBALS  ------------------------

p1, s1 = 0.0064, 20          # logistic anchors
p2, s2 = 0.10,   60
L1, L2  = math.log(p1/(1-p1)), math.log(p2/(1-p2))
a       = (L2-L1)/(s2-s1)
b       = s1 - L1/a

P_FLOOR          = 0.02
MAX_FAIR         = 50.0
SB_SCALE         = 0.35
BLEND_MODEL      = 0.6
TOTAL_HOLES      = 72
COURSE_FIT_SCALE = 100
commission       = 0.02       # exchange commission
KELLY_FRACTION   = 0.25       # ¼‑Kelly

golfer_rows = []              # session storage – one dict per Calculate

# ------------------------  MONTE‑CARLO (vectorised)  ------------------------

def simulate_win_prob(shots_behind, holes_left, sg_expect_round,
                      contenders=20, sims=5000, rnd_sd=2.4):
    md   = -(sg_expect_round * holes_left / 18.0)
    sd   = rnd_sd * math.sqrt(holes_left / 18.0)
    you  = shots_behind + np.random.normal(md, sd, size=sims)
    oth  = np.random.normal(0, sd, size=(sims, contenders-1))
    return np.mean(you[:, None] <= oth.min(axis=1))

# ------------------------  MAIN CALCULATION  ------------------------

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

        holes_left    = int(holes_left_entry.get())
        n_contenders  = int(n_contenders_entry.get())
        quality_key   = quality_var.get()
        shots_behind  = float(shots_behind_entry.get())
    except ValueError:
        messagebox.showerror("Input Error",
                             "Please enter valid numbers in every field.")
        return

    sg_diff   = sg_true - sg_expected_pre
    avg_last5 = sum(finishes) / len(finishes)
    pressure  = sg_diff * 15

    score = (
        50 + xwins
        + total_shots * 0.5 + putt * 0.5 + T2G * 0.5
        + pressure + course_fit * COURSE_FIT_SCALE
        - ranking * 0.5 - leaderboard_pos * 0.3 - avg_last5 * 0.5
        + sg_off_tee * 0.5 + sg_approach * 0.5
        + sg_putting * 0.5 + sg_around_green * 0.5
    )

    score -= (shots_behind / math.sqrt(max(holes_left, 1))) * SB_SCALE
    score /= {"weak": 0.9, "average": 1.0, "strong": 1.1}[quality_key]
    score  = float(np.clip(score, 0, 100))

    p_model = 1 / (1 + math.exp(-a * (score - b)))
    p_model = max(p_model, P_FLOOR)

    holes_played = TOTAL_HOLES - holes_left
    if holes_played:
        sg_so_far       = sg_off_tee + sg_approach + sg_putting + sg_around_green
        sg_rate_per_hole = sg_so_far / holes_played
        sg_remaining     = 0.5 * sg_expected_pre + 0.5 * sg_rate_per_hole * holes_left
    else:
        sg_remaining = sg_expected_pre

    p_sim   = simulate_win_prob(shots_behind, holes_left, sg_remaining,
                                n_contenders)
    p_final = BLEND_MODEL * p_model + (1 - BLEND_MODEL) * p_sim

    p_mkt   = 1 / live_odds
    edge    = p_final - p_mkt
    fair_m  = min(1 / p_final, MAX_FAIR)
    fair_bl = 0.7 * fair_m + 0.3 * live_odds
    ev_back = p_final * (live_odds - 1) - (1 - p_final)

    print(f"{name} | Model:{p_final*100:6.2f}%  Market:{p_mkt*100:6.2f}%  "
          f"Edge:{edge*100:+6.2f}%  EV:{ev_back:+.3f}")

    golfer_rows.append({
        "Name": name,
        "p_model": p_final,
        "Market_p": p_mkt,
        "Edge": edge,
        "FairOdds": fair_bl,
        "LiveOdds": live_odds,
        "EV": ev_back
    })

# ------------------------  COMPARISON & KELLY TABLE  ------------------------

def compare_golfers():
    if not golfer_rows:
        messagebox.showinfo("Compare", "No golfers entered yet.")
        return

    try:
        bankroll = float(bankroll_entry.get())
        if bankroll <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Bankroll",
                             "Please enter a positive bankroll amount first.")
        return

    df = pd.DataFrame(golfer_rows)

    # market probability
    df["q_mkt"] = 1 / (df["LiveOdds"] * (1 - commission))

    # Kelly liability fraction (¼‑Kelly, negatives clipped to 0)
    df["LiabFrac"] = (df["q_mkt"] - df["p_model"]) / (df["LiveOdds"] - 1)
    df.loc[df["LiabFrac"] < 0, "LiabFrac"] = 0

    # scale to keep portfolio feasible
    scaler = 1 / (1 + (df["LiabFrac"] * df["q_mkt"]).sum())
    df["LiabFrac"] *= scaler * KELLY_FRACTION
    df["StakeFrac"] = df["LiabFrac"] / (df["LiveOdds"] - 1)

    # absolute money and %
    df["Liability"] = df["LiabFrac"] * bankroll
    df["Stake"]     = df["StakeFrac"] * bankroll
    df["Liab_%"]    = df["LiabFrac"]  * 100
    df["Stake_%"]   = df["StakeFrac"] * 100

    # ranking / ordering
    df["RankProb"] = df["p_model"].rank(ascending=False, method="min")
    df["RankEdge"] = df["Edge"].rank(ascending=False, method="min")
    df.sort_values("Edge", ascending=False, inplace=True)

    # ---------- display ----------
    top = tk.Toplevel(root)
    top.title("Model vs. Market — Kelly stakes shown")

    cols = ["Name", "p_model", "Market_p", "Edge",
            "FairOdds", "LiveOdds",
            "Stake", "Liability",
            "Stake_%", "Liab_%", "RankProb", "RankEdge"]
    pretty = {
        "Name": "Name",          "p_model": "Model %",
        "Market_p": "Market %",  "Edge": "Edge %",
        "FairOdds": "Fair Odds", "LiveOdds": "Lay Odds",
        "Stake": "Stake £",      "Liability": "Liab £",
        "Stake_%": "Stake %",    "Liab_%": "Liab %",
        "RankProb": "RankProb",  "RankEdge": "RankEdge"
    }

    tree = ttk.Treeview(top, columns=cols, show="headings", height=len(df)+1)
    vsb  = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
    tree.configure(yscroll=vsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    top.grid_rowconfigure(0, weight=1)
    top.grid_columnconfigure(0, weight=1)

    for c in cols:
        tree.heading(c, text=pretty.get(c, c))
        tree.column(c,
                    width=110 if c not in ("Name", "Stake", "Liability") else 130,
                    anchor="center")

    for _, r in df.iterrows():
        tree.insert("", "end",
            values=[r["Name"],
                    f"{r['p_model']*100:.2f}%",
                    f"{r['Market_p']*100:.2f}%",
                    f"{r['Edge']*100:+.2f}%",
                    f"{r['FairOdds']:.2f}",
                    f"{r['LiveOdds']:.2f}",
                    f"£{r['Stake']:.2f}",
                    f"£{r['Liability']:.2f}",
                    f"{r['Stake_%']:.2f}",
                    f"{r['Liab_%']:.2f}",
                    int(r["RankProb"]),
                    int(r["RankEdge"])])

def clear_list():
    golfer_rows.clear()
    messagebox.showinfo("Reset", "Stored golfer list cleared.")

# ------------------------  GUI BUILD  ------------------------

root = tk.Tk()
root.title("Odds Apex - Golf Model")

fields = [
    "Golfer Name", "Expected Wins (xwins)", "Total Shots Gained",
    "Putt", "T2G", "SG True", "SG Expected",
    "Course Fit", "Current Ranking", "Live Odds",
    "Leaderboard Position", "Shots Behind Leader"
]
entries = {}
for i, lbl in enumerate(fields):
    tk.Label(root, text=lbl).grid(row=i, column=0, sticky="e", padx=4, pady=2)
    e = tk.Entry(root); e.grid(row=i, column=1, padx=4, pady=2)
    entries[lbl] = e

(name_entry, xwins_entry, total_shots_entry, putt_entry, t2g_entry,
 sg_true_entry, sg_expected_entry, course_fit_entry, ranking_entry,
 live_odds_entry, leaderboard_pos_entry, shots_behind_entry) = [
    entries[l] for l in fields]

# last‑5 finishes
last5_row = len(fields)
tk.Label(root, text="Last 5 Finishes").grid(row=last5_row, column=0,
                                            sticky="e", padx=4, pady=2)
finish_frame = tk.Frame(root)
finish_frame.grid(row=last5_row, column=1, pady=2)
finish_entries = []
for j in range(5):
    e = tk.Entry(finish_frame, width=4)
    e.grid(row=0, column=j, padx=2)
    finish_entries.append(e)

# in‑play SG
sg_stats = ["SG Off Tee", "SG Approach", "SG Putting", "SG Around the Green"]
for k, lbl in enumerate(sg_stats, start=last5_row + 1):
    tk.Label(root, text=lbl).grid(row=k, column=0, sticky="e", padx=4, pady=2)
    e = tk.Entry(root); e.grid(row=k, column=1, padx=4, pady=2)
    if lbl == "SG Off Tee":            sg_off_tee_entry   = e
    elif lbl == "SG Approach":         sg_approach_entry  = e
    elif lbl == "SG Putting":          sg_putting_entry   = e
    else:                              around_green_entry = e

# round inputs
new_row = last5_row + 1 + len(sg_stats)
tk.Label(root, text="Holes Remaining").grid(row=new_row, column=0,
                                            sticky="e", padx=4, pady=2)
holes_left_entry = tk.Entry(root); holes_left_entry.grid(row=new_row, column=1)

tk.Label(root, text="Contenders").grid(row=new_row+1, column=0,
                                       sticky="e", padx=4, pady=2)
n_contenders_entry = tk.Entry(root)
n_contenders_entry.grid(row=new_row+1, column=1)

tk.Label(root, text="Field Quality").grid(row=new_row+2, column=0,
                                          sticky="e", padx=4, pady=2)
quality_var = tk.StringVar(root); quality_var.set("average")
tk.OptionMenu(root, quality_var, "weak", "average", "strong") \
    .grid(row=new_row+2, column=1, padx=4, pady=2)

# bankroll
tk.Label(root, text="Bankroll (£)").grid(row=new_row+3, column=0,
                                         sticky="e", padx=4, pady=2)
bankroll_entry = tk.Entry(root); bankroll_entry.grid(row=new_row+3, column=1)

# buttons
tk.Button(root, text="Calculate Score & EV", command=calculate_score) \
    .grid(row=new_row+4, column=0, columnspan=2, pady=6)
tk.Button(root, text="Compare Entered Golfers", command=compare_golfers) \
    .grid(row=new_row+5, column=0, columnspan=2, pady=4)
tk.Button(root, text="Clear List", command=clear_list) \
    .grid(row=new_row+6, column=0, columnspan=2, pady=2)

root.mainloop()
