#!/usr/bin/env python3
"""Odds Apex — Golf (Lay‑Only)
Full version with SG‑imbalance penalty helper and clean layout
Last updated 2025‑07‑24
"""

import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import math

# ───────────────────── CONFIG ───────────────────── #

p1, s1 = 0.0064, 20
p2, s2 = 0.10, 60
a, b = 0.30, 15

P_FLOOR, MAX_FAIR    = 0.02, 50.0
SB_SCALE, BLEND_MODEL, FAIR_BLEND_WT = 0.35, 0.25, 0.65
TOTAL_HOLES, COURSE_FIT_SCALE      = 72, 100
commission, KELLY_FRACTION          = 0.02, 0.50
FIELD_CONTENDERS = {"weak":30, "average":70, "strong":120}
EDGE_FLOOR_LAY = 3.0

# helper: imbalance penalty
def imbalance_penalty(sg_off, sg_app, sg_putt, sg_agrn,
                      baseline=0.30, scale=20.0) -> float:
    """How many points to subtract if SG components are overly concentrated."""
    vec = np.array([sg_off, sg_app, sg_agrn, sg_putt])
    total = float(vec.sum() or 1e-8)
    shares = vec / total
    hhi = float((shares**2).sum())
    return max(0.0, hhi - baseline) * scale

# session storage
golfer_rows: list[dict] = []

# ─────────────────── MONTE‑CARLO ─────────────────── #

def simulate_win_prob(shots_diff: float, holes_left: int,
                      sg_expect_round: float,
                      contenders: int=70, sims: int=5000,
                      rnd_sd: float=2.4) -> float:
    md = -(sg_expect_round * holes_left / 18)
    sd = rnd_sd * math.sqrt(holes_left/18)
    you = shots_diff + np.random.normal(md, sd, size=sims)
    oth = np.random.normal(0, sd, size=(sims, contenders-1))
    return float(np.mean(you[:,None] <= oth.min(axis=1)))

# ────────────────── MAIN CALCULATION ────────────────── #

def calculate_score():
    try:
        # basic
        name = name_entry.get().strip()
        xwins = float(xwins_entry.get())
        total_shots = float(total_shots_entry.get())
        putt = float(putt_entry.get())
        t2g = float(t2g_entry.get())
        sg_true = float(sg_true_entry.get())
        sg_exp = float(sg_expected_entry.get())
        course_fit = float(course_fit_entry.get())
        ranking = float(ranking_entry.get())
        live_odds = float(live_odds_entry.get())
        lb_pos = float(leaderboard_pos_entry.get())
        finishes = [float(e.get()) for e in finish_entries]
        # SG comp
        sg_off = float(sg_off_tee_entry.get())
        sg_app = float(sg_approach_entry.get())
        sg_putt = float(sg_putting_entry.get())
        sg_agrn = float(around_green_entry.get())
        # misc
        holes_left = int(holes_left_entry.get())
        quality = quality_var.get()
        shots_diff = float(shots_diff_entry.get())
    except ValueError:
        messagebox.showerror("Input Error","Please enter valid numbers.")
        return

    # penalty for hot‐stat
    score_penalty = imbalance_penalty(sg_off, sg_app, sg_putt, sg_agrn)

    # raw score
    avg5 = sum(finishes)/len(finishes) if finishes else 0
    pressure = (sg_true - sg_exp) * 15
    score = (
        50 + xwins
        + (total_shots+putt+t2g)*0.5
        + pressure + course_fit*COURSE_FIT_SCALE
        - ranking*0.5 - lb_pos*0.3 - avg5*0.5
        + (sg_off+sg_app+sg_putt+sg_agrn)*0.5
        - score_penalty
    )
    score -= (shots_diff/math.sqrt(max(holes_left,1))) * SB_SCALE
    score = float(np.clip(score,0,100))

    # model prob
    p_model = max(1/(1+math.exp(-a*(score-b))), P_FLOOR)

    # forward SG projection
    played = TOTAL_HOLES-holes_left
    sg_rem = sg_exp
    if played:
        t2g_h = (sg_off+sg_app+sg_agrn)/played
        putt_h = sg_putt/played
        rate_h = 0.8*t2g_h + 0.2*putt_h
        sg_rem = 0.7*sg_exp + 0.3*rate_h*holes_left

    p_sim = simulate_win_prob(shots_diff, holes_left, sg_rem,
                              FIELD_CONTENDERS.get(quality,70))
    p_final = BLEND_MODEL*p_model + (1-BLEND_MODEL)*p_sim
    p_final = min(max(p_final,P_FLOOR),0.25)

    # market math
    p_mkt = 1/live_odds
    edge = p_final-p_mkt
    fair = min(1/p_final, MAX_FAIR)
    fair_bl = FAIR_BLEND_WT*fair + (1-FAIR_BLEND_WT)*live_odds
    ev = (1-p_final) - p_final*(live_odds-1)

    row = dict(Name=name, p_model=p_final, Market_p=p_mkt,
               Edge=edge, FairOdds=fair_bl, LiveOdds=live_odds,
               EV=ev, holes_left=holes_left,
               Imbalance=score_penalty,
               sg_off=sg_off, sg_app=sg_app,
               sg_agrn=sg_agrn, sg_putt=sg_putt)
    if golfer_rows and golfer_rows[-1]["Name"]==name:
        golfer_rows[-1]=row
    else:
        golfer_rows.append(row)

    print(f"{name:12s}| Model {p_final:.2%} Edge {edge:.2%} "
          f"Imb {score_penalty:.2f}")

# rest of code unchanged (hedge_pct, compare_golfers, GUI layout, mainloop)


# ────────────────── HEDGE PERCENT FUNCTION ────────────────── #

def recommended_hedge_pct(odds, holes_left, direction):
    if odds <= 2.5:  return 0.90
    if odds <= 4.0:  return 0.60 if holes_left<27 else 0.50
    if odds <= 8.0:  return 0.60
    if odds <= 15.0: return 0.60
    if odds <= 30.0: return 0.675
    return 0.125

# ────────────────── COMPARISON POP‑UP ────────────────── #

def compare_golfers():
    if not golfer_rows:
        messagebox.showinfo("Compare", "No golfers entered yet."); return
    try:
        bankroll = float(bankroll_entry.get()); assert bankroll > 0
    except Exception:
        messagebox.showerror("Input", "Enter a positive bankroll."); return

    df = pd.DataFrame(golfer_rows)

    # lay sizing
    def _lay_liab(r):
        if -r.Edge*100 < EDGE_FLOOR_LAY: return 0.0
        return max(0.0, KELLY_FRACTION*bankroll*
                   (r.Market_p - r.p_model)/(r.LiveOdds - 1))
    df["LayLiability"] = df.apply(_lay_liab, axis=1)
    df["LayStake"]     = np.where(df.LayLiability>0,
                                  df.LayLiability/(df.LiveOdds-1), 0.0)
    df["LayHedgePct"]  = df.apply(lambda r:
        recommended_hedge_pct(r.LiveOdds,r.holes_left,"lay") if r.LayLiability>0 else 0, axis=1)
    df["LayHedgeStake"] = df.LayLiability * df.LayHedgePct
    df["LayHedgePctDisplay"] = df.LayHedgePct.apply(lambda x:f"{int(x*100)}%")

    df["RankProb"] = df.p_model.rank(ascending=False, method="min")
    df["RankEdge"] = df.Edge.rank(ascending=False, method="min")
    df.sort_values("Edge", ascending=False, inplace=True)

    # --- pop‑up ---
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

    top = tk.Toplevel(root); top.title("Odds Apex – Golf (Lay-Only)")
    tree = ttk.Treeview(top, columns=cols, show="headings",
                        height=min(len(df)+1, 30))
    vsb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(top, orient="horizontal", command=tree.xview)
    tree.configure(yscroll=vsb.set, xscroll=hsb.set)
    tree.grid(row=0,column=0,sticky="nsew"); vsb.grid(row=0,column=1,sticky="ns")
    hsb.grid(row=1,column=0,sticky="ew");   top.grid_rowconfigure(0,weight=1)
    top.grid_columnconfigure(0,weight=1)

    for c in cols:
        tree.heading(c, text=headers.get(c,c))
        tree.column(c, width=140 if c in (
            "Name","LayStake","LayLiability","LayHedgeStake") else 118, anchor="center")

    for _, r in df.iterrows():
        tree.insert("", tk.END, values=[
            r.Name,
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
            int(r.RankEdge)
        ])

def clear_list():
    golfer_rows.clear()
    messagebox.showinfo("Reset", "Stored list cleared.")

# ───────────────────── GUI ───────────────────── #

root = tk.Tk(); root.title("Odds Apex - Golf (Lay-Only)")

fields = ["Golfer Name","Expected Wins (xwins)","Total Shots Gained","Putt",
          "T2G","SG True","SG Expected","Course Fit","Current Ranking",
          "Live Odds","Leaderboard Position","Shots +/-"]
entries={}
for i, lbl in enumerate(fields):
    tk.Label(root, text=lbl).grid(row=i,column=0,sticky="e",padx=4,pady=2)
    e = tk.Entry(root); e.grid(row=i,column=1,padx=4,pady=2)
    entries[lbl] = e
(name_entry,xwins_entry,total_shots_entry,putt_entry,t2g_entry,
 sg_true_entry,sg_expected_entry,course_fit_entry,ranking_entry,
 live_odds_entry,leaderboard_pos_entry,shots_diff_entry) = [entries[l] for l in fields]

# last‑5 finishes
last5_row=len(fields)
tk.Label(root,text="Last 5 Finishes").grid(row=last5_row,column=0,sticky="e")
finish_frame=tk.Frame(root); finish_frame.grid(row=last5_row,column=1)
finish_entries=[]
for j in range(5):
    e=tk.Entry(finish_frame,width=4); e.grid(row=0,column=j,padx=2)
    finish_entries.append(e)

# SG inputs
sg_inputs=["SG Off Tee","SG Approach","SG Putting","SG Around the Green"]
for k,lbl in enumerate(sg_inputs,start=last5_row+1):
    tk.Label(root,text=lbl).grid(row=k,column=0,sticky="e",padx=4,pady=2)
    e=tk.Entry(root); e.grid(row=k,column=1,padx=4,pady=2)
    if lbl=="SG Off Tee": sg_off_tee_entry=e
    elif lbl=="SG Approach": sg_approach_entry=e
    elif lbl=="SG Putting": sg_putting_entry=e
    else: around_green_entry=e

new_row=last5_row+1+len(sg_inputs)
tk.Label(root,text="Holes Remaining").grid(row=new_row,column=0,sticky="e")
holes_left_entry=tk.Entry(root); holes_left_entry.grid(row=new_row,column=1)

tk.Label(root,text="Contenders").grid(row=new_row+1,column=0,sticky="e")
n_contenders_entry=tk.Entry(root); n_contenders_entry.grid(row=new_row+1,column=1)

tk.Label(root,text="Field Quality").grid(row=new_row+2,column=0,sticky="e")
quality_var=tk.StringVar(root); quality_var.set("average")
quality_menu = tk.OptionMenu(root,quality_var,"weak","average","strong")
quality_menu.grid(row=new_row+2,column=1,padx=4,pady=2)

tk.Label(root,text="Bankroll (£)").grid(row=new_row+3,column=0,sticky="e")
bankroll_entry=tk.Entry(root); bankroll_entry.grid(row=new_row+3,column=1)

tk.Button(root,text="Calculate Score & EV",command=calculate_score)\
    .grid(row=new_row+4,column=0,columnspan=2,pady=6)
tk.Button(root,text="Compare Entered Golfers",command=compare_golfers)\
    .grid(row=new_row+5,column=0,columnspan=2,pady=4)
tk.Button(root,text="Clear List",command=clear_list)\
    .grid(row=new_row+6,column=0,columnspan=2,pady=2)

root.mainloop()
