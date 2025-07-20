# Odds Apex – Live Golf Lay Model

Small, fast Tkinter app that spots **over‑priced golfers** in real‑time and sizes
lays with quarter‑Kelly, so you never blow the bankroll.

<img width="1710" height="654" alt="image" src="https://github.com/user-attachments/assets/f186f955-defc-449f-9b13-7de7efa22abb" />


---

## Features

- **One‑click fair odds**  
  - Logistic form model + 5 000‑run Monte‑Carlo sim re‑run on demand  
  - Handles **negative** *Shots Behind / Ahead* (‑2 = 2‑shot leader)

- **Edge & Kelly stakes**  
  - Shows model %, market %, edge % (pp)  
  - ¼‑Kelly lay stake + liability, auto‑scaled so portfolio liability ≤ 8 % roll  
  - Rank by probability or by edge

- **Multi‑golfer compare**  
  - Enter any number of players, pop up a sortable table  
  - Colour‑code “Edge %” to see value at a glance

- **Minimal inputs mid‑round**  
  - Update only *Shots Behind / Ahead*, *Holes Remaining*, *Live Odds*  
  - Everything else stays from pre‑round

- **Light refresh logic**  
  - Event triggers: score swing ≥ 1, lead change, every 3 holes, 27/18/9 left

---

## Edge filters (lay‑only)

| Lay odds | Minimum edge (pp) |
|----------|-------------------|
| **1.5 – 3.0** | 2 (Thu–Fri) / 3 (Sat–Sun) |
| **3.0 – 8.0** | 3 / 4 |
| **8.0 – 15.0** | 5 |
| **15.0 – 30.0** | 8 + decent queue depth |
| **30.0 – 50.0** | 12 (pp) and micro stake only |

*(pp = percentage‑points difference model % – market %)*

---

## Installation

```bash
git clone https://github.com/your‑handle/odds‑apex.git
cd odds‑apex
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas
python odds_apex.py                # run the GUI
