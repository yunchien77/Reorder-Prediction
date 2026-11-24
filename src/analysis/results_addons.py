"""Adds the sequence models (and the product-segment study of the extra features) to RESULTS.md, without touching the other sections.
  - sections 1 / 2 : rows of the sequence models are (re)inserted before the Ensemble row
  - section 11     : sequence models vs XGBoost (paired per fold) and with / without the 54 tabular features
  - section 12     : AUPRC difference per product / user segment of the external-data and embedding features (if results/segments_ext.csv exists)
  cd src && python -m analysis.results_addons"""
import re

import numpy as np
import pandas as pd

import common

MD = common.ROOT / "RESULTS.md"
R = pd.read_csv(common.RESULTS / "results.csv")
NT = {k: int((pd.read_csv(common.SPLITS / f"fold{k}.csv").split == "test").sum()) for k in range(1, 6)}
for k in ("f1_order", "recall_at_10", "ndcg_at_10"):
    R[k] = R[k] * R.users_eval / R.fold.map(NT)                  # all-user convention (users without a positive count as 0)

SEQ = {"lstm_hybrid": "LSTM + 54 個特徵", "gru_hybrid": "GRU + 54 個特徵", "transformer_hybrid": "Transformer + 54 個特徵",
       "lstm_seq": "LSTM（只有序列）", "gru_seq": "GRU（只有序列）", "transformer_seq": "Transformer（只有序列）",
       "ens_seq_mean": "Ensemble：XGB + LGBM + CatBoost + MLP + LSTM + GRU + Transformer（機率平均）"}
K7 = ["f1", "f1_order", "auprc", "precision", "recall", "recall_at_10", "ndcg_at_10"]


def rows(m):
    return R[R.model == m].sort_values("fold")


def ms(x, k):
    return f"{x[k].mean():.4f} ± {x[k].std():.4f}"


def table(header, body):
    return header + "\n|" + "---|" * (header.count("|") - 1) + "\n" + "\n".join(body)


have = [m for m in SEQ if len(rows(m))]
text = MD.read_text(encoding="utf-8")
text = text.split("\n## 11. ")[0].rstrip() + "\n"                  # drop an older version of the add-on sections
lines = text.split("\n")
names = set(SEQ.values())
lines = [l for l in lines if not (l.startswith("| ") and l.split(" | ")[0][2:] in names)]      # drop older inserted rows
out, sec, metric = [], 0, None
for l in lines:
    if l.startswith("## "):
        sec = int(re.match(r"## (\d+)\.", l).group(1)); metric = None
    if l.startswith("**"):
        metric = {"**AUPRC**": "auprc", "**F1**": "f1", "**F1 (order)**": "f1_order"}.get(l.strip())
    is_ens = l.startswith("| Ensemble：XGB + LGBM + CatBoost + MLP（機率平均）")
    if is_ens and sec == 1:
        for m in have:
            x = rows(m); out.append(f"| {SEQ[m]} | " + " | ".join(ms(x, k) for k in K7) + f" | {len(x)} |")
    if is_ens and sec == 2 and metric:
        for m in have:
            x = rows(m); out.append(f"| {SEQ[m]} | " + " | ".join(f"{v:.4f}" for v in x[metric]) + f" | {x[metric].mean():.4f} |")
    out.append(l)
text = "\n".join(out).rstrip() + "\n"

# ---------------------------------------------------------------- 11 sequence models
x0 = rows("xgb")
if have:
    sec = ["", "## 11. 序列模型：LSTM / GRU / Transformer（5-fold）", "",
           "每個（使用者, 商品）配對取使用者最近 24 張 prior 訂單：商品是否出現、在購物車的相對位置，加上該訂單的間隔天數、星期、時段、購物車大小（不含目標訂單）。"
           "`+ 54 個特徵`：序列編碼器的輸出與 54 個表格特徵（MLP 分支）合併後預測；`只有序列`：不給 54 個特徵，由網路自己從序列學。"
           "差值 = 與 XGBoost（54 個特徵）逐折相減。", ""]
    body = []
    for m in have:
        x = rows(m); d = x.auprc.values - x0[x0.fold.isin(x.fold)].auprc.values
        body.append(f"| {SEQ[m]} | {ms(x, 'auprc')} | {ms(x, 'f1')} | {ms(x, 'f1_order')} | {d.mean():+.4f}（{int((d > 0).sum())}/{len(d)} 折比 XGB 好） | {len(x)} |")
    sec += [table("| 模型 | AUPRC | F1 | F1 (order) | AUPRC 差（對 XGBoost） | fold 數 |", body), "", "**AUPRC 差（對 XGBoost，逐折）**", ""]
    body = []
    for m in have:
        x = rows(m); d = x.auprc.values - x0[x0.fold.isin(x.fold)].auprc.values
        cells = {int(f): v for f, v in zip(x.fold, d)}
        body.append(f"| {SEQ[m]} | " + " | ".join(f"{cells[k]:+.4f}" if k in cells else "—" for k in range(1, 6)) + " |")
    sec += [table("| 模型 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |", body), ""]
    text += "\n".join(sec)

# ---------------------------------------------------------------- 12 segments of the extra features
seg_file = common.RESULTS / "segments_ext.csv"
if seg_file.exists():
    S = pd.read_csv(seg_file)
    g = S.groupby("seg", sort=False).agg(share=("share", "mean"), pos=("pos", "mean"), base=("base", "mean"),
                                         ext=("ext", "mean"), ext_pos=("ext", lambda v: int((v > 0).sum())),
                                         emb=("emb", "mean"), emb_pos=("emb", lambda v: int((v > 0).sum())), nf=("fold", "count"))
    sec = ["", "## 12. 分群：外部資料與向量特徵對哪些分群有幫助？（AUPRC 差，5-fold）", "",
           "同一折、同一 seed 的 XGBoost：`外部資料` = 54 個 + FoodKeeper + FDC + OFF（門檻 0.9）；`向量特徵` = 54 個 + 訂單向量與文字向量的相似度、商品向量（共 94 個）。"
           "分群 AUPRC 只用該群的配對計算；差 = 與只有 54 個的模型相減，括號 = 5 折中變好的折數。", ""]
    cur, body = None, []
    for name, r in g.iterrows():
        grp, label = name.split("|", 1)
        if grp != cur:
            if body:
                sec += [table("| 分群 | 占配對 | 正類率 | 54 個的 AUPRC | 外部資料 差 | 向量特徵 差 |", body), ""]
            sec += [f"**{grp}**", ""]; cur, body = grp, []
        body.append(f"| {label} | {r.share * 100:.1f}% | {r.pos * 100:.1f}% | {r.base:.4f} | {r.ext:+.4f}（{int(r.ext_pos)}/{int(r.nf)}） | {r.emb:+.4f}（{int(r.emb_pos)}/{int(r.nf)}） |")
    sec += [table("| 分群 | 占配對 | 正類率 | 54 個的 AUPRC | 外部資料 差 | 向量特徵 差 |", body), ""]
    text += "\n".join(sec)

MD.write_text(text.rstrip() + "\n", encoding="utf-8")
print("RESULTS.md updated; sequence models:", have, "| segments:", seg_file.exists())
