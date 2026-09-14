"""
STS에서 Base는 못 잡고 NCWP만 잡은 예시 추출.
- Base가 cosine sim을 매우 다르게 예측 vs gold score
- NCWP가 gold에 훨씬 가깝게 예측한 pair
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

STS_DIR    = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark"
CACHE_DIR  = "/workspace/NCWP/sts_ablation_cache"
WEIGHT_DIR = "/workspace/RAG/code/Make_embedding/nanoGPT/ncwp_extended_results"
DEVICE     = torch.device("cuda")

# 데이터
test_data = json.load(open(f"{STS_DIR}/sts_test.json"))
pairs  = [(item['sentence1'], item['sentence2']) for item in test_data]
scores = np.array([item['score'] for item in test_data])  # 0~5

# 임베딩 (qwen-4b)
embs_full = np.load(f"{CACHE_DIR}/qwen-4b_full_embs.npy").astype(np.float32)
sents_full = json.load(open(f"{CACHE_DIR}/qwen-4b_full_sents.json"))
s2i = {s: i for i, s in enumerate(sents_full)}

def normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

# Base 임베딩 (이미 normalize 되어있을 수 있지만 한 번 더)
all_eval_sents = list({s for s1, s2 in pairs for s in (s1, s2)})
base_embs = {s: embs_full[s2i[s]] / (np.linalg.norm(embs_full[s2i[s]]) + 1e-12)
             for s in all_eval_sents}

# NCWP weight (qwen-4b r=320)
ncwp = np.load(f"{WEIGHT_DIR}/qwen-4b_ncwp_dim320.npz")
W = ncwp['W'].astype(np.float32)
mu_in = ncwp['mu_in'].astype(np.float32)

# NCWP projection
def project(x):
    z = (x - mu_in) @ W
    return z / (np.linalg.norm(z) + 1e-12)

ncwp_embs = {s: project(embs_full[s2i[s]]) for s in all_eval_sents}

# 각 pair마다 cosine 계산
data = []
for (s1, s2), gold in zip(pairs, scores):
    base_cos = float(np.dot(base_embs[s1], base_embs[s2]))
    ncwp_cos = float(np.dot(ncwp_embs[s1], ncwp_embs[s2]))
    data.append({
        's1': s1, 's2': s2, 'gold': gold,
        'base_cos': base_cos, 'ncwp_cos': ncwp_cos,
    })
df = pd.DataFrame(data)

# Z-score: gold 5점 만점 → 0~1로, cosine -1~1 → 0~1
# 또는 단순히 NCWP - Base 차이가 큰 경우 분석
df['base_norm'] = (df['base_cos'] - df['base_cos'].min()) / (df['base_cos'].max() - df['base_cos'].min())
df['ncwp_norm'] = (df['ncwp_cos'] - df['ncwp_cos'].min()) / (df['ncwp_cos'].max() - df['ncwp_cos'].min())
df['gold_norm'] = df['gold'] / 5.0

df['base_err'] = (df['base_norm'] - df['gold_norm']).abs()
df['ncwp_err'] = (df['ncwp_norm'] - df['gold_norm']).abs()
df['improvement'] = df['base_err'] - df['ncwp_err']

# 분류:
# Type A: gold 높은데 Base 낮음, NCWP 잘 잡음
type_a = df[(df['gold'] >= 4.0) & (df['base_norm'] < 0.5)].copy()
type_a = type_a.sort_values('improvement', ascending=False).head(5)

# Type B: gold 낮은데 Base 너무 높음, NCWP가 낮춤
type_b = df[(df['gold'] <= 1.5) & (df['base_norm'] > 0.6)].copy()
type_b = type_b.sort_values('improvement', ascending=False).head(5)

print("="*80)
print("Type A — Gold 높음(≥4.0)인데 Base는 낮게 예측 → NCWP가 잡아낸 case")
print("="*80)
for i, r in type_a.iterrows():
    print(f"\nGold={r['gold']:.1f}")
    print(f"  s1: {r['s1']}")
    print(f"  s2: {r['s2']}")
    print(f"  Base cos={r['base_cos']:.3f}(norm {r['base_norm']:.2f}) | NCWP cos={r['ncwp_cos']:.3f}(norm {r['ncwp_norm']:.2f})")
    print(f"  Improvement: {r['improvement']:.3f}")

print("\n" + "="*80)
print("Type B — Gold 낮음(≤1.5)인데 Base는 너무 높게 예측 → NCWP가 잡아낸 case")
print("="*80)
for i, r in type_b.iterrows():
    print(f"\nGold={r['gold']:.1f}")
    print(f"  s1: {r['s1']}")
    print(f"  s2: {r['s2']}")
    print(f"  Base cos={r['base_cos']:.3f}(norm {r['base_norm']:.2f}) | NCWP cos={r['ncwp_cos']:.3f}(norm {r['ncwp_norm']:.2f})")
    print(f"  Improvement: {r['improvement']:.3f}")

# 저장
top = pd.concat([
    type_a.assign(type='A: gold high, base low'),
    type_b.assign(type='B: gold low, base high'),
])
top.to_csv('/workspace/NCWP/sts_cherrypicks_qwen4b.csv', index=False)
print(f"\n저장: /workspace/NCWP/sts_cherrypicks_qwen4b.csv")
