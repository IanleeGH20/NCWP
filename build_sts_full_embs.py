"""
STS 전체 데이터(train+valid+test = 15,487 unique sentences) 임베딩 생성
Qwen-4B 기준. 캐시에 저장.
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

STS = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark"
CACHE_DIR = "/workspace/NCWP/sts_ablation_cache"
HF_NAME = "Qwen/Qwen1.5-4B"
DEVICE = torch.device("cuda")
os.makedirs(CACHE_DIR, exist_ok=True)

# 전체 unique sentences 수집
all_sents = []
seen = set()
for split in ["train", "valid", "test"]:
    with open(f"{STS}/sts_{split}.json") as f:
        data = json.load(f)
    for item in data:
        for s in (item['sentence1'], item['sentence2']):
            if s not in seen:
                seen.add(s); all_sents.append(s)
print(f"Total unique sentences: {len(all_sents)}")

cache_emb  = os.path.join(CACHE_DIR, "qwen-4b_full_embs.npy")
cache_idx  = os.path.join(CACHE_DIR, "qwen-4b_full_sents.json")

# 이미 있으면 스킵
if os.path.exists(cache_emb) and os.path.exists(cache_idx):
    with open(cache_idx) as f:
        saved = json.load(f)
    if len(saved) == len(all_sents):
        print("이미 캐시됨"); exit(0)

# 기존 ablation 캐시 (5385개) 재사용 가능한지 확인
existing = os.path.join(CACHE_DIR, "qwen-4b_all_embs.npy")
existing_idx = os.path.join(CACHE_DIR, "qwen-4b_sent_order.json")
if os.path.exists(existing) and os.path.exists(existing_idx):
    print("기존 5385개 캐시 재사용 + 차이만 추가 인코딩")
    with open(existing_idx) as f:
        old_sents = json.load(f)
    old_embs = np.load(existing)
    old_map = {s: e for s, e in zip(old_sents, old_embs)}
    new_sents = [s for s in all_sents if s not in old_map]
    print(f"  기존 활용: {len(all_sents) - len(new_sents)} / 추가 인코딩 필요: {len(new_sents)}")
else:
    new_sents = all_sents
    old_map = {}

if len(new_sents) > 0:
    print(f"Qwen-4B 로드 + 신규 {len(new_sents)}개 인코딩...")
    tok = AutoTokenizer.from_pretrained(HF_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {"trust_remote_code": True,
          "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    n_gpu = torch.cuda.device_count()
    model = (torch.nn.DataParallel(AutoModel.from_pretrained(HF_NAME, **kw)).cuda()
             if n_gpu > 1 else AutoModel.from_pretrained(HF_NAME, **kw).to(DEVICE))
    model.eval()
    parts = []
    for i in range(0, len(new_sents), 64):
        batch = new_sents[i:i+64]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc) if n_gpu <= 1 else model.module(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        parts.append(F.normalize(emb.float(), dim=-1).cpu().numpy())
        if (i // 64) % 10 == 0: print(f"  {i+len(batch)}/{len(new_sents)}", end="\r")
    print()
    new_embs = np.vstack(parts)
    for s, e in zip(new_sents, new_embs):
        old_map[s] = e

# 전체 순서로 정렬
full_embs = np.array([old_map[s] for s in all_sents]).astype(np.float32)
np.save(cache_emb, full_embs)
with open(cache_idx, "w") as f:
    json.dump(all_sents, f)
print(f"저장 완료: {cache_emb}  shape={full_embs.shape}")
