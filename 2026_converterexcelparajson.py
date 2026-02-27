# -*- coding: utf-8 -*-

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# =====================================================================
# CONSTANTES E CONFIGURAÇÕES
# =====================================================================
BASE_DIR = Path.cwd()

XLSX_INTER = BASE_DIR / "renato review amostra_stereoset (Intersentence).xlsx"
XLSX_INTRA = BASE_DIR / "amostra_stereoset (instrasentence) - final (2).xlsx"
JSON_GOLD = BASE_DIR / "dev.json"

OUT_MERGED = BASE_DIR / "stereoset_reconstruido_merged.json"

ALLOWED_GOLD = {"stereotype", "anti-stereotype", "unrelated"}
ALLOWED_LABELS = {"stereotype", "anti-stereotype", "unrelated", "related"}

NUM_GOLD_MAP = {
    "0": "anti-stereotype",
    "1": "stereotype",
    "2": "unrelated",
    0: "anti-stereotype",
    1: "stereotype",
    2: "unrelated",
}

WORD_CHAR_RE = re.compile(r"\w", re.UNICODE)

# =====================================================================
# FUNÇÕES UTILITÁRIAS E DE NORMALIZAÇÃO
# =====================================================================
def norm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm_key(x: Any) -> str:
    return norm(x).lower()

def stable_id(*parts: str) -> str:
    return hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()

def normalize_gold_label(gl: Any) -> str:
    gl = NUM_GOLD_MAP.get(gl, gl)
    return norm(gl)

def ensure_nonempty_labels(labels: Any, gold_label: str) -> List[Dict[str, str]]:
    if isinstance(labels, list) and labels:
        return labels
        
    gl = norm(gold_label) or "unrelated"
    if gl not in ALLOWED_LABELS:
        gl = "unrelated"
    return [{"human_id": f"synthetic_{i}", "label": gl} for i in range(1, 6)]

def load_xlsx_all_sheets(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, engine="openpyxl")
    frames = []
    for sheet_name, sdf in sheets.items():
        sdf = sdf.fillna("")
        sdf.columns = [str(c).strip() for c in sdf.columns]
        sdf["_sheet"] = sheet_name
        frames.append(sdf)
    return pd.concat(frames, ignore_index=True)

def find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    cols = list(df.columns)
    cols_l = {c.lower(): c for c in cols}

    def canon(x: str) -> str:
        return re.sub(r"[\s_]+", "", x.lower())

    cols_c = {canon(c): c for c in cols}

    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
        cc = canon(cand)
        if cc in cols_c:
            return cols_c[cc]
    return None

def lcp(strings: List[str]) -> str:
    if not strings:
        return ""
    s1 = min(strings)
    s2 = max(strings)
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]

def lcs(strings: List[str]) -> str:
    if not strings:
        return ""
    return lcp([s[::-1] for s in strings])[::-1]

def trim_prefix_to_boundary(prefix: str) -> str:
    if not prefix:
        return prefix
    if WORD_CHAR_RE.search(prefix[-1]):
        last_boundary = None
        for i in range(len(prefix) - 1, -1, -1):
            if not WORD_CHAR_RE.search(prefix[i]):
                last_boundary = i
                break
        return prefix[: last_boundary + 1] if last_boundary is not None else ""
    return prefix

def trim_suffix_to_boundary(suffix: str) -> str:
    if not suffix:
        return suffix
    if WORD_CHAR_RE.search(suffix[0]):
        first_boundary = None
        for i, ch in enumerate(suffix):
            if not WORD_CHAR_RE.search(ch):
                first_boundary = i
                break
        return suffix[first_boundary:] if first_boundary is not None else ""
    return suffix

def resolve_overlap(prefix: str, suffix: str, min_len: int) -> str:
    if len(prefix) + len(suffix) > min_len:
        overlap = len(prefix) + len(suffix) - min_len
        if 0 < overlap <= len(suffix):
            return suffix[overlap:]
    return suffix

def normalize_context_from_sentences(sentences: List[str]) -> str:
    sents = [norm(s) for s in sentences if norm(s)]
    if not sents:
        return "BLANK"

    prefix = lcp(sents)
    suffix = lcs(sents)
    min_len = min(len(s) for s in sents)

    suffix = resolve_overlap(prefix, suffix, min_len)
    
    prefix = trim_prefix_to_boundary(prefix)
    suffix = trim_suffix_to_boundary(suffix)

    min_len = min(len(s) for s in sents)
    suffix = resolve_overlap(prefix, suffix, min_len)

    ctx = f"{prefix}BLANK{suffix}"
    ctx = re.sub(r"\s+", " ", ctx).strip()
    ctx = re.sub(r"(?<=\w)BLANK", " BLANK", ctx)
    ctx = re.sub(r"BLANK(?=\w)", "BLANK ", ctx)
    ctx = re.sub(r"\s+", " ", ctx).strip()
    return ctx if ctx else "BLANK"

# =====================================================================
# CARREGAMENTO E INDEXAÇÃO DO GOLD JSON
# =====================================================================
def load_gold_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def build_gold_indexes(gold: dict):
    inter_by_id: Dict[str, dict] = {}
    intra_by_key: Dict[Tuple[str, str, tuple], dict] = {}

    for ex in gold.get("data", {}).get("intersentence", []):
        ex_id = ex.get("id")
        if ex_id:
            inter_by_id[ex_id] = ex

    for ex in gold.get("data", {}).get("intrasentence", []):
        bt = ex.get("bias_type")
        tgt = ex.get("target")
        sents = [norm_key(s.get("sentence")) for s in ex.get("sentences", [])]
        if bt and tgt and len(sents) == 3:
            intra_by_key[(bt, tgt, tuple(sorted(sents)))] = ex

    return inter_by_id, intra_by_key

def find_sentence_in_gold(gold_ex: Optional[dict], gold_label: str) -> Optional[dict]:
    if not gold_ex:
        return None
    gl = norm(gold_label)
    for s in gold_ex.get("sentences", []):
        if norm(s.get("gold_label")) == gl:
            return s
    return None

# =====================================================================
# CONSTRUÇÃO DE DATASETS
# =====================================================================
def build_intersentence(df_inter: pd.DataFrame, gold_inter_by_id: Dict[str, dict]) -> List[dict]:
    c_id = find_col(df_inter, "ID", "id")
    c_task = find_col(df_inter, "Tarefa", "Task")
    c_bias = find_col(df_inter, "Viés", "Vies", "bias_type", "bias")
    c_target = find_col(df_inter, "Target", "target")
    c_target_pt = find_col(df_inter, "Target_PT", "trad_target", "trad target", "target_pt")
    c_ctx_pt = find_col(df_inter, "Contexto_PT", "Contexto PT", "context_pt", "context")

    sent_cols = []
    for i in (1, 2, 3):
        c_lab = find_col(df_inter, f"Frase_{i}_Label", f"Sentence_{i}_Label")
        c_pt = find_col(df_inter, f"Frase_{i}_PT", f"Sentence_{i}_PT", f"Sentence_{i}")
        sent_cols.append((c_lab, c_pt, i))

    out = []
    for _, r in df_inter.iterrows():
        row = r.to_dict()

        task = norm(row.get(c_task)) if c_task else ""
        if task and task.lower() != "intersentence":
            continue

        ex_id = norm(row.get(c_id)) if c_id else ""
        bias = norm(row.get(c_bias)) if c_bias else ""
        target = norm(row.get(c_target)) if c_target else ""
        target_pt = norm(row.get(c_target_pt)) if c_target_pt else ""
        context = norm(row.get(c_ctx_pt)) if c_ctx_pt else ""

        gold_ex = gold_inter_by_id.get(ex_id) if ex_id else None
        if not ex_id:
            ex_id = stable_id("intersentence", bias, target, context)

        ex_obj = {
            "id": ex_id,
            "bias_type": bias,
            "target": target,
            "Target_PT": target_pt,
            "context": context,
            "sentences": [],
        }

        for c_lab, c_pt, i in sent_cols:
            gl = normalize_gold_label(row.get(c_lab)) if c_lab else ""
            sent_pt = norm(row.get(c_pt)) if c_pt else ""
            
            if not sent_pt:
                continue

            gold_sent = find_sentence_in_gold(gold_ex, gl)
            sent_id = gold_sent.get("id") if gold_sent else stable_id(ex_id, f"{i}", gl, sent_pt)
            labels = gold_sent.get("labels", []) if gold_sent else []
            labels = ensure_nonempty_labels(labels, gl)

            ex_obj["sentences"].append({
                "id": sent_id,
                "sentence": sent_pt,
                "labels": labels,
                "gold_label": gl
            })

        out.append(ex_obj)
        
    return out

def build_intrasentence(df_intra: pd.DataFrame, gold_intra_by_key: Dict[Tuple[str, str, tuple], dict]) -> List[dict]:
    c_bias = find_col(df_intra, "Viés", "Vies", "bias_type", "bias")
    c_target = find_col(df_intra, "Target", "target")
    c_target_pt = find_col(df_intra, "Target_PT", "target_pt")
    c_ctx_pt = find_col(df_intra, "Contexto_PT", "Contexto PT", "context_pt", "context")

    sent_cols = []
    for i in (1, 2, 3):
        c_lab = find_col(df_intra, f"Frase_{i}_Label")
        c_en = find_col(df_intra, f"Frase_{i}_EN")
        c_pt = find_col(df_intra, f"Frase_{i}_PT")
        sent_cols.append((c_lab, c_en, c_pt, i))

    out = []
    for _, r in df_intra.iterrows():
        row = r.to_dict()

        bias = norm(row.get(c_bias)) if c_bias else ""
        target = norm(row.get(c_target)) if c_target else ""
        target_pt = norm(row.get(c_target_pt)) if c_target_pt else ""
        context_pt = norm(row.get(c_ctx_pt)) if c_ctx_pt else ""

        s_en, s_pt, gls = [], [], []
        
        for c_lab, c_en, c_pt, _ in sent_cols:
            gl = normalize_gold_label(row.get(c_lab)) if c_lab else ""
            en = norm(row.get(c_en)) if c_en else ""
            pt = norm(row.get(c_pt)) if c_pt else ""
            
            gls.append(gl)
            s_en.append(en)
            s_pt.append(pt)

        if not (bias and target and context_pt):
            continue
            
        if any(g not in ALLOWED_GOLD for g in gls):
            continue
            
        if any(not t for t in s_pt):
            continue

        key = (bias, target, tuple(sorted([norm_key(x) for x in s_en])))
        gold_ex = gold_intra_by_key.get(key)

        ex_id = gold_ex.get("id") if gold_ex and gold_ex.get("id") else stable_id(
            "intrasentence", bias, target, context_pt, "||".join(sorted(s_pt))
        )

        ex_obj = {
            "id": ex_id,
            "bias_type": bias,
            "target": target,
            "Target_PT": target_pt,
            "context": context_pt,
            "sentences": [],
        }

        for idx, (gl, pt) in enumerate(zip(gls, s_pt)):
            i = sent_cols[idx][3]
            
            gold_sent = find_sentence_in_gold(gold_ex, gl) if gold_ex else None
            sent_id = gold_sent.get("id") if gold_sent else stable_id(ex_id, f"{i}", gl, pt)
            labels = gold_sent.get("labels", []) if gold_sent else []
            labels = ensure_nonempty_labels(labels, gl)

            ex_obj["sentences"].append({
                "id": sent_id,
                "sentence": pt,
                "labels": labels,
                "gold_label": gl
            })

        out.append(ex_obj)
        
    return out

# =====================================================================
# FILTRO E NORMALIZAÇÃO FINAL
# =====================================================================
def filter_and_normalize(intersentence: List[dict], intrasentence: List[dict]):
    removed = {"intersentence": 0, "intrasentence": 0}

    kept_inter = []
    for ex in intersentence:
        sents = ex.get("sentences", [])
        golds = {s.get("gold_label") for s in sents}
        if len(sents) != 3 or golds != ALLOWED_GOLD:
            removed["intersentence"] += 1
            continue
        kept_inter.append(ex)

    kept_intra = []
    for ex in intrasentence:
        sents = ex.get("sentences", [])
        golds = {s.get("gold_label") for s in sents}
        if len(sents) != 3 or golds != ALLOWED_GOLD:
            removed["intrasentence"] += 1
            continue

        if any("BLANK" in (s.get("sentence", "") or "") for s in sents):
            removed["intrasentence"] += 1
            continue

        ex["context"] = normalize_context_from_sentences([s["sentence"] for s in sents])
        ctx = ex.get("context", "")
        
        if ctx.count("BLANK") != 1 or re.search(r"\wBLANK|BLANK\w", ctx):
            removed["intrasentence"] += 1
            continue
            
        kept_intra.append(ex)

    return kept_inter, kept_intra, removed

# =====================================================================
# MAIN
# =====================================================================
def main():
    for p in (XLSX_INTER, XLSX_INTRA, JSON_GOLD):
        if not p.exists():
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado (Strict): {p.name}")

    df_inter = load_xlsx_all_sheets(XLSX_INTER)
    df_intra = load_xlsx_all_sheets(XLSX_INTRA)

    gold = load_gold_json(JSON_GOLD)
    version = gold.get("version", "1.1")
    gold_inter_by_id, gold_intra_by_key = build_gold_indexes(gold)

    intersentence = build_intersentence(df_inter, gold_inter_by_id)
    intrasentence = build_intrasentence(df_intra, gold_intra_by_key)

    intersentence, intrasentence, removed = filter_and_normalize(intersentence, intrasentence)

    out = {
        "version": version,
        "data": {
            "intersentence": intersentence,
            "intrasentence": intrasentence
        }
    }
    
    OUT_MERGED.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK:", OUT_MERGED.name)
    print("intersentence:", len(intersentence), "removidos:", removed["intersentence"])
    print("intrasentence:", len(intrasentence), "removidos:", removed["intrasentence"])

if __name__ == "__main__":
    main()
