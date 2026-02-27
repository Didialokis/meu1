def build_intrasentence(df_intra: pd.DataFrame, gold_intra_by_key: Dict[Tuple[str, str, tuple], dict]) -> List[dict]:
    # 1. Pré-calcula as colunas base ANTES do loop (Evita buscas repetidas)
    c_bias = find_col(df_intra, "Viés", "Vies", "bias_type", "bias")
    c_target = find_col(df_intra, "Target", "target")
    c_target_pt = find_col(df_intra, "Target_PT", "target_pt")
    c_ctx_pt = find_col(df_intra, "Contexto_PT", "Contexto PT", "context_pt", "context")

    # 2. Pré-calcula as colunas das 3 sentenças ANTES do loop
    sent_cols = []
    for i in (1, 2, 3):
        c_lab = find_col(df_intra, f"Frase_{i}_Label")
        c_en = find_col(df_intra, f"Frase_{i}_EN")
        c_pt = find_col(df_intra, f"Frase_{i}_PT")
        sent_cols.append((c_lab, c_en, c_pt))

    out = []
    # 3. Loop principal de iteração nas linhas
    for _, r in df_intra.iterrows():
        row = r.to_dict()

        bias = norm(row.get(c_bias)) if c_bias else ""
        target = norm(row.get(c_target)) if c_target else ""
        target_pt = norm(row.get(c_target_pt)) if c_target_pt else ""
        context_pt = norm(row.get(c_ctx_pt)) if c_ctx_pt else ""

        s_en, s_pt, gls = [], [], []
        
        # 4. Extração das frases usando as colunas já mapeadas
        for c_lab, c_en, c_pt in sent_cols:
            gl = normalize_gold_label(row.get(c_lab)) if c_lab else ""
            en = norm(row.get(c_en)) if c_en else ""
            pt = norm(row.get(c_pt)) if c_pt else ""
            
            gls.append(gl)
            s_en.append(en)
            s_pt.append(pt)

        # 5. Validações originais (mantidas intactas)
        if not (bias and target and context_pt):
            continue
        if any(g not in ALLOWED_GOLD for g in gls):
            continue
        if any(not t for t in s_pt):
            continue

        # 6. Busca no Gold JSON original
        key = (bias, target, tuple(sorted([norm_key(x) for x in s_en])))
        gold_ex = gold_intra_by_key.get(key)

        ex_id = gold_ex.get("id") if gold_ex and gold_ex.get("id") else stable_id(
            "intrasentence", bias, target, context_pt, "||".join(sorted(s_pt))
        )

        # 7. Construção do objeto principal
        ex_obj = {
            "id": ex_id,
            "bias_type": bias,
            "target": target,
            "Target_PT": target_pt,
            "context": context_pt,
            "sentences": []
        }

        # 8. Popula as 3 sentenças no objeto final
        for i in range(3):
            gl = gls[i]
            pt = s_pt[i]
            
            gold_sent = find_sentence_in_gold(gold_ex, gl) if gold_ex else None
            sent_id = gold_sent.get("id") if gold_sent else stable_id(ex_id, f"{i+1}", gl, pt)
            
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
