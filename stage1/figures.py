"""English plots, source tables, and evidence-backed experiment reports."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score

from .data_io import (CLASS_NAMES, CONFIG, FEATURES, FIGURES, LABELS, REPORTS, RESULTS,
                      cache_identity, file_sha, load_data, read_json, write_json)
from .embeddings import load_pretrained, train_word_vectors
from .probe import comparison_detail


def summary_table(scores):
    rows = []
    for config in CONFIG["configs"]:
        part = scores[scores.config == config]
        oof = pd.read_csv(RESULTS / f"oof_{config}.csv", encoding="utf-8")
        rows.append({"config": config,
                     "r0_oof_macro_f1": f1_score(oof.y_true, oof.y_pred, labels=LABELS, average="macro", zero_division=0),
                     "macro_f1_mean": part.macro_f1.mean(), "macro_f1_sd": part.macro_f1.std(ddof=1),
                     **{name: part[name].mean() for name in ["acc", "f1_neg", "f1_pos", "f1_irr", "f1_neu"]}})
    table = pd.DataFrame(rows)
    table["rank"] = table.macro_f1_mean.rank(method="min", ascending=False).astype(int)
    return table.sort_values("rank").reset_index(drop=True)


def save_figure(fig, name, data):
    data.to_csv(FIGURES / f"{name}.csv", index=False, encoding="utf-8")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_tsne(frame, pretrained, recompute=False, cache_only=False):
    path = FEATURES / "tsne_coordinates.csv"
    metadata_path = FEATURES / "tsne.json"
    if path.exists() and not recompute:
        metadata = read_json(metadata_path)
        assert metadata["identity"] == cache_identity()
        assert metadata["sha256"] == file_sha(path)
        return pd.read_csv(path, encoding="utf-8")
    if cache_only:
        raise FileNotFoundError("Cached visualization missing; run figures once before cache-only notebook")
    rows = []
    for config in ["w2v", "ft", "glove_stem", "bertweet"]:
        if config in ["w2v", "ft"]:
            vectors, _ = train_word_vectors(config, frame.Phrase.tolist(), np.arange(len(frame)), np.array([], dtype=int))
            np.save(FEATURES / f"visualization_only_{config}.npy", vectors)
        else:
            vectors = pretrained[config]
        coordinates = TSNE(n_components=2, perplexity=30, random_state=42, init="pca",
                           learning_rate="auto").fit_transform(vectors)
        rows.extend({"config": config, "row_index": i, "label": int(frame.Sentiment.iloc[i]),
                     "x": float(x), "y": float(y)} for i, (x, y) in enumerate(coordinates))
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False, encoding="utf-8")
    write_json(metadata_path, {"identity": cache_identity(), "sha256": file_sha(path),
               "purpose": "Visualization only; full-training word models never used by probes",
               "perplexity": 30, "random_state": 42})
    return table


def create_figures(recompute=False, cache_only=False):
    frame, _ = load_data("figures")
    scores = pd.read_csv(RESULTS / "fold_scores.csv", encoding="utf-8")
    table = summary_table(scores)
    table.to_csv(RESULTS / "summary.csv", index=False, encoding="utf-8")
    ordered = table.set_index("config").loc[CONFIG["configs"]].reset_index()
    colors = ["#4086a8", "#4caa91", "#a089c5", "#c4aad5", "#ed9860", "#efc07c", "#8a9098"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(ordered))
    ax.bar(positions, ordered.macro_f1_mean * 100, yerr=ordered.macro_f1_sd * 100,
           color=colors, capsize=4, alpha=0.85, label="25-fold mean ± SD")
    ax.scatter(positions, ordered.r0_oof_macro_f1 * 100, color="black", s=30, zorder=4, label="R0 pooled OOF")
    baseline = ordered.loc[ordered.config == "baseline_ref", "macro_f1_mean"].iloc[0] * 100
    ax.axhline(baseline, color="#555555", linestyle="--", label="Reference mean")
    ax.set(xticks=positions, xticklabels=ordered.config, ylabel="Macro-F1 (%)", title="Frozen representations with one linear probe")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, "F1_macro_f1", ordered)

    class_columns = ["f1_neg", "f1_pos", "f1_irr", "f1_neu"]
    fig, ax = plt.subplots(figsize=(8, 5))
    values = ordered[class_columns].to_numpy() * 100
    im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for i in range(7):
        for j in range(4):
            ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", color="white" if values[i,j] > 55 else "black")
    ax.set(xticks=range(4), xticklabels=CLASS_NAMES, yticks=range(7), yticklabels=ordered.config,
           title="Per-class F1: mean over 25 folds")
    fig.colorbar(im, ax=ax, label="F1 (%)")
    save_figure(fig, "F2_class_f1", ordered[["config", *class_columns]])

    coverage = pd.read_csv(RESULTS / "coverage.csv", encoding="utf-8")
    adapted = coverage.groupby("embedding", sort=False).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    lexical = adapted[adapted.embedding != "bertweet"]
    x = np.arange(len(lexical))
    axes[0].bar(x - .18, lexical.oov_token_rate * 100, .36, label="Occurrences")
    axes[0].bar(x + .18, lexical.oov_type_rate * 100, .36, label="Unique words")
    axes[0].plot(x, lexical.unseen_token_rate * 100, "k.", label="Unseen occurrences")
    axes[0].set(xticks=x, xticklabels=lexical.embedding, ylabel="Unrepresentable words (%)", title="Vocabulary coverage")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)
    b = coverage[coverage.embedding == "bertweet"].iloc[0]
    axes[1].bar(["2+ subwords"], [b.split_word_rate * 100], color="#ed9860")
    axes[1].set(ylabel="Word occurrences (%)", title=f"BERTweet: {b.subwords_per_word:.2f} subwords/word")
    axes[2].bar(["Unknown\ntokens", "Unknown\nrows", "Truncated\nrows"],
                [b.unk_tokens, b.unk_rows, b.truncated_rows], color="#ed9860")
    axes[2].set(ylabel="Count", title="BERTweet input diagnostics")
    fig.suptitle("Adaptation diagnostics (word models: outer validation folds; pretrained: all training rows)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "F3_coverage", coverage)

    tsne = create_tsne(frame, load_pretrained(), recompute, cache_only)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    palette = ["#cf5b58", "#4a9c75", "#a47abd", "#487fb6"]
    for ax, config in zip(axes.ravel(), ["w2v", "ft", "glove_stem", "bertweet"]):
        part = tsne[tsne.config == config]
        for label, name, color in zip(LABELS, CLASS_NAMES, palette):
            group = part[part.label == label]
            ax.scatter(group.x, group.y, c=color, s=11, alpha=.65, label=name)
        ax.set(title=config, xticks=[], yticks=[])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("t-SNE of sentence means (visualization only)")
    save_figure(fig, "F4_tsne", tsne)

    delta = np.zeros((7, 7))
    win = np.zeros((7, 7), dtype=int)
    matrix_rows = []
    for i, a in enumerate(CONFIG["configs"]):
        for j, b in enumerate(CONFIG["configs"]):
            item = comparison_detail(scores, a, b)
            delta[i, j], win[i, j] = item["mean_delta"] * 100, item["wins_a"]
            matrix_rows.append(item)
    fig, ax = plt.subplots(figsize=(10, 8))
    bound = max(1, abs(delta).max())
    im = ax.imshow(delta, cmap="RdBu", vmin=-bound, vmax=bound)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, "—" if i == j else f"{delta[i,j]:+.2f}\n{win[i,j]}/25", ha="center", va="center",
                    fontsize=8, color="white" if abs(delta[i,j]) > bound * .65 else "black")
    ax.set(xticks=range(7), xticklabels=CONFIG["configs"], yticks=range(7), yticklabels=CONFIG["configs"],
           title="Paired macro-F1 difference: row minus column\nMean percentage points; row wins / 25")
    ax.tick_params(axis="x", rotation=35)
    fig.colorbar(im, ax=ax, label="Mean difference (percentage points)")
    save_figure(fig, "F5_pairwise_delta", pd.DataFrame(matrix_rows))


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_reports():
    frame, _ = load_data("reports")
    scores = pd.read_csv(RESULTS / "fold_scores.csv", encoding="utf-8")
    summary = summary_table(scores)
    display = summary.copy()
    display["25折 macro-F1"] = [f"{m*100:.2f} ± {s*100:.2f}" for m, s in zip(display.macro_f1_mean, display.macro_f1_sd)]
    for name in ["r0_oof_macro_f1", "acc", "f1_neg", "f1_pos", "f1_irr", "f1_neu"]:
        display[name] = display[name].map(lambda v: f"{v*100:.2f}")
    display = display[["rank", "config", "r0_oof_macro_f1", "25折 macro-F1", "acc", "f1_neg", "f1_pos", "f1_irr", "f1_neu"]]
    comparisons = pd.DataFrame([comparison_detail(scores, a, b) for a, b in [
        ("ft", "w2v"), ("glove_stem", "w2v"), ("bertweet", "glove_stem"),
        ("glove_stem", "glove_direct"), ("bertweet", "bertweet_pca100")]])
    comparisons.to_csv(RESULTS / "key_comparisons.csv", index=False, encoding="utf-8")
    comparisons_display = comparisons.copy()
    for name in ["mean_delta", "sd_delta"]:
        comparisons_display[name] = comparisons_display[name].map(lambda v: f"{v*100:+.2f}")
    reference_comparisons = pd.DataFrame([comparison_detail(scores, name, "baseline_ref")
                                         for name in ["w2v", "ft", "glove_stem", "bertweet"]])
    reference_comparisons.to_csv(RESULTS / "baseline_comparisons.csv", index=False, encoding="utf-8")
    reference_display = reference_comparisons.copy()
    for name in ["mean_delta", "sd_delta"]:
        reference_display[name] = reference_display[name].map(lambda v: f"{v*100:+.2f}")
    coverage = pd.read_csv(RESULTS / "coverage.csv", encoding="utf-8")
    cov_display = coverage.groupby("embedding", sort=False)[["oov_token_rate", "oov_type_rate", "unseen_token_rate", "zero_rows"]].mean().reset_index()
    for name in ["oov_token_rate", "oov_type_rate", "unseen_token_rate"]:
        cov_display[name] = cov_display[name].map(lambda v: "—" if pd.isna(v) else f"{v*100:.2f}%")
    cov_display.zero_rows = cov_display.zero_rows.map(lambda v: f"{v:.2f}")
    c_counts = pd.crosstab(scores.config, scores.C).reindex(CONFIG["configs"], fill_value=0).reset_index()
    c_counts.columns = [str(c) for c in c_counts.columns]
    boundaries = read_json(RESULTS / "grid_boundaries.json")
    best_irr = summary.sort_values("f1_irr", ascending=False).iloc[0]
    irrelevant_count = int((frame.Sentiment == 2).sum())
    candidates = summary[summary.config.isin(["w2v", "ft", "glove_stem", "bertweet"])].head(2)
    recommendation = "、".join(candidates.config)
    costs = {"w2v": "100 维、无需预训练权重下载", "ft": "100 维、能用子词表示训练词表之外的词",
             "glove_stem": "100 维、预训练特征可重复缓存使用", "bertweet": "768 维、冻结特征可缓存，抽取成本高于静态词向量"}
    reasons = "\n".join(f"- `{row.config}`：25 折 macro-F1 为 {row.macro_f1_mean*100:.2f} ± {row.macro_f1_sd*100:.2f}，Irrelevant F1 为 {row.f1_irr*100:.2f}；{costs[row.config]}。"
                        for row in candidates.itertuples())
    alignment = comparison_detail(scores, "glove_stem", "glove_direct")
    subword = comparison_detail(scores, "ft", "w2v")
    b = coverage[coverage.embedding == "bertweet"].iloc[0]
    merges = pd.read_csv(RESULTS / "glove_merges_top20.csv", encoding="utf-8")
    rescued = pd.read_csv(RESULTS / "glove_rescued_top20.csv", encoding="utf-8")
    merge_metadata = read_json(FEATURES / "glove_stem_map.json")
    check = read_json(RESULTS / "check.json")
    repro_path = RESULTS / "reproducibility.json"
    repro = read_json(repro_path) if repro_path.exists() else None
    repro_text = (f"同机两次完整折内训练与嵌套探针运行逐字节一致，SHA256：`{repro['first_sha256']}`。"
                  if repro and repro["byte_identical"] and repro["second_sha256"] == file_sha(RESULTS / "fold_scores.csv")
                  else "尚无有效的同机两次运行一致性证据。")
    text = f"""# 第一阶段：固定线性探针的 embedding 对比

本阶段固定分类器，只改变文本表示，比较四种 embedding 及三个对照；目的在于为第二阶段提供候选建议。全部实验只使用原始训练集的 563 行，未读取测试集、审计文件或 ZIP 内容，未启动第二阶段。Logistic Regression 是测量探针，不计入课程要求的 analytics models。

## 总体结果

下表分数单位均为百分数；25 折标准差采用样本标准差（ddof=1），accuracy 与各类别 F1 为折均值。名次按 25 折平均 macro-F1 排序，是描述性排序，不等同于预先规则判定的优势。

{markdown_table(display)}

R0 为五折预测合并后计算的 OOF macro-F1，区别于五个单折 F1 的平均。`baseline_ref` 是按照相同嵌套选参流程重新训练的参照；历史 43.66 只作背景，不是本次复现值，缺少历史实现也不能据此宣称完全复现。

四种主 embedding 相对于相同流程基线的配对差与判定如下（Δ 与 SD 单位为百分点）：

{markdown_table(reference_display)}

## 预先规定的配对判断

Δ = A − B；下表 Δ 和 SD 单位为百分点。只有绝对平均差至少 1 点、且优势方向赢至少 18/25 折，才判定更好或更差；其他情况均为无法区分。不使用 p 值选择结果。

{markdown_table(comparisons_display)}

`ft − w2v` 对应子词信息；`glove_stem − w2v` 同时改变预训练规模与训练算法，因此不能把差异单独归因于预训练；`bertweet − glove_stem` 还改变架构、预训练任务与维度，因此不能作为上下文作用的独立因果证据。两个辅助对照分别检查词表对齐和维度压缩。25 折存在训练样本重叠，不应当成 25 个独立数据集。

## Irrelevant 类别

训练集中 Irrelevant 有 {irrelevant_count} 行（{irrelevant_count/563*100:.2f}%）。该类别平均 F1 最高的配置为 `{best_irr['config']}`，达到 {best_irr['f1_irr']*100:.2f}%。各类别结果见总表及 F2；不能仅根据总体 macro-F1 代替该类表现。类别语义异质性可能影响识别，但本实验不能验证该解释。

## 表示适配度

{markdown_table(cov_display)}

词模型以上为 25 个外层验证折统计的平均值；预训练表示为训练集全文统计。`unseen_token_rate` 表示词未出现在当前训练词表；FastText 实际可用子词合成向量，因此其实际 OOV 为零。零向量列对词模型是每折平均行数，对预训练表示是全文行数。

BERTweet 每个词平均 {b.subwords_per_word:.3f} 个子词，{b.split_word_rate*100:.2f}% 的词出现被切成至少两个子词；抽取后真实 token 中 `<unk>` 为 {int(b.unk_tokens)} 个，涉及 {int(b.unk_rows)} 行；超过 128 token 而被截断的行数为 {int(b.truncated_rows)}。碎片率以全部未截断文本的词出现为分母，unknown 计数以实际抽取的非特殊 token 为范围。

G-direct 的出现次数 OOV 为 {coverage[coverage.embedding=='glove_direct'].oov_token_rate.iloc[0]*100:.2f}%，G-stem 为 {coverage[coverage.embedding=='glove_stem'].oov_token_rate.iloc[0]*100:.2f}%；对应 G-stem − G-direct 的平均 F1 差为 {alignment['mean_delta']*100:+.2f} 点，G-stem 赢 {alignment['wins_a']}/25 折，按规则为“{alignment['decision']}”。FastText 对训练未见词可合成向量，对应 ft − w2v 的平均差为 {subword['mean_delta']*100:+.2f} 点，按规则为“{subword['decision']}”。词表对齐可能改善查词，也可能合并语义不同的词，仅凭覆盖率不能断言分类一定改善。

完整 GloVe 词表逐词转换，无预筛选，共 {merge_metadata['pretrained_words']} 个原词、{merge_metadata['aligned_forms']} 个单词形式，其中 {merge_metadata['merged_forms']} 个形式合并了多个原词。空输出和多词输出均跳过，详情保存在缓存元数据。

救回词出现次数前 20：

{markdown_table(rescued)}

合并最多的 20 个形式及其完整原词：

{markdown_table(merges)}

## 调参证据

每种配置的 C 选择次数如下；全部 1575 个候选评估记录在 `tuning.csv`。

{markdown_table(c_counts)}

网格边缘选择次数（共 25 折）：{', '.join(f'{k}={v}' for k,v in boundaries.items())}。超过半数的配置写入 OPEN_QUESTIONS，不扩展网格。

## 方法限制与复现证据

Word2Vec/FastText 在各外层训练部分训练，内层选参复用这些向量。因此内层验证文本在无监督词向量训练时已被看到，内层分数可能偏乐观；外层验证文本始终不参与训练。标准化、PCA 和稀疏基线特征分别只在对应内层训练部分拟合，选 C 后再在整个外层训练部分拟合。PCA 使用原始 BERTweet 向量先降到 100 维，再使用相同 StandardScaler 与 LR。

所有特征统一平均池化。BERTweet 关闭 tokenizer normalization，逐字节检查输入，冻结 float32 参数，以最后四层均值再对非特殊 token 求均值；抽取前后 state_dict 哈希相同。token 级 float16 缓存仅保存，不用于本阶段分类。

只验证 {len(check['verified_hashes'])} 个允许文件的原始哈希；原始清单其余 {len(check['unverified_manifest_entries'])} 项未读取、未验证，不声称完整数据包全部哈希通过。每个阶段的 Phrase 哈希与 check 一致。{repro_text}

F4 的全训练集 Word2Vec/FastText 模型只用于可视化，未用于任何分类评分。t-SNE 图不能作为分类优势的证据。运行环境、模型版本和冻结校验见 results 与 features 的 JSON 文件。

## 第二阶段候选建议

建议小组优先考虑 **{recommendation}**：它们是四种主 embedding 中按 25 折平均 macro-F1 排序的前两种，具体分数、折间波动、Irrelevant 表现和配对判定见上表。该建议不将 G-direct 或 PCA 对照重复计算为独立 embedding，也不将稀疏基线计入 embedding。若配对规则为无法区分，应保留这一不确定性，并结合计算成本和后续模型适配性由小组最终决定。

{reasons}
"""
    (REPORTS / "embedding_comparison.md").write_text(text, encoding="utf-8")
    method = """We compared four frozen sentence representations on the unchanged 563-row v3 training set: 100-dimensional skip-gram Word2Vec and FastText trained only on each outer training partition, pretrained 100-dimensional GloVe-Twitter, and frozen 768-dimensional vinai/bertweet-base. All representations used mean pooling; BERTweet averaged hidden layers 9–12 before averaging non-special, non-padding tokens, with tokenizer normalization disabled, a maximum length of 128, float32 inference, and identical model-state checksums before and after extraction. GloVe was evaluated both directly and after applying the original lemma-and-stem normalizer to every pretrained vocabulary entry, averaging vectors that mapped to the same single-word form, without changing any input Phrase. A fold-local BERTweet-PCA100 control applied PCA before the shared probe. Evaluation used the supplied grouped five-fold split plus four shuffled StratifiedGroupKFold repeats (seeds 1–4), yielding 25 outer folds. Each outer training partition used three-fold stratified grouped inner cross-validation (seed 42) to select C from [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10], choosing the smallest C within strictly less than 0.001 macro-F1 of the best. StandardScaler and balanced lbfgs Logistic Regression (max_iter=5000) were shared by all embeddings; PCA and scaling were fitted separately within each inner training partition and refitted on the outer training partition. Word-vector training was not repeated within inner folds, so inner validation text was exposed during unsupervised representation training, potentially making tuning optimistic. A sparse word-unigram plus char_wb 3–5 reference followed the same nested selection, with each feature branch independently L2-normalized and no scaling. We report pooled R0 out-of-fold macro-F1, 25-fold means and sample standard deviations, per-class F1, and paired fold differences. Differences were considered distinguishable only if the absolute mean difference was at least one percentage point and the favored configuration won at least 18 of 25 folds; overlapping folds are not independent replications. No test or audit data were read. Logistic Regression served solely as a measurement probe and does not count toward the course's required analytics models.\n"""
    (REPORTS / "method_en.md").write_text(method, encoding="utf-8")
    questions_path = REPORTS / "OPEN_QUESTIONS.md"
    original = questions_path.read_text(encoding="utf-8").split("\n## Stage 1 execution decisions")[0]
    original = original.replace("本文件仅记录资料整理阶段发现的问题。尚未执行实验，不包含结果。",
                                "前五项保留资料整理阶段的问题；本次执行决定及实际证据见文末和结果报告。")
    decisions = """
## Stage 1 execution decisions

- OQ-001：按既有保守方案仅校验 5 个白名单文件；其余清单条目全部标记未验证。没有读取 ZIP、测试或审计内容。
- OQ-002：原始 normalize_input 返回二元组，词表对齐只使用第一个元素。对产生空输出的原词跳过；多词输出也跳过。其他异常直接报错。
- OQ-003：FastText 同时记录显式训练词表未见率和实际无法表示的 OOV 率（后者为零）。未见词使用子词合成向量。
- OQ-004：实际 Python、包版本和机器信息见 results/stage1/env_*.txt；安装时使用 CPU PyTorch 2.5.1，Transformers 固定 4.44.2。NLTK 3.10.4 下载需显式信任受管代理，资源未用于额外实验数据。
- OQ-006：PCA100 先在原始 BERTweet 句向量上做 PCA，再使用统一 StandardScaler；内层训练、外层重拟合分别拟合两者。PCA 固定 full SVD 与 seed=42。
- OQ-007：所有标准差采用 ddof=1；汇总 accuracy 与类别 F1 为 25 折不加权均值。C 的近似并列以全网格最高均值为基准，严格小于 0.001 时选最小值。
- OQ-008：负方向优势使用 B 胜折数，同分不计任一方胜。重叠折不能视为独立样本；不计算 p 值。
- OQ-009：基线词 unigram 使用空格分词、lowercase=False，避免默认 token pattern 丢弃单字符或重分词；字符分支使用 char_wb。两支保留各自 L2 归一化后直接拼接，不做拼接后归一化。
- OQ-010：BERTweet 碎片率对全部词出现计数；unk 是截断后真实 token，截断判定基于加上起止符后的完整长度。token 缓存不含特殊 token，并包含行偏移和 token IDs。
- OQ-011：GloVe 不预筛选，全部词运行原始函数；合并均值采用 float32。合并数 top20 并列按词形字典序稳定排列。源码字节码写入不允许指向只读数据目录。
- OQ-012：同机重跑重新训练全部折内词模型并重新拟合全部探针，复用冻结预训练特征；没有声称跨设备或跨版本位级一致。
- OQ-013：第二阶段建议从四个主方案按 25 折均值取前两名，同时保留配对规则的“无法区分”结论。仅作建议，不执行第二阶段。
"""
    boundary_lines = [f"- 网格边缘记录：{config} 有 {count}/25 折选择边缘 C；{'超过半数，不扩展网格。' if count > 12 else '未超过半数。'}" for config, count in boundaries.items()]
    questions_path.write_text(original + decisions + "\n" + "\n".join(boundary_lines) + "\n", encoding="utf-8")
