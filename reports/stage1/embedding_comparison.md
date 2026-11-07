# 第一阶段：固定线性探针的 embedding 对比

本阶段固定分类器，只改变文本表示，比较四种 embedding 及三个对照；目的在于为第二阶段提供候选建议。全部实验只使用原始训练集的 563 行，未读取测试集、审计文件或 ZIP 内容，未启动第二阶段。Logistic Regression 是测量探针，不计入课程要求的 analytics models。

## 总体结果

下表分数单位均为百分数；25 折标准差采用样本标准差（ddof=1），accuracy 与各类别 F1 为折均值。名次按 25 折平均 macro-F1 排序，是描述性排序，不等同于预先规则判定的优势。

| rank | config | r0_oof_macro_f1 | 25折 macro-F1 | acc | f1_neg | f1_pos | f1_irr | f1_neu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | bertweet | 46.86 | 47.42 ± 4.16 | 49.31 | 58.85 | 51.18 | 32.21 | 47.44 |
| 2 | bertweet_pca100 | 45.77 | 46.50 ± 3.68 | 48.32 | 57.89 | 50.44 | 29.83 | 47.84 |
| 3 | baseline_ref | 43.38 | 41.91 ± 3.50 | 45.18 | 58.12 | 45.49 | 21.10 | 42.93 |
| 4 | glove_direct | 41.48 | 40.78 ± 3.61 | 43.66 | 54.01 | 47.69 | 18.46 | 42.98 |
| 5 | glove_stem | 38.99 | 39.52 ± 4.63 | 41.45 | 50.12 | 42.80 | 23.40 | 41.76 |
| 6 | ft | 39.24 | 37.90 ± 3.54 | 40.21 | 49.82 | 40.48 | 22.03 | 39.27 |
| 7 | w2v | 38.62 | 36.33 ± 4.26 | 38.85 | 48.89 | 39.19 | 19.13 | 38.10 |

R0 为五折预测合并后计算的 OOF macro-F1，区别于五个单折 F1 的平均。`baseline_ref` 是按照相同嵌套选参流程重新训练的参照；历史 43.66 只作背景，不是本次复现值，缺少历史实现也不能据此宣称完全复现。

四种主 embedding 相对于相同流程基线的配对差与判定如下（Δ 与 SD 单位为百分点）：

| config_a | config_b | mean_delta | sd_delta | wins_a | wins_b | decision |
| --- | --- | --- | --- | --- | --- | --- |
| w2v | baseline_ref | -5.58 | +5.22 | 3 | 22 | 更差 |
| ft | baseline_ref | -4.01 | +5.44 | 7 | 18 | 更差 |
| glove_stem | baseline_ref | -2.39 | +4.83 | 8 | 17 | 无法区分 |
| bertweet | baseline_ref | +5.51 | +5.48 | 20 | 5 | 更好 |

## 预先规定的配对判断

Δ = A − B；下表 Δ 和 SD 单位为百分点。只有绝对平均差至少 1 点、且优势方向赢至少 18/25 折，才判定更好或更差；其他情况均为无法区分。不使用 p 值选择结果。

| config_a | config_b | mean_delta | sd_delta | wins_a | wins_b | decision |
| --- | --- | --- | --- | --- | --- | --- |
| ft | w2v | +1.57 | +4.54 | 19 | 6 | 更好 |
| glove_stem | w2v | +3.19 | +6.19 | 18 | 7 | 更好 |
| bertweet | glove_stem | +7.90 | +5.03 | 23 | 2 | 更好 |
| glove_stem | glove_direct | -1.26 | +3.70 | 8 | 17 | 无法区分 |
| bertweet | bertweet_pca100 | +0.92 | +3.25 | 15 | 10 | 无法区分 |

`ft − w2v` 对应子词信息；`glove_stem − w2v` 同时改变预训练规模与训练算法，因此不能把差异单独归因于预训练；`bertweet − glove_stem` 还改变架构、预训练任务与维度，因此不能作为上下文作用的独立因果证据。两个辅助对照分别检查词表对齐和维度压缩。25 折存在训练样本重叠，不应当成 25 个独立数据集。

## Irrelevant 类别

训练集中 Irrelevant 有 101 行（17.94%）。该类别平均 F1 最高的配置为 `bertweet`，达到 32.21%。各类别结果见总表及 F2；不能仅根据总体 macro-F1 代替该类表现。类别语义异质性可能影响识别，但本实验不能验证该解释。

## 表示适配度

| embedding | oov_token_rate | oov_type_rate | unseen_token_rate | zero_rows |
| --- | --- | --- | --- | --- |
| w2v | 15.95% | 36.88% | 15.95% | 0.00 |
| ft | 0.00% | 0.00% | 15.95% | 0.00 |
| glove_direct | 8.00% | 18.70% | 8.00% | 0.00 |
| glove_stem | 4.84% | 12.68% | 4.84% | 0.00 |
| bertweet | — | — | — | 0.00 |

词模型以上为 25 个外层验证折统计的平均值；预训练表示为训练集全文统计。`unseen_token_rate` 表示词未出现在当前训练词表；FastText 实际可用子词合成向量，因此其实际 OOV 为零。零向量列对词模型是每折平均行数，对预训练表示是全文行数。

BERTweet 每个词平均 1.210 个子词，15.55% 的词出现被切成至少两个子词；抽取后真实 token 中 `<unk>` 为 7 个，涉及 7 行；超过 128 token 而被截断的行数为 0。碎片率以全部未截断文本的词出现为分母，unknown 计数以实际抽取的非特殊 token 为范围。

G-direct 的出现次数 OOV 为 8.00%，G-stem 为 4.84%；对应 G-stem − G-direct 的平均 F1 差为 -1.26 点，G-stem 赢 8/25 折，按规则为“无法区分”。FastText 对训练未见词可合成向量，对应 ft − w2v 的平均差为 +1.57 点，按规则为“更好”。词表对齐可能改善查词，也可能合并语义不同的词，仅凭覆盖率不能断言分类一定改善。

完整 GloVe 词表逐词转换，无预筛选，共 1193514 个原词、1011291 个单词形式，其中 66429 个形式合并了多个原词。空输出和多词输出均跳过，详情保存在缓存元数据。

救回词出现次数前 20：

| word | count |
| --- | --- |
| 2 | 50 |
| 3 | 24 |
| 1 | 23 |
| 5 | 22 |
| 4 | 15 |
| 0 | 11 |
| redempt | 10 |
| 100 | 8 |
| 6 | 7 |
| 19 | 6 |
| 10 | 6 |
| communiti | 5 |
| 8 | 5 |
| 21 | 4 |
| ps4 | 4 |
| 7 | 4 |
| 20 | 4 |
| hearthston | 3 |
| 99 | 3 |
| warzon | 3 |

合并最多的 20 个形式及其完整原词：

| v3_form | merged_words | source_words |
| --- | --- | --- |
| o | 34 | o / :o / º / ｏ / ⓞ / o. / ;o / >:o / :o) / =o / :-o / ;o) / :'o / =-o / <o> / :o( / ᵒ / oing / ⒪ / =o) / oed / o$ / #o / ℴ / >;o / :o/ / >=o / <:o) / ='o / ;'o / =o( / <:o / ;-o / ;o( |
| follow | 22 | follow / followers / followed / following / follower / follows / ｆｏｌｌｏｗ / ｆｏｌｌｏｗｅｒｓ / followe / #follow / ｆoｌｌoｗeｒ / followes / followings / followerss / #ｆｏｌｌｏｗ / ｆｏｌｌｏｗｓ / ℱollow / followal / folloｗ / followent / #followers / followeers |
| love | 22 | love / loves / loved / lovely / loving / lovee / lovelies / #love / loveliest / lovingly / lovelys / ｌｏｖｅ / loveing / lovees / love. / ℒℴvℯ / loveed / #lovely / ℒove / ˡºᵛᵉ / lovings / loveli |
| activ | 19 | active / activity / activities / activate / activated / actively / activism / activation / activates / activ / activating / activator / actives / activer / activations / activement / activiti / actived / activent |
| n | 19 | n / 'n / n. / ’n / 'n' / ⓝ / ｎ / #n / n@… / <n> / ⁿ / ’n’ / n$ / n@+[ / n@: / <n / ⒩ / ’n' / ℕ |
| inform | 18 | information / informe / inform / informed / informative / informes / informal / informing / informate / informations / informer / informs / informational / informant / informants / informally / informers / informeer |
| person | 18 | person / personal / personality / personally / persone / persons / personalities / personalized / persones / personals / personer / personalize / personalization / personable / personeli / personaliti / personalitys / personalizing |
| accept | 17 | accept / accepted / acceptable / accepting / acceptance / accepts / accepte / accepter / accepteer / acceptent / acceptances / acception / acceptes / acceptation / acceptably / acceptability / acceptant |
| domin | 17 | domination / dominate / dominant / dominic / dominated / dominating / dominance / dominates / domine / dominical / domin / dominent / dominator / dominer / domineering / domines / dominators |
| intern | 17 | international / internal / intern / interns / internationally / interne / internally / internationals / interning / internment / internation / internes / interned / interner / internals / internalize / internalized |
| organ | 17 | organic / organization / organized / organ / organize / organizing / organizations / organizer / organs / organizers / organism / organics / organisms / organizes / organically / organe / organes |
| respect | 17 | respect / respects / respected / respectful / respecting / respecte / respectable / respecter / respective / respectfully / respectively / respectent / respectes / respecteer / respectability / respectant / #respect |
| retweet | 17 | retweet / retweets / ｒｅｔｗｅｅｔ / retweeting / retweeted / retweeters / retweeter / #retweet / retweete / <retweet> / retweetable / ｒｅｔｗｅｅｔｅｒｓ / rｅｔｗｅｅｔ / retweetes / retweetant / ℛetweet / retweetent |
| anim | 16 | animal / animals / anime / animation / animated / animes / animate / anims / anim / animations / animator / animators / animating / animer / #anime / animates |
| continu | 16 | continue / continues / continued / continuing / continuously / continuous / continually / continuer / continu / continuation / continuity / continuate / continuent / continual / continuance / continuant |
| control | 16 | control / controller / controle / controls / controlling / controlled / controles / controllers / controlate / controll / controler / controling / controllable / controleer / controled / controllate |
| depend | 16 | depende / depends / depend / depending / depender / dependent / dependable / depended / dependency / dependence / dependant / dependes / dependents / dependability / dependencies / dependance |
| excit | 16 | excited / exciting / excitement / excite / excites / excitment / excitedly / excitable / excit / exciter / excitant / excitation / excitements / #excited / excitent / excitingly |
| initi | 16 | initials / initial / initiative / initially / initiatives / initiate / initiation / initiated / initiating / initiates / initiator / initiations / initi / initializing / initier / initialize |
| instal | 16 | install / installed / installation / installing / instal / installment / installer / installations / installs / instale / instalment / installers / installous / installments / instalments / installent |

## 调参证据

每种配置的 C 选择次数如下；全部 1575 个候选评估记录在 `tuning.csv`。

| config | 0.001 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 | 3.0 | 10.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w2v | 7 | 10 | 2 | 3 | 1 | 2 | 0 | 0 | 0 |
| ft | 13 | 3 | 4 | 2 | 0 | 0 | 1 | 1 | 1 |
| glove_stem | 6 | 2 | 9 | 5 | 2 | 1 | 0 | 0 | 0 |
| glove_direct | 10 | 9 | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| bertweet | 9 | 12 | 2 | 1 | 0 | 0 | 0 | 0 | 1 |
| bertweet_pca100 | 6 | 5 | 5 | 7 | 1 | 1 | 0 | 0 | 0 |
| baseline_ref | 8 | 2 | 4 | 2 | 3 | 2 | 2 | 1 | 1 |

网格边缘选择次数（共 25 折）：baseline_ref=9, bertweet=10, bertweet_pca100=6, ft=14, glove_direct=10, glove_stem=6, w2v=7。超过半数的配置写入 OPEN_QUESTIONS，不扩展网格。

## 方法限制与复现证据

Word2Vec/FastText 在各外层训练部分训练，内层选参复用这些向量。因此内层验证文本在无监督词向量训练时已被看到，内层分数可能偏乐观；外层验证文本始终不参与训练。标准化、PCA 和稀疏基线特征分别只在对应内层训练部分拟合，选 C 后再在整个外层训练部分拟合。PCA 使用原始 BERTweet 向量先降到 100 维，再使用相同 StandardScaler 与 LR。

所有特征统一平均池化。BERTweet 关闭 tokenizer normalization，逐字节检查输入，冻结 float32 参数，以最后四层均值再对非特殊 token 求均值；抽取前后 state_dict 哈希相同。token 级 float16 缓存仅保存，不用于本阶段分类。

只验证 5 个允许文件的原始哈希；原始清单其余 47 项未读取、未验证，不声称完整数据包全部哈希通过。每个阶段的 Phrase 哈希与 check 一致。同机两次完整折内训练与嵌套探针运行逐字节一致，SHA256：`954d25a0d2d860c217e6cb3ce1db106512815787091f12fe3b88ac22879fdfcb`。

F4 的全训练集 Word2Vec/FastText 模型只用于可视化，未用于任何分类评分。t-SNE 图不能作为分类优势的证据。运行环境、模型版本和冻结校验见 results 与 features 的 JSON 文件。

## 第二阶段候选建议

建议小组优先考虑 **bertweet、glove_stem**：它们是四种主 embedding 中按 25 折平均 macro-F1 排序的前两种，具体分数、折间波动、Irrelevant 表现和配对判定见上表。该建议不将 G-direct 或 PCA 对照重复计算为独立 embedding，也不将稀疏基线计入 embedding。若配对规则为无法区分，应保留这一不确定性，并结合计算成本和后续模型适配性由小组最终决定。

- `bertweet`：25 折 macro-F1 为 47.42 ± 4.16，Irrelevant F1 为 32.21；768 维、冻结特征可缓存，抽取成本高于静态词向量。
- `glove_stem`：25 折 macro-F1 为 39.52 ± 4.63，Irrelevant F1 为 23.40；100 维、预训练特征可重复缓存使用。
