# Stage 1 open questions and preparation notes

本文件仅记录资料整理阶段发现的问题。尚未执行实验，不包含结果。

## OQ-001：完整哈希校验与禁止读取文件

原始说明要求验证 `SHA256SUMS.json` 全部哈希，同时禁止读取测试集及审计文件。
当前 `DATA_DIR` 只提取允许使用的输入、归一化代码及元数据；完整包保留为原始 ZIP。
保守执行建议：只对允许读取且已提取的文件校验原始清单哈希，逐项报告范围。
未提取及禁止读取的条目保持未验证，不声称全部通过；不为校验而读取测试或审计内容。
这是一项待确认的协议差异，不是对原始任务说明的静默改写。

## OQ-002：normalize_input 的实际返回类型

原始 `code/unified_data.py` 中 `normalize_input(text, morphology="lemma_stem")`
返回 `(output, details)` 二元组。GloVe 词表对齐应取第一个元素 `output`，
再检查是否恰为一个词。不要将整个二元组当作查词键，也不要将该函数用于 `Phrase`。

## OQ-003：FastText 的 OOV 术语

原始说明同时提到“与 Word2Vec 相同的 OOV 处理”和“FastText 不会出现 OOV”。
保守解释建议：统计未出现在训练词表中的词，但用 FastText 的子词向量生成能力处理这些词，
不因词不在显式词表就跳过；保持其余参数与 Word2Vec 一致。实现时说明统计口径。

## OQ-004：环境清单尚未验证

`requirements-stage1.txt` 是安装输入，不是已验证锁定文件。
实际云端安装、模型下载、NLTK 资源、推理与 notebook 检查仍需执行，
通过后记录准确版本、机器信息和模型缓存信息。

## OQ-005：参考报告未上传

用户明确要求不将 `DSAI4205_Integrated_Interim_Report_final.docx` 与
`DSAI4205_Interim_Report.pdf` 上传至公开仓库。课程项目说明 PDF 获得单独上传授权。
第一阶段基线按 `STAGE1_TASK.md` 的参数实现；缺少历史代码或证据时如实说明，
不要以原始说明给出的 43.66 作为本次实际复现结果。
