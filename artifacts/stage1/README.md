# 第一阶段结果下载

- `stage1_results.zip`：便于查看的小型结果包，包含实现、测试、允许的训练输入、报告、5 张图及对应 CSV、全部评分和验收证据、已执行 notebook。不包含特征缓存；其中的 notebook 已保留执行输出，重新运行缓存模式请使用下方完整包。
- `stage1_complete.zip.part01` 等：完整包的连续二进制分卷，额外包含所有特征缓存、GloVe 对齐词表和 BERTweet token 特征。每卷最多 48 MiB，符合 GitHub 普通文件限制。
- `DOWNLOAD_MANIFEST.json`：两种包和每个分卷的大小、文件数及 SHA256。
- `unpack.py`：使用 Python 标准库合并、校验和解压完整包。

## 下载与解压

只查看结果时，下载 `stage1_results.zip` 即可。完整复用缓存时，请下载本目录的全部分卷、`DOWNLOAD_MANIFEST.json` 和 `unpack.py`，放在同一目录，然后运行：

```bash
python unpack.py --output stage1_download
```

如果已克隆结果分支，从仓库根目录运行：

```bash
python artifacts/stage1/unpack.py --output stage1_download
```

输出目录须为空。脚本会逐卷校验、合并并验证完整 ZIP，再核对解压后的每个文件。后续环境和 notebook 运行方式见解压目录的 `stage1/README.md`。

包内没有测试集、审计文件、原始数据 ZIP、下载的预训练模型、虚拟环境或凭证。完整包包含的只是本次实验所需的允许输入和生成缓存。

结果摘要：7 种配置 × 25 折；BERTweet macro-F1 为 47.42 ± 4.16%；两次完整探针运行逐字节一致；10 项测试通过；缓存模式 notebook 已执行，并已在无 stage1 代码目录的环境中验证。
