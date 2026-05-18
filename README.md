# paper-reads

每天追一篇 huggingface daily papers 榜首论文的四段精读（中英双语），存档并发布到 GitHub Pages。

公开站点：https://yingwang.github.io/paper-reads/

## 内容形式

每篇笔记按"问题 - 方法 - 实验 - 局限"四段展开（英文版用 The Problem / Method / Experiments / Limitations），文末附一句小结（中文 60 字以内 / 英文一句）。技术词第一次出现时顺手解释，目标读者是已有一定 ML 基础但不一定专门做该子领域的工程师。

中文版先写，英文版从中文直译而来，保持结构与详略一致。

## 文件结构

```
paper-reads/
├── docs/
│   ├── index.md              # 中文首页
│   ├── en/
│   │   ├── index.md          # 英文首页
│   │   └── papers/           # 英文论文笔记
│   ├── papers/               # 中文论文笔记
│   │   ├── .pages            # awesome-pages 配置（最新在前）
│   │   ├── YYYY-MM-DD-slug-1.md
│   │   └── YYYY-MM-DD-slug-2.md
│   └── javascripts/mathjax.js
├── mkdocs.yml
└── .github/workflows/deploy-docs.yml
```

## 自动化

笔记由 [yingwang/claude-skills](https://github.com/yingwang/claude-skills) 的 `paper-read` skill 生成：

```bash
/paper-read --trending 1                       # 自动追当日榜首
/paper-read https://arxiv.org/abs/2401.12345   # 手动指定论文
```

skill 在本地存档完成后，会把 `.md` 复制到本仓库 `docs/papers/` 并 `git push`，GitHub Actions 自动构建 + 发布到 Pages。

## 本地构建

```bash
pip install "mkdocs-material>=9.5" "mkdocs-awesome-pages-plugin>=2.10" \
    "pymdown-extensions>=10.7" "mkdocs>=1.6" "mdx-truly-sane-lists>=1.3"
mkdocs serve
```

## License

文字部分：CC BY-NC-SA 4.0（见 [LICENSE](LICENSE)）。引用论文图表与公式归原作者所有。
