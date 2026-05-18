# 每日论文精读

每天追一篇 huggingface daily papers 榜首论文，用中文做四段结构化精读：

- **问题**：要解决什么、为什么是问题、前人解法在哪里不够
- **方法**：核心做法与关键设计，符号与公式按出现顺序解释清楚
- **实验**：数据集、baseline、主要数字、最具说服力的 ablation
- **局限**：作者自己承认的与读者能看出来的潜在问题

风格遵循"对一个进 ML 业界三年的工程师能读懂"为底线：技术词第一次出现时顺手解释，避免不必要的术语堆砌，不写"该论文具有重要价值"这种 LLM 收尾。

每篇笔记文末有一句 60 字以内的"一句话"小结，给读者未来在 reading list 里快速回忆用。

## 自动化

源头是一条定时任务，每天早晨抓 huggingface daily papers 当日榜首，按四段结构精读、归档、push 到本仓库。生成路径由 [yingwang/claude-skills](https://github.com/yingwang/claude-skills) 里的 `paper-read` skill 控制。

## 列表

最新在前。
