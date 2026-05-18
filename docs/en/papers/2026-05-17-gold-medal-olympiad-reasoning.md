# Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling

> **Original title**: Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling
> **Authors**: Yafu Li, Runzhe Zhan, Haoran Zhang, and 25 others (28 authors total)
> **Institutions**: Shanghai AI Laboratory, Chinese University of Hong Kong, Tsinghua University, Shanghai Jiao Tong University, Peking University
> **Year**: 2026 (arxiv ID 2605.13301, submitted 2026-05-13)
> **Subject**: cs.AI / cs.CL
> **Link**: [https://arxiv.org/abs/2605.13301](https://arxiv.org/abs/2605.13301)
> **Reading date**: 2026-05-17

## I. The Problem

In the large-language-model field, the ability to solve mathematics and physics olympiad problems has long served as the canonical touchstone for "rigorous reasoning." An IMO problem is not like a standard AIME multiple-choice question, where producing a numerical answer is enough to score. Graders inspect the proof line by line, and any logical gap, skipped step, or invocation of an unproven lemma costs points. The model must not only "compute correctly" but also "prove clearly," and it must stay on topic and internally consistent across a reasoning chain that can exceed one hundred thousand tokens.

Over the past year, mainstream approaches have fallen into three camps. The first is to throw raw compute at the problem, sustaining long reasoning chains through sheer scale (for instance DeepSeek-V3.2's 943.7 billion tokens of continued training), at extraordinary training cost. The second is heavy multi-branch search combined with voting, gathering scattered correct paths through majority voting; the downside is that no single complete proof can be guaranteed internally consistent. The third leverages the chain-of-thought capability that general-purpose models already possess, applying SFT plus RL fine-tuning. This third line, however, often runs into an awkward paradox: the post-trained base has already gone through RLHF on general dialogue, and stacking new SFT on top tends to disrupt its existing prompt-following and self-checking behaviors.

The question this paper sets out to answer is whether there exists a single, deliberately simple recipe, free of elaborate search machinery, that can convert an already post-trained reasoning base into a "proof-grade" olympiad solver, and let a 30B-class model (a sparse Mixture-of-Experts with 3B activated per forward pass, referred to below as 30B-A3B) clear the gold-medal threshold at top-tier competitions such as the IMO and IPhO. Two terms need clarification first. **SFT** (Supervised Fine-Tuning) means supervised fine-tuning: feeding the model high-quality human or synthetic solutions so it learns to imitate them. **RL** (Reinforcement Learning) means reinforcement learning: letting the model attempt problems on its own and adjusting parameters based on some reward signal. Combining the two is standard practice in the post-training stage of LLMs, but how they are combined, what rewards drive them, and in what curriculum order they are scheduled all have outsized effects on final capability.

## II. Method

The full recipe runs in three stages: reverse-perplexity-curriculum SFT, a two-phase RL pipeline, and test-time scaling. The common thread across the three is "first recover and reshape behavioral patterns, then anchor rigor with reward signals," avoiding the failure mode where RL applied too early shatters the base model's dialogue abilities.

### Reverse-Perplexity-Curriculum SFT

The core innovation of the first fine-tuning stage lies in how the curriculum is ordered. The authors prepare 338K solution trajectories under 8K tokens in length as training samples, but they are not fed in random order. Each sample's perplexity is first computed against the base model P1-30B-A3B (perplexity, abbreviated PPL, is defined as the exponential of the average negative log-likelihood PPL = exp(-1/T · Σ log π(y|x)), and reflects how "unfamiliar" a sample looks to the model). Within each epoch, samples are then sorted by perplexity from high to low: the model is trained first on what looks "most unfamiliar," and later on what is already "familiar."

The motivation for this reverse curriculum is to preserve the dialogue and prompt-following abilities the post-trained base already has. Sorting from low to high PPL (familiar to unfamiliar) lets high-PPL data perturb the base in the late stages of training, causing degradation on general-purpose tasks. Sorting in the opposite direction, from high to low, has the model first learn new proof-search and self-checking behaviors on hard samples, then "consolidate" existing skills on the low-PPL familiar data, effectively reinforcing the prior abilities rather than overwriting them.

Beyond direct solution trajectories, the data mixture also includes two kinds of synthetic samples: **self-verification trajectories** and **self-correction trajectories**. In the former, an external strong model, DeepSeek-V3.2-Speciale, generates a "step-by-step verification" commentary on each solution. In the latter, once the commentary flags an error, a corrected version is generated. As a result, the model has already seen the full three-stage behavior pattern of "solve once, look back and check, then revise" during SFT. Training runs for 4 epochs at batch size 128, with learning rate 1e-5 and warmup.

### Two-Phase RL Pipeline

After SFT comes RL, but unlike generic RLHF that scores an entire output directly, the pipeline goes coarse-then-fine in two phases.

**Coarse phase (Coarse RL, verifiable rewards)**: 8,967 problems whose answers can be checked by machine serve as prompts. In each episode the model generates a complete solution, and only the final answer is graded: reward 1 if correct, 0 if not. The optimizer here is **GSPO** (Group Sequence Policy Optimization), which augments classical PPO with a sequence-length-normalized importance sampling ratio, preventing long outputs from being over-amplified in the gradient. Verification proceeds through three cascading layers: first a normalized-text exact match, then a semantic match through the Math-Verify rule engine if that fails, and finally a generative verification call to the external model gpt-oss-120b if the rule engine still cannot decide.

**Refined phase (Refined RL, proof-level rewards)**: this phase switches to 16,287 problems whose answers are not machine-verifiable, such as IMO-style propositions that begin with "prove that." Three mechanisms are stacked here.

The first is a **generative reward**, in which the external scoring model DeepSeekMath-V2 grades the full proof and judges whether "the reasoning path is mathematically valid, sufficiently rigorous, and complete." The second is a **self-correction mechanism**: when the mean reward of a sampled group drops below 0.5, "correction problems" are mixed into the training batch at a 20% ratio, with prompts containing the original problem, the previous incorrect solution, and an instruction asking the model to critique and revise. The third is **experience replay**: for problems with one or two successes across multiple samples (denoted in the paper as 0 < n₊(q) < 2), successful trajectories are saved into a buffer, and subsequent training samples from the buffer at ratio ρ = 0.25 into each batch, always selecting the lowest-entropy trajectory as the demonstration (the model's "most confident" successful path). The final objective combines the two streams with a weighted sum: 𝒥refined = (1-ρ)·𝒥_fresh + ρ·𝒥_replay.

### Test-Time Scaling (TTS)

At inference time, a "self-review" loop is introduced. Given a problem, the model first generates a complete solution. Then the solution and the original problem are fed back together, and the model writes an "error report" as a verification step. Finally the model issues a verdict: accept, reject, or revise. If the verdict is revise, the model rewrites the solution, and the loop continues until convergence or until the budget is exhausted. The paper reports that the median lengths of a full TTS trajectory on USAMO 2026 are: initial generation 106K tokens, revision stage 83K tokens, verification stage 28.7K tokens. In other words, a single problem can stretch to the 200K-token scale at inference, with compute cost roughly tens of times that of ordinary short-reasoning tasks.

## III. Experiments

The main results split into four categories: answer-verifiable benchmarks, proof-level benchmarks, physics olympiads, and full-paper grading on mathematics olympiads.

**Answer-verifiable benchmarks (Table 1)**: the model SU-01 scores 77.5% on AnswerBench, 59.8% on AMO-Bench, 94.6% and 93.3% on AIME 2025 and AIME 2026 respectively, and 62.5% on FrontierScience-Olympiad. The overall average is 77.3%, essentially tied with the same-scale Qwen3.6-35B-A3B at 77.4%. This means SU-01 does not trade away accuracy on traditional short-answer problems to acquire proof ability, a balance that prior "specialized" models have rarely managed.

**Proof-level benchmarks (Table 3)**: IMO-ProofBench scores 57.6% in direct inference (77.1% on basic problems, 38.1% on advanced), rising to 70.2% with TTS (91.0% basic, 49.5% advanced). The harder FrontierScience-Research set reaches only 11.7%. The TTS gains come mostly from advanced problems, indicating that the self-review mechanism is most useful on the problems that actually test rigor.

**Physics olympiads (Table 2)**: IPhO 2024 scores 23.5 points (25.3 with TTS) against a gold-medal threshold of 20.8; IPhO 2025 scores 20.3 (21.7 with TTS) against a gold-medal threshold of 19.7. Both editions clear the gold-medal line.

**Full-paper math olympiads (Table 4, human-graded)**:

- IMO 2025: 21 points in direct inference (bronze threshold 19, gold threshold 35), reaching the gold-medal threshold of 35 points with TTS. By problem: P1 through P5 full marks, P6 zero.
- USAMO 2026: 15 points in direct inference (bronze threshold 11, gold threshold 25), reaching 35 with TTS. P1, P3, P4, P5, P6 full marks, P2 zero. The paper notes that this 35-point USAMO 2026 score "matches the highest reported score among 340 human contestants."

**Ablation (Figure 4 and Figure 6)**: starting from P1-30B, AnswerBench is 69.2% and advanced proofs only 6.2%; after SFT, AnswerBench temporarily drops to 59.8% (an intentional "reshaping" phase) while advanced proofs rise to 14.8%; after Coarse RL, AnswerBench returns to 77.2% and advanced proofs reach 25.2%; after Refined RL, AnswerBench is 77.5% and advanced proofs 38.1%. The three stages contribute +8.6, +10.4, and +12.9 percentage points to advanced proofs respectively, a nearly even progression.

The curriculum-order ablation is even more telling. On AnswerBench, random order scores 39.5%, reverse (high to low PPL) scores 55.8%, and forward (low to high) only 24.3%. The truncation rate (the fraction of generations forcibly cut off for exceeding the maximum length) is 7.3% for random order, 0.3% for reverse, and far worse for forward. **The reverse PPL curriculum is not an optional minor trick but one of the decisive factors in whether the recipe works at all.**

The last comparison worth flagging is training scale. SU-01 uses 338K SFT trajectories trained for 4 epochs, plus 25K RL prompts trained for 200 steps. By contrast, DeepSeek-V3.2 consumed 943.7 billion tokens during its continued-training stage, and Nemotron-Cascade-2 consumed 26.6 million SFT samples across nine categories. In other words, SU-01's training data is more than two orders of magnitude smaller than comparable systems, yet it achieves equal or better results on olympiad benchmarks.

## IV. Limitations

The authors themselves point out two classes of failure cases. On IMO 2025's P6, the model produced an "invalid column-permutation reduction" and scored zero on the entire problem. On USAMO 2026's P2, the model "failed to preserve a finely-tuned process invariant," again scoring zero. The authors attribute both failures to a common pattern: **the model handles problems that reduce to rigid formal representations well, but is unreliable on problems whose core challenge lies in preserving combinatorial structure or proving a delicate process invariant.** The observation points toward future work: combinatorics and discrete process invariants remain the hard bones of rigorous LLM reasoning.

Several latent issues the authors do not state explicitly are also visible in retrospect. **First**, the entire training pipeline relies heavily on external strong models as reward signals. The generative reward comes from DeepSeekMath-V2, the verification during self-correction comes from gpt-oss-120b, and the self-verification trajectories in the SFT data are generated by DeepSeek-V3.2-Speciale. If the training cost of these external strong models were folded into the accounting, the claim that "training data is more than two orders of magnitude smaller than the baselines" would need a discount. **Second**, under TTS a single problem requires 200K tokens of inference, with corresponding cost and latency that are far from "deployable in production." Offline leaderboard scores look clean, but in a setting like ChatGPT where users expect responses within seconds, the recipe is entirely impractical. **Third**, the reported full-paper scores on IMO and USAMO depend on human grading. The grading procedure is disclosed, but neither the full rubric nor inter-rater agreement metrics are provided, so independent external reproduction of these two specific scores would require considerable resources. **Fourth**, the 30B-A3B sparse-activation architecture is itself a relatively niche choice, and the paper does not discuss how the recipe transfers to dense 30B or larger sparse models; the generality of the recipe remains to be verified.

## One Sentence

A simple pipeline of "reverse-PPL-curriculum SFT plus two-phase RL plus test-time self-checking" pushes a 30B sparse model to the gold-medal threshold at both IMO and IPhO.
