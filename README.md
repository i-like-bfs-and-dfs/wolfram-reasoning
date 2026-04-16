# Incentivizing Wolfram-based thinking in VLMs via reinforcement learning
* I'm experimenting with methods to incentivize Wolfram-based representation and reasoning of visual math concepts in Qwen3-VL-2B-Instruct model via post-training paradigms including cold-start SFT, in-context learning, CoT reasoning, and GRPO-based RL exploration.
* The PoC results show a 3.33% improvement in accuracy while reducing reasoning tokens by 75% over Python-based reasoning.
* Although this project was limited in compute with only 4x Nvidia H200 GPU nodes, further improvements in distributed training via tensor parallelism, context parallelism could yield a larger search space. 

## Thinking in Wolfram
<img width="1170" height="297" alt="image" src="https://github.com/user-attachments/assets/1ef96c5a-0be7-4d95-8601-b358e38c60dd" />

**Figure**: Thinking math concepts with Wolfram Language and Engine.

## Overview of training methodology
<img width="671" height="562" alt="image" src="https://github.com/user-attachments/assets/f7cc8a11-fbe5-41f5-824c-b52591bdff7e" />

**Figure**: For each prompt, G = 10 output sequences from the model are explored, with advantages calculated amongst these outputs based on the reward model. LoRA (injected into each attention layer in the base model) is used for GRPO updates.

## Training script
https://github.com/i-like-bfs-and-dfs/wolfram-reasoning/blob/main/src/train.py

## Dataset
A subset of [🤗 ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) introduced in [VL-Rethinker](https://neurips.cc/virtual/2025/loc/san-diego/poster/119918) paper (see results).

## Latest results (WIP)
<img width="804" height="233" alt="image" src="https://github.com/user-attachments/assets/91d8ab2a-1916-4b7f-a458-bfc8226a3bb9" />

**Table**: Evaluation results with in-context learning and GRPO-based exploration and fine-tuning. Code column denotes fraction of outputs with Wolfram code. No Error column denotes fraction of
outputs with error-free Wolfram code. Accuracy column denotes fraction of correct answers after deduction from Wolfram Engine. The prompt and
output token lengths are averaged category-wise, with mean and std. dev. mentioned. 

* Although fraction of error-free code is high, further optimization is essential (increasing G, batch size, number of epochs, etc.) to increase accuracy. 

## References

1. D. Guo et al. "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning." *Nature*, 2025.
2. T. H. Trinh et al. "Solving olympiad geometry without human demonstrations." *Nature*, 2024.
3. T. Hubert et al. "Olympiad-level formal mathematical reasoning with reinforcement learning." *Nature*, 2025.
4. S. Bai et al. "Qwen3-VL Technical Report." 2025.
5. T. Kojima et al. "Large Language Models are Zero-Shot Reasoners." *NeurIPS*, 2022.
6. T. B. Brown et al. "Language Models are Few-Shot Learners." 2020.
7. T. Schick et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." *NeurIPS*, 2023.
8. D. Das et al. "MathSensei: Mathematical Reasoning with a Tool-Augmented Large Language Model." *ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models*, 2024.
9. T. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS*, 2023.
10. E. J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.
11. H. Wang et al. "VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning." *NeurIPS*, 2025.
