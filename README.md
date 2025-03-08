# Misinformation detection at healthcare mews

This project explores the fine-tuning and reinforcement learning (RLHF) of large language models (LLMs) for misinformation detection at healthcare mews. The workflow includes standard fine-tuning, reinforcement learning with human feedback (RLHF) using different optimization strategies, and comparative experiments with BERT-based models as baselines. The datasets primarily focus on misinformation at healthcare news. During this project, we used the following LLMs:

* LLaMA-3.2-1B-Instruct: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct 
* Falcon3-3B-Instruct: https://huggingface.co/tiiuae/Falcon3-3B-Instruct 
* Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct 
* Phi-3.5-Mini-Instruct: https://huggingface.co/microsoft/Phi-3.5-mini-instruct 



## Directory Structure

```
├── dataset
│   ├── data_with_news
│   ├── FakeHealth.csv
│   ├── FakeHealth_test.csv
│   ├── FakeHealth_train.csv
│   ├── HealthRelease.csv
│   ├── HealthStory.csv
│   ├── raw_data
│   ├── ReCOVery.csv
│   ├── ReCOVery_test.csv
│   ├── ReCOVery_train.csv
│   ├── train
│   ├── test
├── finetuning_bert.py
├── finetuning_llms.py
├── finetuning_llms_rlhf_bco.py
├── finetuning_llms_rlhf_cpo.py
├── finetuning_llms_rlhf_reg.py
├── finetuning_llms_rlhf_reg_cv.py
├── mcpo.py
├── notebooks
├── README.md
├── results
├── src
├── standardized_prompting.py
├── assets (To be added for storing artifacts)
```

## Project Phases

1. **Standard LLM Fine-Tuning:**
   - Initial experiments fine-tuning LLMs on the dataset.
   
2. **LLM Fine-Tuning:**
   - More advanced fine-tuning methods applied to improve performance.

3. **RLHF with BCO and CPO:**
   - RLHF using Best-Case Optimization (BCO) and Conservative Policy Optimization (CPO) techniques.

4. **RLHF with Regularization of CPO:**
   - Adding regularization to the CPO method to enhance stability and generalization.

5. **Phi-3.5 Mini Instruct Model with 5-Fold Cross Validation:**
   - A Phi-3.5 Mini-Instruct model is used to perform five-fold cross-validation to evaluate robustness.

6. **Baseline Experiments with BERT Models:**
   - Fine-tuning two BERT models as baseline comparisons against LLM-based approaches.

## Datasets

The `dataset/` directory contains various datasets related to misinformation detection and health news. Key datasets include:
- `FakeHealth.csv`, `FakeHealth_train.csv`, `FakeHealth_test.csv`
- `ReCOVery.csv`, `ReCOVery_train.csv`, `ReCOVery_test.csv`
- `HealthRelease.csv`, `HealthStory.csv`
- `data_with_news`
- `raw_data`
- `train/` and `test/`

## Results and Artifacts

- The `results/` directory contains all experimental outcomes, metrics, and analysis.
- A new `assets/` directory will be added to store artifacts such as model checkpoints, visualizations, and logs.

## Scripts Overview

- `finetuning_llms.py`: Standard fine-tuning for LLMs.
- `finetuning_llms_rlhf_bco.py`: RLHF using BCO.
- `finetuning_llms_rlhf_cpo.py`: RLHF using CPO.
- `finetuning_llms_rlhf_reg.py`: RLHF with regularization on CPO.
- `finetuning_llms_rlhf_reg_cv.py`: Phi-3.5 Mini-Instruct with 5-fold cross-validation.
- `finetuning_bert.py`: Fine-tuning two BERT models as baselines.
- `mcpo.py`: Additional implementation for model evaluation.
- `standardized_prompting.py`: Predefined prompts used in fine-tuning and evaluation.


