---
license: mit
---
# **Electrocardiogram–Language Model for Few-Shot Question Answering with Meta Learning**
 
Electrocardiogram (ECG) interpretation requires specialized expertise, often involving synthesizing insights from ECG signals with complex clinical queries posed in natural language. The scarcity of labeled ECG data coupled with the diverse nature of clinical inquiries presents a significant challenge for developing robust and adaptable ECG diagnostic systems. This work introduces a **novel multimodal meta-learning** method for **few-shot ECG question answering**, addressing the challenge of limited labeled data while leveraging the rich knowledge encoded within **large language models (LLMs)**. Our **LLM-agnostic approach** integrates a pre-trained ECG encoder with a frozen LLM (e.g., LLaMA and Gemma) via a trainable fusion module, enabling the language model to reason about ECG data and generate clinically meaningful answers. Extensive experiments demonstrate superior generalization to unseen diagnostic tasks compared to supervised baselines, achieving notable performance even with limited ECG leads. For instance, in a 5-way 5-shot setting, our method using LLaMA-3.1-8B achieves accuracy of 84.6%, 77.3%, and 69.6% on single verify, choose and query question types, respectively. These results highlight the potential of our method to enhance clinical ECG interpretation by combining signal processing with the nuanced language understanding capabilities of LLMs, particularly in **data-constrained scenarios**.

---

## Features

- **🧩 Task Diversification:** Restructured ECG-QA tasks promote rapid few-shot adaptation.
- **🔗 Fusion Mapping:** A lightweight multimodal mapper bridges ECG and language features.
- **🌐 Model Generalization:** LLM-agnostic design ensures broad transferability and robustness.
 
## Installation

To get started, you can clone this repository and install the necessary dependencies:
 
```bash
git clone https://github.com/Tang-Jia-Lu/FSL_ECG_QA.git

cd FSL_ECG_QA 
```
 
 
### Load the Model
 
#### Download ECG Model Weights

Download the pre-trained ECG model weights (e.g., `checkpoint_ecg.pt`) and place them in the `/ecg_checkpoint/` directory:

```
/ecg_checkpoint/checkpoint_ecg.pt
```

This allows the `load_careqa_model` function to locate and load the model parameters correctly.

#### Download LLM Model Weights

You can download the pre-trained LLaMA-3.1-8B model weights from Hugging Face. For example, visit the [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) repository and follow the instructions to request access and download the files.

After downloading, place the model files (such as `pytorch_model.bin`, `config.json`, etc.) in the `/llm_checkpoint/llama3.1-8b` directory:

```
/llm_checkpoint/llama3.1-8b
```

### Alternative LLM Model Checkpoints

You can also use the following model checkpoints as alternatives. Please refer to their respective documentation for download and usage instructions:

- [llama3.2_1b](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [phi_2](https://huggingface.co/microsoft/phi-2)
- [gamma2-7b](https://huggingface.co/google/gemma-7b)

#### Download Meta-Mapper Weights

Download the pre-trained meta-mapper weights specifically designed for LLaMA-3.1-8B and the single-verify, 5-way 5-shot setting. Place the downloaded model file at the following location:

```
/models/1_5-way_5-shot.pt
```

This checkpoint is tailored for optimal performance with LLaMA-3.1-8B on single-verify tasks in a 5-way 5-shot few-shot learning scenario.

A `2_2-way_5-shot.pt` checkpoint is also provided.


The load_careqa_model function allows you to download and load the pre-trained FSL_ECG_QA model from Hugging Face.
 

```bash
python train.py --experiment_id 1 --n_way 5 --k_spt 5 --k_qry 5
```

- `--n_way`: Number of classes per episode (default: 5)
- `--k_spt`: Number of support examples per class (default: 5)
- `--k_qry`: Number of query examples per class (default: 5)

You can adjust these arguments as needed for your experiments.


 