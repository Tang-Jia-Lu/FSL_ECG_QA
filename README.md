# Electrocardiogram–Language Model for Few-Shot Question Answering with Meta Learning
<div align="center">

<div>
    <a href='https://tang-jia-lu.github.io/' target='_blank'>Jialu Tang</a><sup>1</sup>&emsp;
    <a href='https://xtxiatong.github.io/' target='_blank'>Tong Xia</a><sup>2</sup>&emsp;
    <a href=https://www.tue.nl/en/research/researchers/yuan-lu' target='_blank'>Yuan Lu</a><sup>1</sup>&emsp;
    <a href='https://www.cl.cam.ac.uk/~cm542/' target='_blank'>Cecilia Mascolo</a><sup>2</sup>&emsp;
    <a href='https://aqibsaeed.github.io/' target='_blank'>Aaqib Saeed</a><sup>1</sup>&emsp;
</div>
<div>
<sup>1</sup><a href="https://www.tue.nl/en/our-university/departments/industrial-design/research/our-research-labs/decentralized-artificial-intelligence-research-lab" target="_blank" rel="noopener noreferrer">
                        Decentralized Artificial Intelligence Research Lab, Eindhoven University of Technology, NL
                      </a>&emsp;

<sup>2</sup>Technology at the University of Cambridge, UK;
</div>
</div>


<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2409.02189-blue?logo=arxiv&logoColor=orange)](https://arxiv.org/html/2410.14464v1)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Citations-purple.svg)](https://scholar.google.nl/scholar?cluster=4185769924704359587&hl=zh-CN&as_sdt=0,5&as_vis=1)
[![Project Page](https://img.shields.io/badge/Project%20Page-Online-brightgreen)](https://tang-jia-lu.github.io/projects/ECG_QA/)
[![AHLI CHIL 2025](https://img.shields.io/badge/IEEE%20CHIL%202025-Accepted-yellow.svg)](https://ieeexplore.ieee.org/document/10888594)

</div>


## 📢 Updates

[10/2024] Arxiv has been released.

[04/2025] Accepted to CHIL 2025

[06/2025] [Code](https://github.com/Tang-Jia-Lu/FSL_ECG_QA), [Model](https://huggingface.co/jialucode/FSL_ECG_QA/), and [Dataset](https://huggingface.co/datasets/jialucode/FSL_ECG_QA_Dataset/) have been released.


## 📝 Abstract
Electrocardiogram (ECG) interpretation requires specialized expertise, often involving synthesizing insights from ECG signals with complex clinical queries posed in natural language. The scarcity of labeled ECG data coupled with the diverse nature of clinical inquiries presents a significant challenge for developing robust and adaptable ECG diagnostic systems. This work introduces a **novel multimodal meta-learning** method for **few-shot ECG question answering**, addressing the challenge of limited labeled data while leveraging the rich knowledge encoded within **large language models (LLMs)**. Our **LLM-agnostic approach** integrates a pre-trained ECG encoder with a frozen LLM (e.g., LLaMA and Gemma) via a trainable fusion module, enabling the language model to reason about ECG data and generate clinically meaningful answers. Extensive experiments demonstrate superior generalization to unseen diagnostic tasks compared to supervised baselines, achieving notable performance even with limited ECG leads. For instance, in a 5-way 5-shot setting, our method using LLaMA-3.1-8B achieves accuracy of 84.6%, 77.3%, and 69.6% on single verify, choose and query question types, respectively. These results highlight the potential of our method to enhance clinical ECG interpretation by combining signal processing with the nuanced language understanding capabilities of LLMs, particularly in **data-constrained scenarios**.

<div align="left">
<h3>⚓ Overview ⚓</h3>
<img src="img/model_structure.gif" width="80%">
<h3>📚 Three Pillars of FedNS</h3>
</div>

- **🧩 Task Diversification:** Restructured ECG-QA tasks promote rapid few-shot adaptation.
- **🔗 Fusion Mapping:** A lightweight multimodal mapper bridges ECG and language features.
- **🌐 Model Generalization:** LLM-agnostic design ensures broad transferability and robustness.

## 🔧 Requirements
###  Environment 

1. **Clone and install [fairseq-signals](https://github.com/Jwoo5/fairseq-signals):**
  ```bash
  git clone https://github.com/Jwoo5/fairseq-signals.git
  cd fairseq-signals
  pip install --editable ./
  ```

2. **Python version:**  
  Python 3.8

3. **Install dependencies:**
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install hydra-core omegaconf
  pip install numpy scipy scikit-learn wfdb
  ```


### 📊 Dataset

The ECG-QA dataset used in this project is available on [Hugging Face](https://huggingface.co/datasets/jialucode/FSL_ECG_QA_Dataset/). Please download the dataset from the link and follow the usage instructions provided in the repository.

### 🏋️ Model Weights

Pre-trained model weights can be downloaded from [Hugging Face - FSL_ECG_QA](https://huggingface.co/jialucode/FSL_ECG_QA/).

## 💡 Running scripts

```bash
python train.py \
  --experiment_id 1 \
  --batchsz_train 10000 \
  --batchsz_test 1000 \
  --paraphrased_path /ecgqa/ptbxl/paraphrased/ \
  --test_dataset ptb-xl \
  --model_name /llm_checkpoint/llama3.1-8b \
  --model_type "" \
  --data_root ecg_qa \
  --data_path /home/jtang/data/ecg_qa_500 \
  --question_type single-verify \
  --epoch 1 \
  --n_way 5 \
  --k_spt 5 \
  --k_qry 5 \
  --prompt 1 \
  --dif_exp 0 \
  --frozen_gpt 1 \
  --frozen_features 1 \
  --repeats 0 \
  --seq_len 30 \
  --seq_len_a 30 \
  --prefix_length 4 \
  --mapper_type MLP \
  --task_num 1 \
  --meta_lr 5e-4 \
  --update_lr 0.05 \
  --update_step 15 \
  --update_step_test 15 \
  --num_workers 8
```

## 🚀 Model Inference

To run inference with all available arguments, use the following command:

```bash
python inference.py \
  --experiment_id 1 \
  --batchsz_test 10 \
  --paraphrased_path ecgqa/ptbxl/paraphrased/ \
  --test_dataset ptb-xl \
  --model_type "" \
  --model_name /gpfs/home1/jtang1/multimodal_fsl_99/mimic_iv_infer/LLARVA/llama3_2_1B/ \
  --question_type single-verify \
  --n_way 5 \
  --k_spt 5 \
  --k_qry 5 \
  --prompt 1 \
  --dif_exp 0 \
  --frozen_gpt 1 \
  --frozen_features 1 \
  --repeats 0 \
  --seq_len 30 \
  --seq_len_a 30 \
  --prefix_length 4 \
  --mapper_type MLP \
  --task_num 1 \
  --update_lr 0.05 \
  --num_workers 8 \
  --meta_lr 5e-4 \
  --update_step 15 \
  --update_step_test 15
```

## 💭 Correspondence
If you have any questions or suggestions, feel free to reach out via [email](mailto:jialu.tang@tue.nl) or open an [issue](https://github.com/tang-jia-lu/ECG_QA/issues) on GitHub.

## Citing FedNS
If you use any content from this repository for your work, please cite the following:

```bibtex
@article{tang2024electrocardiogram,
  title={Electrocardiogram-Language Model for Few-Shot Question Answering with Meta Learning},
  author={Tang, Jialu and Xia, Tong and Lu, Yuan and Mascolo, Cecilia and Saeed, Aaqib},
  journal={arXiv preprint arXiv:2410.14464},
  year={2024}
}
```
