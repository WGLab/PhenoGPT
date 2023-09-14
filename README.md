# PhenoGPT

PhenoGPT is an advanced phenotype recognition model, leveraging the robust capabilities of large language models. It employs a fine-tuned implementation on the publicly accessible [BiolarkGSC+ dataset](https://github.com/lasigeBioTM/IHP), to enhance prediction accuracy and alignments. Like GPT's broad utilization, PhenoGPT can process diverse clinical abstracts for improved flexibility. For enhanced model precision and specialization, you have the option to further fine-tune the proposed PhenoGPT model on your own clinical datasets. This process is elaborated in the subsequent [section](##Fine-tuning).

PhenoGPT is distributed under the [MIT License by Wang Genomics Lab](https://wglab.mit-license.org/).

## Installation
We need to install the required packages for model fine-tuning and inference. 
```
!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
!pip install -q datasets bitsandbytes einops
```
In the command above, we utilize the accelerate package for model sharding. PEFT package is used for efficient fine-tuning like LORA.
bitsandbytes package is used for model quantization.

To use LLaMA model, please apply for access first and download it into the local drive. [Instruction](https://huggingface.co/docs/transformers/main/model_doc/llama)

## Reproduce Model
You can reproduce PhenoGPT model via OpenAI using BiolarkGSC+ dataset. We recommend to follow this [notebook](https://github.com/WGLab/PhenoGPT/blob/main/PhenoGPT.ipynb) script for details.

## Fine-tuning
You can reproduce PhenoGPT model with different base models on BiolarkGSC+ dataset. To fine-tune a specialized phenotype recognition language model, we recommend to follow this [notebook](https://github.com/WGLab/PhenoGPT/blob/main/PhenoGPT.ipynb) script for details.

## Inference
If you want to simply implement PhenoGPT on your local machine for  inference, the fine-tuned models are saved in the [model](https://github.com/WGLab/PhenoGPT/tree/main/model) directory. Please follow the inference section of the script to run your model.
