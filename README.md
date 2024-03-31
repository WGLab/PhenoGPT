# PhenoGPT

PhenoGPT is an advanced phenotype recognition model, leveraging the robust capabilities of large language models. It employs a fine-tuned implementation on the publicly accessible [BiolarkGSC+ dataset](https://github.com/lasigeBioTM/IHP), to enhance prediction accuracy and alignments. Like GPT's broad utilization, PhenoGPT can process diverse clinical abstracts for improved flexibility. For enhanced model precision and specialization, you have the option to further fine-tune the proposed PhenoGPT model on your own clinical datasets. This process is elaborated in the subsequent [section](##Fine-tuning). 

Llama 2 is the default model as it performs the best compared to other models such as GPT-J and Falcon.

PhenoGPT is distributed under the [MIT License by Wang Genomics Lab](https://wglab.mit-license.org/).

## Installation
We need to install the required packages for model fine-tuning and inference. 
```
conda create -n llm_phenogpt python=3.11
conda activate llm_phenogpt
conda install pandas numpy scikit-learn matplotlib seaborn requests joblib
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-12.1" cuda-toolkit
conda install -c conda-forge jupyter
conda install intel-openmp blas
conda install mpi4py
pip install transformers datasets
pip install fastobo sentencepiece einops protobuf
pip install evaluate sacrebleu scipy accelerate deepspeed
pip install git+https://github.com/huggingface/peft.git
pip install flash-attn --no-build-isolation
pip install xformers
pip install bitsandbytes
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=llm_phenogpt
```
In the command above, we utilize the accelerate package for model sharding. PEFT package is used for efficient fine-tuning like LORA.
bitsandbytes package is used for model quantization.
Please pip uninstalll package and pip install package if you encounter any running issues.

## Set Up Model, Input, and Output directories
1. Model:
    - To use LLaMA 2 model, please apply for access first and download it into the local drive. [Instruction](https://huggingface.co/docs/transformers/main/model/llama2)
    - Save model in the ./model/llama2/llama_hf/
    - Download the updated fine-tuning LoRA weights in the release section on GitHub
    - Save LoRA weights in the ./model/llama2/lora_weights/
2. Input:
    - Input files should be txt files
    - Input argument can be either a single txt file or a whole directory containing all input txt files
    - Please see the input and output directories for reference

## Fine-tuning
You can reproduce PhenoGPT model with different base models on BiolarkGSC+ dataset. To fine-tune a specialized phenotype recognition language model, we recommend to follow this [notebook](https://github.com/WGLab/PhenoGPT/blob/main/run_phenogpt.ipynb) script for details. (The notebook is for both llama and falcon model implementation. For gpt-j, please refer to this [script](https://github.com/WGLab/PhenoGPT/blob/main/model/gpt-j/Finetune_gpt_j_6B_8bit_biolark.ipynb).)

## Inference
If you want to simply implement PhenoGPT on your local machine for inference, the fine-tuned models are saved in the [model](https://github.com/WGLab/PhenoGPT/tree/main/model) directory. Please follow the inference section of the [script](https://github.com/WGLab/PhenoGPT/blob/main/inference.py) to run your model.

Please use the following command:
python inference.py -i your_input_folder_directory -o your_output_folder_directory

## Regarding PhenoBCBERT
Since PhenoBCBERT was fine-tuned on the CHOP Proprietary dataset, we cannot publish the model publicly. Please refer to the [paper](https://doi.org/10.1016%2Fj.patter.2023.100887) for results.

## Citation
Yang, J., Liu, C., Deng, W., Wu, D., Weng, C., Zhou, Y., & Wang, K. (2023). Enhancing phenotype recognition in clinical notes using large language models: PhenoBCBERT and PhenoGPT. Patterns (New York, N.Y.), 5(1), 100887. https://doi.org/10.1016/j.patter.2023.100887