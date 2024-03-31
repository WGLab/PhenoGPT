import pandas as pd
import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import os
import sys
from typing import List

from peft import PeftModel
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
import re
from itertools import chain
import torch
from datasets import load_dataset
import pandas as pd
import json, glob
import argparse
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
parser = argparse.ArgumentParser(description="PhenoGPT Medical Term Detector",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required = True, help="directory to input folder")
parser.add_argument("-o", "--output", required = True, help="directory to output folder")
args = parser.parse_args()
## please replace the following lines as your directories to the Llama 2 7B base model & Lora-weight training above
# BASE_MODEL = "/mnt/isilon/wang_lab/jingye/projects/gpt/llama_hf"
# lora_weights = '/mnt/isilon/wang_lab/jingye/projects/PhenoGPT/llama/experiments'
BASE_MODEL = os.getcwd() + "/model/llama2/llama_hf/"
lora_weights = os.getcwd() + './model/llama2/lora_weights/'
load_8bit = False
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"
generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.5)
def remove_hpo(text):
    # Define the pattern to match HP:XXXXXXX
    pattern = r'\bHP\w*\b'
    # Replace matched patterns with an empty string
    cleaned_text = re.sub(pattern, '', text)
    #cleaned_text = re.sub(pattern2, '', cleaned_text)
    return cleaned_text
def generate_output(text):
    prompt = f"""Input: {text}
    ### Response:
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    # model.to(DEVICE)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=300,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    if len(input_ids[0]) > 2048:
        print("WARNING: Your text input has more than the predefined maximum 2048 tokens. The results may be defective.")
    return(output)
def clean_output(output):
    output = remove_hpo(output)
    if "### Response:":
        output = output.split("### Response:")[-1].split("\n")
        if len(output) > 0:
            output_clean = [t.split("|") for t in output]
            output_clean = list(set(chain(*output_clean)))
            output_clean = [re.sub(r'^[\s\W]+|[\s\W]+$', '', t) for t in output_clean if not t.strip().startswith("END")]
            output_clean = [t for t in output_clean if t]
        else:
            print("No medical terms were detected")
            output_clean = []
    else:
        print("No medical terms were detected")
        output_clean = []
    return(output_clean)
def read_text(input_file):
    if ".txt" in input_file:
        input_list=[input_file]
    else:
        input_list = glob.glob(input_file + "/*.txt")
    input_dict = {}
    for f in input_list:
        file_name = f.split('/')[-1][:-4]
        with open(f, 'r') as r:
            data = r.readlines()
            if len(data) > 1:
                data = "\n".join(data)
        input_dict[file_name] = data
    return(input_dict)
def main():
    # set up model
    model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=load_8bit,
            device_map = "auto"
        )
    model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    input_dict = read_text(args.input) 
    for file_name, text in input_dict.items():
        # generate raw response
        output = generate_output(text)
        # clean up response
        output_clean = clean_output(output)
        # save output
        with open(args.output+"/"+file_name+"_phenogpt.txt", 'w') as f:
            for o in output_clean:
                f.write(o+"\n")
        print(output_clean)

if __name__ == "__main__":
    main()