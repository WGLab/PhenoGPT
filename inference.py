import pandas as pd
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import os, sys, re, torch, json, glob, argparse, sent2vec, joblib, nltk
from typing import List
from peft import PeftModel
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from itertools import chain
from datasets import load_dataset
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
nltk.download('stopwords')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
stop_words = set(stopwords.words('english'))
parser = argparse.ArgumentParser(description="PhenoGPT Medical Term Detector",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required = True, help="directory to input folder")
parser.add_argument("-o", "--output", required = True, help="directory to output folder")
parser.add_argument("-id", "--hpoid", choices=['yes', 'no'], default = 'yes', required = False, help="determine if HPO IDs should be predicted")
args = parser.parse_args()
## please replace the following lines as your directories to the Llama 2 7B base model & Lora-weight training above
BASE_MODEL = os.getcwd() + "/model/llama2/llama2_base"
lora_weights = os.getcwd() + '/model/llama2/llama2_lora_weights'
load_8bit = False
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.5)
##set up model
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
hpo_database = joblib.load('hpo_database.json')

def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()

    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

    return ' '.join(tokens)
def remove_hpo(text):
    # Define the pattern to match HP:XXXXXXX
#     pattern1 = r'\bHP_\d{7}\b'
#     pattern2 = r'\bHP:\d{7}\b'
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
def phenogpt_output(raw_output, biosent2vec, termDB2vec, convert2hpo = 'yes'):
    answer_clean = clean_output(raw_output)
    if convert2hpo == 'yes':
        all_terms = list(termDB2vec.keys())
        all_terms_vec = list(termDB2vec.values())
        answers_preprocessed = [preprocess_sentence(txt) for txt in answer_clean]
        answer_vec = biosent2vec.embed_sentences(answers_preprocessed)
        term2hpo = {}
        for i,phenoterm in enumerate(answer_vec):
            all_distances = {}
            dist = []
            for j, ref in enumerate(all_terms_vec):
                dis = distance.cosine(phenoterm, ref)
                if dis > 0:
                    all_distances[all_terms[j]] = 1 - dis
                    dist.append(1-dis)
            if len(dist) != 0:
                matched_pheno = list(all_distances.keys())[np.argmax(dist)]
                hpo_id = hpo_database[matched_pheno]
                term2hpo[answer_clean[i]] = hpo_id
        return term2hpo
    else:
        return answer_clean
def main():
    #please replace your model path here
    biosent2vec_path = './BioSentVec/model/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
    biosent2vec = sent2vec.Sent2vecModel()
    try:
        print("Loading BioSent2Vec model")
        biosent2vec.load_model(biosent2vec_path)
        print('model successfully loaded')
        all_terms = list(hpo_database.keys())
        all_terms_preprocessed = [preprocess_sentence(txt) for txt in all_terms]
        all_terms_vec = biosent2vec.embed_sentences(all_terms_preprocessed)
        ##{Term : Numerical Vector}
        termDB2vec = {k:v for k,v in zip(all_terms, all_terms_vec)}
        print('start phenogpt')
        input_dict = read_text(args.input)
        for file_name, text in input_dict.items():
            try:
                # generate raw response
                raw_output = generate_output(text[0])
                # clean up response
                output = phenogpt_output(raw_output, biosent2vec, termDB2vec, args.hpoid)
                # save output
                with open(args.output+"/"+file_name+"_phenogpt.txt", 'w') as f:
                    if args.hpoid == 'yes':
                        for k,v in output.items():
                            f.write(k+"\t"+v+"\n")
                    else:
                        for t in output:
                            f.write(t+'\n')
                print(output)
            except Exception as e:
                print(e)
                print("Cannot produce results for " + file_name)
    except Exception as e:
        raise ImportError

if __name__ == "__main__":
    main()