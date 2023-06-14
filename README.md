# PhenoGPT

PhenoGPT is an advanced phenotype recognition model, leveraging the robust capabilities of GPT-3. It employs a fine-tuned implementation on the publicly accessible [BiolarkGSC+ dataset](https://github.com/lasigeBioTM/IHP), to enhance prediction accuracy and alignments. Like GPT-3's broad utilization, PhenoGPT can process diverse clinical abstracts for improved flexibility. For enhanced model precision and specialization, you have the option to further fine-tune the proposed PhenoGPT model utilizing OpenAI API packages. This process is elaborated in the subsequent [section](##Fine-tuning).

PhenoGPT is distributed under the [MIT License by Wang Genomics Lab](https://wglab.mit-license.org/).

## Installation
Since our model was hosted on the OpenAI cloud, the main package to install is the [OpenAI API](https://platform.openai.com/docs/api-reference).
In your Python/Conda environment, run
```
pip install openai
```
to install OpenAI API.

On the OpenAI [website](https://platform.openai.com/account/api-keys), please require a personal API token for model usage.

**Remember that your API key is a secret! Every instance of usage and inference through PhenoGPT will incur a fee as stipulated in the OpenAI [documentation](https://openai.com/pricing). Our fine-tuned model is baesd on Davinci.**

To establish your OPENAI_API_KEY environment variable, incorporate the subsequent line into your shell initialization script (for instance, .bashrc, zshrc, etc.). Alternatively, you can execute it in the command line prior to running the fine-tuning command.
```
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```
## Reproduce Model
You can reproduce PhenoGPT model via OpenAI using BiolarkGSC+ dataset. We recommend to follow this [notebook](https://github.com/WGLab/PhenoGPT/blob/main/PhenoGPT.ipynb) script for details.

Once you successfully fine-tuned a PhenoGPT model, it will generate a model id like `davinci:ft-personal:phenogpt-2023-03-20-04-23-25`. You can save this model id for future use. Notice that the model will be affiliated to your API token directly and **not allowed** to share with others. You can build an API upon this for 'public use' if necessary.

## Examples

Please refer to [Example](https://github.com/WGLab/DeepMod2/blob/main/docs/Example.md) detailed instructions on how to run phenogpt on BiolarkGSC+ test dataset.

## Fine-tuning
To fine-tune a specialized phenotype recognition language model, we recommend to follow this [notebook](https://github.com/WGLab/PhenoGPT/blob/main/PhenoGPT.ipynb) script for details.
