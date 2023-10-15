# Fact_retrieval_SME


This repo contains the code and data of EMNLP 2023 accepted paper:

Improving Sequential Model Editing  with Fact Retrieval




## Steps for running Transformer-Patcher

## Directory Structure

```
data/ --> data folder including data of FEVER and zsRE
datasets.py --> load and make the Dataloader for datasets
editor.py --> code for the core editor
models.py -->  base LLM model
seeting.py --> experience setting
train_stage.py --> code for training the  editor
eval.py --> code for evaluate the editor
utils.py --> code for some utils
requirements.txt --> contains required packages
```
### Setup

#### Environment

Our codes are based on Python 3.8.10. Other versions may work as well.

Create a virtual environment and install the dependencies ([conda](https://www.anaconda.com/) can help you do this) :

```
$ conda create --name R-Patcher python=3.8.10
$ conda activate R-Patcher
(R-Patcher)$ pip install -r requirements.txt
```

### Data

The data used is already in [data.zip](https://drive.google.com/drive/folders/1Ago-9aiB9P87hj9OMBPsEsXicVMiDoYP?usp=sharing) file, please unzip this file and implement the following pre-processing steps:



## Running the code

### Training initial models

Before conducting Sequential Model Editing, we first need an initial model.

For fine-tuning a BERT base model on FEVER dataset, you can run:
Follow the [Transformer-Patcher](https://github.com/ZeroYuHuang/Transformer-Patcher) to train the initial model and its checkpoint is saved in `log/models/bert_binary/version_0/checkpoints` or `log/models/bart_seq2seq/version_0/checkpoints` 

```
(R-Patcher)$ python scripts/train_bert_fever.py
```

For fine-tuning a BART base model on zsRE dataset, you can run:

```
(R-Patcher)$ python scripts/train_bart_seq2seq.py
```

### Running Transformer-Patcher

Running R-Patcher requires several arguments:

```
(R-Patcher)$ python train_stage.py 
--setting test 
--model_path *.ckpt 
--box_path *.pkl 
--training True 
--cl_type bert
 --train_way sen_prompt  # sen_prompt means +Pt;  ori menas R-Patcher 
 --log_path ./ZSRE_LOG/testing 
 --editor_tp patch 
 --seed 7 
 --if_rephrase True # if evaluate the GR
 --re_vaild True  # if +Eq in the paper 

```
model_path: the finetuned model to edited.
box_path: the retrieval model. U can use the [contriever-msmarco](https://huggingface.co/facebook/contriever-msmarco) or The CL-based model can get at [Google_drive](https://drive.google.com/file/d/1NEf33sqGuJf-cM2BR6XF-tZoZrj4AW8W/view?usp=sharing)

Evaluate  R-Patcher:
```
python eval.py
--task  fever or zsre 
--setting  *any_name_you_want
--model_path  *.ckpt 

--box_path *.pkl or contriever-msmarco

--cl_type bert
--train_way sen_prompt
--log_path  *your_log_path
--editor_tp
patch

--if_rephrase True # Follow train_stage setting
--re_vaild True # Follow train_stage setting
--weight_path  * # train_stage output


```

# Acknowledgement
This repo benefits from [Transformer-Patcher](https://github.com/ZeroYuHuang/Transformer-Patcher), and [MEMIT](https://github.com/kmeng01/memit). Thanks for their wonderful works.

#
