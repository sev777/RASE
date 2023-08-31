import copy
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)

from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre,eval_gpt
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
import tqdm
ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

'''
REPLACE the original evaluate in MEMIT (https://github.com/kmeng01/memit)

'''


def get_proj_for_bert(self,**kwargs):
    with torch.no_grad():
        inputs=self.tokenizer([kwargs['src']],return_tensors="pt",
            padding=True,
            max_length=32,
            truncation=True,)
        input_fact_src_input_ids=inputs['input_ids']
        input_fact_src_attention_mask=inputs['attention_mask']

        src_emb = self.encode_model(**{'input_ids': input_fact_src_input_ids.to(self.encode_model.device),
                                       'attention_mask': input_fact_src_attention_mask.to(self.encode_model.device)},
                                    output_hidden_states=True, return_dict=True).pooler_output
        if 'facts' in kwargs:
            inputs = self.tokenizer([kwargs['facts']], return_tensors="pt",
                                    padding=True,
                                    max_length=32,
                                    truncation=True, )
            input_fact_input_ids = inputs['input_ids']
            input_fact_attention_mask = inputs['attention_mask']
            fact_emb = self.encode_model(**{'input_ids': input_fact_input_ids.to(self.encode_model.device),
                                            'attention_mask': input_fact_attention_mask.to(self.encode_model.device)},
                                         output_hidden_states=True, return_dict=True).pooler_output

            return self.sen_base_proj(src_emb), self.rel_base_proj(fact_emb)
        else:

            return self.sen_base_proj(src_emb), None
import numpy as np
def edit_cls(inds, xxx, editor):

    return (inds[0][0].cpu().item() - inds[0][1].cpu().item() > 0.15 and np.std(inds[0].cpu().tolist()) > 0.05) or inds[0][0].cpu().item()>0.9, \
        inds[1][0].detach().cpu().item()

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def juge_apply(model,record,mem_cache,keys,md):
    #  model,is_edit = juge_apply(model,record,mem_cache)
    query,_= get_proj_for_bert(md, **{
            'src':   record["requested_rewrite"]['prompt'].format(record["requested_rewrite"]['subject']),
        })
    inds = torch.topk(torch.nn.functional.cosine_similarity(query, keys, dim=1), 6)
    if_edit,choose=(inds[0][0].cpu().item() - inds[0][1].cpu().item() > 0.15 and np.std(inds[0].cpu().tolist()) > 0.05) or inds[0][0].cpu().item()>0.9, \
        inds[1][0].detach().cpu().item()
    if if_edit:
        deltas = mem_cache[1][choose]

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                w[...] += upd_matrix.to(w.device)

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model,if_edit
def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    use_retri: int = 0,
batched: int = 0
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    # if (
    #     continue_from_run is None
    #     or not (run_dir = RESULTS_DIR / dir_name / continue_from_run).exists()
    # ):
    #     continue_from_run = None
    # if continue_from_run is None:
    alg_dir = RESULTS_DIR / dir_name
    if alg_dir.exists():
        id_list = [
            int(str(x).split("_")[-1])
            for x in alg_dir.iterdir()
            if str(x).split("_")[-1].isnumeric()
        ]
        run_id = 0 if not id_list else max(id_list) + 1
    else:
        run_id = 0
    run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)

    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")


    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained('../LMs/huggingface/gpt2-xl').cuda()
        # model = AutoModelForCausalLM.from_pretrained("/root/data/LMs/GPTJ").cuda()
        tok = AutoTokenizer.from_pretrained('../LMs/huggingface/gpt2-xl')
        # tok = AutoTokenizer.from_pretrained("/root/data/LMs/GPTJ")
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data",DATA_DIR)
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]

    ds_ed = ds_class('./data/zsre-edit.jsonl', tok=tok, size=dataset_size_limit)
    ds_tain = ds_class('./data/zsre-train.jsonl', tok=tok, size=dataset_size_limit)
    ds_test = ds_class('./data/zsre-dev-kilt_hd.jsonl', tok=tok, size=dataset_size_limit)


    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")
    ###################
    #eval before edit##
    ###################
    gen_test_vars = [snips, vec]
    args_conserve_memory = (
        dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
        if conserve_memory
        else dict()
    )
    etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()
    #
    before_train,before_dev=[],[]
    for ri, record in enumerate(ds_test):
        print(f'ds_test There are {len(ds_test)} data, we are calculate the {ri} data.',flush=True)
        pre_res = ds_eval_method(
            model,
            tok,
            record,
            *(
                gen_test_vars
                if record["case_id"] % generation_test_interval == 0
                else [None, None]
            ),  # Only test generation every generation_test_interval cases
        )
        resu = sum(pre_res['rewrite_prompts_correct']) / len(pre_res['rewrite_prompts_correct'])
        before_dev.append(resu)
    print(f"before dev res is {sum(before_dev)/len(before_dev)}",flush=True)
    for ri, record in enumerate(ds_tain):
        print(f'ds_tain There are {len(ds_tain)} data, we are calculate the {ri} data.')
        pre_res = ds_eval_method(
            model,
            tok,
            record,
            *(
                gen_test_vars
                if record["case_id"] % generation_test_interval == 0
                else [None, None]
            ),
        )
        resu = sum(pre_res['rewrite_prompts_correct']) / len(pre_res['rewrite_prompts_correct'])
        before_train.append(resu)
    print(f"before train res is {sum(before_train) / len(before_train)}",flush=True)
    #
    if batched==1:

        edited_model=None
        before_res=[]
        after_res=[]
        ds=ds_ed
        edited=0
        ed_data=[]
        for ri, record in enumerate(ds_ed):
            print(f'There are {len(ds_ed)} data, we are calculate the {ri} data.', flush=True)

            pre_res = ds_eval_method(
                model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
            )
            resu = sum(pre_res['rewrite_prompts_correct']) / len(pre_res['rewrite_prompts_correct'])
            print(resu)
            before_res.append([ri, resu])

            # edit
            if resu < 1:
                ed_data.append( [
                         record["case_id"], record["requested_rewrite"]
                    ])
                if len(ed_data)>num_edits:
                    print('break cause the:',num_edits)
                    break

                #APPLY THE EDITS

                print(f'APPLY edits {len(ed_data)}')
                edited_model, weights_copy = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record[0], **record[1]}
                        for record in ed_data
                    ],
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )
        # for ri, record in enumerate(ds_ed):
                aft_res = ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),
                )
                aft_res_acc, aft_res_re = sum(aft_res['rewrite_prompts_correct']) / len(
                    aft_res['rewrite_prompts_correct']), sum(aft_res['paraphrase_prompts_correct']) / len(
                    aft_res['paraphrase_prompts_correct'])
                after_res.append([ri, aft_res_acc, aft_res_re])


        after_ed_res = []
        for ri, record in enumerate(ds_ed):
            print(f'ds_ed There are {len(ds_ed)} data, we are calculate the {ri} data.', flush=True)
            pre_res = ds_eval_method(
                edited_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
            )

            aft_res_acc, aft_res_re = sum(pre_res['rewrite_prompts_correct']) / len(
                pre_res['rewrite_prompts_correct']), sum(pre_res['paraphrase_prompts_correct']) / len(
                pre_res['paraphrase_prompts_correct'])

            after_ed_res.append([ri, aft_res_acc, aft_res_re])
        # print(f"after  edit res is {sum(after_ed_res) / len(after_ed_res)}")

        after_train, after_dev = [], []
        for ri, record in enumerate(ds_test):
            print(f'ds_test There are {len(ds_test)} data, we are calculate the {ri} data.', flush=True)
            pre_res = ds_eval_method(
                edited_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
            )
            resu = sum(pre_res['rewrite_prompts_correct']) / len(pre_res['rewrite_prompts_correct'])
            after_dev.append(resu)
        print(f"after  after_dev res is {sum(after_dev) / len(after_dev)}", flush=True)

        for ri, record in enumerate(ds_tain):
            print(f'ds_tain There are {len(ds_tain)} data, we are calculate the {ri} data.', flush=True)
            pre_res = ds_eval_method(
                edited_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),
            )
            resu = sum(pre_res['rewrite_prompts_correct']) / len(pre_res['rewrite_prompts_correct'])
            after_train.append(resu)
        print(f"after  after_train res is {sum(after_train) / len(after_train)}", flush=True)

        for i, j in zip([before_train, before_dev], [after_train, after_dev]):
            bef = sum(i) / len(i)
            after = sum(j) / len(j)
            print(f'RETAIN (TRAIN and dev): {after}/{bef}={after / bef}', flush=True)
        SR = sum([i[1] for i in after_res]) / len([i[1] for i in after_res])
        after_res_ind = [i[0] for i in after_res]
        edits_data = [i for i in after_ed_res if i[0] in after_res_ind]
        ER = sum([i[1] for i in edits_data]) / len([i[1] for i in edits_data])
        GR = sum([i[2] for i in edits_data]) / len([i[2] for i in edits_data])
        GR_S = sum([i[2] for i in after_res]) / len([i[2] for i in after_res])
        print(f'SR：{SR}, ER: {ER}, GR:{GR} GR_S：{GR_S}', flush=True)
        # before_res
        # after_res
        # after_ed_res
        out_file = Path(str(run_dir / "{}_edits-res.pkl").format(num_edits))

        if out_file.exists():
            print(f"Skipping {out_file}; already exists")

        torch.save({
            'before_train': before_train,
            'before_dev': before_dev,
            'before_res': before_res,
            'after_train': after_train,
            'after_dev': after_dev,
            'after_res_edit': after_res,  # edit res
            'after_res': after_ed_res,  # edit dataset res
        }, out_file)
        print('save to ', out_file, flush=True)

        pass
    else:
        if use_retri==1:

            #load the CL model
            # from experiments.mine.cl_model import cl_for_fact, laod_model
            # md = laod_model()
            # try:
            #     md.load_state_dict(torch.load('/root/data/Transformer-Patcher-main/Retrieval_based_editing/CL_PARAM.ckpt'))
            # except:
            #     md.load_state_dict(torch.load('/root/sev777/Transformer-Patcher-main/Retrieval_based_editing/CL_PARAM.ckpt'))
            # tokenizer = md.tokenizer

            # load the contriever
            # md = AutoModel.from_pretrained("contriever-msmarco").cuda()
            # tokenizer= AutoTokenizer.from_pretrained("contriever-msmarco")

            mem_cache=[[],[],[]]#key,vale,ori

        edited_model=None
        before_res=[]
        after_res=[]
        ds=ds_ed
        edited=0

        for ri,record in enumerate(ds):
            print(f'There are {len(ds)} data, we are calculate the {ri} data.',flush=True)
            pre_res=ds_eval_method(
                model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
            )
            resu=sum(pre_res['rewrite_prompts_correct']) / len(pre_res['rewrite_prompts_correct'])
            print(resu)
            before_res.append([ri,resu])

            #edit
            if resu<1:

                edited+=1
                print(f'data {ri} need editing !!',flush=True)

                start = time()
                # rewrite_with=record["requested_rewrite"]
                # rewrite_with['prompt']=record['facts_prompt']+' || '+  rewrite_with['prompt']
                # record['paraphrase_prompts']=[record['facts_prompt']+' || '+ii  for ii in record['paraphrase_prompts']]
                edited_model, weights_copy = apply_algo(
                    model if (use_retri==1 or edited_model==None) else edited_model,
                    tok,
                    [
                        {"case_id": record["case_id"], **record["requested_rewrite"]}
                    ],
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )
                exec_time = time() - start
                print("Execution took", exec_time,flush=True)
                if use_retri==1:
                    sen_proj, fact_proj = get_proj_for_bert(md, **{
                        'src': record["requested_rewrite"]['prompt'].format(record["requested_rewrite"]['subject']),
                        'facts': record['facts']

                    })
                    # sims = torch.nn.functional.cosine_similarity(sen_proj, fact_proj, dim=1)

                    mem_cache[0].append(fact_proj.detach().cpu())
                    mem_cache[1].append(deltas)



                    aft_res=ds_eval_method(
                        edited_model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),
                    )
                    aft_res_acc,aft_res_re=sum(aft_res['rewrite_prompts_correct']) / len(aft_res['rewrite_prompts_correct']),sum(aft_res['paraphrase_prompts_correct']) / len(aft_res['paraphrase_prompts_correct'])
                    after_res.append([ri,aft_res_acc,aft_res_re])
                    # Restore original weights
                    print(f'***** Restore original weights **********', flush=True)
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(model, k)[...] = v.to("cuda")
                    if  mem_cache[2]==[]:
                        mem_cache[2]={ i:j.detach().cpu() for i,j in weights_copy.items()}
                    if edited>num_edits:
                        print(f'editing {num_edits} edits ', flush=True)
                        break
                else:

                    aft_res = ds_eval_method(
                        edited_model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),
                    )
                    aft_res_acc, aft_res_re = sum(aft_res['rewrite_prompts_correct']) / len(
                        aft_res['rewrite_prompts_correct']), sum(aft_res['paraphrase_prompts_correct']) / len(
                        aft_res['paraphrase_prompts_correct'])
                    after_res.append([ri, aft_res_acc, aft_res_re])

                    if edited > num_edits:
                        print(f'editing {num_edits} edits ', flush=True)
                        break
        if use_retri==0:
            after_ed_res=[]
            for ri,record in enumerate(ds_ed):

                print(f'ds_ed There are {len(ds_ed)} data, we are calculate the {ri} data.',flush=True)
                pre_res=ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                )

                aft_res_acc, aft_res_re = sum(pre_res['rewrite_prompts_correct']) / len(
                    pre_res['rewrite_prompts_correct']), sum(pre_res['paraphrase_prompts_correct']) / len(
                    pre_res['paraphrase_prompts_correct'])

                after_ed_res.append([ri,aft_res_acc,aft_res_re])
            # print(f"after  edit res is {sum(after_ed_res) / len(after_ed_res)}")


            after_train,after_dev=[],[]
            for ri, record in enumerate(ds_test):
                print( f'ds_test There are {len(ds_test)} data, we are calculate the {ri} data.',flush=True)
                pre_res = ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                )
                resu = sum(pre_res['rewrite_prompts_correct']) / len(pre_res['rewrite_prompts_correct'])
                after_dev.append(resu)
            print(f"after  after_dev res is {sum(after_dev) / len(after_dev)}",flush=True)


            for ri, record in enumerate(ds_tain):
                print(f'ds_tain There are {len(ds_tain)} data, we are calculate the {ri} data.',flush=True)
                pre_res = ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),
                )
                resu = sum(pre_res['rewrite_prompts_correct']) / len(pre_res['rewrite_prompts_correct'])
                after_train.append(resu)
            print(f"after  after_train res is {sum(after_train) / len(after_train)}",flush=True)

            for i,j in zip([before_train,before_dev],[after_train,after_dev]):
                bef=sum(i)/len(i)
                after=sum(j)/len(j)
                print(f'RETAIN (TRAIN and dev): {after}/{bef}={after/bef}',flush=True)
            SR=sum([ i[1] for i in after_res])/len([ i[1] for i in after_res])
            after_res_ind=[i[0] for i in after_res]
            edits_data=[i  for i in after_ed_res if i[0] in after_res_ind]
            ER=sum([ i[1] for i in edits_data])/len([ i[1] for i in edits_data])
            GR=sum([ i[2] for i in edits_data])/len([ i[2] for i in edits_data])
            GR_S = sum([i[2] for i in after_res]) / len([i[2] for i in after_res])
            print(f'SR：{SR}, ER: {ER}, GR:{GR} GR_S：{GR_S}',flush=True)
            # before_res
            # after_res
            # after_ed_res
            out_file = Path(str(run_dir / "{}_edits-res.pkl").format(num_edits))

            if out_file.exists():
                print(f"Skipping {out_file}; already exists")

            torch.save({
                'before_train':before_train,
                'before_dev':before_dev,
                'before_res':before_res,
                'after_train':after_train,
                'after_dev':after_dev,
                'after_res_edit':after_res,#edit res
                'after_res':after_ed_res,#edit dataset res
            },out_file)
            print('save to ',out_file,flush=True)
        else:
            keys=torch.cat(mem_cache[0])
            after_ed_res = []
            for ri, record in enumerate(ds_ed):
                print(f'ds_ed There are {len(ds_ed)} data, we are calculate the {ri} data.', flush=True)
                #juge if eedit
                model,is_edit = juge_apply(model,record,mem_cache,keys,md)
                #eval edit
                pre_res = ds_eval_method(
                    model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                )

                aft_res_acc, aft_res_re = sum(pre_res['rewrite_prompts_correct']) / len(
                    pre_res['rewrite_prompts_correct']), sum(pre_res['paraphrase_prompts_correct']) / len(
                    pre_res['paraphrase_prompts_correct'])

                after_ed_res.append([ri, aft_res_acc, aft_res_re,is_edit])
                #resotr
                print(f'***** Restore original weights **********', flush=True)
                with torch.no_grad():
                    for k, v in mem_cache[2].items():
                        nethook.get_parameter(model, k)[...] = v.to("cuda")


            after_train, after_dev = [], []
            for ri, record in enumerate(ds_test):
                print(f'ds_test There are {len(ds_test)} data, we are calculate the {ri} data.', flush=True)

                #juge if eedit
                model,is_edit = juge_apply(model,record,mem_cache,keys,md)
                #edit
                pre_res = ds_eval_method(
                    model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                )
                resu = sum(pre_res['rewrite_prompts_correct']) / len(pre_res['rewrite_prompts_correct'])

                after_dev.append([resu,is_edit])
                #resotr
                print(f'***** Restore original weights **********', flush=True)
                with torch.no_grad():
                    for k, v in mem_cache[2].items():
                        nethook.get_parameter(model, k)[...] = v.to("cuda")


            print(f"after  after_dev res is {sum([i for i in after_dev[0]]) / len(after_dev)}", flush=True)

            for ri, record in enumerate(ds_tain):
                print(f'ds_tain There are {len(ds_tain)} data, we are calculate the {ri} data.', flush=True)

                #juge if eedit
                model,is_edit = juge_apply(model,record,mem_cache,keys,md)
                #edit

                pre_res = ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),
                )
                resu = sum(pre_res['rewrite_prompts_correct']) / len(pre_res['rewrite_prompts_correct'])
                after_train.append([resu,is_edit])

                #resotr
                print(f'***** Restore original weights **********', flush=True)
                with torch.no_grad():
                    for k, v in mem_cache[2].items():
                        nethook.get_parameter(model, k)[...] = v.to("cuda")


            print(f"after  after_train res is {sum([i for i in after_train[0]]) / len(after_train)}", flush=True)

            for i, j in zip([before_train, before_dev], [after_train, after_dev]):
                shot=[jj[1] for jj in j]
                j=[jj[0] for jj in j]

                bef = sum(i) / len(i)
                after = sum(j) / len(j)
                print(f'RETAIN (TRAIN and dev): {after}/{bef}={after / bef}', flush=True)
                print(f'SHOT (TRAIN and dev): {sum(shot)}/{len(shot)}={sum(shot) / len(shot)}', flush=True)

            SR = sum([i[1] for i in after_res]) / len([i[1] for i in after_res])
            after_res_ind = [i[0] for i in after_res]
            edits_data = [i for i in after_ed_res if i[0] in after_res_ind]
            shots = [i[3] for i in after_ed_res if i[0] in after_res_ind]
            ER = sum([i[1] for i in edits_data]) / len([i[1] for i in edits_data])
            GR = sum([i[2] for i in edits_data]) / len([i[2] for i in edits_data])
            print(f'SR：{SR}, ER: {ER}, GR:{GR}, SHOT: {sum(shots)/len(shots)}', flush=True)
            # before_res
            # after_res
            # after_ed_res
            out_file = Path(str(run_dir / "{}_edits-res.pkl").format(num_edits))

            if out_file.exists():
                print(f"Skipping {out_file}; already exists")

            torch.save({
                'before_train': before_train,
                'before_dev': before_dev,
                'before_res': before_res,
                'after_train': after_train,
                'after_dev': after_dev,
                'after_res_edit': after_res,  # edit res
                'after_res': after_ed_res,
                'mem':mem_cache # edit dataset res
            }, out_file)
            print('save to ', out_file, flush=True)

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        # required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        # required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json"
                ,
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        # required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="zsre",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=10,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_retri",
        type=int,
        default=1,
        help="use_retri==1: SME with ROME or MEMIT; ==0: SME with retrieval",
    )
    parser.add_argument(
        "--batched",
        type=int,
        default=1,
        help="batched==1: eval afer edits N data. ==0: eval after every edits",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        use_retri=args.use_retri,
    batched=args.batched
    )
    print('DONE')
