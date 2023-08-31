import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
#test
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .editor import BartSeq2SeqEditor, BertBinaryEditor
from .settings import args, LOG, log_path
from .datasets import SeqEditDataSet
from .utils import split_data_n_sets, edit_or_not_seq2seq, get_proj_for_bert, edit_or_not_binary

# Retrieval_based_editing
from .models import BertBinary
from  .models import BartSeq2Seq
import numpy as np



def edit_cls(inds, xxx, editor):
    if args.cls_type == 'hard_thre':
        return (inds[0][0].cpu().item() - inds[0][1].cpu().item() > 0.15 and np.std(inds[0].cpu().tolist()) > 0.05) or inds[0][0].cpu().item()>0.9, \
            inds[1][0].detach().cpu().item()

    elif args.cls_type == 'two_way':

        editor_list = [i.detach().cpu().item() for i in inds[1]]
        sen2key = [i for i in inds[0]]
        fact_list = [editor.editor.editors_his_d[i][0]['fact_src'] for i in editor_list]
        edited_list = [editor.editor.editors_his_f_h[f] for f in fact_list]

        sim_scores_with_edited = [
            torch.nn.functional.cosine_similarity(xxx, torch.cat(e, dim=0).cuda(), dim=1).mean()
            for e in edited_list]
        scores = [((i + j) / 2).detach().cpu().item() for i, j in zip(sen2key, sim_scores_with_edited)]
        max_sim_score = max(scores)
        scores_ind = inds[1][scores.index(max_sim_score)].detach().cpu().item()

        sort_sc = sorted(scores, reverse=True)
        diffs = sort_sc[0] - sort_sc[1]
        stds = np.std(sort_sc)

        return (diffs > 0.15 and stds > 0.05) or sort_sc[0]>0.9, scores_ind


def final_test(edit_sets, seq_edit_data, editor, boxe_model, desc="", if_rephrase=False, prompt_patch=False,
               log_path='./'):
    ##load the edits memory
    cache_edits = [i[0]['src'] for i in editor.editor.editors_his_d]
    Querys = []
    edits_res = []
    if args.eval_train_rate:
        edit_sets = split_data_n_sets(seq_edit_data.TrainR, len(seq_edit_data.TrainR))
        LOG.info('EVAL  train_rate:')
    if args.eval_dev_rate:
        edit_sets = split_data_n_sets(seq_edit_data.TestR, len(seq_edit_data.TestR))

        LOG.info('EVAL dev_rate')
    editsNUm = 0
    for s, ds in enumerate(tqdm(edit_sets)):
        Quert_tem = []
        # make mini batch
        if args.eval_train_rate:
            dl = DataLoader(dataset=ds, batch_size=1, collate_fn=seq_edit_data.TrainR.collate_fn)
        elif args.eval_dev_rate:
            dl = DataLoader(dataset=ds, batch_size=1, collate_fn=seq_edit_data.TestR.collate_fn)
        else:
            dl = DataLoader(dataset=ds, batch_size=1, collate_fn=seq_edit_data.edit_data.collate_fn)
        d0 = [j for j in dl][0]

        # eval before edit
        aft_edit_res, aft_re_res, aft_hard_res = [], [], []
        if args.task == 'zsre':
            edit_res, re_res, hard_res = edit_or_not_seq2seq(editor.editor.original_model, data_point=d0,
                                                             device=args.device, args=args)

        else:
            edit_res, re_res, hard_res = edit_or_not_binary(editor.editor.original_model, data_point=d0,
                                                            device=args.device, args=args)

        # get the input's rep for CL
        xxx, _ = get_proj_for_bert(boxe_model, **{
            'src': d0['raw'][0]['src'],
        })

        # LOG the treu label
        if edit_res[2]:  # need edit
            if editsNUm == args.stop_number:
                LOG.info(f"We are calculate the {editsNUm} edits,So we stop")
                break
            editsNUm += 1

        # Eval if need edit
        keys = torch.cat(editor.editor.editors_his_k).to(xxx.device)
        inds = torch.topk(torch.nn.functional.cosine_similarity(xxx, keys, dim=1), 6)

        eds, choose = edit_cls(inds, xxx, editor)
     #need edit
        if eds:  # is the edit
            is_edit = 1  # edit !!
            # laod the editor
            editor.editor.editors = []
            editor.editor.set_editors()
            editor.editor.load_edits(choose)
            # load the edits fact
            edits_fact = editor.editor.editors_his_d[choose][0]['fact_src']
            edits_token = editor.editor.original_model.tokenizer(edits_fact, return_tensors="pt",
                            padding=True,
                            max_length=32,
                            truncation=True,)

            if args.task == 'zsre':

                aft_edit_res, aft_re_res, aft_hard_res = edit_or_not_seq2seq(editor.editor.model, data_point={
                    'src_input_ids': d0['src_input_ids'],
                    're_src_input_ids': d0['re_src_input_ids'],
                    'src_attention_mask': d0['src_attention_mask'],
                    're_src_attention_mask': d0['re_src_attention_mask'],
                    'fact_src_input_ids': edits_token['input_ids'],
                    'fact_src_attention_mask': edits_token['attention_mask'],
                    'raw':d0['raw']
                },
                                                                             device=args.device, edit=True, args=args)
            else:
                aft_edit_res, aft_re_res, aft_hard_res = edit_or_not_binary(editor.editor.model, data_point={
                    'src_input_ids': d0['src_input_ids'],
                    're_src_input_ids': d0['re_src_input_ids'],
                    'src_attention_mask': d0['src_attention_mask'],
                    're_src_attention_mask': d0['re_src_attention_mask'],
                    'fact_src_input_ids': edits_token['input_ids'],
                    'fact_src_attention_mask': edits_token['attention_mask'],
                    'raw':d0['raw'],
                    'labels':d0['labels'],
                    're_labels':d0['re_labels'],
                },
                                                                            device=args.device, edit=True, args=args)

            # clear the edit
            editor.editor.clear_editors()

            re_after_edit = []
            if if_rephrase:  # and edit_res[2]:
                # test rephrase edit

                for ri, (i, a) in enumerate(zip(d0['re_src_input_ids'], d0['re_src_attention_mask'])):
                    # get the before res
                    if args.task == 'zsre':
                        re_bef_edit_res, re_bef_re_res, re_bef_hard_res = edit_or_not_seq2seq(
                            editor.editor.original_model,
                            data_point={
                                "src_input_ids": i.view(1,
                                                        -1),
                                "src_attention_mask": a.view(
                                    1, -1),
                                'raw': d0['raw']},
                            test_rephrases=False,
                            device=args.device, args=args)
                    else:
                        lab = d0['re_labels'][ri]
                        re_bef_edit_res, re_bef_re_res, re_bef_hard_res = edit_or_not_binary(
                            editor.editor.original_model,
                            data_point={
                                "src_input_ids": i.view(1,
                                                        -1),
                                "src_attention_mask": a.view(
                                    1, -1),
                                'labels': lab.view(1)},
                            device=args.device,
                            single=True, args=args)

                    # get the rephrase src emb
                    try:
                        reedit_src = d0['raw'][0]['rephrase'][ri]
                        xxx, _ = get_proj_for_bert(boxe_model, **{
                            'src': reedit_src,
                        })
                    except:
                        reedit_src = d0['raw'][0]['rephrases'][ri]
                        xxx, _ = get_proj_for_bert(boxe_model, **{
                            'src': reedit_src,
                        })

                    # get the score
                    keys = torch.cat(editor.editor.editors_his_k).to(xxx.device)
                    inds = torch.topk(torch.nn.functional.cosine_similarity(xxx, keys, dim=1), 6)

                    re_eds, re_choose = edit_cls(inds, xxx, editor)

                    if re_eds:  # need edit
                        re_is_edit = 1
                        editor.editor.editors = []
                        editor.editor.set_editors()
                        editor.editor.load_edits(re_choose)
                        if args.task == 'zsre':
                            re_aft_edit_res, re_aft_re_res, re_aft_hard_res = edit_or_not_seq2seq(editor.editor.model,
                                                                                                  data_point={
                                                                                                      "src_input_ids": i.view(
                                                                                                          1, -1),
                                                                                                      "src_attention_mask": a.view(
                                                                                                          1, -1),
                                                                                                      "fact_src_input_ids":edits_token['input_ids'], "fact_src_attention_mask":edits_token['attention_mask'],
                                                                                                      'raw': d0['raw']
                                                                                                  },
                                                                                                  test_rephrases=False,
                                                                                                  device=args.device,
                                                                                                  edit=True, args=args)
                        else:
                            re_aft_edit_res, re_aft_re_res, re_aft_hard_res = edit_or_not_binary(editor.editor.model,
                                                                                                 data_point={
                                                                                                     "src_input_ids": i.view(
                                                                                                         1,
                                                                                                         -1),
                                                                                                     "src_attention_mask": a.view(
                                                                                                         1, -1),
                                                                                                     "fact_src_input_ids":edits_token['input_ids'], "fact_src_attention_mask":edits_token['attention_mask'],
                                                                                                     'labels': lab.view(1)},
                                                                                                 device=args.device,
                                                                                                 single=True, edit=True,
                                                                                                 args=args)

                        editor.editor.clear_editors()
                    else:  # need not edit
                        re_is_edit = 0
                        if args.task == 'zsre':
                            re_aft_edit_res, re_aft_re_res, re_aft_hard_res = edit_or_not_seq2seq(editor.editor.model,
                                                                                                  data_point={
                                                                                                      "src_input_ids": i.view(
                                                                                                          1, -1),
                                                                                                      "src_attention_mask": a.view(
                                                                                                          1,
                                                                                                          -1),
                                                                                                      'raw': d0['raw']
                                                                                                  },
                                                                                                  test_rephrases=False,
                                                                                                  device=args.device,
                                                                                                  args=args)
                        else:
                            re_aft_edit_res, re_aft_re_res, re_aft_hard_res = edit_or_not_binary(editor.editor.model,
                                                                                                 data_point={
                                                                                                     "src_input_ids": i.view(
                                                                                                         1,
                                                                                                         -1),
                                                                                                     "src_attention_mask": a.view(
                                                                                                         1, -1),
                                                                                                     'labels': lab.view(
                                                                                                         1)},
                                                                                                 device=args.device,
                                                                                                 single=True, args=args
                                                                                                 )

                    re_after_edit.append(
                        [re_bef_edit_res, re_bef_re_res, re_bef_hard_res, re_aft_edit_res, re_aft_re_res,
                         re_aft_hard_res,
                         re_is_edit, re_choose])
            else:
                re_after_edit=[]
        else:  # not the edit
            is_edit = 0
            re_after_edit = []
            if args.task == 'zsre':
                aft_edit_res, aft_re_res, aft_hard_res = edit_or_not_seq2seq(editor.editor.model, data_point=d0,
                                                                             device=args.device, edit=False, args=args)
            else:
                aft_edit_res, aft_re_res, aft_hard_res = edit_or_not_binary(editor.editor.model, data_point=d0,

                                                                               device=args.device, edit=False, args=args)
        # Only the edits for input need to evaluate the GA_MI
        if if_rephrase:
            edits_res.append([
                [edit_res, re_res, hard_res, is_edit, choose],
                [aft_edit_res, aft_re_res, aft_hard_res, is_edit, choose],
                re_after_edit
            ])
        else:
            edits_res.append([[edit_res, re_res, hard_res, is_edit, choose],
                              [aft_edit_res, aft_re_res, aft_hard_res, is_edit, choose]])

        Querys.append(Quert_tem)
    torch.save(Querys, f'{log_path}/{desc}_Querys.pkl')
    print(f'SAVEING QUERY AT {log_path}/{desc}_Querys.pkl')
    return edits_res


def an_res(res):
    needs = []  # how many data need edit

    edit_succ = []  # the ES
    if_edit = []  # if the edit is edit

    GR_mi = []  # the GR_mi
    GR_ma = []  # the GR_ma

    GR_mi_glo = []  # the mi for all data
    GR_ma_glo = []

    affect = []  # need not but edit
    tru_affect = []  # affect the results

    before_res = []
    after_res = []
    for r in res:
        before_edit = r[0]
        after_edit = r[1]
        if before_edit[0][2]:  # need edit:
            needs.append(1)

            if before_edit[3]:
                if_edit.append(1)  # need and is edit
            else:
                if_edit.append(0)  # need but not edit

            if before_edit[3] and 1 - after_edit[0][2]:
                edit_succ.append(1)  # succ
            else:
                edit_succ.append(0)  # failed

            if before_edit[3]:  # if edit then calculate the GR
                GR_ma.append(after_edit[1][2])
                GR_mi = [rr for r in GR_ma for rr in r]

            try:
                GR_ma_glo.append(after_edit[1][2])
            except:
                GR_ma_glo.append(before_edit[1][2])

            GR_mi_glo = [rr for r in GR_ma_glo for rr in r]


        else:  # need not be edited
            needs.append(0)
            if before_edit[3]:
                affect.append(1)  # need not but edit
                tru_affect.append(before_edit[0][0] != after_edit[0][0])
            else:
                affect.append(0)  # need not and retain

        before_res.append(1 - before_edit[0][2])
        try:
            after_res.append(1 - after_edit[0][2])
        except:
            after_res.append(1 - before_edit[0][2])

    try:
        if args.eval_train_rate or args.eval_dev_rate:
            LOG.info(f'The RES before edit is {sum(before_res)}/{len(before_res)}={sum(before_res) / len(before_res)}')
            LOG.info(f'The RES after edit is {sum(after_res)}/{len(after_res)}={sum(after_res) / len(after_res)}')
            LOG.info(
                f'The Retain Rate {sum(after_res) / len(after_res)}/ {sum(before_res) / len(before_res)}={(sum(after_res) / len(after_res)) / (sum(before_res) / len(before_res))}')
        else:
            LOG.info(f'There are totall {len(res)} edits')
            LOG.info(f'There are totall {sum(needs)} need edit')
            LOG.info(f'There are totall {len([n for n in needs if n == 0])} need  NOT edit')

            LOG.info(f'The ES for edits is {sum(edit_succ)}/{len(edit_succ)}={sum(edit_succ) / len(edit_succ)}')
            LOG.info(f'The shot_for_edit is {sum(if_edit)}/{len(if_edit)}={sum(if_edit) / len(if_edit)} ')

            LOG.info(f'The GR_Mi for edits is: {sum(GR_mi)}/{len(GR_mi)}={sum(GR_mi) / len(GR_mi)}')
            LOG.info(
                f'The GR_Ma for edits is: {sum([sum(r) / len(r) for r in GR_ma])}/{len(GR_ma)}={sum([sum(r) / len(r) for r in GR_ma]) / len(GR_ma)}')

            LOG.info(f'The GR_Mi_GLO for edits is: {sum(GR_mi_glo)}/{len(GR_mi_glo)}={sum(GR_mi_glo) / len(GR_mi_glo)}')
            LOG.info(
                f'The GR_Ma_GLO for edits is: {sum([sum(r) / len(r) for r in GR_ma_glo])}/{len(GR_ma_glo)}={sum([sum(r) / len(r) for r in GR_ma_glo]) / len(GR_ma_glo)}')

            if len(affect) == 0:
                LOG.info(f'No affect')
            else:
                LOG.info(f'The affect number is {sum(affect)}/{len(affect)}={sum(affect) / len(affect)}')
                LOG.info(f'The True affect number is {sum(tru_affect)}/{len(affect)}={sum(tru_affect) / len(affect)}')
            LOG.info(f'The RES before edit is {sum(before_res)}/{len(before_res)}={sum(before_res) / len(before_res)}')
            LOG.info(f'The RES after edit is {sum(after_res)}/{len(after_res)}={sum(after_res) / len(after_res)}')
            LOG.info(
                f'The Retain Rate {sum(after_res) / len(after_res)}/ {sum(before_res) / len(before_res)}={(sum(after_res) / len(after_res)) / (sum(before_res) / len(before_res))}')

            LOG.info(
                f'GR, ER, TrainR, N:  {sum([sum(r) / len(r) for r in GR_ma]) / len(GR_ma)} {sum(edit_succ) / len(edit_succ)} {(sum(after_res) / len(after_res)) / (sum(before_res) / len(before_res))} {sum(needs)}')
    except Exception as e:
        LOG.info(f'Some thing is wrong: {e}')
    try:

        all_res = []
        Ga_mi = []
        for d in res:
            # each edits
            r = d[2]
            re = d[0]
            e_need = re[0][2]
            tes = []
            for rr in r:
                # each edit's rephrase
                # only eval the data which the src need input.
                if e_need:
                    is_edit = rr[6]
                    need_edit = rr[0][2]
                    before_res = 1 - rr[0][2]
                    after_res = 1 - rr[3][2]
                    all_res.append([is_edit, need_edit, before_res, after_res])  #
                    if need_edit:
                        tes.append(after_res)
            Ga_mi.append(tes)
        Ge_ma_res = sum([sum(i) / len(i) for i in Ga_mi if i != []]) / len(Ga_mi)
        Ge_mi_res = sum([ii for i in Ga_mi for ii in i]) / len([ii for i in Ga_mi for ii in i])

        shot = sum([1 if r[0] == 1 else 0 for r in all_res if r[1] == 1]) / len(
            [1 if r[0] == 1 else 0 for r in all_res if r[1] == 1])

        res_be = sum([r[-2] for r in all_res]) / len(all_res)
        res_af = sum([r[-1] for r in all_res]) / len(all_res)
        ES_GR = sum([1 if r[-1] == 1 else 0 for r in all_res if r[1] == 1]) / len(
            [1 if r[-1] == 1 else 0 for r in all_res if r[1] == 1])

        LOG.info(f'The ES_GR is {ES_GR}, The Shot is  {shot}')

    except:
        LOG.info("NO RE[3] data")


if __name__ == "__main__":

    # load model and editor
    if args.task == 'zsre':
        model_to_edit = BartSeq2Seq.load_from_checkpoint(args.model_path)
        # if args.editor_tp == 'patch':
        editor = BartSeq2SeqEditor(**vars(args))

    else:
        model_to_edit = BertBinary.load_from_checkpoint(args.model_path)
        # if args.editor_tp == 'patch':
        editor = BertBinaryEditor(**vars(args))

    # load data:
    if args.task == 'zsre':
        seq_edit_data = SeqEditDataSet(
            task_name=args.task, tokenizer=model_to_edit.tokenizer, data_path=args.data_path,
            batch_size=args.batch_size, num_workers=args.num_workers, use_length=args.use_length
        )
    else:

        seq_edit_data = SeqEditDataSet(
            task_name=args.task, tokenizer=model_to_edit.tokenizer, data_path=args.data_path,
            batch_size=args.batch_size, num_workers=args.num_workers, use_length=args.use_length
        )

    # load the CL model for classify
    boxe_model = torch.load(args.box_path)
    boxe_model.tokenizer = AutoTokenizer.from_pretrained("/media/sev/win/huggingface/unsup-simcse-roberta-large")

    # split the edits dataset
    edit_sets = split_data_n_sets(seq_edit_data.edit_data, len(seq_edit_data.edit_data))

    # load the train weight
    weights = torch.load(args.weight_path)
    editor.editor.editors_his_k = weights[0]
    editor.editor.editors_his_v = weights[1]
    editor.editor.editors_his_d = weights[2]
    editor.editor.editors_his_f = dict(weights[3])
    editor.editor.editors_his_f_h = dict(weights[4])
    LOG.info(f'Loading  {len(edit_sets)} data,\n'
             f'with {len(editor.editor.editors_his_k)} memorys.\n'
             )
    del weights

    # Begin to eval
    with torch.no_grad():
        LOG.info('EVAL ON EDITING')
        args.cls_type = 'hard_thre'
        res1 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                          desc='edit_hard', if_rephrase=args.if_rephrase,log_path=log_path)
        LOG.info('Hard Thre: \n')
        an_res(res1)

        args.cls_type = 'two_way'
        res2 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                          desc='edit_two_way', if_rephrase=args.if_rephrase,log_path=log_path)
        LOG.info('Two way: \n')
        an_res(res2)
        torch.save([res1, res2, []], f'{log_path}/edit_res.pkl')
        if True:  # not args.if_rephrase:
            #####################################################################

            args.eval_train_rate = True
            args.eval_dev_rate = False

            LOG.info('EVAL ON TRAIN ')
            args.cls_type = 'hard_thre'
            res1 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                              desc='TRAIN_RETAIN_HARD', if_rephrase=False,log_path=log_path)
            LOG.info('Hard Thre: \n')
            an_res(res1)

            args.cls_type = 'two_way'
            res2 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                              desc='TRAIN_RETAIN_TWO_WAY', if_rephrase=False,log_path=log_path)
            LOG.info('Two way: \n')
            an_res(res2)
            torch.save([res1, res2, []], f'{log_path}/train_res.pkl')

            #################################################
            args.eval_dev_rate = True
            args.eval_train_rate = False

            LOG.info('EVAL ON DEV')
            args.cls_type = 'hard_thre'
            res1 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                              desc='DEV_HARD', if_rephrase=False,log_path=log_path)
            LOG.info('Hard Thre: \n')
            an_res(res1)

            args.cls_type = 'two_way'
            res2 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                              desc='DEV_TWO_WAY', if_rephrase=False,log_path=log_path)
            LOG.info('Two way: \n')
            an_res(res2)
            torch.save([res1, res2, []], f'{log_path}/dev_res.pkl')
