import os
import sys
from random import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from .editor import BartSeq2SeqEditor, BertBinaryEditor
from .eval import final_test, an_res

from .settings import args, LOG, log_path
from .datasets import SeqEditDataSet
from .utils import split_data_n_sets, edit_or_not_seq2seq, get_proj_for_bert, edit_or_not_binary

# Retrieval_based_editing
from .models import BertBinary
from .models import BartSeq2Seq

from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer


def get_callbacks(args_):
    lr_callback = LearningRateMonitor(logging_interval="step")
    early_stopping_callback = EarlyStopping(
        monitor="stop_editing",
        patience=100000,
        stopping_threshold=0.99,
        mode='max',
    )
    model_checkpoint_callback = ModelCheckpoint(
        monitor=None,  # args_.ckpt_monitor,
        mode=args_.ckpt_metric_mode,
        dirpath=args_.ckpt_path,
        save_top_k=0,
        filename="model",  # auto-filled,
        save_weights_only=True
    )
    return [lr_callback, early_stopping_callback, model_checkpoint_callback]


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def save_and_evl():

    torch.save([train_res, Faied], f'{log_path}/{editsNUm}_train_res.pkl')

    torch.save(
        [
            editor.editor.editors_his_k,
            editor.editor.editors_his_v,
            editor.editor.editors_his_d,
            editor.editor.editors_his_f,
            editor.editor.editors_his_f_h,
            Faied
        ], f'{log_path}/{editsNUm}_edits_weightsl.pkl')

    LOG.info(f'Training finish, save the weight results to {log_path}/{editsNUm}_edits_weightsl.pkl')

    LOG.info(f'There are {editsNUm} need edit, ')
    LOG.info(f'the edit succ number is {edits_right_num}, ')
    LOG.info(f'SUCC is {edits_right_num / editsNUm}')
    LOG.info(f'There are {len(Faied)} data edits failed.')
    LOG.info(f'The memory size is {len(editor.editor.editors_his_k)}')

    edit_res = [t for t in train_res if t[1] != [[], [], []]]

    es_res_for_train = [1 - i[1][0][2] for i in edit_res]
    gr_res_for_train = [i[1][1][2] for i in edit_res]
    gr_mi = [ii for i in gr_res_for_train for ii in i]
    gr_ma = [sum(i) / len(i) for i in gr_res_for_train]
    LOG.info(
        f'The ES for train is {sum(es_res_for_train)}/{len(es_res_for_train)}={sum(es_res_for_train) / len(es_res_for_train)}')
    LOG.info(f'The GR_mi for train is {sum(gr_mi)}/{len(gr_mi)}={sum(gr_mi) / len(gr_mi)}')
    LOG.info(f'The GR_ma for train is {sum(gr_ma)}/{len(gr_ma)}={sum(gr_ma) / len(gr_ma)}')

    LOG.info('EVAL ON EDITING reshapes')
    args.cls_type = 'hard_thre'
    res1 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                      desc=f'{editsNUm}_EDIT_HARD', if_rephrase=True,log_path=log_path)
    LOG.info('Hard Thre: \n')
    an_res(res1)

    args.cls_type = 'two_way'
    res2 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                      desc=f'{editsNUm}_EDIT_TWO_WAY', if_rephrase=True,log_path=log_path)
    LOG.info('Two way: \n')
    an_res(res2)
    torch.save([res1, res2, []], f'{log_path}/{editsNUm}_edit_with_re_res.pkl')

    #####################################################################
    args.eval_train_rate = True
    args.eval_dev_rate = False

    LOG.info('EVAL ON TRAIN ')
    args.cls_type = 'hard_thre'
    res1 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                      desc=f'{editsNUm}_TRAIN_HARD', if_rephrase=False,log_path=log_path)
    LOG.info('Hard Thre: \n')
    an_res(res1)

    args.cls_type = 'two_way'
    res2 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                      desc=f'{editsNUm}_TRAIN_WTO_WAY', if_rephrase=False,log_path=log_path)
    LOG.info('Two way: \n')
    an_res(res2)
    torch.save([res1, res2, []], f'{log_path}/{editsNUm}_train_res.pkl')

    #################################################
    args.eval_dev_rate = True
    args.eval_train_rate = False

    LOG.info('EVAL ON DEV')
    args.cls_type = 'hard_thre'
    res1 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                      desc=f'{editsNUm}_DEV_HARD', if_rephrase=False,log_path=log_path)
    LOG.info('Hard Thre: \n')
    an_res(res1)

    args.cls_type = 'two_way'
    res2 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                      desc=f'{editsNUm}_DEV_TWO_WAY', if_rephrase=False,log_path=log_path)
    LOG.info('Two way: \n')
    an_res(res2)
    torch.save([res1, res2, []], f'{log_path}/{editsNUm}_dev_res.pkl')

    ############################################################################

if __name__ == "__main__":

    # load model and editor
    if args.task == 'zsre':
        model_to_edit = BartSeq2Seq.load_from_checkpoint(args.model_path)

        editor = BartSeq2SeqEditor(**vars(args))

    else:
        model_to_edit = BertBinary.load_from_checkpoint(args.model_path)

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

    #load the contriever
    # boxe_model = AutoModel.from_pretrained("contriever-msmarco").cuda()
    # boxe_model.tokenizer= AutoTokenizer.from_pretrained("contriever-msmarco")

    # split the edits dataset
    edit_sets = split_data_n_sets(seq_edit_data.edit_data, len(seq_edit_data.edit_data))


    # start editing

    # init the res record
    train_res = []
    edits_right_num = 0
    editsNUm = 0
    Faied = []
    LOC_loader = DataLoader(dataset=seq_edit_data.TrainR, batch_size=1, collate_fn=seq_edit_data.TrainR.collate_fn,
                            shuffle=True)
    for s, ds in enumerate(edit_sets):
        # make mini batch
        LOG.info(f"We are calculate the {s} data, all: {len(edit_sets)}")
        dl = DataLoader(dataset=ds, batch_size=1, collate_fn=seq_edit_data.edit_data.collate_fn)
        d0 = [j for j in dl][0]

        if args.task == 'zsre':
            edit_res, re_res, hard_res = edit_or_not_seq2seq(editor.editor.original_model, data_point=d0,
                                                             device=args.device,args=args)
        else:
            edit_res, re_res, hard_res = edit_or_not_binary(editor.editor.original_model, data_point=d0,
                                                            device=args.device,args=args)
        # prepare the after logs
        aft_edit_res, aft_re_res, aft_hard_res = [], [], []

        if edit_res[2]:  # need edit

            if editsNUm == args.stop_number:
                LOG.info(f"We are calculate the {editsNUm} edits,So we stop")
                # try:

                save_and_evl()

            editsNUm += 1
            sen_proj, fact_proj = get_proj_for_bert(boxe_model, **{
                'src': d0['raw'][0]['src'],
                'facts': d0['raw'][0]['fact_src']

            })
            sen_proj = sen_proj[0].view(1, -1)
            fact_proj = fact_proj[0].view(1, -1)

            editor.editor.editors = []

            editor.editor.set_editors()


            callbacks = get_callbacks(args)
            edit_trainer = Trainer(
                callbacks=callbacks, gpus=args.gpus, logger=TensorBoardLogger(log_path, name=None),
                check_val_every_n_epoch=args.check_val_every_n_epoch, log_every_n_steps=args.check_val_every_n_epoch,
                max_epochs=args.max_edit_step, num_sanity_val_steps=0,
                gradient_clip_val=5.0,
            )
            edit_trainer.use_index = -1


            edit_trainer.loc = LOC_loader._get_iterator().next()

            # different edit method
            if args.editor_tp == 'fact_emb':
                editor.editor.editors[0]['editor'].set_fact_emb(fact_proj)

            # if the edit in memory
            if d0["raw"][0]['fact_src'] in editor.editor.editors_his_f:
                indexs = editor.editor.editors_his_f[d0["raw"][0]['fact_src']][0]
                editor.editor.load_edits(indexs)
                edit_trainer.use_index = indexs

            # editing !!
            edit_trainer.fit(editor, train_dataloaders=dl, val_dataloaders=dl)

            # eval after editing.

            if args.task == 'zsre':
                aft_edit_res, aft_re_res, aft_hard_res = edit_or_not_seq2seq(editor.editor.model, data_point=d0,
                                                                             device=args.device,edit=True,args=args)
            else:
                aft_edit_res, aft_re_res, aft_hard_res = edit_or_not_binary(editor.editor.model, data_point=d0,
                                                                            device=args.device,edit=True,args=args)

            # if edits falied:
            if aft_edit_res[2]:
                LOG.info(f'EDITING FAILED {d0["raw"][0]["src"]}')
                Faied.append([s, d0["raw"][0]["src"]])
            else:
                edits_right_num += 1

                # save the edits information
                sims = torch.nn.functional.cosine_similarity(sen_proj, fact_proj, dim=1)

                editor.editor.save_edits(d0['raw'][0], fact_proj, sen_proj, sims=sims.detach().cpu().item())

            editor.editor.clear_editors()  # remove the edit

        train_res.append([[edit_res, re_res, hard_res], [aft_edit_res, aft_re_res, aft_hard_res]])

    torch.save([train_res, Faied], f'{log_path}/train_res.pkl')

    torch.save(
        [
            editor.editor.editors_his_k,
            editor.editor.editors_his_v,
            editor.editor.editors_his_d,
            editor.editor.editors_his_f,
            editor.editor.editors_his_f_h,
            Faied
        ], f'{log_path}/edits_weightsl.pkl')

    LOG.info(f'Training finish, save the weight results to {log_path}/edits_weightsl.pkl')

    LOG.info(f'There are {editsNUm} need edit, ')
    LOG.info(f'the edit succ number is {edits_right_num}, ')
    LOG.info(f'SUCC is {edits_right_num / editsNUm}')
    LOG.info(f'There are {len(Faied)} data edits failed.')
    LOG.info(f'The memory size is {len(editor.editor.editors_his_k)}')

    edit_res = [t for t in train_res if t[1] != [[], [], []]]

    es_res_for_train = [1 - i[1][0][2] for i in edit_res]
    gr_res_for_train = [i[1][1][2] for i in edit_res]
    gr_mi = [ii for i in gr_res_for_train for ii in i]
    gr_ma = [sum(i) / len(i) for i in gr_res_for_train]
    LOG.info(
        f'The ES for train is {sum(es_res_for_train)}/{len(es_res_for_train)}={sum(es_res_for_train) / len(es_res_for_train)}')
    LOG.info(f'The GR_mi for train is {sum(gr_mi)}/{len(gr_mi)}={sum(gr_mi) / len(gr_mi)}')
    LOG.info(f'The GR_ma for train is {sum(gr_ma)}/{len(gr_ma)}={sum(gr_ma) / len(gr_ma)}')

    LOG.info('EVAL ON EDITING reshapes')
    args.cls_type = 'hard_thre'
    res1 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                      desc='EDIT_HARD', if_rephrase=True,log_path=log_path)
    LOG.info('Hard Thre: \n')
    an_res(res1)

    args.cls_type = 'two_way'
    res2 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                      desc='EDIT_TWO_WAY', if_rephrase=True,log_path=log_path)
    LOG.info('Two way: \n')
    an_res(res2)
    torch.save([res1, res2, []], f'{log_path}/edit_with_re_res.pkl')

    #####################################################################
    args.eval_train_rate = True
    args.eval_dev_rate = False

    LOG.info('EVAL ON TRAIN ')
    args.cls_type = 'hard_thre'
    res1 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                      desc='TRAIN_HARD', if_rephrase=False,log_path=log_path)
    LOG.info('Hard Thre: \n')
    an_res(res1)

    args.cls_type = 'two_way'
    res2 = final_test(edit_sets, seq_edit_data=seq_edit_data, editor=editor, boxe_model=boxe_model,
                      desc='TRAIN_WTO_WAY', if_rephrase=False,log_path=log_path)
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

