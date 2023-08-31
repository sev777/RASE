
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.seq2seq_modules import BartSeq2Seq
from src.models.class_modules import BertBinary
# from src.models.t5_modules import T5Seq2Seq
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy
from tqdm import tqdm
import copy
from torch.nn import init
import torch
import torch.nn.functional as F
from torch import Tensor
import collections
from torch.optim.lr_scheduler import ReduceLROnPlateau






def kl_loc_loss(pre, post, mask=None):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.view(-1, pre.shape[-1])
    post_ = post.view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            # assert mask is not None
            kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1)
            mask_ = mask.contiguous().view(pre_.shape[0])
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError



def get_callbacks(args_):
    lr_callback = LearningRateMonitor(logging_interval="step")
    early_stopping_callback = EarlyStopping(
        monitor="stop_editing",
        patience=100000,
        stopping_threshold=0.99,
        mode='max',
    )
    model_checkpoint_callback = ModelCheckpoint(
        monitor=None,#args_.ckpt_monitor,
        mode=args_.ckpt_metric_mode,
        dirpath=args_.ckpt_path,
        save_top_k=0,
        filename="model",  # auto-filled,
        save_weights_only=True
    )
    return [lr_callback, early_stopping_callback,model_checkpoint_callback]

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
########################################################################

class ModuleDetector(nn.Module):

    def __init__(self, model: nn.Module,  mode='input', memory_loc=0, hidden_loc=0):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.mode = mode

        self.hidden_button = False

        self.memory = []
        self.tmp_hidden = None

        self.memory_mask = None


    def set_hidden_loc(self, hl):
        self.hidden_loc = hl

    def forward(self, hidden_states):
        output = self.model(hidden_states)

        m_tmp = copy.deepcopy(hidden_states if self.mode == 'input' else output)
        hidden_size = m_tmp.size(-1)
        if self.hidden_button:
            if self.hidden_loc == 'bart_seq':
                # The hidden is usually used for one example, we do not need the mask for padding token
                self.tmp_hidden = m_tmp.reshape(-1, hidden_size)
            else:
                self.tmp_hidden = m_tmp[:, self.hidden_loc].reshape(-1, hidden_size)
        return output

    def get_hidden(self):
        return self.tmp_hidden

    def turn_on_hidden(self):
        self.hidden_button = True

    def turn_off_hidden(self):
        self.hidden_button = False
class Editor(nn.Module):

    def __init__(self,
                 model, hidden_size=768, device=None,
                 max_add_neuron_num=5,

                 freeze_model=True, freeze_k=False, freeze_a=False,
                 amplify_v=False, amplify_con=10.0,
                 drop_num=0, drop_rate=0.5,
                 act_margin_val=0.0, margin_val1=0.0, margin_val2=0.0,args=None
                 ):
        super().__init__()
        self.args=args
        self.model = copy.deepcopy(model)
        self.original_model = copy.deepcopy(model)
        self.hidden_size = hidden_size
        self.device = device
        self.max_add_neuron_num = max_add_neuron_num


        self.freeze_model = freeze_model
        self.freeze_a = freeze_a
        self.amplify_con = amplify_con
        if self.freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False

        self.model_named_modules = None
        self.get_named_modules()

        self.editors = []
        self.detectors = []

        self.amplify_v = amplify_v
        self.freeze_k = freeze_k

        self.drop_num = drop_num
        self.drop_rate = drop_rate

        self.act_margin_val = act_margin_val
        self.margin_val1 = margin_val1
        self.margin_val2 = margin_val2


        self.editors_his_k=[]# editors memory Key
        self.editors_his_v=[]# editors memory Value


        self.editors_his_d=[]# editors memory data
        self.editors_his_f=collections.defaultdict(list)# editors memory for sim score
        self.editors_his_f_h=collections.defaultdict(list)# editors memory for data rep

        if self.args.task == 'zsre':

            self.name_edit_type = {
                'model.model.decoder.layers.5.fc1': 'output',
                'model.model.decoder.layers.5.fc2': 'input'
            }
        else:


            self.name_edit_type = {
                'model.model.encoder.layer.11.output.dense': 'input',
                'model.model.encoder.layer.11.intermediate.dense': 'output'
            }


    def reset_model(self, model, clear_memory):
        self.model = copy.deepcopy(model)
        if self.freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False
        self.model_named_modules = None
        self.get_named_modules()
        self.editors = []


    def clear_detectors(self):
        for d in self.detectors:
            self.model_named_modules[d['module']]._modules[d['child']] = d['original_module']
        self.detectors = []


    def clear_editors(self):
        for e in self.editors:
            self.model_named_modules[e['module']]._modules[e['child']] = e['original_module']
        self.get_named_modules()


    def set_editors(self, batch=None, init_weights=None, error_count=1, select_index=0):

        self.get_editors(
            batch,
            init_weights=dict() if init_weights is None else init_weights,
            error_count=error_count, select_index=select_index
        )
        for e in self.editors:
            self.model_named_modules[e['module']]._modules[e['child']] = e['editor']

    def set_detectors(self):
        for d in self.detectors:
            self.model_named_modules[d['module']]._modules[d['child']] = d['detector']

    def load_edits(self,ind):
        for ei, e in enumerate(self.editors):
            # #print(f'LOADS {''.join([e['module'],e['child'],data])}')
            e['editor'].import_patch(self.editors_his_v[ind][ei])
            self.model_named_modules[e['module']]._modules[e['child']] = e['editor']

        self.get_named_modules()



    def save_edits(self,data,fact_hid,sen_proj,sims=-1,index=-1):

        edits = []
        for e in self.editors:
            edits.append(e['editor'].export_patch())
        if index == -1:
            self.editors_his_k.append(fact_hid.detach().cpu())
            self.editors_his_v.append(edits)
            self.editors_his_d.append([data])
            self.editors_his_f[data['fact_src']].append(len(self.editors_his_k)-1)
            self.editors_his_f[data['fact_src']].append([sims])
            self.editors_his_f[data['fact_src']].append(data['src'])
            self.editors_his_f_h[data['fact_src']].append(sen_proj.detach().cpu())
        else:
            self.editors_his_v[index]=edits
            self.editors_his_d[index].append(data)
            self.editors_his_f[data['fact_src']][1].append(sims)
            self.editors_his_f[data['fact_src']].append(data['src'])
            self.editors_his_f_h[data['fact_src']].append(sen_proj.detach().cpu())

    def step(self):

        for e in self.editors:
            self.model_named_modules[e['module']]._modules[e['child']] = e['editor'].assign_layer()
        self.editors = []
        self.get_named_modules()

    def get_hidden(self, index=None):
        res = dict()
        for d in self.detectors:
            k = d['module'] + '.' + d['child']
            v = d['detector'].get_hidden()
            res[k] = v[index] if index is not None else v
        return res


    def get_named_modules(self):
        # For now we just edit one linear layer once
        self.model_named_modules = None
        self.model_named_modules = {x[0]: x[1] for x in self.model.named_modules()}

    def get_editors(self, batch, init_weights=None, error_count=None, select_index=None):
        name_edit_type = self.name_edit_type
        for name, edit_type in name_edit_type.items():
            e_tmp = dict()
            n = name.rsplit('.', 1)
            e_tmp['module'], e_tmp['child'] = n[0], n[-1]
            if edit_type == 'input':
                e_tmp['editor'] = ModifyLinearInput_single(
                    self.model_named_modules[n[0]].__getattr__(n[-1]),
                    amplify=self.amplify_v, freeze_a=self.freeze_a,
                    amplify_con=self.amplify_con,
                    add_neuron_num=self.max_add_neuron_num,device=self.device,args=self.args
                )
            else:
                init_weight = init_weights[name] if name in init_weights.keys() else None
                # train_memo, val_memo = None, None

                e_tmp['editor'] = ModifyLinearOutput_single(
                    self.model_named_modules[n[0]].__getattr__(n[-1]),
                    init_weight=init_weight,  freeze=self.freeze_k,
                    drop_num=self.drop_num, drop_rate=self.drop_rate,
                    add_neuron_num=self.max_add_neuron_num,device=self.device
                )
            e_tmp['original_module'] = self.model_named_modules[n[0]].__getattr__(n[-1])
            self.editors.append(e_tmp)

    def repeat_tensor(self, t):
        return torch.repeat_interleave(t, self.drop_num + 1, dim=0)

    def feed_kl_input(self, memo_loader, his_edit_data, total_loc_num):
        # self.memo_loader = memo_loader
        self.total_loc_num = total_loc_num
        self.his_edit_data = his_edit_data

    def do_not_act_val(self):
        for e in self.editors:
            e['editor'].activate_loss = 'non_use'

    def do_act_val(self):
        for e in self.editors:
            e['editor'].activate_loss = self.activate_loss

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,fact_inds=None,labels=None):

        if self.args.task == 'zsre':
            target_for_loss = decoder_input_ids[:, 1:]
            res = dict()

            logits = self.model(
                input_ids, attention_mask,
                decoder_input_ids[:, :-1], decoder_attention_mask[:, :-1]
            )
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs=logits.log_softmax(-1), target=target_for_loss,
                epsilon=self.model.hparams.eps, ignore_index=self.model.tokenizer.pad_token_id,
            )
            # ntokens = decoder_attention_mask[:, 1:].sum()
            res['logtis'] = logits
            res['loss']=loss

            return res
        else:
            res = self.model(input_ids, attention_mask, labels)
            return res

    def get_detectors(self, *args, **kwargs):
        detected_modules = kwargs.get("detected_modules")

        hidden_loc = kwargs.get("hidden_loc") if "hidden_loc" in kwargs else 0
        mode = kwargs.get("mode") if "mode" in kwargs else "input"
        for module_name, child in detected_modules.items():
            detector = ModuleDetector(
                model=self.model_named_modules[module_name]._modules[child],
                 mode=mode,
                 hidden_loc=hidden_loc
            )
            self.detectors.append({
                'module': module_name, 'child': child,
                'detector': detector, 'original_module': self.model_named_modules[module_name]._modules[child]
            })


#Editor for zsre
class BartSeq2SeqEditor(LightningModule):

    def __init__(self,toks=None, *args, **kwargs):
        super().__init__()
        self.toks=toks
        # add_neuron_num = kwargs.get('add_neuron_num')
        self.save_hyperparameters()
        self.current_device = torch.device('cuda', self.hparams.gpus[0])
        ##
        if self.hparams.editor_tp=='patch':
            self.editor = Editor(
                model=BartSeq2Seq.load_from_checkpoint(self.hparams.model_path),
                max_add_neuron_num=self.hparams.max_add_neuron_num,
                freeze_model=self.hparams.freeze_model,
                freeze_k=self.hparams.freeze_k,
                freeze_a=self.hparams.freeze_a,
                args=self.hparams_initial,
                amplify_v=self.hparams.amplify_v,
                act_margin_val=self.hparams.act_margin_val,

                device=self.current_device
            )

        self.edit_acc = Accuracy()
        self.val_loader = None
        self.valid_memory_loss = []
        self.valid_metric = []
        self.save_ckpt = 0
        self.stop_editing = False
        self.start_example_editing = False
        self.BIG_CONSTANT = 10000
        self.has_stepped = False

    def on_train_start(self):
        self.valid_memory_loss = []
        self.valid_metric = []
        self.save_ckpt = 0
        self.start_example_editing = False
        self.stop_editing = False
        self.has_stepped = False


    @staticmethod
    def early_stop_editing(metrics, mode='min', thd=0.0, patience=1):
        best_step = 0
        for step, vm in enumerate(metrics):
            if mode == 'min':
                if vm < metrics[best_step] - thd:
                    best_step = step
            else:
                if vm > metrics[best_step] + thd:
                    best_step = step

        if best_step < len(metrics) - patience:
            return True
        else:
            return False

    @staticmethod
    def save_editing_ckpt(metrics, mode='min'):
        # save the model if new val metric is attained
        return (mode == 'min' and min(metrics) == metrics[-1]) or (mode == 'max' and max(metrics) == metrics[-1])

    def fed_val_loader(self, dl):
        self.val_loader = dl

    def reset(self, clear_memory):
        self.editor.reset_model(
            BartSeq2Seq.load_from_checkpoint(self.hparams.model_path),
            clear_memory=clear_memory
        )

    def joint_training(self, batch, batch_idx=None,fact_emb=None,indexs=-1):
            if self.hparams_initial.train_way=='ori':
                input_ids = batch["src_input_ids"]
                input_atts =batch["src_attention_mask"]

            elif self.hparams_initial.train_way=='sen_prompt':
                input_ids = batch["src_input_ids"][:,1:]
                input_atts =batch["src_attention_mask"][:,1:]
                facts_input='||'.join(batch['raw'][0]['fact_src'].split('||')[:-1])
                fact_tok=self.editor.model.tokenizer(facts_input, return_tensors="pt",
                    padding=True, max_length=30,
                    truncation=True,)
                fact_ids=fact_tok['input_ids'].to(input_ids.device)
                fact_att=fact_tok['attention_mask'].to(input_ids.device)
                input_ids=torch.cat([torch.repeat_interleave(fact_ids,len(input_ids),0),input_ids],dim=1,)
                input_atts=torch.cat([torch.repeat_interleave(fact_att,len(input_atts),0),input_atts],dim=1,)

            elif self.hparams_initial.train_way == 'sen_combine':
                input_ids = batch["src_input_ids"]
                input_atts =batch["src_attention_mask"]
                fact_ids=batch['fact_src_input_ids']
                fact_att=batch['fact_src_attention_mask']
                input_ids=torch.cat([torch.repeat_interleave(fact_ids,len(input_ids),0),input_ids],dim=1,)
                input_atts=torch.cat([torch.repeat_interleave(fact_att,len(input_atts),0),input_atts],dim=1,)

            elif self.hparams_initial.train_way == 'fact_aware':
                input_ids = batch["src_input_ids"]#[:,1:]
                input_atts =batch["src_attention_mask"]#[:,1:]
                self.editor.editors[0]['editor'].set_fact_emb(self.trainer.fact_emb)

            res = self.editor(
                input_ids, input_atts,
                batch["trg_input_ids"], batch["trg_attention_mask"]
            )
            loss = res['loss']


            if self.hparams_initial.KL:
                mask = self.trainer.loc["trg_attention_mask"][:, :-1].to(self.device) if 'trg_attention_mask' in self.trainer.loc.keys() else None
                if self.hparams_initial.task=='zsre':
                    ori_res = self.editor.original_model(
                        self.trainer.loc['src_input_ids'].cuda(),
                        self.trainer.loc['src_attention_mask'].cuda(),
                        self.trainer.loc["trg_input_ids"][:, :-1].cuda(),
                        self.trainer.loc["trg_attention_mask"][:, :-1].cuda())

                    after_res=self.editor.model(
                        self.trainer.loc['src_input_ids'].cuda(),
                        self.trainer.loc['src_attention_mask'].cuda(),
                        self.trainer.loc["trg_input_ids"][:, :-1].cuda(),
                        self.trainer.loc["trg_attention_mask"][:, :-1].cuda())
                else:
                    ori_res = self.editor.original_model(
                        self.trainer.loc['src_input_ids'].cuda(),
                        self.trainer.loc['src_attention_mask'].cuda(),
                       )
                    after_res = self.editor.model(
                        self.trainer.loc['src_input_ids'].cuda(),
                        self.trainer.loc['src_attention_mask'].cuda(),
                      )
                kl_loss = kl_loc_loss(ori_res, after_res, mask)
                loss=loss+  kl_loss

            return {"loss": loss}

    def training_step(self, batch, batch_idx=None, optimizer_idx=None,):
        #print(self.trainer.use_index)

        if  self.trainer.use_index== -1:
            return self.joint_training(batch, batch_idx)
        else:
            return self.joint_training(batch, batch_idx,indexs = self.trainer.use_index)

    def joint_validation(self, batch, batch_idx=None,indexs=-1):
        stop_editing, save_ckpt = False, False
        if self.hparams_initial.train_way == 'fact_aware':
            self.editor.editors[0]['editor'].set_fact_emb(self.trainer.fact_emb)
        if not self.hparams_initial.re_vaild:

            if self.hparams_initial.train_way=='ori':
                input_ids = batch["src_input_ids"]#[:,1:]
                input_atts =batch["src_attention_mask"]#[:,1:]

            elif self.hparams_initial.train_way=='sen_prompt':
                input_ids = batch["src_input_ids"][:,1:]
                input_atts =batch["src_attention_mask"][:,1:]
                # fact_ids=batch['fact_src_input_ids'][:,:-1]
                # fact_att=batch['fact_src_attention_mask'][:,:-1]
                facts_input = '||'.join(batch['raw'][0]['fact_src'].split('||')[:-1])
                fact_tok=self.editor.model.tokenizer(facts_input, return_tensors="pt",
                    padding=True, max_length=30,
                    truncation=True,)
                fact_ids=fact_tok['input_ids'].cuda()
                fact_att=fact_tok['attention_mask'].cuda()
                input_ids=torch.cat([torch.repeat_interleave(fact_ids,len(input_ids),0),input_ids],dim=1,)
                input_atts=torch.cat([torch.repeat_interleave(fact_att,len(input_atts),0),input_atts],dim=1,)

            elif self.hparams_initial.train_way == 'sen_combine':
                input_ids = batch["src_input_ids"]
                input_atts =batch["src_attention_mask"]
                fact_ids=batch['fact_src_input_ids']
                fact_att=batch['fact_src_attention_mask']
                input_ids=torch.cat([torch.repeat_interleave(fact_ids,len(input_ids),0),input_ids],dim=1,)
                input_atts=torch.cat([torch.repeat_interleave(fact_att,len(input_atts),0),input_atts],dim=1,)


            b_size = input_ids.size(0)
            res = self.editor(
                input_ids, input_atts,
                batch["trg_input_ids"], batch["trg_attention_mask"]
            )
            self.log("val_loss", res['loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=b_size)


            model_gen = self.editor.model.model.generate(
                input_ids=input_ids, attention_mask=input_atts,
                min_length=1, num_return_sequences=1, use_cache=True
            )

        else:

            if self.hparams_initial.train_way=='ori':
                input_ids = batch["vaild_input_ids"]
                input_atts =batch["vaild_attention_mask"]

            elif self.hparams_initial.train_way=='sen_prompt':
                input_ids = batch["vaild_input_ids"][:,1:]
                input_atts =batch["vaild_attention_mask"][:,1:]
                fact_ids=batch['fact_src_input_ids'][:,:-1]
                fact_att=batch['fact_src_attention_mask'][:,:-1]
                input_ids=torch.cat([torch.repeat_interleave(fact_ids,len(input_ids),0),input_ids],dim=1,)
                input_atts=torch.cat([torch.repeat_interleave(fact_att,len(input_atts),0),input_atts],dim=1,)

            elif self.hparams_initial.train_way == 'sen_combine':
                input_ids = batch["vaild_input_ids"]
                input_atts =batch["vaild_attention_mask"]
                fact_ids=batch['fact_src_input_ids']
                fact_att=batch['fact_src_attention_mask']
                input_ids=torch.cat([torch.repeat_interleave(fact_ids,len(input_ids),0),input_ids],dim=1,)
                input_atts=torch.cat([torch.repeat_interleave(fact_att,len(input_atts),0),input_atts],dim=1,)


            b_size = input_ids.size(0)
            res = self.editor(
                input_ids, input_atts,
                batch["trg_input_ids"][:b_size,:], batch["trg_attention_mask"][:b_size,:]
            )
            self.log("val_loss", res['loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=b_size)

            model_gen = self.editor.model.model.generate(
                input_ids=input_ids, attention_mask=input_atts,
                min_length=1, num_return_sequences=1, use_cache=True
            )

        # self.editor.do_act_val()
        target_len = batch["trg_input_ids"].size(1)
        edit_acc = model_gen[:, :target_len].equal(batch["trg_input_ids"][:b_size,:])

        stop_editing = float(edit_acc)
        save_ckpt = edit_acc

        self.save_ckpt += save_ckpt
        self.stop_editing = stop_editing
        if self.stop_editing == 1:
            self.has_stepped = True
        self.log("stop_editing", float(stop_editing), on_step=False, on_epoch=True, prog_bar=True)
        self.log("save_ckpt", self.save_ckpt, on_step=False, on_epoch=True, prog_bar=True)


    def validation_step(self, batch, batch_idx=None):
        # #print('Vailde:',batch['raw'])
        self.joint_validation(batch, batch_idx)

    def get_optimizer(self, params, lr=None, optim=None):
        if lr is None:
            lr = self.hparams.lr
        if optim is None:
            optim = self.hparams.optim
        if optim == "adam":
            return torch.optim.Adam(params=params, lr=lr, weight_decay=self.hparams.weight_decay)
        if optim == 'rmsprop':
            return torch.optim.RMSprop(params=params, lr=lr)
        if optim == 'sgd':
            return torch.optim.SGD(params=params, lr=lr, weight_decay=self.hparams.weight_decay, momentum=0.9)

    def configure_optimizers(self):
        # for the joint editing style, we just need one parameter
        parameters = [p for p in self.editor.parameters() if p.requires_grad]
        optimizer = self.get_optimizer(parameters)

        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer, mode='min',
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
            threshold=0.05
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch", "frequency": self.hparams.check_val_every_n_epoch,
            "monitor": "val_loss", "strict": True
        }
        optimizer_list = [optimizer]
        lr_scheduler_config_list = [lr_scheduler_config]

        return optimizer_list, lr_scheduler_config_list

class ModifyLinearOutput_single(nn.Module):  # nn.Linear(input_size, output_size) -> nn.Linear(input_size, output_size+1)

    def __init__(self,
                 linear: nn.Linear,
                 add_neuron_num=1, init_weight=None,
                 drop_num=0, drop_rate=0.5,
                 freeze=False,device=None
                 ):
        super().__init__()

        self.device = device
        self.linear = copy.deepcopy(linear)
        self.hidden_size = min(self.linear.weight.size())
        self.intermediate_size = max(self.linear.weight.size())

        self.add_neuron_num = add_neuron_num

        self.add_neuron_loc = [-(i + 1) for i in range(add_neuron_num)]


        self.extra_output = nn.Linear(self.hidden_size, self.add_neuron_num)
        if init_weight is not None:
            assert init_weight.size(0) == add_neuron_num
            self._reset_parameters(init_weight=init_weight)

        self.drop_num = drop_num
        self.drop_rate = drop_rate

        self.freeze = freeze
        if self.freeze:
            for p in self.extra_output.parameters():
                p.requires_grad = False
        self.fact_emb=None

    def freeze_self(self):
        for p in self.extra_output.parameters():
            p.requires_grad = False
    def init_fact_emb(self):
        self.fact_emb = None
    def set_fact_emb(self,emb):
        self.fact_emb=emb
    def unfreeze_self(self):
        for p in self.extra_output.parameters():
            p.requires_grad = True

    def _reset_parameters(self, init_weight):
        scale = torch.norm(init_weight, dim=-1).unsqueeze(-1)
        self.extra_output.weight = nn.Parameter(init_weight / (scale ** 2))
        init.constant_(self.extra_output.bias, 0.)


    def get_dif_dropout(self, h):
        res = [h]
        for i in range(self.drop_num):
            p = self.drop_rate if not isinstance(self.drop_rate, list) else self.drop_rate[i]
            res.append(F.dropout(h, p=p))
        return torch.cat(res, dim=0)

    def forward(self, hidden_states):

        w, b = self.get_modified_weight_bias()

        if self.fact_emb!=None:
            hidden_states[:,-1,:]+=self.fact_emb.to(hidden_states.device)
            self.init_fact_emb()
        # hiddent state is mpdified
        output = torch.add(torch.matmul(hidden_states, w.T), b)
        return output

    def get_modified_weight_bias(self):
        wd = self.linear.weight.clone().detach()
        we = self.extra_output.weight
        if self.linear.bias is not None:
            bd = self.linear.bias.clone().detach()
        else:
            bd = torch.tensor([0] * wd.size(0)).to(wd.device)
        be = self.extra_output.bias


        w = torch.cat((wd, we), dim=0)
        b = torch.cat((bd, be), dim=0)

        return w, b

    def export_patch(self):
        we = self.extra_output.weight
        be = self.extra_output.bias
        return {'extra_output.weight':we.detach().cpu(),'extra_output.bias':be.detach().cpu()}

    def import_patch(self,weights):

        self.extra_output.weight=nn.Parameter(weights['extra_output.weight'].clone())
        self.extra_output.bias=nn.Parameter(weights['extra_output.bias'].clone())
        self.extra_output.to(self.device)

    def assign_layer(self):
        #print('assign')
        w, b = self.get_modified_weight_bias()
        new_layer = nn.Linear(self.hidden_size, self.intermediate_size + self.add_neuron_num)
        new_layer.weight = nn.Parameter(w.clone().detach())
        new_layer.bias = nn.Parameter(b.clone().detach())
        new_layer.bias.requires_grad = False
        new_layer.weight.requires_grad = False
        return new_layer

class ModifyLinearInput_single(nn.Module):  # nn.Linear(input_size, output_size) -> nn.Linear(input_size+1, output_size)
    def __init__(self, linear: nn.Linear, loc: int = -1,
                 amplify=False, freeze_a=False, amplify_con=10.,
                 add_neuron_num=1,device=None,args=None):
        super().__init__()
        self.device=device
        self.args=args
        self.linear = copy.deepcopy(linear)
        self.add_neuron_num = add_neuron_num
        self.hidden_size = min(self.linear.weight.size())
        self.intermediate_size = max(self.linear.weight.size())
        self.loc = loc
        self.extra_input = nn.Parameter(torch.randn([self.hidden_size, self.add_neuron_num]))
        self.drp=nn.Dropout(0.2)
        self.amplify = amplify
        if self.amplify:
            self.a = nn.Parameter(torch.randn([self.hidden_size, self.add_neuron_num]))
            self.amplify_con = amplify_con
        self._reset_parameters()
        if freeze_a:
            self.a.requires_grad = False

    def freeze_self(self):
        self.extra_input.requires_grad = False
        self.a.requires_grad = False

    def unfreeze_self(self):
        self.extra_input.requires_grad = True
        self.a.requires_grad = True

    def _reset_parameters(self):
        if self.amplify:
            init.constant_(self.a, self.amplify_con)

    def get_modified_weight_bias(self):
        wd = self.linear.weight.clone().detach()
        if self.linear.bias is not None:
            b = self.linear.bias.clone().detach()
        else:
            b = torch.tensor([0] * wd.size(0)).to(wd.device)
        we = self.extra_input
        if self.amplify:
            we = self.extra_input * self.a

        w = torch.cat((wd, we), dim=1)
        return w, b

    def forward(self, hidden_states):
        # #print('Pass input')
        w, b = self.get_modified_weight_bias()
        if  self.args.patch_drop:
            output = torch.add(torch.matmul( self.drp(hidden_states), w.T), b)
        else:
            output = torch.add(torch.matmul(hidden_states, w.T), b)
        return output

    def export_patch(self):
        we =  self.extra_input
        a=None
        if self.amplify:
            # we = self.extra_input * self.a
            a=self.a
        return {'extra_input': we.detach().cpu(),'a':a.detach().cpu()}

    def import_patch(self, weights):
        self.extra_input = nn.Parameter(weights['extra_input'].clone().to(self.device))
        if weights['a']!=None:
            self.a=nn.Parameter(weights['a'].clone().to(self.device))

        # self.extra_input.to(self.device)


    def assign_layer(self):
        #print('assign')
        w, b = self.get_modified_weight_bias()
        new_layer = nn.Linear(self.intermediate_size + self.add_neuron_num, self.hidden_size)
        new_layer.weight = nn.Parameter(w.clone().detach())
        new_layer.bias = nn.Parameter(b.clone().detach())
        new_layer.bias.requires_grad = False
        new_layer.weight.requires_grad = False

        return new_layer


#Editor for fever
class BertBinaryEditor(LightningModule):


    def __init__(self,toks=None, *args, **kwargs):
        super().__init__()
        self.toks=toks
        self.save_hyperparameters()
        # add_neuron_num = kwargs.get('add_neuron_num')
        self.current_device = torch.device('cuda', self.hparams.gpus[0])

        if self.hparams.editor_tp == 'patch':

            self.editor = Editor(
                model=BertBinary.load_from_checkpoint(self.hparams.model_path),
                max_add_neuron_num=self.hparams.max_add_neuron_num,
                freeze_model=self.hparams.freeze_model,
                amplify_v=self.hparams.amplify_v == 1,

                freeze_k=self.hparams.freeze_k == 1,
                freeze_a=self.hparams.freeze_a == 1,
                act_margin_val=self.hparams.act_margin_val,
                device=self.current_device,
                args=self.hparams
            )
        ##

        self.val_loader = None
        self.valid_memory_loss = []
        self.valid_metric = []
        self.save_ckpt = 0
        self.start_example_editing = False
        self.stop_editing = False
        self.BIG_CONSTANT = 10000
        self.has_stepped = False

    def on_train_start(self):
        self.valid_memory_loss = []
        self.valid_metric = []
        self.save_ckpt = 0
        self.start_example_editing = False
        self.stop_editing = False
        self.has_stepped = False

    @staticmethod
    def early_stop_editing(metrics, mode='min', thd=0.0, patience=1):
        best_step = 0
        for step, vm in enumerate(metrics):
            if mode == 'min':
                if vm < metrics[best_step] - thd:
                    best_step = step
            else:
                if vm > metrics[best_step] + thd:
                    best_step = step

        if best_step < len(metrics) - patience:
            return True
        else:
            return False

    @staticmethod
    def save_editing_ckpt(metrics, mode='min'):
        # save the model if new val metric is attained
        return (mode == 'min' and min(metrics) == metrics[-1]) or (mode == 'max' and max(metrics) == metrics[-1])

    def fed_val_loader(self, dl):
        self.val_loader = dl

    def reset(self, clear_memory):
        self.editor.reset_model(
            BertBinary.load_from_checkpoint(self.hparams.model_path),
            clear_memory=clear_memory
        )

    def forward(self, input_ids, attention_mask, labels):
        res = self.editor(input_ids, attention_mask, labels)
        return res

    def joint_training(self, batch, batch_idx=None):
        input_ids, attention_mask, labels= batch["src_input_ids"], batch["src_attention_mask"], batch["labels"]
        res = self.editor(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = res['loss']

        return {"loss": loss}

    def training_step(self, batch, batch_idx=None, optimizer_idx=None):
        return self.joint_training(batch, batch_idx)

    def joint_validation(self, batch, batch_idx=None):
        stop_editing, save_ckpt = False, False
        if not self.hparams_initial.re_vaild:
            input_ids, attention_mask, labels = batch["src_input_ids"], batch["src_attention_mask"], batch["labels"]
            b_size = input_ids.size(0)
            res = self.editor(input_ids=input_ids,attention_mask= attention_mask, labels=labels)
        else:
            input_ids, attention_mask, labels = batch["vaild_input_ids"], batch["vaild_attention_mask"], batch["va_labels"]
            b_size = input_ids.size(0)
            res = self.editor(input_ids=input_ids,attention_mask= attention_mask, labels=labels)

        self.log("val_loss", res['loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=b_size)

        stop_editing = float(res['metric'] == 1)
        save_ckpt = res['metric'] == 1

        self.save_ckpt += float(save_ckpt)
        self.stop_editing = stop_editing
        if stop_editing == 1:

            self.has_stepped = True
        self.log("stop_editing", stop_editing, on_step=False, on_epoch=True, batch_size=b_size, prog_bar=True)
        self.log("save_ckpt", self.save_ckpt, on_step=False, on_epoch=True, batch_size=b_size, prog_bar=True)

    def validation_step(self, batch, batch_idx=None):
        self.joint_validation(batch, batch_idx)

    def test_step(self, batch, batch_idx=None):
        input_ids, attention_mask, labels = batch["src_input_ids"], batch["src_attention_mask"], batch["labels"]
        b_size = input_ids.size(0)
        res = self.editor(input_ids, attention_mask, labels)
        test_loss = res['loss']
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=b_size)

    def get_optimizer(self, params, lr=None, optim=None):
        if lr is None:
            lr = self.hparams.lr
        if optim is None:
            optim = self.hparams.optim
        if optim == "adam":
            return torch.optim.Adam(params=params, lr=lr, weight_decay=self.hparams.weight_decay)
        if optim == 'rmsprop':
            return torch.optim.RMSprop(params=params, lr=lr)
        if optim == 'sgd':
            return torch.optim.SGD(params=params, lr=lr, weight_decay=self.hparams.weight_decay, momentum=0.9)


    def configure_optimizers(self):
        # for the joint editing style, we just need one parameter
        parameters = [p for p in self.editor.parameters() if p.requires_grad]
        optimizer = self.get_optimizer(parameters)

        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer, mode='min',
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
            threshold=0.05
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch", "frequency": self.hparams.check_val_every_n_epoch,
            "monitor": "val_loss", "strict": True
        }
        optimizer_list = [optimizer]
        lr_scheduler_config_list = [lr_scheduler_config]

        return optimizer_list, lr_scheduler_config_list
########################################################################