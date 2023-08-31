
import os
from argparse import ArgumentParser

import torch

import torch.nn as nn


#
#
from typing import Optional
import numpy as np
from pytorch_lightning import LightningModule

from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torchmetrics import Accuracy
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp,pair=False):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.pair=pair

    def forward(self, x, y):

        return self.cos(x, y) / self.temp

class cl_for_fact(nn.Module):
    def __init__(self, encode_model,proj_layer='highway',n_proj_layer=2,proj_fact=True,fact_size=None,batch_size=128,hard=False,data_tp=None,args=None,tokenizer=None):
        super().__init__()
        self.args = args
        self.encode_model=encode_model
        self.tokenizer=  self.tokenizer=tokenizer
        if self.args.type=='Bart':
            transformer_hidden_size=self.encode_model.model.config.d_model
        else:
            transformer_hidden_size=1024
        box_dim = self.args.box_dim
        self.batch_size=batch_size

        self.sim=Similarity(temp=0.1, args=self.args)
        self.ent_sim=Similarity(temp=self.args.ent2sen_temp,args=self.args)
        self.hard=hard
        self.loss_fct=nn.CrossEntropyLoss()
        self.data_tp=data_tp
        for n,p in self.encode_model.named_parameters():
            p.requires_grad=False
        #
        self.dp=nn.Dropout(0.5)
        if not proj_fact:
            # fact_embedding= nn.Embedding(len(entity_vocab.keys()),box_dim)
            fact_embedding = np.random.uniform(
                low=-0.05, high=0.05, size=fact_size
            )
            fact_embedding[0] = np.zeros(box_dim)
            self.fact_embedding.weight = nn.Parameter(
                torch.FloatTensor(fact_embedding)
            )
            self.fact_embedding.weight.requires_grad = True


        if proj_layer == "linear":
            self.sen_base_proj =LinearProjection(
                transformer_hidden_size,
                box_dim )
            if  proj_fact:
                self.rel_base_proj =LinearProjection(
                    transformer_hidden_size,
                    box_dim )
        elif proj_layer == "mlp":
            self.sen_base_proj = SimpleFeedForwardLayer(
                transformer_hidden_size,
                box_dim ,
                activation=nn.Sigmoid())
            if proj_fact:
                self.rel_base_proj =SimpleFeedForwardLayer(
                    transformer_hidden_size,
                    box_dim ,
                    activation=nn.Sigmoid())

        elif proj_layer == "highway":
            self.sen_base_proj =  HighwayNetwork(
                transformer_hidden_size,
                box_dim ,
                n_proj_layer,
                activation=nn.ReLU())
            if proj_fact:
                self.rel_base_proj =  HighwayNetwork(
                    transformer_hidden_size,
                    box_dim ,
                    n_proj_layer,
                    activation=nn.ReLU())

        else:
            raise ValueError(args.proj_layer)

    def Avgpool(self,inpt,laebls_len):

        res = []
        h = 0
        for ii, i in enumerate(laebls_len):
            res.append([])
            l = 0
            for jj, j in enumerate(laebls_len):

                res[-1].append(torch.mean(inpt[h:i + h, l:j + l]))
                l += j
            h += i

        return torch.tensor(res).to(inpt.device)

    def forward(self, **kwargs):
        # training_step defines the train loop.
        # it is independent of forward
        if self.data_tp:
            input_fact_input_ids = kwargs['fact_src_input_ids']
            input_fact_attention_mask = kwargs['fact_src_attention_mask']
            input_fact_src_input_ids = kwargs['src_input_ids']
            input_fact_src_attention_mask = kwargs['src_attention_mask']

            fact_labels=kwargs['src_fact_labels']

            fact_in_src=kwargs['trg_input_ids']
            fact_in_src_att=kwargs['trg_attention_mask']


            with torch.no_grad():
                fact_emb = self.encode_model(**{'input_ids': input_fact_input_ids.to(self.encode_model.device), 'attention_mask': input_fact_attention_mask.to(self.encode_model.device)},output_hidden_states=True, return_dict=True).pooler_output

                src_emb = self.encode_model(**{'input_ids': input_fact_src_input_ids.to(self.encode_model.device), 'attention_mask': input_fact_src_attention_mask.to(self.encode_model.device)}, output_hidden_states=True, return_dict=True).pooler_output

                src_fact = self.encode_model(**{'input_ids': fact_in_src.to(self.encode_model.device), 'attention_mask': fact_in_src_att.to(self.encode_model.device)},output_hidden_states=True, return_dict=True).pooler_output

            pooler_output = fact_emb.view((int(fact_emb.shape[0]/2), 2, fact_emb.size(-1)))  #
            fact_1, fact_2 = pooler_output[:, 0], pooler_output[:, 1]
            pooler_output = src_emb.view((int(src_emb.shape[0]/2), 2, src_emb.size(-1)))  #
            src_1, src_2 = pooler_output[:, 0], pooler_output[:, 1]

            #rep for CSE
            fact_1, fact_2 = self.rel_base_proj(fact_1), self.rel_base_proj(fact_2)
            src_1, src_2 = self.sen_base_proj(src_1), self.sen_base_proj(src_2)
            fact_insrc_1= self.sen_base_proj(src_fact)
            if kwargs['training']:

                 #sen 2 sen
                sen2sen_sim = self.sim(src_1.unsqueeze(1), src_2.unsqueeze(0))
                labels = torch.arange(sen2sen_sim.size(0)).long().to(sen2sen_sim.device)
                sen2sen_loss = self.loss_fct(sen2sen_sim, labels)
                #fact 2 fact
                fact2fact_sim = self.sim(fact_1.unsqueeze(1), fact_2.unsqueeze(0))
                labels = torch.arange(fact2fact_sim.size(0)).long().to(fact2fact_sim.device)
                fact2fact_loss = self.loss_fct(fact2fact_sim, labels)
                #fact 2 sen
                fact2sen_sim = self.ent_sim(src_1.unsqueeze(1), fact_1.unsqueeze(0))
                fact2sen_src_sim = self.ent_sim(fact_insrc_1.unsqueeze(1), fact_1.unsqueeze(0))
                fact2sen_sim= fact2sen_sim*0.4+fact2sen_src_sim*0.6
                if self.hard:
                    with torch.no_grad():
                        hard_emb=self.encode_model(**{
                            'input_ids':kwargs['pos_fact_hard_input_ids'].to(self.encode_model.device),
                            'attention_mask':kwargs['pos_fact_hard_attention_mask'].to(self.encode_model.device)  })[1]
                    hard_emb=self.sen_base_proj(hard_emb)
                    fact2sen_sim=torch.cat([fact2sen_sim,self.ent_sim(hard_emb.unsqueeze(1), fact_1.unsqueeze(0))],dim=0)

                labels = torch.tensor(fact_labels).long().to(fact2sen_sim.device)
                fact2sen_loss = self.loss_fct(fact2sen_sim, labels)

                return sen2sen_loss,fact2fact_loss,fact2sen_loss
            else:

                with torch.no_grad():

                    sims = nn.functional.cosine_similarity(src_1.unsqueeze(1), fact_1.unsqueeze(0),
                                                           dim=-1).detach().cpu()
                    max_values, max_index = torch.max(sims, dim=1)

                return {
                    'max_values': max_values.tolist(),
                    'max_index': max_index.tolist(),
                    'true_label': fact_labels,
                }
        else:

            input_fact_input_ids=kwargs['input_fact_input_ids']
            input_fact_attention_mask=kwargs['input_fact_attention_mask']
            input_fact_src_input_ids=kwargs['input_fact_src_input_ids']
            input_fact_src_attention_mask=kwargs['input_fact_src_attention_mask']

            with torch.no_grad():
                if self.args.type!='Bart':
                    fact_emb = self.encode_model(**{'input_ids': input_fact_input_ids.to(self.encode_model.device), 'attention_mask': input_fact_attention_mask.to(self.encode_model.device)}, output_hidden_states=True, return_dict=True).pooler_output

                    src_emb = self.encode_model(**{'input_ids': input_fact_src_input_ids.to(self.encode_model.device), 'attention_mask': input_fact_src_attention_mask.to(self.encode_model.device)}, output_hidden_states=True, return_dict=True).pooler_output
                else:
                    fact_emb = self.encode_model.model(**{'input_ids': input_fact_input_ids.to(self.encode_model.model.device),
                                                    'attention_mask': input_fact_attention_mask.to(
                                                        self.encode_model.model.device)}, output_hidden_states=True)

                    eos_mask = input_fact_input_ids.eq(self.encode_model.model.config.eos_token_id)
                    hidden_states = fact_emb.decoder_hidden_states[-1]

                    fact_emb = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                                         hidden_states.size(-1))[:, -1, :]

                    src_emb = self.encode_model.model(**{'input_ids': input_fact_src_input_ids.to(self.encode_model.model.device),
                                                   'attention_mask': input_fact_src_attention_mask.to(
                                                       self.encode_model.model.device)}, output_hidden_states=True)

                    eos_mask = input_fact_src_input_ids.eq(self.encode_model.model.config.eos_token_id)
                    hidden_states = src_emb.decoder_hidden_states[-1]

                    src_emb = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                                         hidden_states.size(-1))[:, -1, :]


            pooler_output = fact_emb.view((int(fact_emb.shape[0] / 2), 2, fact_emb.size(-1)))  #
            fact_1, fact_2 = pooler_output[:, 0], pooler_output[:, 1]

            pooler_output = src_emb.view((int(src_emb.shape[0] / 2), 2, src_emb.size(-1)))  #
            src_1, src_2 = pooler_output[:, 0], pooler_output[:, 1]

            # rep for CSE
            fact_1, fact_2 = self.rel_base_proj(fact_1), self.rel_base_proj(fact_2)
            src_1, src_2 = self.sen_base_proj(src_1), self.sen_base_proj(src_2)

        if kwargs['training']:

            # sen 2 sen
            # sen2sen_sim = self.sim(src_1.unsqueeze(1), src_2.unsqueeze(0))
            # labels = torch.arange(sen2sen_sim.size(0)).long().to(sen2sen_sim.device)
            # sen2sen_loss = self.loss_fct(sen2sen_sim, labels)

            group_sen_sim = self.Avgpool(self.sim(src_1.unsqueeze(1), src_1.unsqueeze(0)), kwargs['labels_len'])
            group_sen2sen_loss = self.loss_fct(group_sen_sim, torch.arange(group_sen_sim.size(0)).long().to(group_sen_sim.device))
            sen2sen_loss=group_sen2sen_loss
            # fact 2 fact
            fact2fact_sim = self.sim(fact_1.unsqueeze(1), fact_2.unsqueeze(0))
            labels = torch.arange(fact2fact_sim.size(0)).long().to(fact2fact_sim.device)
            fact2fact_loss = self.loss_fct(fact2fact_sim, labels)
            # fact 2 sen
            fact2sen_sim = self.ent_sim(src_1.unsqueeze(1), fact_1.unsqueeze(0),kwargs['labels'])
            # if self.hard:
            #     with torch.no_grad():
            #         hard_emb = self.encode_model(**{
            #             'input_ids': kwargs['pos_fact_hard_input_ids'].to(self.encode_model.device),
            #             'attention_mask': kwargs['pos_fact_hard_attention_mask'].to(self.encode_model.device)})[1]
            #     hard_emb = self.sen_base_proj(hard_emb)
            #     fact2sen_sim = torch.cat([fact2sen_sim, self.ent_sim(hard_emb.unsqueeze(1), fact_1.unsqueeze(0))],
            #                              dim=0)

            # labels = torch.arange(fact2sen_sim.size(0)).long().to(fact2fact_sim.device)
            fact2sen_loss = self.loss_fct(fact2sen_sim, torch.tensor(kwargs['labels']).long().to(fact2sen_sim.device))

            return sen2sen_loss, fact2fact_loss, fact2sen_loss
        else:

            with torch.no_grad():

                sims = nn.functional.cosine_similarity(src_1.unsqueeze(1), fact_1.unsqueeze(0),
                                                       dim=-1).detach().cpu()
                max_values, max_index = torch.max(sims, dim=1)

            return {
                'max_values': max_values.tolist(),
                'max_index': max_index.tolist(),
                'true_label':kwargs['labels'],
            }


class HighwayNetwork(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               n_layers: int,
               activation: Optional[nn.Module] = None):
    super(HighwayNetwork, self).__init__()
    self.n_layers = n_layers
    self.nonlinear = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    self.gate = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    for layer in self.gate:
      layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
    self.final_linear_layer = nn.Linear(input_dim, output_dim)
    self.activation = nn.ReLU() if activation is None else activation
    self.sigmoid = nn.Sigmoid()
    self.dropout=nn.Dropout(0.2)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    for layer_idx in range(self.n_layers):
      gate_values = self.sigmoid(self.gate[layer_idx](self.dropout(inputs)))
      nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
      inputs = gate_values * nonlinear + (1. - gate_values) * inputs
    return self.final_linear_layer(inputs)

class LinearProjection(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               bias: bool = True):
    super(LinearProjection, self).__init__()
    self.linear = nn.Linear(input_dim, output_dim, bias=bias)
    self.dp=   nn.Dropout(0.2)
  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    outputs =  self.dp(self.linear(inputs))
    return outputs


class SimpleFeedForwardLayer(nn.Module):
  """2-layer feed forward"""
  def __init__(self,
               input_dim: int,
               output_dim: int,
               bias: bool = True,
               activation: Optional[nn.Module] = None):
    super(SimpleFeedForwardLayer, self).__init__()
    self.linear_projection1 = nn.Linear(input_dim,
                                        (input_dim + output_dim) // 2,
                                        bias=bias)
    self.linear_projection2 = nn.Linear((input_dim + output_dim) // 2,
                                        output_dim,
                                        bias=bias)
    self.activation = activation if activation else nn.Sigmoid()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    inputs = self.activation(self.linear_projection1(inputs))
    inputs = self.activation(self.linear_projection2(inputs))
    return inputs


class BertBinary(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--train_data_path", type=str,
            default='data/fever_data/fever-train.jsonl'
        )
        parser.add_argument(
            "--dev_data_path", type=str,
            default='data/fever_data/fever-val.jsonl'
        )
        parser.add_argument(
            "--test_data_path", type=str,
            default='data/fever_data/fever-dev-kilt.jsonl'
        )
        # parser.add_argument("--add_pooling_layer", type=bool, default=False)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_steps", type=int, default=10000)
        parser.add_argument("--warmup_updates", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=32)
        parser.add_argument("--model_name", type=str, default="bert-base-uncased")
        parser.add_argument("--eps", type=float, default=0.1)

        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        model_dir = os.path.join(self.hparams.cache_dir, self.hparams.model_name) \
            if "cache_dir" in self.hparams else self.hparams.model_name
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.model = BertClassifier(model_dir)
        except:
            print("The transformer cache can not be used")
            self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model_name)  # have internet
            self.model = BertClassifier(self.hparams.model_name)  # have internet
        self.train_acc = Accuracy(threshold=0.0)
        self.valid_acc = Accuracy(threshold=0.0)

    def train_dataloader(self, shuffle=True):
        if not hasattr(self, "train_dataset") or not hasattr(self, "train_loader"):
            self.train_dataset = FeverData(
                tokenizer=self.tokenizer,
                data_path=self.hparams.train_data_path, max_length=self.hparams.max_length
            )
            print("The training dataset has {} data\n".format(len(self.train_dataset)))
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.train_dataset.collate_fn,
                num_workers=self.hparams.num_workers,
                shuffle=shuffle,
            )
        return self.train_loader

    def val_dataloader(self):
        if not hasattr(self, "val_dataset") or not hasattr(self, "val_loader"):
            self.val_dataset = FeverData(
                tokenizer=self.tokenizer,
                data_path=self.hparams.dev_data_path, max_length=self.hparams.max_length
            )
            print("The validation dataset has {} data\n".format(len(self.val_dataset)))
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.val_dataset.collate_fn,
                num_workers=self.hparams.num_workers,
            )
        return self.val_loader

    def test_dataloader(self):
        if not hasattr(self, "test_dataset") or not hasattr(self, "test_loader"):
            self.test_dataset = FeverData(
                tokenizer=self.tokenizer,
                data_path=self.hparams.test_data_path, max_length=self.hparams.max_length
            )
            print("The test dataset has {} data\n".format(len(self.test_dataset)))
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.test_dataset.collate_fn,
                num_workers=self.hparams.num_workers,
            )
        return self.test_loader

    def forward(self, input_ids, attention_mask, labels):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cross_entropy = F.binary_cross_entropy_with_logits(logits, labels)
        # todo KnowledgeEditor这里加入了这个entropy，不知道为什么，需要研究一下
        entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean(-1)
        loss = cross_entropy - self.hparams.eps * entropy
        metric = self.train_acc(logits, labels.int())

        return {"loss": loss, "logits": logits, "metric": metric}

    def training_step(self, batch, batch_idx=None):
        logits = self.model(input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"])
        cross_entropy = F.binary_cross_entropy_with_logits(logits, batch['labels'])
        # todo KnowledgeEditor这里加入了这个entropy，不知道为什么，需要研究一下
        entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean(-1)

        loss = cross_entropy - self.hparams.eps * entropy
        self.log("CE", cross_entropy, on_step=True, on_epoch=False, prog_bar=True, batch_size=logits.size(0))
        self.log("E", entropy, on_step=True, on_epoch=False, prog_bar=True, batch_size=logits.size(0))
        self.train_acc(logits, batch["labels"].int())
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True, batch_size=logits.size(0))
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx=None):
        logits = self.model(input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"])
        self.valid_acc(logits, batch["labels"].int())
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=logits.size(0))
        return {"logits": logits}

    def sample(self, sentences, **kwargs):
        with torch.no_grad():
            return self.model(
                **{
                    k: v.to(self.device)
                    for k, v in self.tokenizer(
                        sentences,
                        return_tensors="pt",
                        padding=True,
                        max_length=self.hparams.max_length,
                        truncation=True,
                    ).items()
                }
            )

    def test_step(self, batch, batch_idx=None):
        logits = self.model(input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"])
        metric = self.valid_acc(logits, batch["labels"].int())
        self.log("metric", metric, batch_size=logits.size(0))  # , on_step=False, on_epoch=True, prog_bar=True)
        return {"logits": logits, "metric": metric}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            # bias 和 LayerNorm.weight 使用的是不同的weight_decay
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            }
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_steps,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, hidden_dim=768):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, *args, **kwargs):
        return self.classifier(self.model(*args, **kwargs)[1]).squeeze(-1)


class BartSeq2Seq(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--train_data_path", type=str,
            default='data/zsre_data/zsre-train.jsonl'
        )
        parser.add_argument(
            "--dev_data_path", type=str,
            default='data/zsre_data/zsre-val.jsonl'
        )
        # par
        parser.add_argument("--num_beams", type=int, default=NUM_BEAMS)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_updates", type=int, default=50000)
        parser.add_argument("--warmup_updates", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--model_name", type=str, default="facebook/bart-base")
        parser.add_argument("--eps", type=float, default=0.1)
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        model_dir = os.path.join(self.hparams.cache_dir, self.hparams.model_name) \
            if "cache_dir" in self.hparams else self.hparams.model_name
        try:
            self.tokenizer = BartTokenizer.from_pretrained('/media/sev/win/huggingface/facebook/bart-base')
            self.model = BartForConditionalGeneration.from_pretrained('/media/sev/win/huggingface/facebook/bart-base')
        except:
            print("The cache can not be used")
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_name)  # have internet
            self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_name)  # have internet
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

    def train_dataloader(self, shuffle=True):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = Seq2SeqData(
                tokenizer=self.tokenizer,
                data_path=self.hparams.train_data_path,
                max_length=self.hparams.max_length,
            )
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.hparams.num_workers, shuffle=shuffle,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = Seq2SeqData(
                tokenizer=self.tokenizer,
                data_path=self.hparams.dev_data_path,
                max_length=self.hparams.max_length,
                validation=True,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=16,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        # batch_size x trg_len x vocab_size
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        ).logits

    def training_step(self, batch, batch_idx=None):
        input_ids = batch["src_input_ids"]
        attention_mask = batch["src_attention_mask"]
        decoder_input_ids = batch["trg_input_ids"][:, :-1]
        decoder_attention_mask = batch["trg_attention_mask"][:, :-1]
        logits = self.forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
        loss, nll_loss = label_smoothed_nll_loss(
            logits.log_softmax(-1), batch["trg_input_ids"][:, 1:],
            epsilon=self.hparams.eps, ignore_index=self.tokenizer.pad_token_id,
        )

        ntokens = batch["trg_attention_mask"][:, 1:].sum()
        loss, nll_loss = loss / ntokens, nll_loss / ntokens
        self.log("nll_loss", nll_loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        trg = [b["trg"] for b in batch["raw"]]
        pred = self.tokenizer.batch_decode(
            self.model.generate(
                input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"],
                min_length=0, num_beams=self.hparams.num_beams, num_return_sequences=1, use_cache=True
            ),
            skip_special_tokens=True
        )
        acc = torch.tensor(
            [
                p.lower().strip() in [t_.lower().strip() for t_ in t]
                for t, p in zip(trg, pred)
            ]
        ).long()
        self.valid_acc(acc, torch.ones_like(acc))
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(trg))

    def test_step(self, batch, batch_idx=None):
        trg = [b["trg"] for b in batch["raw"]]
        pred = self.tokenizer.batch_decode(
            self.model.generate(
                input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"],
                min_length=0, num_beams=self.hparams.num_beams, num_return_sequences=1, use_cache=True
            ),
            skip_special_tokens=True
        )
        acc = [
                p.lower().strip() == t.lower()
                for t, p in zip(trg, pred)
        ]
        acc = torch.tensor(acc).long()
        self.valid_acc(acc, torch.ones_like(acc))
        self.log("test_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]


