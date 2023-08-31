import torch
from torch.utils.data import DataLoader, Subset, random_split, Dataset
def split_data_n_sets(d, n_set=1):
    data_len = len(d) // n_set
    print(data_len)
    # assert  data_len==0
    data_size, i = [], 0
    while i + data_len < len(d):
        data_size.append(data_len)
        i += data_len
    data_size.append(len(d) - i)
    edit_sets_list = random_split(dataset=d, lengths=data_size, generator=torch.Generator().manual_seed(42))
    return edit_sets_list


def lower_and_strip_list(inp):
    return [i.lower().strip() for i in inp]

def edit_or_not_seq2seq(model, data_point, device, test_rephrases=True,single=-1,test_imp=False,args=None,edit=False):

    with torch.no_grad():
        model.eval()
        model.to(device)
        batch = data_point
        if edit:
            # print('EVAL  EDITING')
            if args.train_way=='ori':
                input_ids = batch["src_input_ids"]
                input_atts =batch["src_attention_mask"]
                if 're_src_input_ids' in batch.keys():
                    re_input_ids=batch["re_src_input_ids"]
                    re_input_atts=batch["re_src_attention_mask"]
            elif args.train_way=='sen_prompt':
                input_ids = batch["src_input_ids"][:,1:]
                input_atts =batch["src_attention_mask"][:,1:]
                # fact_ids=batch['fact_src_input_ids'][:,:-1]
                # fact_att=batch['fact_src_attention_mask'][:,:-1]
                facts_input = '||'.join(batch['raw'][0]['fact_src'].split('||')[:-1])
                fact_tok = model.tokenizer(facts_input, return_tensors="pt",
                                                       padding=True, max_length=30,
                                                       truncation=True, )
                fact_ids = fact_tok['input_ids'].to(input_ids.device)
                fact_att = fact_tok['attention_mask'].to(input_ids.device)
                input_ids=torch.cat([torch.repeat_interleave(fact_ids,len(input_ids),0),input_ids],dim=1,)
                input_atts=torch.cat([torch.repeat_interleave(fact_att,len(input_atts),0),input_atts],dim=1,)
                if 're_src_input_ids' in batch.keys():
                    re_input_ids=torch.cat([torch.repeat_interleave(fact_ids,len(batch["re_src_input_ids"]),0),batch["re_src_input_ids"]],dim=1)
                    re_input_atts=torch.cat([torch.repeat_interleave(fact_att,len(batch["re_src_attention_mask"]),0),batch["re_src_attention_mask"]],dim=1,)
            elif args.train_way == 'sen_combine':
                input_ids = batch["src_input_ids"]
                input_atts =batch["src_attention_mask"]
                fact_ids=batch['fact_src_input_ids']
                fact_att=batch['fact_src_attention_mask']
                input_ids=torch.cat([torch.repeat_interleave(fact_ids,len(input_ids),0),input_ids],dim=1,)
                input_atts=torch.cat([torch.repeat_interleave(fact_att,len(input_atts),0),input_atts],dim=1,)
                if 're_src_input_ids' in batch.keys():
                    re_input_ids=torch.cat([torch.repeat_interleave(fact_ids,len(batch["re_src_input_ids"]),0),batch["re_src_input_ids"]],dim=1)
                    re_input_atts=torch.cat([torch.repeat_interleave(fact_att,len(batch["re_src_attention_mask"]),0),batch["re_src_attention_mask"]],dim=1,)
        else:
            # print('EVAL without EDITING')
            input_ids = batch["src_input_ids"]#[:,1:]
            input_atts =batch["src_attention_mask"]#[:,1:]
            if 're_src_input_ids' in batch.keys():
                re_input_ids=batch["re_src_input_ids"]
                re_input_atts=batch["re_src_attention_mask"]

        prediction = model.model.generate(
            input_ids=input_ids[[0]].to(device),
            attention_mask=input_atts[[0]].to(device),
            min_length=0, num_return_sequences=1, use_cache=True
        )
        prediction = model.tokenizer.batch_decode(prediction, skip_special_tokens=True)
        prediction = lower_and_strip_list(prediction)
        targets = lower_and_strip_list(batch["raw"][0]["trg"])
        need_edit = 1 if prediction[0] not in targets else 0
        if test_rephrases:
            prediction_re = model.model.generate(
                input_ids=re_input_ids.to(device),
                attention_mask=re_input_atts.to(device),
                min_length=0, num_return_sequences=1, use_cache=True
            )
            prediction_re = lower_and_strip_list(model.tokenizer.batch_decode(prediction_re, skip_special_tokens=True))
            correct_count = 0
            for p in prediction_re:
                correct_count += float(p in targets)
            correct_count /= len(prediction_re)
            re_num = [p in targets for p in prediction_re]

            return [prediction,targets,1 if prediction[0] not in targets else 0],\
                   [prediction_re,targets,[p in targets for p in prediction_re]],\
                   []
        else:
            correct_count = 0
            re_num = []
    return [prediction,targets,1 if prediction[0] not in targets else 0],[],[]


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

def edit_or_not_binary(model, data_point, device, args=None,single=False,edit=False):

    with torch.no_grad():
        model.eval()
        model.to(device)
        batch = data_point

        if edit:
            # print('EVAL  EDITING')
            if args.train_way == 'ori':
                input_ids = batch["src_input_ids"]
                input_atts = batch["src_attention_mask"]
                if not single:
                    re_input_ids = batch["re_src_input_ids"]
                    re_input_atts = batch["re_src_attention_mask"]
            elif args.train_way == 'sen_prompt':
                input_ids = batch["src_input_ids"][:, 1:]
                input_atts = batch["src_attention_mask"][:, 1:]
                fact_ids = batch['fact_src_input_ids'][:, :-1]
                fact_att = batch['fact_src_attention_mask'][:, :-1]
                input_ids = torch.cat([torch.repeat_interleave(fact_ids, len(input_ids), 0), input_ids], dim=1, )
                input_atts = torch.cat([torch.repeat_interleave(fact_att, len(input_atts), 0), input_atts], dim=1, )
                if not single:
                    re_input_ids = torch.cat([torch.repeat_interleave(fact_ids, len(batch["re_src_input_ids"]), 0),
                                              batch["re_src_input_ids"]], dim=1)
                    re_input_atts = torch.cat(
                        [torch.repeat_interleave(fact_att, len(batch["re_src_attention_mask"]), 0),
                         batch["re_src_attention_mask"]], dim=1, )
            elif args.train_way == 'sen_combine':
                input_ids = batch["src_input_ids"]
                input_atts = batch["src_attention_mask"]
                fact_ids = batch['fact_src_input_ids']
                fact_att = batch['fact_src_attention_mask']
                input_ids = torch.cat([torch.repeat_interleave(fact_ids, len(input_ids), 0), input_ids], dim=1, )
                input_atts = torch.cat([torch.repeat_interleave(fact_att, len(input_atts), 0), input_atts], dim=1, )
                if not single:
                    re_input_ids = torch.cat([torch.repeat_interleave(fact_ids, len(batch["re_src_input_ids"]), 0),
                                              batch["re_src_input_ids"]], dim=1)
                    re_input_atts = torch.cat(
                        [torch.repeat_interleave(fact_att, len(batch["re_src_attention_mask"]), 0),
                         batch["re_src_attention_mask"]], dim=1, )
        else:
            # print('EVAL without EDITING')
            input_ids = batch["src_input_ids"]  # [:,1:]
            input_atts = batch["src_attention_mask"]
            # [:,1:]
            if not single:
                re_input_ids = batch["re_src_input_ids"]
                re_input_atts = batch["re_src_attention_mask"]

        res = model(
            input_ids[[0]].to(device),input_atts[[0]].to(device),
            batch["labels"][[0]].to(device),
        )

        if single:
            return [res['logits'].cpu().item(), batch["labels"].cpu().item(), 1 - res['metric'].cpu().item()], \
                [], \
                []
        else:
            re_res = model(
                re_input_ids.to(device), re_input_atts.to(device),
                batch["re_labels"].to(device),
            )
            return [res['logits'].cpu().item(), batch["labels"][[0]].cpu().item(),1 - res['metric'].cpu().item()],\
                   [(re_res['logits']>0).bool().cpu().tolist(),batch["re_labels"].cpu().tolist(), ((re_res['logits']>0).bool().cpu()==batch["re_labels"].bool().cpu()).tolist()],\
                   []


