import os

import jsonlines
import random
import torch
from torch.utils.data import Dataset



def data_split(dataset, ratio, shuffle=True):
    res, start = {}, 0
    offsets = {n: int(len(dataset)*r) for n, r in ratio.items()}
    if shuffle:
        random.shuffle(dataset)
    for n, offset in offsets.items():
        res[n] = dataset[start:start + offset]
        start += offset
    return res



class Seq2SeqData_FoeBox(Dataset):
    def __init__(self, tokenizer, data_path, max_length=32, example_repeat=16,
                 all_views=False, return_view=5, validation=False, edit=False,use_length=-1):
        """
        :param tokenizer:
        :param data_path:
        :param max_length:
        :param validation:
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.validation = validation
        self.edit = edit
        self.data = []
        self.example_repeat = example_repeat

        def extract(d):
            ex={}
            try:
                ex['input']=d['input']
                ex['subject']=d['subject']
                ex['answers']=d['output']
                ex['rephrases']=d['rephrases']
                ex['tem_q']=d['tem_q']
                ex['imp']=d['impl']

                # if   ex['imp']==[]:
                #     return {}
            except:
                ex={}
            return ex

        with jsonlines.open(data_path) as f:
            for di,d in enumerate(f):
                if di==use_length:
                    break
                extracted = extract(d)
                if  extracted!={}:
                    self.data.append(extracted)
        print(f'The original dataset size is {di}.')
        print(f'The loaded dataset size is {len(self.data)}.')
        if not self.edit:
            data_sub = [d['subject'][0][0] for d in self.data]
            new_data = []
            for i, j in zip(self.data, data_sub):
                tem = i
                tem['prompt'] = i['input'].replace(j, '{}')
                tem['subs'] = j
                new_data.append(tem)
            self.data=new_data
            del new_data
            del data_sub
            del tem
            self.tem_dict = {}
            for i in self.data:
                if i['prompt'] not in self.tem_dict:
                    self.tem_dict[i['prompt']] = [i]
                else:
                    self.tem_dict[i['prompt']].append(i)

            for i in self.data:
                i['negs']=[{
                    'src':d['input'],
                    'subject':d['subject'],
                    'template':d['tem_q'],
                    'rephrases':d['rephrases'],
                } for d in self.tem_dict[i['prompt']] if d['subject']!=i['subject']]
                i['negs_len']=len(i['negs'])
            self.mins=min([len(d['negs']) for d in self.data])
            self.maxs=max([len(d['negs']) for d in self.data])
        #     pass
        self.max_length = max_length
        self.all_views = all_views
        self.return_view = return_view

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        if self.edit:
            return {
                "src": self.data[item]["input"],
                "trg": self.data[item]["answers"],
                "rephrases": random.sample(
                    self.data[item]["rephrases"],
                    k=min(self.return_view, len(self.data[item]["rephrases"]))) if not self.all_views else self.data[item]["rephrases"],
                "subject":self.data[item]['subject'][0],
                "template":self.data[item]['tem_q'][0],
                "fact_src":self.data[item]["subject"][0][0] + '||' + self.data[item]["tem_q"][0]+ '||'+ self.data[item]["answers"][0],
                "impl":self.data[item]['imp'],


                "item":item,
            }
        else:
            return {
                "src": self.data[item]["input"],
                "trg": self.data[item]["answers"],
                "rephrases": random.sample(
                    self.data[item]["rephrases"],
                    k=min(self.return_view, len(self.data[item]["rephrases"]))) if not self.all_views else self.data[item]["rephrases"],
                "subject":self.data[item]['subject'][0],
                "template":self.data[item]['tem_q'][0],
                "fact_src":self.data[item]["subject"][0][0] + '||' + self.data[item]["tem_q"][0]+ '||'+ self.data[item]["answers"][0],
                "item":item,
                "prompt":self.data[item]["prompt"],
                "negs":self.data[item]["negs"],
                "negs_len":self.data[item]["negs_len"],
            }
    def collate_fn(self, batch):
        # batches = {}
        # b["subject"][0] + '||' + b["template"] + '||' + b['trg'][0]
        batches = {}
        if self.edit:
            vaild= [b['rephrases']+[b['src']] for b in batch]
            for name in ("src","fact_src",) + (() if self.validation else ("trg",)):
                tokenizer_input = [b[name] for b in batch]
                tokenizer_output = self.tokenizer(
                    tokenizer_input, return_tensors="pt",
                    padding=True, max_length=self.max_length,
                    truncation=True,
                )
                for k, v in tokenizer_output.items():
                    if name == 'src' and self.edit and self.example_repeat > 1:
                        v_ = [v for _ in range(self.example_repeat)]
                        batches["{}_{}".format(name, k)] = torch.cat(v_, dim=0)
                    else:
                        batches["{}_{}".format(name, k)] = v

            for i, j in self.tokenizer(
            vaild[0],
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
            ).items():
                 batches[f"vaild_{i}"]=j

            assert len(batch) == 1
            tokenizer_trg = self.tokenizer(
                [b["trg"][0] for b in batch], return_tensors="pt",
                padding=True, max_length=self.max_length,
                truncation=True,
            )
            for k, v in tokenizer_trg.items():
                if self.example_repeat == 1:
                    batches["{}_{}".format("trg", k)] = v
                else:
                    v_ = [v for _ in range(self.example_repeat)]
                    batches["{}_{}".format("trg", k)] = torch.cat(v_, dim=0)

            tokenize_rephrases = self.tokenizer(
                batch[0]["rephrases"],
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
            for k, v in tokenize_rephrases.items():
                batches['{}_{}'.format('re_src', k)] = v

            if "trg_input_ids" in batches:
                # batches["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id
                b_size = batches["trg_input_ids"].size(0)
                eos = torch.tensor([[self.tokenizer.eos_token_id] for _ in range(b_size)])
                mask = torch.tensor([[1] for _ in range(b_size)])
                batches["trg_input_ids"] = torch.cat((eos, batches["trg_input_ids"]), dim=-1)
                batches["trg_attention_mask"] = torch.cat((mask, batches["trg_attention_mask"]), dim=-1)

            imps=batch[0]['impl']
            imp_inpt=[i[0] for i in imps]
            imp_label=[i[1]  for i in imps]
            if imp_inpt!=[]:
                tokenize_imp = self.tokenizer(
                    imp_inpt,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                )
                for k, v in tokenize_imp.items():
                    batches['{}_{}'.format('imp_src', k)] = v

                trg_imp = self.tokenizer(
                    imp_label,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                )
                for k, v in trg_imp.items():
                    batches['{}_{}'.format('imp_trg', k)] = v


            batches["raw"] = batch
        else:
            src = []
            fact_src=[]
            reshapes=[]
            raws=[]
            for b in batch:
                # for i in range(2):  ## src repeat for CL
                src.append(b['src'])
                fact_src.append(b["subject"][0] + '||' + b["template"])
                reshapes.append(b['rephrases'])
                raws.append(b)
                # for i in range(b['negs_len'] if b['negs_len']<2 else 2 ):
                for i in range(0):
                    src.append(b['negs'][i]['src'])
                    fact_src.append(b['negs'][i]['subject'][0][0]+ '||' + b['negs'][i]['template'][0])
                    reshapes.append(b['negs'][i]['rephrases'][:len(b['negs'][i]['rephrases']) if len(b['negs'][i]['rephrases'])<self.return_view else self.return_view ])
                    raws.append(b['negs'][i])
            fact_input = list(set(fact_src))
            # fact_labels=[{i:fi  for fi,i in enumerate(set(fact_src))}[f] for f in fact_input]
            src_fact_labels=[{i:fi  for fi,i in enumerate(set(fact_src))}[f] for f in fact_src]

            batches = {
                f"{k1}_{k2}": v2
                for k1, v1 in {
                    "src": src,
                    # "trg": trg,
                    "fact_src": fact_input,
                    # "re_src": [b["rephrases"][0] for b in batch]
                }.items()
                for k2, v2 in self.tokenizer(
                    v1,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                ).items()
            }
            batches['src_fact_labels']=src_fact_labels

            tokenize_rephrases = [self.tokenizer(
                b,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ) for b in reshapes]
            batches['tokenize_rephrases'] = tokenize_rephrases

            batches["raw"] =raws
        return batches


class FeverEditData_Formine(Dataset):

    def __init__(self, tokenizer=None, data_path=None, max_length=32, all_rephrase=True, example_repeat=16,use_length=-1,return_view=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.data_path = data_path
        self.max_length = max_length
        self.all_rephrase = all_rephrase
        self.example_repeat = example_repeat
        self.return_view= return_view
        wrongs=[]
        def extract(d):
            ex={}
            try:
                ex['input']=d['input']
                if '==' in ex['input']:
                    ex = {}
                    return ex
                # ex['subject']=d['subject']
                ex['answers']=d['output'][0]["answer"]
                ex['rephrases']=[di for di in d['rephrases'] if '=' not in di]
                ex['entity']=list(set([dd[0][2] for dd in d['evi']]))
                if '=' in ex['input']:
                    wrongs.append(ex['input'])
                for r in  ex['rephrases']:
                    if '=' in r:
                        wrongs.append(r)
                # ex['tem_q']=d['tem_q']
                if len( ex['rephrases']) ==0:
                    ex={}
            except:
                ex={}
            return ex


        if self.data_path is not None:
            with jsonlines.open(self.data_path) as f:
                for di,d in enumerate(f):
                    if di==use_length:
                        break
                    extracted = extract(d)
                    if extracted != {}:
                        self.data.append(extracted)

                    # if len(d["rephrases"]) > 0:
                    #     self.data.append({
                    #         "input": d["input"],
                    #         "rephrases": d["rephrases"],
                    #         "label": d["output"][0]["answer"]
                    #     })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "src": self.data[item]["input"],
            "trg": self.data[item]["answers"] == "SUPPORTS",
            "rephrase": random.sample(
                self.data[item]["rephrases"],
                k=min(self.return_view, len(self.data[item]["rephrases"]))) ,
            "fact_src": '||'.join(self.data[item]['entity'])+'||'+self.data[item]["input"],
            "item": item,
        }
#     "fact": '||'.join(self.data[item]['entity'])+'||'+self.data[item]["input"],

    def collate_fn(self, batch):
        """
        :return: batches = {
                    "src_input_ids": 1 x padded_sent_len,
                    "src_rephrase_ids": 1 x rephrase_num x sent_len
                    "label": label x rephrase_num
                }
        The rephrased sentences does not contain original sentence.
        """
        # For editing dataset, we just consider one model at once
        batches={}
        vaild = [b['rephrase'] + [b['src']] for b in batch]
        for name in ("src", "fact_src",):
            tokenizer_input = [b[name] for b in batch]
            tokenizer_output = self.tokenizer(
                tokenizer_input, return_tensors="pt",
                padding=True, max_length=self.max_length,
                truncation=True,
            )
            for k, v in tokenizer_output.items():
                if name == 'src' and  self.example_repeat > 1:
                    v_ = [v for _ in range(self.example_repeat)]
                    batches["{}_{}".format(name, k)] = torch.cat(v_, dim=0)
                else:
                    batches["{}_{}".format(name, k)] = v

        for i, j in self.tokenizer(
                vaild[0],
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
        ).items():
            batches[f"vaild_{i}"] = j

        assert len(batch) == 1
        labels = torch.tensor([b_["trg"] for b_ in batch]).float()
        batches["labels"] = torch.cat([labels for _ in range(self.example_repeat)], dim=0)
        batches["re_labels"] = torch.cat([labels for _ in range( len(batch[0]["rephrase"]))], dim=0)
        batches["va_labels"] = torch.cat([labels for _ in range(len(vaild[0]))], dim=0)

        tokenize_rephrases = self.tokenizer(
            batch[0]["rephrase"],
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )
        for k, v in tokenize_rephrases.items():
            batches['{}_{}'.format('re_src', k)] = v

        batches["raw"] = batch

        return batches


#Edit dataclass
class SeqEditDataSet(object):

    def __init__(self, task_name='fever', tokenizer=None, data_path=None,
                 train_sub_size=10000,  batch_size=128, num_workers=1,use_length=-1):

        self.tokenizer = tokenizer
        self.train_sub_size = train_sub_size

        self.batch_size = batch_size
        self.task_name = task_name
        self.num_workers = num_workers

        # train_path = os.path.join(data_path, '{}_data/{}-train.jsonl'.format(task_name,task_name))
        edit_path = os.path.join(data_path, '{}_data/{}-edit.jsonl'.format(task_name,task_name))
        val_path = os.path.join(data_path, '{}_data/{}-val.jsonl'.format(task_name,task_name))
        dev_path = os.path.join(data_path, '{}_data/{}-dev-kilt.jsonl'.format(task_name,task_name))


        # Creating datasets
        if task_name == 'fever':



            # self.edit_data = FeverEditData_Formine(tokenizer=tokenizer, data_path=edit_path,use_length=use_length)


            # self.TrainR = FeverEditData_Formine(tokenizer=tokenizer, data_path=val_path,use_length=use_length)

            # self.TestR = FeverEditData_Formine(tokenizer=tokenizer, data_path=dev_path,use_length=use_length)

            # replace the data file with yours    
            self.edit_data = FeverEditData_Formine(tokenizer=tokenizer, data_path='/media/sev/Linux/pycharm_pro/Transformer-Patcher-main/data/fever_data/.fever-edit-evi.jsonl',use_length=use_length)
    
            self.TrainR = FeverEditData_Formine(tokenizer=tokenizer, data_path='/media/sev/Linux/pycharm_pro/Transformer-Patcher-main/data/fever_data/.fever-train-evi.jsonl',use_length=use_length)

            self.TestR = FeverEditData_Formine(tokenizer=tokenizer, data_path='../data/fever_data/fever-dev-kilt-with-evi.jsonl',use_length=use_length)


        elif task_name == 'zsre':


            self.edit_data = Seq2SeqData_FoeBox(tokenizer=tokenizer, data_path=edit_path,edit=True,validation=True,use_length=use_length)


            self.TrainR = Seq2SeqData_FoeBox(tokenizer=tokenizer, data_path=val_path,edit=True,validation=True,use_length=use_length)

            self.TestR = Seq2SeqData_FoeBox(tokenizer=tokenizer, data_path=dev_path,edit=True,validation=True,use_length=10)




