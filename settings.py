
import argparse
import logging
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='zsre')
parser.add_argument("--setting", type=str, default='zsre')
parser.add_argument("--data_path", type=str, default='../data')
parser.add_argument("--model_path", type=str, default='../log/models/bart_seq2seq/version_1/checkpoints/bart-seq2seq-epoch=10-valid_acc=0.4881.ckpt')
#data/fever_data/FC_model.ckpt
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument('--task_id', type=str, default=None, help='name for logging txt')
parser.add_argument('--gpu_nums', type=int, default=0)
parser.add_argument('--tasks_per_gpu', type=int, default=1)
parser.add_argument('--log_path', type=str, default='./logsV2')
parser.add_argument('--log_name', type=str, default='log.txt')

parser.add_argument('--seed', type=int, default=77)

parser.add_argument('--max_edit_step', type=int, default=1000)
parser.add_argument("--gpus", type=list, default=[0])
parser.add_argument("--device", type=int, default=0)

# early stopping hp
parser.add_argument('--early_patience', type=int, default=1)
parser.add_argument('--early_mode', type=str, default='max')
parser.add_argument('--early_thd', type=float, default=0.01)
parser.add_argument('--start_val_epoch', type=int, default=90)

# checkpoint hp
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--ckpt_monitor", default="save_ckpt", type=str)
parser.add_argument("--ckpt_metric_mode", default="max", type=str)

# optimizer hp
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=2e-2)
parser.add_argument("--weight_decay", type=float, default=0.02)
parser.add_argument('--lr_scheduler_factor', type=float, default=0.7)
parser.add_argument('--lr_scheduler_patience', type=int, default=1)

# initialization hp
parser.add_argument('--use_init_weight', type=int, default=1)
parser.add_argument('--amplify_v', type=int, default=1)
parser.add_argument('--amplify_con', type=float, default=10.0)
parser.add_argument('--freeze_a', type=bool, default=False)
parser.add_argument('--freeze_k', type=bool, default=False)
parser.add_argument('--margin_val1', type=float, default=3)
parser.add_argument('--margin_val2', type=float, default=-3)

# trainer hp
parser.add_argument('--check_val_every_n_epoch', type=int, default=30)


# activate hp
parser.add_argument('--activate_loss', type=str, default='top5_exp')  # margin|exp|non_use
parser.add_argument('--act_loss_thd', type=float, default=0.1)  # the thd for stopping the training
 # the thd for stopping the training
parser.add_argument('--alc', type=float, default=1.0)
parser.add_argument('--act_margin_val', type=float, default=0.0)

# freeze hp
parser.add_argument('--freeze_model', type=bool, default=True)
parser.add_argument('--training', type=bool, default=False)
parser.add_argument('--train_boxe', type=bool, default=False)
parser.add_argument('--emb_sim', type=bool, default=False)
parser.add_argument('--only_box_evl', type=bool, default=False)
parser.add_argument('--cls_sim', type=bool, default=True)
parser.add_argument('--test_reshape', type=bool, default=False)
parser.add_argument('--use_threshold', type=bool, default=False)
parser.add_argument('--eval_cl', type=bool, default=False)
parser.add_argument('--patch_drop', type=bool, default=False)
parser.add_argument('--re_vaild', type=bool, default=False)
parser.add_argument('--prompt_patch', type=bool, default=False)
parser.add_argument('--impls', type=bool, default=False)
parser.add_argument('--test_sem_relation', type=bool, default=False)
parser.add_argument('--eval_train_rate', type=bool, default=False)
parser.add_argument('--eval_dev_rate', type=bool, default=False)
parser.add_argument('--if_rephrase', type=bool, default=False)
parser.add_argument('--KL', type=bool, default=False)



parser.add_argument('--threhold', type=float, default=0.8)

parser.add_argument('--max_add_neuron_num', type=int, default=7)
parser.add_argument('--use_length', type=int, default=-1)
parser.add_argument('--add_neuron_num', type=int, default=1)

# use_val = 0: we edit just until the editing example is corrected
# use_val = 1: we edit and use an external validation set to decide when to early stop
parser.add_argument('--use_val', type=int, default=1)
parser.add_argument('--promt_length', type=int, default=7)
parser.add_argument('--stop_number', type=int, default=-1)

parser.add_argument('--weight_path', type=str, default='None')
parser.add_argument('--box_path', type=str, default='None')
parser.add_argument('--cl_type', type=str, default='bart')
parser.add_argument('--train_way', type=str, default='ori')
parser.add_argument('--cls_type', type=str, default='hard_thre')
parser.add_argument('--pseudo_token', type=str, default='[PROMPT]')
parser.add_argument('--editor_tp', type=str, default='patch')
args, _ = parser.parse_known_args()

args.gpus = [args.device]
# args.device = torch.device('cuda', args.device)




uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime())
file_name=f'{args.setting}'
log_path=f'{args.log_path}/{file_name}/'
if not os.path.exists(log_path):
    os.makedirs(log_path)

LOG = logging.getLogger(__name__)
LOG = logging.getLogger(__name__)
LOG.setLevel(level=logging.INFO)
fh = logging.FileHandler(log_path+'log.txt')
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

LOG.addHandler(fh)
LOG.addHandler(ch)
LOG.info(f'LOG path is {log_path}')
LOG.info(f'setting is {args}')
