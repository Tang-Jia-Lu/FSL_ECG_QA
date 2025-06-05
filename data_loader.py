import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from load_class import change_ecg_to_qa, prepare_ecg_qa_data
from utils import set_device
import matplotlib.pyplot as plt
import argparse
from meta_trainer import MetaTrainer
import warnings
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)

PROJECT_ROOT = str(Path.cwd().parent.parent)  # project path
LOG_PATH = PROJECT_ROOT + "/logs/"
MODELS_PATH = PROJECT_ROOT + "/models/"

class FSL_ECG_QA_DataLoader(Dataset):
    """
    This is DataLoader for episodic training on FSL_ECG_QA dataset
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets / tasks
    sets: conains n_way * k_shot for meta-train set, n_way * k_query for meta-test set.
    """

    def __init__(self, mode, batchsz, n_way, k_shot, k_query, seq_len, seq_len_a, repeats, tokenizer,
                 prefix_length, startidx=0, all_ids=None, in_templates=None, prompt=1, paraphrased_path="", test_dataset=""):
        self.batchsz = batchsz  
        self.n_way = n_way  
        self.k_shot = k_shot  
        self.k_query = k_query  
        self.repeats = repeats
        self.setsz = self.n_way * self.k_shot if self.repeats == 0 else self.n_way * self.k_shot * (self.repeats + 1)
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.seq_len = seq_len  # sentence seq length
        self.seq_len_a = seq_len_a
        self.prefix_length = prefix_length
        self.startidx = startidx  # index label not from 0, but from startidx
        self.device = set_device()
        print('shuffle DB: %s, b:%d, %d-way, %d-shot, %d-query, %d-repeats' % (mode, batchsz, n_way, k_shot,
                                                                                          k_query, repeats))
        self.gpt_tokenizer = tokenizer
        self.mode = mode
        self.all_ids = all_ids
        self.prompt = prompt
        self.test_dataset=test_dataset
        
        # Set the base path depending on the test_dataset parameter
        if test_dataset == "ptb-xl":
            self.ecg_base_path = "/gpfs/home1/jtang1/multimodal_fsl_99/process_ptbxl2"
            # self.ecg_base_path = "/path/to/ptbxl/data"  # Update with the actual path
        elif test_dataset == "mimic":
            self.ecg_base_path = "/gpfs/home1/jtang1/multimodal_fsl_99/process_mimic"
            # self.ecg_base_path = "/path/to/mimic/data"  # Update with the actual path
        else:
            # Default path or raise an error for unknown dataset
            self.ecg_base_path = "/gpfs/home1/jtang1/multimodal_fsl_99/process_ptbxl2"
            # self.ecg_base_path = "/path/to/default/data"  # Update with the actual path
        
        json_data_ecg = change_ecg_to_qa(all_ids, in_templates, paraphrased_path, test_dataset=test_dataset)
      
        self.data = []
        self.img2caption = {}
        
        for i, (category_name, ecg_q_as) in enumerate(json_data_ecg.items()):
            self.data.append(ecg_q_as)

        self.cls_num = len(self.data)        
        self.create_batch(self.batchsz)


    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        # Creating of tasks; batchsz is the num. of iterations when sampling from the task distribution
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, replace=False)  # no duplicate
            support_x = []
            query_x = []
            for cls in selected_cls:
                selected_question = np.random.choice(len(self.data[cls]), 1)[0]
                selected_imgs_idx = np.random.choice(len(self.data[cls][selected_question]), self.k_shot + self.k_query)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls][selected_question])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls][selected_question])[indexDtest].tolist())
                if self.repeats > 0:
                    for i in range(self.repeats):
                        support_x.append(np.array(self.data[cls][selected_question])[indexDtrain].tolist())

            # shuffle the corresponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
                            
            # shuffle the corresponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def get_ptbxl_data_path(self, ecg_id):
        return os.path.join(
            f"{int(ecg_id / 1000) * 1000 :05d}",
            f"{ecg_id:05d}_hr"
        )
        
    def gen_prompt(self, q_str):    
        if self.prompt == 1:
            return "Question: " + q_str + "Answer: "
        elif self.prompt == 2:
            return q_str
        elif self.prompt == 3:
            return q_str + "the answer can be both, none or in question."
        else:
            # Default case or error handling
            return q_str
        
    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        support_x = torch.FloatTensor(self.setsz, 12, 2500)
        query_x = torch.FloatTensor(self.querysz, 12, 2500)
        
        support_y_q = []
        support_y_a = []
        support_y_q_mask = []
        support_y_a_mask = []
        query_y_q = []
        query_y_a = []
        query_y_q_mask = []
        query_y_a_mask = []
        
        flatten_support_x = [f"{self.ecg_base_path}/{self.get_ptbxl_data_path(sample['ecg_id'][0])}"
                             for sublist in self.support_x_batch[index] for sample in sublist]
        flatten_query_x = [f"{self.ecg_base_path}/{self.get_ptbxl_data_path(sample['ecg_id'][0])}"
                           for sublist in self.query_x_batch[index] for sample in sublist]
        
        for sublist in self.support_x_batch[index]:
            for sample in sublist:
                q_str = sample["question"].lower()
                for num_a, content in enumerate(sample["answer"]):
                    if num_a != 0:
                        a_str += ", " + content.lower()
                    else:
                        a_str = content.lower()
                
                q_str_tokenized = self.gpt_tokenizer(self.gen_prompt(q_str), return_tensors="pt")['input_ids']

                caption_padded_q, mask_0_q = pad_tokens(q_str_tokenized, self.seq_len, self.prefix_length,
                                                self.gpt_tokenizer.eos_token_id)
                support_y_q.append(caption_padded_q)
                support_y_q_mask.append(mask_0_q)
                
                a_str_tokenized = self.gpt_tokenizer(a_str, return_tensors="pt")['input_ids']
                caption_padded_a, mask_0_a = pad_tokens(a_str_tokenized, self.seq_len_a, self.prefix_length,
                                                self.gpt_tokenizer.eos_token_id)
                support_y_a.append(caption_padded_a)
                support_y_a_mask.append(mask_0_a)
                
        support_y_q = torch.stack(support_y_q)
        support_y_a = torch.stack(support_y_a)
        support_y_q_mask = torch.stack(support_y_q_mask)
        support_y_a_mask = torch.stack(support_y_a_mask)
        
        for sublist in self.query_x_batch[index]:
            for sample in sublist:
                q_str = sample["question"].lower()
                for num_a, content in enumerate(sample["answer"]):
                    if num_a != 0:
                        a_str += ", " + content.lower()
                    else:
                        a_str = content.lower()

                q_str_tokenized = self.gpt_tokenizer(self.gen_prompt(q_str), return_tensors="pt")['input_ids']
                caption_padded_q, mask_0_q = pad_tokens(q_str_tokenized, self.seq_len, self.prefix_length,
                                                        self.gpt_tokenizer.eos_token_id)
                query_y_q.append(caption_padded_q)
                query_y_q_mask.append(mask_0_q)

                a_str_tokenized = self.gpt_tokenizer(a_str, return_tensors="pt")['input_ids']
                caption_padded_a, mask_0_a = pad_tokens(a_str_tokenized, self.seq_len_a, self.prefix_length,
                                                        self.gpt_tokenizer.eos_token_id)
                query_y_a.append(caption_padded_a)
                query_y_a_mask.append(mask_0_a)

        query_y_q = torch.stack(query_y_q)
        query_y_q_mask = torch.stack(query_y_q_mask)
        query_y_a = torch.stack(query_y_a)
        query_y_a_mask = torch.stack(query_y_a_mask)

        # Reading of ecgs:
        for i, path in enumerate(flatten_support_x):
            ecg = loadmat(path)['feats']
            support_x[i] = torch.tensor(ecg)

        for i, path in enumerate(flatten_query_x):
            ecg = loadmat(path)['feats']
            query_x[i] = torch.tensor(ecg)

        return support_x, support_y_q, support_y_a, support_y_q_mask, support_y_a_mask, flatten_support_x, query_x, query_y_q, query_y_a, query_y_q_mask, query_y_a_mask, flatten_query_x

    def __len__(self):
        return self.batchsz


def pad_tokens(tokens, seq_len, prefix_length, eos_token_id):
    tokens = tokens.squeeze(0)
    padding = seq_len - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        tokens = tokens[:seq_len]
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = eos_token_id
    mask = mask.float()
    mask = torch.cat((torch.ones(prefix_length), mask), dim=0)  # adding prefix mask
    return tokens, mask 
    
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--experiment_id', type=int, default=666)
    argparser.add_argument('--batchsz_train', type=int, default=10000)
    argparser.add_argument('--batchsz_test', type=int, default=1000)
    argparser.add_argument('--model_name', type=str, help="path to model download from hugging face", 
                          #default="path/to/model")
                          default="/gpfs/home1/jtang1/multimodal_fsl_99/src/gamma/")
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=15)
    
    argparser.add_argument('--paraphrased_path', type=str, default='/gpfs/home1/jtang1/multimodal_fsl_99/paraphrased',
                        #default='path/to/paraphrased', 
                          help='path to ./paraphrased containing trian/val/test ECG-QA json files')
    argparser.add_argument('--question_type', type=str, help='question types, single-verify, single-choose, single-query,all', default='single-verify')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--prompt', type=int, help='1,Question: +q_str+Answer:,2,q_str,3,q_str+the answer can be both, none or in question.', default=1)
    argparser.add_argument('--dif_exp', type=int, help='0,same_exp,1,dif_exp', default=0)
    argparser.add_argument('--frozen_gpt', type=int, help='0,unfrozen_gpt,1,frozen_gpt', default=1)  
    argparser.add_argument('--frozen_features', type=int, help='0,unfrozen_features,1,frozen_features', default=1)    
    argparser.add_argument('--repeats', type=int, help='repeats for support set', default=0)
    argparser.add_argument('--seq_len', help='for padding batch', type=int, default=30) 
    argparser.add_argument('--seq_len_a', help='for padding batch', type=int, default=30)
    argparser.add_argument('--prefix_length', type=int, default=4)
    argparser.add_argument('--mapper_type', type=str, help='ATT MLP', default="MLP")
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=5e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.05)
    argparser.add_argument('--test_dataset', type=str, default="ptb-xl", choices=["ptb-xl", "mimic"], help='Dataset to use (ptb-xl or mimic)')
    args = argparser.parse_args()

    class_qa, train_temp, test_temp = prepare_ecg_qa_data(args)
    
    device = set_device()
    meta = MetaTrainer(args, args.experiment_id, is_pretrained=False).to(device)
    params = list(filter(lambda p: p.requires_grad, meta.model.parameters()))
    params_summed = sum(p.numel() for p in params)
    print("Total num of params: {} ".format(params_summed))
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_loader_train = FSL_ECG_QA_DataLoader(mode='train', n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=args.batchsz_train,
                                      seq_len=args.seq_len, seq_len_a=args.seq_len_a,repeats=args.repeats, tokenizer=gpt_tokenizer,
                                      prefix_length=args.prefix_length,all_ids=class_qa, in_templates=train_temp, prompt=args.prompt,
                                      paraphrased_path= args.paraphrased_path, test_dataset=args.test_dataset)
    data_loader_test  = FSL_ECG_QA_DataLoader(mode='test', n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=args.batchsz_train,
                                      seq_len=args.seq_len, seq_len_a=args.seq_len_a,repeats=args.repeats, tokenizer=gpt_tokenizer,
                                      prefix_length=args.prefix_length,all_ids=class_qa, in_templates=test_temp, prompt=args.prompt,
                                      paraphrased_path= args.paraphrased_path, test_dataset=args.test_dataset)
    batch = next(iter(data_loader_train))
    
    if isinstance(batch, dict):
        for key, value in batch.items():
            print(f"{key}: {value}")
    elif isinstance(batch, (list, tuple)):
        for i, item in enumerate(batch):
            print(f"Item {i}: {item}")
    else:
        print(batch)

