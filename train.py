import argparse
import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from load_class import prepare_ecg_qa_data
from data_loader import FSL_ECG_QA_DataLoader 
from meta_trainer import MetaTrainer
from utils import *
import numpy as np
import os

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)

device = set_device()
if torch.cuda.is_available():
    print('Training on GPU!')
else:
    print('Training on CPU!')

PATH = str(Path.cwd())
LOGS_PATH = PATH + "/logs/"

def write_data_to_txt(file_path, data):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.exists(file_path):
        with open(file_path, 'a', newline='') as file:
            file.write(data)
    else:  # Create the file
        with open(file_path, 'w') as file:
            file.write(data)

def main_episodic():
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    experiment_id = "{}_{}-way_{}-shot{}".format(args.experiment_id, args.n_way, args.k_spt,args.model_type)
    meta = MetaTrainer(args, experiment_id, is_pretrained=False).to(device)
    if args.frozen_features==1:
        for param in meta.model.feature_extract.parameters():
            param.requires_grad = False
    if args.frozen_gpt==1:
        for param in meta.model.gpt.parameters():
            param.requires_grad = False
    params = list(filter(lambda p: p.requires_grad, meta.model.parameters()))
    params_summed = sum(p.numel() for p in params)
    print("Total num of params: {} ".format(params_summed))
    log_file_path = LOGS_PATH + "log_{}.txt".format(experiment_id)
    write_data_to_txt(log_file_path, "Experiment ID: {} Date: {}, {}-way, {}-shot (support), {}-shot (query), Mapper: {}".format(experiment_id, datetime.datetime.now(),args.n_way, args.k_spt, args.k_qry,args.mapper_type))

    class_qa, train_temp, test_temp = prepare_ecg_qa_data(args)  

    data_loader_train = FSL_ECG_QA_DataLoader(mode='train', n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=args.batchsz_train,
                                      seq_len=args.seq_len, seq_len_a=args.seq_len_a,repeats=args.repeats, tokenizer=gpt_tokenizer,
                                      prefix_length=args.prefix_length,all_ids=class_qa, in_templates=train_temp, prompt=args.prompt,
                                      paraphrased_path= args.paraphrased_path, test_dataset=args.test_dataset)
    data_loader_test  = FSL_ECG_QA_DataLoader(mode='test', n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=args.batchsz_test,
                                      seq_len=args.seq_len, seq_len_a=args.seq_len_a,repeats=args.repeats, tokenizer=gpt_tokenizer,
                                      prefix_length=args.prefix_length,all_ids=class_qa, in_templates=test_temp, prompt=args.prompt,
                                      paraphrased_path= args.paraphrased_path, test_dataset=args.test_dataset)
    db_train = DataLoader(data_loader_train, batch_size=args.task_num, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                    drop_last=True)
    db_test = DataLoader(data_loader_test, batch_size=args.task_num, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)
    
    for epoch in range(args.epoch):
        for step, batch in enumerate(db_train):
            (
                x_spt, y_spt_q, y_spt_a, y_spt_mask_q, y_spt_mask_a, id_spt,
                x_qry, y_qry_q, y_qry_a, y_qry_mask_q, y_qry_mask_a, qry_img_id
            ) = batch

            x_spt        = x_spt.to(device)
            y_spt_q      = y_spt_q.to(device)
            y_spt_a      = y_spt_a.to(device)
            y_spt_mask_q = y_spt_mask_q.to(device)
            y_spt_mask_a = y_spt_mask_a.to(device)
            # id_spt stays on CPU
            x_qry        = x_qry.to(device)
            y_qry_q      = y_qry_q.to(device)
            y_qry_a      = y_qry_a.to(device)
            y_qry_mask_q = y_qry_mask_q.to(device)
            y_qry_mask_a = y_qry_mask_a.to(device)
            # qry_img_id stays on CPU

            accs, losses = meta(
                x_spt, y_spt_q, y_spt_a, y_spt_mask_q, y_spt_mask_a,
                x_qry, y_qry_q, y_qry_a, y_qry_mask_q, y_qry_mask_a
            )

            if step % 100 == 0:
                print(
                    f"------ Meta-training {args.n_way}-way, {args.k_spt}-shot ({args.k_qry}-query) "
                    f"{args.mapper_type}-mapper {args.prefix_length}-prefix tokens------"
                )
                write_data_to_txt(
                    file_path=LOGS_PATH + f"log_{experiment_id}.txt",
                    data=f"Step: {step} \tTraining acc: {accs} \n"
                )
                meta.save_mapper_model(para="")

            # Evaluation every 10 steps
            if step % 400 == 0 and step != 0:
                test_accs_all = []
                for test_step, test_batch in enumerate(db_test):
                    (
                        spt_x, spt_y_q, spt_y_a, spt_mask_q, spt_mask_a, spt_ids,
                        qry_x, qry_y_q, qry_y_a, qry_mask_q, qry_mask_a, qry_img_ids
                    ) = test_batch

                    spt_x        = spt_x.to(device)
                    spt_y_q      = spt_y_q.to(device)
                    spt_y_a      = spt_y_a.to(device)
                    spt_mask_q   = spt_mask_q.to(device)
                    spt_mask_a   = spt_mask_a.to(device)
                    # spt_ids stays on CPU
                    qry_x        = qry_x.to(device)
                    qry_y_q      = qry_y_q.to(device)
                    qry_y_a      = qry_y_a.to(device)
                    qry_mask_q   = qry_mask_q.to(device)
                    qry_mask_a   = qry_mask_a.to(device)
                    # qry_img_ids stays on CPU

                    test_acc = meta.finetunning(
                        spt_x, spt_y_q, spt_y_a, spt_mask_q, spt_mask_a,
                        qry_x, qry_y_q, qry_mask_q, qry_mask_a, qry_y_a
                    )
                    test_accs_all.append(test_acc)
                test_accs_all = np.stack(test_accs_all, axis=0).mean(axis=0).astype(np.float16)
                print("------ Meta-test {}-way, {}-shot ({}-query) ------".format(args.n_way, args.k_spt, args.k_qry))
                print("Step: {} \tTest acc: {} \n".format(step, test_accs_all))
                write_data_to_txt(file_path=log_file_path, data="Step: {} \tTest acc: {} \n".format(step, test_accs_all))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--experiment_id', type=int, default=123456)
    argparser.add_argument('--batchsz_train', type=int, default=10000)
    argparser.add_argument('--batchsz_test', type=int, default=1000)
    argparser.add_argument('--paraphrased_path', type=str, default='/ecgqa/ptbxl/paraphrased/',
                          help='path to ./paraphrased containing trian/val/test ECG-QA json files')
    argparser.add_argument('--test_dataset', type=str, default="ptb-xl", choices=["ptb-xl", "mimic"], help='Dataset to use (ptb-xl or mimic)')
    argparser.add_argument('--model_name', type=str, help="path to llm model",
                           default="/llm_checkpoint/llama3.1-8b") 
                        #    default="/gpfs/home1/jtang1/multimodal_fsl_99/mimic_iv_infer/LLARVA/llama3_2_1B/")
    argparser.add_argument('--model_type', type=str, help='model need to test', default="") # "acc_1" "acc2" "
    argparser.add_argument('--data_root', type=str, default='ecg_qa')
    argparser.add_argument('--data_path', type=str, help='the path to datasets', default='/home/jtang/data/ecg_qa_500')
    argparser.add_argument('--question_type', type=str, help='question types, single-verify, single-choose, single-query, all', default='single-verify')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--prompt', type=int, help='1,Question: +q_str+Answer:,2,q_str,3,q_str+the answer can be both, none or in question.', default=1)
    argparser.add_argument('--dif_exp', type=int, help='0,same_exp,1,dif_exp', default=0)
    argparser.add_argument('--frozen_gpt', type=int, help='0,unfrozen_gpt,1,frozen_gpt', default=1)  
    argparser.add_argument('--frozen_features', type=int, help='0,unfrozen_features,1,frozen_features', default=1)    
    argparser.add_argument('--repeats', type=int, help='repeats for support set', default=0)
    argparser.add_argument('--seq_len', type=int, default=30)  
    argparser.add_argument('--seq_len_a', type=int, default=30)  
    argparser.add_argument('--prefix_length', type=int, default=4)
    argparser.add_argument('--mapper_type', type=str, default="MLP", help='Type of mapper to use: MLP or ATT')
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=5e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.05)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=15)
    argparser.add_argument('--update_step_test', type=int, help='update steps for fine-tunning', default=15)
    argparser.add_argument('--num_workers', type=int, default=8)

    args = argparser.parse_args()

    main_episodic()

