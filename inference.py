import argparse
import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from load_class import prepare_ecg_qa_data
from data_loader import FSL_ECG_QA_DataLoader 
from meta_trainer import MetaTrainer
from utils import *
import numpy as np

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)

device = set_device()
if torch.cuda.is_available():
    print('Inference on GPU!')
else:
    print('Inference on CPU!')

PATH = str(Path.cwd())
LOGS_PATH = PATH + "/logs/"

def main_inference():
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    experiment_id = "{}_{}-way_{}-shot{}".format(args.experiment_id, args.n_way, args.k_spt, args.model_type)
    meta_trainer = MetaTrainer(args, experiment_id, is_pretrained=False).to(device)
    
    if args.frozen_features == 1:
        for param in meta_trainer.model.feature_extract.parameters():
            param.requires_grad = False
    if args.frozen_gpt == 1:
        for param in meta_trainer.model.gpt.parameters():
            param.requires_grad = False
            
    params = list(filter(lambda p: p.requires_grad, meta_trainer.model.parameters()))
    params_summed = sum(p.numel() for p in params)
    print("Total num of params: {} ".format(params_summed))
    
    log_file_path = LOGS_PATH + "log_{}.txt".format(experiment_id)
    write_data_to_txt(log_file_path, "Experiment ID: {} Date: {}, {}-way, {}-shot (support), {}-shot (query), Test Dataset: {}\n"
                      .format(experiment_id, datetime.datetime.now(), args.n_way, args.k_spt, args.k_qry, args.test_dataset))

    class_qa, train_temp, test_temp = prepare_ecg_qa_data(args)
    
    data_loader_test = FSL_ECG_QA_DataLoader(mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                     k_query=args.k_qry, batchsz=args.batchsz_test, 
                                     seq_len=args.seq_len, seq_len_a=args.seq_len_a,
                                     repeats=args.repeats, tokenizer=gpt_tokenizer,
                                     prefix_length=args.prefix_length, all_ids=class_qa, 
                                     in_templates=test_temp, prompt=args.prompt,
                                     paraphrased_path=args.paraphrased_path, 
                                     test_dataset=args.test_dataset)
                                     
    db_test = DataLoader(data_loader_test, batch_size=args.task_num, shuffle=True, 
                        num_workers=args.num_workers, pin_memory=True)
                        
    print("Num. of tasks: {}".format(len(db_test)))
    
    accs_all_test = []
    metrics_results = []
    
    for step, batch in enumerate(db_test):
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

        accs = meta_trainer.finetunning(
            x_spt, y_spt_q, y_spt_a, y_spt_mask_q, y_spt_mask_a,
            x_qry, y_qry_q, y_qry_mask_q, y_qry_mask_a, y_qry_a, qry_img_id
        )
        
        accs_all_test.append(accs)
        
        print("------ Meta-test {}-way, {}-shot ({}-query) ------".format(args.n_way, args.k_spt, args.k_qry))
        print("Step: {} \tTest acc: {}\n".format(step, accs))
        write_data_to_txt(file_path=log_file_path, 
                          data="Step: {} \tTest acc: {}\n".format(step, accs))

    # Calculate average accuracy across all test tasks
    # Extract just the array portion from each tuple in accs_all_test
    acc_arrays = [item[0] for item in accs_all_test]
    
    # Now compute the mean of just the accuracy arrays
    accs = np.array(acc_arrays).mean(axis=0).astype(np.float16)
    
    # If you need mean metrics, calculate them separately
    avg_bertscore = np.mean([item[1]['f1_bertscore'] for item in accs_all_test])
    avg_meteor = np.mean([item[1]['meteor'] for item in accs_all_test])
    avg_rouge = np.mean([item[1]['rouge'] for item in accs_all_test])
    
    avg_bleu1 = np.mean([item[2]['BLEU-1'] for item in accs_all_test])
    avg_bleu2 = np.mean([item[2]['BLEU-2'] for item in accs_all_test])
    avg_bleu3 = np.mean([item[2]['BLEU-3'] for item in accs_all_test])
    avg_bleu4 = np.mean([item[2]['BLEU-4'] for item in accs_all_test])

    metrics_str = (
        f"FINAL METRICS:\n"
        f"  BERTScore F1: {avg_bertscore:.4f}\n"
        f"  METEOR: {avg_meteor:.4f}\n"
        f"  ROUGE: {avg_rouge:.4f}\n"
        f"  BLEU-1: {avg_bleu1:.4f}\n"
        f"  BLEU-2: {avg_bleu2:.4f}\n"
        f"  BLEU-3: {avg_bleu3:.4f}\n"
        f"  BLEU-4: {avg_bleu4:.4f}\n"
    )
    
    write_data_to_txt(file_path=log_file_path, data=metrics_str)
    print("FINAL: \tTest acc: {} \n".format(accs))
    write_data_to_txt(file_path=log_file_path, data="FINAL: \tTest acc: {} \n".format(accs))
    write_data_to_txt(log_file_path, "Experiment completed: {} Date: {}\n".format(experiment_id, datetime.datetime.now()))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--experiment_id', type=int, default=123456)
    argparser.add_argument('--batchsz_test', type=int, default=10)
    argparser.add_argument('--paraphrased_path', type=str, default='ecgqa/ptbxl/paraphrased/',
                          help='path to ./paraphrased containing train/val/test ECG-QA json files')
    argparser.add_argument('--test_dataset', type=str, default="ptb-xl", choices=["ptb-xl", "mimic"], 
                          help='Dataset to use (ptb-xl or mimic)')
    argparser.add_argument('--model_type', type=str, help='model need to test', default="")  # "acc_1" "acc2" ""
    argparser.add_argument('--model_name', type=str, 
                          default="/gpfs/home1/jtang1/multimodal_fsl_99/mimic_iv_infer/LLARVA/llama3_2_1B/",
                          help="Path to model")
    argparser.add_argument('--question_type', type=str, default='single-verify',
                          help='question types: single-verify, single-choose, single-query, all')
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--prompt', type=int, default=1,
                          help='1=Question: +q_str+Answer:, 2=q_str, 3=q_str+answer options')
    argparser.add_argument('--dif_exp', type=int, help='0=same_exp, 1=dif_exp', default=0)
    argparser.add_argument('--frozen_gpt', type=int, help='0=unfrozen_gpt, 1=frozen_gpt', default=1)
    argparser.add_argument('--frozen_features', type=int, help='0=unfrozen_features, 1=frozen_features', default=1)
    argparser.add_argument('--repeats', type=int, help='repeats for support set', default=0)
    argparser.add_argument('--seq_len', type=int, default=30)
    argparser.add_argument('--seq_len_a', type=int, default=30)
    argparser.add_argument('--prefix_length', type=int, default=4)
    argparser.add_argument('--mapper_type', type=str, default="MLP", help='Type of mapper: MLP or ATT')
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.05)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=5e-4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=15)
    argparser.add_argument('--update_step_test', type=int, help='update steps for fine-tunning', default=15)
    args = argparser.parse_args()
    
    experiment_id = "{}_{}-way_{}-shot{}".format(args.experiment_id, args.n_way, args.k_spt, args.model_type)
    log_file_path = LOGS_PATH + "log_{}.txt".format(experiment_id)
    write_data_to_txt(log_file_path, "Inference started. Experiment ID: {} Date: {}\n"
                      .format(args.experiment_id, datetime.datetime.now()))
                      
    main_inference()
