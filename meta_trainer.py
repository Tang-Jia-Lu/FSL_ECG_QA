import os
from copy import deepcopy
from compute_scores import compute_bmr_metrics, compute_bleu_metrics
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from meta_learner import MetaLearner
from utils import *

PATH = str(Path.cwd())
MODELS_PATH = PATH + "/models/"

class MetaTrainer(nn.Module):
    def __init__(self, args, experiment_id, is_pretrained, new_words=False):
        super(MetaTrainer, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.prefix_length = args.prefix_length
        self.seq_len = args.seq_len
        self.seq_len_a = args.seq_len_a
        self.device = 'cuda:0'
        self.experiment_id=experiment_id
        self.model_weight = "{}.pt".format(experiment_id)
        self.log_file_path = PATH + "/logs/log_{}.txt".format(experiment_id)
        self.new_words=new_words
        self.model_name=args.model_name
        self.mapper_type=args.mapper_type
        self.model = MetaLearner(model_name=self.model_name,prefix_length=self.prefix_length, seq_len=self.seq_len, seq_len_a=self.seq_len_a,
                                 new_words=self.new_words, mapper_type=self.mapper_type)

        # Loading pre-trained model
        if is_pretrained:
            model_dict = torch.load(MODELS_PATH + self.model_weight, map_location=torch.device(self.device))
            self.model.mapper_net.load_state_dict(model_dict['mapper_net'])
            
        self.model.to(self.device)
        self.meta_optim = optim.AdamW(self.model.parameters(), lr=self.meta_lr)
        self.pad_token_id = self.model.gpt_tokenizer.eos_token_id

    def forward(self, x_spt, y_spt_q, y_spt_a, y_spt_mask_q, y_spt_mask_a, x_qry, y_qry_q, y_qry_a, y_qry_mask_q, y_qry_mask_a):

        task_num = x_spt.shape[0]
        a_seq_len = int(torch.sum(y_qry_mask_a[:, :, self.prefix_length:]))

        losses_q = torch.zeros((self.update_step + 1)).to(self.device)
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            logits = self.model(x_spt[i], y_spt_q[i], y_spt_mask_q[i], y_spt_mask_a[i], list(self.model.mapper_net.parameters()), get_pred_tokens=False)
            y_spt_a_mask = y_spt_a[i][y_spt_mask_a[i][:, self.prefix_length:] == 1]
            loss = F.cross_entropy(logits.reshape(-1, self.model.gpt.vocab_size), y_spt_a_mask.flatten(), ignore_index=self.pad_token_id)
            grad = torch.autograd.grad(loss, self.model.mapper_net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.model.mapper_net.parameters())))
            question = y_qry_q[i]
            answer_mask = y_qry_a[i][y_qry_mask_a[i][:, self.prefix_length:] == 1]

            with torch.no_grad():
                logits_q, pred_tokens = self.model(x_qry[i], question, y_qry_mask_q[i], y_qry_mask_a[i], list(self.model.mapper_net.parameters()))
                loss_q = F.cross_entropy(logits_q.reshape(-1, self.model.gpt.vocab_size), answer_mask.flatten(), ignore_index=self.pad_token_id)
                losses_q[0] += loss_q
                correct = torch.eq(pred_tokens, answer_mask).sum().item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                logits_q, pred_tokens = self.model(x_qry[i], question, y_qry_mask_q[i], y_qry_mask_a[i], fast_weights)
                loss_q = F.cross_entropy(logits_q.reshape(-1, self.model.gpt.vocab_size), answer_mask.flatten(), ignore_index=self.pad_token_id)
                losses_q[1] += loss_q
                correct = torch.eq(pred_tokens, answer_mask).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                logits = self.model(x_spt[i], y_spt_q[i], y_spt_mask_q[i], y_spt_mask_a[i], fast_weights, get_pred_tokens=False)
                y_spt_a_mask = y_spt_a[i][y_spt_mask_a[i][:, self.prefix_length:] == 1]
                loss = F.cross_entropy(logits.reshape(-1, self.model.gpt.vocab_size), y_spt_a_mask.flatten(), ignore_index=self.pad_token_id)
                grad = torch.autograd.grad(outputs=loss, inputs=fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q, pred_tokens = self.model(x_qry[i], question, y_qry_mask_q[i], y_qry_mask_a[i], fast_weights)
                loss_q = F.cross_entropy(logits_q.reshape(-1, self.model.gpt.vocab_size), answer_mask.flatten(), ignore_index=self.pad_token_id)
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    correct = torch.eq(pred_tokens, answer_mask).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        loss_q = torch.mean(losses_q[2:]) / task_num

        self.meta_optim.zero_grad()
        loss_q.backward(inputs=list(self.model.mapper_net.parameters()))
        nn.utils.clip_grad_norm_(self.model.mapper_net.parameters(), max_norm=1)
        self.meta_optim.step()
        accs = np.array(corrects) / (a_seq_len)
        losses_q_ = [round(loss.item(), 4) for loss in losses_q]

        return accs, losses_q_

    def finetunning(self, x_spt, y_spt_q, y_spt_a, y_spt_mask_q, y_spt_mask_a, x_qry, y_qry, y_qry_mask_q, y_qry_mask_a, qry_answer, calc_metrics=False):
        bmr_results_list = []
        bleu_results_list = []
        
        a_seq_len = int(torch.sum(y_qry_mask_a[:, :, self.prefix_length:]))
        corrects = [0 for _ in range(self.update_step_test + 1)]

        model_state_dict = deepcopy(self.model.state_dict())
        model = MetaLearner(model_name=self.model_name, prefix_length=self.prefix_length, seq_len=self.seq_len, seq_len_a=self.seq_len_a,
                            new_words=self.new_words)
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        task_num = x_spt.shape[0]
        for i in range(task_num):
            logits = model(x_spt[i], y_spt_q[i], y_spt_mask_q[i], y_spt_mask_a[i], fast_weights=list(model.mapper_net.parameters()), get_pred_tokens=False)
            y_spt_a_mask = y_spt_a[i][y_spt_mask_a[i][:, self.prefix_length:] == 1]
            loss = F.cross_entropy(logits.reshape(-1, self.model.gpt.vocab_size), y_spt_a_mask.flatten(),
                                  ignore_index=self.pad_token_id)
            grad = torch.autograd.grad(outputs=loss, inputs=model.mapper_net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, model.mapper_net.parameters())))

            with torch.no_grad():
                logits_q, pred_tokens = model(x_qry[i], y_qry[i], y_qry_mask_q[i], y_qry_mask_a[i], fast_weights=list(model.mapper_net.parameters()))
                qry_answer_mask = qry_answer[i][y_qry_mask_a[i][:, self.prefix_length:] == 1]
                correct = torch.eq(pred_tokens, qry_answer_mask).sum().item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                logits_q, pred_tokens = model(x_qry[i], y_qry[i], y_qry_mask_q[i], y_qry_mask_a[i], fast_weights=fast_weights)
                correct = torch.eq(pred_tokens, qry_answer_mask).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step_test):
                y_spt_a_mask = y_spt_a[i][y_spt_mask_a[i][:, self.prefix_length:] == 1]
                logits = model(x_spt[i], y_spt_q[i], y_spt_mask_q[i], y_spt_mask_a[i], fast_weights=fast_weights, get_pred_tokens=False)
                loss = F.cross_entropy(logits.reshape(-1, self.model.gpt.vocab_size), y_spt_a_mask.flatten(),
                                      ignore_index=self.pad_token_id)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q, pred_tokens = model(x_qry[i], y_qry[i], y_qry_mask_q[i], y_qry_mask_a[i], fast_weights=fast_weights, get_pred_tokens=True)
                with torch.no_grad():
                    correct = torch.eq(pred_tokens, qry_answer_mask).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

            if calc_metrics:
                all_length = 0
                gt_answer_list = []
                pred_answer_list = []
                for idx in range(x_qry[i].shape[0]):
                    gt_answer = self.model.gpt_tokenizer.decode(qry_answer[i][idx], skip_special_tokens=True).strip()
                    answer_length = int(torch.sum(y_qry_mask_a[i][idx][self.prefix_length:]))
                    pred_answer = self.model.gpt_tokenizer.decode(pred_tokens[all_length:all_length + answer_length], skip_special_tokens=True).strip()
                    gt_answer_list.append(gt_answer)
                    pred_answer_list.append(pred_answer)
                    all_length += answer_length
                    write_data_to_txt(self.log_file_path, ("GT answer: {}, Pred. answer: {}\n".format(gt_answer, pred_answer)))

                bmr_results = compute_bmr_metrics(gt_answer_list, pred_answer_list)
                bmr_results_list.append(bmr_results)

                bleu_results = compute_bleu_metrics(gt_answer_list, pred_answer_list)
                bleu_results_list.append(bleu_results)
            
        def compute_average_metrics(results_list):
            average_dict = {}
            num_entries = len(results_list)
            for result in results_list:
                for key, value in result.items():
                    if key not in average_dict:
                        average_dict[key] = 0
                    average_dict[key] += value
            for key in average_dict:
                average_dict[key] = average_dict[key] / num_entries
            return average_dict

        if calc_metrics and bmr_results_list and bleu_results_list:
            average_results_bmr = compute_average_metrics(bmr_results_list)
            average_results_blue = compute_average_metrics(bleu_results_list)
            print("Average BMR Results:", average_results_bmr, average_results_blue)
        else:
            average_results_bmr = {}
            average_results_blue = {}

        del model
        accs = np.array(corrects) / a_seq_len

        if calc_metrics:
            return accs, average_results_bmr, average_results_blue
        else:
            return accs

    def save_mapper_model(self,para=None):
        model_dict={'mapper_net': self.model.mapper_net.state_dict()}
        torch.save(model_dict, os.path.join(MODELS_PATH, f"{self.experiment_id}{para}.pt"))                        
        print("Model saved on path {}".format(MODELS_PATH))


def write_data_to_txt(file_path, data):
    if path.exists(file_path):
        with open(file_path, 'a', newline='') as file:
            file.write(data)
    else:  # Create the file
        with open(file_path, 'w') as file:
            file.write(data)
