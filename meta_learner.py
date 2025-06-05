import math
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
from torch.nn import functional as F
from fairseq_cli.extract_model import hydra_main
from utils import set_device
from transformers import AutoTokenizer

class MetaLearner(nn.Module):
    def __init__(self, model_name,prefix_length, seq_len,seq_len_a, new_words=False,mapper_type="ATT"):
        super(MetaLearner, self).__init__()
        self.device = set_device()
        self.prefix_length = prefix_length
        self.seq_len = seq_len
        self.seq_len_a = seq_len_a
        self.mapper_dim = 768
        
        self.mapper_type = mapper_type
        self.gpt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gpt = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        
        for param in self.gpt.parameters():
            param.requires_grad = False
        self.gpt_embedding_size = self.gpt.config.hidden_size
        
        self.feature_extract = hydra_main()
        for param in self.feature_extract.parameters():
            param.requires_grad = False

        
        self.mapper_net = AttentionMapper(dim_clip=self.mapper_dim, dim_gpt_embedding=self.gpt_embedding_size,
                                          prefix_length=self.prefix_length)


        self.text_generator = TextSampler(self.gpt, self.gpt_tokenizer, self.prefix_length)

        if new_words:
            self.gpt.resize_token_embeddings(len(self.gpt_tokenizer))
            self.reinit_word_matrix()


    def forward(self, ecg, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None, mask_a: Optional[torch.Tensor] = None, fast_weights=None,
                labels: Optional[torch.Tensor] = None, get_pred_tokens=True):

        padding_mask = torch.zeros(ecg.shape[0], 12, 5000, dtype=torch.bool)
        proj_clip = self.feature_extract(**{
                'source': ecg,
                'padding_mask': padding_mask
            })
        clip_prefix2 = proj_clip['encoder_out'].to('cuda:0')
        tokens_embed = self.get_gpt_embeddings(tokens).to('cuda:0')
        proj_clip = self.mapper_net(clip_prefix2, fast_weights).to('cuda:0')
        embedding_cat = torch.cat((proj_clip, tokens_embed), dim=1)
        
        batch_size = ecg.shape[0]
        if labels is not None:
            dummy_token = self.get_dummy_token(batch_size=batch_size, dummy_token_len=self.prefix_length)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(**{'inputs_embeds':embedding_cat})
        out_logits = out.logits[:, self.prefix_length-1:self.prefix_length-1+self.seq_len_a]
        out_logits = out_logits[mask_a[:,self.prefix_length:] == 1]
        if get_pred_tokens:
            gen_tokens = self.generate_text(prefix_embed=embedding_cat,mask_a=mask_a,mask=mask)
            return out_logits, gen_tokens

        return out_logits

    def get_dummy_token(self, batch_size: int, dummy_token_len: int) -> torch.Tensor:
        return torch.zeros(batch_size, dummy_token_len, dtype=torch.int64, device=self.device)

    def get_gpt_embeddings(self, tokens):
        return self.gpt.get_input_embeddings()(tokens) 

    def reinit_word_matrix(self):
        params = self.gpt.state_dict()
        embeddings = params['transformer.wte.weight']
        pre_expansion_embeddings = embeddings[:-3, :]
        mu = torch.mean(pre_expansion_embeddings, dim=0)
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5 * sigma)
        new_embeddings = torch.stack(tuple((dist.sample() for _ in range(3))), dim=0)
        embeddings[-3:, :] = new_embeddings
        params['transformer.wte.weight'][-3:, :] = new_embeddings
        self.gpt.load_state_dict(params)

    def generate_text(self, prefix_embed=None,mask_a=None,mask=None):
        gen_tokens_list = []
        logits_sum_list = []
        for i in range(prefix_embed.shape[0]):
            gen_tokens= self.text_generator.generate(embed=prefix_embed[i],tokens_mask=mask_a[i,self.prefix_length:],mask=mask[i, :self.prefix_length])
            gen_tokens_list.append(gen_tokens.squeeze(0))
        gen_tokens_ = torch.cat(gen_tokens_list, dim=0).view(-1)
        return gen_tokens_
    

class AttentionMapper(nn.Module):
    def __init__(self, dim_clip, dim_gpt_embedding, prefix_length):
        super(AttentionMapper, self).__init__()
        # print(88888888,dim_clip, dim_gpt_embedding, prefix_length)
        self.dim_V = dim_gpt_embedding
        self.num_heads = 8
        self.prefix_length = prefix_length
        self.config = [
            # ( name of param ), [out_size, in_size],
            ('parameter', [prefix_length, dim_clip]),
            # ('linear', [dim_clip, dim_gpt_embedding])
            ('fc_q_linear', [dim_gpt_embedding, dim_clip]),
            ('fc_k_linear', [dim_gpt_embedding, dim_clip]),
            ('fc_v_linear', [dim_gpt_embedding, dim_clip]),
            ('fc_o_linear', [dim_gpt_embedding, dim_gpt_embedding]),
            ('layer_norm_1', [dim_gpt_embedding]),
            ('layer_norm_2', [dim_gpt_embedding])
        ]

        self.vars = nn.ParameterList()
        for i, (name, param) in enumerate(self.config):
            if 'linear' in name:
                w = nn.Parameter(torch.ones(*param))
                b = nn.Parameter(torch.zeros(param[0]))
                torch.nn.init.xavier_uniform_(w)
                self.vars.append(w)
                self.vars.append(b)
            elif 'parameter' in name:  # the visual prefix
                param_learn = nn.Parameter(torch.randn(*param), requires_grad=True)
                self.vars.append(param_learn)
            elif 'layer_norm' in name:
                layer_norm_w = nn.Parameter(torch.ones(*param))
                layer_norm_b = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(layer_norm_w)
                self.vars.append(layer_norm_b)

    def forward(self, clip_x, fast_weights=None):
        clip_x = clip_x.float().unsqueeze(1)
        batch_size, clip_len = clip_x.shape[:2]
        fast_weights = list(fast_weights)
        prefix = fast_weights[0].unsqueeze(0).expand(batch_size, *fast_weights[0].shape)
        x_prefix = torch.cat((prefix, clip_x), dim=1)

        Q = F.linear(x_prefix, weight=fast_weights[1], bias=fast_weights[2])
        K = F.linear(x_prefix, weight=fast_weights[3], bias=fast_weights[4])
        V = F.linear(x_prefix, weight=fast_weights[5], bias=fast_weights[6])

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = F.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = F.dropout(O, p=0.5)
        O = F.layer_norm(O, normalized_shape=[O.shape[-1]], weight=fast_weights[9], bias=fast_weights[10]) \
            if 'layer_norm_1' in [c[0] for c in self.config] else O

        O = O + F.leaky_relu(F.linear(O, weight=fast_weights[7], bias=fast_weights[8]))
        O = F.dropout(O, p=0.5)
        O = F.layer_norm(O, normalized_shape=[O.shape[-1]], weight=fast_weights[11],  bias=fast_weights[12]) \
            if 'layer_norm_2' in [c[0] for c in self.config] else O
        O = O[:, :self.prefix_length]

        return O

class TextSampler:
    def __init__(self, gpt, gpt_tokenizer, prefix_len):
        self.gpt = gpt
        self.tokenizer = gpt_tokenizer
        self.prefix_len = prefix_len

    def generate(self,
                 tokens=None,
                 logits_sum=None,
                 tokens_mask=None,
                 mask=None,                 
                 prompt=None,
                 embed=None,
                 entry_count=1,
                 top_p=0.8,
                 temperature=1.0):
        """
        Adapted from:
        https://github.com/rmokady/CLIP_prefix_caption/blob/1ad805a844a62ab2e5480479aa021bccf0d4d12a/predict.py
        """

        entry_length = int(torch.sum(tokens_mask))
        self.gpt.eval()
        filter_value = -float("Inf")
        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed.unsqueeze(0)
                outputs = self.gpt(**{'inputs_embeds':generated})
                logits0 = outputs.logits
                for i in range(entry_length):
                    logits = logits0[:, self.prefix_len-1+i, :] / (temperature if temperature > 0 else 1.0)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p  # nucleus sampling
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits_c=logits.clone()
                    logits_c[:, indices_to_remove] = filter_value
                    next_token = torch.argmax(logits_c, -1).unsqueeze(1)
                    if tokens is None:
                        tokens = next_token
                    else:
                        tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
