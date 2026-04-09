'''Aligning Medical Images with General  Knowledge from Large Language Models,miccai2024'''
import os.path as osp
import torch
import torch.nn as nn
import json
from collections import OrderedDict
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import sys
sys.path.append('/disk3/wjr/workspace/sec_proj4/proj4_baseline/')
from model_cls.ViP.clip import clip
from model_cls.ViP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

#["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]
def load_clip_to_cpu(backbone_name="ViT-B/16"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ViP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class MergePrompt(nn.Module):
    def __init__(self, dim, dtype):
        super().__init__()
        self.dim = dim
        self.scale = dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(dim, dim, bias=False, dtype=dtype)
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.q_proj.weight)

    def forward(self, query, key=None, value=None):
        if key is None:
            key = query
        if value is None:
            value = key

        q = self.q_proj(query)
        k = self.k_proj(key)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = (query @ key.transpose(-2, -1)) * self.scale

        attn = torch.softmax(attn, dim=-1)

        # out = attn @ v
        out = attn @ value

        out = query + out
        return out


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final  # layernorm
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        out = OrderedDict()
        for i, (k, v) in enumerate(prompts.items()):
            x = v + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_prompts[k].argmax(dim=-1)]
            x = x @ self.text_projection  # text_projection: (512,1024)
            out[k] = x

        return out


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, CTP='end',  # class token position (end)
                 NCTX=4,  # number of context tokens
                 CSC=False,   # class-specific context (False or True)
                 DESCRIPTOR_PATH='./model_cls/ViP/vip_descriptors/descriptors_pneumoconiosis.json'
    ):
        super().__init__()
        n_cls = len(classnames)  # num class = 100
        n_ctx = NCTX  # num context token = 16
        # ctx_init = cfg.TRAINER.VIP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # text feature shape
        with open(DESCRIPTOR_PATH, 'r') as fp:
            gpt_descriptions = json.load(fp)

        # random initialization
        if CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # learnable token para to be optimized

        tokenized_prompts = OrderedDict()
        embedding = OrderedDict()
        name_lens = OrderedDict()
        for i, (k, v) in enumerate(gpt_descriptions.items()):
            name_lens[k] = [len(_tokenizer.encode(item)) for item in v]
            prompts = [prompt_prefix + " " + item + "." for item in v]
            tokenized_prompts[k] = torch.cat([clip.tokenize(p) for p in prompts])
            with torch.no_grad():
                embedding[k] = clip_model.token_embedding(tokenized_prompts[k]).type(dtype)

        self.token_prefix = OrderedDict()
        self.token_suffix = OrderedDict()
        for i, (k, v) in enumerate(embedding.items()):
            self.token_prefix[k] = embedding[k][:, :1, :].cuda()
            self.token_suffix[k] = embedding[k][:, 1 + n_ctx:, :].cuda()

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = CTP

    def forward(self):
        ctx = self.ctx

        prompts = OrderedDict()
        for index, (k, v) in enumerate(self.tokenized_prompts.items()):
            prefix = self.token_prefix[k]
            suffix = self.token_suffix[k]

            if ctx.dim() == 2:
                ctx_k = ctx.unsqueeze(0).expand(prefix.shape[0], -1, -1)  # all prompt has the same template
            else:
                ctx_k = ctx[index].unsqueeze(0).expand(prefix.shape[0], -1, -1)  # class-specific prompt

            if self.class_token_position == "end":
                prompts[k] = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_k,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )

            elif self.class_token_position == "middle":
                single_desc_prompts = []
                half_n_ctx = self.n_ctx // 2
                for i in range(len(self.name_lens[k])):  # for class k, there are m descriptors
                    name_len = self.name_lens[k][i]
                    prefix_i = prefix[i:i + 1, :, :]
                    descriptor_i = suffix[i:i + 1, :name_len, :]
                    suffix_wo_desc_i = suffix[i:i + 1, name_len:, :]
                    ctx_half_i_1 = ctx_k[i:i + 1, :half_n_ctx, :]
                    ctx_half_i_2 = ctx_k[i:i + 1, half_n_ctx:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            ctx_half_i_1,  # (1, n_ctx//2, dim)
                            descriptor_i,  # (1, name_len, dim)
                            ctx_half_i_2,  # (1, n_ctx//2, dim)
                            suffix_wo_desc_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                    single_desc_prompts.append(prompt)
                prompts[k] = torch.cat(single_desc_prompts, dim=0)

            elif self.class_token_position == "front":
                single_desc_prompts = []
                for i in range(len(self.name_lens[k])):
                    name_len = self.name_lens[k][i]
                    prefix_i = prefix[i: i + 1, :, :]
                    descriptor_i = suffix[i: i + 1, :name_len, :]
                    suffix_wo_desc_i = suffix[i: i + 1, name_len:, :]
                    ctx_k_i = ctx_k[i: i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            descriptor_i,  # (1, name_len, dim)
                            ctx_k_i,  # (1, n_ctx, dim)
                            suffix_wo_desc_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                    single_desc_prompts.append(prompt)
                prompts[k] = torch.cat(prompts, dim=0)

            else:
                raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, CTP='end',  # class token position (end)
                 NCTX=4,  # number of context tokens
                 CSC=False,   # class-specific context (False or True)
                 DESCRIPTOR_PATH='model_cls/ViP/vip_descriptors/descriptors_pneumoconiosis.json'):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, CTP, NCTX, CSC, DESCRIPTOR_PATH)
        self.classnames = classnames
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        token = torch.empty((len(classnames), clip_model.ln_final.weight.shape[0]), dtype=self.dtype)
        self.learnable_token = nn.Parameter(token)  # learnable token
        nn.init.normal_(self.learnable_token, std=0.02)

        self.attn = MergePrompt(dim=clip_model.ln_final.weight.shape[0], dtype=self.dtype)

    def forward(self, image):
        B, C, H, W = image.shape
        logit_scale = self.logit_scale.exp()
        logits = torch.zeros((B, len(self.classnames)), dtype=self.dtype, requires_grad=True).cuda()
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        for i, (k, v) in enumerate(text_features.items()):
            grouped_text_features = self.attn(self.learnable_token[i:i + 1, :], v)
            normalized_text_features = grouped_text_features / grouped_text_features.norm(dim=-1, keepdim=True)
            score = image_features @ normalized_text_features.t()
            logits[:, i:i + 1] += logit_scale * score

        return logits

#["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]

class ViP(nn.Module):
    def __init__(self, backbone_name="ViT-B/16", classnames=['normal', 'pneumoconiosis'],
                 VIP_PREC="fp16",
                 device = torch.device("cuda"),
                 CTP='end',  # class token position (end)
                 NCTX=4,  # number of context tokens
                 CSC=False,   # class-specific context (False or True)
                 DESCRIPTOR_PATH='/disk3/wjr/workspace/sec_proj4/proj4_baseline/model_cls/ViP/vip_descriptors/descriptors_pneumoconiosis.json'):
        super().__init__()
        self.device = device
        self.VIP_PREC=VIP_PREC
        print(f"Loading CLIP (backbone: {backbone_name})")
        clip_model = load_clip_to_cpu(backbone_name)

        if VIP_PREC == "fp32" or VIP_PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(classnames, clip_model, CTP, NCTX, CSC, DESCRIPTOR_PATH)
        self.model.to(self.device)
    def forward(self, image):
        prec = self.VIP_PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
        else:
            output = self.model(image)
        return output
import random
import argparse
import datetime
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import numpy as np
def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #将CuDNN的性能优化模式设置为关闭
    cudnn.benchmark = False
    #将CuDNN的确定性模式设置为启用,确保CuDNN在相同的输入下生成相同的输出
    cudnn.deterministic = True
    #CuDNN加速
    cudnn.enabled = True
    print(cudnn.benchmark, cudnn.deterministic, cudnn.enabled)
if __name__ == '__main__':
    setup_seed(1)
    aa = torch.ones((2, 3, 224, 224)).cuda()
    model = ViP()
    y=model(aa)
    print(y)