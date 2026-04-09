import torch
import json
import sys
sys.path.append('/disk3/wjr/workspace/sec_proj4/proj4_baseline/')
from model_cls.pneullm.model import ModelArgs, Transformer
from model_cls.pneullm.tokenizer import Tokenizer
from model_cls.pneullm.mm_adapter import set_Clip_Adapter, set_MMAdapter, set_rpo_MMAdapter
from pathlib import Path


def _load_and_redistribute_checkpoint(llama_model_path, model_name):
    # with open(Path(llama_model_path) / model_name / 'params.json') as f:
    with open(Path(llama_model_path) / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B':
        checkpoint = torch.load(llama_model_path + '/consolidated.00.pth', map_location="cpu")
        # checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params
    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)
    loaded = []
    for x in checkpoints:
        print('loading from', x)
        loaded.append(torch.load(x, map_location='cpu'))
    full_state_dict = {}
    split_dims = {}

    def add_weight_with_split_dim(name, dim):
        if dim < 0:  # bcast without split
            full_state_dict[name] = loaded[0][name].clone()
        else:
            full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
        for x in loaded:
            del x[name]
        split_dims[name] = dim

    add_weight_with_split_dim('tok_embeddings.weight', 1)
    add_weight_with_split_dim('norm.weight', -1)
    add_weight_with_split_dim('output.weight', 0)
    for i in range(params['n_layers']):
        print('gathering layer %d of %d' % (i, params['n_layers']))
        layer_prefix = f'layers.{i}.'
        bcast_names = [
            'attention_norm.weight',
            'ffn_norm.weight',
        ]
        column_parallel_names = [
            'attention.wq.weight',
            'attention.wk.weight',
            'attention.wv.weight',
            'feed_forward.w1.weight',
            'feed_forward.w3.weight',
        ]
        row_parallel_names = [
            'attention.wo.weight',
            'feed_forward.w2.weight',
        ]
        for key in bcast_names:
            add_weight_with_split_dim(layer_prefix + key, -1)
        for key in column_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 0)
        for key in row_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 1)

    checkpoint=full_state_dict
    return checkpoint, tokenizer, params

# write here
from model_cls.pneullm.rpo_model import Transformer as rpo_Transformer
def phellm(adapter_type='attn', adapter_dim=8, RPO_K=4, multiscale=4, adapter_scale=1., temperature=10.,
           visual_adapter_type='router', gradient_checkpointing=True, llama_model_path='/disk3/wjr/workspace/llama7B', llm_model='7B', max_seq_len=10,
           hidden_proj=128, drop_path=0.):

    llama_model_path =llama_model_path
    model_name = llm_model
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(llama_model_path, model_name)
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=32, hidden_proj=hidden_proj, drop_path=drop_path, **params
    )
    model_args.vocab_size = tokenizer.n_words
    model_args.RPO_K = RPO_K
    # load with GPU
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_args.multiscale=multiscale
    llama = rpo_Transformer(model_args)
    # delete language encoder
    del llama.backbone.transformer

    torch.set_default_tensor_type(torch.FloatTensor)
    llama.load_state_dict(checkpoint, strict=False)
    if adapter_type=='block' or adapter_type=='attn':
        # aa=0
        # set_MMAdapter(llama,adapter_type,dim=adapter_dim,s=adapter_scale,t=temperature,gradient_checkpointing=gradient_checkpointing)
        set_rpo_MMAdapter(llama, adapter_type, dim=adapter_dim, s=adapter_scale, t=temperature,
                          gradient_checkpointing=gradient_checkpointing)
        set_Clip_Adapter(llama.backbone.visual,visual_adapter_type,dim=adapter_dim,s=adapter_scale,t=temperature)

    learnable_keys = ['prompt', 'adapter']
    total_learn = 0.
    total = 0.
    trainable_names = []
    for name, param in llama.named_parameters():
        total += param.nelement()
        for key in learnable_keys:
            if key in name:
                param.requires_grad = True
                param.data = param.data.float()
                total_learn += param.nelement()
                trainable_names.append(name)
            else:
                param.requires_grad = False
    print('  + Number of trainable params: %.10fM' % (total_learn / 1e6))
    print('  + Number of params: %.10fM' % (total / 1e6))
    return llama
from model_cls.pneullm.rpo_model import Transformer_vision, Transformer_vision_simple
def phellm_vision(adapter_type='attn', adapter_dim=8, RPO_K=4, multiscale=4, adapter_scale=1., temperature=10.,
           visual_adapter_type='router', gradient_checkpointing=True, llama_model_path='/disk3/wjr/workspace/llama7B', llm_model='7B', max_seq_len=10,
           hidden_proj=128, drop_path=0.):
    llama = Transformer_vision()
    torch.set_default_tensor_type(torch.FloatTensor)
    if adapter_type=='block' or adapter_type=='attn':
        # aa=0
        # set_MMAdapter(llama,adapter_type,dim=adapter_dim,s=adapter_scale,t=temperature,gradient_checkpointing=gradient_checkpointing)
        set_Clip_Adapter(llama.backbone,visual_adapter_type,dim=adapter_dim,s=adapter_scale,t=temperature)

    learnable_keys = ['prompt', 'adapter']
    total_learn = 0.
    total = 0.
    trainable_names = []
    for name, param in llama.named_parameters():
        total += param.nelement()
        for key in learnable_keys:
            if key in name:
                param.requires_grad = True
                param.data = param.data.float()
                total_learn += param.nelement()
                trainable_names.append(name)
            else:
                param.requires_grad = False
    print('  + Number of trainable params: %.10fM' % (total_learn / 1e6))
    print('  + Number of params: %.10fM' % (total / 1e6))
    return llama

def phellm_vision_simple(adapter_type='attn', adapter_dim=8, RPO_K=4, multiscale=4, adapter_scale=1., temperature=10.,
           visual_adapter_type='router', gradient_checkpointing=True, llama_model_path='/disk3/wjr/workspace/llama7B', llm_model='7B', max_seq_len=10,
           hidden_proj=128, drop_path=0.):
    llama = Transformer_vision_simple()

    total_learn = 0.
    total = 0.
    trainable_names = []
    for name, param in llama.named_parameters():
        total += param.nelement()
        param.requires_grad = True
        # param.data = param.data.float()
        if 'adapter' in name:
            param.data = param.data.float()
        total_learn += param.nelement()
        trainable_names.append(name)
    print('  + Number of trainable params: %.10fM' % (total_learn / 1e6))
    print('  + Number of params: %.10fM' % (total / 1e6))
    return llama