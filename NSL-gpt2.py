import numpy as np
import torch
import time
import math
torch.set_printoptions(8)

# 全局KV缓存
kv_cache = []

def gelu(x):
    """GELU激活函数的近似计算实现
    计算公式: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def softmax(x):
    """Softmax函数实现"""
    return torch.nn.functional.softmax(x, dim=-1)


def layer_norm(x, g_b, eps:float = 1e-5):
    """层归一化实现
    
    参数:
        x: 输入张量
        g_b: 包含gamma和bias的字典
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
    
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return g * normalized + b


def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """线性层实现"""
    w, b = w_b['w'], w_b['b']
    return torch.matmul(x, torch.Tensor(w)) + torch.Tensor(b)


def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """前馈神经网络: linear -> gelu -> linear"""
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    x = linear(x, w_b1)
    x = gelu(x)
    x = linear(x, w_b2)
    return x


def attention(q, k, v, mask, use_cache=False, cache=None):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """注意力计算实现
    参考论文: https://arxiv.org/abs/1706.03762
    """
    d_k = k.shape[-1]
    
    if use_cache and cache is not None:
        cached_k, cached_v = cache
        k = torch.cat([cached_k, k], dim=0)
        v = torch.cat([cached_v, v], dim=0)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask
    
    weights = softmax(scores)
    output = torch.matmul(weights, v)
    
    if use_cache:
        updated_cache = (k, v)
        return output, updated_cache
    else:
        return output


def mha(x, attn, n_head, use_cache=False, layer_cache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """多头注意力实现"""
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    x = linear(x, c_attn)
    
    qkv = x.chunk(3, dim=-1)
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]
    qkv_heads = list(zip(*qkv_heads))

    if use_cache:
        head_caches = layer_cache if layer_cache else [None] * n_head
        new_head_caches = []
        out_heads = []
        
        for i, (q, k, v) in enumerate(qkv_heads):
            head_cache = head_caches[i]
            
            if head_cache is not None:
                causal_mask = None
            else:
                n_seq = q.shape[0]
                causal_mask = torch.triu(torch.full((n_seq, n_seq), float('-inf')), diagonal=1)
            
            out, new_cache = attention(q, k, v, causal_mask, use_cache=True, cache=head_cache)
            out_heads.append(out)
            new_head_caches.append(new_cache)
        
        updated_layer_cache = new_head_caches
    else:
        n_seq = x.shape[0]
        causal_mask = torch.triu(torch.full((n_seq, n_seq), float('-inf')), diagonal=1)
        out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]
        updated_layer_cache = None
    
    x = torch.cat(out_heads, dim=-1)
    x = linear(x, c_proj)
    
    if use_cache:
        return x, updated_layer_cache
    else:
        return x


def transformer_block(x, block, n_head, use_cache=False, layer_idx=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """Transformer块实现"""
    global kv_cache
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    layer_cache = kv_cache[layer_idx] if use_cache and layer_idx is not None and layer_idx < len(kv_cache) else None
    
    if use_cache:
        ln_1_output = layer_norm(x, ln_1)
        attn_output, updated_layer_cache = mha(ln_1_output, attn, n_head=n_head, use_cache=True, layer_cache=layer_cache)
        x = x + attn_output
        
        if layer_idx is not None:
            if layer_idx < len(kv_cache):
                kv_cache[layer_idx] = updated_layer_cache
            else:
                kv_cache.append(updated_layer_cache)
    else:
        x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)

    x = x + ffn(layer_norm(x, ln_2), mlp)

    return x


def gpt2(inputs, params, n_head, use_cache=False, is_prefix=False):  # [n_seq] -> [n_seq, n_vocab]
    """GPT-2模型实现"""
    global kv_cache
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    
    if is_prefix or not use_cache:
        if is_prefix:
            kv_cache.clear()
            
        x = wte[inputs] + wpe[range(len(inputs))]
        x = torch.Tensor(x)
    else:
        last_idx = len(inputs) - 1
        x = wte[inputs[-1:]] + wpe[last_idx:last_idx+1]
        x = torch.Tensor(x)
    
    for layer_idx, block in enumerate(blocks):
        x = transformer_block(x, block, n_head=n_head, use_cache=use_cache, layer_idx=layer_idx)

    x = layer_norm(x, ln_f)
    
    wte_tensor = torch.Tensor(wte)
    return torch.matmul(x, wte_tensor.T)


def generate(inputs, params, n_head, n_tokens_to_generate):
    """自回归生成文本"""
    from tqdm import tqdm
    
    inputs = inputs.copy()
    
    logits = gpt2(inputs, params, n_head=n_head, use_cache=True, is_prefix=True)
    next_id = np.argmax(logits[-1].numpy())
    inputs.append(int(next_id))
    
    for _ in tqdm(range(n_tokens_to_generate - 1), "generating"):
        logits = gpt2(inputs, params, n_head=n_head, use_cache=True, is_prefix=False)
        next_id = np.argmax(logits[-1].numpy())
        inputs.append(int(next_id))

    return inputs[len(inputs) - n_tokens_to_generate :]


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)