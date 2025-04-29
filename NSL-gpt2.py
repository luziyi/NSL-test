import numpy as np
import torch
import time
import math
torch.set_printoptions(8)

# 全局KV缓存，存储每一层的key和value缓存
kv_cache = []

def gelu(x):
    r"""
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    return torch.nn.functional.softmax(x, dim=-1)


def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
    
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return g * normalized + b


def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = w_b['w'], w_b['b']
    return torch.matmul(x, torch.Tensor(w)) + torch.Tensor(b)


def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    x = linear(x, w_b1)  # 第一个线性变换
    x = gelu(x)         # GELU激活函数
    x = linear(x, w_b2)  # 第二个线性变换
    return x


def attention(q, k, v, mask, use_cache=False, cache=None):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            use_cache: 是否使用KV缓存
            cache: 当前的KV缓存
        Output: Tensor, (updated_cache) 当use_cache=True时
    """
    d_k = k.shape[-1]
    
    # 如果启用缓存且有缓存数据，则合并当前K和V与缓存的K和V
    if use_cache and cache is not None:
        cached_k, cached_v = cache
        k = torch.cat([cached_k, k], dim=0)
        v = torch.cat([cached_v, v], dim=0)
    
    # 计算注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [n_q, n_k]
    
    # 应用mask
    if mask is not None:
        scores = scores + mask
    
    # 应用softmax获得注意力权重
    weights = softmax(scores)  # [n_q, n_k]
    
    # 加权求和得到输出
    output = torch.matmul(weights, v)  # [n_q, d_v]
    
    # 更新缓存
    if use_cache:
        updated_cache = (k, v)
        return output, updated_cache
    else:
        return output


def mha(x, attn, n_head, use_cache=False, layer_cache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
            use_cache: 是否使用KV缓存
            layer_cache: 该层的KV缓存
        Output: Tensor 或 (Tensor, updated_layer_cache) 当use_cache=True时
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    # Split into qkv
    qkv = x.chunk(3, dim=-1)  # [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # 转为 [n_head, 3, n_seq, n_embd/n_head]

    # 根据是否使用缓存处理不同的情况
    if use_cache:
        # 对于自回归生成，如果使用缓存，我们只关心最后一个位置
        head_caches = layer_cache if layer_cache else [None] * n_head
        new_head_caches = []
        out_heads = []
        
        for i, (q, k, v) in enumerate(qkv_heads):
            head_cache = head_caches[i]
            
            # 处理上下文关系的mask
            if head_cache is not None:
                n_prev = head_cache[0].shape[0]  # 缓存中的token数量
                # 只为最后一个token创建mask，允许它关注所有之前的token
                causal_mask = None  # 当使用缓存时，我们不需要mask，因为只计算最后一个token的输出
            else:
                n_seq = q.shape[0]
                causal_mask = torch.triu(torch.full((n_seq, n_seq), float('-inf')), diagonal=1)
            
            # 执行attention操作，并获取更新后的缓存
            out, new_cache = attention(q, k, v, causal_mask, use_cache=True, cache=head_cache)
            out_heads.append(out)
            new_head_caches.append(new_cache)
        
        # 更新层缓存
        updated_layer_cache = new_head_caches
    else:
        # 标准因果mask
        n_seq = x.shape[0]
        causal_mask = torch.triu(torch.full((n_seq, n_seq), float('-inf')), diagonal=1)
        
        # 执行标准注意力计算
        out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]
        updated_layer_cache = None
    
    # Merge heads
    x = torch.cat(out_heads, dim=-1)  # [n_seq, n_embd] 或 [1, n_embd]
    
    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    if use_cache:
        return x, updated_layer_cache
    else:
        return x


def transformer_block(x, block, n_head, use_cache=False, layer_idx=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
    Transformer块处理函数
    
    Input:
        x: 输入张量
        block: 块参数
        n_head: 注意力头数
        use_cache: 是否使用KV缓存
        layer_idx: 层索引，用于获取对应层的缓存
    Output:
        x: 输出张量
    """
    global kv_cache
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # 获取当前层的缓存
    layer_cache = kv_cache[layer_idx] if use_cache and layer_idx is not None and layer_idx < len(kv_cache) else None
    
    # multi-head causal self attention
    if use_cache:
        # 应用layernorm
        ln_1_output = layer_norm(x, ln_1)
        
        # 多头注意力计算，获取输出和更新的缓存
        attn_output, updated_layer_cache = mha(ln_1_output, attn, n_head=n_head, use_cache=True, layer_cache=layer_cache)
        
        # 残差连接
        x = x + attn_output
        
        # 更新缓存
        if layer_idx is not None:
            if layer_idx < len(kv_cache):
                kv_cache[layer_idx] = updated_layer_cache
            else:
                kv_cache.append(updated_layer_cache)
    else:
        x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head, use_cache=False, is_prefix=False):  # [n_seq] -> [n_seq, n_vocab]
    """
    GPT-2模型的主函数
    
    Input:
        inputs: 输入token ids
        params: 模型参数
        n_head: 注意力头数
        use_cache: 是否使用KV缓存
        is_prefix: 是否为前缀（首次处理整个序列）
    Output:
        logits: 模型输出的logits
    """
    global kv_cache
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    
    # 如果是前缀模式或不使用缓存，则处理整个序列并初始化KV缓存
    if is_prefix or not use_cache:
        # 清空KV缓存
        if is_prefix:
            kv_cache.clear()
            
        # token + positional embeddings
        x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
        x = torch.Tensor(x)
    else:
        # 如果使用缓存，只处理最后一个token
        last_idx = len(inputs) - 1
        # 只获取最后一个token的embedding和位置编码
        x = wte[inputs[-1:]] + wpe[last_idx:last_idx+1]  # [1] -> [1, n_embd]
        x = torch.Tensor(x)
    
    # forward pass through n_layer transformer blocks
    for layer_idx, block in enumerate(blocks):
        x = transformer_block(x, block, n_head=n_head, use_cache=use_cache, layer_idx=layer_idx)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    # 避免NumPy弃用警告，使用torch.matmul或显式转换
    wte_tensor = torch.Tensor(wte)
    return torch.matmul(x, wte_tensor.T)  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    """
    自回归生成函数
    
    Input:
        inputs: 输入token ids
        params: 模型参数
        n_head: 注意力头数
        n_tokens_to_generate: 需要生成的token数量
    Output:
        生成的token ids
    """
    from tqdm import tqdm
    
    # 复制输入，避免修改原始输入
    inputs = inputs.copy()
    
    # 首先处理整个前缀序列，初始化KV Cache
    logits = gpt2(inputs, params, n_head=n_head, use_cache=True, is_prefix=True)
    next_id = np.argmax(logits[-1].numpy())
    inputs.append(int(next_id))
    
    # 然后逐个生成剩余的tokens，利用KV Cache
    for _ in tqdm(range(n_tokens_to_generate - 1), "generating"):  # auto-regressive decode loop
        # 现在只需要处理最新添加的token
        logits = gpt2(inputs, params, n_head=n_head, use_cache=True, is_prefix=False)
        next_id = np.argmax(logits[-1].numpy())  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)