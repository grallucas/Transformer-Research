import numpy as np

########################
### MODEL OPERATIONS ###
########################

def matmul(x, M):
    return x @ M

def matmul_backward(x, M, grad_output):
    grad_x = grad_output @ M.T

    x_2d = x.reshape(-1, x.shape[-1])
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    grad_M = x_2d.T @ grad_output_2d

    return grad_x, grad_M

def add(x, y):
    return x + y

def add_backward(x, y, grad_output):
    # NOTE: this may not work correctly for broadcasting cases where x and y have different shapes
    grad_x = grad_output
    grad_y = grad_output
    return grad_x, grad_y

def silu(x):
    # NOTES: 1) could use logaddexp for numerical stability. 2) could cache the sigmoid itself `s` and pass it thru instead of recomputing it in the backward pass.
    return x * (1 / (1 + np.exp(-x)))

def silu_backward(x, grad_output):
    sigmoid = 1 / (1 + np.exp(-x))
    grad_x = grad_output * (sigmoid + x * sigmoid * (1 - sigmoid))
    return grad_x

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_backward(x, grad_output):
    s = softmax(x)
    grad_x = grad_output - np.sum(grad_output * s, axis=-1, keepdims=True)
    grad_x *= s
    return grad_x

def softmax_cross_entropy(logits, labels):
    # computes softmax and cross-entropy loss for the given logits and labels, instead of separate softmax and cross-entropy steps. This makes the gradients easier and more numerically stable
    logits_2d = logits.reshape(-1, logits.shape[-1])
    labels_1d = labels.reshape(-1)

    shifted_logits = logits_2d - np.max(
        logits_2d, axis=-1, keepdims=True
    )

    log_probs = shifted_logits - np.log(
        np.sum(np.exp(shifted_logits), axis=-1, keepdims=True)
    )

    token_losses = -log_probs[np.arange(labels_1d.size), labels_1d]
    return np.mean(token_losses)

def softmax_cross_entropy_backward(logits, labels, grad_output=1.0):
    original_shape = logits.shape
    logits_2d = logits.reshape(-1, logits.shape[-1])
    labels_1d = labels.reshape(-1)

    grad_logits = softmax(logits_2d)
    grad_logits[np.arange(labels_1d.size), labels_1d] -= 1
    grad_logits /= labels_1d.size

    return grad_output * grad_logits.reshape(original_shape)

def rmsnorm(x, weight, eps=1e-8):
    mean_square = np.mean(x**2, axis=-1, keepdims=True)
    inv_rms = 1 / np.sqrt(mean_square + eps)
    x_normalized = x * inv_rms

    return x_normalized * weight, (inv_rms, x_normalized)

def rmsnorm_backward(x, weight, grad_output, cache):
    inv_rms, x_normalized = cache
    grad_normalized = grad_output * weight

    mean_dot = np.mean(
        x * grad_normalized,
        axis=-1,
        keepdims=True,
    )

    grad_x = (
        grad_normalized * inv_rms
        - x * mean_dot * inv_rms**3
    )

    grad_weight = np.sum(
        grad_output * x_normalized,
        axis=tuple(range(x.ndim - 1)),
    )

    return grad_x, grad_weight

def embedding(x, weight):
    return weight[x]

def embedding_backward(x, weight, grad_output):
    grad_weight = np.zeros_like(weight)
    np.add.at(grad_weight, x, grad_output)
    return grad_weight

def scaled_dot_product_attention(q, k, v):
    d_k = q.shape[-1]
    scale = 1 / np.sqrt(d_k)

    scores = (q @ np.swapaxes(k, -1, -2)) * scale

    sequence_length = q.shape[-2]
    causal_mask = np.triu(
        np.ones((sequence_length, sequence_length), dtype=bool),
        k=1,
    )
    scores = np.where(causal_mask, -np.inf, scores)

    attn = softmax(scores)
    output = attn @ v

    return output, attn

def scaled_dot_product_attention_backward(q, k, v, grad_output, attn):
    scale = 1 / np.sqrt(q.shape[-1])

    # output = attn @ v
    grad_attn = grad_output @ np.swapaxes(v, -1, -2)
    grad_v = np.swapaxes(attn, -1, -2) @ grad_output

    # attn = softmax(scores)
    grad_scores = (
        grad_attn - np.sum(grad_attn * attn, axis=-1, keepdims=True)
    ) * attn

    # scores = q @ k.T * scale
    grad_q = (grad_scores @ k) * scale
    grad_k = (np.swapaxes(grad_scores, -1, -2) @ q) * scale

    return grad_q, grad_k, grad_v

def self_attention(x, attention_heads, W_o):
    W_qs = np.stack([head["W_q"] for head in attention_heads], axis=0)
    W_ks = np.stack([head["W_k"] for head in attention_heads], axis=0)
    W_vs = np.stack([head["W_v"] for head in attention_heads], axis=0)

    x_expanded = np.expand_dims(x, axis=1)
    q = matmul(x_expanded, W_qs)
    k = matmul(x_expanded, W_ks)
    v = matmul(x_expanded, W_vs)

    head_outputs, attn = scaled_dot_product_attention(q, k, v)
    head_outputs = np.swapaxes(head_outputs, 1, 2)
    concatenated = head_outputs.reshape(x.shape[0], x.shape[1], -1)
    output = matmul(concatenated, W_o)

    cache = {
        "q": q,
        "k": k,
        "v": v,
        "attn": attn,
        "concatenated": concatenated,
    }
    return output, cache

def self_attention_backward(
    x, attention_heads, W_o, grad_output, cache
):
    grad_concatenated, grad_W_o = matmul_backward(
        cache["concatenated"], W_o, grad_output
    )

    n_heads = len(attention_heads)
    d_head = cache["q"].shape[-1]
    grad_head_outputs = grad_concatenated.reshape(
        x.shape[0], x.shape[1], n_heads, d_head
    )
    grad_head_outputs = np.swapaxes(grad_head_outputs, 1, 2)

    grad_q, grad_k, grad_v = scaled_dot_product_attention_backward(
        cache["q"], cache["k"], cache["v"], grad_head_outputs, cache["attn"]
    )

    # Broadcast x across heads, then sum over the batch for weight gradients.
    x_transposed = np.swapaxes(x, -1, -2)[:, np.newaxis]
    grad_W_qs = np.sum(x_transposed @ grad_q, axis=0)
    grad_W_ks = np.sum(x_transposed @ grad_k, axis=0)
    grad_W_vs = np.sum(x_transposed @ grad_v, axis=0)

    W_qs = np.stack([head["W_q"] for head in attention_heads], axis=0)
    W_ks = np.stack([head["W_k"] for head in attention_heads], axis=0)
    W_vs = np.stack([head["W_v"] for head in attention_heads], axis=0)

    # Every head reads the same input, so its contributions add together.
    grad_x = np.sum(grad_q @ np.swapaxes(W_qs, -1, -2), axis=1)
    grad_x += np.sum(grad_k @ np.swapaxes(W_ks, -1, -2), axis=1)
    grad_x += np.sum(grad_v @ np.swapaxes(W_vs, -1, -2), axis=1)

    grad_attention_heads = [
        {"W_q": grad_W_qs[i], "W_k": grad_W_ks[i], "W_v": grad_W_vs[i]}
        for i in range(n_heads)
    ]
    return grad_x, grad_attention_heads, grad_W_o

def positional_encoding(seq_len, d_model, dtype=np.float32):
    pos = np.arange(seq_len, dtype=dtype)[:, np.newaxis]
    i = np.arange(d_model, dtype=dtype)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        2 * np.floor(i / 2) / d_model,
    )
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads

def feed_forward(x, weight1, weight2):
    hidden = matmul(x, weight1)
    hidden_act = silu(hidden)
    output = matmul(hidden_act, weight2)
    return output, (hidden, hidden_act)

def feed_forward_backward(x, weight1, weight2, grad_output, cache):
    hidden, hidden_act = cache

    # output = hidden_act @ weight2
    grad_hidden_act, grad_weight2 = matmul_backward(
        hidden_act,
        weight2,
        grad_output,
    )

    # hidden_act = silu(hidden)
    grad_hidden = silu_backward(
        hidden,
        grad_hidden_act,
    )

    # hidden = x @ weight1
    grad_x, grad_weight1 = matmul_backward(
        x,
        weight1,
        grad_hidden,
    )

    return grad_x, grad_weight1, grad_weight2

def transformer_block(x, params):
    norm1, norm1_cache = rmsnorm(x, params["norm1"])

    attn_output, attention_cache = self_attention(
        norm1,
        params["attention_heads"],
        params["W_o"],
    )
    x_after_attention = x + attn_output

    norm2, norm2_cache = rmsnorm(x_after_attention, params["norm2"])
    ffn_output, ffn_cache = feed_forward(norm2, params["W_1"], params["W_2"])
    output = x_after_attention + ffn_output

    cache = {
        "norm1": norm1,
        "norm1_cache": norm1_cache,
        "attention": attention_cache,
        "x_after_attention": x_after_attention,
        "norm2": norm2,
        "norm2_cache": norm2_cache,
        "ffn": ffn_cache,
    }
    return output, cache

def transformer_block_backward(x, params, grad_output, cache):
    norm1 = cache["norm1"]
    x_after_attention = cache["x_after_attention"]
    norm2 = cache["norm2"]

    # output = x_after_attention + ffn_output
    grad_x_after_attention = grad_output
    grad_ffn_output = grad_output

    grad_norm2, grad_W_1, grad_W_2 = feed_forward_backward(
        norm2, params["W_1"], params["W_2"], grad_ffn_output, cache["ffn"]
    )
    grad_x_from_norm2, grad_norm2_weight = rmsnorm_backward(
        x_after_attention, params["norm2"], grad_norm2, cache["norm2_cache"]
    )
    grad_x_after_attention = grad_x_after_attention + grad_x_from_norm2

    # x_after_attention = x + attn_output
    grad_x_residual = grad_x_after_attention
    grad_attn_output = grad_x_after_attention
    grad_norm1, grad_attention_heads, grad_W_o = self_attention_backward(
        norm1,
        params["attention_heads"],
        params["W_o"],
        grad_attn_output,
        cache["attention"],
    )
    grad_x_from_norm1, grad_norm1_weight = rmsnorm_backward(
        x, params["norm1"], grad_norm1, cache["norm1_cache"]
    )
    grad_x = grad_x_residual + grad_x_from_norm1

    grads = {
        "norm1": grad_norm1_weight,
        "attention_heads": grad_attention_heads,
        "W_o": grad_W_o,
        "norm2": grad_norm2_weight,
        "W_1": grad_W_1,
        "W_2": grad_W_2,
    }
    return grad_x, grads

def transformer(token_ids, params):
    x = embedding(token_ids, params["embedding"])

    positions = positional_encoding(
        seq_len=token_ids.shape[-1],
        d_model=x.shape[-1],
        dtype=x.dtype,
    )
    x = x + positions

    block_inputs = []
    block_caches = []

    for block_params in params["blocks"]:
        block_inputs.append(x)
        x, block_cache = transformer_block(x, block_params)
        block_caches.append(block_cache)

    before_final_norm = x
    x, final_norm_cache = rmsnorm(x, params["final_norm"])
    final_normalized = x

    logits = matmul(x, params["output_weight"])

    cache = {
        "token_ids": token_ids,
        "block_inputs": block_inputs,
        "block_caches": block_caches,
        "before_final_norm": before_final_norm,
        "final_norm_cache": final_norm_cache,
        "final_normalized": final_normalized,
    }

    return logits, cache

def transformer_backward(params, cache, grad_logits):
    # Vocabulary projection
    grad_x, grad_output_weight = matmul_backward(
        cache["final_normalized"],
        params["output_weight"],
        grad_logits,
    )

    # Final RMSNorm
    grad_x, grad_final_norm = rmsnorm_backward(
        cache["before_final_norm"],
        params["final_norm"],
        grad_x,
        cache["final_norm_cache"],
    )

    # Transformer blocks in reverse order
    block_grads = [None] * len(params["blocks"])

    for i in reversed(range(len(params["blocks"]))):
        grad_x, block_grads[i] = transformer_block_backward(
            cache["block_inputs"][i],
            params["blocks"][i],
            grad_x,
            cache["block_caches"][i],
        )

    # Embedding lookup
    grad_embedding = embedding_backward(
        cache["token_ids"],
        params["embedding"],
        grad_x,
    )

    grads = {
        "embedding": grad_embedding,
        "blocks": block_grads,
        "final_norm": grad_final_norm,
        "output_weight": grad_output_weight,
    }

    return grads

def xavier_normal(rng, shape, dtype=np.float32):
    fan_in, fan_out = shape[-2:]
    std = np.sqrt(2 / (fan_in + fan_out))
    return rng.normal(0, std, shape).astype(dtype)

def initialize_parameters(
    vocab_size,
    d_model,
    d_ff,
    n_layers,
    n_heads,
    seed=0,
    dtype=np.float32,
):
    rng = np.random.default_rng(seed)
    residual_scale = 1 / np.sqrt(2 * n_layers)

    blocks = []

    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_attention = d_model // n_heads

    for _ in range(n_layers):
        attention_heads = [
            {
                "W_q": xavier_normal(
                    rng, (d_model, d_attention), dtype
                ),
                "W_k": xavier_normal(
                    rng, (d_model, d_attention), dtype
                ),
                "W_v": xavier_normal(
                    rng, (d_model, d_attention), dtype
                ),
            }
            for _ in range(n_heads)
        ]
        blocks.append({
            "norm1": np.ones(d_model, dtype=dtype),

            "attention_heads": attention_heads,
            "W_o": (
                xavier_normal(rng, (d_model, d_model), dtype)
                * residual_scale
            ),

            "norm2": np.ones(d_model, dtype=dtype),

            "W_1": xavier_normal(
                rng, (d_model, d_ff), dtype
            ),
            "W_2": (
                xavier_normal(rng, (d_ff, d_model), dtype)
                * residual_scale
            ),
        })

    params = {
        "embedding": rng.normal(
            0,
            1 / np.sqrt(d_model),
            (vocab_size, d_model),
        ).astype(dtype),

        "blocks": blocks,

        "final_norm": np.ones(
            d_model,
            dtype=dtype,
        ),

        "output_weight": xavier_normal(
            rng,
            (d_model, vocab_size),
            dtype,
        ),
    }

    return params

def matmul(x, M):
    return x @ M

def matmul_backward(x, M, grad_output):
    grad_x = grad_output @ M.T

    x_2d = x.reshape(-1, x.shape[-1])
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    grad_M = x_2d.T @ grad_output_2d

    return grad_x, grad_M

def add(x, y):
    return x + y

def add_backward(x, y, grad_output):
    # NOTE: this may not work correctly for broadcasting cases where x and y have different shapes
    grad_x = grad_output
    grad_y = grad_output
    return grad_x, grad_y

def silu(x):
    # NOTES: 1) could use logaddexp for numerical stability. 2) could cache the sigmoid itself `s` and pass it thru instead of recomputing it in the backward pass.
    return x * (1 / (1 + np.exp(-x)))

def silu_backward(x, grad_output):
    sigmoid = 1 / (1 + np.exp(-x))
    grad_x = grad_output * (sigmoid + x * sigmoid * (1 - sigmoid))
    return grad_x

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_backward(x, grad_output):
    s = softmax(x)
    grad_x = grad_output - np.sum(grad_output * s, axis=-1, keepdims=True)
    grad_x *= s
    return grad_x

def softmax_cross_entropy(logits, labels):
    # computes softmax and cross-entropy loss for the given logits and labels, instead of separate softmax and cross-entropy steps. This makes the gradients easier and more numerically stable
    logits_2d = logits.reshape(-1, logits.shape[-1])
    labels_1d = labels.reshape(-1)

    shifted_logits = logits_2d - np.max(
        logits_2d, axis=-1, keepdims=True
    )

    log_probs = shifted_logits - np.log(
        np.sum(np.exp(shifted_logits), axis=-1, keepdims=True)
    )

    token_losses = -log_probs[np.arange(labels_1d.size), labels_1d]
    return np.mean(token_losses)

def softmax_cross_entropy_backward(logits, labels, grad_output=1.0):
    original_shape = logits.shape
    logits_2d = logits.reshape(-1, logits.shape[-1])
    labels_1d = labels.reshape(-1)

    grad_logits = softmax(logits_2d)
    grad_logits[np.arange(labels_1d.size), labels_1d] -= 1
    grad_logits /= labels_1d.size

    return grad_output * grad_logits.reshape(original_shape)

def rmsnorm(x, weight, eps=1e-8):
    mean_square = np.mean(x**2, axis=-1, keepdims=True)
    inv_rms = 1 / np.sqrt(mean_square + eps)
    x_normalized = x * inv_rms

    return x_normalized * weight, (inv_rms, x_normalized)

def rmsnorm_backward(x, weight, grad_output, cache):
    inv_rms, x_normalized = cache
    grad_normalized = grad_output * weight

    mean_dot = np.mean(
        x * grad_normalized,
        axis=-1,
        keepdims=True,
    )

    grad_x = (
        grad_normalized * inv_rms
        - x * mean_dot * inv_rms**3
    )

    grad_weight = np.sum(
        grad_output * x_normalized,
        axis=tuple(range(x.ndim - 1)),
    )

    return grad_x, grad_weight

def embedding(x, weight):
    return weight[x]

def embedding_backward(x, weight, grad_output):
    grad_weight = np.zeros_like(weight)
    np.add.at(grad_weight, x, grad_output)
    return grad_weight

def scaled_dot_product_attention(q, k, v):
    d_k = q.shape[-1]
    scale = 1 / np.sqrt(d_k)

    scores = (q @ np.swapaxes(k, -1, -2)) * scale

    sequence_length = q.shape[-2]
    causal_mask = np.triu(
        np.ones((sequence_length, sequence_length), dtype=bool),
        k=1,
    )
    scores = np.where(causal_mask, -np.inf, scores)

    attn = softmax(scores)
    output = attn @ v

    return output, attn

def scaled_dot_product_attention_backward(q, k, v, grad_output, attn):
    scale = 1 / np.sqrt(q.shape[-1])

    # output = attn @ v
    grad_attn = grad_output @ np.swapaxes(v, -1, -2)
    grad_v = np.swapaxes(attn, -1, -2) @ grad_output

    # attn = softmax(scores)
    grad_scores = (
        grad_attn - np.sum(grad_attn * attn, axis=-1, keepdims=True)
    ) * attn

    # scores = q @ k.T * scale
    grad_q = (grad_scores @ k) * scale
    grad_k = (np.swapaxes(grad_scores, -1, -2) @ q) * scale

    return grad_q, grad_k, grad_v

def self_attention(x, attention_heads, W_o):
    W_qs = np.stack([head["W_q"] for head in attention_heads], axis=0)
    W_ks = np.stack([head["W_k"] for head in attention_heads], axis=0)
    W_vs = np.stack([head["W_v"] for head in attention_heads], axis=0)

    x_expanded = np.expand_dims(x, axis=1)
    q = matmul(x_expanded, W_qs)
    k = matmul(x_expanded, W_ks)
    v = matmul(x_expanded, W_vs)

    head_outputs, attn = scaled_dot_product_attention(q, k, v)
    head_outputs = np.swapaxes(head_outputs, 1, 2)
    concatenated = head_outputs.reshape(x.shape[0], x.shape[1], -1)
    output = matmul(concatenated, W_o)

    cache = {
        "q": q,
        "k": k,
        "v": v,
        "attn": attn,
        "concatenated": concatenated,
    }
    return output, cache

def self_attention_backward(
    x, attention_heads, W_o, grad_output, cache
):
    grad_concatenated, grad_W_o = matmul_backward(
        cache["concatenated"], W_o, grad_output
    )

    n_heads = len(attention_heads)
    d_head = cache["q"].shape[-1]
    grad_head_outputs = grad_concatenated.reshape(
        x.shape[0], x.shape[1], n_heads, d_head
    )
    grad_head_outputs = np.swapaxes(grad_head_outputs, 1, 2)

    grad_q, grad_k, grad_v = scaled_dot_product_attention_backward(
        cache["q"], cache["k"], cache["v"], grad_head_outputs, cache["attn"]
    )

    # Broadcast x across heads, then sum over the batch for weight gradients.
    x_transposed = np.swapaxes(x, -1, -2)[:, np.newaxis]
    grad_W_qs = np.sum(x_transposed @ grad_q, axis=0)
    grad_W_ks = np.sum(x_transposed @ grad_k, axis=0)
    grad_W_vs = np.sum(x_transposed @ grad_v, axis=0)

    W_qs = np.stack([head["W_q"] for head in attention_heads], axis=0)
    W_ks = np.stack([head["W_k"] for head in attention_heads], axis=0)
    W_vs = np.stack([head["W_v"] for head in attention_heads], axis=0)

    # Every head reads the same input, so its contributions add together.
    grad_x = np.sum(grad_q @ np.swapaxes(W_qs, -1, -2), axis=1)
    grad_x += np.sum(grad_k @ np.swapaxes(W_ks, -1, -2), axis=1)
    grad_x += np.sum(grad_v @ np.swapaxes(W_vs, -1, -2), axis=1)

    grad_attention_heads = [
        {"W_q": grad_W_qs[i], "W_k": grad_W_ks[i], "W_v": grad_W_vs[i]}
        for i in range(n_heads)
    ]
    return grad_x, grad_attention_heads, grad_W_o

def positional_encoding(seq_len, d_model, dtype=np.float32):
    pos = np.arange(seq_len, dtype=dtype)[:, np.newaxis]
    i = np.arange(d_model, dtype=dtype)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        2 * np.floor(i / 2) / d_model,
    )
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads

def feed_forward(x, weight1, weight2):
    hidden = matmul(x, weight1)
    hidden_act = silu(hidden)
    output = matmul(hidden_act, weight2)
    return output, (hidden, hidden_act)

def feed_forward_backward(x, weight1, weight2, grad_output, cache):
    hidden, hidden_act = cache

    # output = hidden_act @ weight2
    grad_hidden_act, grad_weight2 = matmul_backward(
        hidden_act,
        weight2,
        grad_output,
    )

    # hidden_act = silu(hidden)
    grad_hidden = silu_backward(
        hidden,
        grad_hidden_act,
    )

    # hidden = x @ weight1
    grad_x, grad_weight1 = matmul_backward(
        x,
        weight1,
        grad_hidden,
    )

    return grad_x, grad_weight1, grad_weight2

def transformer_block(x, params):
    norm1, norm1_cache = rmsnorm(x, params["norm1"])

    attn_output, attention_cache = self_attention(
        norm1,
        params["attention_heads"],
        params["W_o"],
    )
    x_after_attention = x + attn_output

    norm2, norm2_cache = rmsnorm(x_after_attention, params["norm2"])
    ffn_output, ffn_cache = feed_forward(norm2, params["W_1"], params["W_2"])
    output = x_after_attention + ffn_output

    cache = {
        "norm1": norm1,
        "norm1_cache": norm1_cache,
        "attention": attention_cache,
        "x_after_attention": x_after_attention,
        "norm2": norm2,
        "norm2_cache": norm2_cache,
        "ffn": ffn_cache,
    }
    return output, cache

def transformer_block_backward(x, params, grad_output, cache):
    norm1 = cache["norm1"]
    x_after_attention = cache["x_after_attention"]
    norm2 = cache["norm2"]

    # output = x_after_attention + ffn_output
    grad_x_after_attention = grad_output
    grad_ffn_output = grad_output

    grad_norm2, grad_W_1, grad_W_2 = feed_forward_backward(
        norm2, params["W_1"], params["W_2"], grad_ffn_output, cache["ffn"]
    )
    grad_x_from_norm2, grad_norm2_weight = rmsnorm_backward(
        x_after_attention, params["norm2"], grad_norm2, cache["norm2_cache"]
    )
    grad_x_after_attention = grad_x_after_attention + grad_x_from_norm2

    # x_after_attention = x + attn_output
    grad_x_residual = grad_x_after_attention
    grad_attn_output = grad_x_after_attention
    grad_norm1, grad_attention_heads, grad_W_o = self_attention_backward(
        norm1,
        params["attention_heads"],
        params["W_o"],
        grad_attn_output,
        cache["attention"],
    )
    grad_x_from_norm1, grad_norm1_weight = rmsnorm_backward(
        x, params["norm1"], grad_norm1, cache["norm1_cache"]
    )
    grad_x = grad_x_residual + grad_x_from_norm1

    grads = {
        "norm1": grad_norm1_weight,
        "attention_heads": grad_attention_heads,
        "W_o": grad_W_o,
        "norm2": grad_norm2_weight,
        "W_1": grad_W_1,
        "W_2": grad_W_2,
    }
    return grad_x, grads

def transformer(token_ids, params):
    x = embedding(token_ids, params["embedding"])

    positions = positional_encoding(
        seq_len=token_ids.shape[-1],
        d_model=x.shape[-1],
        dtype=x.dtype,
    )
    x = x + positions

    block_inputs = []
    block_caches = []

    for block_params in params["blocks"]:
        block_inputs.append(x)
        x, block_cache = transformer_block(x, block_params)
        block_caches.append(block_cache)

    before_final_norm = x
    x, final_norm_cache = rmsnorm(x, params["final_norm"])
    final_normalized = x

    logits = matmul(x, params["output_weight"])

    cache = {
        "token_ids": token_ids,
        "block_inputs": block_inputs,
        "block_caches": block_caches,
        "before_final_norm": before_final_norm,
        "final_norm_cache": final_norm_cache,
        "final_normalized": final_normalized,
    }

    return logits, cache

def transformer_backward(params, cache, grad_logits):
    # Vocabulary projection
    grad_x, grad_output_weight = matmul_backward(
        cache["final_normalized"],
        params["output_weight"],
        grad_logits,
    )

    # Final RMSNorm
    grad_x, grad_final_norm = rmsnorm_backward(
        cache["before_final_norm"],
        params["final_norm"],
        grad_x,
        cache["final_norm_cache"],
    )

    # Transformer blocks in reverse order
    block_grads = [None] * len(params["blocks"])

    for i in reversed(range(len(params["blocks"]))):
        grad_x, block_grads[i] = transformer_block_backward(
            cache["block_inputs"][i],
            params["blocks"][i],
            grad_x,
            cache["block_caches"][i],
        )

    # Embedding lookup
    grad_embedding = embedding_backward(
        cache["token_ids"],
        params["embedding"],
        grad_x,
    )

    grads = {
        "embedding": grad_embedding,
        "blocks": block_grads,
        "final_norm": grad_final_norm,
        "output_weight": grad_output_weight,
    }

    return grads

#############################################
### PARAMETER INITIALIZATION AND TRAINING ###
############################################

def xavier_normal(rng, shape, dtype=np.float32):
    fan_in, fan_out = shape[-2:]
    std = np.sqrt(2 / (fan_in + fan_out))
    return rng.normal(0, std, shape).astype(dtype)

def initialize_parameters(
    vocab_size,
    d_model,
    d_ff,
    n_layers,
    n_heads,
    seed=0,
    dtype=np.float32,
):
    rng = np.random.default_rng(seed)
    residual_scale = 1 / np.sqrt(2 * n_layers)

    blocks = []

    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_attention = d_model // n_heads

    for _ in range(n_layers):
        attention_heads = [
            {
                "W_q": xavier_normal(
                    rng, (d_model, d_attention), dtype
                ),
                "W_k": xavier_normal(
                    rng, (d_model, d_attention), dtype
                ),
                "W_v": xavier_normal(
                    rng, (d_model, d_attention), dtype
                ),
            }
            for _ in range(n_heads)
        ]
        blocks.append({
            "norm1": np.ones(d_model, dtype=dtype),

            "attention_heads": attention_heads,
            "W_o": (
                xavier_normal(rng, (d_model, d_model), dtype)
                * residual_scale
            ),

            "norm2": np.ones(d_model, dtype=dtype),

            "W_1": xavier_normal(
                rng, (d_model, d_ff), dtype
            ),
            "W_2": (
                xavier_normal(rng, (d_ff, d_model), dtype)
                * residual_scale
            ),
        })

    params = {
        "embedding": rng.normal(
            0,
            1 / np.sqrt(d_model),
            (vocab_size, d_model),
        ).astype(dtype),

        "blocks": blocks,

        "final_norm": np.ones(
            d_model,
            dtype=dtype,
        ),

        "output_weight": xavier_normal(
            rng,
            (d_model, vocab_size),
            dtype,
        ),
    }

    return params

###############################
### OPTIMIZATION ALGORITHMS ###
###############################

def zeros_like_tree(tree):
    if isinstance(tree, dict):
        return {
            key: zeros_like_tree(value)
            for key, value in tree.items()
        }

    if isinstance(tree, list):
        return [
            zeros_like_tree(value)
            for value in tree
        ]

    return np.zeros_like(tree)

def adam_step(
    params,
    grads,
    state,
    learning_rate=3e-4,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
):
    state["step"] += 1
    step = state["step"]

    def update(params, grads, momentum, squared_grad):
        if isinstance(params, dict):
            for key in params:
                update(
                    params[key],
                    grads[key],
                    momentum[key],
                    squared_grad[key],
                )

        elif isinstance(params, list):
            for parameter, gradient, m, v in zip(
                params,
                grads,
                momentum,
                squared_grad,
            ):
                update(parameter, gradient, m, v)

        else:
            momentum[...] = (
                beta1 * momentum
                + (1 - beta1) * grads
            )

            squared_grad[...] = (
                beta2 * squared_grad
                + (1 - beta2) * grads**2
            )

            corrected_momentum = (
                momentum / (1 - beta1**step)
            )

            corrected_squared_grad = (
                squared_grad / (1 - beta2**step)
            )

            params[...] -= (
                learning_rate
                * corrected_momentum
                / (np.sqrt(corrected_squared_grad) + eps)
            )

    update(
        params,
        grads,
        state["momentum"],
        state["squared_grad"],
    )


#################
### INFERENCE ###
#################

from IPython.display import display
from attention_visualize import attention_viewer, wrapped_output

def generate(
    tokenizer,
    params,
    prompt,
    number_of_tokens,
    context_length,
    temperature=1.0,
    seed=0,
    SHOW_ATTENTION=True,
):
    encode, decode, VOCAB_SIZE = tokenizer

    rng = np.random.default_rng(seed)
    token_ids = list(encode(prompt))
    prompt_token_count = len(token_ids)
    generated_text = decode(token_ids)
    attention_steps = [] if SHOW_ATTENTION else None

    output = display(
        wrapped_output(generated_text),
        display_id=True,
    )

    for _ in range(number_of_tokens):
        context = np.asarray(
            token_ids[-context_length:],
            dtype=np.int64,
        )[np.newaxis, :]

        logits, cache = transformer(context, params)
        if SHOW_ATTENTION:
            context_start = len(token_ids) - context.shape[-1]
            probabilities_by_layer = np.stack([
                block_cache["attention"]["attn"][0, :, -1, :]
                for block_cache in cache["block_caches"]
            ])
            attention_steps.append((context_start, probabilities_by_layer))

        next_logits = logits[0, -1] / temperature
        probabilities = softmax(next_logits)

        next_token = rng.choice(
            VOCAB_SIZE,
            p=probabilities,
        )

        token_ids.append(int(next_token))

        generated_text = decode(token_ids)

        output.update(wrapped_output(generated_text))

    if SHOW_ATTENTION and attention_steps:
        viewer, behavior = attention_viewer(
            [decode([token_id]) for token_id in token_ids],
            prompt_token_count,
            attention_steps,
            len(params["blocks"]),
            len(params["blocks"][0]["attention_heads"]),
            separator=(
                " "
                if len(token_ids) > 1
                and decode(token_ids)
                != "".join(decode([token_id]) for token_id in token_ids)
                else ""
            ),
        )
        output.update(viewer)
        display(behavior)

    return generated_text

#################################
### SAVING AND LOADING MODELS ###
#################################

import dill

def save_model(params, adam_state, encode, decode, filename="model.pkl"):
    with open(filename, "wb") as f:
        dill.dump((params, adam_state, encode, decode), f)

def load_model(filename="model.pkl"):
    with open(filename, "rb") as f:
        return dill.load(f)
