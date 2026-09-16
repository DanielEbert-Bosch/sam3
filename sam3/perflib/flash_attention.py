import torch


def flash_attn_func(q, k, v):
    """Run installed FlashAttention on dense batched Q/K/V tensors."""
    major, _ = torch.cuda.get_device_capability(q.device)
    if major >= 8:
        from sam3.perflib.fa3 import flash_attn_func as flash_attn_func_v3

        return flash_attn_func_v3(q, k, v)

    from flash_attn.flash_attn_interface import flash_attn_unpadded_func

    batch_size, query_length, num_heads, head_dim = q.shape
    key_length = k.shape[1]
    cu_query = torch.arange(
        0,
        (batch_size + 1) * query_length,
        query_length,
        dtype=torch.int32,
        device=q.device,
    )
    cu_key = torch.arange(
        0,
        (batch_size + 1) * key_length,
        key_length,
        dtype=torch.int32,
        device=q.device,
    )
    output = flash_attn_unpadded_func(
        q.contiguous().view(-1, num_heads, head_dim),
        k.contiguous().view(-1, num_heads, head_dim),
        v.contiguous().view(-1, num_heads, head_dim),
        cu_query,
        cu_key,
        query_length,
        key_length,
        0.0,
    )
    return output.view(batch_size, query_length, num_heads, head_dim)
