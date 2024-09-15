import torch
import json
import os
from einops import rearrange
from torch import Tensor
from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management

# グローバルリストを作成（スクリプトの先頭で定義）
global_tensor_list = []

def retrieve_and_remove_tensor(tensor_name):
    for i, tensor_data in enumerate(global_tensor_list):
        if tensor_data['name'] == tensor_name:
            # リストからテンソルデータを削除し、返す
            return global_tensor_list.pop(i)

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, pe_ref: Tensor, pe_ref2: Tensor) -> Tensor:
    ##############

    # JSONファイルから元のメタデータを読み込む
    with open('variables.json', 'r') as f:
        metadata = json.load(f)

    mode = metadata.get('mode', 'normal')  # デフォルトはnormal
    timesteps = float(metadata['timesteps'])
    i = int(metadata['i'])
    blockcls = metadata['blockcls']
    attmode = metadata['attmode']
    kfactor = metadata['kfactor']
    vfactor = metadata['vfactor']
    tfactor_pairs = metadata.get('tfactor_pairs', [])

    # timestepに基づいてtfactorを計算
    tfactor = 1.0
    for threshold, factor in sorted(tfactor_pairs, reverse=True):
        if timesteps <= threshold:
            tfactor = factor


    # tfactorを乗算
    kfactor *= tfactor

    tensor_name = f"{i}_{blockcls}"

    if mode == 'ref':

        if attmode == 'source':
            # テンソルをグローバルリストに追加
            k_ref = k.clone()
            v_ref = v.clone()
            global_tensor_list.append({
                'name': tensor_name,
                'k_ref': k_ref,
                'v_ref': v_ref
            })

            q, k = apply_rope(q, k, pe_ref)

        elif attmode == 'target':

            # テンソルをpopする
            retrieved_tensor = {}
            retrieved_tensor = retrieve_and_remove_tensor(tensor_name)
            k_ref1 = retrieved_tensor['k_ref']
            v_ref1 = retrieved_tensor['v_ref']
            k_ref2 = k_ref1.clone()
            v_ref2 = v_ref1.clone()
            q_ref1 = q.clone()
            q_ref2 = q.clone()

            for ref in [k_ref1, v_ref1, k_ref2, v_ref2]:
                ref = ref.to(dtype=k.dtype, device=k.device)

            q, k = apply_rope(q, k, pe)
            q_ref1, k_ref1 = apply_rope(q_ref1, k_ref1, pe_ref)
            q_ref2, k_ref2 = apply_rope(q_ref2, k_ref2, pe_ref2)


            k_ref1 = k_ref1[:, :, 256:, :]
            v_ref1 = v_ref1[:, :, 256:, :]  
            k_ref2 = k_ref2[:, :, 256:, :]
            v_ref2 = v_ref2[:, :, 256:, :]                         

            k_ref1 = k_ref1 * kfactor
            v_ref1 = v_ref1 * vfactor
            k_ref2 = k_ref2 * kfactor
            v_ref2 = v_ref2 * vfactor 

            if 0.85 <= timesteps:

                k_ref1[:, :, :, 16+0 :16+0 +28+8-1] *= 0.1
                k_ref1[:, :, :, 16+56:16+56+28+8-1] *= 0.1
                k_ref2[:, :, :, 16+0 :16+0 +28+8-1] *= 0.1
                k_ref2[:, :, :, 16+56:16+56+28+8-1] *= 0.1

            elif 0.6 <= timesteps < 0.85:
            
                k_ref1[:, :, :, 16+0 :16+0 +28-20-1] *= 0.1
                k_ref1[:, :, :, 16+56:16+56+28-20-1] *= 0.1
                k_ref2[:, :, :, 16+0 :16+0 +28-20-1] *= 0.1
                k_ref2[:, :, :, 16+56:16+56+28-20-1] *= 0.1


            k=torch.cat((k, k_ref1,k_ref2), dim=2)
            v=torch.cat((v, v_ref1,v_ref2), dim=2)



    elif mode == 'normal':
        q, k = apply_rope(q, k, pe)

    ##############    
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if comfy.model_management.is_device_mps(pos.device) or comfy.model_management.is_intel_xpu():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
