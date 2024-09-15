#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass
import json
import os

import torch
from torch import Tensor, nn

from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

from einops import rearrange, repeat
import comfy.ldm.common_dit

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.img_ids_ref = []
        params = FluxParams(**kwargs)
        self.params = params
        self.in_channels = params.in_channels * 2 * 2
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = operations.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype, device=device, operations=operations)

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control=None,
    ) -> Tensor:
        
        
            ##############

        img_org = img.clone()
        txt_org = txt.clone()

        # JSONファイルから元のメタデータを読み込む
        with open('variables.json', 'r') as f:
            metadata = json.load(f)
        mode = metadata.get('mode', 'normal')  # デフォルトはnormal

        # サブフォルダのパスを指定
        subfolder = "tensor_data"

        # テンソル名とファイル名の設定（サブフォルダを含む）
        tensor_name = f"{timesteps.item()}"
        tensor_file = os.path.join(subfolder, f'{tensor_name}.pt')

        # サブフォルダが存在しない場合は作成
        os.makedirs(subfolder, exist_ok=True)

        if mode == 'write':
        # テンソルを書き出す
            torch.save({
                'refimg': img
            }, tensor_file)

            return img
            ##############

        if mode == 'ref':
            attmodes = ["source", "target"]
        if mode == 'normal':
            attmodes = ["normal"]
        for attmode in attmodes:


            if attmode == "source":
                

                # テンソルを読み込む
                if not os.path.exists(tensor_file):
                    print(f"エラー: テンソルファイル {tensor_file} が見つかりません。")
                    return

                loaded_tensors = torch.load(tensor_file)
                img_loaded = loaded_tensors['refimg']
                img_loaded = img_loaded.to(dtype=img.dtype, device=img.device)
                img = img_loaded

            if attmode == "target":
                img = img_org.clone()
                txt = txt_org.clone()
        
            if img.ndim != 3 or txt.ndim != 3:
                raise ValueError("Input img and txt tensors must have 3 dimensions.")

            # running on sequences img
            img = self.img_in(img)
            vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
            if self.params.guidance_embed:
                if guidance is None:
                    raise ValueError("Didn't get guidance strength for guidance distilled model.")
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

            vec = vec + self.vector_in(y)
            txt = self.txt_in(txt)
            ids = torch.cat((txt_ids, img_ids), dim=1)
            ids_ref = torch.cat((txt_ids, self.img_ids_ref[0]), dim=1)
            ids_ref2 = torch.cat((txt_ids, self.img_ids_ref[1]), dim=1)
            pe = self.pe_embedder(ids)
            pe_ref = self.pe_embedder(ids_ref)
            pe_ref2 = self.pe_embedder(ids_ref2)

            for i, block in enumerate(self.double_blocks):
                ###################
                blockcls = "double"

                # JSONファイルから元のメタデータを読み込む
                with open('variables.json', 'r') as f:
                    metadata = json.load(f)
                # 更新したい変数
                data_new = {
                    "timesteps": timesteps.item(),  # tensor を単一の値に変換
                    "i": i,
                    "blockcls": blockcls,
                    "attmode": attmode
                }

                # 元のメタデータを更新
                metadata.update(data_new)

                # 更新されたデータをJSONファイルに書き込む
                with open('variables.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
                ###################

                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, pe_ref=pe_ref, pe_ref2=pe_ref2)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            img = torch.cat((txt, img), 1)

            for i, block in enumerate(self.single_blocks):
                ###################
                blockcls = "single"

                # JSONファイルから元のメタデータを読み込む
                with open('variables.json', 'r') as f:
                    metadata = json.load(f)

                # 更新したい変数
                data_new = {
                    "timesteps": timesteps.item(),  # tensor を単一の値に変換
                    "i": i,
                    "blockcls": blockcls,
                    "attmode": attmode
                }

                # 元のメタデータを更新
                metadata.update(data_new)

                # 更新されたデータをJSONファイルに書き込む
                with open('variables.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
                ###################

                img = block(img, vec=vec, pe=pe, pe_ref=pe_ref, pe_ref2=pe_ref2)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x, timestep, context, y, guidance, control=None, **kwargs):
        bs, c, h, w = x.shape
        patch_size = 2
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        self.img_ids_ref.clear()
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        self.img_ids_ref.append(torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype))
        self.img_ids_ref.append(torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype))
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        self.img_ids_ref[0][..., 1] = self.img_ids_ref[0][..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        self.img_ids_ref[0][..., 2] = self.img_ids_ref[0][..., 2] + torch.linspace(w_len + 32, 2*w_len + 32 - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        self.img_ids_ref[1][..., 1] = self.img_ids_ref[1][..., 1] + torch.linspace(h_len + 32, 2*h_len + 32 - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        self.img_ids_ref[1][..., 2] = self.img_ids_ref[1][..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]

        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        self.img_ids_ref[0] = repeat(self.img_ids_ref[0], "h w c -> b (h w) c", b=bs)
        self.img_ids_ref[1] = repeat(self.img_ids_ref[1], "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control)
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
