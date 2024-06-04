# coding=utf-8
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

import enum
from enum import Enum

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.fused_moe import fused_moe, quant_fused_moe
from vllm.model_executor.layers.linear import (adjust_marlin_shard,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear,
                                               MergedColumnParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.dbrx import DbrxConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm import _custom_ops as ops

class ExllamaState(Enum):

    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()

class DbrxRouter(nn.Module):
    """A Router implementation for DBRX that returns logits for each expert
    per token.
    """

    def __init__(
        self,
        config: DbrxConfig,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.d_model = config.d_model
        self.layer = ReplicatedLinear(
            self.d_model,
            self.num_total_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.layer(hidden_states)
        return router_logits


class DbrxExperts(nn.Module):
    """A tensor-parallel MoE implementation for DBRX.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: DbrxConfig,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.top_k = config.ffn_config.moe_top_k
        self.d_model = config.d_model
        self.intermediate_size = (config.ffn_config.ffn_hidden_size //
                                  self.tp_size)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.router = DbrxRouter(config, self.params_dtype)
        self.quant_config = quant_config
        self.use_fused_gptq_moe = isinstance(quant_config, GPTQConfig)
        if self.use_fused_gptq_moe:
            self.intermediate_size = config.ffn_config.ffn_hidden_size
            self.output_sizes = [self.intermediate_size] * 2
            self.ws = ReplicatedLinear(
                    self.d_model, divide(self.intermediate_size * 2, self.tp_size),
                                    bias=False,
                                    quant_config=quant_config)
            self.w2s = ReplicatedLinear(divide(self.intermediate_size, self.tp_size),
                                    self.d_model,
                                    bias=False,
                                    quant_config=quant_config)
            self.ws.quant_method.create_moe_weights(layer=self.ws, 
                                                    input_size_per_partition=self.d_model, 
                                                    output_partition_sizes=[divide(self.intermediate_size * 2, self.tp_size)],
                                                    input_size=self.d_model,
                                                    output_size=self.intermediate_size * 2,
                                                    params_dtype=self.params_dtype,
                                                    num_experts=self.num_total_experts,
                                                    weight_loader=self.weight_loader)
            self.w2s.quant_method.create_moe_weights(layer=self.w2s, 
                                                    input_size_per_partition=divide(self.intermediate_size, self.tp_size), 
                                                    output_partition_sizes=[self.d_model],
                                                    input_size=self.intermediate_size,
                                                    output_size=self.d_model,
                                                    params_dtype=self.params_dtype,
                                                    num_experts=self.num_total_experts,
                                                    weight_loader=self.weight_loader)
        else:
            self.ws = nn.Parameter(
                torch.empty(
                    self.num_total_experts,
                    2 * self.intermediate_size,
                    self.d_model,
                    device="cuda",
                    dtype=self.params_dtype,
                ))
            self.w2s = nn.Parameter(
                torch.empty(
                    self.num_total_experts,
                    self.d_model,
                    self.intermediate_size,
                    device="cuda",
                    dtype=self.params_dtype,
                ))

            set_weight_attrs(
                self.ws,
                {
                    "weight_loader": self.weight_loader,
                },
            )
            set_weight_attrs(
                self.w2s,
                {
                    "weight_loader": self.weight_loader,
                },
            )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int = -1, loaded_shard_id: int = None):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        param_data = param.data
        if (expert_id != -1) and self.use_fused_gptq_moe:
            param_data = param_data[expert_id]
            if "w2" in weight_name:
                input_dim = getattr(param, "input_dim", None)
                if input_dim is not None:
                    shard_size = param_data.shape[input_dim]
                    start_idx = tp_rank * shard_size
                    loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                        shard_size)
            if ("w1" in weight_name) or ("v1" in weight_name):
                output_dim = getattr(param, "output_dim", None)
                if output_dim is not None:
                    shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
                    shard_size = self.output_sizes[loaded_shard_id] // tp_size
                    # Special case for quantization.
                    # If quantized, we need to adjust the offset and size to account
                    # for the packing.
                    packed_dim = getattr(param, "packed_dim", None)
                    if packed_dim == output_dim:
                        shard_size = shard_size // param.pack_factor
                        shard_offset = shard_offset // param.pack_factor
                        # Special case for Marlin.
                        shard_size, shard_offset = adjust_marlin_shard(
                            param, shard_size, shard_offset)

                    param_data = param_data.narrow(output_dim, shard_offset,
                                                shard_size)
                    start_idx = tp_rank * shard_size
                    loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                        shard_size)
            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
            return

        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        # DBRX uses GLU for each experts.
        # GLU has 3 linear layers: w1, v1 and w2.
        if weight_name.endswith("w1"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model],
            )
            param_data[:, 0:shard_size, :] = loaded_weight[:, shard, :]
        if weight_name.endswith("v1"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model],
            )
            param_data[:,
                       shard_size:2 * shard_size, :] = loaded_weight[:,
                                                                     shard, :]
        if weight_name.endswith("w2"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model],
            ).transpose(1, 2)
            param_data[:] = loaded_weight[:, :, shard]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.d_model)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.router(hidden_states)
        
        if self.use_fused_gptq_moe:
            # shuffle weights for exllama
            for w in [self.ws, self.w2s]:
                if w.exllama_state == ExllamaState.UNINITIALIZED:
                    if self.quant_config.desc_act:
                        w.g_idx.data[:] = torch.argsort(w.g_idx.data[:],
                                                dim=-1).to(torch.int)
                    else:
                        w.g_idx.data[:] = torch.arange(
                            w.g_idx.data.shape[1], device=w.g_idx.device).unsqueeze(0).repeat(
                                w.g_idx.data.shape[0], 1)
                    w.exllama_state = ExllamaState.READY
                    ops.gptq_shuffle(w.qweight, w.g_idx,
                                    self.quant_config.weight_bits)

            # For memory bound workloads: decode and small prefills, use the
            # fused quant moe. Otherwise, dequantize them individually
            
            final_hidden_states = quant_fused_moe(
                hidden_states,
                self.ws.qweight,
                self.ws.scales,
                self.ws.qzeros,
                self.ws.g_idx,
                self.w2s.qweight,
                self.w2s.scales,
                self.w2s.qzeros,
                self.w2s.g_idx,
                router_logits,
                self.top_k,
                True,
                self.quant_config.weight_bits,
            )
        else:
            final_hidden_states = fused_moe(
                hidden_states,
                self.ws,
                self.w2s,
                router_logits,
                self.top_k,
                renormalize=True,
                inplace=True,
            )

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_size)


class DbrxAttention(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.head_dim = self.d_model // self.total_num_heads
        self.total_num_kv_heads = config.attn_config.kv_n_heads
        self.clip_qkv = config.attn_config.clip_qkv
        self.rope_theta = config.attn_config.rope_theta
        self.max_position = config.max_seq_len

        # pylint: disable=invalid-name
        self.Wqkv = QKVParallelLinear(
            self.d_model,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=False,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_world_size
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size
        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              quant_config=quant_config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        hidden_states, _ = self.out_proj(attn_output)
        return hidden_states


class DbrxFusedNormAttention(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.attn = DbrxAttention(config, quant_config)
        self.norm_1 = nn.LayerNorm(self.d_model)
        self.norm_2 = nn.LayerNorm(self.d_model)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        x = self.attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + x
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        return hidden_states, residual


class DbrxBlock(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.norm_attn_norm = DbrxFusedNormAttention(config,
                                                     quant_config)
        self.ffn = DbrxExperts(config, quant_config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states, residual = self.norm_attn_norm(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = self.ffn(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class DbrxModel(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        self.blocks = nn.ModuleList([
            DbrxBlock(config, cache_config, quant_config)
            for _ in range(config.n_layers)
        ])
        self.norm_f = nn.LayerNorm(config.d_model, eps=1e-5)
        for module in self.modules():
            if hasattr(module, "bias") and isinstance(module.bias,
                                                      nn.Parameter):
                # Remove the bias term in Linear and LayerNorm.
                module.register_parameter("bias", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            hidden_states = block(
                position_ids,
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )
        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class DbrxForCausalLM(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.use_fused_gptq_moe = isinstance(quant_config, GPTQConfig)
        self.unpadded_vocab_size = config.vocab_size
        self.transformer = DbrxModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
        )
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        if not self.use_fused_gptq_moe:
            expert_params_mapping = [(
                "ws" if weight_name in ["w1", "v1"] else "w2s",
                f"experts.mlp.{weight_name}",
            ) for weight_name in ["w1", "v1", "w2"]]
            for name, loaded_weight in weights:
                for param_name, weight_name in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, weight_name)
                    break
                else:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
        else:
            expert_params_mapping = [
                ("ws" if weight_name in ["w1", "v1"] else "w2s",
                f"experts.mlp.{weight_name}", shard_id)
                for weight_name, shard_id in [("w1", 0), ("v1", 1), ("w2", None)]
            ]
            for name, loaded_weight in weights:
                for (param_name, weight_name, shard_id) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    expert_id = int(name.split(".")[-2].split("_")[1])
                    name = name.replace(weight_name + f"_{expert_id}", param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                    loaded_weight,
                                    weight_name=weight_name,
                                    expert_id=expert_id,
                                    loaded_shard_id=shard_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Skip experts that are not assigned to this worker.
                    if ("ffn.experts.mlp." in name and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
