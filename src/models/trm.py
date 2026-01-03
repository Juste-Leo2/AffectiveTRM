# src/models/trm.py

from typing import Tuple, List, Dict
from dataclasses import dataclass
import torch
from torch import nn
from pydantic import BaseModel

# Imports depuis layers (inchangés)
from .layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, trunc_normal_init_

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    input_dim: int
    num_classes: int
    H_cycles: int
    L_cycles: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    no_ACT_continue: bool = True
    forward_dtype: str = "float32"

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = torch.float32 if config.forward_dtype == "float32" else torch.bfloat16
        
        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_size, bias=False)
        
        # Tête de sortie du réseau profond
        self.output_head = nn.Linear(self.config.hidden_size, 2, bias=False)
        
        # Tête de sortie directe (Shortcut)
        self.shortcut_head = nn.Linear(self.config.input_dim, 2, bias=True)

        self.q_head = nn.Linear(self.config.hidden_size, 2, bias=True)
        
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=2,
                                              base=self.config.rope_theta)

        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.L_layers)]
        )

        self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02))
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02))

        # --- CORRECTION D'INITIALISATION ---
        with torch.no_grad():
            # 1. On laisse le shortcut apprendre vite (Xavier Init ou défaut)
            nn.init.xavier_uniform_(self.shortcut_head.weight)
            nn.init.zeros_(self.shortcut_head.bias)
            
            # 2. On met la tête profonde à ZERO pour commencer comme un modèle linéaire
            # Cela évite le collapse to mean immédiat dû au bruit du réseau profond
            self.output_head.weight.zero_()
            self.output_head.bias.zero_() if self.output_head.bias is not None else None
            
            # 3. Init du Halting (pour encourager à ne pas s'arrêter tout de suite ou selon logique)
            self.q_head.weight.data.normal_(0, 0.02)
            self.q_head.bias.fill_(-3.0) 

    def _input_projection(self, sensor_data: torch.Tensor):
        projected = self.input_proj(sensor_data)
        return projected.unsqueeze(1)

    def empty_carry(self, batch_size: int, device: torch.device):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.zeros(batch_size, 1, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.zeros(batch_size, 1, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        mask = reset_flag.view(-1, 1, 1)
        h_init_expanded = self.H_init.view(1, 1, -1).expand(mask.shape[0], -1, -1)
        l_init_expanded = self.L_init.view(1, 1, -1).expand(mask.shape[0], -1, -1)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(mask, h_init_expanded, carry.z_H),
            z_L=torch.where(mask, l_init_expanded, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        inputs_raw = batch["inputs"]
        input_projected = self._input_projection(inputs_raw)

        z_H, z_L = carry.z_H, carry.z_L
        
        # Récursion
        for _H_step in range(self.config.H_cycles-1):
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_projected, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_projected, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)
        
        output_state = z_H.squeeze(1)
        
        # Combinaison : Deep + Shortcut
        deep_logits = self.output_head(output_state)
        shortcut_logits = self.shortcut_head(inputs_raw)
        
        output_logits = deep_logits + shortcut_logits

        q_logits = self.q_head(output_state).to(torch.float32)
        return new_carry, output_logits, (q_logits[..., 0], q_logits[..., 1])

class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config):
        super().__init__()
        self.config = config
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = next(self.parameters()).device
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),
            current_data={k: torch.zeros_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.tensor(0, device=carry.steps.device, dtype=torch.int32), carry.steps)
        
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), 
                batch[k], 
                carry.current_data[k]
            ) for k in batch.keys()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        
        outputs = {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            if self.config.no_ACT_continue: should_halt = (q_halt_logits > 0)
            else: should_halt = (q_halt_logits > q_continue_logits)
            halted = is_last_step | should_halt
            if self.training and self.config.halt_exploration_prob > 0:
                mask = torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                halted = halted & ~mask

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs