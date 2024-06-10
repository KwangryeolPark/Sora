"""
Author: Kwangryeol Park
E-mail: pkr7098@unist.ac.kr
"""

import math

import torch
import torch.nn as nn

class SoraLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0
    ) -> torch.Tensor:
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else nn.Identity()
        
        self.lora_A = nn.Parameter(weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(weight.new_zeros((out_features, r)))
        self.lora_G = nn.Parameter(torch.randn(1, r))
        self.scaling = lora_alpha / r
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lora_dropout(x)
        out = (out @ self.lora_A.T) * self.lora_G
        out = out @ self.lora_B.T
        out = out * self.scaling
        return out