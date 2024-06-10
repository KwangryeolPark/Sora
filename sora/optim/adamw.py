import math
import warnings
from typing import Iterable, Tuple, Union, List

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        lambda_sparse: float = 0.1,
        lambda_schedule: Union[List, str] = None,
        lambda_max: int = None,
        lambda_num: int = None,
        multiply_lr: bool = False
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
            
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        
        self.lambda_sparse = lambda_sparse
        self.lambda_idx = 0
        self.lambda_schedule = lambda_schedule
        self.multiply_lr = multiply_lr
        self._build_lambda_list(lambda_max, lambda_num)
    
    def _build_lambda_list(
        self,
        lambda_max,
        lambda_num
    ):
        if self.lambda_shedule is None:
            self._lambdas = None
            return
        if isinstance(self.lambda_schedule, list):
            self._lambdas = self.lambda_schedule
        else:
            assert lambda_max is not None and  lambda_num is not None
            if self.lambda_schedule == "linear":
                self._lambdas = np.linspace(self.lambda_sparse, lambda_max, lambda_num)
            elif self.lambda_schedule == "log_linear":
                self._lambdas = np.log(np.linspace(np.exp(self.lambda_sparse), np.exp(lambda_max), lambda_num))
            elif self.lambda_schedule == "exp_linear":
                self._lambdas = np.exp(np.linspace(np.log(self.lambda_sparse)), np.log(lambda_max), lambda_num)
            else:
                raise NotImplementedError
        
    def step_lambda(self):
        if self._lambdas is None:
            return
        else:
            if self.lambda_idx < len(self._lambdas) - 1:
                self.lambda_idx += 1
                self.lambda_sparse = self._lambdas[self.lambda_idx]
            else:
                pass
    
    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                # Original AdamW
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients, please consider SparseAdam instead")
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                step_size = group['lr']
                if group['correct_bias']:
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                to_add = torch.div(exp_avg, denom) * (-step_size)
                if group['weight_decay'] > 0.0:
                    to_add = to_add + (-group['lr'] * group['weight_decay']) * p.data
                p.data.add_(to_add)
                
                # Sparse Update
                if self.lambda_sparse > 0.0:
                    multiply = group['lr'] if self.multiply_lr else 1.0
                    p.data[p.data > self.lambda_sparse * multiply] -= (self.lambda_sparse * multiply)
                    p.data[p.data < -self.lambda_sparse * multiply] += (self.lambda_sparse * multiply)
                    p.data[abs(p.data) < self.lambda_sparse * multiply] = 0.0
        
        return loss