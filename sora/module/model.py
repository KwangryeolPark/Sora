"""
Author: Kwangryeol Park
E-mail: pkr7098@unist.ac.kr
"""

from typing import Optional, Union, List

import torch.nn as nn
from opendelta.basemodel import DeltaBase

from .config import SoraConfig
from .layer import SoraLinear

class SoraModel(DeltaBase):
    config_class = SoraConfig
    delta_type = "sora"
    default_modified_modules = [
        'attn@.q@',
        'attn@.v@',
        'attn@.k@',
        'attn@.proj@',
        'ff@.w1@',
        'ff@.w2@'
    ]
    _supported_backends = ['hf', 'bmt']
    _need_pseudo_data = False
    
    def __init__(
        self,
        backbone_model: nn.Module,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout:float = 0.0,
        modified_modules: Optional[List[str]] = None,
        unfrozen_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        common_structure: Optional[bool] = None,
        interactive_modify: Optional[Union[bool, int]] = False,
    ):        
        DeltaBase.__init__(
            self,
            backbone_model,
            modified_modules=modified_modules,
            unfrozen_modules=unfrozen_modules,
            exclude_modules=exclude_modules,
            common_structure=common_structure,
            interactive_modify=interactive_modify
        )
        
        # Make all arguments of init to self.arguments.
        self.backbone_model = backbone_model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
            
        self.delta_modules = nn.ModuleList()
        self.add_all_delta_to_backbone(
            self.backbone_model,
            self.modified_modules
        )
        
    def update_module(
        self,
        module: nn.Module,
        key: str
    ) -> None:
        parent_ref, child_name, child_ref = self.find_module(module, key)
        
        parallel_module = self.new_module_like(child_module=child_ref)
        self.insert_parallel_module(child_ref, delta_module=parallel_module, delta_name="sora")
    
    def _pseudo_data_to_instantiate(self, backbone: nn.Module):
        # No need to pass pseudo input, so overwrite it
        pass

    def new_module_like(self, child_module: nn.Module):
        if isinstance(child_module, nn.Linear):
            in_features, out_features = child_module.in_features, child_module.out_features
            new_module = SoraLinear(
                in_features=in_features,
                out_features=out_features,
                weight=child_module.weight,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout
            )
            self.delta_modules.append(new_module)
        else:
            raise NotImplementedError
        return new_module