"""
Author: Kwangryeol Park
E-mail: pkr7098@unist.ac.kr
"""

from dataclasses import dataclass

from opendelta.utils.signature import get_arg_names_inside_func
from opendelta import BaseDeltaConfig

@dataclass
class SoraArguments:
    r: int = 8,
    lora_alpha: int = 16
    lora_dropout: float = 0.0

class SoraConfig(BaseDeltaConfig):
    def __init__(
        self,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name):
                setattr(self, arg_name, locals()[arg_name])