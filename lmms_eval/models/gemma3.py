from io import BytesIO
from copy import deepcopy
import os
import base64
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
import requests as url_requests
import time
import logging

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils
import torch
from accelerate import Accelerator, DistributedType
from transformers import AutoTokenizer, Gemma3ForCausalLM
from tqdm import tqdm


@register_model("gemma3")
class Gemma3(lmms):
    def __init__(
        self,
        pretrained: str = "google/gemma-3-1b-pt",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Load model with bfloat16 as recommended in the Gemma 3 documentation
        self.model = Gemma3ForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        
        # Use AutoTokenizer instead of AutoProcessor for text-only cases
        # This follows the recommended pattern in the Gemma 3 documentation
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
            self.accelerator = accelerator

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = []
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm([reg.args for reg in requests]):
            # Process image data if available
            if doc_to_visual and hasattr(self, 'task_dict') and task in self.task_dict and split in self.task_dict[task] and doc_id in self.task_dict[task][split]:
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                
                # For multimodal input, we need a processor that can handle images
                # This would require importing AutoProcessor but only when needed
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(self.model.config._name_or_path)
                
                # Process text and image inputs using the processor
                model_inputs = processor(text=contexts, images=visuals[0][0], return_tensors="pt")
                
                # Move inputs to the device
                for key in model_inputs:
                    model_inputs[key] = model_inputs[key].to(self.model.device)
                
                input_len = model_inputs["input_ids"].shape[-1]
                
                # Generate text
                with torch.inference_mode():
                    generation = self.model.generate(
                        **model_inputs, 
                        max_new_tokens=gen_kwargs.get("max_new_tokens", 100), 
                        do_sample=gen_kwargs.get("do_sample", False)
                    )
                    generation = generation[0][input_len:]
                    decoded = processor.decode(generation, skip_special_tokens=True)
            else:
                # Text-only generation following Gemma 3 documentation
                model_inputs = self.tokenizer(contexts, return_tensors="pt").to(self.model.device)
                input_len = model_inputs["input_ids"].shape[-1]
                
                # Generate text
                with torch.inference_mode():
                    generation = self.model.generate(
                        **model_inputs, 
                        max_new_tokens=gen_kwargs.get("max_new_tokens", 100), 
                        do_sample=gen_kwargs.get("do_sample", False)
                    )
                    generation = generation[0][input_len:]
                    decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
            
            res.append(decoded)
        return res
    
    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError
