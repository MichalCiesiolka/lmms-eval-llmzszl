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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
from PIL import Image


@register_model("pixtral")
class Pixtral(lmms):
    def __init__(
        self,
        pretrained: str = "mistralai/Pixtral-12B-Base-2409",
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

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        
        # Initialize processor for handling both text and images
        self.processor = AutoProcessor.from_pretrained(pretrained)
        
        # Also initialize tokenizer for text-only scenarios
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
                
                # Process text and image inputs using the processor
                # For Pixtral, we need to process images and text together
                model_inputs = self.processor(
                    text=contexts, 
                    images=visuals[0][0], 
                    return_tensors="pt"
                )
                
                # Move inputs to the device
                for key in model_inputs:
                    model_inputs[key] = model_inputs[key].to(self.model.device)
                
                input_len = model_inputs["input_ids"].shape[-1]
                
                # Generate text
                with torch.inference_mode():
                    generation = self.model.generate(
                        **model_inputs, 
                        max_new_tokens=gen_kwargs.get("max_new_tokens", 100), 
                        do_sample=gen_kwargs.get("do_sample", False),
                        temperature=gen_kwargs.get("temperature", 0)
                    )
                    generation = generation[0][input_len:]
                    decoded = self.processor.decode(generation, skip_special_tokens=True)
            else:
                # Text-only generation
                model_inputs = self.tokenizer(contexts, return_tensors="pt").to(self.model.device)
                input_len = model_inputs["input_ids"].shape[-1]
                
                # Generate text
                with torch.inference_mode():
                    generation = self.model.generate(
                        **model_inputs, 
                        max_new_tokens=gen_kwargs.get("max_new_tokens", 100), 
                        do_sample=gen_kwargs.get("do_sample", False),
                        temperature=gen_kwargs.get("temperature", 0)
                    )
                    generation = generation[0][input_len:]
                    decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
            
            res.append(decoded)
        return res
    
    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError 