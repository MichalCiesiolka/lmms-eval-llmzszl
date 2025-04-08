from io import BytesIO
from copy import deepcopy
import os
import base64
from typing import List, Tuple, Union, Optional, Any, Dict
from tqdm import tqdm
import requests as url_requests
import time
import logging
import json
from pathlib import Path

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils
import torch
from accelerate import Accelerator, DistributedType
from vllm import LLM
from vllm.sampling_params import SamplingParams
from PIL import Image


@register_model("pixtral")
class Pixtral(lmms):
    def __init__(
        self,
        pretrained: str = "mistralai/Pixtral-12B-2409",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        tokenizer_mode: str = "mistral",
        max_tokens: int = 100,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Parse any remaining kwargs
        self.kwargs = kwargs
        
        # Initialize vLLM model
        self.llm = LLM(
            model=pretrained,
            tokenizer_mode=tokenizer_mode,
            dtype=torch.float16,
            limit_mm_per_prompt={"image_url": 4}
        )
        
        self.max_tokens = max_tokens
        self._device = device
        self._rank = 0
        self._world_size = 1
        
        # Setup for distributed computing if needed
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            if accelerator.is_local_main_process:
                logging.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        
        self.accelerator = accelerator
        
    def set_task_dict(self, task_dict: Dict):
        """Set task_dict to be used by model during evaluation"""
        self.task_dict = task_dict

    def _convert_image_to_data_url(self, image_path):
        """Convert image path to data URL format for vLLM."""
        if image_path.startswith("http"):
            return {"url": image_path}
        
        # For local files, convert to base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = "image/jpeg"  # Default to JPEG, could be made smarter
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"
        
        return {"url": f"data:{mime_type};base64,{encoded}"}

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood calculation not implemented for Pixtral with vLLM")

    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = []
        sampling_params = SamplingParams(max_tokens=self.max_tokens, temperature=0)
        
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm([reg.args for reg in requests]):
            # Initialize chat message structure
            chat = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": contexts}
                    ]
                }
            ]
            
            # Process image data if available
            if doc_to_visual and hasattr(self, 'task_dict') and task in self.task_dict and split in self.task_dict[task] and doc_id in self.task_dict[task][split]:
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                
                # Add images to the chat message
                for visual in visuals[0]:
                    if isinstance(visual, str):  # Assuming it's a file path
                        image_data = self._convert_image_to_data_url(visual)
                        chat[0]["content"].append({
                            "type": "image_url",
                            "image_url": image_data
                        })
                    elif isinstance(visual, Image.Image):
                        # Convert PIL image to bytes
                        buffered = BytesIO()
                        visual.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        chat[0]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                        })
            
            # Update sampling parameters from gen_kwargs
            if gen_kwargs:
                custom_params = {}
                if "max_new_tokens" in gen_kwargs:
                    custom_params["max_tokens"] = gen_kwargs["max_new_tokens"]
                if "do_sample" in gen_kwargs and gen_kwargs["do_sample"]:
                    custom_params["use_beam_search"] = False
                if "temperature" in gen_kwargs:
                    custom_params["temperature"] = gen_kwargs["temperature"]
                
                sampling_params = SamplingParams(**custom_params)
            
            # Generate response using vLLM
            outputs = self.llm.chat(messages=chat, sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text
            
            res.append(output_text)
        
        return res
    
    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError("Multi-round generation not implemented for Pixtral with vLLM") 