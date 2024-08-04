import os
import sys
import csv

import tensorrt
# import torch2trt
import torch_tensorrt
# torch
import torch
from torch.utils.data import DataLoader
import torch.utils.checkpoint as checkpoint
import tensorrt
# huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (LoraConfig, PeftModel, AutoPeftModelForCausalLM,
                  prepare_model_for_kbit_training , get_peft_model)
from accelerate import dispatch_model, init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
# modules
from function.models.model_cfg import model_choice
from function.utils.sft_dataset import SFT_Dataset, DataCollatorForSupervisedDataset



os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
class sft_PP_run():
    def __init__(self, model_name):
        self.CFG = model_choice[model_name]

        self.tokenizer = self._prepare_tokenizer()

        
    def _prepare_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
                                        self.CFG['MODEL_SAVE_DIR'],
                                        padding_side="left",
                                        bos_token='</s>', 
                                        eos_token='</s>', 
                                        unk_token='</s>', 
                                        pad_token='</s>',
                                        # add_eos_token=True,
                                        # add_bos_token=True,
                                        )
        # tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _prepare_quantization_model(self):
        device_map = self._set_memory(33, 47)
        max_memory={0: "24GiB", 1: "24GiB"}
        # bnb_config = BitsAndBytesConfig(
        #                         load_in_4bit=True,
        #                         bnb_4bit_use_double_quant=True,
        #                         bnb_4bit_compute_dtype=torch.bfloat16, #if your gpu supports it 
        #                         bnb_4bit_quant_type = "nf4",
        #                         )
        bnb_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_threshold=6.,
                                llm_int8_skip_modules=["lm_head"],
                                # load_in_8bit_fp32_cpu_offload=True,
                                # llm_int8_has_fp16_weight=True,
                                )
        base_model = AutoModelForCausalLM.from_pretrained(
                                            self.CFG['MODEL_SAVE_DIR'],
                                            # torch_dtype=torch.bfloat16,
                                            torch_dtype=torch.float16,
                                            attn_implementation="flash_attention_2",
                                            quantization_config=bnb_config,
                                            max_memory=max_memory,
                                            device_map=device_map
                                            )
        
        base_model.gradient_checkpointing_enable() #this to checkpoint grads 
        base_model = prepare_model_for_kbit_training(base_model) #quantising the model (due to compute limits)
        
        return base_model
        
    def _prepare_lora_model(self, add_training=False):
        base_model = self._prepare_quantization_model()
        peft_config = LoraConfig(r=32,
                                 lora_alpha=64,
                                 lora_dropout=0.1, 
                                 bias="none",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj", "lm_head"],
                                 task_type="CAUSAL_LM")
                                 
        if add_training:
            peft_model = PeftModel.from_pretrained(base_model,
                                                self.CFG['PEFT_MODEL'],
                                                is_trainable=add_training)
            is_add_training = True
        else:
            peft_model = get_peft_model(model=base_model, 
                                    peft_config=peft_config) # get_peft_model 새로운 PEFT 모델을 만들 때 사용
            is_add_training = False
        self._print_parameters(peft_model)
        return peft_model, is_add_training
    

    def _prepare_inference_model(self):
        base_model = self._prepare_quantization_model()
        inf_model = PeftModel.from_pretrained(base_model,
                                              self.CFG['PEFT_MODEL'])
        
        inf_model.config.use_cache = True
        inf_model.gradient_checkpointing_disable()
        inf_model.eval()
        return inf_model
    
    
    def _prepare_rt_model(self):
        # from optimum
        device_map = self._set_memory(27, 47)
        max_memory={0: "24GiB", 1: "24GiB"}
        bnb_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                # llm_int8_threshold=6.,
                                llm_int8_skip_modules=["lm_head"],
                                # load_in_8bit_fp32_cpu_offload=True
                                )
        merge_model = AutoModelForCausalLM.from_pretrained(
                                            self.CFG['MERGE_MODEL'],
                                            # torch_dtype=torch.bfloat16,
                                            torch_dtype=torch.half,
                                            # attn_implementation="flash_attention_2",
                                            # quantization_config=bnb_config,
                                            max_memory=max_memory,
                                            device_map=device_map,
                                            # device_map={"": "cpu"}
        )
        
        # torch._dynamo.disable()

        
        # def forward(self, input_ids):
        #     if input_ids.dtype != torch.long:
        #         input_ids = input_ids.long()
        #     return self.model(input_ids)

        # Prepare Model
        # torch_script_module = torch.jit.script(self.inf_model)
        # enabled_precisions={torch.float, torch.half}
        # trt_ts_module = torch_tensorrt.compile(
        #                                 self.inf_model, 
        #                                 inputs=inputs, 
        #                                 enabled_precisions=enabled_precisions
        # )
        # torch.jit.save(torch_script_module, "torch_script_module.ts")
        ### 해볼 것 !###
        # inputs = [
        #     torch_tensorrt.Input(
        #         min_shape=[1, 2200],
        #         opt_shape=[1, 2700],
        #         max_shape=[1, 3107],
        #         dtype=torch.half,
        #     )
        # ]
        enabled_precisions = {torch.half}
        compile_spec = {
            "inputs": [
                torch_tensorrt.Input(
                    min_shape=(1, 2200),
                    opt_shape=(1, 2700),
                    max_shape=(1, 3100),
                    dtype=torch.half,
                )
            ],
            "enabled_precisions": enabled_precisions,
            "ir": "dynamo",
        }
        dummy_inputs = [torch.randn((1, 3107)).to("cuda").half()]
        optimized_model = torch_tensorrt.compile(
            merge_model,
            ir="torch_compile",
            inputs=compile_spec,
            enabled_precisions={torch.float, torch.half},
            debug=True,
            workspace_size=48 << 30, # 48GB
            min_block_size=1 << 20, # 1MB
            torch_executed_ops={},
        )
        print("compile 끝")
        return optimized_model

    
    def _print_parameters(self, model):
        trainable_param = 0
        total_params = 0
        for name , param in model.named_parameters():
            # print(name)
            # print(f"{name} is on device {param.device}")
            total_params += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
                
        print(f"Total params : {total_params} , trainable_params : {trainable_param} , trainable % : {100 * trainable_param / total_params} ")

    def _set_memory(self, divide_layer, last_layer):
        device_map = {
            'model': 0,
            'model.embed_tokens': 0,
            'model.layers': 0,

            'model.norm': 1,
            'model.rotary_emb': 1,
            'lm_head': 1
        }
        for i in range(last_layer+1):
            if i < divide_layer:
                n = 0
            else :
                n = 1
            update_dict = {    
                    f"model.layers.{i}": n,
                    f"model.layers.{i}.self_attn": n,
                    f"model.layers.{i}.self_attn.q_proj": n,
                    f"model.layers.{i}.self_attn.k_proj": n,
                    f"model.layers.{i}.self_attn.v_proj": n,
                    f"model.layers.{i}.self_attn.o_proj": n,
                    f"model.layers.{i}.self_attn.rotary_emb": n,
                    f"model.layers.{i}.mlp": n,
                    f"model.layers.{i}.mlp.gate_proj": n,
                    f"model.layers.{i}.mlp.up_proj": n,
                    f"model.layers.{i}.mlp.down_proj": n,
                    f"model.layers.{i}.mlp.act_fn": n,
                    f"model.layers.{i}.input_layernorm": n,
                    f"model.layers.{i}.post_attention_layernorm": n
            }
            device_map.update(update_dict)

        return device_map

    # def _set_memory(self, divide_layer, last_layer):
    #     device_map = {
    #         f"base_model.model.model.embed_tokens.weight": "cpu",
    #         f"base_model.model.model.norm.weight": 1,
    #         f"base_model.model.lm_head.base_layer.weight": 1,
    #         f"base_model.model.lm_head.lora_A.default.weight": 1,
    #         f"base_model.model.lm_head.lora_B.default.weight": 1,
    #     }
    #     for i in range(last_layer+1):
    #         if i < divide_layer:
    #             n = 0
    #         else :
    #             n = 1
    #         update_dict = {    
    #                 f"base_model.model.model.layers.{i}.self_attn.q_proj.base_layer.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.q_proj.base_layer.SCB": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.q_proj.base_layer.weight_format": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.k_proj.base_layer.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.k_proj.base_layer.SCB": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.k_proj.base_layer.weight_format": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_B.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.v_proj.base_layer.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.v_proj.base_layer.SCB": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.v_proj.base_layer.weight_format": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.o_proj.base_layer.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.o_proj.base_layer.SCB": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.o_proj.base_layer.weight_format": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_B.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.gate_proj.base_layer.weight": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.gate_proj.base_layer.SCB": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.gate_proj.base_layer.weight_format": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.gate_proj.lora_A.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.gate_proj.lora_B.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.up_proj.base_layer.weight": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.up_proj.base_layer.SCB": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.up_proj.base_layer.weight_format": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.up_proj.lora_A.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.up_proj.lora_B.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.down_proj.base_layer.weight": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.down_proj.base_layer.SCB": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.down_proj.base_layer.weight_format": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.down_proj.lora_A.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.mlp.down_proj.lora_B.default.weight": n, 
    #                 f"base_model.model.model.layers.{i}.input_layernorm.weight": n, 
    #                 f"base_model.model.model.layers.{i}.post_attention_layernorm.weight": n, 
    #         }
    #         device_map.update(update_dict)

    #     return device_map