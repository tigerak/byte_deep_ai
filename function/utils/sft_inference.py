
from time import time
# TensorRT
import torch_tensorrt
# torch
import torch
# huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM
# modules
from function.utils.sft_PP import sft_PP_run
from function.prompts.prompt import Prompt
from function.prompts.summ_prompt import Summary_Prompt

class SFT_inference():
    def __init__(self, model_name):
        pp = sft_PP_run(model_name)
        # self.inf_model = pp._prepare_inference_model()
        self.tokenizer = pp._prepare_tokenizer()
        # self.inf_model = AutoModelForCausalLM.from_pretrained(r'/home/deep_ai/Project/merged_model_directory',
        #                                                       local_files_only=True)
        # self.inf_model = pp._prepare_merge_model()
        self.inf_model = pp._prepare_onnx_model()

    def inference(self, data):
        start_time = time()
        # Prepare Input Data
        data = data.to_dict()
        input = self._data_preprocessing(data=data)
        # Inference
        with torch.no_grad():
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast():
                generation_args = dict(   
                                    num_beams=6,
                                    temperature=1.5,
                                    top_p=1.5,
                                    do_sample=True,
                                    max_new_tokens=400,
                                    # top_k=50,
                                    early_stopping=True
                                )
                output = self.inf_model.generate(**input, 
                                            **generation_args)
                
                result_text = self.tokenizer.decode(output[0],
                                            skip_special_tokens=True)
                torch.cuda.empty_cache() 
                
        print(result_text)
        end_time = time()
        processing_time = round((end_time - start_time), 3)
        processing_time = str(processing_time) + "초"
        print(processing_time)

        result_dict = self._output_dict(result_text, processing_time)
        return result_dict

    def _output_dict(self, result_text, processing_time):
        def extract_text(raw, delimiter):
            index = raw.find(delimiter)
            if index != -1:
                return raw[index + len(delimiter):].strip(), raw[:index].strip()
            return '', raw.strip()

        tags = {
            'medium_class': "중분류(Medium Classification):",
            'major_class': "대분류(Major Classification):",
            'sub': "sub: ",
            'main': "### 분류 태그(Classification Tag)\n기업 태그(Company Tag):",
            'summary_title': "### 요약문 제목(Summary Title):",
            'summary': "### 요약문(Summary):",
            'summary_reason': "### 요약문  구성 방법(Method of Summary Composition):"
        }

        parsed_texts = {}
        for key, delimiter in tags.items():
            parsed_texts[key], result_text = extract_text(result_text, delimiter)

        summary_count = str(len(parsed_texts['summary']))

        result_dict = {
            'summary_reason': parsed_texts['summary_reason'],
            'summary': parsed_texts['summary'],
            'summary_title': parsed_texts['summary_title'],
            'main': parsed_texts['main'],
            'sub': parsed_texts['sub'],
            'major_class': parsed_texts['major_class'],
            'medium_class': parsed_texts['medium_class'],
            'summary_count': summary_count,
            'processing_time': processing_time
        }

        return result_dict
    
    def _data_preprocessing(self, data):
        if len(data['articleDiv']) > 2500:
            print(len(data['articleDiv']))
            data['articleDiv']=data['articleDiv'][:2500]

        prompt = Prompt()
        query = prompt.make_source(article_date=data['dateDiv'],
                                   title=data['titleDiv'],
                                   article=data['articleDiv'])
        input_ids = self.tokenizer.encode(query, 
                                    return_tensors='pt', 
                                    return_token_type_ids=False)
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        input = {'input_ids':input_ids,
                 'attention_mask':attention_mask}
        
        return input

    def merge(self):
        #### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
        from peft import AutoPeftModelForCausalLM
        
        # Load PEFT model on CPU
        model = AutoPeftModelForCausalLM.from_pretrained(
            '/home/deep_ai/Project/data/output/sft/24_07_27/solar_recovery_1st/checkpoint-280',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        )
        # Merge LoRA and base model and save
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained('/home/deep_ai/Project/data/output/sft/merged_model',
                                     safe_serialization=True, 
                                     max_shard_size="2GB")
        print(f"병합 끝")

    def make_onnx(self):
        # Prepare Input Data
        from data.sft_test_data.sft_test import data
        if len(data['articleDiv']) > 2500:
            print(len(data['articleDiv']))
            data['articleDiv']=data['articleDiv'][:2500]

        prompt = Prompt()
        query = prompt.make_source(article_date=data['dateDiv'],
                                   title=data['titleDiv'],
                                   article=data['articleDiv'])
        input_ids = self.tokenizer.encode(query, 
                                    return_tensors='pt', 
                                    return_token_type_ids=False)
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        input_ids = input_ids.to('cpu')
        attention_mask = attention_mask.to('cpu')
        input = {'input_ids':input_ids,
                 'attention_mask':attention_mask}

        # Prepare Model
        traced_model = torch.jit.trace(self.inf_model, input['input_ids'])
        torch.jit.save(traced_model, "/home/deep_ai/Project/data/output/sft/traced_solar_model.pt")


        # optimized_model = torch_tensorrt.dynamo.compile(
        #                         self.inf_model,
        #                         inputs=input,
        #                         enabled_precisions={torch.float16},
        #                         debug=True,
        #                         workspace_size=46,
        #                         min_block_size=3,
        #                         torch_executed_ops={})

        print("끝")