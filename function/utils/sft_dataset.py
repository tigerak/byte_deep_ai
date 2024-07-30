import csv
import json
from tqdm import tqdm
import copy
# torch
import torch
from torch.utils.data import Dataset
#modules
from function.prompts.prompt import Prompt
from function.prompts.summ_prompt import Summary_Prompt
from function.prompts.tag_prompt import Tag_Prompt

class SFT_Dataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        data_list = []
        with open(csv_path, newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data_list.append(row)

        self.tokenizer = tokenizer
        self.prompt = Prompt()
        self._prompt_complete(data_list)


    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx],
                    labels=self.labels[idx])
    

    def _prompt_complete(self, data_list):
        source_prompts = []
        total_prompts = []
        for data in tqdm(data_list):
            # Article 길이 조절 2500자
            if len(data['article']) > 2500:
                data['article'] = data['article'][:2500]
            # 회사 태그 전처리
            main, sub = self._company_tag_preprocessing(data['company_tag'])

            # Source Prompt
            source_prompt = self.prompt.make_source(
                                            article_date=data['article_date'],
                                            title=data['title'],
                                            article=data['article']
            )
            source_prompts.append(source_prompt)
            # Target Prompt
            target_prompt = self.prompt.make_target(
                                            methood=data['modified_reason'],
                                            summary=data['modified_summary'],
                                            summary_title=data['modified_summary_title'],
                                            main_company=main,
                                            sub_company=sub,
                                            major=data['primary_tag'],
                                            medium=data['secondary_tag']
            )
            # Total Prompt
            total_prompt = source_prompt + target_prompt + self.tokenizer.eos_token
            total_prompts.append(total_prompt)
        # Tokenization
        source_tokenized = self._tokeniz_fn(source_prompts)
        total_tokenized = self._tokeniz_fn(total_prompts)
        # label의 source 부분 masking
        input_ids = total_tokenized['input_ids']
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, source_tokenized['input_ids_len']):
            label[:source_len] = -100
        # 완성
        data_dict = dict(input_ids=input_ids,
                         labels=labels)
        
        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']

        # 최대 토큰 수 계산
        print(f"최대 토근 수 : {max(total_tokenized['input_ids_len'])}")
        

    
    def _company_tag_preprocessing(self, company_tag):
        tag_list = json.loads(company_tag)
        result = {
            'main': [],
            'sub': []
        }
        main = []
        sub = []
        if tag_list != None:
            for entry in tag_list:
                for key, value in entry.items():
                    if value is not None and key != "relation":
                        # '-'의 뒷부분을 키로 사용
                        _, suffix = key.split('-')
                        if suffix in result:
                            if value not in result[suffix]:
                                result[suffix].append(value)
        main = ', '.join(result['main'])
        sub = ', '.join(result['sub'])
        return main, sub
    
    
    def _tokeniz_fn(self, prompts_list):
        tokenized_list = [
                self.tokenizer(
                            prompt,
                            return_tensors="pt",
                ) for prompt in prompts_list
        ]
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_len = [
            tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(input_ids=input_ids,
                    input_ids_len=input_ids_len)


class DataCollatorForSupervisedDataset(object): 
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value= -100)
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )