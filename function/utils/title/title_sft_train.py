import os
# huggingface
from transformers import Trainer, TrainingArguments
# modules
from function.models.model_cfg import model_choice
from function.utils.sft_PP import sft_PP_run
from function.utils.title.title_sft_dataset import Title_SFT_Dataset, DataCollatorForSupervisedDataset


class Title_SFT_train():
    def __init__(self, model_name):
        self.pp = sft_PP_run(model_name)
        self.tokenizer = self.pp._prepare_tokenizer()

        self.CFG = model_choice[model_name]


    def train(self, date, time, add_training=False):
        print(f"학습 시작 : {date} {time}")
        # Model
        peft_model, is_add_training = self.pp._prepare_lora_model(add_training)
        print(f"추가 학습인가? : {add_training}")
        if add_training:
            print(f"원본 모델 : {self.CFG['TITLE_PEFT_MODEL']}")
        peft_model.train()
        peft_model.config.use_cache = False
        # Data
        train_dataset, data_collator = self._prepare_dataset()
        # Train
        args = TrainingArguments(
                        num_train_epochs = 120,
                        per_device_train_batch_size = 4,
                        gradient_accumulation_steps = 32,
                        gradient_checkpointing =True,

                        weight_decay=0.001,
                        optim=self.CFG['OPTIM'],
                        learning_rate=self.CFG['LEARNING_RATE'],
                        lr_scheduler_type= "cosine",
                        fp16=True,

                        logging_strategy= "steps",
                        logging_steps=1,
                        logging_dir=f"{self.CFG['TITLE_OUTPUT_DIR']}/{date}/{time}",

                        save_strategy="steps",
                        save_steps = 40,
                        save_total_limit=5,
                        save_safetensors = True,
                        output_dir = f"{self.CFG['TITLE_OUTPUT_DIR']}/{date}/{time}",
                        logging_nan_inf_filter = False, 

                        resume_from_checkpoint=self.CFG['TITLE_PEFT_MODEL'] if is_add_training else False, # 중단한 학습을 계속 할 경우 사용
                        )   
        peft_trainer = Trainer(model=peft_model,
                               args=args,
                               train_dataset=train_dataset,
                               data_collator=data_collator,)
        
        peft_trainer.train()

        
    def _prepare_dataset(self):
        print(f"학습에 사용하는 CSV file: {self.CFG['TITLE_DATA_PATH']}")
        
        train_dataset = Title_SFT_Dataset(json_path=self.CFG['TITLE_DATA_PATH'],
                                          tokenizer=self.tokenizer)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

        return train_dataset, data_collator