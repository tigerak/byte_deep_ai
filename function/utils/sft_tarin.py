import os
# huggingface
from transformers import Trainer, TrainingArguments
# modules
from function.models.model_cfg import model_choice
from function.utils.sft_PP import sft_PP_run
from function.utils.sft_dataset import SFT_Dataset, DataCollatorForSupervisedDataset


class SFT_train():
    def __init__(self, model_name):
        self.pp = sft_PP_run(model_name)
        self.tokenizer = self.pp._prepare_tokenizer()

        self.CFG = model_choice[model_name]

    def _prepare_dataset(self):
        csv_files = []
        for root, dir, files in os.walk(self.CFG['DATASET_DIR']):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        if len(csv_files) != 1:
            raise Exception(f"CSV file을 많이 찾음: {csv_files}")
        else:
            print(f"학습에 사용하는 CSV file: {csv_files}")
        
        train_dataset = SFT_Dataset(csv_path=csv_files[0],
                                    tokenizer=self.tokenizer)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

        return train_dataset, data_collator

    def train(self, date, time, add_training=False):
        print(f"학습 시작 : {date} {time}")
        # Model
        peft_model, is_add_training = self.pp._prepare_lora_model(add_training)
        print(f"추가 학습인가? : {add_training}")
        if add_training:
            print(f"원본 모델 : {self.CFG['PEFT_MODEL']}")
        peft_model.train()
        peft_model.config.use_cache = False
        # Data
        train_dataset, data_collator = self._prepare_dataset()
        # train_dataloader = DataLoader(train_dataset,
        #                               batch_size=1,
        #                               shuffle=True,
        #                               collate_fn=data_collator,
        #                               pin_memory=False)
        # Train
        args = TrainingArguments(
                        num_train_epochs = 120,
                        per_device_train_batch_size = 2,
                        gradient_accumulation_steps = 64,
                        gradient_checkpointing =True,

                        weight_decay=0.001,
                        optim=self.CFG['OPTIM'],
                        learning_rate=self.CFG['LEARNING_RATE'],
                        # warmup_steps = 4,
                        lr_scheduler_type= "cosine",
                        # bf16=True, 
                        fp16=True,

                        logging_strategy= "steps",
                        logging_steps=1,
                        logging_dir=f"{self.CFG['OUTPUT_DIR']}/{date}/{time}",

                        save_strategy="steps",
                        save_steps = 40,
                        save_total_limit=5,
                        save_safetensors = True,
                        output_dir = f"{self.CFG['OUTPUT_DIR']}/{date}/{time}",
                        logging_nan_inf_filter = False, 

                        resume_from_checkpoint=self.CFG['PEFT_MODEL'] if is_add_training else False, # 중단한 학습을 계속 할 경우 사용
                        )   
        peft_trainer = Trainer(model=peft_model,
                               args=args,
                               train_dataset=train_dataset,
                               data_collator=data_collator,)
        
        peft_trainer.train()

        # # 로그 플러시
        # sys.stdout.flush()