from transformers import AutoModelForCausalLM,AutoTokenizer
import yaml
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb
wandb.init(mode='disabled')

def load_model_and_tokenizer(model_name,tokenizer_name,device_map:str='auto'):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,trust_remote_code=True)

    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'

    return model, tokenizer

def load_configs(config_path):
    with open(config_path, "r") as f:
        configs = yaml.load(f,Loader=yaml.FullLoader)
    return configs

def train_model(model,tokenizer,configs):
    train_dataset = load_from_disk(configs['train_dataset_path'])
    test_dataset = load_from_disk(configs['test_dataset_path'])
    
    training_arguments = TrainingArguments(
        output_dir = configs["output_dir"],
        num_train_epochs = configs['num_train_epochs'],
        logging_steps=50,
        evaluation_strategy='steps',
        per_device_train_batch_size = configs['per_device_train_batch_size'],
        gradient_accumulation_steps = configs['gradient_accumulation_steps'],
        optim = configs['optim'],
        learning_rate = configs['learning_rate'],
        weight_decay = configs['weight_decay'],
        fp16 = configs['fp16'],
        max_grad_norm = configs['max_grad_norm'],
        warmup_ratio = configs['warmup_ratio'],
        group_by_length = configs['group_by_length'],
        lr_scheduler_type = configs['lr_scheduler_type'],
        save_safetensors=True,
        seed=42,
        save_strategy='steps',
        do_eval=True,
        )
    train_dataset = train_dataset.shuffle().select(range(10000))
    eval_dataset = test_dataset.shuffle().select(range(500))
    
    trainer = SFTTrainer(
        args = training_arguments,
        model= model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        tokenizer = tokenizer,
        packing = False,
        max_seq_length=configs['max_seq_length']
        )
    trainer.train()
    trainer.model.save_pretrained(configs['save_model_at'])

if __name__ == "__main__":
    config = load_configs('gpt2_config.yaml')
    model,tokenizer = load_model_and_tokenizer(config['saved_model_name'],config['saved_tokenizer_name'])

    train_model(model,tokenizer,config)