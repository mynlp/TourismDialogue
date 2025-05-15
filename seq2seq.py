import datasets, transformers
from datasets import Dataset, DatasetDict
from transformers import GenerationConfig, AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler, DataCollatorForSeq2Seq
from tqdm import tqdm
import torch
import math, os, json, sys, logging
from torch.utils.data import DataLoader
import numpy as np

os.environ["WANDB_DISABLED"] = "True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logger = logging.getLogger("transformers")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = logging.DEBUG
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

def prepare_dataset_origin(MODEL, given_path):
    path = dict()
    path['train'] = 'dataset/{}/train_'.format(given_path)
    path['valid'] = 'dataset/{}/dev_'.format(given_path)
    path['test'] = 'dataset/{}/test_'.format(given_path)

    data = dict()
    for target in path:
        data[target] = {'input': [], 'output': []}
        for line in open(path[target] + 'src.txt', encoding='utf-8'):
            if not line.strip():
                continue
            src = 'Generate Japanese entities from the given sentence: {}\nAnswer:'.format(line.strip())
            data[target]['input'].append(src)
        for line in open(path[target] + 'tgt.txt', encoding='utf-8'):
            if not line.strip():
                continue
            tgt = '{}'.format(line.strip().strip())
            data[target]['output'].append(tgt)
    
    train = Dataset.from_dict(data['train'], split = 'train')
    valid = Dataset.from_dict(data['valid'], split = 'validation')
    test = Dataset.from_dict(data['test'], split = 'test')

    raw_datasets = DatasetDict(
        {
            'train': train,
            'valid': valid,
            'test': test
        }
    )

    train.to_json('./raw.json', force_ascii=False)

    if 'mbart' in MODEL:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang='ja_XX', tgt_lang='en_XX')
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element['input'],
            text_target = element['output'],
            truncation=True,
            padding=True, #'max_length',
            max_length=512,
        )
        return outputs

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    return raw_datasets, tokenized_datasets


def prepare_dataset(MODEL, given_path):
    path = dict()
    path['train'] = 'dataset/{}/train.json'.format(given_path)
    path['valid'] = 'dataset/{}/dev.json'.format(given_path)
    path['test'] = 'dataset/{}/test.json'.format(given_path)

    data = dict()
    for target in path:
        data[target] = {'input': [], 'output': []}
        with open(path[target],'r') as f:
            insts =json.load(f)
        for inst in insts:
            src = 'Generate Japanese entities from the given sentence: {}\nAnswer:'.format(inst['src'])
            data[target]['input'].append(src)
            tgt = ' '.join([f'{k}:{v}' for k,v in inst['label'].items()])
            data[target]['output'].append(tgt)

    train = Dataset.from_dict(data['train'], split='train')
    valid = Dataset.from_dict(data['valid'], split='validation')
    test = Dataset.from_dict(data['test'], split='test')

    raw_datasets = DatasetDict(
        {
            'train': train,
            'valid': valid,
            'test': test
        }
    )

    train.to_json('./raw.json', force_ascii=False)

    if 'mbart' in MODEL:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang='ja_XX', tgt_lang='en_XX')
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element['input'],
            text_target=element['output'],
            truncation=True,
            padding=True,  # 'max_length',
            max_length=512,
        )
        return outputs

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    return raw_datasets, tokenized_datasets


def train(MODEL, given_path, tokenized_datasets):
    if 'mbart' in MODEL:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang='ja_XX', tgt_lang='en_XX')
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)

    tokenized_datasets['train'].to_json('./tokenized.json', force_ascii=False)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    train_dataloader = DataLoader(
        tokenized_datasets['train'], collate_fn=data_collator, batch_size=4
    )
    dev_dataloader = DataLoader(
        tokenized_datasets['valid'], collate_fn=data_collator, batch_size=1
    )

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1.5e-5)
    num_train_epochs = 20
    max_train_steps = num_train_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=max_train_steps * 1,
    )

    num_update_steps_per_epoch = len(train_dataloader)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(tokenized_datasets['train'])}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {1}")
    logger.info(f"  Gradient Accumulation steps = {1}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    running_loss = 0.
    best_dev_loss = 99.
    patience = 0

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            #logger.info(batch)
            batch = batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            completed_steps += 1
            if completed_steps >= max_train_steps:
                break

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                logger.info('\nstep {} loss: {}\n'.format(i + 1, last_loss))
                running_loss = 0.

        model.eval()
        losses = []
        for _, batch in enumerate(dev_dataloader):
            batch = batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            #logger.info('\n')
            #logger.info(tokenizer.decode(labels, skip_special_tokens=True))
            #logger.info('\n')
            loss = outputs.loss
            losses.append(loss.repeat(1))

        losses = torch.cat(losses)
        try:
            dev_loss = torch.mean(losses)
            perplexity = math.exp(dev_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info('\n')
        logger.info(f"epoch {epoch}: perplexity: {perplexity} dev_loss: {dev_loss}")
        logger.info('\n')

        if best_dev_loss < dev_loss:
            patience += 1
        else:
            best_dev_loss = dev_loss
            patience = 0

        if patience == 3:
            logger.info("--- early stopping ! ---")
            break

    if not os.path.exists(given_path):
        os.mkdir(given_path)
    with open(os.path.join(given_path, "all_results.json"), "w") as f:
        json.dump({"perplexity": perplexity}, f)    
    tokenizer.save_pretrained(given_path)
    model.save_pretrained(given_path)

def gen(given_path, raw_datasets):
    fw = open('{}/pred_es.txt'.format(given_path), 'w', encoding='utf-8')
    model = AutoModelForSeq2SeqLM.from_pretrained(given_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(given_path)
    generation_config = GenerationConfig.from_pretrained(given_path, 
            temperature = 0.7, 
            max_new_tokens = 512,
            repetition_penalty = 1.5,
            #encoder_repetition_penalty = 1.5,
            #no_repeat_ngram_size = 10,
            ) 
    test = raw_datasets['test']
    logger.info('generating output for test data')
    for i in tqdm(range(len(test))):
        input_ids = tokenizer(test[i]['input'], return_tensors="pt").to(device)
        outputs = model.generate(**input_ids, generation_config=generation_config)
        fw.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
        fw.write('\n')
    fw.close()
    return 1

def main(given_path, out_path):
    #MODEL = 'facebook/mbart-large-50'
    MODEL = 'google/mt5-base'
    #MODEL = 'bigscience/mt0-large'

    raw_datasets, tokenized_datasets = prepare_dataset(MODEL, given_path)
    train(MODEL, out_path, tokenized_datasets)
    gen(out_path, raw_datasets)
    
if __name__ == '__main__':
    main('proc_data_wz3','exps/mt5-base-wz3')
    #main('tourism')



