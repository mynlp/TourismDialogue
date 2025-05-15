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


def input_prompt(text, tokenizer):
    prompt = (f'対話の記録：{text}　{tokenizer.sep_token} 対話の記録によると、彼らが話題にしていた場所は\
    「方面 {tokenizer.mask_token}」\
    「県 {tokenizer.mask_token}」\
    「サブエリア {tokenizer.mask_token}」\
    「位置情報コピー {tokenizer.mask_token}」です。\
    また、話していた活動は、\
    大ジャンルでは「{tokenizer.mask_token}」、\
    中ジャンルでは「{tokenizer.mask_token}」、\
    小ジャンルでは「{tokenizer.mask_token}」とのことです。\
    さらに、\
    旅行の特徴条件は「{tokenizer.mask_token}」で、\
    キーワードは「{tokenizer.mask_token}」とされています。')
    return prompt

def output_prompt2(answers):
    prompt_template = ('方面：{}；県：{}；サブエリア：{}；位置情報コピー：{}；大ジャンル：{}；中ジャンル：{}；小ジャンル：{}；\
                       旅行の特徴条件：{}；キーワード：{}')
    prompt=prompt_template.format(
        answers['方面'],
        answers['県'],
        answers['サブエリア'],
        answers['位置情報コピー'],
        answers['大ジャンル'],
        answers['中ジャンル'],
        answers['小ジャンル'],
        answers['特徴条件'],
        answers['キーワード']
    )
    return prompt

def output_prompt(answers):
    prompt = ''
    for k,v in answers.items():
        prompt += f'{k}：{v}；'
    return prompt

def prepare_dataset2(MODEL, given_path):
    if 'mbart' in MODEL:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang='ja_XX', tgt_lang='en_XX')
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    path = dict()
    path['train'] = 'dataset/{}/train.json'.format(given_path)
    path['valid'] = 'dataset/{}/dev.json'.format(given_path)
    path['test'] = 'dataset/{}/test.json'.format(given_path)

    data = dict()
    for target in path:
        data[target] = {'input': [], 'output': []}
        with open(path[target], 'r') as f:
            insts = json.load(f)
        for inst in insts:
            src = input_prompt(inst['src'].strip(), tokenizer)
            data[target]['input'].append(src)
            label= {k:'なし' for k in keys}
            if inst['label'] is not None:
                for k in inst['label'].keys():
                    label[k] = inst['label'][k]
            tgt = output_prompt(label)
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



    def tokenize(element):
        outputs = tokenizer(
            element['input'],
            text_target = element['output'],
            truncation=True,
            padding=True, #'max_length',
            max_length=1024,
        )
        return outputs

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    return raw_datasets, tokenized_datasets


def prepare_dataset(MODEL, given_path):
    if 'mbart' in MODEL:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang='ja_XX', tgt_lang='en_XX')
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    path = dict()
    path['train'] = 'dataset/{}/train.json'.format(given_path)
    path['valid'] = 'dataset/{}/dev.json'.format(given_path)
    path['test'] = 'dataset/{}/test.json'.format(given_path)

    data = dict()
    for target in path:
        data[target] = {'input': [], 'output': []}
        with open(path[target], 'r') as f:
            insts = json.load(f)
        for inst in insts:
            src = input_prompt(inst['src'].strip(), tokenizer)
            data[target]['input'].append(src)
            label = inst['label']
            #if inst['label'] is not None:
            #    for k in inst['label'].keys():
            #        label[k] = inst['label'][k]
            tgt = output_prompt(label)
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

    def tokenize(element):
        outputs = tokenizer(
            element['input'],
            text_target=element['output'],
            truncation=True,
            padding=True,  # 'max_length',
            max_length=1024,
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
        tokenized_datasets['train'], collate_fn=data_collator, batch_size=1
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
    gw = open('{}/gold_es.txt'.format(given_path), 'w', encoding='utf-8')
    model = AutoModelForSeq2SeqLM.from_pretrained(given_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(given_path)

    test = raw_datasets['test']
    logger.info('generating output for test data')
    for i in tqdm(range(len(test))):
        #input_ids = tokenizer(test[i]['input'], return_tensors="pt").to(device)
        #outputs = model.generate(**input_ids, generation_config=generation_config)
        answers = generate_key_value_pairs(model,tokenizer,test[i]['input'],keys,value_set)
        fw.write(str(answers))
        fw.write('\n')
        gw.write(str(test[i]['output']))
        gw.write('\n')
    fw.close()
    gw.close()
    return 1

def generate_key_value_pairs(model, tokenizer, src, keys, value_set):
    # Tokenize the input text `src` for the encoder
    encoder_inputs = tokenizer(src, return_tensors='pt').to(device)

    # Initialize the decoder input as an empty string
    decoder_input = ""

    # Initialize dictionary to store key-value pairs
    key_value_pairs = {}

    # Iterate over each key in the key list
    for key in keys:
        # Get the candidate values for the current key
        candidates = value_set[key]

        # Prepare a batch of decoder inputs for all candidate values
        candidate_sequences = [f"{decoder_input}{key}：{candidate}；" for candidate in candidates]
        decoder_inputs = tokenizer(candidate_sequences, return_tensors='pt', padding=True, truncation=True).to(device)

        # Forward pass: compute the logits for the batch of candidate sequences
        with torch.no_grad():
            outputs = model(
                input_ids=encoder_inputs['input_ids'].expand(len(candidates), -1),
                attention_mask=encoder_inputs['attention_mask'].expand(len(candidates), -1),
                decoder_input_ids=decoder_inputs['input_ids'],
                decoder_attention_mask=decoder_inputs['attention_mask']
            )
            logits = outputs.logits.cpu()  # Shape: (batch_size, sequence_length, vocab_size)

        # Compute the likelihood scores for all candidate sequences in the batch
        likelihoods = compute_likelihood_batch(logits, decoder_inputs['input_ids'].cpu(), tokenizer)

        # Find the candidate with the highest score
        best_index = torch.argmax(likelihoods).item()
        best_candidate = candidates[best_index]

        # Update the decoder input with the best key-value pair
        decoder_input += f"{key}：{best_candidate}；"

        # Store the key-value pair
        key_value_pairs[key] = best_candidate

    return key_value_pairs

def compute_likelihood_batch(logits, input_ids, tokenizer):
    """
    Compute the likelihood of the input sequences given the logits in a batch.
    """
    # Shift logits and input_ids to align for likelihood computation
    shift_logits = logits[:, :-1, :].contiguous()  # Remove last logit (no prediction for it)
    shift_labels = input_ids[:, 1:].contiguous()   # Remove first token (no target for it)

    # Compute the log-likelihood for each token
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    log_likelihood = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Sum the log-likelihoods to get the total likelihood of each sequence in the batch
    total_likelihood = log_likelihood.sum(dim=-1)

    return total_likelihood

def main(given_path):
    MODEL = 'facebook/mbart-large-50'
    #MODEL = 'mt5-base'
    #MODEL = 'bigscience/mt0-large'
    #MODEL='ku-nlp/bart-base-japanese'

    raw_datasets, tokenized_datasets = prepare_dataset(MODEL, given_path)
    train(MODEL, given_path, tokenized_datasets)
    gen(given_path, raw_datasets)



if __name__ == '__main__':
    with open('value_set.json','r') as f:
        value_set = json.load(f)
    keys = list(value_set.keys())
    main('mbart_large_wz3')
    #main('tourism')



