import json
import os
import torch
import wandb

from MagicTools import PadSequence, TrainUtils
from transformers import AutoTokenizer, BartForConditionalGeneration, AutoModelForSeq2SeqLM

import logging
import argparse

wandb.login(key='your key here!')
device = torch.device('cuda')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s')
logger = logging.getLogger(__name__)

def input_prompt_instruct(text, tokenizer):
    prompt='Generate Japanese entities from the given sentence: {}\nAnswer:'.format(text)
    return prompt


def input_prompt_instruct2(text, tokenizer):
    prompt='次の文から日本語のエンティティを生成する：{}\n回答：'.format(text)
    return prompt



def output_prompt(answers):
    if answers is None:
        return 'なし:なし;'
    key_values = ''.join([f'{k}:{v};' for k, v in answers.items()])
    prompt = key_values
    return prompt


def compute_score(records):
    prec=0.0
    reca=0.0
    for record in records:
        label = set([kv for kv in record['output_text'].split(';') if kv.strip()])
        pred = set([kv for kv in record['predict'].split(';') if kv.strip()])
        inter = label.intersection(pred)
        prec += len(inter) / len(pred)
        reca += len(inter) / len(label)
    prec = prec / len(records)
    reca = reca / len(records)
    return prec, reca


def get_best_candidate(model, tokenizer, encoder_inputs, decoder_input_texts,
                       batch_size=16):
    decoder_start_token_id = (
        model.config.decoder_start_token_id
        if model.config.decoder_start_token_id is not None
        else tokenizer.bos_token_id
    )

    batches = [decoder_input_texts[i:i + batch_size] for i in range(0, len(decoder_input_texts), batch_size)]
    likelihoods = []
    for batch in batches:
        batch_input_ids = tokenizer(batch, padding=True, return_tensors='pt',
                                    add_special_tokens=False)['input_ids']

        batch_attention_mask = tokenizer(batch, padding=True, return_tensors='pt',
                                         add_special_tokens=False)['attention_mask']
        decoder_input_ids = torch.cat(
            [torch.zeros(batch_input_ids.size(0), 1).fill_(decoder_start_token_id) if decoder_start_token_id is not None else torch.zeros(0),
             torch.zeros(batch_input_ids.size(0), 1).fill_(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else torch.zeros(0),
             batch_input_ids], dim=1).long().to(device)
        # decoder_input_ids=batch_input_ids.to(device)

        decoder_attention_mask = torch.cat(
            [torch.ones(batch_input_ids.size(0), 1) if decoder_start_token_id is not None else torch.ones(0),
             torch.ones(batch_input_ids.size(0), 1) if tokenizer.bos_token_id is not None else torch.ones(0),
             batch_attention_mask], dim=1).to(device)
        # decoder_attention_mask = batch_attention_mask.to(device)
        with torch.no_grad():
            outputs = model(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                input_ids=encoder_inputs.repeat(batch_input_ids.size(0), 1))
        logits = outputs.logits
        likelihoods.append(compute_likelihood_batch(logits, decoder_input_ids, tokenizer))
    likelihoods = torch.cat(likelihoods, dim=0)
    best_index = torch.argmax(likelihoods).item()
    # print(likelihoods)
    return best_index


def generate_key_value_pairs(model, tokenizer, batch, value_set):
    # Tokenize the input text `src` for the encoder
    # encoder_inputs = tokenizer(batch, return_tensors='pt').to(device)
    colon_mark = ':'
    semi_mark = ';'

    # encoder_outputs = model.get_encoder()(
    #    input_ids=batch['input_ids'],
    #    attention_mask=batch['attention_mask'],
    #    return_dict=True
    # )
    # print(encoder_outputs)
    encoder_inputs = batch['input_ids'].to(device)
    # Iterate over each key in the key list
    generated_key_vals = []
    for i in range(10):
        generated_tokens = f''.join([f'{k}{colon_mark}{v}{semi_mark}' for k, v in generated_key_vals])

        candidates = list(value_set.keys()) + [tokenizer.eos_token]
        decoder_input_texts = [generated_tokens + f'{k}' for k in candidates]
        print(decoder_input_texts)
        best_index = get_best_candidate(model, tokenizer, encoder_inputs, decoder_input_texts)
        best_key = candidates[best_index]
        if best_key == tokenizer.eos_token:
            break
        candidates = value_set[best_key]
        decoder_input_texts = [generated_tokens + f'{best_key}{colon_mark}{v}' for v in candidates]
        best_index = get_best_candidate(model, tokenizer, encoder_inputs, decoder_input_texts)
        best_val = candidates[best_index]
        generated_key_vals.append((best_key, best_val))
    return dict(generated_key_vals), ''.join([f'{k}:{v};' for k,v in generated_key_vals])


def compute_likelihood_batch(logits, input_ids, tokenizer):
    """
    Compute the likelihood of the input sequences given the logits in a batch.
    """
    # Shift logits and input_ids to align for likelihood computation
    shift_logits = logits[:, :-1, :].contiguous()  # Remove last logit (no prediction for it)
    shift_labels = input_ids[:, 1:].contiguous()  # Remove first token (no target for it)

    # Compute the log-likelihood for each token
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    # log_probs=shift_logits
    log_probs[shift_labels == tokenizer.pad_token_id] = 0
    log_likelihood = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    seq_lens = (shift_labels != tokenizer.pad_token_id).sum(dim=-1)
    # print(seq_lens)
    # Sum the log-likelihoods to get the total likelihood of each sequence in the batch
    total_likelihood = log_likelihood.sum(dim=-1) #/ seq_lens

    return total_likelihood



class MyTrainUtils(TrainUtils):
    def __init__(self, mode):
        super().__init__(mode=mode)
        self.value_set = None
        self.mode=mode
        if self.mode=='test':
            with open(self.config.key_val_file,'r') as f:
                value_set = json.load(f)
                value_set['なし']=['なし']
            self.value_set = value_set


    def inference(self, model, tokenizer, batch, do_sample):
        if self.mode=='train':
            loss = self.loss_function(model, batch)
            records = []
            for index in batch['index']:
                record = {
                    'index': index,
                    'loss': loss.item()
                }
                records.append(record)
            return records
        else:
            if self.config.constraint_decode:
                key_val_dict, key_val_text = generate_key_value_pairs(model.module, tokenizer, batch, self.value_set)
            else:
                key_val_text = tokenizer.decode(
                    model.module.generate(input_ids=batch['input_ids'].to(device), temperature = 0.7, max_new_tokens = 512,
                                          repetition_penalty = 1.5)[0],
                    skip_special_tokens=True)
                key_val_text = key_val_text.replace(' ','')
            records = []
            for index in batch['index']:
                record = {
                    'index': index,
                    'predict': key_val_text
                }
                records.append(record)
            return records


    def process_outs(self, tokenizer, outs):
        return outs


    def loss_function(self, model, batch):
        #print(batch['decoder_input_ids'].size())
        loss = model(
            input_ids=batch['input_ids'].to(model.device),
            attention_mask=batch['attention_mask'].to(model.device),
            labels=batch['decoder_input_ids'].to(model.device)
        ).loss
        return loss


    def compute_score(self, records):
        if self.mode =='train':
            avg_loss = sum([r['loss'] for r in records]) / len(records)
            return avg_loss
        else:
            prec,reca = compute_score(records)
            wandb.log({
                'precision': prec,
                'recall': reca
            })
            os.makedirs(os.path.join(self.config.log_dir,self.config.eval_out),exist_ok=True)
            with open(os.path.join(self.config.log_dir,self.config.eval_out, 'full_score.json'),'w') as f:
                json.dump({'precision':prec,'recall':reca},f)
            return prec

    def construct_instance(self, data, tokenizer, is_train, is_chinese):
        insts = []
        for record in data:
            if any([kw in self.config.model_name for kw in ['bart']]):
                input_text = input_prompt_instruct2(record['src'], tokenizer)
                logger.info('==>>> using mask prompt...')
            else:
                input_text = input_prompt_instruct(record['src'], tokenizer)
                logger.info('==>>> using instruction prompt...')
            output_text = output_prompt(record['label'])
            insts.append({
                'input_text': input_text,
                'output_text': output_text,
                'index': len(insts)
            })
            #print('input text:', input_text)
            #print('output text:', output_text)
        return insts

    def process_inputs(self, tokenizer, instance, is_train):
        inputs = tokenizer(instance['input_text'], truncation=True, add_special_tokens=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        decoder_inputs = tokenizer(instance['output_text'], truncation=True, add_special_tokens=True)
        decoder_input_ids = decoder_inputs['input_ids']
        decoder_attention_mask = decoder_inputs['attention_mask']
        index = instance['index']
        instance = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'index': index
        }
        return instance

    def collate_fn(self, batch):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, indexs = [list() for i in range(5)]
        for inst in batch:
            input_ids.append(inst['input_ids'])
            attention_mask.append(inst['attention_mask'])
            decoder_input_ids.append(inst['decoder_input_ids'])
            decoder_attention_mask.append(inst['decoder_attention_mask'])
            indexs.append(inst['index'])

        max_len = max([len(x) for x in input_ids])
        input_ids = PadSequence(input_ids, max_len=max_len, pad_token_id=self.tokenizer.pad_token_id)
        attention_mask = PadSequence(attention_mask, max_len=max_len, pad_token_id=0)
        decoder_max_len = max([len(x) for x in decoder_input_ids])
        decoder_input_ids = PadSequence(decoder_input_ids, max_len=decoder_max_len,
                                        pad_token_id=self.tokenizer.pad_token_id)
        decoder_attention_mask = PadSequence(decoder_attention_mask, max_len=decoder_max_len, pad_token_id=0)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
        decoder_input_ids[decoder_input_ids == self.tokenizer.pad_token_id] = -100
        decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)
        #print(input_ids.size(),attention_mask.size(),decoder_input_ids.size(),decoder_attention_mask.size())
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'decoder_input_ids': decoder_input_ids,
                 'decoder_attention_mask': decoder_attention_mask, 'index': indexs}
        return batch

    def get_train_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_file', type=str, required=True)
        parser.add_argument('--dev_file', type=str, required=True)
        parser.add_argument('--model_name', type=str, required=True)
        parser.add_argument('--cache_dir', type=str, default='./cache')
        parser.add_argument('--use_cache', action='store_true')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--adam_epsilon', type=float, default=1e-5)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--seed', type=int, default=123)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--warmup_rate', type=float, default=0.1)
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--only_load_model', action='store_true')
        parser.add_argument('--log_dir', type=str, required=True)
        parser.add_argument('--project_name', type=str, default='MagicToolsTest')
        parser.add_argument('--run_name', type=str, default='SST2')
        parser.add_argument('--optimize_direction', type=str, default='max')
        parser.add_argument('--accumulated_size', type=int, default=1)
        parser.add_argument('--init_eval_score', type=float, default=0.0)
        parser.add_argument('--is_chinese', action='store_true')
        parser.add_argument('--epoch_based', action='store_true')
        parser.add_argument('--save_strategy', choices=['epoch','step'], default='epoch')
        args = parser.parse_args()
        return args

    def get_test_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--test_file', type=str, required=True)
        parser.add_argument('--key_val_file', type=str, required=True)
        parser.add_argument('--model_name', type=str, required=True)
        parser.add_argument('--cache_dir', type=str, default='./cache')
        parser.add_argument('--use_cache', action='store_true')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--seed', type=int, default=123)
        parser.add_argument('--only_load_model', action='store_true')
        parser.add_argument('--log_dir', type=str, required=True)
        parser.add_argument('--project_name', type=str, default='MagicToolsTest')
        parser.add_argument('--run_name', type=str, default='SST2')
        parser.add_argument('--is_chinese', action='store_true')
        parser.add_argument('--epoch_based', action='store_true')
        parser.add_argument('--eval_out', type=str, default='eval')
        parser.add_argument('--constraint_decode', action='store_true')

        args = parser.parse_args()
        return args

    def get_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
        return model

    def get_tokenizer(self):
        if 'mbart' in self.config.model_name:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, src_lang='ja_XX', tgt_lang='ja_XX')
            logger.info('==>>> loading tokenizer for mbart model...')
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if 'bart' in self.config.model_name:
                tokenizer.add_tokens([':',';'])
                self.model.resize_token_embeddings(len(tokenizer))
        return tokenizer
