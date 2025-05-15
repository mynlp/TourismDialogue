import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,GenerationConfig
import json,torch
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open('./../value_set2.json','r') as f:
    value_set = json.load(f)
    value_set['なし']=['なし']

def input_prompt_instruct(text):
    prompt = ('次の会話履歴に記載されているエンティティタイプとエンティティ値を特定してください。'
              f'会話履歴:{text}。'
              'エンティティタイプ: 方面、県、サブエリア、位置情報コピー、大ジャンル、中ジャンル、小ジャンル、特徴条件は、キーワード。')
    return prompt

def get_model(model_name,model_file):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_file,map_location=device)['model'])
    print('finish loading model file...')
    return model, tokenizer#,generation_config

def generate(batch, model, tokenizer):
    batch = [input_prompt_instruct(sample) for sample in batch]
    batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    batch = batch.to(device)
    output1 = generate_key_value_pairs(model,tokenizer,batch, value_set=value_set)
    output2 = tokenizer.decode(model.generate(**batch,do_sample=False)[0],skip_special_tokens=True)
    outputs = {'output1':output1, 'output2':output2}
    return outputs


def get_context(dialogue, line_idx):
    wz = 7
    context = dialogue[max(line_idx-wz,0):line_idx+wz+1]
    context = '\n'.join(['{}：{}'.format('操作員' if turn["speaker"]=='operator' else '顧客',turn['utterance'])
                         for turn in context])
    return context



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
    return dict(generated_key_vals)


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


model_file = "/home/u00483/repos/TourismDialogue/browser/mt5-base/tourism"
batch=['本日はお問い合わせ頂き、ありがとうございます。 はい。えーとこの度一日よろしくお願い致します。 お願い致します。はい。 はい。スーッ。えーとではお客様えーと今回はー、えーっとご旅行のご相談ということで、お間違いないでしょうか？ はい。 はい。かしこまりました。 えー、それではー、えーっと…。 もう行きたいー、その場所だったりー、えー地方だったり県とかってもう、決まってらっしゃいますか？ はい。えー…、沖縄～…ですね。ええ。']