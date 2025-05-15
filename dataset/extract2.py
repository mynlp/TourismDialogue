import json
import os
import random
from collections import defaultdict,Counter

def generate_training_instances(annotations, window_size):
    training_instances = []

    for i in range(len(annotations)):
        # Determine the start and end indices for the context window
        start = max(0, i - window_size)
        end = min(len(annotations), i + window_size + 1)

        # Extract the context annotations
        context_annotations = annotations[start:end]

        # Combine consecutive utterances from the same speaker
        combined_utterances = []

        for ann in context_annotations:
            current_speaker = '操作員' if ann["speaker"]=='operator' else '顧客'
            current_utterance = ann["utterance"]
            combined_utterances.append(f"{current_speaker}: {current_utterance.strip()}")

        # Join the combined utterances into a single training instance
        training_instance = "\n".join(combined_utterances)

        # Extract all query dictionaries from the I-th annotation
        queries = [ann_dict["query"] for ann_dict in annotations[i]["annotation"] if ann_dict["query"] is not None]

        # Decide how to handle multiple queries (e.g., concatenate, choose the first one, etc.)
        # Here, we concatenate all queries into a single dictionary
        label = {}
        for query in queries:
            label.update(query)

        # If no queries are found, set the label to None
        if not label:
            label = None

        # Add the training instance and label to the list
        if label is not None:
            training_instances.append({'src':training_instance, 'label':label})
        else:
            training_instances.append({'src':training_instance,'label':None})

    return training_instances




def balance_records(records, split, balance_data):
    # Separate records into two lists based on the label
    print(f'{split} key statistics before balance:{Counter(sum([list(r["label"].keys()) for r in records if r["label"] is not None],[]))}')
    label_dict = defaultdict(list)
    none_records=[]
    for record in records:
        if record['label'] is not None:
            for k in record['label'].keys():
                label_dict[k].append(record)
        else:
            none_records.append(record)

    max_num = max([len(vs) for vs in label_dict.values()])
    if split == 'train' and balance_data:
        for k in label_dict.keys():
            label_dict[k]=label_dict[k]+random.choices(label_dict[k], k=max_num-len(label_dict[k]))
        balanced_records = sum(label_dict.values(), [])
    else:
        balanced_records = [r for r in records if r['label'] is not None]
    balanced_records+=random.choices(none_records, k=max_num)
    random.shuffle(balanced_records)
    print(f'{split} key statistics after balance:{Counter(sum([list(r["label"].keys()) for r in balanced_records if r["label"] is not None],[]))}')

    return balanced_records


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--split-info", type=str, required=True)
    parser.add_argument("--annotation-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--balance-data", action="store_true")
    args = parser.parse_args()
    with open(args.split_info, "r") as f:
        split_info = json.load(f)
    for split in split_info.keys():
        insts = []
        for filename in os.listdir(args.annotation_dir):
            if filename in split_info[split]:
                with open(os.path.join(args.annotation_dir, filename), 'r') as f:
                    annos = json.load(f)
                insts += generate_training_instances(annos, args.window_size)
        insts = balance_records(insts, split, args.balance_data)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"{split}.json"), 'w') as f:
            json.dump(insts,f,indent=2)


