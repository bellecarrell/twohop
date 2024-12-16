import pandas as pd
import fire
import os

from build_two_hop_training_splits import read_jsonl_file, write_jsonl_file

def thousand_test(split_path='/n/holyscratch01/kempner_lab/Everyone/data/twohop-1/ent_10k_dataset/archive/processed/splits', size=40000):
    # For each split, load train file
    for split in ['two-hop','two-hop-5perc-cot','two-hop-one-hop','one-hop','two-hop-cot']:
        train_path = os.path.join(split_path, f'train_{size}_{split}.jsonl')
        train_facts = read_jsonl_file(train_path)
        print(f'Len of train facts for {split} is {len(train_facts)}')
        # Save 1000 examples
        train_facts = train_facts[:1000]
        train_path = os.path.join(split_path, f'train_1000_{split}.jsonl')
        write_jsonl_file(train_path, train_facts)

if __name__ == '__main__':
    fire.Fire(thousand_test)