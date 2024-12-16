from build_two_hop_training_splits import read_jsonl_file, write_jsonl_file

# set DATA_DIR to /n/holyscratch01/kempner_lab/Everyone/data/twohop-1/processed/splits
DATA_DIR = '/n/holyscratch01/kempner_lab/Everyone/data/twohop-1/ent_10k_dataset/archive/processed/splits'
# read train_40000-two-hop.jsonl from DATA_DIR
train_facts = read_jsonl_file(DATA_DIR + '/train_40000_two-hop.jsonl')

# add "input" key to each dictionary in train_facts containing "Let's think step by step."
for i, fact in enumerate(train_facts):
    fact['input'] = "Let's think step by step."

    if i % 1000 == 0:
        print(f'{i} facts processed')

# write train facts to train_40000_two_hop-dummy-cot.jsonl in DATA_DIR
write_jsonl_file(DATA_DIR + '/train_40000_two-hop-dummy-cot.jsonl', train_facts)