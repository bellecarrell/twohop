from build_two_hop_training_splits import read_jsonl_file, write_jsonl_file

# read /n/holyscratch01/kempner_lab/Everyone/data/twohop-1/ent_10k_dataset/archive/processed/splits/test.jsonl
test_facts = read_jsonl_file('/n/holyscratch01/kempner_lab/Everyone/data/twohop-1/ent_10k_dataset/archive/processed/splits/test.jsonl')

# add "input" key to each dictionary in test_facts containing "Let's think step by step."
for i, fact in enumerate(test_facts):
    fact['input'] = "Let's think step by step."

    if i % 1000 == 0:
        print(f'{i} facts processed')

# write test facts to test-dummy-cot.jsonl in /n/holyscratch01/kempner_lab/Everyone/data/twohop-1/ent_10k_dataset/archive/processed/splits
write_jsonl_file('/n/holyscratch01/kempner_lab/Everyone/data/twohop-1/ent_10k_dataset/archive/processed/splits/test-dummy-cot.jsonl', test_facts)

