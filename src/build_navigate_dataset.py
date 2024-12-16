# Import read_json from wikidata.util
from wikidata.utils import load_json, write_json
# Import defaultdict from collections
from collections import defaultdict
import random
from build_two_hop_training_splits import write_jsonl_file

# set DATA_DIR to the path of the data directory: /n/holyscratch01/kempner_lab/Everyone/data/twohop-1/nav
DATA_DIR = '/n/holyscratch01/kempner_lab/Everyone/data/twohop-1/nav'

# read 'task.json' in DATA_DIR
task = load_json(DATA_DIR + '/task.json')

data = task['examples']
description = task['description']
print(f'description: {description}')
quit()

def extract_output(d):
    return 'True' if d['target_scores']['True'] else 'False'

# format each example in data
for d in data:
    d['instruction'] = description + '\n' + d['input']
    d['output'] = extract_output(d)
    d['input'] = ""
    # remove all other keys
    for k in list(d.keys()):
        if k not in ['instruction', 'output', 'n_sentences', 'input']:
            d.pop(k)

# group data by 'n_sentences'
grouped_data = defaultdict(list)
for d in data:
    grouped_data[d['n_sentences']].append(d)

# shuffle each grouped_data list
for k in grouped_data:
    random.shuffle(grouped_data[k])

# split each grouped_data list into train, and test
train_data = []
test_data = []
for k in grouped_data:
    split = int(len(grouped_data[k]) * 0.8)
    train_data += grouped_data[k][:split]
    test_data += grouped_data[k][split:]

# save test data to 'test.json' in DATA_DIR
write_jsonl_file(DATA_DIR + '/test.jsonl', test_data)

# save train data to 'train.json' in DATA_DIR
write_jsonl_file(DATA_DIR + '/train.jsonl', train_data)

# add 'input' to each train_data example
for d in train_data:
    d['input'] = "Let's think step by step."

# save train data to 'train_prompt.json' in DATA_DIR
write_jsonl_file(DATA_DIR + '/train_prompt.jsonl', train_data)

# set BBH to the path of the bbh directory: /n/home01/nsaphra/workplace/Big-Bench-Hard/cot-prompts
BBH = '/n/home01/nsaphra/workplace/BIG-Bench-Hard/cot-prompts'

with open(BBH + '/navigate.txt', 'r') as f:
    lines = f.readlines()

# remove first line in lines
lines = lines[1:]

# concatenate lines into one string
cot_examples = ''.join(lines)

# add cot_examples to the front of instruction for each training point
for d in train_data:
    d['instruction'] = cot_examples + '\n' + d['instruction']

# save train data to 'train_prompt_cot.json' in DATA_DIR
write_jsonl_file(DATA_DIR + '/train_prompt_cot.jsonl', train_data)