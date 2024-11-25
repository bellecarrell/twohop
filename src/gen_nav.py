from build_two_hop_training_splits import read_jsonl_file, write_jsonl_file
from nav import Location

data = read_jsonl_file('/n/holyscratch01/kempner_lab/Everyone/data/twohop-1/nav/train.jsonl')
write_jsonl_file('/n/netscratch/kempner_sham_lab/Lab/acarrell/twohop/nav/train.jsonl', data)
print(f'data 0: {data[0]}')
print(f'len of data: {len(data)}')

# Count n_sentences in data
n_sentences = {}
for d in data:
    n = d['n_sentences']
    if n not in n_sentences:
        n_sentences[n] = 0
    n_sentences[n] += 1

print(f'n_sentences: {n_sentences}')
input_prefix = 'Let\'s think step by step.\n We start at the origin (0, 0), facing the positive y-axis.'

for j, d in enumerate(data):
    location = Location()
    instruction  = d['instruction']

    # remove question 
    instruction = instruction.split('?')[1]

    # remove options
    instruction = instruction.split('Options:')[0]

    instruction = instruction.split('.')
    print(f'instruction: {instruction}')
    d_input = ''

    for idx, i in enumerate(instruction):
        if i.strip() == '':
            continue
        location.update(i)
        if not d_input:
            d_input = f'{input_prefix}\n'
        d_input += f'({idx + 1}) {i}: {location}\n'
    
    if location.at_origin():
        d_input += 'Since (0, 0) is (0, 0), we are indeed where we started. So the answer is Yes.\n'
    else:
        d_input += f'Since ({location.x}, {location.y}) is not (0, 0), we are not where we started. So the answer is No.\n'
    d['input'] = d_input

write_jsonl_file('/n/netscratch/kempner_sham_lab/Lab/acarrell/twohop/nav/train_cot.jsonl', data)
