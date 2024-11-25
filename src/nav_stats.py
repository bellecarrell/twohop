from build_two_hop_training_splits import read_jsonl_file

data = read_jsonl_file('/n/holyscratch01/kempner_lab/Everyone/data/twohop-1/nav/train.jsonl')
print(f'len of data: {len(data)}')
print(f'data 0: {data[0]}')


# Count n_sentences in data
n_sentences = {}
for d in data:
    n = d['n_sentences']
    if n not in n_sentences:
        n_sentences[n] = 0
    n_sentences[n] += 1

print(f'n_sentences: {n_sentences}')

# Print each type of instruction sentence
instructions = {}
n_steps = {}
ff = 0
for d in data:
    instruction  = d['instruction']
    if 'Always face forward' in instruction:
        #print(f'instruction: {instruction}')
        ff += 1

    # remove question 
    instruction = instruction.split('?')[1]

    # remove options
    instruction = instruction.split('Options:')[0]

    instruction = instruction.split('.')


    for i in instruction:
        i = i.strip()
        if i.startswith('Take'):
            s = i.split('Take ')[1].split(' ')[0]
            if s not in n_steps:
                n_steps[s] = 0
            n_steps[s] += 1
            if i.endswith('steps') or i.endswith('step'):
                i = 'Take steps'
            elif i.endswith('steps backward'):
                i = 'Take steps backward'
            elif i.endswith('steps forward'):
                i = 'Take steps forward'
            elif i.endswith('steps left'):
                i = 'Take steps left'
            elif i.endswith('steps right'):
                i = 'Take steps right'
        elif i.startswith('Turn'):
            if i.endswith('left'):
                i = 'Turn left'
            elif i.endswith('right'):
                i = 'Turn right'
        else:
            pass
        if i not in instructions:
            instructions[i] = 0
        instructions[i] += 1


true = 0
for d in data:
    if 'Yes' in d['output']:
        true += 1

print(f'true: {true}')
print(f'false: {len(data) - true}')

# for d in data:
#     if 'Yes' in d['output'] and (d['n_sentences'] == 4):
#         true.append(d['instruction'].strip('Q: If you follow these instructions, do you return to the starting point?'))

# # sort based on whether Always face forward is in the instruction
# true = sorted(true)

# for t in true:
#     print(t)

# write true to text file
# with open('/n/home01/acarrell/workplace/twohop-1/src/true_4.txt', 'w') as f:
#     for t in true:
#         f.write(t + '\n')

# print(f'instructions: {instructions}')
# print(f'ff: {ff}')
# print(f'n ff {len(data) - ff}')
# # sort n_steps by key
# n_steps = dict(sorted(n_steps.items()))
# print(f'n_steps: {n_steps}')