from query import Query
import argparse
import os
import random
from query import Query
from wikidata.utils import load_json
from build_one_hop_dataset import to_file
import glob
import json
from relation import Relation
from query import Query, TwoHopQuery

def write_jsonl_file(fname, data):
    with open(fname, 'w+') as file:
        for entry in data:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')

def read_jsonl_file(fname):
    data = []
    # Reading from a JSON Lines file
    with open(fname, 'r') as file:
        for line in file:
            data_entry = json.loads(line)
            # Add each entry to data
            data.append(data_entry)
    return data

def process_train_facts(facts, test=False, cot=False, ent_only=False, partial=0.0):
    train_facts = []
    for i, f in enumerate(facts):
        if i % 100 == 0:
            print(f'Processing fact {i}')
        f['instruction'] = str(f['prompt'])
        # TODO: shuffle answers first?
        if test:
            f['output'] = str(f['answers'][0]['value'])
            answers = list(f['answers'])
            alist = []
            if len(answers) > 1:
                for a in answers:
                    a = dict(a)
                    alist.append(a['value'])
                    for alias in a['aliases']:
                        alist.append(alias)
            else:
                a = dict(answers[0])
                alist.append(a['value'])
                for alias in a['aliases']:
                    alist.append(alias)
            answers = alist
            f['answer'] = str(answers)
        else:
            f['output'] = str(f['answers'][0]['value'])
        f['input'] = ''
        if cot:
            # Construct one-hop query to get query prompt and answer for cot
            one_hop_query = Query(f['subject_id'], Relation.string_to_enum(f['relation']), f['target_ids'])

            if len(one_hop_query.get_answers()[0]) == 0:
                print(f'No answers for {one_hop_query.get_query_prompt()}')
                continue
            answer = one_hop_query.get_answers()[0][0]
            f['input'] = f'{one_hop_query.get_query_prompt()} {answer}'
            if ent_only:
                f['input'] = f'{answer}'
            elif partial:
                # Set full_context to one-hop query prompt
                f['full_context'] = one_hop_query.get_query_prompt()
                # Set partial percentage to partial/100
                partial /= 100
                # Get number of words in one-hop query prompt
                total_words = len(f['full_context'].split())
                
                # Get number of words to keep
                num_words = int(total_words * partial)

                # Get partial context
                partial_context = ' '.join(f['full_context'].split()[(total_words-num_words):])
                f['input'] = f'{partial_context} {answer}'


        # Remove all fields that are not 'instruction' 'input' or 'output'
        for key in list(f.keys()):
            if key not in ['instruction', 'input', 'full_context', 'answer', 'output']:
                del f[key]
        train_facts.append(f)
    return train_facts

def build_two_hop_training_splits(args):
    # Set seed -- DO NOT change so that splits have same test set
    seed = 12345
    ent_only = False

    random.seed(seed)
    DATA_DIR = os.path.join('/n/netscratch/kempner_sham_lab/Lab/acarrell/twohop/',args.ent_dir,'processed')

    two_hop_facts = {}
    for fname in glob.glob(os.path.join(DATA_DIR, 'two_hop*.json')):
        facts = load_json(fname)
        two_hop_facts.update(facts)
    

    # Convert two-hop facts to list
    two_hop_facts = [value for value in two_hop_facts.values()]
    print(f'Len of two-hop facts {len(two_hop_facts)}')
    random.shuffle(two_hop_facts)
    test_facts = two_hop_facts[-args.test_size:]

    # Remove test set from two-hop facts
    two_hop_facts = two_hop_facts[:-args.test_size]

    # Check if test-set has been created; make if not
    # test_path = os.path.join(DATA_DIR, 'splits','test.jsonl')
    # test_facts = process_train_facts(test_facts, test=True)
    # if not os.path.exists(test_path):
    #     write_jsonl_file(test_path, test_facts)
    # # Otherwise, check saved test = test split
    # else:
    #     saved_test_facts = read_jsonl_file(os.path.join(DATA_DIR, 'splits', 'test.jsonl'))
    #     assert saved_test_facts == test_facts
    
    # Create training set
    train_facts = two_hop_facts[:args.train_size]

    # Assert test set not in train set
    for f in test_facts:
        assert f not in train_facts

    # If one-hop in split-type, create one-hop facts
    if 'one-hop' in args.split_type:
        one_hop_facts = {}
        for fname in glob.glob(os.path.join(DATA_DIR, 'one_hop*.json')):
            facts = load_json(fname)
            one_hop_facts.update(facts)
        one_hop_facts = [value for value in one_hop_facts.values()]
        random.shuffle(one_hop_facts)
        print(f'len of one hop facts {len(one_hop_facts)}')
    
        # If split type is one-hop, make facts one hop facts only
        if args.split_type == 'one-hop':
            train_facts = one_hop_facts[:args.train_size]
        else:
            train_facts = []
            two_hop_size = args.train_size // 2
            one_hop_size = args.train_size - two_hop_size
            train_facts.extend(two_hop_facts[:two_hop_size])
            train_facts.extend(one_hop_facts[:one_hop_size])
    # If split type is not one-hop, make facts two-hop facts only
    else:
        train_facts = two_hop_facts[:args.train_size]    
    
    # Process train facts
    # If 5 percent cot, split train facts into 95% no cot, 5% cot
    if '5perc-cot' in args.split_type:
        cot_facts = train_facts[int(0.95*args.train_size):]
        train_facts = train_facts[:int(0.95*args.train_size)]
        print(f'len of cot {len(cot_facts)}')
        train_facts = process_train_facts(train_facts, cot=False)
        if 'ent-only' in args.split_type:
            cot_facts = process_train_facts(cot_facts, cot=True, ent_only=True)
        else:
            cot_facts = process_train_facts(cot_facts, cot=True)
        train_facts.extend(cot_facts)
    else:
        cot_facts = train_facts
        if 'ent-only' in args.split_type:
            cot_facts = process_train_facts(cot_facts, ent_only=True, cot='cot' in args.split_type)
        elif 'cot-perc-context' in args.split_type:
            cot_facts = process_train_facts(cot_facts, cot='cot' in args.split_type, partial=args.partial)
        else:
            cot_facts = process_train_facts(cot_facts, cot='cot' in args.split_type)


    # Write train facts to file
    train_fname = f'train_{args.train_size}_{args.split_type}.jsonl'

    # If partial context, add to train_fname
    if 'cot-perc-context' in args.split_type:
        train_fname = f'train_{args.train_size}_{args.split_type}_{int(args.partial)}.jsonl'

    print(f'Writing to file for {args.split_type}')
    write_jsonl_file(os.path.join(DATA_DIR, 'splits', train_fname), train_facts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ent_dir", type=str, default='ent_10k_dataset_small')
    parser.add_argument("--split_type", type=str, default='', choices=['two-hop','two-hop-5perc-cot','two-hop-5perc-cot-ent-only','two-hop-one-hop','one-hop','two-hop-cot','two-hop-cot-ent-only', 'two-hop-cot-perc-context'])
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--test_size", type=int, default=10000)
    parser.add_argument("--partial", type=float, default=0.0)
    build_two_hop_training_splits(parser.parse_args())