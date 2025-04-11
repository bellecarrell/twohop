import torch
import random
import names
import argparse
import pickle
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AutoModelForCausalLM
import wandb
import copy

import names
import random

import json
from typing import List, Tuple

import inflect  # using inflect==5.3.0

# Inspired by Big Bench object counting, code modified from https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/object_counting/generate_task.py

DATA = {
    "animals": [
        "cat", "rabbit", "donkey", "dog", "mouse", "cow", "snake", "fish",
        "bear", "snail", "chicken", "pig", "duck", "goat", "frog"
    ],
    "objects": [
        "car", "chair", "couch", "table", "bed", "lamp", "microwave", "oven",
        "stove", "fridge", "toaster"
    ],
    "vegetables": [
        "carrot", "head of broccoli", "cabbage", "yam", "garlic", "cauliflower",
        "lettuce head", "stalk of celery", "potato", "onion"
    ],
    "musical instruments": [
        "violin", "trumpet", "flute", "clarinet", "drum", "piano", "trombone",
        "accordion"
    ],
    "fruits": [
        "strawberry", "banana", "grape", "peach", "orange", "apple",
        "blackberry", "plum", "nectarine", "raspberry"
    ],
}



# Get key from dictionary based on value

p = inflect.engine()

def get_counting_word(word: str, number: int) -> str:
  is_vowel = word[0] in "aeiou"
  article = "an" if is_vowel else "a"
  return [article, "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][number - 1]


def add_and(items: List[str]) -> List[str]:
  items[-1] = "and " + items[-1]


def add_count(items: List[str]) -> int:

  counts = {}
  p = inflect.engine()

  for i in range(len(items)):
    word = items[i]
    #number = random.choices([1, 2, 3, 4, 5], weights=[80, 10, 6, 3, 1])[0]
    # vary weight number = random.choices([1, 2, 3, 4, 5], weights=[60, 20, 12, 6, 2])[0]
    number = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])[0]
    counting_word = get_counting_word(word, number)
    word = p.plural(word, number)

    items[i] = f"{counting_word} {word}"
    counts[items[i]] = number

  return counts


def add_decoy(items: List[str], exclude_kind: str) -> List[str]:
  k = random.randint(1, 3)
  kind = random.choice([kind for kind in DATA if kind != exclude_kind])
  new_items = random.sample(DATA[kind], k)
  add_count(new_items)
  new_items.extend(items)
  random.shuffle(new_items)
  return new_items


def make_example(k=None):

  if not k:
    k = random.randint(2, 10)
  kind = random.choice(list(DATA))
  items = random.sample(DATA[kind], min(len(DATA[kind]), k))
  counts = add_count(items)
  items_list = copy.deepcopy(items)
  count = sum(counts.values())
  if random.random() > 0.50 and kind != "objects":
    #if random.random() > 0.75 and kind != "objects":
    items = add_decoy(items, kind)
  add_and(items)
  items = ", ".join(items)
  return (f"Question: I have {items}. How many {kind} do I have?",
          (p.number_to_words(count), str(count)), (counts, items_list, kind))

# +

def generate_problem(num=4, add_cot=False, face_forward=True, end_at_origin=False, max_distance=10, max_steps=10, multiclass=False):

    problem, answer, (counts, items, kind) = make_example(k=num)

    if add_cot:
        cot = []
        cot.append("Let's think step by step.")
        cot.append(f"We first identify the {kind} on the list and include their quantity in parentheses:")
        for i in range(len(items)):
            cot.append(f'{items[i]} ({counts[items[i]]})')
        cot.append("Now, let's add the numbers in parentheses: ")
        counts_str = '+'.join([str(x) for x in counts.values()])
        cot.append(f'{counts_str} = {sum(counts.values())}. So the answer is {sum(counts.values())}.')
        cot = "\n".join(cot) + "\n"

    if add_cot:
        return (problem, cot + 'Answer: ' + answer[1])
    else:
        return (problem, 'Answer: ' + answer[1])

def get_batch(batch_size=16, num=5, add_cot=False, max_distance=10, max_steps=10, multiclass=False):
    problems = []
    answers = []
    # generate batch random coin flips for face forward
    ff = [random.choice([True, False]) for _ in range(batch_size)]
    end_at_origin = [random.choice([True, False]) for _ in range(batch_size)]
    if multiclass:
        end_at_origin = [False for _ in range(batch_size)]

    for i in range(batch_size):
        ffi = ff[i]
        end_at_origini = end_at_origin[i]
        problem, answer = generate_problem(num, add_cot, ffi, end_at_origini, max_distance=max_distance, max_steps=max_steps, multiclass=multiclass)
        problems.append(problem)
        answers.append(answer)
    return problems, answers

def compute_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) | (labels == -100)
    example_correct = torch.all(correct, dim=-1)
    return example_correct.float().mean()

def main(num, num_steps, batch_size, architecture, learning_rate, weight_decay, use_cot=False, info=False, max_distance=10, max_steps=10, multiclass=False):

    # Initialize wandb
    config = {
        "num_sentences": num,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "architecture": architecture,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_cot": use_cot
    }
    
    if info:
        wandb.init(project="obj_count", config=config)
        print(wandb.run.id)
        # change run name to include num, num_steps, batch_size, architecture, learning_rate, weight_decay, use_cot
        wandb.run.name = f'{info}_cot{use_cot}_num{num}_steps{num_steps}_arch{architecture.split("/")[-1]}_lr{learning_rate}'
        print(config)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Generate and print a sample problem
    sample_problem = generate_problem(num=num, add_cot=use_cot)
    sample_str = sample_problem[0] + '\n' + sample_problem[1]
    logging.info(f"Sample problem:\n{sample_str}")
    if info:
        wandb.log({"sample_problem": sample_str})

    model_name = architecture
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

    # Print and log number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    formatted_params = f"{num_params / 1e6:.1f}M" if num_params < 1e9 else f"{num_params / 1e9:.1f}B"
    logging.info(f"Number of model parameters: {formatted_params}")
    if info:
        wandb.log({"num_parameters": num_params, "formatted_num_parameters": formatted_params})

    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Calculate warmup steps (10% of total steps)
    warmup_steps = int(0.1 * num_steps)
    
    # Create scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    num_true = 0

    progress_bar = tqdm(range(num_steps), desc="Training", ncols=100)
    for step in progress_bar:
        problem_batch, answer_batch = get_batch(batch_size=batch_size, num=num, add_cot=use_cot, max_distance=max_distance, max_steps=max_steps, multiclass=multiclass)
        tokenized_problems = tokenizer(problem_batch)
        tokenized_answers = tokenizer(answer_batch)
        tokenized_batch = {'input_ids': [], 'attention_mask': [], 'loss_mask': []}
        for i in range(len(tokenized_problems['input_ids'])):
            problem_input_ids = torch.tensor(tokenized_problems['input_ids'][i])
            problem_attention_mask = torch.tensor(tokenized_problems['attention_mask'][i])
            answer_input_ids = torch.tensor(tokenized_answers['input_ids'][i])
            answer_attention_mask = torch.tensor(tokenized_answers['attention_mask'][i])
            tokenized_batch['input_ids'].append(torch.cat([problem_input_ids, answer_input_ids]))
            tokenized_batch['attention_mask'].append(torch.cat([problem_attention_mask, answer_attention_mask]))
            tokenized_batch['loss_mask'].append(torch.cat([torch.zeros_like(problem_input_ids), torch.ones_like(answer_input_ids)]))
        
        batch_max_length = max([len(x) for x in tokenized_batch['input_ids']])
        for k in tokenized_batch.keys():
            for i in range(len(tokenized_batch[k])):
                tokenized_batch[k][i] = torch.nn.functional.pad(tokenized_batch[k][i], (0, batch_max_length - len(tokenized_batch[k][i])), value=tokenizer.pad_token_id)
            tokenized_batch[k] = torch.stack(tokenized_batch[k])
        
        if CUDA:
            tokenized_batch = {k: v.cuda() for k, v in tokenized_batch.items()}
    
        rel = tokenized_batch['loss_mask'] == 1
        preds = model(tokenized_batch['input_ids'], attention_mask=tokenized_batch['attention_mask'])
        logits = preds.logits[:, :-1, :]
        labels = tokenized_batch["input_ids"][:, 1:]
        rel = rel[:, 1:]
        labels = torch.where(rel, labels, -100)
        loss = criterion(logits.reshape(-1, preds.logits.shape[-1]), labels.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        accuracy = compute_accuracy(logits, labels)

        if info:
            # Log metrics to wandb
            wandb.log({
                "loss": loss.item(),
                "accuracy": accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
                "learning_rate": scheduler.get_last_lr()[0]
            })

        if step % 10 == 0 and step > 0:
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Accuracy': f'{accuracy:.4f}'})

    # Save results to pickle file
    results = {
        'num': num,
        'num_steps': num_steps,
        'batch_size': batch_size,
        'architecture': architecture,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'use_cot': use_cot
    }
    filename = f'training_results_num{num}_steps{num_steps}_batch{batch_size}_arch{architecture.split("/")[-1]}_lr{learning_rate}_wd{weight_decay}'
    if use_cot:
        filename += '_cot'
    filename += '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    if info:
        wandb.save(filename)
        # Finish the wandb run
        wandb.finish()
    progress_bar.set_postfix_str("Training completed. Results saved to wandb and local file.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on nav problem")
    parser.add_argument("--num", type=int, default=4, help="Number of people in each problem")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--architecture", type=str, default="EleutherAI/pythia-160m", help="Model architecture to use")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--use_cot", action="store_true", help="Use the 'Cot' statement in the problem")
    parser.add_argument("--info", type=str, help="Use info in wandb name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_steps", type=int, default=10, help="Max number of steps in a single instruction")
    parser.add_argument("--max_distance", type=int, default=10, help="Max distance from origin")
    parser.add_argument("--multiclass", action="store_true", help="Use multiclass classification")
    args = parser.parse_args()
    
    main(args.num, args.num_steps, args.batch_size, args.architecture, args.learning_rate, args.weight_decay, args.use_cot, args.info, args.max_distance, args.max_steps, args.multiclass)
