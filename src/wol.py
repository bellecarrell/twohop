import torch
import random
import names
import argparse
import pickle
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AutoModelForCausalLM
import wandb

import names
import random

def generate_problem(num=5, add_cot=False):
    person_names = [names.get_first_name() for _ in range(num)]
    liars = [random.choice([True, False]) for _ in range(num)]
    is_liar = [(liars[i] and not liars[i+1]) or (not liars[i] and liars[i+1]) for i in range(num-1)]
    statements = []
    if not liars[0]:
        statements.append(f"{person_names[0]} tells the truth.")
    else:
        statements.append(f"{person_names[0]} lies.")

    for i in range(len(person_names) - 1):
        next_person = person_names[i + 1]
        if is_liar[i]:
            statements.append(f"{next_person} says {person_names[i]} tells the truth.")
        else:
            statements.append(f"{next_person} says {person_names[i]} lies.")
            

    statements.append(f"Is {person_names[-1]} telling the truth?")

    cot = []
    for i in range(len(person_names)):
        if liars[i]:
            cot.append(person_names[i] + " lies.")
        else:
            cot.append(person_names[i] + " tells the truth.")

    answer = "Yes" if (sum(is_liar) + liars[0]) % 2 == 0 else "No"

    # Combine the statements into the problem
    problem = "\n".join(statements) + "\n"
    cot = "\n".join(cot) + "\n"
    if add_cot:
        return ('Question:\n' + problem, cot + 'Answer: ' + answer)
    else:
        return ('Question:\n' + problem, 'Answer: ' + answer)

def get_batch(batch_size=16, num=5, add_cot=False):
    problems = []
    answers = []
    for _ in range(batch_size):
        problem, answer = generate_problem(num, add_cot)
        problems.append(problem)
        answers.append(answer)
    print(problems)
    print(answers)
    return problems, answers

def compute_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) | (labels == -100)
    example_correct = torch.all(correct, dim=-1)
    return example_correct.float().mean()

def main(num, num_steps, batch_size, architecture, learning_rate, weight_decay, use_cot=False):
    # Initialize wandb
    config = {
        "num_people": num,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "architecture": architecture,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_cot": use_cot
    }
    wandb.init(project="world-of-lies", config=config)
    print(wandb.run.id)
    print(config)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Generate and print a sample problem
    sample_problem = generate_problem(num, use_cot)
    sample_str = sample_problem[0] + '\n' + sample_problem[1]
    logging.info(f"Sample problem:\n{sample_str}")
    wandb.log({"sample_problem": sample_str})

    model_name = architecture
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

    # Print and log number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    formatted_params = f"{num_params / 1e6:.1f}M" if num_params < 1e9 else f"{num_params / 1e9:.1f}B"
    logging.info(f"Number of model parameters: {formatted_params}")
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

    progress_bar = tqdm(range(num_steps), desc="Training", ncols=100)
    for step in progress_bar:
        problem_batch, answer_batch = get_batch(batch_size=batch_size, num=num, add_cot=use_cot)
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

    wandb.save(filename)
    progress_bar.set_postfix_str("Training completed. Results saved to wandb and local file.")

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on World of Lies problem")
    parser.add_argument("--num", type=int, default=5, help="Number of people in each problem")
    parser.add_argument("--num_steps", type=int, default=5000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--architecture", type=str, default="EleutherAI/pythia-160m", help="Model architecture to use")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--use_cot", action="store_true", help="Use the 'Cot' statement in the problem")
    args = parser.parse_args()
    
    main(args.num, args.num_steps, args.batch_size, args.architecture, args.learning_rate, args.weight_decay, args.use_cot)
