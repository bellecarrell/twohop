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
class Location:
    def __init__(self, x=0, y=0, heading=0):
        self.x = x
        self.y = y
        self.heading = heading
    
    def update(self, instruction):
        instruction = instruction.strip()
        instruction = instruction.strip('.')
        if instruction.startswith('Take'):
            steps = int(instruction.split(' ')[1])
            if instruction.endswith('forward'):
                self.y += steps
            elif instruction.endswith('backward'):
                self.y -= steps
            elif instruction.endswith('left'):
                self.x -= steps
            elif instruction.endswith('right'):
                self.x += steps
            elif instruction.endswith('steps') or instruction.endswith('step'):
                if self.heading == 0:
                    self.y += steps
                elif self.heading == 90:
                    self.x += steps
                elif self.heading == 180:
                    self.y -= steps
                elif self.heading == 270:
                    self.x -= steps
        elif instruction.startswith('Turn'):
            if instruction.endswith('left'):
                self.heading -= 90
            elif instruction.endswith('right'):
                self.heading += 90
            elif instruction.endswith('around'):
                self.heading += 180
            self.heading = self.heading % 360
        else:
            pass

    def go_to_origin(self, num_instructions=1, instructions=None):
        if num_instructions != 1:
            raise ValueError("Only 1 instruction is supported at this time.")
        if self.x > 0:
            return f'Take {self.x} steps left.'
        elif self.x < 0:
            return f'Take {abs(self.x)} steps right.'
        elif self.y > 0:
            return f'Take {self.y} steps backward.'
        elif self.y < 0:
            return f'Take {abs(self.y)} steps forward.'
            
    def print_heading(self):
        if self.heading == 0:
            return 'facing the positive y-axis'
        elif self.heading == 90:
            return 'facing the positive x-axis'
        elif self.heading == 180:
            return 'facing the negative y-axis'
        elif self.heading == 270:
            return 'facing the negative x-axis'
        else:
            return 'Invalid heading'

    def end_at_origin(self):
        return self.x == 0 and self.y == 0

    def __str__(self):
        return f'({self.x}, {self.y}), {self.print_heading()}.'
    
    def reset(self):
        self.x = 0
        self.y = 0
        self.heading = 0

def generate_instruction(i=0, face_forward=True):
    if face_forward:
        if i == 0:
            instruction = 'Always face forward.'
        else:
            steps = random.randint(1, 10)
            direction = random.choice(['forward', 'backward', 'left', 'right'])
            if steps == 1:
                instruction = f'Take 1 step {direction}.'
            else:
                instruction = f'Take {steps} steps {direction}.'
    else:
        turn = random.choice([True, False])
        if turn:
            instruction = random.choice(['Turn left.', 'Turn right.', 'Turn around.'])
        else:
            steps = random.randint(1, 10)
            if steps == 1:
                instruction = f'Take 1 step.'
            else:
                instruction = f'Take {steps} steps.'
    return instruction



def generate_problem(num=3, add_cot=False, face_forward=True, end_at_origin=False):
    if num != 3:
        raise ValueError("Only 3 sentence problems are supported at this time.")
    prefix = 'Q: If you follow these instructions, do you return to the starting point?'
    ff = 'Always face forward.'
    location = Location()
    instructions = []
    cot = []
    if end_at_origin:
        if face_forward:
            instructions.append(ff)
            
            steps = random.randint(1, 10)
            direction = random.choice(['forward', 'backward', 'left', 'right'])
            if steps == 1:
                instruction = f'Take 1 step {direction}.'
            else:
                instruction = f'Take {steps} steps {direction}.'
            instructions.append(instruction)

            location.update(instruction)
            instructions.append(location.go_to_origin())
        else:
            steps = random.randint(1, 10)
            if steps == 1:
                instruction = f'Take 1 step.'
            else:
                instruction = f'Take {steps} steps.'
            instructions.append(instruction)
            instructions.append('Turn around.')
            instructions.append(instruction)
    
    else:
        for i in range(num):
            instruction = generate_instruction(i, face_forward)

            
            location.update(instruction)
            if i == num - 1:
                if location.end_at_origin():
                    end_at_origin = True
            instructions.append(instruction)
                
    location.reset()
    
    if add_cot:
        cot.append("Let's think step by step. \nWe start at the origin (0, 0), facing the positive y-axis.")
        for i in instructions:
            location.update(i)
            cot.append(f'{i} {str(location)}')

        if end_at_origin:
            cot.append('Since (0, 0) is (0, 0), we are indeed where we started. So the answer is Yes.\n')
        else:
            cot.append(f'Since ({location.x}, {location.y}) is not (0, 0), we are not where we started. So the answer is No.\n')


    # Combine the statements into the problem
    problem = "\n".join(instructions) + "\n"
    cot = "\n".join(cot) + "\n"
    answer = 'Yes' if end_at_origin else 'No'
    if add_cot:
        return (prefix + problem, cot + 'Answer: ' + answer)
    else:
        return (prefix + problem, 'Answer: ' + answer)            
                
def get_batch(batch_size=16, num=5, add_cot=False):
    problems = []
    answers = []
    # generate batch random coin flips for face forward
    ff = [random.choice([True, False]) for _ in range(batch_size)]
    end_at_origin = [random.choice([True, False]) for _ in range(batch_size)]
    for i in range(batch_size):
        ffi = ff[i]
        end_at_origini = end_at_origin[i]
        problem = None
        while problem is None:
            try:
                problem, answer = generate_problem(num, add_cot, ffi, end_at_origini)
            except ValueError:
                pass
            else:
                break
        #problem, answer = generate_problem(num, add_cot, ff.pop(), end_at_origin.pop())
        problems.append(problem)
        answers.append(answer)
    return problems, answers

def compute_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) | (labels == -100)
    example_correct = torch.all(correct, dim=-1)
    return example_correct.float().mean()

def main(num, num_steps, batch_size, architecture, learning_rate, weight_decay, use_cot=False):

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
    wandb.init(project="nav", config=config)
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

    num_true = 0

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
    parser = argparse.ArgumentParser(description="Train model on nav problem")
    parser.add_argument("--num", type=int, default=3, help="Number of people in each problem")
    parser.add_argument("--num_steps", type=int, default=5000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--architecture", type=str, default="EleutherAI/pythia-160m", help="Model architecture to use")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--use_cot", action="store_true", help="Use the 'Cot' statement in the problem")
    args = parser.parse_args()
    
    main(args.num, args.num_steps, args.batch_size, args.architecture, args.learning_rate, args.weight_decay, args.use_cot)