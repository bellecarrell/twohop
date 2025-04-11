import torch
import random
import names
import argparse
import pickle
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AutoModelForCausalLM
import wandb
import math
import names
import random


# +
def turn(direction):
    return f'Turn {direction}.'

def generate_turn(num, rotation):
    if rotation not in [0, 90, 180] or num not in [1, 2]:
        raise ValueError("Invalid rotation or number of instructions.")
    if rotation == 0:
        dirs = ['left', 'right', 'around']
        if num == 1:
            return [turn(random.choice(dirs))]
        elif num == 2:
            return [turn(random.choice(dirs)), turn(random.choice(dirs))]
    if rotation == 90:
        directions = ['left', 'right']
        if num == 1:
            return [turn(random.choice(directions))]
        elif num == 2:
            return [turn('around'), random.choice(directions)]
    elif rotation == 180:
        if num == 1:
            return [turn('around')]
        elif num == 2:
            dir = random.choice(['left', 'right'])
            return [turn(dir), turn(dir)]

def step(step, direction=None):
    step = abs(step)
    if direction is None:
        if step == 1:
            return 'Take 1 step.'
        return f'Take {step} steps.'
    else:
        if step == 1:
            return f'Take 1 step {direction}.' 
        return f'Take {step} steps {direction}.' 

class Location:
    def __init__(self, x=0, y=0, heading=0, face_forward=False, max_distance=10):
        self.x = x
        self.y = y
        self.heading = heading
        self.face_forward = face_forward
        self.max_distance = max_distance
    
    def update(self, instruction, verbose=False):

        instruction = instruction.strip()
        instruction = instruction.strip('.')
        if verbose:
            print(f'Instruction: {instruction}')
        
        if instruction.startswith('Take'):
            if verbose:
                print(f'xy before step: {self.x} {self.y}')
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
            if verbose:
                print(f'xy after step: {self.x} {self.y}')
        elif instruction.startswith('Turn'):
            if verbose:
                print(f'Heading before turn: {self.heading}')

            if instruction.endswith('left'):
                self.heading -= 90
            elif instruction.endswith('right'):
                self.heading += 90
            elif instruction.endswith('around'):
                self.heading += 180
            self.heading = self.heading % 360
            if verbose:
                print(f'Heading after turn: {self.heading}')
        else:
            pass
        

    def return_direction(self, axis='x'):
        if axis not in ['x', 'y']:
            raise ValueError("Invalid axis.")
        if axis == 'x':
            if self.x > 0:
                return 270
            else:
                return 90
        if axis == 'y':
            if self.y > 0:
                return 180
            else:
                return 0
            
    def turn_to(self, direction):
        if direction not in [0, 90, 180, 270]:
            raise ValueError('Invalid direction.')
        
        turn = {90: 'right', -270: 'right', -90: 'left', 270: 'left', 180: 'around', -180: 'around'}

        heading = self.heading if self.heading != 0 else 0
        turn_direction = direction - heading
        return turn[turn_direction]
            
    def no_turn(self, direction):
        if self.heading == direction:
            return True
        return False

    def go_to_origin(self, num_instructions=None, instructions=None):
        if not num_instructions:
            num_instructions = self.instr_to_origin()
        instructions = []
        if self.face_forward:
            if self.x > 0:
                instructions.append(step(self.x, 'left'))
            elif self.x < 0:
                instructions.append(step(self.x, 'right'))
            if self.y > 0:
                instructions.append(step(self.y, 'backward'))
            elif self.y < 0:
                instructions.append(step(self.y, 'forward'))
            if num_instructions > 1:
                random.shuffle(instructions)
            
            if num_instructions == 1:
                return instructions[0]
            return instructions
        else:
            return_x = self.return_direction('x')
            return_y = self.return_direction('y')
            ins_x = []
            ins_y = []
            if self.x != 0 and self.y != 0:
                x_first = random.choice([True, False])
                if x_first:
                    if not self.no_turn(return_x):
                        ins_x.append(turn(self.turn_to(return_x)))
                    ins_x.append(step(self.x))
                    for i in ins_x:
                        instructions.append(i)
                        self.update(i)
                    if not self.no_turn(return_y):
                        ins_y.append(turn(self.turn_to(return_y)))
                    ins_y.append(step(self.y))
                    for i in ins_y:
                        instructions.append(i)
                        self.update(i)
                else:
                    if not self.no_turn(return_y):
                        ins_y.append(turn(self.turn_to(return_y)))
                        ins_y.append(step(self.y))
                        for i in ins_y:
                            instructions.append(i)
                            self.update(i)
                    if not self.no_turn(return_x):
                        ins_x.append(turn(self.turn_to(return_x)))
                        ins_x.append(step(self.x))
                        for i in ins_x:
                            instructions.append(i)
                            self.update(i)
            elif self.x != 0:
                if not self.no_turn(return_x):
                    ins_x.append(turn(self.turn_to(return_x)))
                ins_x.append(step(self.x))
                for i in ins_x:
                    instructions.append(i)
                    self.update(i)
            elif self.y != 0:
                if not self.no_turn(return_y):
                    ins_y.append(turn(self.turn_to(return_y)))
                ins_y.append(step(self.y))
                for i in ins_y:
                    instructions.append(i)
                    self.update(i)
            if num_instructions == 1:
                instructions = ''.join(instructions)
            # if not self.end_at_origin():
            #     return ValueError("Did not end at origin.")
            else:
                self.reset()
            
            return instructions
        
    def nonzero(self):
        nonzero = 0
        if self.x != 0:
            nonzero += 1
        if self.y != 0:
            nonzero += 1
        return nonzero
    
    def turns_to_origin(self):
        if self.face_forward:
            return ValueError("Cannot calculate turns to origin when facing forward.")
        turns = 0
        if self.x > 0:
            if self.heading != 270:
                turns += 1
        elif self.x < 0:
            if self.heading != 90:
                turns += 1
        if self.y > 0:
            if self.heading != 180:
                turns += 1
        elif self.y < 0:
            if self.heading != 0:
                turns += 1
        return turns

    def instr_to_origin(self):
        if self.face_forward:
            return self.nonzero()
        else:
            return self.nonzero() + self.turns_to_origin()
            
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

    # def __str__(self):
    #     return f'({self.x}, {self.y}, {self.heading}).'
    
    def reset(self):
        self.x = 0
        self.y = 0
        self.heading = 0

    def valid(self):
        return self.x <= self.max_distance and self.x >= -self.max_distance and self.y <= self.max_distance and self.y >= -self.max_distance

# +
def generate_reverse_direction(direction):
    if direction == 'forward':
        return 'backward'
    elif direction == 'backward':
        return 'forward'
    elif direction == 'left':
        return 'right'
    elif direction == 'right':
        return 'left'
    else:
        return None

def generate_direction(face_forward=True, used_direction=None):
    ff_directions = ['forward', 'backward', 'left', 'right']
    if used_direction is not None:
        ff_directions.remove(used_direction)
        ff_directions.remove(generate_reverse_direction(used_direction))
    if face_forward:
        return random.choice(ff_directions)
    else:
        return random.choice(['around', 'left', 'right'])


# -

def generate_complementary_direction(direction):
    if direction == 'forward' or direction == 'backward':
        return random.choice(['forward', 'backward'])
    elif direction == 'left' or direction == 'right':
        return random.choice(['left', 'right'])
    else:
        return None


def generate_steps_instruction(steps, direction):
    if direction:
        if steps == 1:
            instruction = f'Take 1 step {direction}.'
        else:
            instruction = f'Take {steps} steps {direction}.'
    else:
        if steps == 1:
            instruction = f'Take 1 step.'
        else:
            instruction = f'Take {steps} steps.'
    return instruction

# +
def validate_instruction(instruction, location, same_distance=False, verbose=False):
    if not instruction:
        return False
    num_instructions = location.instr_to_origin()
    location_copy = Location(location.x, location.y, location.heading, face_forward=location.face_forward)
    location_copy.update(instruction)
    if verbose:
        print(f'------- validating instruction: {instruction} -------')
        print(f'location: {location}')
        print(f'location_copy: {location_copy}')
        print(f'num_instructions to origin: {num_instructions}')
        print(f'num instructions copy: {location_copy.instr_to_origin()}')

    if same_distance:
        if location_copy.instr_to_origin() > num_instructions:
            if verbose:
                print('not same distance or less')
            return False
    return location_copy.valid()

def generate_instruction(i=0, face_forward=True, end_at_origin=False, max_distance=10):
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
        from numpy.random import choice
        turn = choice([True, False],p=[0.3,0.7])
        if turn:
            instruction = random.choice(['Turn left.', 'Turn right.', 'Turn around.'])
        else:
            steps = random.randint(1, max_distance)
            if steps == 1:
                instruction = f'Take 1 step.'
            else:
                instruction = f'Take {steps} steps.'
    return instruction


# -

def generate_numbers(n, target_sum):
    """Generates n numbers that sum to target_sum."""

    numbers = []
    remaining_sum = target_sum

    for i in range(n - 1):
        # Generate a random number less than the remaining sum
        num = random.randint(1, remaining_sum - (n - i - 1)) 
        numbers.append(num)
        remaining_sum -= num

    # Add the last number to ensure the sum is correct
    numbers.append(remaining_sum)

    return numbers


def generate_ff_instructions(num, used_direction=None):
    instructions = []
    min_steps = max([2, num])
    steps = random.randint(min_steps, 10)
    direction = generate_direction(True, used_direction)
    back = generate_reverse_direction(direction)
    instructions_direction = random.randint(1, num-1)
    instructions_back = num - instructions_direction

    for dir, num_instructions in [(direction, instructions_direction), (back, instructions_back)]:
        if num_instructions == 1:
            steps_dir = [steps]
        else:
            steps_dir = generate_numbers(num_instructions, steps)
    
        for i in range(num_instructions):
            instruction = generate_steps_instruction(steps_dir[i], dir)
            instructions.append(instruction)
    random.shuffle(instructions)
    return instructions


# +
MIN_DIRS_FF = 2
MIN_DIRS_NO_FF = 3


def generate_step(num, steps):
    if num == 1:
        return [step(steps)]
    else:
        nums = generate_numbers(num, steps)
        return [step(num) for num in nums]
    
def max_steps(steps_turns):
    max_steps = 0
    for i in range(len(steps_turns)-1):
        if i % 2 != 0:
            if steps_turns[i] > max_steps:
                max_steps = steps_turns[i]
    return max_steps

def generate_turn_directions(num):
    if num < 3:
        raise ValueError("Number of instructions must be at least 3.")
    steps_turns = [1] * 3
    steps_turns.insert(0,0)
    steps_turns.append(0)
    to_add = num - 3
    for i in range(to_add):
        idx = random.randint(0, len(steps_turns)-1)
        # Ensure there are not more than 2 instructions for 180 turn 
        while idx == 2 and steps_turns[idx] == 2:
            idx = random.randint(0, len(steps_turns)-1)
        steps_turns[idx] += 1
    instructions = []
    min_steps = max_steps(steps_turns)
    steps = random.randint(min_steps, 10) 
    for i, num_instructions in enumerate(steps_turns):
        if i % 2 == 0:
            if i in [0, 4]:
                rotation = 0
                if num_instructions != 0:
                    instructions.extend(generate_turn(num_instructions, rotation))
            elif i in [2]:
                rotation = 180
                instructions.extend(generate_turn(num_instructions, rotation))
        else:
            instructions.extend(generate_step(num_instructions, steps))
    return instructions

def generate_two_turn_six_steps(num):
    if num == 6:
        instructions = generate_turn_directions(3)
        instructions.extend(generate_turn_directions(3))
    else:
        instructions_turn_one = random.randint(MIN_DIRS_NO_FF, num - MIN_DIRS_FF)
        instructions_turn_two = num - instructions_turn_one
        instructions = generate_turn_directions(instructions_turn_one)
        instructions.extend(generate_turn_directions(instructions_turn_two))
    return instructions

def generate_two_turn_directions(num):
    if num < 6:
        raise ValueError("Number of instructions must be at least 6.")
    if num == 6:
        return generate_two_turn_six_steps(num)
    else:
        six_steps_path = random.choice([True, False])
        if six_steps_path:
            return generate_two_turn_six_steps(num)
        else:
            m = random.randint(1, 10)
            n = random.randint(1, 10)
            lr = random.choice(['left', 'right'])
            instructions = []
            instructions.append(step(m))
            instructions.append(turn(lr))
            instructions.append(step(n))
            instructions.append(turn(lr))
            instructions.append(step(m))
            instructions.append(turn(lr))
            instructions.append(step(n))
            return instructions
    


# +
import copy
def generate_instruction_same_dist_to_origin(i, location):
    instruction = generate_instruction(i, False)
    location_copy = Location(x=location.x, y=location.y, heading=location.heading, face_forward=False)
    num_instr_to_origin = location.instr_to_origin()
    while not validate_instruction(instruction, location_copy, same_distance=True):
        instruction = generate_instruction(i, False)

    return instruction

def generate_problem(num=4, add_cot=False, face_forward=True, end_at_origin=False, max_distance=10, max_steps=10, multiclass=None):
    max_distance = 10
    max_steps = 10
    
    if not multiclass:
        multiclass = True

    prefix = 'Question: If you follow these instructions, do you return to the starting point?\n'
    if multiclass:
        prefix = 'Question: If you follow these instructions, what is your location?\n'

    ff = 'Always face forward.'
    location = Location(face_forward=face_forward, max_distance=max_distance)
    instructions = []
    cot = []
    remaining = num
    i = 0

    while remaining > 0:
        instr_added = 1

        instruction = generate_instruction(i, face_forward, max_distance=max_steps)
        while not validate_instruction(instruction, location):
            instruction = generate_instruction(i, face_forward, max_distance=max_steps)
        
        remaining -= instr_added
        i += instr_added
        
        if type(instruction) == str:
            location.update(instruction)
        
        if type(instruction) == list:
            instructions.extend(instruction)
            location.reset()
            for j in instructions:
                location.update(j)
        else:
            instructions.append(instruction)
    
    FF_TO_ORIGIN = 2
    NO_FF_TO_ORIGIN = 4
    instr_to_origin = {True: FF_TO_ORIGIN, False: NO_FF_TO_ORIGIN}

    # if end_at_origin:
    #     if not location.end_at_origin():
    #         # Get min of instr_to_origin or len of instructions
    #         to_remove = min(instr_to_origin[face_forward], len(instructions))
    #         instructions = instructions[:-to_remove]

    #         location.reset()
    #         for i in instructions:
    #             location.update(i)
            
    #         if not location.end_at_origin():
    #             new_instruction = location.go_to_origin()
    #             if type(new_instruction) == str:
    #                 new_instruction = [new_instruction]
    #             instructions.extend(new_instruction)

    #             if len(new_instruction) != to_remove:
    #                 remaining = to_remove - len(new_instruction)
    #                 dirs = ['left', 'right', 'around']
    #                 new_instruction = [turn(random.choice(dirs)) for i in range(remaining)]
    #                 instructions.extend(new_instruction)
    #         else:
    #             end_at_origin = True
    #             dirs = ['left', 'right', 'around']
    #             new_instruction = [turn(random.choice(dirs)) for i in range(to_remove)]
    #             instructions.extend(new_instruction)
    # else:
    #     if location.end_at_origin():
    #         end_at_origin = True

    location.reset()

    cot.append("We start at the origin, (0, 0, 0).")
    for i, instr in enumerate(instructions):
        location.update(instr)
        if i != len(instructions) - 1:
            cot.append(f'{instr} {str(location)}')
        else:
            cot.append(f'{instr}')

    problem = "\n".join(instructions) + "\n"
    cot = "\n".join(cot) + "\n"
    answer = 'Yes' if end_at_origin else 'No'
    if multiclass:
        answer = f'({location.x}, {location.y})'

    if add_cot:
        return (prefix + problem, cot, 'Answer: ' + answer)
    else:
        return (prefix + problem, '', 'Answer: ' + answer)

def get_batch(batch_size=16, num=5, add_cot=False, max_distance=10, max_steps=10, multiclass=False):
    problems = []
    cots = []
    answers = []
    # generate batch random coin flips for face forward
    ff = [random.choice([True, False]) for _ in range(batch_size)]
    end_at_origin = [random.choice([True, False]) for _ in range(batch_size)]
    if multiclass:
        end_at_origin = [False for _ in range(batch_size)]

    for i in range(batch_size):
        ffi = ff[i]
        end_at_origini = end_at_origin[i]
        problem, cot, answer = generate_problem(num, add_cot, ffi, end_at_origini, max_distance=max_distance, max_steps=max_steps, multiclass=multiclass)
        problems.append(problem)
        answers.append(answer)
        cots.append(cot)
    return problems, cots, answers

def compute_accuracy(logits, labels, tokenizer):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) | (labels == -100)
    example_correct = torch.all(correct, dim=-1)
    return example_correct.float().mean()

def compute_accuracy_answer_only(logits, labels, tokenizer, logstep=False):
    predictions = torch.argmax(logits, dim=-1)
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    total = len(labels)
    correct = 0
    for i in range(total):
        prediction = extract_answer(predictions[i], tokenizer)
        label = extract_answer(labels[i], tokenizer)
        # prediction = predictions[i]
        # label = f'{cot[i]} {answer[i]}'
        # prediction = prediction.replace('\n', ' ')
        # label = label.replace('\n', ' ')
        if logstep and i == 0:
            print(f'Prediction: {prediction}')
            print(f'Label: {label}')
        if label in prediction:
            correct += 1
    if logstep:
        print(f'acc: {correct}/{total}')
    return correct / total

def extract_answer(text, tokenizer):
    #text = tokenizer.decode(tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    text = text.split('Answer:')[-1]
    answer = text.strip()
    return answer


def compute_lambda_distribution(removal_smoothing_lambda, truncate_length=100):
    if removal_smoothing_lambda == float('inf'):
        lambda_distribution = torch.zeros(truncate_length)
        lambda_distribution[0] = 1
    else:
        positions = torch.arange(truncate_length)
        lambda_distribution = (1 - math.exp(-removal_smoothing_lambda)) * positions.mul(-removal_smoothing_lambda).exp()
        cum_prob = lambda_distribution.sum()
        assert cum_prob <= 1
        lambda_distribution[-1] = lambda_distribution[-1] + (1-cum_prob)
    return lambda_distribution

def main(num, num_steps, batch_size, architecture, learning_rate, weight_decay, use_cot=False, info=False, max_distance=10, max_steps=10, multiclass=False, seed=42, removal_smoothing_lambda=float('inf'), remove_cot=False, examples_per_removed_token=1000, pause_token='', answer_only=False, tokens_removed_per=1, dummy=False):

    keep_last_step = False

    # Initialize wandb
    config = {
        "num_sentences": num,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "architecture": architecture,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_cot": use_cot, 
        "examples_per_removed_token": examples_per_removed_token,
        "removal_smoothing_lambda": removal_smoothing_lambda, 
        "keep_last_step": keep_last_step,
    }
    
    if info:
        wandb.init(project="nav", config=config)
        print(wandb.run.id)
        # change run name to include num, num_steps, batch_size, architecture, learning_rate, weight_decay, use_cot
        wandb.run.name = f'{info}_cot{use_cot}_num{num}_remove{examples_per_removed_token}_steps{num_steps}_tokens{tokens_removed_per}'
        print(config)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    verbose = False

    # Generate and print a sample problem
    sample_problem = generate_problem(num=num, add_cot=use_cot, multiclass=multiclass)
    sample_str = sample_problem[0] + '\n' + sample_problem[1] + '\n' + sample_problem[2]
    logging.info(f"Sample problem:\n{sample_str}")
    if info:
        wandb.log({"sample_problem": sample_str})

    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

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

    if examples_per_removed_token > 0:
        remove_cot = True

    if remove_cot:
        steps_per_removal = examples_per_removed_token // batch_size
        lambda_distribution = compute_lambda_distribution(removal_smoothing_lambda)
        scheduled_to_remove = 0
        all_cot_removed_in_prev_batch = False
        logging.info("remove_cot is True. Removing cot tokens during training.")

    progress_bar = tqdm(range(num_steps), desc="Training", ncols=100)
    for i ,step in enumerate(progress_bar):
        if remove_cot:
            prev_scheduled_to_remove = scheduled_to_remove
            scheduled_to_remove = i // steps_per_removal
            random_removal_offset = torch.multinomial(lambda_distribution, batch_size, replacement=True) 
            reset_opt = False
            reset_step = 0
            to_remove = scheduled_to_remove + random_removal_offset
            if scheduled_to_remove > prev_scheduled_to_remove:
                print(f" -step {step}. removing: {scheduled_to_remove * tokens_removed_per}")
                if (not all_cot_removed_in_prev_batch):
                    print ('RESETTING OPTIMIZER')
                    optimizer.zero_grad(set_to_none=True)
                    del optimizer
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    reset_opt = True
                    reset_step = step

        problem_batch, cot_batch, answer_batch = get_batch(batch_size=batch_size, num=num, add_cot=use_cot, max_distance=max_distance, max_steps=max_steps, multiclass=multiclass)
        
        if pause_token != 0 and use_cot:
            for i in range(len(cot_batch)):
                cot_batch[i] = ''.join(['<pause>' for _ in range(pause_token)])
        
        tokenized_problems = tokenizer(problem_batch)
        tokenized_cots = tokenizer(cot_batch)

        tokenized_answers = tokenizer(answer_batch)
        tokenized_batch = {'input_ids': [], 'attention_mask': [], 'loss_mask': []}
        all_cot_removed_in_batch = False
        len_cot = 0
        for i in range(len(tokenized_problems['input_ids'])):
            problem_input_ids = torch.tensor(tokenized_problems['input_ids'][i])
            problem_attention_mask = torch.tensor(tokenized_problems['attention_mask'][i])
            cot_input_ids = torch.tensor(tokenized_cots['input_ids'][i])
            cot_attention_mask = torch.tensor(tokenized_cots['attention_mask'][i])
            answer_input_ids = torch.tensor(tokenized_answers['input_ids'][i])
            answer_attention_mask = torch.tensor(tokenized_answers['attention_mask'][i])
            len_cot = len(cot_input_ids)

            if remove_cot:
                to_remove[i] = to_remove[i] * tokens_removed_per
                if to_remove[i] > 0:
                    if keep_last_step and ((len(cot_input_ids) - to_remove[i]) < tokens_removed_per):
                        to_remove[i] = len(cot_input_ids) - tokens_removed_per
                        
                    # print(f'len cot_input_ids: {len(cot_input_ids)}')
                    # print(f'to_remove: {to_remove[i]}')
                    # print(f'len cot_input_ids[to_remove[i]:]: {len(cot_input_ids[to_remove[i]:])}')
                    cot_input_ids = cot_input_ids[to_remove[i]:]
                    cot_attention_mask = cot_attention_mask[to_remove[i]:]

                    answer_input_ids = torch.cat([cot_input_ids, answer_input_ids])
                    answer_attention_mask = torch.cat([cot_attention_mask, answer_attention_mask])
                else:
                    answer_input_ids = torch.cat([cot_input_ids, answer_input_ids])
                    answer_attention_mask = torch.cat([cot_attention_mask, answer_attention_mask])
                if to_remove[i] >= len(cot_input_ids):
                    all_cot_removed_in_batch = True
                tokenized_cots['input_ids'][i] = cot_input_ids
                tokenized_cots['attention_mask'][i] = cot_attention_mask
                      
            else:
                if use_cot:
                    answer_input_ids = torch.cat([cot_input_ids, answer_input_ids])
                    answer_attention_mask = torch.cat([cot_attention_mask, answer_attention_mask])

            tokenized_batch['input_ids'].append(torch.cat([problem_input_ids, answer_input_ids]))
            tokenized_batch['attention_mask'].append(torch.cat([problem_attention_mask, answer_attention_mask]))
            tokenized_batch['loss_mask'].append(torch.cat([torch.zeros_like(problem_input_ids), torch.ones_like(answer_input_ids)]))
        
        
        if remove_cot and reset_opt:
            print(f'len cot input ids: {len_cot}')

        all_cot_removed_in_prev_batch = all_cot_removed_in_batch
        if all_cot_removed_in_batch and dummy:
            verbose = True

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


        logstep = False
        # if (all_cot_removed_in_batch and dummy and i % 100 == 0) or (i % 500 == 0):
        if (dummy and step % 50 == 0 and remove_cot) or (dummy and remove_cot and step > 700 and step < 1100 and step % 10 == 0):
            p = torch.argmax(logits, dim=-1)
            p = tokenizer.batch_decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            els = tokenizer.batch_decode(labels, skip_special_tokens=True)
            probs = tokenizer.batch_decode(tokenized_problems['input_ids'], skip_special_tokens=True)
            cot = tokenizer.batch_decode(tokenized_cots['input_ids'], skip_special_tokens=True)
            answer = tokenizer.batch_decode(tokenized_answers['input_ids'], skip_special_tokens=True)
            with open(f'{num}_{info}_predictions.txt', 'a') as f:
                f.write('--------------------------------------------------\n')
                # f.write(f'Last reset opt step: {reset_step}\n\n')
                f.write(f'Step: {step}\n\n')
                f.write(f'Problem: {probs[0]}\n\n')
                f.write(f'COT + answer: {cot[0]} {answer[0]}\n\n')
                f.write(f'Prediction: {p[0]}\n\n')
                f.write('\n')
            logstep = True

        if answer_only:
            accuracy = compute_accuracy_answer_only(logits, labels, tokenizer, logstep=logstep)

        rel = rel[:, 1:]
        labels = torch.where(rel, labels, -100)
        loss = criterion(logits.reshape(-1, preds.logits.shape[-1]), labels.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if not answer_only: 
            accuracy = compute_accuracy(logits, labels, tokenizer)

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
    parser.add_argument('--removal_smoothing_lambda', type=float, default=float('inf'))
    parser.add_argument('--remove_cot', action='store_true')
    parser.add_argument('--examples_per_removed_token', type=int, default=0)
    parser.add_argument('--pause_token', type=int, default=0)
    parser.add_argument('--answer_only', action='store_true')
    parser.add_argument('--tokens_removed_per', type=int, default=0)
    parser.add_argument('--dummy', action='store_true')

    args = parser.parse_args()
    
    main(args.num, args.num_steps, args.batch_size, args.architecture, args.learning_rate, args.weight_decay, args.use_cot, args.info, args.max_distance, args.max_steps, args.multiclass, args.seed, args.removal_smoothing_lambda, args.remove_cot, args.examples_per_removed_token, args.pause_token, args.answer_only, args.tokens_removed_per, args.dummy)
