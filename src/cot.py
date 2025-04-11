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
from nav import generate_problem, get_batch
from wol import generate_problem as wol_generate_problem, get_batch as wol_get_batch

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

def main(num, num_steps, batch_size, architecture, learning_rate, weight_decay, use_cot=False, info=False, seed=42, removal_smoothing_lambda=float('inf'), remove_cot=False, examples_per_removed_token=1000, pause_token='', answer_only=False, tokens_removed_per=1, dummy=False):

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
    parser.add_argument('--removal_smoothing_lambda', type=float, default=float('inf'))
    parser.add_argument('--remove_cot', action='store_true')
    parser.add_argument('--examples_per_removed_token', type=int, default=0)
    parser.add_argument('--pause_token', type=int, default=0)
    parser.add_argument('--answer_only', action='store_true')
    parser.add_argument('--tokens_removed_per', type=int, default=0)
    parser.add_argument('--dummy', action='store_true')

    args = parser.parse_args()
    
    main(args.num, args.num_steps, args.batch_size, args.architecture, args.learning_rate, args.weight_decay, args.use_cot, args.info, args.seed, args.removal_smoothing_lambda, args.remove_cot, args.examples_per_removed_token, args.pause_token, args.answer_only, args.tokens_removed_per, args.dummy)
