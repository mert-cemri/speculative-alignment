import torch
from tqdm import tqdm

from utils import norm_logits, sample, max_fn, Decoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import numpy as np
import time
@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None,tokenizer=None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    correlatations = []
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    start_time = time.time()
    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                normalized_logits = norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p)
                # print(top_k)
                # print(q.shape)
                # print(q[:, -1, :].shape) #(batch, vocab)
                # print("normalized_logits:",normalized_logits.shape) #(batch, vocab)
                num_samples=1
                next_tok = sample(normalized_logits,num_samples=num_samples)
                # print("next_tok:",next_tok)
                # assert False
                
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                
                acceptance_prob = torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j])
                # print()
                # print(torch.max(q[:, prefix_len + i - 1, :],dim=1)[0].item())
                correlatations.append(acceptance_prob.squeeze().cpu().item())
                if r < acceptance_prob:
                    # accept, and update n
                    new_tuple = [(p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]).cpu().item(),torch.max(q[:, prefix_len + i - 1, :],dim=1)[0].cpu().item()]
                    # correlatations.append(new_tuple)
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    is_all_accept = False
                    break
                
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)

    import numpy as np
    # correlatations = np.array(correlatations).T
    execution_time = time.time() - start_time

    return prefix, correlatations, execution_time

def return_reward(prompt,response,device,model_name="Skywork/Skywork-Reward-Llama-3.1-8B"):
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        # attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]

    # Format and tokenize the conversations
    conv_formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
    conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)

    # Get the reward scores
    with torch.no_grad():
        score = rm(**conv_tokenized).logits[0][0].item()
    return score

def sample_with_alignment(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None,tokenizer=None,
                         model_name="Skywork/Skywork-Reward-Llama-3.1-8B") -> torch.Tensor:
    device = approx_model.device

    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"

    correlatations = []
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    start_time = time.time()
    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = target_model(x).logits
                num_samples = 5
                normalized_logits = norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p)
                next_tok = sample(normalized_logits,num_samples=num_samples)
                # print("next_tok",next_tok.shape)
                x_draft = [torch.cat((x, next_tok[:,i:i+1]), dim=1) for i in range(num_samples)]                
                # print("x_draft",x_draft.shape)
                x_draft_text = [tokenizer.decode(x_draft[i][0], skip_special_tokens=True) for i in range(num_samples)]
                rewards = torch.zeros(num_samples,device=x.device)
                for i in range(num_samples):
                    rewards[i] = return_reward(prompt= tokenizer.decode(x_draft[i][0], skip_special_tokens=True),
                                response=x_draft_text[i],
                                device=x.device)
                reward_coeff = 0.01
                # print(next_tok)
                # assert False
                print(normalized_logits.device)
                print(rewards.device)
                scores = normalized_logits[0,next_tok[0].long()] * torch.exp(reward_coeff*rewards)
                scores = scores/torch.sum(scores) 
                aligned_token_index = sample(scores)
                print(next_tok)
                print(aligned_token_index)
                x = torch.cat((x, next_tok[:,aligned_token_index]), dim=1)
                # assert False
            
            # normalize the logits
            # for i in range(q.shape[1]):
            #     q[:,i,:] = norm_logits(q[:,i,:],
            #                     temperature, top_k, top_p)
                
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            # p = target_model(x).logits
            # for i in range(p.shape[1]):
            #     p[:,i,:] = norm_logits(p[:,i,:],
            #                     temperature, top_k, top_p)
            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            # n = prefix_len - 1
            # for i in range(gamma):
            #     if random_seed:
            #         torch.manual_seed(random_seed)
            #     r = torch.rand(1, device = p.device)
            #     j = x[:, prefix_len + i]
                
            #     acceptance_prob = torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j])
            #     correlatations.append(acceptance_prob.squeeze().cpu().item())
            #     if r < acceptance_prob:
            #         # accept, and update n
            #         new_tuple = [(p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]).cpu().item(),torch.max(q[:, prefix_len + i - 1, :],dim=1)[0].cpu().item()]
            #         n += 1
            #     else:
            #         # reject
            #         t = sample(max_fn(p[:, n, :] - q[:, n, :]))
            #         is_all_accept = False
            #         break
                
            # prefix = x[:, :n + 1]
            
            prefix = x
            # if is_all_accept:
            #     t = sample(p[:, -1, :])
            
            # prefix = torch.cat((prefix, t), dim=1)
            # pbar.update(n - pbar.n)

    # correlatations = np.array(correlatations).T
    execution_time = time.time() - start_time

    return prefix, correlatations, execution_time

