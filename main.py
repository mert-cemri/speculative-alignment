import torch
from sampling_algorithms import speculative_sampling, sample_with_alignment,return_reward

import argparse
import contexttimer

import matplotlib as mpl

# mpl.style.use('seaborn-vO_8-dark-palette')
#mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 14
mpl.rcParams["figure.figsize"] = (16, 8)
#mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['legend.fontsize'] = 13
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = .3

from matplotlib import pyplot as plt
from utils import create_models, color_print

MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "llama3-8b":"solidrust/Meta-Llama-3-8B-Instruct-AWQ",
    "llama3-13b":"solidrust/Llama-3-13B-Instruct-v0.1-AWQ",
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["llama3-8b"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["llama3-13b"])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=10, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    return args

def generate(input_text, small_model, large_model, tokenizer, num_tokens=10, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = small_model.device
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    torch.manual_seed(123)
    output,acceptance_probs_specdec,execution_time_spec = speculative_sampling(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed,tokenizer=tokenizer)
    generated_text_spec = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"deepmind's speculative_sampling: {generated_text_spec}")   
    
    # acceptance_probs_osd, execution_time_osd = acceptance_probs_specdec, execution_time_spec

    torch.manual_seed(123)
    output,acceptance_probs_osd,execution_time_osd = sample_with_alignment(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed,tokenizer=tokenizer)
    generated_text_align = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"incontext online speculative_sampling: {generated_text_align}")  

    # acceptance_probs_specdec, execution_time_spec = acceptance_probs_osd, execution_time_osd
    return [acceptance_probs_specdec,acceptance_probs_osd, execution_time_spec, execution_time_osd,
            generated_text_spec, generated_text_align]

def average_accross_time(some_list_of_lists):
    max_length = 0
    for some_list in some_list_of_lists:
        if max_length < len(some_list):
            max_length = len(some_list)
    
    final_list = []
    for i in range(max_length):
        sums = 0
        counts = 0
        for some_list in some_list_of_lists:
            try:
                sums += some_list[i]
                counts += 1
            except:
                pass
        final_list.append(sums/counts)
    return final_list

if __name__ == "__main__":
    args = parse_arguments()
    
    import json

    with open('../ShareGPT_Vicuna_unfiltered/ShareGPT_2023.05.04v0_Wasteland_Edition.json','r') as file:
        sharegpt_data = json.load(file)

    test_data = []
    for i in range(300):
        for j in range(len(sharegpt_data[i]['conversations'])):
            if sharegpt_data[i]['conversations'][j]['from'] == 'human':
                test_data.append(sharegpt_data[i]['conversations'][j]['value'])
        if len(test_data) > 500:
            break

    totals = 0
    specdec_acceptances = []
    osd_acceptances = []
    specdec_times = []
    osd_times = []
    draft_model, target_model, tokenizer = create_models(args.approx_model_name, args.target_model_name)
    count = 0
    spec_rewards = []
    align_rewards = []
    for test in test_data[1:10]:
        count +=1 
        print("Doing",count)
        if len(test) > 30 :
            totals += 1
            # try:
            acceptance_probs_spec,acceptance_probs_osd,time_spec,time_osd,generated_text_spec,generated_text_align = generate(test, draft_model, target_model, tokenizer, num_tokens=args.max_tokens, gamma=args.gamma,
                                    random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)
            reward_spec = return_reward(test,generated_text_spec,device=target_model.device)
            reward_align = return_reward(test,generated_text_align,device=target_model.device)
            spec_rewards.append(reward_spec)
            align_rewards.append(reward_align)
            specdec_acceptances.append(acceptance_probs_spec)
            osd_acceptances.append(acceptance_probs_osd)
            specdec_times.append(time_spec)
            osd_times.append(time_osd)
            # except:
            #     pass
        if totals > 5:
            break

    specdec_acceptances = average_accross_time(specdec_acceptances)
    osd_acceptances = average_accross_time(osd_acceptances)
    # print(specdec_acceptances)
    # print(osd_acceptances)
    # plt.figure()
    # plt.xlabel('t')
    # plt.ylabel('time')
    # plt.plot(specdec_times,'*')
    # plt.plot(osd_times,'o')
    # plt.legend(['specdec','in context'])
    # plt.show()
    # # plt.savefig("../avg_time_longer_7b_1b.jpg")

    # plt.figure()
    # plt.xlabel('t')
    # plt.ylabel('acceptance prob')
    # plt.plot(specdec_acceptances,'*')
    # plt.plot(osd_acceptances,'o')
    # plt.legend(['specdec','in context'])
    # plt.show()
    # plt.savefig("../avg_acceptance_probs_accross_time_13b_8b_prefix_middle.jpg")

    plt.figure()
    plt.xlabel('sentences')
    plt.ylabel('rewards')
    plt.plot(spec_rewards,'*')
    plt.plot(align_rewards,'o')
    plt.legend(['specdec','aligned'])
    plt.show()
    plt.savefig("figures/newstuff.jpg")
# python main.py \
#     --input "The quick brown fox jumps over the lazy " \
#     --target_model_name solidrust/Llama-3-13B-Instruct-v0.1-AWQ \
#     --approx_model_name solidrust/Meta-Llama-3-8B-Instruct-AWQ