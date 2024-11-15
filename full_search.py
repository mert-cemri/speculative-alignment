import torch
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
device = "cuda:1"
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    # attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
def rate_answer(question,answer):
    conv1 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    conv1_formatted = rm_tokenizer.apply_chat_template(conv1, tokenize=False)
    conv1_tokenized = rm_tokenizer(conv1_formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        score1 = rm(**conv1_tokenized).logits[0][0].item()
    return score1

from getpass import getpass
model = "NousResearch/Meta-Llama-3-8B-Instruct"
api_key = "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"
base_url="http://localhost:8000/v1"

from openai import OpenAI
import re

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

def chat_completion_request_openai(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]

    chat_response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=1.0,
    max_tokens=1500
    )
    if chat_response.choices:
        completion_text = chat_response.choices[0].message.content
    else:
        completion_text = None
    return completion_text

def get_answer(question, draft_answer='', critique=''):
    prompt = (
    f"Question: {question}\n"
    f"Draft answer: {draft_answer}\n"
    f"Previous thoughts: {critique}\n"
    "Please improve the draft answer based on the previous answers. Return output in this manner:\n"
    "Reasoning Process: <step-by-step reasoning process>\n"
    "Final answer: <imrpoved and verified answer>\n"
    )
    return chat_completion_request_openai(prompt)

import random
import numpy as np
import math

max_children = 3
exploration_weight = 1.41

class Node:
    def __init__(self,question,answer,prev_answers='',parent=None):
        self.question = question
        self.answer = answer
        self.prev_answers = prev_answers
        self.parent = parent
        self.children = []
        self.value = 0.0
        self.visits = 0

    def is_fully_expanded(self):
        # check if we can expand further
        return len(self.children) >= max_children
    
    def best_child(self):
        #return index of best child
        choice_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float('inf')
            else:
                weight = child.value/child.visits + exploration_weight* math.sqrt((2*math.log(self.visits) / child.visits))
            choice_weights.append(weight)
        return self.children[np.argmax(choice_weights)] 
    
    def most_visited_child(self):
        # return most visited child
        return max(self.children, key=lambda child:child.visits)
    
    def add_child(self,child_node):
        # add child_node to current node
        self.children.append(child_node)

class MCTS:
    def __init__(self,question='',max_iterations=3):
        self.question = question
        self.max_iterations = max_iterations
        first_answer = get_answer(question)
        self.root = Node(question,first_answer)
    def search(self):
        for i in range(self.max_iterations):
            print(f"Iteration: {i}/{self.max_iterations}")
            node = self.select(self.root)
            print(f"Selected node: {node.answer}")
            if not node.is_fully_expanded():
                node = self.expand(node)
            reward = self.evaluate(node)
            print(f"Assigned reward: {reward}")
            self.backpropagate(node,reward)
        print(f"Visits to most visited child: {self.root.most_visited_child().visits}")
        return self.root.most_visited_child().answer
    def select(self, node):
        while node.is_fully_expanded() and node.children:
            # if node has a child and it is fully expanded, go deep
            # if at a leaf or not fully expanded, return that node and expand
            node = node.best_child()
        return node
    def expand(self, node):
        for j in range(max_children - len(node.children)):
            child_node = Node(self.question, node.answer, parent=node)
            node.add_child(child_node)
            answer = get_answer(self.question, child_node.answer, child_node.prev_answers)
            child_node.prev_answers += answer
            print(f"\n --Answer {j}--:\n{answer}")
            child_node.answer = answer
        return random.choice(node.children)
    def evaluate(self, node):
        return rate_answer(self.question, node.answer)
    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
        
question = "What is the capital of France?"
mcts = MCTS(question,max_iterations=2)
best_answer = mcts.search()

print(mcts.root.answer)