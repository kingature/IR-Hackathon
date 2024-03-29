#!/usr/bin/env python3

import json
import math
from tira.third_party_integrations import ir_datasets
from torch import bfloat16
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizerFast, TextStreamer


class Llama2Wrapper:
    def __init__(self, min_len, max_len, temperature, modelpath="../models/llama2-7b-chat-pytorch", **kwargs):
        self.min_len = min_len
        self.max_len = max_len
        self.temperature = temperature
        self.tokenizer = LlamaTokenizerFast.from_pretrained(modelpath, use_default_system_prompt=False)
        self.model = LlamaForCausalLM.from_pretrained(modelpath, torch_dtype=bfloat16, device_map="auto", load_in_4bit=True)
        self.sysprompt = "Answer the following query. Be short and concise, 50 words at max. Answer in full sentences."
        self.prompt = "<s>[INST] <<SYS>>\n" + self.sysprompt + "\n<</SYS>>\n\n"

    def tokenize_and_count(self, string):
        model_input = self.tokenizer(string, return_tensors="pt").to("cuda")
        num_tokens = model_input["input_ids"].shape[-1]
        return model_input, num_tokens

    def process_query(self, query, prompttype="plain"):
        print("Query: " + query)
        supported_prompttypes = ["plain", "cot"]
        if prompttype not in supported_prompttypes:
            print(f"Prompttype {prompttype} is unsupported! Please use one of {supported_prompttypes}.")
            exit(1)
        match prompttype:
            case "plain":
                query = self.prompt + query.strip() + " [/INST]"
            case "cot":
                query = "<s>[INST] <<SYS>>\nBe short and concise, 100 words max. Answer in full sentences, while briefly writing down your steps towards the response.\n<</SYS>>\n\nAnswer the following query: " + query.strip() + ". Give the rationale before answering. [/INST]"

        input, num_input_tokens = self.tokenize_and_count(query)
        output = self.model.generate(**input, max_new_tokens=self.max_len, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, temperature=self.temperature, min_length=self.min_len)
        output = self.tokenizer.decode(output[0][num_input_tokens:], skip_special_tokens=True).strip()
        print("\nResponse: " + output)
        return output

    def chain_of_thoughts(self, queries):
        responses = []
        for idx, q in enumerate(queries):
            print("\n[" + " " * (math.ceil(math.log10(len(queries)) - 1) - math.floor(math.log10(idx + 1))) + str(idx+1) + "/" + str(len(queries)) + "]", end=" ")
            output = self.process_query(q.text, prompttype="cot")
            responses.append(output)
        return responses

    def extract_keywords(self):
        with open("msmarco-passage-trec-dl-2019-judged-20230107-training.jsonl", "r") as f:
            responses = [json.loads(line)["response"] for line in f.readlines()]

        input = "<s>[INST] <<SYS>>\n" + "Extract 10 relevant keywords from the following document. Also think of 10 other relevant keywords. List your keywords comma-separated." + "\n<</SYS>>\n\n"
        input += responses[0] + " [/INST]"
        input, num_input_tokens = self.tokenize_and_count(input)
        output = self.model.generate(**input, max_new_tokens=200, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, temperature=5.1, num_return_sequences=3)
        print(output)
        output0 = self.tokenizer.decode(output[0][num_input_tokens:], skip_special_tokens=True).strip()
        output1 = self.tokenizer.decode(output[1][num_input_tokens:], skip_special_tokens=True).strip()
        output2 = self.tokenizer.decode(output[2][num_input_tokens:], skip_special_tokens=True).strip()

        print("\nResponse: " + "\n".join([output0, output1, output2]))

    def vary_queries(self, queries):
        for q in queries:
            input = "<s>[INST] <<SYS>>\n" + "Assume the following as a query in a natural language retrieval system. Operate in 2 steps: First, write a list of relevant terms. Second, formulate one query with the terms." + "\n<</SYS>>\n\n"
            input += q.text + " [/INST]"
            input, num_input_tokens = self.tokenize_and_count(input)
            output = self.model.generate(**input, max_new_tokens=200, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, temperature=1.1, num_return_sequences=1)
            output = self.tokenizer.decode(output[0][num_input_tokens:], skip_special_tokens=True).strip()
            print("\nQuery: " + q.text + "\nResponse: " + output)
