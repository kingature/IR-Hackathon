#!/usr/bin/env python3

import json
import math
import tqdm

from torch import bfloat16
from transformers import LlamaForCausalLM, LlamaTokenizerFast, TextStreamer

from layout import Layout

def mp_helper(params):
    queries = params[0]
    experiment = params[1]
    show_output = params[2]
    self = params[3]

    return list(range(len(queries)))

    tokenizer = LlamaTokenizerFast.from_pretrained(modelpath, use_default_system_prompt=False)
    model = LlamaForCausalLM.from_pretrained(modelpath, torch_dtype=bfloat16, device_map="auto", load_in_4bit=True)

    outputs = []
    for q in queries:
        outputs.append(self.process_query(q.text, prompt, prompttype=experiment, show_output=show_output, model=model, tokenizer=tokenizer))

    return outputs

class Llama2Wrapper(Layout):
    def __init__(self, min_len, max_len, temperature, modelpath="../models/llama2-7b-chat-pytorch", **kwargs):
        self.min_len = min_len
        self.max_len = max_len
        self.temperature = temperature
        self.modelpath = modelpath
        self.tokenizer = LlamaTokenizerFast.from_pretrained(modelpath, use_default_system_prompt=False)
        self.model = LlamaForCausalLM.from_pretrained(modelpath, torch_dtype=bfloat16, device_map="auto", load_in_4bit=True)

    def make_prompt(self, sysprompt, prompt):
        return "<s>[INST] <<SYS>>\n" + sysprompt + "\n<</SYS>>\n\n" + prompt.strip() + " [/INST]"

    def tokenize_and_count(self, string, tokenizer):
        model_input = self.tokenizer(string, return_tensors="pt").to("cuda")
        num_tokens = model_input["input_ids"].shape[-1]
        return model_input, num_tokens

    def process_query(self, query, model, tokenizer, prompttype="zs", show_output=True, **kwargs):
        if show_output:
            print("Query: " + query)

        supported_prompttypes = ["zs", "cot"]
        if prompttype not in supported_prompttypes:
            print(f"Prompttype {prompttype} is unsupported! Please use one of {supported_prompttypes}.")
            exit(1)
        match prompttype:
            case "zs":
                sysprompt = "Answer the following query. Be short and concise, 50 words at max. Answer in full sentences."
                input = self.make_prompt(sysprompt, query)
            case "cot":
                sysprompt = "Be short and concise, 100 words max. Answer in full sentences, while briefly writing down your steps towards the response."
                input = self.make_prompt(sysprompt, query)

        input, num_input_tokens = self.tokenize_and_count(query, tokenizer)
        output = model.generate(**input, max_new_tokens=self.max_len, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=self.temperature, min_length=self.min_len)
        output = tokenizer.decode(output[0][num_input_tokens:], skip_special_tokens=True).strip()
        print("\nResponse: " + output)
        return output

    def process_queries(self, queries, experiment="cot", show_output=True):
        responses = []
        if show_output:
            for idx, q in enumerate(queries):
                print("\n[" + " " * (math.ceil(math.log10(len(queries)) - 1) - math.floor(math.log10(idx + 1))) + str(idx+1) + "/" + str(len(queries)) + "]", end=" ")
                output = self.process_query(q.text, prompttype=experiment, show_output=True, model=self.model, tokenizer=self.tokenizer)
                responses.append(output)

        else:
            ngpus = torch.cuda.device_count()
            if ngpus > 1:
                with multiprocessing.Pool(ngpus) as pool:
                    print(f"Distributing {len(queries)} queries across {ngpus}")
                    batches = [queries[i*ngpus : i*ngpus+ngpus] for i in range(ngpus+1)]
                    batches[-2] += batches[-1]
                    batches = batches[:-1]
                    for b in batches:
                        print(len(b))

                    for b in batches:
                        output = pool.map(mp_helper, (b, experiment, True, self))
#                        output = self.process_query(q.text, prompttype=experiment, show_output=False, pid)
                        responses += output

            else:
                for q in tqdm.tqdm(queries):
                    output = self.process_query(q.text, prompttype=experiment, show_output=False, model=self.model, tokenizer=self.tokenizer)
                    responses.append(output)

        return responses

    def chain_of_thoughts(self, queries):
        return self.process_queries(queries, experiment="cot", show_output=False)

    def similar_queries_fs(self, queries):
        return self.process_queries(queries, experiment="fs", show_output=False)

    def similar_queries_zs(self, queries):
        return self.process_queries(queries, experiment="zs", show_output=False)

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

if __name__ == "__main__":
    from tira.third_party_integrations import ir_datasets
    import psutil
    import torch
    import multiprocessing

    ds = "msmarco-passage-trec-dl-2019-judged-20230107-training"
    ds = ir_datasets.load("ir-benchmarks/" + ds)
    queries = list(ds.queries)
    print(queries[0])
    print("loading llama...")
    llama = Llama2Wrapper(20, 200, 1.1, modelpath="/home/s9037008/ir/models/llama2-7b-chat-pytorch")
    print("starting to generate queries...")
    out = llama.chain_of_thoughts(queries)
    print(out)