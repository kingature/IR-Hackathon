#!/usr/bin/env python

from experiments.experiments import ChainOfThoughts, SimilarQueriesFS, SimilarQueriesZS
from models.flan_ul2 import FlanUL2Wrapper
from models.llama import Llama2Wrapper
from models.chatgpt import ChatGPTWrapper
from util.utility import get_all_datasets, get_by_priority, split_list
import sys


if __name__ == '__main__':
    dset_list = get_by_priority(0)

    if len(dset_list) == 0:
        dset_list = get_all_datasets()

    batchnum = int(sys.argv[1])
    ngpus = int(sys.argv[2])

    dset_list = split_list(dset_list, ngpus)[batchnum-1]
    if len(dset_list) == 0:
        print(f"Process {batchnum}/{ngpus} has nothing to do, exiting...")
        exit()
    print(f"Process {batchnum}/{ngpus} has datasets {dset_list}")

    flan_model = FlanUL2Wrapper(min_len=10, max_len=200, temperature=0.5, name="flan-ul2")
    llama_model = Llama2Wrapper(min_len=10, max_len=200, temperature=1.1, name="llama", modelpath="/beegfs/ws/1/s9037008-ir-hackaton-queries/models/llama2-7b-chat-pytorch")
    chatgpt_model = None    # ChatGPTWrapper(max_len=200, temperature=0.5, name="gpt")

    chain_of_thoughts = ChainOfThoughts(long_name="chain-of-thoughts", short_name="CoT", flan=flan_model, llama=llama_model, gpt=chatgpt_model, dsets=dset_list)
    chain_of_thoughts.run()
    # chain_of_thoughts.eval([chatgpt_model.name])

    similar_queries_fs = SimilarQueriesFS(long_name="similar-queries-fs", short_name="Q2E/FS", flan=flan_model, llama=llama_model, gpt=chatgpt_model, dsets=dset_list)
    similar_queries_fs.run()
    # similar_queries_fs.eval([flan_model.name, llama_model.name, chatgpt_model.name])

    similar_queries_zs = SimilarQueriesZS(long_name="similar-queries-zs", short_name="Q2E/ZS", flan=flan_model, llama=llama_model, gpt=chatgpt_model, dsets=dset_list)
    similar_queries_zs.run()
    # similar_queries_zs.eval([flan_model.name, llama_model.name, chatgpt_model.name])
