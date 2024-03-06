#!/usr/bin/env python

from src.experiments.experiments import ChainOfThoughts, SimilarQueriesFS, SimilarQueriesZS
from src.models.chatgpt import ChatGPTWrapper
from src.models.flan_ul2 import FlanUL2Wrapper
from src.models.llama import Llama2Wrapper
from src.util.utility import get_all_datasets, get_similar

if __name__ == '__main__':
    dset_list = [get_similar('marco')]

    if len(dset_list) == 0:
        dset_list = get_all_datasets()

    flan_model = None   # FlanUL2Wrapper(min_len=10, max_len=200, temperature=0.5, name="flan-ul2")
    llama_model = None  # Llama2Wrapper(min_len=10, max_len=200, temperature=1.1, name="llama")
    chatgpt_model = ChatGPTWrapper(max_len=200, temperature=0.5, name="gpt")

    chain_of_thoughts = ChainOfThoughts(long_name="chain-of-thoughts", short_name="CoT", flan=flan_model, llama=llama_model, gpt=chatgpt_model, dsets=dset_list)
    chain_of_thoughts.run()
    chain_of_thoughts.eval([chatgpt_model.name])

    # similar_queries_fs = SimilarQueriesFS(long_name="similar-queries-fs", short_name="Q2E/FS", flan=flan_model, llama=llama_model, gpt=chatgpt_model, dsets=dset_list)
    # similar_queries_fs.run()
    # similar_queries_fs.eval([flan_model.name, llama_model.name, chatgpt_model.name])

    # similar_queries_zs = SimilarQueriesZS(long_name="similar-queries-zs", short_name="Q2E/ZS", flan=flan_model, llama=llama_model, gpt=chatgpt_model, dsets=dset_list)
    # similar_queries_zs.run()
    # similar_queries_zs.eval([flan_model.name, llama_model.name, chatgpt_model.name])
