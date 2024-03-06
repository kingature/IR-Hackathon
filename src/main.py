#!/usr/bin/env python

from experiments.experiments import ChainOfThoughts, SimilarQueriesFS, SimilarQueriesZS
from models.flan_ul2 import FlanUL2Wrapper
from models.llama import Llama2Wrapper
from util.utility import get_all_datasets, get_similar
import sys

def split_list(lst, parts):
    if parts > len(lst):
        ret = [[lst[i]] for i in range(len(lst))]
        if len(ret) < parts:
            for i in range(parts-len(ret)):
                ret.append([])
#        print("returning " + str(ret))
        return ret

    avg = len(lst) / float(parts)
    out = []
    last = 0.0

    while last < len(lst):
        out.append(lst[int(last):int(last + avg)])
        last += avg

    return out

if __name__ == '__main__':
    dset_list = [get_similar('marco')]
    dset_list = ['msmarco-passage-trec-dl-2019-judged-20230107-training', 'msmarco-passage-trec-dl-2020-judged-20230107-training', 'longeval-long-september-20230513-training', 'longeval-short-july-20230513-training', 'longeval-train-20230513-training', 'longeval-heldout-20230513-training', 'antique-test-20230107-training', 'argsme-touche-2021-task-1-20230209-training', 'argsme-touche-2020-task-1-20230209-training']
    dset_list = [dset_list[3]]

    if len(dset_list) == 0:
        dset_list = get_all_datasets()

    batchnum = int(sys.argv[1])
    ngpus = int(sys.argv[2])

    print(batchnum, ngpus)
    dset_list = split_list(dset_list, ngpus)[batchnum-1]
    print(dset_list)
    if len(dset_list) == 0:
        print(f"Process {batchnum} has nothing to do, exiting...")
        exit()

    flan_model = FlanUL2Wrapper(min_len=10, max_len=200, temperature=0.5)
    llama_model = Llama2Wrapper(min_len=10, max_len=200, temperature=1.1, modelpath="/beegfs/ws/1/s9037008-ir-hackaton-queries/models/llama2-7b-chat-pytorch")

#    chain_of_thoughts = ChainOfThoughts(name="CoT", flan=flan_model, llama=llama_model, dsets=dset_list)
#    chain_of_thoughts.run()
#    chain_of_thoughts.eval(["chain-of-thoughts"], ["llama", "flan-ul2"])
#
#    similar_queries_fs = SimilarQueriesFS(name="Q2E/FS", flan=flan_model, llama=llama_model, dsets=dset_list)
#    similar_queries_fs.run()
#    similar_queries_fs.eval(["chain-of-thoughts"], ["llama", "flan-ul2"])

    similar_queries_zs = SimilarQueriesZS(name="Q2E/ZS", flan=flan_model, llama=llama_model, dsets=dset_list)
    similar_queries_zs.run()
#    similar_queries_zs.eval(["chain-of-thoughts"], ["llama", "flan-ul2"])
