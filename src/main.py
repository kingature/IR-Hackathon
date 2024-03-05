from tira.third_party_integrations import ir_datasets
from flan_ul2 import FlanUL2Wrapper
from utility import *

TIREX_DATASETS = [
    'antique-test-20230107-training', 'argsme-touche-2021-task-1-20230209-training', 'argsme-touche-2020-task-1-20230209-training',
    'clueweb09-en-trec-web-2009-20230107-training', 'clueweb09-en-trec-web-2010-20230107-training', 'clueweb09-en-trec-web-2011-20230107-training',
    'clueweb09-en-trec-web-2012-20230107-training', 'clueweb12-touche-2020-task-2-20230209-training', 'clueweb12-touche-2021-task-2-20230209-training',
    'clueweb12-trec-misinfo-2019-20240214-training', 'clueweb12-trec-web-2013-20230107-training', 'clueweb12-trec-web-2014-20230107-training',
    'cord19-fulltext-trec-covid-20230107-training', 'cranfield-20230107-training', 'disks45-nocr-trec-robust-2004-20230209-training',
    'disks45-nocr-trec7-20230209-training', 'disks45-nocr-trec8-20230209-training', 'gov-trec-web-2002-20230209-training',
    'gov-trec-web-2003-20230209-training', 'gov-trec-web-2004-20230209-training', 'gov2-trec-tb-2004-20230209-training',
    'gov2-trec-tb-2005-20230209-training', 'gov2-trec-tb-2006-20230209-training', 'longeval-heldout-20230513-training',
    'longeval-long-september-20230513-training', 'longeval-short-july-20230513-training', 'longeval-train-20230513-training',
    'medline-2004-trec-genomics-2004-20230107-training', 'medline-2004-trec-genomics-2005-20230107-training', 'medline-2017-trec-pm-2017-20230211-training',
    'medline-2017-trec-pm-2018-20230211-training', 'msmarco-passage-trec-dl-2019-judged-20230107-training', 'msmarco-passage-trec-dl-2020-judged-20230107-training',
    'nfcorpus-test-20230107-training', 'trec-tip-of-the-tongue-dev-20230607-training', 'vaswani-20230107-training',
    'wapo-v2-trec-core-2018-20230107-training'
]


def main():
    flan = FlanUL2Wrapper(10, 200, 0.5)

    for dset_name in TIREX_DATASETS:
        dataset = ir_datasets.load(f'ir-benchmarks/{dset_name}')
        json_res = flan.chain_of_thoughts(list(dataset.queries_iter()))
        save_queries_cot("flan-ul2", dset_name, list(dataset.queries_iter()), json_res)


if __name__ == '__main__':
    main()
