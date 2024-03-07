#!/usr/bin/env python

import json
import os
import matplotlib
from matplotlib import pyplot as plt
from statistics import mean, stdev

# colors
orange = "#ffa500"
violet = "#868ad1"
darkorange = "#f47264"
turquoise = "#84cbc5"
purple = "#7030a0"
navyblue = "#1f75bb"

def load_evals():
    cot_evals = set(os.listdir("generated/chain-of-thoughts/evaluation"))
    fs_evals = set(os.listdir("generated/similar-queries-fs/evaluation"))
    zs_evals = set(os.listdir("generated/similar-queries-zs/evaluation"))

    # check if all dirs contain evals for the same datasets and skip datasets for which not all evals are present
    if not (cot_evals == fs_evals and cot_evals == zs_evals):
        print("There are evaluation data mismatches!")
        all_dss = cot_evals.union(fs_evals, zs_evals)
        eval_dss = cot_evals.intersection(fs_evals, zs_evals)
        print(f"Evaluating for {len(eval_dss)}/{len(all_dss)}, skipping {all_dss - eval_dss}")
    else:
        eval_dss = cot_evals

    data = {}
    for ds in eval_dss:
        dshat = ds[5:-6]
        data[dshat] = {"baselines": {}}
        for exp in ["chain-of-thoughts", "similar-queries-fs", "similar-queries-zs"]:
            data[dshat][exp] = {}
            with open(f"generated/{exp}/evaluation/{ds}", "r") as f:
                tmpdata = [json.loads(line) for line in f.readlines()]
                for line in tmpdata:
                    # baselines are always identical between models and experiments
                    if line["name"] in ["BM25", "BM25+RM3", "BM25+KL"]:
                        data[dshat]["baselines"][line["name"]] = {}
                        data[dshat]["baselines"][line["name"]]["NDCG@10"] = line["ndcg_cut.10"]
                        data[dshat]["baselines"][line["name"]]["Recall@1000"] = line["recall_1000"]
                    else:
                        data[dshat][exp][line["model"]] = {}
                        data[dshat][exp][line["model"]]["NDCG@10"] = line["ndcg_cut.10"]
                        data[dshat][exp][line["model"]]["Recall@1000"] = line["recall_1000"]

    return data

def barchart(data, name, score):
    experiments = ["chain-of-thoughts", "similar-queries-fs", "similar-queries-zs"]
#    plt.xkcd()
    matplotlib.rcParams.update({'font.size': 13})
    plt.figure(figsize=(8, 6))

    barwidth = 0.9
    # baseline
    xs_baseline = [1,2,3]
    ys_baseline = [data[name]["baselines"][m][score] for m in ["BM25", "BM25+RM3", "BM25+KL"]]
    plt.bar(xs_baseline, ys_baseline, barwidth, label=["BM25", "BM25+RM3", "BM25+KL"], color=[orange, violet, darkorange])
    plt.hlines(ys_baseline, [x-barwidth/2 for x in xs_baseline], 15+barwidth/2, colors=[orange, violet, darkorange], linestyles="dashed")
    # flan
    xs_flan = [5,9,13]
    ys_flan = [data[name][exp]["flan-ul2"][score] for exp in experiments]
    plt.bar(xs_flan, ys_flan, barwidth, label="flan", color=turquoise)
    # llama
    xs_llama = [6,10,14]
    ys_llama = [data[name][exp]["llama"][score] for exp in experiments]
    plt.bar(xs_llama, ys_llama, barwidth, label="llama", color=purple)
    # gpt
    xs_gpt = [7,11,15]
    ys_gpt = [data[name][exp]["gpt"][score] for exp in experiments]
    plt.bar(xs_gpt, ys_gpt, barwidth, label="gpt", color=navyblue)

    ymin = min(ys_baseline + ys_flan + ys_llama + ys_gpt)
    ymax = max(ys_baseline + ys_flan + ys_llama + ys_gpt)
    scale_factor = 1.5
    plt.ylim(max(0, ymin-2*(ymax-ymin)), min(1, ymax+0.5*(ymax-ymin)))

    plt.xticks([2,6,10,14], ["Baseline", "CoT/ZS", "Q2E/FS", "Q2E/ZS"], rotation=0)
    plt.xlabel("Experiment")
    plt.ylabel(score)
    plt.title(name+"\n", fontweight="bold")
    plt.legend(loc="lower right", framealpha=0.95)
#    plt.show()
    plt.savefig("fig/" + name + "-" + score + ".png", dpi=500)
    plt.close()

def barchart_all(data):
    bm25_recs = [data[d]["baselines"]["BM25"]["Recall@1000"] for d in data.keys()]
    bm25_precs = [data[d]["baselines"]["BM25"]["NDCG@10"] for d in data.keys()]
    bm25rm3_recs = [data[d]["baselines"]["BM25+RM3"]["Recall@1000"] for d in data.keys()]
    bm25rm3_precs = [data[d]["baselines"]["BM25+RM3"]["NDCG@10"] for d in data.keys()]
    bm25kl_recs = [data[d]["baselines"]["BM25+KL"]["Recall@1000"] for d in data.keys()]
    bm25kl_precs = [data[d]["baselines"]["BM25+KL"]["NDCG@10"] for d in data.keys()]
    cot_flan_recs = [data[d]["chain-of-thoughts"]["flan-ul2"]["Recall@1000"] for d in data.keys()]
    cot_flan_precs = [data[d]["chain-of-thoughts"]["flan-ul2"]["NDCG@10"] for d in data.keys()]
    cot_llama_recs = [data[d]["chain-of-thoughts"]["llama"]["Recall@1000"] for d in data.keys()]
    cot_llama_precs = [data[d]["chain-of-thoughts"]["llama"]["NDCG@10"] for d in data.keys()]
    cot_gpt_recs = [data[d]["chain-of-thoughts"]["gpt"]["Recall@1000"] for d in data.keys()]
    cot_gpt_precs = [data[d]["chain-of-thoughts"]["gpt"]["NDCG@10"] for d in data.keys()]
    fs_flan_recs = [data[d]["similar-queries-fs"]["flan-ul2"]["Recall@1000"] for d in data.keys()]
    fs_flan_precs = [data[d]["similar-queries-fs"]["flan-ul2"]["NDCG@10"] for d in data.keys()]
    fs_llama_recs = [data[d]["similar-queries-fs"]["llama"]["Recall@1000"] for d in data.keys()]
    fs_llama_precs = [data[d]["similar-queries-fs"]["llama"]["NDCG@10"] for d in data.keys()]
    fs_gpt_recs = [data[d]["similar-queries-fs"]["gpt"]["Recall@1000"] for d in data.keys()]
    fs_gpt_precs = [data[d]["similar-queries-fs"]["gpt"]["NDCG@10"] for d in data.keys()]
    zs_flan_recs = [data[d]["similar-queries-zs"]["flan-ul2"]["Recall@1000"] for d in data.keys()]
    zs_flan_precs = [data[d]["similar-queries-zs"]["flan-ul2"]["NDCG@10"] for d in data.keys()]
    zs_llama_recs = [data[d]["similar-queries-zs"]["llama"]["Recall@1000"] for d in data.keys()]
    zs_llama_precs = [data[d]["similar-queries-zs"]["llama"]["NDCG@10"] for d in data.keys()]
    zs_gpt_recs = [data[d]["similar-queries-zs"]["gpt"]["Recall@1000"] for d in data.keys()]
    zs_gpt_precs = [data[d]["similar-queries-zs"]["gpt"]["NDCG@10"] for d in data.keys()]

    # recall
    plt.bar(1, mean(bm25_recs), yerr=stdev(bm25_recs), color=orange)
    plt.bar(2, mean(bm25rm3_recs), yerr=stdev(bm25rm3_recs), color=violet)
    plt.bar(3, mean(bm25kl_recs), yerr=stdev(bm25kl_recs), color=darkorange)
    plt.bar(5, mean(cot_flan_recs), yerr=stdev(cot_flan_recs), color=turquoise)
    plt.bar(6, mean(cot_llama_recs), yerr=stdev(cot_llama_recs), color=purple)
    plt.bar(7, mean(cot_gpt_recs), yerr=stdev(cot_gpt_recs), color=navyblue)
    plt.bar(9, mean(fs_flan_recs), yerr=stdev(fs_flan_recs), color=turquoise)
    plt.bar(10, mean(fs_llama_recs), yerr=stdev(fs_llama_recs), color=purple)
    plt.bar(11, mean(fs_gpt_recs), yerr=stdev(fs_gpt_recs), color=navyblue)
    plt.bar(13, mean(zs_flan_recs), yerr=stdev(zs_flan_recs), color=turquoise)
    plt.bar(14, mean(zs_llama_recs), yerr=stdev(zs_llama_recs), color=purple)
    plt.bar(15, mean(zs_gpt_recs), yerr=stdev(zs_gpt_recs), color=navyblue)
    plt.show()
    plt.savefig()
    plt.close()

    #ndcg

def main():
    data = load_evals()
    datasets = list(data.keys())
    experiments = list(data[datasets[0]].keys())
    scores = ["Recall@1000", "NDCG@10"]

#    for d in datasets:
#        for s in scores:
#            barchart(data, d, s)

    barchart_all(data)

if __name__ == "__main__":
    main()
