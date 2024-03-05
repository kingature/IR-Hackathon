import jsonlines
import os
import json


def save_queries(model_name, dset_name, queries, responses):
    path = f'generated/{model_name}'

    if not os.path.exists(path):
        os.makedirs(path)

    with jsonlines.open(f'{path}/{dset_name}.jsonl', mode='w') as writer:
        for query, response in zip(queries, responses):
            elem = {
                "query-id": query.query_id,
                "query-text": query.default_text(),
                "response": response,
                "metadata": {"model": model_name}
            }
            writer.write(elem)


def read_queries(model_name, dset_name):
    path = f'chain-of-thoughts/generated/{model_name}/{dset_name}.jsonl'
    with open(path) as file:
        data = [line.rstrip() for line in file]

    return [json.loads(entry) for entry in data]
