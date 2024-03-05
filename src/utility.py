import json
import os


def save_queries_cot(model_name, dset_name, queries, responses):
    results = []
    for query, response in zip(queries, responses):
        elem = {
            "query-id": query.query_id,
            "query-text": query.default_text(),
            "query-CoT": response
        }
        results.append(elem)
    path = f'generated/{model_name}/cot'

    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{path}/{dset_name}.json', 'w') as file:
        file.write(json.dumps(results, indent=4))
