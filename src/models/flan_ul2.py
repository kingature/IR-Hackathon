from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

from layout import Layout


class FlanUL2Wrapper(Layout):
    def __init__(self, min_len, max_len, temperature):
        self.min_len = min_len
        self.max_len = max_len
        self.temperature = temperature

        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

    def process_queries(self, prompts):
        result = []
        batches = torch.utils.data.DataLoader(prompts, batch_size=5)

        for batch in batches:
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
            output = self.model.generate(inputs, do_sample=True, min_length=self.min_len, max_length=self.max_len,
                                         temperature=self.temperature)
            result.extend([elem for elem in self.tokenizer.batch_decode(output, skip_special_tokens=True)])
        return result

    def chain_of_thoughts(self, queries):
        def add_context(query):
            return (f'Answer the following query:'
                    f''
                    f'{query}'
                    f''
                    f'Give the rationale before answering.')

        prompts = [add_context(query.default_text()) for query in queries]
        return self.process_queries(prompts)

    def similar_queries_fs(self, queries):
        def add_context(query):
            return (f'Suggest 5 queries that are similar to the following query. Here are some examples first:'
                    f''
                    f'Original query: How to tie a windsor knot?'
                    f'Similar query: Instructions for tying a windsor knot'
                    f''
                    f'Original query: How is the weather tomorrow morning?'
                    f'Similar query: Weather tomorrow morning'
                    f''
                    f'Original query: Simple vegan cooking recipes'
                    f'Similar query: What are some delicious and simple vegan cooking recipes?'
                    f''
                    f'Query: {query}')

        prompts = [add_context(query.default_text()) for query in queries]
        return self.process_queries(prompts)

    def similar_queries_zs(self, queries):
        def add_context(query):
            return (f'Suggest 5 queries that are similar to the following query:'
                    f''
                    f'Query: {query}')

        prompts = [add_context(query.default_text()) for query in queries]
        return self.process_queries(prompts)
