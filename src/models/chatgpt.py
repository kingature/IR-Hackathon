from openai import OpenAI
from src.models.layout import Layout


class ChatGPTWrapper(Layout):

    def __init__(self, max_len, temperature, name):
        # TODO Remove API KEY
        super().__init__(name)
        self.client = OpenAI(api_key='...')
        self.max_len = max_len
        self.temperature = temperature

    def process_queries(self, prompts):
        result = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": f"{prompt}"},
                ],
                max_tokens=self.max_len,
                temperature=self.temperature
            )
            result.append(response.choices[0].message.content)
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
