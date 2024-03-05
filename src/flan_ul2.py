from transformers import T5ForConditionalGeneration, AutoTokenizer


class FlanUL2Wrapper:

    def __init__(self, min_len, max_len, temperature):
        self.min_len = min_len
        self.max_len = max_len
        self.temperature = temperature

        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

    def process_queries(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
        output = self.model.generate(inputs, do_sample=True, min_length=self.min_len, max_length=self.max_len,
                                     temperature=self.temperature)
        return [elem for elem in self.tokenizer.batch_decode(output, skip_special_tokens=True)]

    def chain_of_thoughts(self, queries):
        def add_context(query):
            return (f'Answer the following query:'
                    f''
                    f'{query}'
                    f''
                    f'Give the rationale before answering.')

        prompts = [add_context(query.default_text()) for query in queries]
        return self.process_queries(prompts)
