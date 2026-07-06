from datasets import load_dataset
import random
import numpy as np
import os
from transformers import AutoTokenizer

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

MODEL_NAME = os.getenv("MODEL_NAME","Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class DummyDataset:

    def get_prompt(
        self,
        context_length,
        sample_index=0
    ):

        prompt = (
            "hello "
            * context_length
        )

        return prompt

class LongBench:
    def __init__(self):
        self.dataset = load_dataset(
                        "hotpotqa/hotpot_qa",
                        "distractor",
                        split="validation"
                    )
    def build_context(self, sample):
        """
        Convert the HotpotQA context format into one text block.
        """

        context = ""

        titles = sample["context"]["title"]
        paragraphs = sample["context"]["sentences"]

        for title, sentences in zip(titles, paragraphs):
            context += f"{title}\n"
            context += " ".join(sentences)
            context += "\n\n"

        return context

    def get_prompt(
        self,
        context_length,
        sample_index=None
    ):
        #candidates = [x for x in self.dataset if abs(x["length"]-context_length)< 256]
        if sample_index is None:
            sample = random.choice(self.dataset)
        else:
            sample = self.dataset[sample_index]
        #sample = self.dataset[sample_index]
        #print("[DEBUG] Sample:", sample.keys())
        sample["context"] = self.build_context(sample)
        #print("BEFORE TRUNCATE [DEBUG] context:", sample["context"])

        context_ids = tokenizer.encode(sample["context"],add_special_tokens=False)
        #print("BEFORE TRUNCATE context_ids:", len(context_ids))
        context_ids = context_ids[:context_length]
        #print("AFTER TRUNCATE context_ids:", len(context_ids))
        sample["context"] = tokenizer.decode(context_ids,skip_special_tokens=True)
        #print("AFTER TRUNCATE [DEBUG] context:", sample["context"])

        prompt = f"""Context:{sample['context']} Question:{sample['question']}"""

        return prompt

class PromptBuilder:
    def __init__(self, dataset_choice='dummy'):
        self.dataset_choice = dataset_choice
        if self.dataset_choice == 'dummy':
            self.dataset = DummyDataset()
        elif self.dataset_choice == 'longbench':
            self.dataset = LongBench()
        else:
            raise ValueError(f"Unknown Dataset choice {dataset_choice}")
    
    def get_prompt(self, context_length, sample_index=None):
        return self.dataset.get_prompt(context_length, sample_index)
