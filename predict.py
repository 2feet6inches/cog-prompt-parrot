import os
from typing import Optional, List

import torch
import torch.nn as nn
from torch import autocast
from PIL import Image
from cog import BasePredictor, Input, Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

start_token = "<BOP>"
pad_token = "<PAD>"
end_token = "<EOP>"
special_tokens_dict = {
    'bos_token': start_token, 
    'eos_token': end_token, 
    'pad_token': pad_token
}

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = AutoModelForCausalLM.from_pretrained("./model")
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir="./model")

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="")
    ) -> str:

        style = "Manual_Prompt" #@param ["Starting_Words", "Promptless", "Manual_Prompt"]
        manual_prompt=prompt #@param{type:"string"}
        num_start_words = 2 #@param{type:"integer"}

        num_prompts_to_generate=10 #@param{type:"integer"}
        #markdown Maximum and min length of the generated prompts. Will cut off mid word. This is expected behavior
        max_prompt_length=50 #@param{type:"integer"}
        min_prompt_length=30 #@param{type:"integer"}
        #@markdown `temperature`: Default: 1.2. Turning up will inject more chaos.
        temperature=1.2 #@param{type:"number"}
        #@markdown `top_k`: Default 70. The number of top tokens returned by the AI. Will be randomly selected for generation.
        top_k=70 #@param{type:"integer"}
        #@markdown `top_p`: Default 0.9. The total percent to consider from the `top_k` returned tokens. For more information refer to [this guide!]( https://docs.cohere.ai/token-picking/)
        top_p=0.9 #@param{type:"number"}


        if style == "Starting_Words":
            prompt_starts = list(set([" ".join(p.split()[0:num_start_words]).replace(",", "") for p in all_prompts if len(p.split()) > 1]))
            prompt = random.choice(prompt_starts)
        elif style == "Promptless":
            prompt = start_token
        else:
            if not manual_prompt:
                raise UserWarning("manual_prompt must be at least 1 letter")
            prompt = manual_prompt
    
        encoded_prompt = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        encoded_prompt = encoded_prompt.to(self.model.device)

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=max_prompt_length,
            min_length=min_prompt_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_prompts_to_generate,
            pad_token_id=self.tokenizer.pad_token_id # gets rid of warning
            )

        tokenized_start_token = self.tokenizer.encode(start_token)

        prompt_overrides = ""
        prompts_to_run = []
        for generated_sequence in output_sequences:
            # precision is a virtue
            tokens = []
            for i, s in enumerate(generated_sequence):
                if s in tokenized_start_token and i != 0:
                    if len(tokens) >= min_prompt_length:
                        break
                tokens.append(s)

            text = self.tokenizer.decode(tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            text = text.strip().replace("\n", " ").replace("/", ",") # / remove slash. It causes problems in namings
            prompts_to_run.append(text)
        return "\n".join(prompts_to_run)
