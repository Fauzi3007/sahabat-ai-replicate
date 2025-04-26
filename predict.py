import torch
from transformers import pipeline
from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def setup(self):
        model_id = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
        self.terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def predict(
        self,
        prompt: str = Input(description="Prompt dari user"),
        max_new_tokens: int = Input(description="Jumlah token yang dihasilkan", default=512),
        temperature: float = Input(description="Tingkat kreativitas (semakin rendah = semakin deterministik)", default=0.7),
        top_p: float = Input(description="Top-p sampling", default=0.9),
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        output = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.terminators,
            do_sample=True
        )
        return output[0]["generated_text"]
