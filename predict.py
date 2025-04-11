import torch
from cog import BasePredictor, Input
from transformers import pipeline, AutoTokenizer

class Predictor(BasePredictor):
    def setup(self):
        self.model_id = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
        self.tokenizer = self.pipeline.tokenizer
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def predict(
        self,
        message: str = Input(description="Masukkan pertanyaan dalam Bahasa Indonesia, Jawa, atau Sunda."),
        max_new_tokens: int = Input(description="Jumlah maksimum token keluaran", default=256),
    ) -> str:
        messages = [{"role": "user", "content": message}]

        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.terminators,
        )

        return outputs[0]["generated_text"]
