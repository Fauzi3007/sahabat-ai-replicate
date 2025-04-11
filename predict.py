from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory"""
        print("Loading model...")
        model_name = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded!")

    def predict(
        self,
        prompt: str = Input(description="Prompt berisi pertanyaan atau instruksi untuk model"),
        max_new_tokens: int = Input(description="Jumlah maksimum token yang akan dihasilkan", default=512),
        temperature: float = Input(description="Nilai temperature untuk sampling (0.1-2.0)", default=0.7, ge=0.1, le=2.0),
        top_p: float = Input(description="Nilai top-p untuk sampling", default=0.95, ge=0.0, le=1.0),
        top_k: int = Input(description="Nilai top-k untuk sampling", default=50, ge=0),
        repetition_penalty: float = Input(description="Penalty untuk repetisi", default=1.0, ge=0.0),
    ) -> str:
        """Jalankan model gemma2-9b-cpt-sahabatai-v1-instruct dengan input yang diberikan"""
        
        # Format untuk model Gemma2 instruction
        if not prompt.strip().startswith("<start_of_turn>"):
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            formatted_prompt = prompt
            
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate output
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the model's response
        try:
            model_response = decoded_output.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
        except:
            model_response = decoded_output
            
        return model_response