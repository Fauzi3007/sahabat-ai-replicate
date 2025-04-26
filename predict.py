from cog import BasePredictor, Input, Path
import os
import json
import subprocess
import time
import tempfile
import shutil
from typing import List, Optional
from huggingface_hub import hf_hub_download

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Download the model from HuggingFace Hub"""
        # Path to llama.cpp executable
        self.executable_path = "./llama.cpp/main"
        
        # Download the model
        print("Downloading model...")
        self.model_path = hf_hub_download(
            repo_id="gmonsoon/gemma2-9b-cpt-sahabatai-v1-instruct-GGUF",
            filename="gemma2-9b-cpt-sahabatai-v1-instruct.Q4_K_M.gguf"
        )
        print(f"Model downloaded to {self.model_path}")

    def predict(
        self,
        prompt: str = Input(description="The prompt to generate text from"),
        system_prompt: str = Input(
            description="System prompt to guide the model's behavior",
            
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=1024,
            ge=1,
            le=8192
        ),
        temperature: float = Input(
            description="Sampling temperature; higher values make output more random",
            default=0.7,
            ge=0.0,
            le=2.0
        ),
        top_p: float = Input(
            description="Nucleus sampling probability threshold",
            default=0.95,
            ge=0.0,
            le=1.0
        ),
        stop_sequences: List[str] = Input(
            description="Sequences that will stop generation when produced",
            default=[]
        )
    ) -> str:
        # Format the prompt based on the model's expected format
        formatted_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        # Create a temporary file for the prompt and output
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as prompt_file:
            prompt_file.write(formatted_prompt)
            prompt_file_path = prompt_file.name
        
        output_file_path = f"{prompt_file_path}.out"

        # Build command
        cmd = [
            self.executable_path,
            "-m", self.model_path,
            "--color", "false",
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "-c", str(max_tokens),
            "-n", str(max_tokens),
            "-f", prompt_file_path,
            "--no-prompt-caching",
            "-ngl", "1"  # Use 1 GPU layer
        ]
        
        # Add stop sequences if provided
        for seq in stop_sequences:
            cmd.extend(["--reverse-prompt", seq])
        
        # Run llama.cpp
        try:
            with open(output_file_path, "w") as output_file:
                process = subprocess.Popen(
                    cmd, 
                    stdout=output_file, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stderr = process.communicate()[1]
                if process.returncode != 0:
                    print(f"Error running llama.cpp: {stderr}")
                    return f"Error generating response: {stderr}"
            
            # Read the results
            with open(output_file_path, "r") as f:
                output = f.read()
            
            # Clean up the output - remove the original prompt
            response = output.replace(formatted_prompt, "").strip()
            
            # Clean up temporary files
            os.unlink(prompt_file_path)
            os.unlink(output_file_path)
            
            return response
            
        except Exception as e:
            print(f"Exception: {e}")
            # Clean up temporary files
            if os.path.exists(prompt_file_path):
                os.unlink(prompt_file_path)
            if os.path.exists(output_file_path):
                os.unlink(output_file_path)
            return f"Error: {str(e)}"