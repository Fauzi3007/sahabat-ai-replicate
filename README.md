# Gemma2-9b-cpt-sahabatai Quantized Model

This is a Replicate deployment of the quantized [gmonsoon/gemma2-9b-cpt-sahabatai-v1-instruct-GGUF](https://huggingface.co/gmonsoon/gemma2-9b-cpt-sahabatai-v1-instruct-GGUF) model.

## Model Details

- **Base Model**: Gemma2-9b
- **Fine-tuning**: Custom fine-tuned by gmonsoon as "sahabatai-v1"
- **Quantization**: Q4_K_M (4-bit quantization with medium quality)
- **Format**: GGUF (compatible with llama.cpp)

## Usage

This model accepts the following parameters:

- `prompt`: The user input to generate a response for
- `system_prompt`: Instructions to guide the model's behavior (default: "You are a helpful AI assistant.")
- `max_tokens`: Maximum number of tokens to generate (default: 1024)
- `temperature`: Controls randomness; higher values make output more random (default: 0.7)
- `top_p`: Controls diversity via nucleus sampling (default: 0.95)
- `stop_sequences`: List of strings that will stop generation when produced

Example:

```python
import replicate

output = replicate.run(
    "your-username/gemma2-9b-cpt-sahabatai",
    input={
        "prompt": "Explain quantum computing in simple terms",
        "system_prompt": "You are a helpful AI assistant that explains complex topics in an easy-to-understand way.",
        "max_tokens": 500,
        "temperature": 0.7
    }
)
print(output)
```

## Notes

This is a quantized version of the original model, offering faster inference with slightly reduced precision. It's ideal for applications requiring good performance with reasonable resource usage.
