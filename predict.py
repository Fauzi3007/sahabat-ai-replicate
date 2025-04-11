from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_model():
    model_name = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

generator = load_model()

def predict(prompt: str):
    results = generator(prompt, max_new_tokens=200)
    return results[0]["generated_text"]
