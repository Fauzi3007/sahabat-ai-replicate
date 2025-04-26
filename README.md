# Sistem QA Multilingual dengan Gemma2-9b-cpt-sahabatai

Ini adalah deployment model [gmonsoon/gemma2-9b-cpt-sahabatai-v1-instruct-GGUF](https://huggingface.co/gmonsoon/gemma2-9b-cpt-sahabatai-v1-instruct-GGUF) yang dioptimalkan untuk sistem tanya jawab berbasis dokumen dalam Bahasa Indonesia, Inggris, Jawa, dan Sunda untuk Politeknik Negeri Padang.

## Detail Model

- **Model Dasar**: Gemma2-9b
- **Fine-tuning**: Fine-tuned oleh GoToCompany sebagai "sahabatai-v1" dan versi gguf oleh gmonsoon
- **Kuantisasi**: Q4_K_M (kuantisasi 4-bit dengan kualitas medium)
- **Format**: GGUF (kompatibel dengan llama.cpp)
- **Bahasa yang Didukung**: Bahasa Indonesia, Bahasa Inggris, Bahasa Jawa, Bahasa Sunda

## Penggunaan

Model ini menerima parameter berikut:

- `question`: Pertanyaan yang diajukan pengguna
- `context`: Konteks dokumen untuk menjawab pertanyaan
- `max_tokens`: Jumlah maksimum token yang dihasilkan (default: 1024)
- `temperature`: Mengontrol keacakan; nilai lebih tinggi membuat output lebih acak (default: 0.7)
- `top_p`: Mengontrol keberagaman melalui nucleus sampling (default: 0.95)
- `raw_prompt`: (Opsional) Prompt mentah lengkap jika Anda ingin mengganti format default

## Contoh Penggunaan

### Bahasa Indonesia

```python
import replicate

output = replicate.run(
    "your-username/gemma2-qa-pnp",
    input={
        "question": "Apa syarat untuk mendaftar program S1?",
        "context": "Program S1 di Politeknik Negeri Padang memiliki beberapa persyaratan pendaftaran: Lulusan SMA/SMK/MA dengan nilai rata-rata minimal 7.5, Lulus ujian masuk tertulis, Membayar biaya pendaftaran sebesar Rp 250.000.",
        "max_tokens": 500,
        "temperature": 0.7
    }
)
print(output)
```

### Bahasa Inggris

```python
import replicate

output = replicate.run(
    "your-username/gemma2-qa-pnp",
    input={
        "question": "What are the requirements to register for the undergraduate program?",
        "context": "The undergraduate program at Politeknik Negeri Padang has several registration requirements: High school graduates with a minimum average score of 7.5, Pass the written entrance exam, Pay a registration fee of Rp 250,000.",
        "max_tokens": 500,
        "temperature": 0.7
    }
)
print(output)
```

### Bahasa Jawa

```python
import replicate

output = replicate.run(
    "your-username/gemma2-qa-pnp",
    input={
        "question": "Apa syarat kanggo ndaftar program S1?",
        "context": "Program S1 ing Politeknik Negeri Padang duwe sawetara syarat pendaftaran: Lulusan SMA/SMK/MA kanthi biji rata-rata minimal 7.5, Lulus ujian mlebu tertulis, Mbayar biaya pendaftaran Rp 250.000.",
        "max_tokens": 500,
        "temperature": 0.7
    }
)
print(output)
```

### Bahasa Sunda

```python
import replicate

output = replicate.run(
    "your-username/gemma2-qa-pnp",
    input={
        "question": "Naon syarat pikeun daftar program S1?",
        "context": "Program S1 di Politeknik Negeri Padang ngabogaan sababaraha sarat pendaptaran: Lulusan SMA/SMK/MA kalawan nilai rata-rata minimal 7.5, Lulus ujian asup tinulis, Mayar biaya pendaptaran Rp 250.000.",
        "max_tokens": 500,
        "temperature": 0.7
    }
)
print(output)
```

## Integrasi dengan Chatbot

Jika Anda menggunakan langchain untuk aplikasi chatbot, Anda dapat menggunakan Replicate API sebagai model backend:

```python
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate

# Definisikan template prompt
prompt = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Kamu adalah asisten dari Politeknik Negeri Padang.
Tugasmu adalah menjawab pertanyaan berdasarkan konteks dokumen yang diberikan oleh pengguna.
Jika pengguna bertanya di luar topik dokumen, jangan tanggapi.
Jika konteks yang diberikan tidak cukup untuk menjawab pertanyaan, katakan bahwa kamu tidak memiliki jawabannya.
Jawablah dalam bahasa yang sama dengan pertanyaan pengguna.

<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Jawablah pertanyaan pengguna berdasarkan konteks berikut:
Konteks: {context}
Pertanyaan: {question}

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Buat template prompt
prompt_template = PromptTemplate(
    template=prompt,
    input_variables=["context", "question"]
)

# Gunakan API Replicate
llm = Replicate(
    model="your-username/gemma2-qa-pnp",
    model_kwargs={"temperature": 0.7, "max_tokens": 500}
)

# Buat pipeline untuk inferensi
def get_response(context, question):
    formatted_prompt = prompt_template.format(context=context, question=question)
    # Kirim prompt yang sudah diformat ke API
    response = llm(
        input={
            "raw_prompt": formatted_prompt
        }
    )
    return response
```

## Catatan

Model ini didesain khusus untuk menjawab pertanyaan dalam Bahasa Indonesia, Inggris, Jawa, dan Sunda berdasarkan konteks dokumen yang diberikan. Model ini telah dikuantisasi untuk mengoptimalkan kinerja dan penggunaan sumber daya.
