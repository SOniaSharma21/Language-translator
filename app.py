from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS
import gradio as gr

device = "cpu"
language_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

language_model = AutoModelForCausalLM.from_pretrained(
    language_model_name,
    low_cpu_mem_usage=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(language_model_name)

# Language dictionary
languages = {
    "translate to english": ("English", "en"),
    "translate to chinese": ("Chinese", "zh-cn"),
    "translate to japanese": ("Japanese", "ja"),
    "translate to hindi": ("Hindi", "hi"),
    "translate to spanish": ("Spanish", "es"),
    "translate to french": ("French", "fr"),
    "translate to german": ("German", "de"),
    "translate to arabic": ("Arabic", "ar"),
    "translate to korean": ("Korean", "ko")
}


def process_input(input_text, action):

    # Translation mode
    if action in languages:
        lang_name, lang_code = languages[action]
        prompt = f"Translate the following sentence into natural {lang_name}. Only return the translated sentence:\n{input_text}"
        lang = lang_code

    # Chat mode
    else:
        prompt = input_text
        lang = "en"

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(text, return_tensors="pt").to(device)

    generate_ids = language_model.generate(
        model_inputs.input_ids,
        max_new_tokens=256
    )

    output_text = tokenizer.decode(
        generate_ids[0][model_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return output_text, lang


def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang, slow=False)
    filename = "output_audio.mp3"
    tts.save(filename)
    return filename


def handle_interaction(input_text, action):
    output_text, lang = process_input(input_text, action)
    audio_filename = text_to_speech(output_text, lang)
    return output_text, audio_filename


action_options = [
    "translate to english",
    "translate to chinese",
    "translate to japanese",
    "translate to hindi",
    "translate to spanish",
    "translate to french",
    "translate to german",
    "translate to arabic",
    "translate to korean",
    "chat"
]


iface = gr.Interface(
    fn=handle_interaction,
    inputs=[
        gr.Textbox(label="Enter text here..."),
        gr.Dropdown(label="Action", choices=action_options, value="chat")
    ],
    outputs=[
        gr.Textbox(label="Output Text"),
        gr.Audio(label="Output Audio", type="filepath")
    ],
    title="Trappoo Translator & Chat",
    description="Translate text into multiple languages or chat with the AI model.",
    theme="gradio/soft"
)

if __name__ == "__main__":
    iface.launch()