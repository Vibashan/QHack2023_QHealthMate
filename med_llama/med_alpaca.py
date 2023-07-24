import os
import warnings
import pyfiglet
import sys

import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def set_cache_directory():
    # Set the cache directory for the Hugging Face model
    current_path = os.getcwd()
    cache_directory = os.path.join(current_path, "huggingface_cache")
    return cache_directory

def clean_generated_text(generated_text):
    # Remove unwanted text tags and extract chatbot's response
    cleaned_text = generated_text.replace("< / FREETEXT >", "")
    cleaned_text = cleaned_text.replace("< / PARAGRAPH >", "")
    cleaned_text = cleaned_text.replace("▃", "")
    cleaned_text = cleaned_text.replace("< ABSTRACT >", "")
    cleaned_text = cleaned_text.replace("<\n>", " ")

    start_idx = generated_text.rfind("Chatbot:")
    if start_idx != -1:
        cleaned_answer = generated_text[start_idx + len("Chatbot:"):].strip()
        return cleaned_answer
    else:
        return generated_text

def generate_response(model, tokenizer, user_input):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        prompt = f"Question: {user_input}\n\nChatbot: "
        answer = generator(prompt, max_length=300, temperature=0.7, do_sample=True)

    cleaned_answer = clean_generated_text(answer[0]['generated_text'])
    return cleaned_answer

def print_qhealthmate():
    figlet_obj = pyfiglet.Figlet()
    ascii_art = figlet_obj.renderText("QHealthMate")
    print(ascii_art)

def print_disclaimer():
    print("Disclaimer: QHealthMate is intended for informational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for personalized medical guidance and recommendations.")
    print()

def main():

    model_name = "medalpaca/medalpaca-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=set_cache_directory())
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=set_cache_directory())
    
    print_qhealthmate()
    print_disclaimer()

    if len(sys.argv) != 2:
        print("Usage: python filename.py 'user_input'")
    else:
        user_input = sys.argv[1]
        response = generate_response(model, tokenizer, user_input)
        print(response)

if __name__ == "__main__":
    main()
