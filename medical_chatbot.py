import os
import warnings
import pyfiglet
import time

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

def print_in_color(text, color="blue"):
    # Print text in the specified color
    color_code = {
        "red": "\033[91m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }

    if color in color_code:
        print(color_code[color] + text + color_code["reset"])
    else:
        print(text)
        
def print_with_typing(text, typing_speed=0.03, color=None):
    # Print text with a typing effect and optional color
    color_code = {
        "red": "\033[91m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }

    if color in color_code:
        print(color_code[color], end='')

    for char in text:
        print(char, end='', flush=True)
        time.sleep(typing_speed)
    print(color_code["reset"])

def initialize_chatbot():
    # Initialize the chatbot by loading the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b", cache_dir=set_cache_directory())
    model = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-7b", cache_dir=set_cache_directory())
    return model, tokenizer

def chat_with_bot(model, tokenizer):
    # Main function to run the chatbot
    figlet_obj = pyfiglet.Figlet()
    ascii_art = figlet_obj.renderText("QHealthMate")
    print(ascii_art)

    print_in_color("Disclaimer: QHealthMate is intended for informational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for personalized medical guidance and recommendations.", color="red")
    print()
    print_with_typing("Hello! I'm your QHealthMate. You can start chatting with me. Type 'exit', 'quit', or 'bye' to end the conversation.", color="green")
    
    context = None
    while True:
        user_input = input("\033[94mYou: \033[0m")

        if user_input.lower() in ["quit", "exit", "bye"]:
            print_with_typing("Goodbye and have a nice day! :)", color="red")
            break

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            prompt = f"Context: {context}\n\nQuestion: {user_input}\n\nChatbot: "
            answer = generator(prompt, max_length=300, temperature=0.6, do_sample=True)

        cleaned_answer = clean_generated_text(answer[0]['generated_text'])
        print_with_typing(cleaned_answer, color="green")

def main():
    model, tokenizer = initialize_chatbot()
    chat_with_bot(model, tokenizer)

if __name__ == "__main__":
    main()
