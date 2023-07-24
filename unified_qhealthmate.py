import os
import sys
import time
import pyfiglet
from PIL import Image
import numpy as np
import argparse

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ecg_classifier import predict_ecg_class
from skin_cancer_classifier import predict_skin_cancer_class

import warnings
warnings.filterwarnings('ignore')


def set_cache_directory():
    # Set the cache directory for the Hugging Face model
    current_path = os.getcwd()
    cache_directory = os.path.join(current_path, "med_llama/huggingface_cache")
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
    print_in_color("Disclaimer: QHealthMate is intended for informational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for personalized medical guidance and recommendations.", color='red')
    print()

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

def main():
    model_name = "medalpaca/medalpaca-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=set_cache_directory())
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=set_cache_directory())

    parser = argparse.ArgumentParser(description='Skin Lesion Classification and Medical QA')
    parser.add_argument('--input_image', type=str, help='Path to the input image for classification')
    parser.add_argument('--input_audio', type=str, help='Path to the input audio for ECG classification')
    parser.add_argument('--image_model_path', type=str, default='./skin_cancer/models/best_model_final.pth', help='Path to the best image model checkpoint file')
    parser.add_argument('--audio_model_path', type=str, default='./ecg/models/best_model_final.pth', help='Path to the best audio model checkpoint file')
    parser.add_argument('--image_llama_vector_path', type=str, default='../data/skin_data/llama_embed_skin.npy', help='Path to the llama vector file for image model')
    parser.add_argument('--audio_llama_vector_path', type=str, default='../data/ecg_data/llama_embed_ecg.npy', help='Path to the llama vector file for audio model')
    args = parser.parse_args()

    print_qhealthmate()
    print_disclaimer()

    if args.input_image:
        # Validate input image path
        if not os.path.isfile(args.input_image):
            print_in_color("Error: Invalid input image path.", color="red")
            exit(1)

        # Validate image model checkpoint path
        if not os.path.isfile(args.image_model_path):
            print_in_color("Error: Invalid image model checkpoint path.", color="red")
            exit(1)

        # Validate image llama vector path
        if not os.path.isfile(args.image_llama_vector_path):
            print_in_color("Error: Invalid image llama vector path.", color="red")
            exit(1)

        # Skin cancer classification prediction
        print_in_color("Loading pretrained weights for the skin lesion classification model...", color="green")
        predicted_class_name = predict_skin_cancer_class(args.input_image, args.image_model_path, args.image_llama_vector_path)
        print_in_color(f"Skin Lesion Classification Model Predicted:", color="green")
        print_with_typing(predicted_class_name, color="blue")
        
        print()

        # Medical QA chatbot response
        print_with_typing("Asking the QHealthMate for advice...", color="red")
        response = generate_response(model, tokenizer, f"What is {predicted_class_name} and what should I do?")
        print_with_typing(f"QHealthMate: {response}", color="green")

    elif args.input_audio:
        # Validate input audio path
        if not os.path.isfile(args.input_audio):
            print_in_color("Error: Invalid input audio path.", color="red")
            exit(1)

        # Validate audio model checkpoint path
        if not os.path.isfile(args.audio_model_path):
            print_in_color("Error: Invalid audio model checkpoint path.", color="red")
            exit(1)

        # Validate audio llama vector path
        if not os.path.isfile(args.audio_llama_vector_path):
            print_in_color("Error: Invalid audio llama vector path.", color="red")
            exit(1)

        # ECG classification prediction
        print_in_color("Loading pretrained weights for the ECG classification model...", color="green")
        predicted_class_name = predict_ecg_class(args.input_audio, args.audio_model_path, args.audio_llama_vector_path)
        print_in_color("ECG Classification Model Predicted:", color="green")
        print_with_typing(predicted_class_name, color="blue")
        
        print()
        
        # Medical QA chatbot response
        print_with_typing("Asking the QHealthMate for advice...", color="red")
        response = generate_response(model, tokenizer, f"What is {predicted_class_name} and what should I do?")
        print_with_typing(f"QHealthMate: {response}", color="green")

    else:
        print_in_color("Error: No input provided. Please provide either --input_image or --input_audio.", color="red")

if __name__ == "__main__":
    main()

