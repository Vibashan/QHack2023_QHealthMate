# QHealthMate [QHack 2023]

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 


#### Contributions
- HealthMate(Chatbot): Given user input queries, generate output text addressing the queries powered by LLAMA model (Large Language Model).
- Multimodal Analyser: Other than text, if you want to give Image or Audio, the these standalone CNN model can be utilized. For instance if you want to check whether you have some skin lesion disease, you can simply capture a photo of your skin, and the CNN model classifies the skin lesion and the output is provided to the llama model which will assist you further.
- ONNX Export: Finally, the whole system is exported to ONNX to run on mobile devices.

## Environment
```angular2
UBUNTU="18.04"
CUDA="11.0"
CUDNN="8"
```

### Installation

```angular2

conda env create -f environment.yaml
conda activate qhack

```
### Data structure
Please download the data please use this [link](https://drive.google.com/drive/folders/1RD8_dWvPvzOriGQug14Eu3TfKg5gffIV?usp=sharing) 
```
../data/
  ecg_data/
    mit_train.csv
    mit_test.csv
  skin_data/
    train/
        ISIC_0034075.jpg
        ISIC_0034075.jpg
    train.csv
    test/
    test.csv
```

### Command to run Chatbot
```angular2
python medical_chatbot.py
```

### Command to run Chatbot + Multimodal Annalyzer
```angular2
# For skin_classification + chatbot
python unified_qhealthmate.py --input_image /path/to/your/image.jpg --image_model_path ./skin_cancer/models/best_model_final.pth --image_llama_vector_path ../data/skin_data/llama_embed_skin.npy

# For ecg_classification + chatbot
python unified_qhealthmate.py --input_audio /path/to/your/audio.wav --audio_model_path ./ecg/models/best_model_final.pth --audio_llama_vector_path ../data/ecg_data/llama_embed_ecg.npy

```

### Command to export to ONNX
```angular2
python skin_cancer/onnx_export.py --input_image path_to_input_image.jpg
```

### Acknowledgement:
1. https://github.com/kbressem/medAlpaca
2. https://github.com/csuustc/ECG-Heartbeat-Classification
3. https://github.com/hoang-ho/Skin_Lesions_Classification_DCNNs

