# Generative Pre-trained transformer Implementation

The implementation was done on Google colab, following the paper "Attention is All You Need" and OpenAI's GPT-2.

Refer  ``` training_code.ipynb``` for codes related to training the model and results,  ```model.py``` for the underlying transformer architecture and GPT class.




## Install Requirements:
For installing the requirements for this software, please run the following: 

 ```
 pip install -r requirements.txt
 ```

## Local Chatbot
To run a chatbot on the terminal on your local machine, copy the files from the ```LocalChatbot``` folder on your local machine,place them all in the same repository, execute ```python generator.py```. The model class is in ```gpt_model.py``` and the trained model weights are in ```trained_model.pth```.

## Dataset
The dataset used for training the model is the ```binary_operation_fine_shuffled_file.csv``` , a self-generated file consisting of binary computations for numbers 1 to 20( addition, subtraction, multiplication, division).
## Paper Reference

https://arxiv.org/abs/1706.03762