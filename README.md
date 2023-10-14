# Generative Pre-trained transformer Implementation

The implementation was done on Google colab, following the paper "Attention is All You Need" and OpenAI's GPT-2.

Refer  ``` training_code.ipynb``` for codes related to training the model and results,  ```model.py``` for the underlying transformer architecture and GPT class.




## Install Requirements:
For installing the requirements for this software, please run the following: 

 ```
 pip install -r requirements.txt
 ```
  
## Dataset
The dataset used for training the model is the ```binary_operation_fine_shuffled_file.csv``` , a self-generated file consisting of binary computations for numbers 1 to 20( addition, subtraction, multiplication, division).
## Paper Reference

https://arxiv.org/abs/1706.03762