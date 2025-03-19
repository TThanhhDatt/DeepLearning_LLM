# DeepLearning And LLM
Build model deep learning for Entity recognition problem

Describe this project:
1. Build data, limit it to a data domain and a certain narrow problem so that the data is concentrated (avoid the phenomenon of sparse data), that is:
  - Restrictions on photo types: for example, only photos of fruit
  - Limitations on question types: for example, only asking about quantity and identification
  Use LLMs to generate data:
  - Collect photos
  - Use LLMs to create questions and answers
2. Build a model to solve this problem with the following requirements:
  - Use CNN to learn image features
  - Use LSTM to generate output 
  - Note: connect CNN and LSTM into a unified model 
  - Test and compare approaches: use pretrain models and train models from scratch
  - Use LSTM to generate sequences with attention and without attention

# Description
Request 1: I use GPT-2 to create caption for the image.
 - GPT-2 (Generative Pre-trained Transformer 2) is a large-scale language model developed by OpenAI and released in 2019. It is the second iteration of the GPT series and was a significant advancement in natural language processing (NLP).

Request 2: 
 - CNN: I use pretrained model to train: VGG16 and custom model, however the accuracy is very low, so i will improve model in the future
 - LSTM: I use LSTM to generate sequences with attention and without attention and i compare it through graph

# Result 
- About LLM, applying LLM (GPT-2) to the problem, when I collect image data, specifically fruits, the result is that the model generates a json file for me, the content includes image description, questions and answers, making model training easier.
- About Deep Learning, CNN plays the role of image recognition, helping to identify the type of fruit. LSTM plays the role of generating questions and answers.

# Inference
The accuracy is very low because epoch is low so my training is not good. I will improve its.
The output is incorrect because of the accuracy
LLM help to generate Q&A correctly, and it saves me less time.  