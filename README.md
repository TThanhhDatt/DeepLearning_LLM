# DeepLearning_LSTM
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
