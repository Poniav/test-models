"""
DeepSeekR1: A deep learning-based tool for fast and accurate prediction of DNA and RNA binding proteins
https://huggingface.co/deepseek-ai/DeepSeek-R1

Not working yet on Local & Google Colab - To be implemented 
"""
from transformers import AutoModelForCausalLM

# Messages to generate
messages = [{"role": "user", "content": "What is the sequence of the DNA?"}]

# Load the DeepSeek-R1 model
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)