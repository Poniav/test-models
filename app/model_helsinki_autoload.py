"""
Helsinki-NLP/opus-mt-fr-en : Modèle pour traduire du français vers l'anglais 
We will use the Helsinki-NLP/opus-mt-fr-en model to translate a French text to English.

Currently working on the translation function that dynamically loads the necessary model.
"""

from transformers import MarianMTModel, MarianTokenizer
import torch

# Function to translate text from source language to target language
def translate_text(text, src_lang, tgt_lang):
    # Define the model name based on the source and target languages
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    
    # Load the model and tokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate translation
    with torch.no_grad():
        translated = model.generate(**inputs)
    
    # Decode the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    res = {
        "model_name": model_name,
        "source_text": text,
        "translated_text": translated_text,
        "source_lang": src_lang,
        "target_lang": tgt_lang
    }
    
    return res

# Usage example
text = "Je suis aller à la neige hier soir et je suis très content de ce que j'ai fait."
src_lang = "fr"  # Langue source (français)
tgt_lang = "en"  # Langue cible (anglais)

translated_text = translate_text(text, src_lang, tgt_lang)
print(f"Model: {translated_text['model_name']}")
print(f"Translated Text: {translated_text}")