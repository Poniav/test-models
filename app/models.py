from transformers import MarianMTModel, MarianTokenizer
import torch
import sentencepiece

# Fonction de traduction qui charge dynamiquement le modèle nécessaire
def translate_text(text, src_lang, tgt_lang):
    # Définir le nom du modèle en fonction de la paire de langues
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    
    # Charger le modèle et le tokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Tokeniser le texte source
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Effectuer la traduction
    with torch.no_grad():
        translated = model.generate(**inputs)
    
    # Décoder la réponse
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    res = {
        "model_name": model_name,
        "source_text": text,
        "translated_text": translated_text,
        "source_lang": src_lang,
        "target_lang": tgt_lang
    }
    
    return res

# Exemple d'utilisation
text = "Je suis aller à la neige hier soir et je suis très content de ce que j'ai fait."
src_lang = "fr"  # Langue source (français)
tgt_lang = "en"  # Langue cible (anglais)

translated_text = translate_text(text, src_lang, tgt_lang)
print(f"Model: {translated_text['model_name']}")
print(f"Translated Text: {translated_text}")