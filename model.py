from transformers import AutoTokenizer, AutoModelForMaskedLM

modelo = "dccuchile/bert-base-spanish-wwm-cased"
ruta_local = "./modelo_beto"

# Descarga y guarda modelo/tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelo)
model = AutoModelForMaskedLM.from_pretrained(modelo)

tokenizer.save_pretrained(ruta_local)
model.save_pretrained(ruta_local)