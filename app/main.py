from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer


model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M', src_lang="en", tgt_lang="fr")


app = FastAPI()

class Records(BaseModel):
    id: str
    text: str

class Item(BaseModel):
    records: List[Records]
    fromLang: str = "en"
    toLang: str = "ja"

@app.post("/translation")
def translation(item: Item):
    tokenizer.src_lang = item.fromLang
    model_inputs = tokenizer(item.records[0].text, return_tensors="pt")
    generated_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id(item.toLang))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return {"result":{"id":item.records[0].id,"result": result[0]}}
