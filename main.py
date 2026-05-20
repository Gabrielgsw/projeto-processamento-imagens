from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

@app.get("/food-facts/{barcode}")
def read_open_food_facts(barcode: str):
    
        
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"

    headers = {
        "User-Agent": "Projeto_PDI_UFRPE - Python/FastAPI - Version 1.0"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()        
       
        if data.get("status") == 0:
            raise HTTPException(status_code=404, detail="Produto não encontrado na base de dados.")
        
        
        product_data = data.get("product", {})        
        
        nutrition_image_url = product_data.get("image_nutrition_url")
        
        
        if not nutrition_image_url:
            raise HTTPException(status_code=404, detail="Produto encontrado, mas não possui foto da tabela nutricional cadastrada.")
        
        
        return {
            "barcode": barcode,
            "nutrition_image_url": nutrition_image_url
        }
    else:
        raise HTTPException(status_code=response.status_code, detail="Erro de comunicação com a API.")