import cv2
import numpy as np
from matplotlib import pyplot as plt
import requests


def main():
    
    barcode = "7891000412855" 
    api_url = f"http://127.0.0.1:8000/food-facts/{barcode}"
    
    api_response = requests.get(api_url)
    
    if api_response.status_code != 200:
        print(f"Erro ao acessar API: {api_response.text}")
        return

    
    image_url = api_response.json().get("nutrition_image_url")
    print(f"URL encontrada: {image_url}")

    img_response = requests.get(image_url)
    
    if img_response.status_code == 200:
        
        img_array = np.frombuffer(img_response.content, np.uint8)        
        img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        plt.figure(figsize=(12, 6))
        
        
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title('Imagem Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        
        plt.imshow(gray_img, cmap='gray')
        plt.title('Escala de Cinza')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    else:
        print(f"Erro ao baixar a imagem. Status: {img_response.status_code}")

if __name__ == "__main__":
    main()