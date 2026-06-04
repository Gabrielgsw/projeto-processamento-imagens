import cv2
import numpy as np
from matplotlib import pyplot as plt
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares para ordenação
# ─────────────────────────────────────────────────────────────────────────────
def ordenar_pontos(pts):
    """ Ordena os 4 pontos: Topo-Esquerda, Topo-Direita, Base-Direita, Base-Esquerda """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares para correção de perspectiva
# ─────────────────────────────────────────────────────────────────────────────
def aplicar_perspectiva(imagem, pts):
    """Planificação caso a imagem não esteja reta """
    rect = ordenar_pontos(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(imagem, M, (maxWidth, maxHeight))
    return warped



# ─────────────────────────────────────────────────────────────────────────────
# Função principal que contém todo a lógica
# ─────────────────────────────────────────────────────────────────────────────
def main():
    barcode = "7891000379691" 
    api_url = f"http://127.0.0.1:8000/food-facts/{barcode}"
    
    try:
        api_response = requests.get(api_url)
        if api_response.status_code != 200:
            print(f"Erro: {api_response.status_code}.")
            return

        image_url = api_response.json().get("nutrition_image_url")
        if not image_url:
            print("Imagem não disponível")
            return

        img_response = requests.get(image_url)
        if img_response.status_code != 200:
            print(f"Erro: {img_response.status_code}")
            return
            
        # ─────────────────────────────────────────────────────────────────────────────
        # Pré-processamento
        # ─────────────────────────────────────────────────────────────────────────────
        img_array = np.frombuffer(img_response.content, np.uint8)
        img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            print("Erro: imagem inválida")
            return
            
        h, w = img_cv2.shape[:2]
        
        # Conversão BGR -> RGB
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        
        # Tons de cinza
        gray_img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        
        # Filtro blateral
        bilateral = cv2.bilateralFilter(gray_img, 9, 75, 75)
        
        # Equalização adaptativa(CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imagem_clahe = clahe.apply(bilateral)

        # ─────────────────────────────────────────────────────────────────────────────
        # Seleção dos objetos
        # ─────────────────────────────────────────────────────────────────────────────
        
        # Canny com filtro gaussiano
        bordas_ext = cv2.Canny(cv2.GaussianBlur(imagem_clahe, (5, 5), 0), 40, 120)
        
        
        # Fechamento
        kernel_ext = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bordas_fechadas = cv2.morphologyEx(bordas_ext, cv2.MORPH_CLOSE, kernel_ext)
        
        # Busca de contornos
        contornos_ext, _ = cv2.findContours(bordas_fechadas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Implementação da aproximação de Douglas-Peucker
        img_poligonos_ext = img_rgb.copy()
        tabela_bgr = None
        tabela_encontrada = False
        
        if contornos_ext:
            contornos_ext = sorted(contornos_ext, key=cv2.contourArea, reverse=True)
            
            for c in contornos_ext:
                peri = cv2.arcLength(c, True)
                aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
                area = cv2.contourArea(c)
                
                if len(aprox) == 4 and (w * h * 0.05) < area < (w * h * 0.95):
                    cv2.drawContours(img_poligonos_ext, [aprox], -1, (0, 255, 0), 4)
                    pontos = aprox.reshape(4, 2)
                    tabela_bgr = aplicar_perspectiva(img_cv2, pontos)
                    tabela_encontrada = True
                    break
            
            if not tabela_encontrada:
                # Busca um recorte em um limiar de 5% e 95% da imagem
                caixas_ext = [cv2.boundingRect(c) for c in contornos_ext if (w * h * 0.05) < cv2.contourArea(c) < (w * h * 0.95)]
                if caixas_ext:
                    for b in caixas_ext:
                        cv2.rectangle(img_poligonos_ext, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255, 0, 0), 2)
                    x_min = max(0, min(b[0] for b in caixas_ext) - 5)
                    x_max = min(w, max(b[0] + b[2] for b in caixas_ext) + 5)
                    y_min = max(0, min(b[1] for b in caixas_ext) - 5)
                    y_max = min(h, max(b[1] + b[3] for b in caixas_ext) + 5)
                    tabela_bgr = img_cv2[y_min:y_max, x_min:x_max]

        if tabela_bgr is None or tabela_bgr.size == 0:
            print("Falha no recorte")
            return
            
        h_tab, w_tab = tabela_bgr.shape[:2]

        # ─────────────────────────────────────────────────────────────────────────────
        # Segmentação
        # ─────────────────────────────────────────────────────────────────────────────
        gray_tab = cv2.cvtColor(tabela_bgr, cv2.COLOR_BGR2GRAY)
        
        # Canny "interno"
        bordas_int = cv2.Canny(cv2.GaussianBlur(gray_tab, (3, 3), 0), 40, 120)
        
        
        # Fechamento "interno"
        kernel_int = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 5)) 
        morph_int = cv2.morphologyEx(bordas_int, cv2.MORPH_CLOSE, kernel_int)
        
        #  Busca de contornos
        contornos_int, _ = cv2.findContours(morph_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        caixas_int = [cv2.boundingRect(c) for c in contornos_int]
        
        # Ordenação
        caixas_validas = [b for b in caixas_int if b[2] > w_tab * 0.25 and b[3] > 8]
        caixas_validas.sort(key=lambda b: b[1]) 
        
        y_div = 0
        img_divisoria_debug = cv2.cvtColor(tabela_bgr, cv2.COLOR_BGR2RGB)
        
        # Recorte
        if caixas_validas:
            bx, by, bw, bh = caixas_validas[0]
            cv2.rectangle(img_divisoria_debug, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
            y_div = by + bh + 3
            
            for (bx_d, by_d, bw_d, bh_d) in caixas_validas[1:]:
                cv2.rectangle(img_divisoria_debug, (bx_d, by_d), (bx_d+bw_d, by_d+bh_d), (255, 0, 0), 1)
        
        if y_div <= 0 or y_div >= h_tab:
            y_div = int(h_tab * 0.15)
        y_div = min(y_div, h_tab - 1)
        
        cv2.line(img_divisoria_debug, (0, y_div), (w_tab - 1, y_div), (0, 255, 0), 2)

        titulo_bgr = tabela_bgr[0:y_div, :]
        dados_bgr = tabela_bgr[y_div:h_tab, :]

        # ─────────────────────────────────────────────────────────────────────────────
        # Plotagem
        # ─────────────────────────────────────────────────────────────────────────────
        imagens_plot = [
            ("1. Original", img_rgb),
            ("2. Grayscale", gray_img),
            ("3. CLAHE", imagem_clahe),
            ("4. Canny Original", bordas_ext),
            ("5. Douglas-Peucker", img_poligonos_ext),
            ("6. Tabela Planificada ", cv2.cvtColor(tabela_bgr, cv2.COLOR_BGR2RGB)),
            ("7. Fechamento Morfológico Interno", morph_int),
            ("8. Divisória Detectada", img_divisoria_debug),
            ("9. Título", cv2.cvtColor(titulo_bgr, cv2.COLOR_BGR2RGB) if titulo_bgr.size > 0 else np.zeros_like(tabela_bgr)),
            ("10. Dados Nutricionais", cv2.cvtColor(dados_bgr, cv2.COLOR_BGR2RGB) if dados_bgr.size > 0 else np.zeros_like(tabela_bgr))
        ]
        
        plt.figure(figsize=(15, 14))
        for i, (titulo, img) in enumerate(imagens_plot):
            plt.subplot(4, 3, i + 1)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.title(titulo, fontsize=11, fontweight="bold")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()