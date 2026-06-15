import cv2
import numpy as np
from matplotlib import pyplot as plt
import requests
import pytesseract
import re
import os

# Caminho da instalação do pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Arquivo com um código de barras por linha
ARQUIVO_BARCODES = "barcodes.txt"

# Pasta onde as imagens originais dos rótulos serão salvas
PASTA_IMGS = "imgs"


# Ordenação de coordenadas espaciais
def ordenar_pontos(pts):
    """Ordena os vértices do polígono no formato: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# Transformação geométrica e correção de perspectiva
def aplicar_perspectiva(imagem, pts):
    """Calcula a matriz de homografia e planifica a região delimitada pelos pontos."""
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

# Extração do texto do rótulo através do OCR pytesseract
def extrair_texto_ocr(imagem_dados):
    """
    Processa a matriz da imagem extraída para otimização de leitura OCR.
    Retorna o texto extraído pelo Tesseract.
    """
    print("\n[Iniciando Leitura OCR...]")

    escala = 2.5
    largura = int(imagem_dados.shape[1] * escala)
    altura = int(imagem_dados.shape[0] * escala)
    img_ampliada = cv2.resize(imagem_dados, (largura, altura), interpolation=cv2.INTER_CUBIC)

    gray_ocr = cv2.cvtColor(img_ampliada, cv2.COLOR_BGR2GRAY)
    blur_ocr = cv2.GaussianBlur(gray_ocr, (3, 3), 0)
    _, bin_ocr = cv2.threshold(blur_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config_tesseract = '--oem 3 --psm 6'

    try:
        texto_extraido = pytesseract.image_to_string(bin_ocr, lang='por', config=config_tesseract)
        print("────────────────── TEXTO EXTRAÍDO ──────────────────")
        print(texto_extraido)
        print("──────────────────────────────────────────────────────\n")
        return texto_extraido
    except Exception as e:
        print(f"Falha no OCR: {e}")
        return None

# Extração dos nutrientes desejados do texto processado pelo ocr
def extrair_nutrientes_do_texto(texto):
    """
    Percorre o texto extraído pelo OCR linha a linha e busca o primeiro
    número encontrado nas linhas que contêm "valor energético" e "carboidratos"

    Usa múltiplas variações de cada palavra-chave para absorver possíveis
    erros no OCR

    Retorna um dicionário com os valores encontrados.
    """
    if not texto:
        return {}

    padroes = {
        "Valor energético": [r"va[lt]or", r"energ", r"kcal", r"keal"],
        "Carboidratos":     [r"carboidr", r"carho", r"arbo"],
    }

    resultado = {}
    linhas = texto.split('\n')

    for i, linha in enumerate(linhas):
        linha_lower = linha.lower()
        for nome, lista_padroes in padroes.items():
            if nome in resultado:
                continue
            if not any(re.search(p, linha_lower) for p in lista_padroes):
                continue
            # Busca o primeiro número na linha atual, caso não ache, tenta a próxima
            nums = re.findall(r'\d[\d,\.]*', linha)
            if not nums and i + 1 < len(linhas):
                nums = re.findall(r'\d[\d,\.]*', linhas[i + 1])
            if nums:
                resultado[nome] = nums[0]

    return resultado

# Busca os valores desejados na API para verificar assertividade
def buscar_valores_api(barcode):
    """
    Consulta diretamente a API Open Food Facts para obter os valores
    nutricionais cadastrados (por 100g/100ml).

    Campos retornados:
      energy-kcal_100g   → Valor energético em kcal
      carbohydrates_100g → Carboidratos em g
    """
    url     = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    headers = {"User-Agent": "Projeto_Processamento_Imagens -  Version 1.0"}

    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"[API] Erro ao buscar dados nutricionais: {resp.status_code}")
            return None, None

        nutriments = resp.json().get("product", {}).get("nutriments", {})
        energia    = nutriments.get("energy-kcal_100g")
        carbo      = nutriments.get("carbohydrates_100g")
        return energia, carbo

    except Exception as e:
        print(f"Erro ao buscar dados da API: {e}")
        return None, None

# Comparação dos valores obtidos pelo ocr e dos valores obtidos via API
def comparar_com_api(valores_ocr, energia_api, carbo_api, tolerancia=0.10):
    """
    Compara os valores extraídos pelo OCR com os da API Open Food Facts.

    A tolerância padrão de 10% existe porque o OCR em fotos raramente é
    perfeito, pois as variações de iluminação e perspectiva podem causar pequenos
    erros de leitura nos dígitos.

    Exibe uma tabela com: valor OCR | valor API | status da comparação.
    """
    print("\n" + "=" * 54)
    print("    COMPARAÇÃO  OCR x API OPEN FOOD FACTS")
    print("=" * 54)
    print(f"{'Nutriente':<25} {'OCR':>9} {'API':>9}  {'Status'}")
    print("-" * 54)

    pares = [
        ("Valor energético", valores_ocr.get("Valor energético"), energia_api, "kcal"),
        ("Carboidratos",     valores_ocr.get("Carboidratos"),     carbo_api,   "g"),
    ]

    for nome, val_ocr, val_api, unidade in pares:
        if val_ocr is None:
            status = "⚠  OCR não leu"
        elif val_api is None:
            status = "⚠  API sem dado"
        else:
            try:
                v_ocr = float(str(val_ocr).replace(',', '.'))
                v_api = float(val_api)
                diff  = abs(v_ocr - v_api) / v_api if v_api != 0 else 1.0
                status = "✓ COMPATÍVEL" if diff <= tolerancia else f"✗ DIVERGENTE ({diff*100:.0f}%)"
            except ValueError:
                status = "⚠  Valor inválido"

        ocr_str = f"{val_ocr} {unidade}" if val_ocr else "—"
        api_str = f"{val_api} {unidade}" if val_api else "—"
        print(f"{nome:<25} {ocr_str:>9} {api_str:>9}  {status}")

    print("=" * 54)



# Leitura dos códigos de barras a partir do arquivo .txt
def carregar_barcodes(caminho: str) -> set[str]:
   
    if not os.path.exists(caminho):
        print(f"[ERRO] Arquivo '{caminho}' não encontrado.")
        return set()

    barcodes = set()
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            codigo = linha.strip()
            if codigo:
                barcodes.add(codigo)

    print(f"{len(barcodes)} código(s) de barras únicos carregados de '{caminho}'")
    return barcodes



# Processamento de um único rótulo 
def processar_rotulo(barcode: str):
    
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


        # 1. Inicio do Pré-processamento
        img_array = np.frombuffer(img_response.content, np.uint8)
        img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            print("Imagem inválida! ")
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


        # Binarização adaptativa
        imagem_binarizada = cv2.adaptiveThreshold(
            imagem_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )


        # 2. Seleção dos objetos e Extração Geométrica

        # Canny
        bordas_ext = cv2.Canny(imagem_binarizada, 40, 120)

        # Fechamento
        kernel_ext = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bordas_fechadas = cv2.morphologyEx(bordas_ext, cv2.MORPH_CLOSE, kernel_ext)

        # Busca por contornos
        contornos_ext, _ = cv2.findContours(bordas_fechadas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Implementação da aproximação de Douglas-Peucker
        img_poligonos_ext = img_rgb.copy()
        tabela_bgr = None
        tabela_encontrada = False

        if contornos_ext:
            contornos_ext = sorted(contornos_ext, key=cv2.contourArea, reverse=True)

            for c in contornos_ext:
                area = cv2.contourArea(c)

                if (w * h * 0.05) < area:
                    peri = cv2.arcLength(c, True)
                    aprox = None

                    for eps in np.linspace(0.01, 0.08, 10):
                        temp_aprox = cv2.approxPolyDP(c, eps * peri, True)
                        if len(temp_aprox) == 4:
                            aprox = temp_aprox
                            break

                    if aprox is None:
                        rect = cv2.minAreaRect(c)
                        box = cv2.boxPoints(rect)
                        aprox = np.intp(box).reshape(4, 1, 2)

                    cv2.drawContours(img_poligonos_ext, [aprox], -1, (0, 255, 0), 4)
                    pontos = aprox.reshape(4, 2)
                    tabela_bgr = aplicar_perspectiva(img_cv2, pontos)
                    tabela_encontrada = True
                    break

            if not tabela_encontrada:
                caixas_ext = [cv2.boundingRect(c) for c in contornos_ext if (w * h * 0.05) < cv2.contourArea(c) < (w * h * 0.99)]
                if caixas_ext:
                    for b in caixas_ext:
                        cv2.rectangle(img_poligonos_ext, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255, 0, 0), 2)
                    x_min = max(0, min(b[0] for b in caixas_ext) - 5)
                    x_max = min(w, max(b[0] + b[2] for b in caixas_ext) + 5)
                    y_min = max(0, min(b[1] for b in caixas_ext) - 5)
                    y_max = min(h, max(b[1] + b[3] for b in caixas_ext) + 5)
                    tabela_bgr = img_cv2[y_min:y_max, x_min:x_max]

        if tabela_bgr is None or tabela_bgr.size == 0:
            print("Usando a imagem inteira")
            tabela_bgr = img_cv2.copy()
            cv2.rectangle(img_poligonos_ext, (0, 0), (w-1, h-1), (0, 0, 255), 6)

        h_tab, w_tab = tabela_bgr.shape[:2]


        # 3. Segmentação Interna (Separação Cabeçalho/Dados)
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
            # Sanity check: um título legítimo ocupa uma fração pequena da
            # altura da tabela. Se o fechamento morfológico fundiu várias
            # linhas em um único blob (comum em rótulos com muitos
            # nutrientes, onde os espaçamentos verticais são menores que o
            # kernel_int), bh fica próximo de h_tab e esse box é descartado
            # — y_div cai no fallback de 15% abaixo.
            if bh < h_tab * 0.25:
                cv2.rectangle(img_divisoria_debug, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                y_div = by + bh + 3
                for (bx_d, by_d, bw_d, bh_d) in caixas_validas[1:]:
                    cv2.rectangle(img_divisoria_debug, (bx_d, by_d), (bx_d+bw_d, by_d+bh_d), (255, 0, 0), 1)

        if y_div <= 0 or y_div >= h_tab:
            y_div = int(h_tab * 0.15)
        y_div = min(y_div, h_tab - 1)

        cv2.line(img_divisoria_debug, (0, y_div), (w_tab - 1, y_div), (0, 255, 0), 2)

        titulo_bgr = tabela_bgr[0:y_div, :]
        dados_bgr  = tabela_bgr[y_div:h_tab, :]


        # 4. Inferência OCR + Extração de Nutrientes + Comparação com API
        if dados_bgr.size > 0:
            texto = extrair_texto_ocr(dados_bgr)

            # Extrai os valores de energia e carboidratos do texto OCR
            valores_ocr = extrair_nutrientes_do_texto(texto)

            # Busca os valores de referência direto na Open Food Facts
            energia_api, carbo_api = buscar_valores_api(barcode)

            # Compara OCR × API e exibe o resultado
            comparar_com_api(valores_ocr, energia_api, carbo_api)
        else:
            print("Matriz de dados vazia")


        # 5. Plotagem
        imagens_plot = [
            ("1. Original", img_rgb),
            ("2. Grayscale", gray_img),
            ("3. Binarização Adaptativa", imagem_binarizada),
            ("4. Canny Original", bordas_ext),
            ("5. Douglas Peucker", img_poligonos_ext),
            ("6. Tabela Planificada", cv2.cvtColor(tabela_bgr, cv2.COLOR_BGR2RGB)),
            ("7. Fechamento Morfológico Interno", morph_int),
            ("8. Divisória Detectada", img_divisoria_debug),
            ("9. Título", cv2.cvtColor(titulo_bgr, cv2.COLOR_BGR2RGB) if titulo_bgr.size > 0 else np.zeros_like(tabela_bgr)),
            ("10. Dados Nutricionais", cv2.cvtColor(dados_bgr, cv2.COLOR_BGR2RGB) if dados_bgr.size > 0 else np.zeros_like(tabela_bgr))
        ]

        plt.figure(figsize=(15, 14))
        plt.suptitle(f"Código de barras: {barcode}", fontsize=14, fontweight="bold")
        for i, (titulo, img) in enumerate(imagens_plot):
            plt.subplot(4, 3, i + 1)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.title(titulo, fontsize=11, fontweight="bold")
            plt.axis('off')

        plt.tight_layout()

        # Salva a figura em /imgs
        os.makedirs(PASTA_IMGS, exist_ok=True)
        caminho_img = os.path.join(PASTA_IMGS, f"{barcode}.jpg")
        plt.savefig(caminho_img, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figura salva em '{caminho_img}'")

        plt.show()

    except Exception as e:
        print(f"Erro: {e}")


# Fluxo principal
def main():
    barcodes = carregar_barcodes(ARQUIVO_BARCODES)

    if not barcodes:
        print("Nenhum código de barras para processar.")
        return

    for i, barcode in enumerate(barcodes, start=1):
        print("\n" + "#" * 60)
        print(f"# [{i}/{len(barcodes)}] Processando código de barras: {barcode}")
        print("#" * 60)
        processar_rotulo(barcode)


if __name__ == "__main__":
    main()