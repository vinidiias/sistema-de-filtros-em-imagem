import numpy as np
import cv2 as cv
from scipy import ndimage
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# threshold (limiarização)
def simple_threshold(img_path, min_threshold): #certo

    img = cv.imread(img_path) #carrega imagem
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # pixel com valor < 127 sera 0. Maior que 127 sera 255
    val, thresh = cv.threshold(gray, min_threshold, 255, cv.THRESH_BINARY)
    
    output_path = os.path.join('img_output/', 'threshould.png')
    
    img_result = cv.imwrite(output_path, thresh)
#simple_threshold('eu_e_kraskof.png', 200)

def laplacian_filter(img_path): # Define a função para aplicar o filtro Laplaciano

    # Carrega a imagem do caminho fornecido
    image = cv.imread(img_path)

    # Converte a imagem para escala de cinza
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Aplica o filtro Laplaciano para detectar bordas
    laplacian = cv.Laplacian(imageGray, cv.CV_64F)

    # Converte a imagem resultante para formato uint8 e calcula a magnitude absoluta
    laplacian = np.uint8(np.absolute(laplacian))

    # Converte a imagem em escala de cinza resultante para formato RGB
    laplacianRGB = cv.cvtColor(laplacian, cv.COLOR_BAYER_BG2BGR)

    # Combina a imagem original e a imagem resultante do filtro lado a lado
    combinado = np.hstack((image, laplacianRGB))
    
    # Define o caminho de saída para salvar a imagem filtrada
    output_path = os.path.join('img_output/', 'laplacian.png')
    
    # Salva a imagem filtrada no arquivo especificado
    img_result = cv.imwrite(output_path, laplacian)
    # cv.imshow("Laplacian filter", laplacian) # Descomente para exibir a imagem filtrada
    # cv.waitKey(0) # Espera por uma tecla para fechar a janela exibida

#laplacian_filter('eu_e_kraskof.png')

def gray_scale(img_path): #certo
    img = cv.imread(img_path)

    #remove a informacao de cor de deixa apenas a intensidade da luz
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #cv.imshow("GrayScale", gray_img)
    output_path = os.path.join('img_output/', 'grayScale.png')
    
    img_result = cv.imwrite(output_path, gray_img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
#gray_scale('eu_e_kraskof.png')

def low_pass_filter(img_path, n): # Define a função para aplicar um filtro passa-baixa

    # Carrega a imagem do caminho fornecido
    img = cv.imread(img_path)

    # Isso cria uma média simples dos pixels ao redor de cada pixel.
    kernel = np.ones((n, n), np.float32) / (n * n)

    # para cada pixel, o valor da media dos pixels na vizinha é substituido com o pixel central
    dst = cv.filter2D(img, -1, kernel)

    # Define o caminho de saída para salvar a imagem filtrada
    output_path = os.path.join('img_output/', 'low_pass.png')

    # Salva a imagem filtrada no arquivo especificado
    img_result = cv.imwrite(output_path, dst)

    # cv.imshow('low pass filter', dst)
    # cv.waitKey(0)

#low_pass_filter('eu_e_kraskof.png', 10)

#filtro passa baixa mediana
def low_med_pass_filter(img_path, n): # Define a função para aplicar um filtro passa-baixa mediana

    # Carrega a imagem do caminho fornecido
    img = cv.imread(img_path)

    # Aplica o filtro de mediana à imagem
    # O tamanho do kernel é definido pelo parâmetro n. O filtro de mediana usa um kernel quadrado (n x n)
    mediana = cv.medianBlur(img, n)

    # Define o caminho de saída para salvar a imagem filtrada
    output_path = os.path.join('img_output/', 'med_low_pass.png')
    
    # Salva a imagem filtrada no arquivo especificado
    img_result = cv.imwrite(output_path, mediana)

    # cv.imshow('Filtro passa baixa mediana', mediana)
    # cv.waitKey(0)


#low_med_pass_filter('eu_e_kraskof.png')

def high_pass_filter(img_path):
    # Lê a imagem do caminho especificado
    img = cv.imread(img_path)

    # Define o kernel para o filtro passa-alta básico
    kernel = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]])

    # Aplica o filtro passa-alta à imagem usando o kernel definido
    filter = cv.filter2D(src=img, ddepth=-1, kernel=kernel)

    # Define o caminho de saída para salvar a imagem filtrada
    output_path = os.path.join('img_output/', 'basic_high_pass.png')
    
    # Salva a imagem filtrada no arquivo especificado
    img_result = cv.imwrite(output_path, filter)
    #cv.imshow("Filtro passa alta básica", filter)
    #cv.waitKey()

def high_reinforcement_filter(img_path, A):
    # Lê a imagem do caminho especificado
    img = cv.imread(img_path)

    # Calcula o valor do parâmetro w para o kernel baseado no valor de A
    w = 9 * A - 1

    # Define o kernel para o filtro de alto reforço
    kernel = np.array([[-1, -1, -1],
                      [-1, w, -1],
                      [-1, -1, -1]])

    # Aplica o filtro de alto reforço à imagem usando o kernel definido
    filter = cv.filter2D(src=img, ddepth=-1, kernel=kernel)

    # Define o caminho de saída para salvar a imagem filtrada
    output_path = os.path.join('img_output/', 'high_reinforcement.png')
    
    # Salva a imagem filtrada no arquivo especificado
    img_result = cv.imwrite(output_path, filter)
    #cv.imshow("Filtro alto reforço", filter)
    #cv.waitKey()

def robert_filter(img_path):
    # Lê a imagem do caminho especificado
    img = cv.imread(img_path)

    # Converte a imagem para escala de cinza
    img_grayScale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Define os kernels para o filtro de Robert (vertical e horizontal)
    robert_v = np.array([[1, 0],
                        [0, -1]])
    
    robert_h = np.array([[0, 1],
                        [-1, 0]])

    # Aplica os kernels de Robert para detectar bordas vertical e horizontalmente
    vertical = ndimage.convolve(img_grayScale.astype(float), robert_v)
    horizontal = ndimage.convolve(img_grayScale.astype(float), robert_h)

    # Calcula a magnitude das bordas combinadas das duas direções
    out_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)

    # Define o caminho de saída para salvar a imagem filtrada
    output_path = os.path.join('img_output/', 'roberts.png')
    
    # Salva a imagem filtrada no arquivo especificado
    img_result = cv.imwrite(output_path, out_img)

    #cv.imshow('Roberts Filter', out_img)
    #cv.waitKey(0)

def sobel_filter(img_path):
    # Lê a imagem do caminho especificado
    image = cv.imread(img_path)
    # Converte a imagem para escala de cinza
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Aplica o filtro Sobel para detectar bordas na direção X
    sobelX = cv.Sobel(imageGray, cv.CV_64F, 1, 0)
    # Aplica o filtro Sobel para detectar bordas na direção Y
    sobelY = cv.Sobel(imageGray, cv.CV_64F, 0, 1)

    # Converte os resultados para valores absolutos e tipo uint8
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    # Combina os resultados das bordas detectadas nas duas direções
    sobelComb = cv.bitwise_or(sobelX, sobelY)

    # Define o caminho de saída para salvar a imagem filtrada
    output_path = os.path.join('img_output/', 'sobel.png')
    
    # Salva a imagem filtrada no arquivo especificado
    img_result = cv.imwrite(output_path, sobelComb)

    #cv.imshow("Sobel X", sobelX) # Mostra o filtro no eixo X
    #cv.imshow("Sobel Y", sobelY) # Mostra o filtro no eixo Y
    #cv.imshow("Sobel Completo", sobelComb)
    #cv.waitKey(0)

def zerocross_filter(img_path):
    # Lê a imagem do caminho especificado em escala de cinza
    image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # Aplica o filtro Laplacian para encontrar bordas
    laplacian = cv.Laplacian(image, cv.CV_64F)
    rows, cols = laplacian.shape

    # Cria uma matriz para armazenar os cruzamentos de zero
    zero_crossings = np.zeros((rows, cols), dtype=np.uint8)
    # Percorre cada pixel da imagem
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Verifica se o pixel é um cruzamento de zero
            if laplacian[i][j] == 0:
                if (laplacian[i - 1][j] < 0 and laplacian[i + 1][j] > 0) or \
                   (laplacian[i][j - 1] < 0 and laplacian[i][j + 1] > 0) or \
                   (laplacian[i - 1][j - 1] < 0 and laplacian[i + 1][j + 1] > 0) or \
                   (laplacian[i - 1][j + 1] < 0 and laplacian[i + 1][j - 1] > 0):
                    zero_crossings[i][j] = 255

    # Define o caminho de saída para salvar a imagem filtrada
    output_path = os.path.join('img_output/', 'zero_crossing.png')
    
    # Salva a imagem filtrada no arquivo especificado
    img_result = cv.imwrite(output_path, zero_crossings)
    #cv.imshow('Zero Crossings', zero_crossings)
    #cv.waitKey(0)

def canny(img_path, upper_threshold):
    # Lê a imagem do caminho especificado
    img = cv.imread(img_path)

    # Define os limites inferiores e superiores para o detector de bordas Canny
    t_lower = upper_threshold / 2
    t_upper = upper_threshold

    # Aplica o detector de bordas Canny à imagem
    edge = cv.Canny(img, t_lower, t_upper)

    # Define o caminho de saída para salvar a imagem filtrada
    output_path = os.path.join('img_output/', 'canny.png')
    
    # Salva a imagem filtrada no arquivo especificado
    img_result = cv.imwrite(output_path, edge)
    #cv.imshow('edge', edge)
    #cv.waitKey(0)

def salt_and_pepper_noise(img_path):
    # Lê a imagem do caminho especificado
    img = cv.imread(img_path)
    # Converte a imagem para escala de cinza
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Cria uma cópia da imagem em escala de cinza para aplicar o ruído
    img_gray_copy = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray_copy1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Cria uma máscara de ruído com valores aleatórios
    noise_mask = np.random.randint(0, 21, size=(img_gray.shape[0], img_gray.shape[1]), dtype=int)

    # Define os pixels para ruído de sal (0) e pimenta (255)
    zeros_pixel = np.where(noise_mask == 0)
    one_pixel = np.where(noise_mask == 20)

    # Aplica o ruído de sal e pimenta à imagem
    img_gray[zeros_pixel] = 0
    img_gray[one_pixel] = 255

    # Define o caminho de saída para salvar a imagem com ruído
    output_path = os.path.join('img_output/', 'saltPepper_noise.png')
    
    # Salva a imagem com ruído no arquivo especificado
    img_result = cv.imwrite(output_path, img_gray)

    #cv.imshow('salt', img_gray)
    #cv.waitKey(0)

def gaussian_noise(img_path, n):
    # Lê a imagem do caminho especificado
    img = cv.imread(img_path)

    # Aplica o filtro de desfoque Gaussiano para adicionar ruído gaussiano
    gau_noise = cv.GaussianBlur(img, (n, n), 0) # n deve ser ímpar

    # Define o caminho de saída para salvar a imagem com ruído gaussiano
    output_path = os.path.join('img_output/', 'gaussian_noise.png')
    
    # Salva a imagem com ruído gaussiano no arquivo especificado
    img_result = cv.imwrite(output_path, gau_noise)

    #cv.imshow('gaussian noise', gau_noise)
    #cv.waitKey(0)

def median_noise(img_path, n):
    # Lê a imagem do caminho especificado
    img = cv.imread(img_path)

    # Aplica o filtro de mediana para adicionar ruído mediano
    median_noise = cv.medianBlur(img, n) # n deve ser ímpar

    # Define o caminho de saída para salvar a imagem com ruído mediano
    output_path = os.path.join('img_output/', 'median_noise.png')
    
    # Salva a imagem com ruído mediano no arquivo especificado
    img_result = cv.imwrite(output_path, median_noise)

    #cv.imshow('median noise', median_noise)
    #cv.waitKey(0)

def prewitt(img_path, n):
    # Lê a imagem do caminho especificado em escala de cinza
    img = cv.imread(img_path, 0)

    # Aplica o filtro de desfoque Gaussiano para suavizar a imagem antes da aplicação do filtro Prewitt
    img_gaussian = cv.GaussianBlur(img, (n, n), 0)

    # Define os kernels para o filtro Prewitt (direção X e Y)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    # Aplica os kernels de Prewitt para detectar bordas nas direções X e Y
    img_prewittx = cv.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv.filter2D(img_gaussian, -1, kernely)

    # Combina os resultados das bordas detectadas nas duas direções
    img_prewitt = img_prewittx + img_prewitty

    # Define o caminho de saída para salvar a imagem filtrada
    output_path = os.path.join('img_output/', 'prewitt.png')
    
    # Salva a imagem filtrada no arquivo especificado
    img_result = cv.imwrite(output_path, img_prewitt)

    #cv.imshow('prewitt', img_prewitt)
    #cv.waitKey(0)

def watershed(img_path, n):
    # Lê a imagem do caminho especificado
    img = cv.imread(img_path)
    # Verifica se a imagem foi lida corretamente
    assert img is not None, "O arquivo não pôde ser lido, verifique com os.path.exists()"
    
    # Converte a imagem para escala de cinza
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Aplica a limiarização (Threshold) usando o método de Otsu
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Remove o ruído com uma operação de abertura morfológica
    kernel = np.ones((n, n), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    
    # Determina a área de fundo certo (background) com dilatação
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    
    # Calcula a distância da transformação para encontrar a área de frente certa (foreground)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Converte a área de frente certa para valores inteiros de 8 bits
    sure_fg = np.uint8(sure_fg)
    
    # Subtrai a área de frente certa da área de fundo para encontrar a região desconhecida
    unknown = cv.subtract(sure_bg, sure_fg)
    
    # Realiza a marcação dos componentes conectados na imagem
    ret, markers = cv.connectedComponents(sure_fg)
    
    # Adiciona 1 a todos os marcadores para que o fundo certo não seja zero, mas 1
    markers = markers + 1
    
    # Marca a região desconhecida como zero
    markers[unknown == 255] = 0
    
    # Aplica o algoritmo Watershed para segmentação de imagem
    markers = cv.watershed(img, markers)
    
    # Pinta as bordas detectadas pelo algoritmo com a cor vermelha
    img[markers == -1] = [255, 0, 0]
    
    # Define o caminho de saída para salvar a imagem segmentada
    output_path = os.path.join('img_output/', 'watershed_segmentation.png')
    img_result = cv.imwrite(output_path, img)

    # Exibe a imagem segmentada (comentado para evitar a abertura da janela)
    cv.imshow('Watershed', img)
    cv.waitKey(0)

def histograma(img_path):
    # Lê a imagem do caminho especificado em escala de cinza
    img = cv.imread(img_path, 0)

    # Aplica a equalização de histograma para melhorar o contraste da imagem
    equ = cv.equalizeHist(img)

    # Calcula o histograma da imagem equalizada
    hist = cv.calcHist([equ], [0], None, [256], [0, 256])

    # Plota e salva o histograma
    plt.clf()
    plt.plot(hist)
    plt.savefig('img_output/hist_equ.png')

    # Define o caminho de saída para salvar a imagem equalizada
    output_path = os.path.join('img_output/', 'histograma.png')
    img_result = cv.imwrite(output_path, equ)

    #cv.imshow('equ', equ)
    #cv.waitKey(0)

def histograma_adaptativo(img_path):
    # Lê a imagem do caminho especificado em escala de cinza
    img = cv.imread(img_path, 0)

    # Cria um objeto CLAHE (Contrast Limited Adaptive Histogram Equalization)
    ahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Aplica o CLAHE à imagem
    ahe_result = ahe.apply(img)

    # Calcula o histograma da imagem com CLAHE
    hist = cv.calcHist([ahe_result], [0], None, [256], [0, 256])
    
    # Plota e salva o histograma
    plt.clf()
    plt.plot(hist)
    plt.savefig('img_output/hist_adpt.png')

    # Define o caminho de saída para salvar a imagem com CLAHE
    output_path = os.path.join('img_output/', 'histograma_adaptativo.png')
    
    # Salva a imagem com CLAHE no arquivo especificado
    img_result = cv.imwrite(output_path, ahe_result)

    #cv.imshow('AHE result', ahe_result)
    #cv.waitKey(0)

def contagem_objetos(img_path):
    # Lê a imagem do caminho especificado
    img = cv.imread(img_path)
    # Converte a imagem para escala de cinza
    img_cinza = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Aplica o desfoque Gaussiano para suavizar a imagem
    img_blur = cv.GaussianBlur(img_cinza, (5, 5), 0)
    # Aplica a limiarização (Threshold) para segmentar objetos
    _, img_limiar = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # Aplica uma operação de fechamento morfológica para remover pequenos buracos
    kernel = np.ones((7, 7), np.uint8)
    img_fechamento = cv.morphologyEx(img_limiar, cv.MORPH_CLOSE, kernel)
    # Aplica dilatação para destacar os objetos
    kernel_dilate = np.ones((5, 5), np.uint8)
    img_dilatada = cv.dilate(img_fechamento, kernel_dilate, iterations=2)
    # Encontra contornos na imagem dilatada
    contornos, _ = cv.findContours(img_dilatada, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Cria uma cópia da imagem original para desenhar os contornos
    img_contorno = img.copy()

    num_objetos = 0

    # Desenha os contornos e numera os objetos
    for i, contorno in enumerate(contornos):
        cv.drawContours(img_contorno, [contorno], -1, (0, 255, 0), 3)
        M = cv.moments(contorno)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.putText(img_contorno, str(i + 1), (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        num_objetos += 1

    # Define o caminho de saída para salvar a imagem com contornos e contagem
    output_path = os.path.join('img_output/', 'conta_objetos.png')
    
    # Salva a imagem com contornos no arquivo especificado
    img_result = cv.imwrite(output_path, img_contorno)
    #cv.imshow('contar obj', img_contorno)
    #cv.waitKey(0)