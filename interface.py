import customtkinter as ctk
import webbrowser, os
from tkinter.filedialog import askdirectory, askopenfilename
import cv2 as cv
from PIL import Image
from tkinter import *
import filtros as filter
import pyautogui as pag

#janela principal
janela = ctk.CTk()
label = None
img = None

#valor do parametro
valor = 0.0
opcao = ""
label_value = None

#label da imagem carregada
img_original = None
label_imgOriginal = None

#frame do lado histograma
frame_4 = None

#label do histograma equalizado
label_hist = None
img_hist = None

#label do histograma adaptativo
label_hist_adpt = None
img_hist_adpt = None

#configurando a janela principal
janela.title("PID")
width_main = 2000
height_main = 1000
janela.geometry("1100x550")
janela.maxsize(width=width_main, height=height_main) #responsividade max
janela.minsize(width=500, height=250) #responsividade min
janela.resizable(width=True, height=False) # true para alteravel ou false para fixo

#metodos
#remove o frame dos histogramas
def remove_frame():
    if frame_4 is not None:
        frame_4.destroy()
        label_hist.destroy()
        label_hist_adpt.destroy()

#atualiza o histograma adaptativo
def update_hist_adpt(hist):
    global frame_4, img_hist_adpt, label_hist, label_hist_adpt, img_hist

    if frame_4 is None:
        frame_4 = ctk.CTkFrame(master=janela)
        frame_4.pack(side="right", expand=True)

    if label_hist is not None:
        label_hist.destroy()
        img_hist = None

    img_hist_adpt = hist

    if label_hist_adpt is None:
        label_hist_adpt = ctk.CTkLabel(master=frame_4, text=None, image=img_hist_adpt)
        label_hist_adpt.pack()
    else:
        label_hist_adpt.configure(image=img_hist_adpt)
        label_hist_adpt.image = img_hist

#atualiza a imagem do histograma equalizado
def update_hist_equ(hist):
    global frame_4, img_hist, label_hist, label_hist_adpt, img_hist_adpt

    if frame_4 is None:
        frame_4 = ctk.CTkFrame(master=janela)
        frame_4.pack(side="right", expand=True)
    
    if label_hist_adpt is not None:
        label_hist_adpt.destroy()
        img_hist_adpt = None

    img_hist = hist

    if label_hist is None:
        label_hist = ctk.CTkLabel(master=frame_4, text=None, image=img_hist)
        label_hist.pack()
    else:
        label_hist.configure(image=img_hist)
        label_hist.image = img_hist

#atualiza imagem original
def update_image(new_img):
    global img, label

    img = new_img  # Atualiza a imagem global

    if label is None:  # Se o label ainda não existir, cria um novo
        label = ctk.CTkLabel(master=frame_3, text=None, image=img)
        label.pack()
    else:  # Se o label já existir, atualiza a imagem
        label.configure(image=img)
        label.image = img 

#pega o valor da opcao
def optionMenu_choice(value):
    global opcao
    opcao = value

#acao de cada algoritmo
def optionmenu_callback():
    global input
    global opcao, valor
    global label, img, img_hist

    valor = float(input.get())

    if opcao == "Limiarização (Threshold)":
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.simple_threshold(path_main, valor)
            img = PhotoImage(file="img_output/threshould.png")
            remove_frame()
            update_image(img)      
    elif(opcao == "Escala de Cinza"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.gray_scale(path_main)
            remove_frame()
            img = PhotoImage(file="img_output/grayScale.png")
            update_image(img)
    elif(opcao == "Passa-Alta Básico"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.high_pass_filter(path_main)
            remove_frame()
            img = PhotoImage(file="img_output/basic_high_pass.png")
            update_image(img)
    elif(opcao == "Passa-Alta AR"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.high_reinforcement_filter(path_main, valor)
            remove_frame()
            img = PhotoImage(file="img_output/high_reinforcement.png")
            update_image(img)
    elif(opcao == "Passa-Baixa Média"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.low_pass_filter(path_main, valor)
            remove_frame()
            img = PhotoImage(file="img_output/low_pass.png")
            update_image(img)
    elif(opcao == "Passa-Baixa Mediana"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.low_med_pass_filter(path_main, valor)
            remove_frame()
            img = PhotoImage(file="img_output/med_low_pass.png")
            update_image(img)
    elif(opcao == "Roberts"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.robert_filter(path_main)
            remove_frame()
            img = PhotoImage(file="img_output/roberts.png")
            update_image(img)
    elif(opcao == "Prewitt"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.prewitt(path_main, valor)
            remove_frame()
            img = PhotoImage(file="img_output/prewitt.png")
            update_image(img)
    elif(opcao == "Sobel"):        
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.sobel_filter(path_main)
            remove_frame()
            img = PhotoImage(file="img_output/sobel.png")
            update_image(img)
    elif(opcao == "Log"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.laplacian_filter(path_main)
            remove_frame()
            img = PhotoImage(file="img_output/laplacian.png")
            update_image(img)
    elif(opcao == "Zerocross"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.zerocross_filter(path_main)
            remove_frame()
            img = PhotoImage(file="img_output/zero_crossing.png")
            update_image(img)
    elif(opcao == "Canny"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.canny(path_main, valor)
            remove_frame()
            img = PhotoImage(file="img_output/canny.png")
            update_image(img)
    elif(opcao == "Salt && Pepper Noise"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.salt_and_pepper_noise(path_main)
            remove_frame()
            img = PhotoImage(file="img_output/saltPepper_noise.png")
            update_image(img)
    elif(opcao == "Gaussian Noise"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.gaussian_noise(path_main, valor)
            remove_frame()
            img = PhotoImage(file="img_output/gaussian_noise.png")
            update_image(img)
    elif(opcao == "Median Noise"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.median_noise(path_main, valor)
            remove_frame()
            img = PhotoImage(file="img_output/median_noise.png")
            update_image(img)
    elif(opcao == "Watershed Segmentation"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.watershed(path_main, valor)
            remove_frame()
            img = PhotoImage(file="img_output/watershed_segmentation.png")
            update_image(img)
    elif(opcao == "Histograma (Esc.Cinza)"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.histograma(path_main)

            img = PhotoImage(file="img_output/histograma.png")
            img_hist = PhotoImage(file="img_output/hist_equ.png")

            update_image(img)
            update_hist_equ(img_hist)
    elif(opcao == "Ajuste Adap. de Hist."):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.histograma_adaptativo(path_main)

            img = PhotoImage(file="img_output/histograma_adaptativo.png")
            img_hist = PhotoImage(file="img_output/hist_adpt.png")

            update_image(img)
            update_hist_adpt(img_hist)
    elif(opcao == "Contagem simples"):
        if path_main == "":
            pag.alert(text="Carregue uma imagem!", title="Aviso!")
        else:
            filter.contagem_objetos(path_main)
            img = PhotoImage(file="img_output/conta_objetos.png")
            update_image(img)

#carregar imagem
def upload_image():
    global path_main
    global label_imgOriginal
    global img_original

    path = askopenfilename(title="Selecione uma imagem do computador")
    path_main = path

    img_original = PhotoImage(file=path)

    if label_imgOriginal is None:
        label_imgOriginal = ctk.CTkLabel(master=frame_2, text=None, image=img_original)
        label_imgOriginal.pack()
    else:
        label_imgOriginal.configure(image=img_original)
        label_imgOriginal.image = img_original

#frames
frame_1 = ctk.CTkFrame(master=janela) #frame à direta
frame_1.pack(side="left", fill="y")

frame_2 = ctk.CTkFrame(master=janela) #frame da imagem original
frame_2.pack(side="left", expand=True)

frame_3 = ctk.CTkFrame(master=janela) #frame da imagem resultante
frame_3.pack(side="left", expand=True)

#botoes
btn_imagem_download = ctk.CTkButton(master=frame_1, text="Carregar Imagem", command=upload_image)
btn_imagem_download.pack(side="bottom")

#selecao de opcoes dos algoritmos
optionMenu = ctk.CTkOptionMenu(master=frame_1, values=["Limiarização (Threshold)", "Escala de Cinza", "Passa-Alta Básico", "Passa-Alta AR", "Passa-Baixa Média", "Passa-Baixa Mediana", "Roberts", "Prewitt", "Sobel", "Log", "Zerocross", "Canny", "Salt && Pepper Noise", "Gaussian Noise", "Median Noise", "Watershed Segmentation", "Histograma (Esc.Cinza)", "Ajuste Adap. de Hist.", "Contagem simples"], command=optionMenu_choice, fg_color="#033333")
optionMenu.pack()

#entrada de valor
input = ctk.CTkEntry(frame_1, placeholder_text='Digite um valor', fg_color='transparent', text_color=("black", "white"), placeholder_text_color=("gray", "lightgray"), font=("Arial", 12), border_width=1)
input.insert(0, 0)
input.pack()

#botao de aplicar
optionApply = ctk.CTkButton(master=frame_1, text='Aplicar', command=optionmenu_callback)
optionApply.pack()

#caminho das imagens carregadas
path_main = ""

#executa a interface
janela.mainloop() # roda a janela