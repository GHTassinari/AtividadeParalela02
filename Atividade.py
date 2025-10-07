# Instalei eles dando pip install -r requirements.txt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

url = "https://quatrorodas.abril.com.br/wp-content/uploads/2023/06/1FLP00991.jpg?quality=70&strip=info&w=1280&h=720&crop=1"
response = requests.get(url)
img_pil = Image.open(BytesIO(response.content))
img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis('off')
plt.tight_layout()
plt.show()

blur = cv2.blur(img, (15, 15))
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
plt.title("Filtro de Suavização por Borramento")
plt.axis('off')
plt.tight_layout()
plt.show()

gaussian = cv2.GaussianBlur(img, (15, 15), 0)
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
plt.title("Filtro Gaussiano")
plt.axis('off')
plt.tight_layout()
plt.show()

median = cv2.medianBlur(img, 15)
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
plt.title("Filtro Mediano Não-linear")
plt.axis('off')
plt.tight_layout()
plt.show()

kernel = np.ones((15, 15), np.float32) / 225
filtro_manual = cv2.filter2D(img, -1, kernel)
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(filtro_manual, cv2.COLOR_BGR2RGB))
plt.title("Filtro Manual de Suavização")
plt.axis('off')
plt.tight_layout()
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)
plt.figure(figsize=(10, 6))
plt.imshow(sobel, cmap='gray')
plt.title("Filtro de Realce Sobel")
plt.axis('off')
plt.tight_layout()
plt.show()

laplacian = cv2.Laplacian(gray, cv2.CV_64F)
plt.figure(figsize=(10, 6))
plt.imshow(laplacian, cmap='gray')
plt.title("Filtro Espacial Laplaciano")
plt.axis('off')
plt.tight_layout()
plt.show()

canny = cv2.Canny(gray, 100, 200)
plt.figure(figsize=(10, 6))
plt.imshow(canny, cmap='gray')
plt.title("Filtro Canny")
plt.axis('off')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(18, 18))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original")
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Borramento")
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Gaussiano")
axes[0, 2].axis('off')

axes[1, 0].imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Mediano")
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(filtro_manual, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Manual")
axes[1, 1].axis('off')

axes[1, 2].imshow(sobel, cmap='gray')
axes[1, 2].set_title("Sobel")
axes[1, 2].axis('off')

axes[2, 0].imshow(laplacian, cmap='gray')
axes[2, 0].set_title("Laplaciano")
axes[2, 0].axis('off')

axes[2, 1].imshow(canny, cmap='gray')
axes[2, 1].set_title("Canny")
axes[2, 1].axis('off')

axes[2, 2].axis('off')

plt.suptitle("Todos os Filtros Aplicados", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()