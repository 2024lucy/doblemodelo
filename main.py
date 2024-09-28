import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Cargar los dos modelos entrenados desde Hugging Face
model_1_id = "2024lucy/2024lucy"  # Modelo 1: Lucy
model_2_id = "2024lucy/2024jose"  # Modelo 2: Jose

# Configurar los pipelines de ambos modelos
pipe_1 = StableDiffusionPipeline.from_pretrained(model_1_id).to("cuda")
pipe_2 = StableDiffusionPipeline.from_pretrained(model_2_id).to("cuda")

# Describir a las dos personas (ajusta los prompts a tu gusto)
prompt_1 = "A detailed image of Lucy smiling in a casual setting"  # Prompt para Lucy
prompt_2 = "A detailed image of Jose looking thoughtful in an office"  # Prompt para Jose

# Generar la imagen de la primera persona (Lucy)
image_1 = pipe_1(prompt_1).images[0]

# Generar la imagen de la segunda persona (Jose)
image_2 = pipe_2(prompt_2).images[0]

# Función para combinar las dos imágenes lado a lado
def combine_images(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Crear una nueva imagen con el doble de ancho
    combined_image = Image.new('RGB', (width1 + width2, max(height1, height2)))

    # Pegar ambas imágenes en la nueva imagen
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1, 0))

    return combined_image

# Combinar las imágenes de Lucy y Jose
final_image = combine_images(image_1, image_2)

# Guardar la imagen final en la carpeta 'output'
final_image.save("output/combined_image.png")

# Mostrar la imagen final (opcional)
final_image.show()
