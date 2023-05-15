from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import segmentation_models_pytorch as smp

app = Flask(__name__)

# Chargement du modèle
model = None
transform = None

def load_model():
    global model, transform

    nom_de_architecture = "DeeplabV3"
    nom_de_l_encodeur = "resnet18"
    poids_de_l_encodeur = "imagenet"
    n_classes = 20

    model = OurModel(archi_name=nom_de_architecture, encoder_name=nom_de_l_encodeur, encoder_weights=poids_de_l_encodeur)
    model.load_state_dict(torch.load('DeeplabV3.resnet18.imagenet.pth'))
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Fonction pour effectuer la prédiction sur une image donnée
def predict(image):
    global model, transform

    input_image = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(input_image)

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    inv_img = inv_normalize(input_image.squeeze(0)).cpu()
    output_x = output.detach().cpu().squeeze(0)
    decoded_output = decode_segmap(torch.argmax(output_x, 0))

    return np.moveaxis(inv_img.numpy(), 0, 2), decoded_output

@app.route('/predict', methods=['POST'])
def perform_prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image = Image.open(request.files['image'])
    input_image, mask = predict(image)

    # Conversion des images en formats compatibles avec la réponse JSON
    input_image = Image.fromarray((input_image * 255).astype(np.uint8))
    mask = Image.fromarray((mask * 255).astype(np.uint8))

    # Création d'un dictionnaire contenant les résultats
    result = {
        'input_image': input_image,
        'mask': mask
    }

    # Conversion du dictionnaire en JSON
    result_json = jsonify(result)

    return result_json

if __name__ == '__main__':
    # Chargement du modèle au démarrage de l'application
    load_model()

    # Démarrage de l'application Flask
    app.run()

