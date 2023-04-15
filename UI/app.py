import sys
sys.path.append('../')
sys.path.append('../ImgProcessing/')

import gradio as gr
import pandas as pd
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
from imProcessingPipeline import improcessing as process

device = torch.device('cpu')

df_colorMapping = pd.read_csv('../Labelling/colorLabelEncodingMapping.csv') 
num_color_class = len(df_colorMapping.index) - 1
labels_color = df_colorMapping['Class'].tolist()

df_textureMapping = pd.read_csv('../Labelling/texture2LabelEncodingMapping.csv') 
num_texture_class = len(df_textureMapping.index) - 1
labels_texture = df_textureMapping['Class'].tolist()

vgg11_color = models.alexnet()
vgg11_color.classifier[6] = nn.Linear(vgg11_color.classifier[6].in_features , num_color_class)
vgg11_color.load_state_dict(torch.load("color_weights.pth", map_location=device))
vgg11_color.eval()

vgg11_texture = models.alexnet()
vgg11_texture.classifier[6] = nn.Linear(vgg11_texture.classifier[6].in_features , num_texture_class)
vgg11_texture.load_state_dict(torch.load("texture_weights.pth", map_location=device))
vgg11_texture.eval()

def get_confidences(img, model, labels, num_class):
    with torch.no_grad():
        output = model(img)
        preds = nn.functional.softmax(output[0], dim = 0)
        confidence = {labels[i]: float(preds[i]) for i in range(num_class)}  
        pred = max(confidence, key = confidence.get)
    return confidence, pred

def predict(img, model_name):
    print(model_name)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    processed_img = process(img)[0]
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.ToTensor()
    input = transform(processed_img).unsqueeze(0)
    # confidence_color, pred_color  = None, None
    confidence_color, pred_color = get_confidences(input, vgg11_color, labels_color, num_color_class)
    confidence_texture, pred_texture = get_confidences(input, vgg11_texture, labels_texture, num_texture_class)
    
    pred_msg = f'This is a {pred_color} {pred_texture} sherd!'
    return processed_img, confidence_color, confidence_texture, pred_msg
    
# with gr.Blocks() as demo:
#     with gr.Column(scale=0.5, min_width=600):
#         name = gr.Image(type="pil")
#         greet_btn = gr.Button("Greet")
#     with gr.Column(scale=0.5, min_width=600):
#         output = gr.Textbox(label="Output Box")
#     greet_btn.click(fn=greet, inputs=name, outputs=output)

description = '''Digitization of archaeology is in great demand. Since 2009, a team of researchers and students led by Dr. Cobb has been investigating the area around Vedi, Armenia, aiming at understanding human life and mobility in the ancient landscapes of the Near East. A large volume of sherds was excavated and documented with photography. Inspired by the recent advancement in computer vision and deep learning, this project attempts to explore various deep learning models to classify and compare sherds unearthed. \n More info can be found in http://openarchaeology.org/armenia/index
'''

demo = gr.Interface(fn=predict, 
            #  inputs=gr.Image(type="pil"),
            #  outputs=[gr.Label(num_top_classes=3), "text"],
            inputs=[gr.Image(label='Please Upload Sherd Image', type="numpy"),
                    gr.Radio(choices = ["AlexNet", "VGG", "ResNet", "SimNet"], value='AlexNet',label="Please select a model")],
            examples=[["exampleImgs/raw_sherd2.jpg",None],
                      ["exampleImgs/raw_sherd4.jpg",None],
                      ["exampleImgs/raw_sherd5.jpg",None]],
            outputs=[gr.Image(label='Processed Image'),
                     gr.Label(label='Color Prediction',num_top_classes=3), 
                     gr.Label(label='Texture Prediction', num_top_classes=3),
                     gr.Textbox(label="Prediction")],
             title = "⛏️⛏️⛏️Machine Learning in Archaeology⛏️⛏️⛏️", 
             description= description,
             allow_flagging="never")



demo.launch(server_port=8080)   

