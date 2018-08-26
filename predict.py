# Imports here
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from PIL import Image

use_gpu = torch.cuda.is_available

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    size = 224
    # TODO: Process a PIL image for use in a PyTorch model
    width, height = image.size
    
    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)
        
    resized_image = image.resize((width, height))
        
    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    cropped_image = image.crop((x0, y0, x1, y1))
    np_image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    np_image_array = (np_image - mean) / std
    np_image_array = np_image.transpose((2, 0, 1))
    
    return np_image_array



def load_checkpoint(args):
    checkpoint_provided = torch.load(args.saved_model)
    if checkpoint_provided['arch'] == 'vgg':
        model = models.vgg16()        
    elif checkpoint_provided['arch'] == 'densenet':
        model = models.densenet121()
        

    # build the classifier part of model
    num_features = model.classifier[0].in_features
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_features, 512)),
                          ('relu', nn.ReLU()),
                          ('drpot', nn.Dropout(p=0.5)),
                          ('hidden', nn.Linear(512, args.hidden_units)),                        
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1)),
                          ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint_provided['state_dict'])
    
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Using GPU")
        else:
            print("Using CPU since GPU is not available/configured")

    class_to_idx = checkpoint_provided['class_to_idx']
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class

def predict(args, image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image = torch.FloatTensor([process_image(Image.open(image_path))])
    
    if use_gpu and args.gpu:
        model = model.cuda()    
    model.eval()

    if use_gpu and args.gpu:
        output = model.forward(Variable(image.cuda()))
    else:
        output = model.forward(Variable(image))
    
    pobabilities = torch.exp(output.cpu()).data.numpy()[0]

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class

def main():

    parser = argparse.ArgumentParser(description='Flower Classification Predictor')
    parser.add_argument('--gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('--image_path', type=str, help='path of image')
    parser.add_argument('--hidden_units', type=int, default=100, help='hidden units for fc layer')
    parser.add_argument('--saved_model' , type=str, default='my_checkpoint_cmd.pth', help='path of your saved model')
    parser.add_argument('--mapper_json' , type=str, default='cat_to_name.json', help='path of your mapper from category to name')
    parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')

    args = parser.parse_args()


    import json
    with open(args.mapper_json, 'r') as f:
        cat_to_name = json.load(f)

    model, class_to_idx, idx_to_class = load_checkpoint(args)
    top_probability, top_class = predict(args, args.image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=args.topk)
                                              
    print('Predicted Classes: ', top_class)
    print ('Class Names: ')
    [print(cat_to_name[x]) for x in top_class]
    print('Predicted Probability: ', top_probability)

if __name__ == "__main__":
    main()
