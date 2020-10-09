######## COMMAND LINE HELPER ########
# python predict.py --image predict.jpg --checkpoint nnmodel.pt --topk 5 --labels cat_to_name.json --gpu true
#
#
#
#
#
####################################################
################ PREDICTION MACHINE ################
####################################################


#### IMPORTS ####

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from train import load_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

import argparse
import os



######## ARGPARSE CODE ########

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Image to predict')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint to use when predicting')
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--labels', type=str, help='JSON file containing label names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args, _ = parser.parse_known_args()


# LABEL NAMES #
with open('cat_to_name.json', 'r') as f:
	cat_to_name = json.load(f)


# PREDICT CLASS OF THE IMAGE FILE #
def predict(image, checkpoint, topk=5, labels='', gpu=False):

	if args.image:
		image = args.image	 
		
	if args.checkpoint:
		checkpoint = args.checkpoint

	if args.topk:
		topk = args.topk
			
	if args.labels:
		labels = args.labels

	if args.gpu:
		gpu = args.gpu
	
	# LOAD CHECKPOINT #
	checkpoint_dict = torch.load(checkpoint)
	arch = checkpoint_dict['arch']
	hidden_units = checkpoint_dict['hidden_units']
	num_labels = len(checkpoint_dict['class_to_idx'])
	class_to_idx = checkpoint_dict['class_to_idx']
	idx_to_class = {val: key for key, val in class_to_idx.items()}
		
	model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

	


	using_gpu = gpu and torch.cuda.is_available()
	device = torch.device("cpu")

	# GPU USAGE #
	if using_gpu:
			print("Using GPU for training!")
			device = torch.device("cuda:0")
			model.cuda()
	elif use_gpu and not using_gpu:
			print("CUDA unavailable. Using CPU for training!")
	else:
			print("Using CPU for training!")
	model.eval()
	
	img = Image.open(image)
	img_loader = transforms.Compose([
		transforms.Resize(256), 
		transforms.CenterCrop(224), 
		transforms.ToTensor()])

	img = img_loader(img).float()
	image = np.array(img) 

	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	image = (np.transpose(image, (1, 2, 0)) - mean)/std    
	image = np.transpose(image, (2, 0, 1))

	image = Variable(torch.FloatTensor(image), requires_grad=True)
	image = image.unsqueeze(0)
	




	if using_gpu:
		image = image.cuda()

	result = model(image).topk(topk)

	if using_gpu:
		probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
		classes = result[1].data.cpu().numpy()[0]
	else:       
		probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
		classes = result[1].data.numpy()[0]


	class_names = [cat_to_name[idx_to_class[cls]] for cls in classes]
	print(class_names)
	print(probs)


	img = mpimg.imread(args.image)
	f, axarr = plt.subplots(2,1)

	axarr[0].imshow(img)

	y_pos = np.arange(len(classes))

	axarr[1].barh(y_pos, probs, align='center', color='dark blue')
	axarr[1].set_yticks(y_pos)
	axarr[1].set_yticklabels(class_names)
	axarr[1].invert_yaxis()
	_ = axarr[1].set_xlabel('Probs')

	plt.show()


# PREDICT FROM COMMAND LINE #
if args.image and args.checkpoint:
	predict(args.image, args.checkpoint)