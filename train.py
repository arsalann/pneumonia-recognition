######## COMMAND LINE HELPER ########
# python train.py --data_dir xray --gpu true --epochs 23 --arch alexnet --learning_rate 0.001 --checkpoint nnmodel.pt --hidden_units 2040
#
#
#
#
#
##################################################
################ TRAINING MACHINE ################
##################################################




# IMPORTS #
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data


import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

import time
import copy
import argparse
import os



#####################################################################

#### ENABLES CPU TO RUN TRAINING WITHOUT FREEZING ####

def run():
	torch.multiprocessing.freeze_support()
	print('loop')

if __name__ == '__main__':
	run()

####################################################################

	######## ARGPARSE CODE ########

	parser = argparse.ArgumentParser(description='This is where you enter the parameters')
	parser.add_argument('--data_dir', type=str, help='path to raw')
	parser.add_argument('--gpu', action='store_true', help='GPU availability')
	parser.add_argument('--epochs', type=int, help='Number of epochs')
	parser.add_argument('--arch', type=str, help='Model architecture')
	parser.add_argument('--learning_rate', type=float, help='Learning rate')
	parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
	parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')

	args, _ = parser.parse_known_args()





	######## DEFINE THE MODEL ########

	def load_model(arch='vgg19', num_labels=102, hidden_units=4096):
		# PRETRAINED MODEL #
		if arch=='vgg19':
			model = models.vgg19(pretrained=True)
		elif arch=='alexnet':
			model = models.alexnet(pretrained=True)
		else:
			raise ValueError('Please enter alexnet or vgg19 architecture')
			
		# LOCK PARAMETERS #
		for param in model.parameters():
			param.requires_grad = False
		
		# DELETE LAST LATER #
		features = list(model.classifier.children())[:-1]

		# # OF FILTERS #
		num_filters = model.classifier[len(features)].in_features

		# NEW LAYERS
		features.extend([
			nn.Dropout(),
			nn.Linear(num_filters, hidden_units),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(hidden_units, hidden_units),
			nn.ReLU(True),
			nn.Linear(hidden_units, num_labels),
		])
		
		model.classifier = nn.Sequential(*features)

		return model





	######## TRAIN THE MODEL ########

	def train_model(image_datasets, arch='alexnet', hidden_units=2040, epochs=23, learning_rate=0.001, gpu=False, checkpoint='nnmodel.pt'):
		
		# ABOVE PARAMETERS ARE THE DEFAULT, BELOW ARE IF SPECIFIED IN COMMAND LINE #
		if args.arch:
			arch = args.arch
			
		if args.hidden_units:
			hidden_units = args.hidden_units

		if args.epochs:
			epochs = args.epochs
				
		if args.learning_rate:
			learning_rate = args.learning_rate

		if args.gpu:
			gpu = args.gpu

		if args.checkpoint:
			checkpoint = args.checkpoint		



		# DEFINE DATALOADER #
		dataloaders = {
			x: data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=2)
			for x in list(image_datasets.keys())
		}



		# DEFINE FUNCTION TO MEASURE DATASET SIZE #
		dataset_sizes = {
			x: len(dataloaders[x].dataset) 
			for x in list(image_datasets.keys())
		}	



		# PARAMETER SANITY CHECK #
		print('Network architecture:', arch)
		print('Number of hidden units:', hidden_units)
		print('Number of epochs:', epochs)
		print('Learning rate:', learning_rate)




		# RELOAD MODEL #
		num_labels = len(image_datasets['train'].classes)
		model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)



		# CUDA GPU FUNCTIONS #
		using_gpu = gpu and torch.cuda.is_available()
		device = torch.device("cpu")

		if using_gpu:
			print("Using GPU for training!")
			device = torch.device("cuda:0")
			model.cuda()
		elif gpu and not using_gpu:
			print("CUDA unavailable. Using CPU for training!")
		else:
			print("Using CPU for training!")

		

		# DEFINE CRITERION, OPTIMIZER, AND SCHEDULE FOR PARAMETERS THAT REQUIRE GRADIENT #
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)
		scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



		# DEFINE SINCE #
		since = time.time()

		# DEFINE BEST MODEL WEIGHTS #
		best_model_wts = copy.deepcopy(model.state_dict())
		best_acc = 0.0



		# TRAIN AND VALIDATE EACH EPOCH #

		for epoch in range(epochs):
			print()
			print('Epoch {}/{}'.format(epoch + 1, epochs))
			print('----------------the machine is learning!----------------')
			
			for phase in ['train', 'validate']:
				if phase == 'train':
					# scheduler.step()
					model.train()
				else:
					model.eval()

				running_loss = 0.0
				running_corrects = 0

				# DATA ITERATION #
				for inputs, labels in dataloaders[phase]:				
					inputs = inputs.to(device)
					labels = labels.to(device)

					# RESET PARAMETER GRADIENTS TO ZERO #
					optimizer.zero_grad()

					# ENABLE OR DISABLE GRADIENT BASED ON PARAMETER #
					with torch.set_grad_enabled(phase == 'train'):
						outputs = model(inputs)
						_, preds = torch.max(outputs, 1)
						loss = criterion(outputs, labels)

						# CALCULATE LOSS FOR PARAMETERS THAT REQUIRE GRADIENT & OPTIMIZE #
						if phase == 'train':
							loss.backward()
							optimizer.step()
							scheduler.step()

					# ANALYZE PERFORMANCE #
					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)

				epoch_loss = running_loss / dataset_sizes[phase]
				epoch_acc = running_corrects.double() / dataset_sizes[phase]

				print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

				# DEEP COPY MODEL #
				if phase == 'validate' and epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())

			print()

		# ANALYZE PERFORMANCE TIME #
		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		print('Best validation Accuracy: {:4f}'.format(best_acc))

		# LOAD BEST WEIGHTS #
		model.load_state_dict(best_model_wts)

		# DETERMINE INDEX CLASSES #
		model.class_to_idx = image_datasets['train'].class_to_idx

		# CHECKPOINT #
		if checkpoint:
			print ('Saving checkpoint to:', checkpoint) 
			checkpoint_dict = {
				'arch': arch,
				'class_to_idx': model.class_to_idx, 
				'state_dict': model.state_dict(),
				'hidden_units': hidden_units
		}
		torch.save(checkpoint_dict, checkpoint)
		return model




	# STANDARD TRANFORMS #

	if args.data_dir:	
		# STANDARD TRANSFORMATIONS #
		data_transforms = {
			'train': transforms.Compose([
				transforms.Resize(256),
				transforms.RandomCrop(224),
				transforms.RandomRotation(45),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			]),
			'test': transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			]),
			'validate': transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
		}
		
		# LOAD DATASETS #
		image_datasets = {
			x: datasets.ImageFolder(root=args.data_dir + '/' + x, transform=data_transforms[x])
			for x in list(data_transforms.keys())
		}
		


	######## COMMENCE LEARNING ########		
		train_model(image_datasets)