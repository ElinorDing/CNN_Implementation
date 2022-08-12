"""
Fun: CNN for MNIST classification
"""
import random

import numpy as np
import time
import h5py
import argparse
import os.path
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# from util import _create_batch
import json
import torchvision
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNNModel
from utils import str2bool
import torch.multiprocessing


## input hyper-paras
parser = argparse.ArgumentParser(description = "nueral networks")
parser.add_argument("-mode", dest="mode", type=str, default='train', help="train or test")
parser.add_argument("-num_epoches", dest="num_epoches", type=int, default=40, help="num of epoches")

parser.add_argument("-fc_hidden1", dest="fc_hidden1", type=int, default=100, help="dim of hidden neurons")
parser.add_argument("-fc_hidden2", dest="fc_hidden2", type=int, default=100, help="dim of hidden neurons")
parser.add_argument("-learning_rate", dest ="learning_rate", type=float, default=0.001, help = "learning rate")
parser.add_argument("-decay", dest ="decay", type=float, default=0.5, help = "learning rate")
parser.add_argument("-batch_size", dest="batch_size", type=int, default=100, help="batch size")
parser.add_argument("-dropout", dest ="dropout", type=float, default=0.4, help = "dropout prob")
parser.add_argument("-rotation", dest="rotation", type=int, default=10, help="image rotation")
parser.add_argument("-load_checkpoint", dest="load_checkpoint", type=str2bool, default=False, help="true of false")

parser.add_argument("-activation", dest="activation", type=str, default='relu', help="activation function")
# parser.add_argument("-MC", dest='MC', type=int, default=10, help="number of monte carlo")
parser.add_argument("-channel_out1", dest='channel_out1', type=int, default=64, help="number of channels")
parser.add_argument("-channel_out2", dest='channel_out2', type=int, default=64, help="number of channels")
parser.add_argument("-k_size", dest='k_size', type=int, default=4, help="size of filter")
parser.add_argument("-pooling_size", dest='pooling_size', type=int, default=2, help="size for max pooling")
parser.add_argument("-stride", dest='stride', type=int, default=1, help="stride for filter")
parser.add_argument("-max_stride", dest='max_stride', type=int, default=2, help="stride for max pooling")
parser.add_argument("-ckp_path", dest='ckp_path', type=str, default="checkpoint", help="path of checkpoint")

args = parser.parse_args()
	

def _load_data(DATA_PATH, batch_size):
	'''Data loader'''

	print("data_path: ", DATA_PATH)
	train_trans = transforms.Compose([transforms.RandomRotation(args.rotation),transforms.RandomHorizontalFlip(),\
								transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
	
	train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True,train=True, transform=train_trans)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
	
	## for testing
	test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
	test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True, train=False, transform=test_trans)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
	
	return train_loader, test_loader



def _compute_accuracy(y_pred, y_batch):
	## --------------------------------------------
	## write the code of computing accuracy below
	## --------------------------------------------
	accy = 100 * y_pred/y_batch
	return accy


def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
	"""Sets the learning rate to the initial LR decayed by 1/10 every args.lr epochs"""
	lr = learning_rate
	if (epoch > 5):
		lr = 0.001
	if (epoch >= 10):
		lr = 0.0001
	if (epoch > 20):
		lr = 0.00001
	
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	# print("learning_rate: ", lr)

def _save_checkpoint(state, ckp_path):
	print("=> Saving checkpoint")
	torch.save(state, ckp_path)

def main():

	use_cuda = torch.cuda.is_available() ## if have gpu or cpu 
	device = torch.device("cuda" if use_cuda else "cpu")
	torch.cuda.set_device(device=0) ## choose gpu number 0
	print("device: ", device)
	if use_cuda:
		torch.cuda.manual_seed(72)

	torch.multiprocessing.set_sharing_strategy('file_system')
	## initialize hyper-parameters
	num_epoches = args.num_epoches
	decay = args.decay
	learning_rate = args.learning_rate
	

	## step 1: Load data
	DATA_PATH = "./data/"
	train_loader, test_loader=_load_data(DATA_PATH, args.batch_size)
	classes = ('0','1','2','3','4','5','6','7','8','9')

	example = iter(test_loader)
	example_data, example_targets = example.next()
	##-------------------------------------------------------
	## please write the code about model initialization below
	##-------------------------------------------------------
	model = CNNModel(args) #kernel size, stride
	## load model to gpu or cpu
	model.to(device)
	
	## --------------------------------------------------
	## Complete code about defining the LOSS FUNCTION
	## --------------------------------------------------
	optimizer = optim.Adam(model.parameters(),lr=learning_rate)  ## optimizer
	loss_fun = nn.CrossEntropyLoss() ## cross entropy loss
	# writer = SummaryWriter()
	##--------------------------------------------
	## load checkpoint below if you need
	##--------------------------------------------
	if args.load_checkpoint:
		checkpoint = torch.load(args.ckp_path)
		CNNModel.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoches = checkpoint['epoch']

	
	##  model training
	if args.mode == 'train':
		train_counter = [x for x in range(num_epoches)]
		model = model.train()
		for epoch in range(num_epoches): #10-50

			## learning rate
			adjust_learning_rate(learning_rate, optimizer, epoch, decay)
			train_loss_record = list()
			for batch_id, (x_batch,y_labels) in enumerate(train_loader):
				x_batch,y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)

				## feed input data x into model
				output_y = model(x_batch)
				
				##---------------------------------------------------
				## write loss function below, refer to tutorial slides
				##----------------------------------------------------
				loss = loss_fun(output_y,y_labels)
				train_loss_record.append(loss.item())

				##----------------------------------------
				## write back propagation below
				##----------------------------------------
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				##------------------------------------------------------
				## get the predict result and then compute accuracy below
				##------------------------------------------------------

				##----------------------------------------------------------
				## loss.item() or use tensorboard to monitor the loss blow
				## if use loss.item(), you may use log txt files to save loss
				##----------------------------------------------------------
				# writer.add_scalar("Loss/train", loss, batch_id)
				mean_loss = sum(train_loss_record)/len(train_loss_record)
				print(f'Loss of epoch {epoch} was {mean_loss:.5f}')
			## -------------------------------------------------------------------
			## save checkpoint below (optional), every "epoch" save one checkpoint
			## -------------------------------------------------------------------
			checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
			_save_checkpoint(checkpoint,args.ckp_path)

		print("finish training")
	# writer.close()
			
				

	##------------------------------------
	##    model testing code below
	##------------------------------------
	model.eval()
	correct = 0
	total = 0
	correct_pred = {classname: 0 for classname in classes}
	total_pred = {classname: 0 for classname in classes}
	val_loss_record = list()
	test_counter = [x for x in range(num_epoches)]
	with torch.no_grad():
		for batch_id, (x_batch,y_labels) in enumerate(test_loader):
			x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
			##---------------------------------------
			## write the predict result below
			##---------------------------------------
			output_y = model(x_batch)

			##---------------------------------------------------
			## write loss function below, refer to tutorial slides
			##----------------------------------------------------
			loss = loss_fun(output_y,y_labels)
			val_loss_record.append(loss.item())


			##--------------------------------------------------
			## complete code for computing the accuracy below
			##---------------------------------------------------
			_, y_pred = torch.max(output_y.data, 1)
			for label, prediction in zip(y_labels,y_pred):
				if label == prediction:
					correct_pred[classes[label]] += 1
				total_pred[classes[label]] += 1

			total += y_labels.size(0)
			correct += (y_pred == y_labels).sum().item()


	# print accuracy for each class
	for classname, correct_count in correct_pred.items():
		# acc = _compute_accuracy(int(correct_count),total_pred[classname])
		accuracy = 100 * float(correct_count) / total_pred[classname]
		print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

	accuracy = _compute_accuracy(correct, total)
	print(f'Accuracy of the network on the test images: {accuracy} %')

	for i in range(6):
		f1 = plt.figure()
		f1.add_subplot(2,3,i+1)
		f1.tight_layout()
		f1.imshow(example_data[i][0], cmap='gray', interpolation='none')
		f1.title("Ground Truth: {}".format(example_targets[i]))
		f1.xticks([])
		f1.yticks([])
		f1.plot(train_counter, random.sample(train_loss_record,10), color='blue')
		f1.scatter(test_counter, random.sample(val_loss_record,10), color='red')
		f1.legend(['Train Loss', 'Test Loss'], loc='upper right')
		f1.xlabel('number of training examples seen')
		f1.ylabel('negative log likelihood loss')
		f1.savefig('Train-Loss.png')

	for i in range(6):
		f2 = plt.figure()
		f2.add_subplot(2,3,i+1)
		f2.tight_layout()
		f2.imshow(example_data[i][0], cmap='gray', interpolation='none')
		f2.title("Prediction")
		# plt.title("Prediction: {}".format(
		# 	y_pred.data.max(1, keepdim=True)[1][i].item()))
		f2.xticks([])
		f2.yticks([])
		f2.savefig('Prediction.png')



if __name__ == '__main__':
	time_start = time.time()
	main()
	time_end = time.time()
	print("running time: ", (time_end - time_start)/60.0, "mins")
	

