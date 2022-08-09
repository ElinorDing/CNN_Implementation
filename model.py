"""
define moduals of model
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
	"""docstring for ClassName"""
	
	def __init__(self, args):
		super(CNNModel, self).__init__()
		##-----------------------------------------------------------
		## define the model architecture here
		## MNIST image input size batch * 28 * 28 (one input channel)
		##-----------------------------------------------------------
		# padding = (args.k_size-1)/2
		# ## define CNN layers below
		# self.conv1 = nn.Conv2d(in_channels=1,out_channels=args.channel_out1,kernel_size= args.k_size, stride= args.stride,padding= padding)
		#
		# self.conv2 = nn.Conv2d(in_channels= args.channel_out1, out_channels=args.channel_out2,kernel_size=args.k_size, stride = args.stride, padding= padding)
		#
		# # self.conv3 = nn.Conv2d(in_channels= args.channel_out2, out_channels=args.channel_out2,kernel_size=args.k_size, stride = args.stride, padding= padding),
		# self.maxPool = nn.MaxPool2d(kernel_size=args.pooling_size,stride= args.stride)
		# self.activation = args.activation
		# self.dropout = args.dropout
		self.cov = nn.Sequential(
			nn.Conv2d(in_channels=1,out_channels=args.channel_out1,kernel_size= args.k_size, stride= args.stride),
			nn.BatchNorm2d(args.channel_out1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=args.pooling_size,stride= args.stride),
			nn.Conv2d(in_channels= args.channel_out1, out_channels=args.channel_out2,kernel_size=args.k_size, stride = args.stride),
			nn.BatchNorm2d(args.channel_out2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=args.pooling_size,stride= args.stride),
			nn.Conv2d(in_channels= args.channel_out1, out_channels=args.channel_out2,kernel_size=args.k_size, stride = args.stride),
			nn.BatchNorm2d(args.channel_out2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=args.pooling_size,stride= args.stride),
			nn.Dropout(args.dropout)
		)

		##------------------------------------------------
		## write code to define fully connected layer below
		##------------------------------------------------
		in_size = args.channel_out2*16*16
		out_size = 10
		self.fc = nn.Linear(in_size, out_size)
		

	'''feed features to the model'''
	def forward(self, x):  #default
		
		##---------------------------------------------------------
		## write code to feed input features to the CNN models defined above
		##---------------------------------------------------------

		# x = self.conv1(x)
		# x = self.activation(x)
		# x = self.maxPool(x)
		# x = self.conv2(x)
		# x = self.activation(x)
		# x = self.maxPool(x)
		# x = self.conv2(x)
		# x = self.activation(x)
		# x = self.maxPool(x)
		# x= self.dropout(x)
		x_out = self.cov(x)

		## write flatten tensor code below (it is done)
		x = torch.flatten(x_out,1) # x_out is output of last layer
		print("x shape",x.shape)
		## ---------------------------------------------------
		## write fully connected layer (Linear layer) below
		## ---------------------------------------------------
		result = self.fc(x) # predict y
		
		
		return result
        
		
		
	
		