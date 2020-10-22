import numpy as np
import os, cv2, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from Counterfactual.counterfactual_dataset import counterfactual_mask
from DistributionalModels.distributional_model import DistributionalModel
from file_management import save_to_pickle, load_from_pickle

class MultichannelPassiveActiveModel(HypothesisModel):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.flat_input_sizes, self.flat_self = list(), list()
		if self.instanced: # TODO: some of the logic on this does not actually check out at the moment
			for name in self.environment_model.object_names:
				for _ in self.environment_model.object_counts[name]:
					if name == self.option_name:
						self.flat_self.append(1)
					else:
						self.flat_self.append(0)
					self.flat_input_sizes.append(self.environment_model.object_sizes[name])
		else:
			for name in self.environment_model.object_names:
				if name == self.option_name:
					self.flat_self.append(1)
				else:
					self.flat_self.append(0)
				self.flat_input_sizes.append(self.environment_model.object_sizes[name] * self.environment_model.object_counts[name])
		self.flat_output_size = self.environment_model.object_sizes[self.option_name] * self.environment_model.object_counts[self.option_name]
		self.diff = True # TODO: only predicts the difference for now
		self.multichannel_network = None
		self.lmda = kwargs["lambda"]
		self.mask_epsilon = kwargs["mask_epsilon"]
		self.iscuda = False

	def train(self, rollouts):
		'''
		trains the necessary components
		'''
		self.multichannel_network = MultiChannelNetwork(input_sizes=self.flat_input_sizes, output_size=sef.flat_output_size, passive_indexes = self.flat_self)
        self.class_optimizer = optim.Adam(self.multichannel_network.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)
        # lossfn = nn.BCELoss()
        # train classifier
        for i in range(10000):
            idxes, batchvals = rollouts.get_batch(20)
            outputs, mask = self.multichannel_network(batchvals.values.state)
            loss = self.multichannel_loss(outputs, batchvales.values.state_diff, mask)
            self.class_optimizer.zero_grad()
            (loss).backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.args.max_grad_norm)
            self.class_optimizer.step()
        self.gather_hypotheses(rollouts)

    def gather_hypotheses(self, rollouts):
    	'''
		Given a learned multichannel network, performs some kind of clustering to gather hypotheses about each of the object's effects on the target

    	'''

    def multichannel_loss(self, outputs, targets, mask):
    	loss = (outputs - targets).norm() + (mask - self.multichannel_network.target_mask).norm() * lmda
    	return loss

    def passive_active(self, masks):
    	pa = torch.ones((masks.size(0)))
    	pa[(masks - self.multichannel_network.target_mask).norm() <= self.mask_epsilon] = 1
    	if self.iscuda:
    		return pa.cuda()
    	return pa

	def predict(self, rollouts):
		'''
		predicts the next state from the factored state, as well as the passive and active components
		'''
		predictions, masks = self.multichannel_network(rollouts.get_values("state"))
		pa = self.passive_active(masks)
		return predictions, pa

	def sample(self, all=False):
		'''
		sample a hypothesis from the set of hypotheses, if @param all is true, returns all of the hypotheses
		'''

class MultiChannelNetwork(nn.Module):
    def __init__(self, input_sizes, output_size, passive_indexes, scale_factor=10):
    	self.target_mask = torch.tensor(passive_indexes).detach()
    	self.channels = list()
    	self.pas_idxes = [0,-1]
    	self.input_sizes = input_sizes
    	self.passive_idx = passive_indexes
    	for i in range(len(input_sizes)):
    		sze = input_sizes[i]
    		if passive_indexes[i] != 1:
    			net = SingleChannelNetwork(sze + output_size, output_size) # assumes output size is the same as flat size
    			if self.pas_idxes[1] < 0:
    				self.pas_idxes[1] = self.pas_idxes[0] + sze
    			else:
    				self.pas_idxes[1] += sze # assumes only a single range of flat indexes
    		else: # use only passive dynamics
    			net = SingleChannelNetwork(sze, output_size)
    		if self.pas_idxes[1] < 0:
    			self.pas_idxes[0] += sze
    		self.channels.append(net)
    	self.channels = torch.ModuleList(self.channels)
    	self.scale_factor = scale_factor


    def forward(self, x):
    	pas_state = x[:,self.flat_idxes[0]:self.flat_idxes[1]]
    	at = 0
    	predictions = list()
    	masks = list()
    	for i in range(len(input_sizes)):
    		sze = input_sizes[i]
    		if passive_indexes[i] != 1:
    			p, m = self.channels[i](torch.cat([pas_state, x[:,at:at+sze]], dim=1))
    			predictions.append(p)
    			masks.append(m * self.scale_factor)
    		else:
    			p,m = self.channels[i](pas_state)
    			predictions.append(p)
    			masks.append(m)
    		at += sze
    	predictions = torch.stack(predictions, dim=1)
    	masks = torch.stack(masks, dim = 1)
    	smx = nn.Softmax(dim=1)
    	masks = smx(masks) # while this normalizes, would prefer a "hard" max
    	# max_masks = masks.max(dim) # 
		predictions.transpose(1,2).transpose(0,1)*masks.transpose(1,0).transpose(2,1)
		return predictions, masks

class SingleChannelNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden = 30, num_int = 30, num_pred = 30):
        self.num_inputs = num_inputs
        self.l1 = nn.Linear(num_inputs, num_hidden)
        self.interaction_hidden = nn.Linear(num_hidden, num_int)
        self.pred_hidden = nn.Linear(num_hidden, num_pred)
       	self.interaction_out = nn.Linear(num_pred, 1)
       	self.pred_out = nn.Linear(num_pred, num_outputs)
       	self.layers = [self.l1, self.interaction_hidden, self.pred_hidden, self.interaction_out, self.pred_out]

    def reset_parameters(self): # TODO: use inheritance so this only has to appear once in the code
        relu_gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            if type(layer) == nn.Conv2d:
                if self.init_form == "orth":
                    nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') 
            elif issubclass(type(layer), Model):
                layer.reset_parameters()
            elif type(layer) == nn.Parameter:
                nn.init.uniform_(layer.data, 0.0, 0.2/np.prod(layer.data.shape))#.01 / layer.data.shape[0])
            else:
                fulllayer = layer
                if type(layer) != nn.ModuleList:
                    fulllayer = [layer]
                for layer in fulllayer:
                    # print("layer", self, layer)
                    if self.init_form == "orth":
                        nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                    if self.init_form == "uni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                         nn.init.uniform_(layer.weight.data, 0.0, 1 / layer.weight.data.shape[0])
                    if self.init_form == "smalluni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                        nn.init.uniform_(layer.weight.data, -.0001 / layer.weight.data.shape[0], .0001 / layer.weight.data.shape[0])
                    elif self.init_form == "xnorm":
                        torch.nn.init.xavier_normal_(layer.weight.data)
                    elif self.init_form == "xuni":
                        torch.nn.init.xavier_uniform_(layer.weight.data)
                    elif self.init_form == "eye":
                        torch.nn.init.eye_(layer.weight.data)
                    if layer.bias is not None:                
                        nn.init.uniform_(layer.bias.data, 0.0, 1e-6)
        print("parameter number", self.count_parameters(reuse=False))


    def forward(self, x):
        x = x / 84
        x = self.l1(x)
        x = F.relu(x)
        p = self.pred_hidden(x)
        p = F.relu(p)
        p = self.pred_out(p)
        i = self.interaction_hidden(x)
        i = F.relu(i)
        i = self.interaction_out(i)
        i = F.tanh(i)
        return p, i
