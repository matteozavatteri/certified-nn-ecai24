import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
import argparse
import shutil 
from conf import *

import random
import csv
import os
import time
import copy
import json

import sys
import random

#device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("Using device: {}".format(device))
torch.set_default_device(device)

def load_and_normalize(dataset):
	# Dataset preprocessing
	Normalized = []

	with open(dataset, 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter=',', skipinitialspace=True, fieldnames=Fieldnames)
		for row in reader: 
			new_row = []
			for f, u in FUB.items():
				new_row.append(max(0,min(1,float(row[f]) / u)))
				assert new_row[-1] >= 0 and new_row[-1] <= 1
			for o in NN_Outputs:
				new_row.append(float(row[o]) / 5)
				assert new_row[-1] >= 0 and new_row[-1] <= 1
			Normalized.append(new_row)
	
	return Normalized
	
def neural_network(hidden_layers, neurons_per_hidden_layer):

	InputNeurons  = len(NN_Inputs)
	OutputNeurons = len(NN_Outputs)

	layers = []
	layers.append(nn.Linear(InputNeurons,neurons_per_hidden_layer))
	layers.append(nn.ReLU())
	
	for _ in range(hidden_layers-1):
		layers.append(nn.Linear(neurons_per_hidden_layer,neurons_per_hidden_layer))
		layers.append(nn.ReLU())

	layers.append(nn.Linear(neurons_per_hidden_layer,OutputNeurons))
	network = nn.Sequential(*layers)
	
	return network

def train(network, X_train, y_train, X_valid, y_valid, epochs, early_stopping, learning_rate):
	optimizer = optim.Adam(network.parameters(), lr=learning_rate)
	batch_size = 100

	loss_fn = MSELoss()
	best_model = copy.deepcopy(network)
	best_loss  = None

	with torch.no_grad():
		y_pred = network(X_valid)
		best_loss = loss_fn(y_pred, y_valid).item()

	assert best_loss is not None

	#print("Initial network: loss(validation) = {}".format(best_loss))

	stop = 0
	X_train_tensor   = torch.tensor(X_train, dtype=torch.float32)
	y_train_tensor   = torch.tensor(y_train, dtype=torch.float32)
	train = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
	train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True, generator=torch.Generator(device=device))

	for epoch in range(epochs):
		for data in train_loader:
			# Every data instance is an input + label pair
			Xbatch, ybatch = data
			y_pred = network(Xbatch)
			loss = loss_fn(y_pred, ybatch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		# compute loss on valid
		with torch.no_grad():
			y_pred = network(X_valid)
			lossV = loss_fn(y_pred, y_valid).item()

		if lossV < best_loss:
			best_model = copy.deepcopy(network)
			best_loss  = lossV
			#print("Update after epoch {}: loss(validation) = {}".format(epoch, lossV))
			stop = 0
		else:
			stop += 1

		if stop > early_stopping:
			break
		
	return best_model, best_loss

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("property", help="1,2,3")
	parser.add_argument("--learning-rate", type=float, default=0.001, help="learning rate (0.01, 0.001, 0.0001; default: 0.001)")
	parser.add_argument("--epochs", type=int, default=100, help="number of epochs (default: 100)")
	parser.add_argument("--early-stopping", type=int, default=100, help="early stopping (default: 100)")
	parser.add_argument("--hidden-layers", type=int, default=7, help="number of hidden layers (default: 7)")
	parser.add_argument("--neurons-per-hidden-layer", type=int, default=8, help="number of neurons per hidden layer (default: 8)")
	parser.add_argument("--seed", type=int, default=1, help="random seed for shuffling dataset (default: 1)")
	parser.add_argument("--counterexamples", default=50, help="max number of counterexamples")
	parser.add_argument("--sampling", default=20, help="sample more points around the repaired counterexample")
	parser.add_argument("--retain", default=False, action='store_true', help="retain the current network (last CEGIS iteration)")
	parser.add_argument("--force", default=False, action='store_true', help="delete previous stat files and do the experiment again")

	args = parser.parse_args()

	PROPERTY = int(args.property)
	assert PROPERTY in {1,2}

	LEARNING_RATE = float(args.learning_rate)
	assert LEARNING_RATE in {1e-2,1e-3,1e-4}

	EPOCHS = int(args.epochs)
	assert EPOCHS >= 1

	EARLY_STOPPING = int(args.early_stopping)
	assert EARLY_STOPPING >= 1 and EARLY_STOPPING <= EPOCHS

	HIDDEN_LAYERS = int(args.hidden_layers)
	assert HIDDEN_LAYERS >= 1

	NEURONS_PER_HIDDEN_LAYER = int(args.neurons_per_hidden_layer)
	assert NEURONS_PER_HIDDEN_LAYER >= 1

	SEED = int(args.seed)
	FORCE = bool(args.force)

	COUNTEREXAMPLES = int(args.counterexamples)
	assert COUNTEREXAMPLES >= 1

	SAMPLING = int(args.sampling)
	assert SAMPLING >= 0
	
	RETAIN = args.retain
	
	# print('PROPERTY: {}'.format(PROPERTY))
	# print('LEARNING_RATE: {}'.format(LEARNING_RATE))
	# print('EPOCHS: {}'.format(EPOCHS))
	# print('EARLY_STOPPING: {}'.format(EARLY_STOPPING))
	# print('HIDDEN_LAYERS: {}'.format(HIDDEN_LAYERS))
	# print('NEURONS_PER_HIDDEN_LAYER: {}'.format(NEURONS_PER_HIDDEN_LAYER))
	# print('SEED: {}'.format(SEED))
	# print('FORCE: {}'.format(FORCE))
	# print('COUNTEREXAMPLES: {}'.format(COUNTEREXAMPLES))
	# print('SAMPLING: {}'.format(SAMPLING))
	# print('RETAIN: {}'.format(RETAIN))
	# exit()

	if not os.path.exists('experiments'):
		os.mkdir('experiments')

	EXPERIMENT = '{}-{}-{}x{}-{}-{}-{}-{}'.format(PROPERTY,LEARNING_RATE,HIDDEN_LAYERS,NEURONS_PER_HIDDEN_LAYER, COUNTEREXAMPLES, SAMPLING, SEED, 'R' if RETAIN else 'L')
	if os.path.exists('experiments/{}'.format(EXPERIMENT)):
		if not FORCE:
			exit('experiment exists, use --force to overwrite and do the experiment again')
		else:
			#print('overwriting experiment {}'.format(EXPERIMENT))
			shutil.rmtree('experiments/{}'.format(EXPERIMENT))
	os.mkdir('experiments/{}'.format(EXPERIMENT))
	
	START = time.time()

	Normalized = load_and_normalize('dataset.csv')
	random.seed(SEED)
	random.shuffle(Normalized)

	# partition
	Training   = Normalized[:8840]
	Validation = Normalized[8840:9945]
	Test = Normalized[9945:]

	N = len(Fieldnames)
	X_train = [i[0:N-8] for i in Training]
	y_train = [i[N-8:]  for i in Training]

	X_valid = torch.tensor([i[0:N-8] for i in Validation], dtype=torch.float32)
	y_valid = torch.tensor([i[N-8:]  for i in Validation], dtype=torch.float32)
	
	X_test  = torch.tensor([i[0:N-8] for i in Test], dtype=torch.float32)
	y_test  = torch.tensor([i[N-8:]  for i in Test], dtype=torch.float32)

	assert len(Normalized) == (len(Training) + len(Validation) + len(Test))
	
	TrainingTIME     		= []
	VerificationTIME 		= []
	RepairAndSamplingTIME 	= []
	LossOnValidation 		= []
	LossOnTest       		= []
	CounterExamples  		= []
	NewPairs         		= [] 

	OriginalInputs = len(X_train)

	loss_fn = MSELoss()
	lossT = None
	cegis_iteration = 0

	from milp import *
	from properties import *

	network = neural_network(HIDDEN_LAYERS, NEURONS_PER_HIDDEN_LAYER)
	while True:				
		cegis_iteration += 1	
		#print("iteration={}".format(cegis_iteration))
		start = time.time()

		if cegis_iteration > 1 and (not RETAIN):
			network = neural_network(HIDDEN_LAYERS, NEURONS_PER_HIDDEN_LAYER) # new one
		
		network, lossV = train(network, X_train, y_train, X_valid, y_valid, EPOCHS, EARLY_STOPPING, LEARNING_RATE)			
		TrainingTime = time.time() - start

		# compute loss on test
		with torch.no_grad():
			y_pred = network(X_test)
			lossT = loss_fn(y_pred, y_test).item()

		#print("Formal verification started")
		Inputs, Outputs, SolvingTime, RSTIME, CE, NewData = MILP(network, P=PROPERTY, C=COUNTEREXAMPLES, S=SAMPLING)
		
		# statistics
		TrainingTIME.append(TrainingTime)
		VerificationTIME.append(SolvingTime)
		RepairAndSamplingTIME.append(RSTIME)
		LossOnValidation.append(lossV)
		LossOnTest.append(lossT)
		CounterExamples.append(CE)
		NewPairs.append(NewData)

		X_train += Inputs
		y_train += Outputs
		
		with open('experiments/{}/training-time.json'.format(EXPERIMENT), 'w') as f:
			json.dump(TrainingTIME, f)

		with open('experiments/{}/formal-verification-time.json'.format(EXPERIMENT), 'w') as f:
			json.dump(VerificationTIME, f)
		
		with open('experiments/{}/repair-and-sampling-time.json'.format(EXPERIMENT), 'w') as f:
			json.dump(RepairAndSamplingTIME, f)
		
		with open('experiments/{}/loss-validation.json'.format(EXPERIMENT), 'w') as f:
			json.dump(LossOnValidation, f)
		
		with open('experiments/{}/loss-test.json'.format(EXPERIMENT), 'w') as f:
			json.dump(LossOnTest, f)
		
		with open('experiments/{}/counterexamples.json'.format(EXPERIMENT), 'w') as f:
			json.dump(CounterExamples, f)
		
		with open('experiments/{}/new-pairs.json'.format(EXPERIMENT), 'w') as f:
			json.dump(NewPairs, f)

		FinalDataset = [x+y for (x,y) in zip(X_train,y_train)]
		with open('experiments/{}/dataset.json'.format(EXPERIMENT), 'w') as f:
			json.dump(NewPairs, f)

		torch.save(network.state_dict(), 'experiments/{}/network-{}.pt'.format(EXPERIMENT,cegis_iteration))

		if len(Inputs) == 0:
			#print("No counterexamples found")
			#print("Total new inputs={}".format(len(X_train)-OriginalInputs))
			break

	END = time.time()
	total_time = END - START
	#print("Total time: {}".format(total_time))
	# statistics
	with open("experiments/{}/stats.txt".format(EXPERIMENT), "w") as f:
		# total time, added pairs, iterations of the CEGIS loop
		f.write("{} {} {}\n".format(total_time, len(X_train)-OriginalInputs, cegis_iteration))

	env.dispose()
	gurobipy.disposeDefaultEnv()