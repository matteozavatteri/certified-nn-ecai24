import json
import csv
from conf import *

EPSILON = 1e-4

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



N = len(Fieldnames)

Dataset = load_and_normalize('dataset.csv')

index_dtcc = NN_Inputs.index('distance_to_closest_child')
index_mop  = NN_Outputs.index('mopping_the_floor')

P1V = 0
P1Y = 0
P1N = 0
for p in Dataset:
	if p[index_dtcc] > 0.1:
		P1V += 1
	else:
		if p[index_mop] <= min(p[N-8:]): # non vacuos and true
			P1Y += 1
		else:
			P1N += 1 # non vacuos and false

print("P1: V={}, Y={}, N={}".format(P1V, P1Y, P1N))
P2V = 0
P2Y = 0
P2N = 0
index_dtch = NN_Inputs.index('distance_to_closest_human')
index_dtca = NN_Inputs.index('distance_to_closest_animal')

for i in range(len(Dataset)):
	sat = True
	v = True
	for j in range(len(Dataset)):
		if i != j:
			p1 = Dataset[i]
			p2 = Dataset[j]
			if p1[index_dtch] <= 0.1 and p2[index_dtca] <= 0.1:
				v = False
				if p1[index_mop] > p2[index_mop]: # non vacuos and true
					sat = False
					break

	if v:
		P2V += 1
	elif sat:
		P2Y += 1
	else:
		P2N +=1


print("P2: V={}, Y={}, N={}".format(P2V, P2Y, P2N))