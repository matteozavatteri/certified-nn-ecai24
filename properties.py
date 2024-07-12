from milp import *
import random
import time

EPSILON = 1e-4

# Property 1
def samplingP1(c, N):
	(cx,cy) = c
	Inputs  = []
	Outputs = []
	index_dtcc = NN_Inputs.index('distance_to_closest_child')
	index_mop  = NN_Outputs.index('mopping_the_floor')

	env = gurobipy.Env(empty=True)
	env.setParam('OutputFlag', 0)
	env.start()
	m = Model("repair",env=env)
	s = []
	for i in range(len(NN_Outputs)):
		s.append(m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s{}".format(i)))
	
	m.update()
	for i in range(len(NN_Outputs)):
		m.addConstr(cy[index_mop] + s[index_mop] <= cy[i] + s[i])
		
	m.setObjective(quicksum(s[i] * s[i] for i in range(len(NN_Outputs))), GRB.MINIMIZE)
	m.update()
	m.optimize()
	assert m.Status == GRB.OPTIMAL

	for i in range(len(NN_Outputs)):
		cy[i] = cy[i] + s[i].X				

	Inputs.append(cx)
	Outputs.append(cy)

	M = len(NN_Inputs) - len(BinaryIndexes)
	ContinuousIndexes = [i for i in range(len(NN_Inputs)) if i not in BinaryIndexes]

	for _ in range(N):
		x = torch.tensor(cx, dtype=torch.float32)
		noise = 0.1 * torch.randn(M)
		for i in ContinuousIndexes:
			x[i] += noise[0]
			noise = noise[1:]
		
		y = torch.tensor(cy, dtype=torch.float32)
		noise = 0.1 * torch.randn(len(NN_Outputs))
		y += noise
		if y[index_mop] <= min(y): # Fpost
			Inputs.append(x.tolist())
			Outputs.append(y.tolist())

	m.dispose()
	return Inputs, Outputs

def property1(network, C, S):
	milp = milp_encoding(network, K=1)
	milp.setParam('PoolSolutions', C)
	milp.setParam('PoolSearchMode', 1)

	# if distance_to_closest_child <= 0.1 then APP(mopping_floor) <= APP(other_actions)

	index_dtcc = NN_Inputs.index('distance_to_closest_child')
	index_mop  = NN_Outputs.index('mopping_the_floor')

	x = milp._x[0,index_dtcc]
	y = milp._y[0,index_mop]

	OtherActionVars = [milp._y[0,i] for i in range(len(NN_Outputs))]
	
	milp.addConstr(x <= 0.1) # close child
	
	min_app = milp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="min_app")
	milp.addConstr(min_app == min_(OtherActionVars), name="MinOtherActions")
	milp.addConstr(y >= min_app + EPSILON) # y > some other actions

	Inputs  = []
	Outputs = []

	start = time.time()
	milp.optimize()
	SolvingTime = time.time() - start
	
	solutions = milp.SolCount
	if milp.Status == GRB.INFEASIBLE:
		return Inputs, Outputs, SolvingTime, 0, solutions, len(Inputs)
	
	for i in range(solutions):
		milp.setParam('SolutionNumber', i)
		cx = [milp._x[0,i].xn for i in range(len(NN_Inputs))]  # counterexample
		cy = [milp._y[0,i].xn for i in range(len(NN_Outputs))] # counterexample

		for i in BinaryIndexes:
			cx[i] = round(cx[i])

		start = time.time()
		tmpI, tmpO = samplingP1((cx,cy), S)
		RepairAndSamplingTIME = time.time() - start
		Inputs += tmpI
		Outputs += tmpO

	milp.dispose()

	#print("Added {} zone(s) of {} points".format(milp.SolCount,len(Inputs)))
	return Inputs, Outputs, SolvingTime, RepairAndSamplingTIME, solutions, len(Inputs)


# Property 2
def samplingP2(cH, cA, N):
	(cxH,cyH) = cH
	(cxA,cyA) = cA

	Inputs  = []
	Outputs = []
	index_dtch = NN_Inputs.index('distance_to_closest_human')
	index_dtca = NN_Inputs.index('distance_to_closest_animal')
	index_mop  = NN_Outputs.index('mopping_the_floor')
	
	env = gurobipy.Env(empty=True)
	env.setParam('OutputFlag', 0)
	env.start()
	m = Model("repair",env=env)
	
	sH = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="sH")			
	sA = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="sA")
	
	m.update()
	m.addConstr(cyH[index_mop] + sH <= cyA[index_mop] + sA)
		
	m.setObjective(sH * sH + sA * sA, GRB.MINIMIZE)
	m.update()
	m.optimize()

	assert m.Status == GRB.OPTIMAL

	cyH[index_mop] += sH.X
	cyA[index_mop] += sA.X

	Inputs.append(cxH)
	Outputs.append(cyH)
	Inputs.append(cxA)
	Outputs.append(cyA)


	M = len(NN_Inputs) - len(BinaryIndexes)
	ContinuousIndexes = [i for i in range(len(NN_Inputs)) if i not in BinaryIndexes]
	for _ in range(N):
		xH = torch.tensor(cxH, dtype=torch.float32)
		noise = 0.1 * torch.randn(M)
		for i in ContinuousIndexes:
			xH[i] += noise[0]
			noise = noise[1:]

		yH = torch.tensor(cyH, dtype=torch.float32)
		noise = 0.1 * torch.randn(len(NN_Outputs))
		yH += noise

		xA = torch.tensor(cxA, dtype=torch.float32)
		noise = 0.1 * torch.randn(M)
		for i in ContinuousIndexes:
			xA[i] += noise[0]
			noise = noise[1:]

		yA = torch.tensor(cyA, dtype=torch.float32)
		noise = 0.1 * torch.randn(len(NN_Outputs))
		yA += noise
		
		if yH[index_mop] <= yA[index_mop]: # Fpost
			Inputs.append(xH.tolist())
			Outputs.append(yH.tolist())
			Inputs.append(xA.tolist())
			Outputs.append(yA.tolist())
	
	m.dispose()
	return Inputs, Outputs


def property2(network, C, S):
	milp = milp_encoding(network, K=2)
	milp.setParam('PoolSolutions', C)
	milp.setParam('PoolSearchMode', 1)
	
	# if distance_to_closest_human <= 0.1, then
	# APP(mopping_floor) <= APP(mopping_floor) when distance_to_closest_animal <= 0.1

	index_dtch = NN_Inputs.index('distance_to_closest_human')
	index_dtca = NN_Inputs.index('distance_to_closest_animal')
	index_mop  = NN_Outputs.index('mopping_the_floor')

	xH = milp._x[0,index_dtch]
	yH = milp._y[0,index_mop]

	xA = milp._x[1,index_dtca]
	yA = milp._y[1,index_mop]

	milp.addConstr(xH <= 0.1) # dist to close human
	milp.addConstr(xA <= 0.1) # dist to close animal
	milp.addConstr(yH >= yA + EPSILON) 
	
	start = time.time()
	milp.optimize()
	SolvingTime = time.time() - start

	Inputs  = []
	Outputs = []

	solutions = milp.SolCount
	if milp.Status == GRB.INFEASIBLE:
		return Inputs, Outputs, SolvingTime, 0, solutions, len(Inputs)
	
	for i in range(solutions):
		milp.setParam('SolutionNumber', i)
		cxH = [milp._x[0,i].X for i in range(len(NN_Inputs))]  # counterexample
		cyH = [milp._y[0,i].X for i in range(len(NN_Outputs))] # counterexample

		cxA = [milp._x[1,i].X for i in range(len(NN_Inputs))]  # counterexample
		cyA = [milp._y[1,i].X for i in range(len(NN_Outputs))] # counterexample
		
		for i in BinaryIndexes:
			cxH[i] = round(cxH[i])
			cxA[i] = round(cxA[i])

		start = time.time()
		tmpI, tmpO = samplingP2((cxH, cyH), (cxA, cyA), S)
		RepairAndSamplingTIME = time.time() - start
		Inputs += tmpI
		Outputs += tmpO

	milp.dispose()

	#print("Added {} zone(s) of {} points".format(milp.SolCount,len(Inputs)))
	return Inputs, Outputs, SolvingTime, RepairAndSamplingTIME, solutions, len(Inputs)	

def MILP(network, P, C, S):		
	f = None
	if P == 1:
		f = property1
	elif P == 2:
		f = property2
	else:
		assert False

	return f(network,C,S)