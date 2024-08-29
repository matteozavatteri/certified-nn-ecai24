import json
import matplotlib.pyplot as plt
import os
import argparse

METRICS = ['training-time', 'formal-verification-time', 'repair-and-sampling-time', 'loss-test', 'loss-validation', 'new-pairs', 'counterexamples']

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("key", nargs='+', help="list of keys")
	args = parser.parse_args()
	KEYS = args.key
	plt.rcParams['text.usetex'] = True
	DATA = dict()

	for key in KEYS:
		DATA[key] = dict()
		for metric in METRICS:
			DATA[key][metric] = None
			path = 'experiments/{}/{}.json'.format(key,metric)
			if os.path.exists(path):
				with open(path, "r") as f:
					DATA[key][metric] = json.load(f)
		
		with open('experiments/{}/stats.txt'.format(key)) as f:
			tt, tp, ti = f.readline().split()

		DATA[key]['total-time'] = int(float(tt))
		DATA[key]['total-pairs'] = int(tp)
		DATA[key]['total-iterations'] = int(ti)

	#print("DATA = {}".format(DATA)) 	     

	# time stats
	for key in KEYS:
		fig, ax = plt.subplots()
		ax.xaxis.set_ticks([i+1 for i in range(1,DATA[key]['total-iterations']+1)])
		for metric in ['training-time','formal-verification-time']:
			ax.set_xlabel('CEGIS iteration')
			ax.set_ylabel('time (s)')
			plt.yscale("log")
			if DATA[key][metric] != None:
				x = [i for i in range(1,len(DATA[key][metric])+1)]
				y = DATA[key][metric]
				ax.plot(x, y, label=metric)
	
		ax.legend(loc='best')
		plt.savefig('img/{}-T-FV.pdf'.format(key), format="pdf")
		plt.close('all')

	# loss
	for key in KEYS:
		fig, ax = plt.subplots()
		ax.xaxis.set_ticks([i+1 for i in range(1,DATA[key]['total-iterations']+1)])
		for metric in ['loss-test']:
			ax.set_xlabel('CEGIS iteration')
			ax.set_ylabel('loss value')
			#plt.yscale("log")
			if DATA[key][metric] != None:
				x = [i for i in range(1,len(DATA[key][metric])+1)]
				y = DATA[key][metric]
				ax.plot(x, y, label=metric)
	
		ax.legend(loc='best')
		plt.savefig('img/{}-LT.pdf'.format(key), format="pdf")
		plt.close('all')
	
	# counter examples
	import numpy as np

	for key in KEYS:
		labels = [i for i in range(1,len(DATA[key]['counterexamples'])+1)]
		data = {
		    'CounterExamples' : DATA[key]['counterexamples'],
		    'New Pairs' : DATA[key]['new-pairs']
		}

		x = np.arange(len(labels))  # the label locations
		width = 0.3  # the width of the bars
		multiplier = 0

		fig, ax = plt.subplots(layout='constrained')
		ax.xaxis.set_ticks([i+1 for i in range(1,DATA[key]['total-iterations']+1)])

		for attribute, measurement in data.items():
		    offset = width * multiplier
		    rects = ax.bar(x + offset, measurement, width, label=attribute)
		    ax.bar_label(rects, padding=3)
		    multiplier += 1

		# Add some text for labels, title and custom x-axis tick labels, etc.
		ax.set_ylabel('Number of pairs')
		ax.set_xticks(x + width, labels)
		ax.legend(loc='upper center', ncols=len(data))
		ax.set_xlabel('CEGIS iteration')
		m = max(DATA[key]['new-pairs'])
		#ax.set_ylim(0, m)

		plt.savefig('img/{}-CE-NP.pdf'.format(key), format="pdf")

