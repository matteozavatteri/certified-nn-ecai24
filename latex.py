import os 
import json
if __name__ == '__main__':
	METRICS = ['training-time', 'formal-verification-time', 'repair-and-sampling-time', 'loss-test', 'loss-validation', 'new-pairs', 'counterexamples']
	Networks = ['6x9','7x8','8x7','9x6','10x10']
	Rates = [0.01,0.001,0.0001]
	DATA = dict()

	for P in [1,2]:
		for N in Networks:
			for R in Rates:
				key = '{}-{}-{}-50-20-100-L'.format(P,R,N)
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


	# latex table
	with open('table.tex', 'w') as f:
		f.write('\\scalebox{0.7}{\\begin{tabular}{|*{11}{c|}}\n')
		f.write('\\hline\n')
		f.write('{{\\bf Property}} & {{\\bf Network}} & {{\\bf Learning Rate}} & {{\\bf CEGIS Iterations}} & {{\\bf Total Time (s)}} & {{\\bf Training Time (s)}} & {{\\bf Verification Time (s)}} & {{\\bf Repair + Sampling Time (s)}} & {{\\bf Initial Loss (test)}} & {{\\bf Final Loss (test)}} & {{\\bf New Pairs}} \\\\\n')
		f.write('\\hline\n')
		
		for P in [1,2]:	
			firstP = True
			f.write("\\multirow{{{}}}{{*}}{{$P_{}$}} ".format(len(Networks)*len(Rates),P))
			for N in Networks:	
				firstN = True
				f.write(" & \\multirow{{{}}}{{*}}{{{}}} ".format(len(Rates),N))
				for R in Rates:	
					if not firstN:
						f.write(" & ")	
					firstN = False
					f.write(" & {} ".format(R))
					key = '{}-{}-{}-50-20-100-L'.format(P,R,N)
					
					text = DATA[key]['total-iterations'] 
					f.write(" & {}".format(text)) # iterations
					
					text = DATA[key]['total-time'] 
					f.write(" & {}".format(text))
					
					text = int(sum(DATA[key]['training-time']))
					f.write(" & {}".format(text))
					
					text = round(sum(DATA[key]['formal-verification-time']),3)
					f.write(" & {}".format(text))
					
					text = round(sum(DATA[key]['repair-and-sampling-time']),3)
					f.write(" & {}".format(text))
					
					text = round(DATA[key]['loss-test'][0],3)
					f.write(" & {}".format(text))

					text = round(DATA[key]['loss-test'][-1],3)
					f.write(" & {}".format(text))
					
					text = DATA[key]['total-pairs'] 
					f.write(" & {} \\\\\n".format(text))
					f.write('\\cline{3-11}\n')
				f.write('\\cline{2-11}\n')
			f.write('\\hline\n')
		f.write('\\end{tabular}}\n')