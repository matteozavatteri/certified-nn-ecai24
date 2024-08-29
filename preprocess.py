import csv
import os

# Remove header
with open('not_normallized.csv', 'r') as csvfile:
	with open('tmp.csv', 'w') as out:
		i = 1
		for line in csvfile:
			if i != 1:
				out.write(line)
			i += 1


# Dataset preprocessing
with open('tmp.csv', 'r') as csvfile:
	with open('dataset.csv', 'w') as out:
		for line in csvfile:
			line = line.strip().split(",")
			del line[2]
			del line[0]
			out.write('{}\n'.format(','.join(line)))

os.remove('tmp.csv')