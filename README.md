# Paper

This repository contains the supplementary material of the paper:

	Matteo Zavatteri, Davide Bresolin, Nicolò Navarin. 
	Automated Synthesis of Certified Neural Networks. 
	27TH EUROPEAN CONFERENCE ON ARTIFICIAL INTELLIGENCE (ECAI 2024).

# Dataset

This code works on the original dataset of the paper:
	
	Jonas Tjomsland, Sinan Kalkan and Hatice Gunes
	Mind Your Manners! A Dataset and A Continual Learning Approach for Assessing Social Appropriateness of Robot Actions
	Frontiers in Robotics and AI, Special Issue on Lifelong Learning and Long-term Human-Robot Interaction, 9:1-18, 2022

[Link to the paper](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.669420/full)

We do not provide the dataset here. However, to get all set up to reproduce the experiments:

- Download [https://github.com/jonastjoms/MANNERS-DB/blob/master/src/data/not_normallized.csv](https://github.com/jonastjoms/MANNERS-DB/blob/master/src/data/not_normallized.csv)
- Run `$ python3 preprocess.py`. This will create the file `dataset.csv`.

This step needs only to be done once.

# Dependencies

- gurobipy
- pytorch (thus numpy too)

To run all the experiments of the paper run:

	$ python3 RunExperiments.py

Recall to download the dataset and preprocess it.

# Usage

To run a single experiment run 

	$ python3 synthesizer.py -h 

to see the help screen (see also inside RunExperiments.py to have example of possible commands).

Once you are done, run

	$ python3 latex.py

to compute the table of numbers (output file table.tex).

Optionally, create the `img` folder and run 

	$ python3 stats.py 1-0.0001-10x10-50-20-100-L
	$ python3 stats.py 1-0.0001-6x9-50-20-100-L
	$ python3 stats.py 1-0.0001-7x8-50-20-100-L
	$ python3 stats.py 1-0.0001-8x7-50-20-100-L
	$ python3 stats.py 1-0.0001-9x6-50-20-100-L
	$ python3 stats.py 1-0.001-10x10-50-20-100-L
	$ python3 stats.py 1-0.001-6x9-50-20-100-L
	$ python3 stats.py 1-0.001-7x8-50-20-100-L
	$ python3 stats.py 1-0.001-8x7-50-20-100-L
	$ python3 stats.py 1-0.001-9x6-50-20-100-L
	$ python3 stats.py 1-0.01-10x10-50-20-100-L
	$ python3 stats.py 1-0.01-6x9-50-20-100-L
	$ python3 stats.py 1-0.01-7x8-50-20-100-L
	$ python3 stats.py 1-0.01-8x7-50-20-100-L
	$ python3 stats.py 1-0.01-9x6-50-20-100-L
	$ python3 stats.py 2-0.0001-10x10-50-20-100-L
	$ python3 stats.py 2-0.0001-6x9-50-20-100-L
	$ python3 stats.py 2-0.0001-7x8-50-20-100-L
	$ python3 stats.py 2-0.0001-8x7-50-20-100-L
	$ python3 stats.py 2-0.0001-9x6-50-20-100-L
	$ python3 stats.py 2-0.001-10x10-50-20-100-L
	$ python3 stats.py 2-0.001-6x9-50-20-100-L
	$ python3 stats.py 2-0.001-7x8-50-20-100-L
	$ python3 stats.py 2-0.001-8x7-50-20-100-L
	$ python3 stats.py 2-0.001-9x6-50-20-100-L
	$ python3 stats.py 2-0.01-10x10-50-20-100-L
	$ python3 stats.py 2-0.01-6x9-50-20-100-L
	$ python3 stats.py 2-0.01-7x8-50-20-100-L
	$ python3 stats.py 2-0.01-8x7-50-20-100-L
	$ python3 stats.py 2-0.01-9x6-50-20-100-L
	
to compute graphics (need matplotlib).
	
	
