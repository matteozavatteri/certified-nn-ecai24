import subprocess

# in seconds
TIMEOUT=3600*24 

P = [1,2]
E = 100 # epochs
S = 20  # early stopping


# layers x neurons
H = [(6,9),(7,8),(8,7),(9,6),(10,10)]

# learning rate
LR = [0.01,0.001,0.0001]

# seeds
SEEDS = [100]

# strategy (counterexamples, sampling)
STRATEGY = [(50,20)]

# command
c = 'python3 synthesizer.py {property} --learning-rate {lr} --epochs {epochs} --early-stopping {earlystopping} --hidden-layers {layers} --neurons-per-hidden-layer {neurons} --counterexamples {counterexamples} --sampling {sampling} --seed {seed}'

cmds = [c.format(property=p, lr=lr, epochs=E, earlystopping=S, layers=layers, neurons=neurons, counterexamples=ce, sampling=sam, seed=seed) for p in P for lr in LR for (layers,neurons) in H for seed in SEEDS for (ce,sam) in STRATEGY]

for cmd in cmds:
    #print("Running: {}".format(cmd))
    try:
        p = subprocess.run(cmd.split(), stdout=subprocess.PIPE, timeout=TIMEOUT)
    except subprocess.CalledProcessError as e:
        print("{}".format(e))
        print("going ahead")
    except:
        print('timeout\n')

