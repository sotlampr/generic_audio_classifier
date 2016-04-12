""" Stand-alone application to demonstrate yaafe's transformations """
import numpy as np
import matplotlib.pyplot as plt
from yaafelib import FeaturePlan, Engine, AudioFileProcessor


FPLAN = FeaturePlan(sample_rate=44100)
FPLAN.addFeature("mfcc: MFCC blockSize=512 stepSize=128")
FPLAN.addFeature("mfcc_stat: MFCC blockSize=512 stepSize=128 > "
                 "StatisticalIntegrator NbFrames=40 StepNbFrames=8")
ENGINE = Engine()
ENGINE.load(FPLAN.getDataFlow())
PROCESSOR = AudioFileProcessor()

PROCESSOR.processFile(ENGINE, 'track.wav')
DATA = ENGINE.readAllOutputs()
ENGINE.flush()

X_MFCC = DATA['mfcc']
X_MFCC_STAT = DATA['mfcc_stat']

MFCC_DESC = list()
for i in range(1, 14):
    desc = "Average for Band {}".format(i)
    MFCC_DESC.append(desc)
for i in range(1, 14):
    desc = "Standard Dev. for Band {}".format(i)
    MFCC_DESC.append(desc)

plt.ion()
FIG = plt.figure()
FIG.set_size_inches(14, 8)
TICK = np.arange(X_MFCC.shape[1])
AXIS = FIG.add_subplot(111)
AXIS.contour(np.swapaxes(X_MFCC, 0, 1))
plt.yticks(TICK, MFCC_DESC[:15])
AXIS.set_xlabel("Time Blocks")
# plt.tight_layout()
plt.draw()
