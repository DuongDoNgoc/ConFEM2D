"""
A code snipet for plot a figure from given data read from txt.file
This code show also how to decorate the output figure, i.e., add text to figure, configure axes text...
"""
import matplotlib.pyplot as plt
import numpy as np

Colors = ['red', 'magenta', 'green', 'yellow', 'blue', 'cyan']
FonSizTi = 'x-large'  # fontsize title x-large
FonSizAx = 'x-medium'  # fontsize axis
# FonSizAx='medium'     # fontsize axis
LiWiCrv = 3  # line width curve in pt
LiWiCrvMid = 2  # line width curve in pt
LiWiCrvThin = 0.5  # line width curve in pt

f5_name = 'C:/Users/regga/Desktop/testcode/E6-03.timeout.txt'
WrNodes = [[210, 2, 421, 'b-u']]
label = 'Reaction force-displacement of node=%s dof=%s' %(WrNodes[0][0], WrNodes[0][1])
sc1, sc2 = -1., -1.
L1, L2 = [0.], [0.]
with open(f5_name, 'rb') as file:
    z1 = file.readline()
    while z1 <> '': # retrieve data from file
        z2 = z1.split()
        if len(z2) > 0:
            L1 += [sc1 * float(z2[0])]
            L2 += [sc2 * float(z2[1])]
        else: continue
        z1 = file.readline()
#Setting up figure
fig, axe = plt.subplots()
axe.plot(L1, L2, c='red', lw= LiWiCrv, ls='-')
axe.set_title(label, fontsize='x-large')
plt.ylabel('reacion force (kN)', fontsize='large')
plt.xlabel('displacement (m)', fontsize='large')
#plt.legend(loc='lower right')
axe.annotate('max load = 15.8 kN', xy=(0.04,15.8), xytext=(0.035,12),
             arrowprops=dict(facecolor='black', shrink=0.01))
axe.text(0.0013, 9.2, r'crack formation state IIa', rotation=35, fontsize='small')

plt.grid()
plt.show()

