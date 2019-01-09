import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

def Plot2D_elemout(inputfile, Time_expected, ElSet, demand_dict, IsoVaFlag):
    """
    :param inputfile: *.nodeout.txt
    :param Time_expected : desired time of plotting
    :param ElSet : type of elements expected to post process
    :param demand_dict: dictionary of desired plot variables. { key:[column number in file] }
    :param IsoVaFlag : Flag whether isovalue lines are ploted
    :return: contour of desired plot variables corresponding to x,y coordinates
    """
    data = {}  # data of values reffered to demand_dict
    #for key in demand_dict.keys():
    #    data[key] = []      # pre-allocate type of object
    # Retrieve data from file
    with open(inputfile,'rb') as f1:
        z1 = f1.readline()
        z2 = z1.split()
        while z1 <> '':
            if z2[0] == 'Time':
                Time = float(z2[1])    # then compare with Time_expected
                label, x, y = [] ,[], []    # nodal coordinates x,y
                data[Time] = {}
                for key in demand_dict.keys(): data[Time][key] = [] # pre-allocate type of object
                elemFlag = False # reset elemFlag at each new Time
            elif z2[0] == 'El': # assign to eFlag
                if z2[3] == ElSet: elemFlag = True
                else : elemFlag = False
            elif elemFlag == True:   # data found
                z2 = map(float, z2)  # convert to floating value
                z2[0] = int(z2[0])  # except for node label
                #label += [z2[0]]
                x += [z2[0]]
                y += [z2[1]]
                for key in demand_dict.keys():
                    col = demand_dict[key]
                    value = z2[col]
                    data[Time][key].append(value)
             #next line
            z1 = f1.readline()
            z2 = z1.split()

        print "Node label=%s" %label
        print "x= %s" % x
        print "y= %s" % y
        print len(x), len(y)
        for key in data.keys():
            print "Time=%s" % key
            for key_ in demand_dict:
                print "data[%s][%s]= %s" % (key, key_, data[key][key_])
                print len(data[key][key_])

    # convert list to numpy 1D array
    X,Y = np.array(x), np.array(y)
    # plot contour maps
    nfig = len(demand_dict.keys())
    fig, ax = plt.subplots(nrows=nfig)
    for time_ in data.keys():
        if time_ == Time_expected:  # plot only at expected Time
            for i in xrange(nfig):
                data_ = data[Time_expected]
                z_ = data_[data_.keys()[i]]
                Z = np.array(z_)
                cntr = ax[i].tricontourf(X, Y, Z, cmap='RdBu_r')  # of contour
                #cntr = ax[i].tricontourf(X, Y, Z, colors=('r','g','b'))  # other setting for colormap
                #cntr.cmap.set_under('yellow')
                #cntr.cmap.set_over('cyan')
                fig.colorbar(cntr, ax=ax[i])
                ax[i].plot(X, Y, 'ko', ms=2)
                ax[i].set_title('Contour of ' + data[Time_expected].keys()[i] + " at Time=%s" % Time_expected)
                if IsoVaFlag == True:
                    cml = ax[i].tricontour(X, Y, Z, levels=cntr.levels[::2], lw=0.5, colors='k')  # color map lines
                    plt.clabel(cml, colors='k', fontsize=8)
            plt.subplots_adjust(hspace=0.5)
            plt.show()

if __name__=="__main__":
    ipfile = "C:/Users/regga/Desktop/testcode/E7-04.elemout.txt"
    Time_expected = 1.0220
    demand = {'Moment Mx': 5, 'Moment My': 6}
    Plot2D_elemout(ipfile, Time_expected, 'PROP1', demand, False)
