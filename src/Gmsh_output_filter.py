"""This module work as a filter providing compatitble ConFEM's inputdata from the .msh file resulted from Gmsh4.0"""

from __future__ import print_function
f1 = open('C:/Users/regga/Desktop/testcode/slab.msh', 'r')   # edit the path to your .msh file
list_f1 = list(f1)
# Preallocate memories
PhysicalEntity = []
MeshFormat = {}
MeshFormat_key = ""
MeshFormat_value = []   #starting and ending line corresponding to MeshFormat_key

# Loop over all lines in file to define MeshFormat dictionnary
for index, line in enumerate(list_f1):
    #MeshFormat
    if line.strip() == "$MeshFormat":
        MeshFormat_key = "MeshFormat"
        MeshFormat[MeshFormat_key] = [index]
    elif line.strip() == "$EndMeshFormat":
        MeshFormat[MeshFormat_key] +=[index]
    #PhysicalNames
    if line.strip() == "$PhysicalNames":
        MeshFormat_key = "PhysicalNames"
        MeshFormat[MeshFormat_key] = [index]
    elif line.strip() == "$EndPhysicalNames":
        MeshFormat[MeshFormat_key] +=[index]
    #Entities
    if line.strip() == "$Entities":
        MeshFormat_key = "Entities"
        MeshFormat[MeshFormat_key] = [index]
    elif line.strip() == "$EndEntities":
        MeshFormat[MeshFormat_key] +=[index]
    #Nodes
    if line.strip() == "$Nodes":
        MeshFormat_key = "Nodes"
        MeshFormat[MeshFormat_key] = [index]
    elif line.strip() == "$EndNodes":
        MeshFormat[MeshFormat_key] +=[index]
    #Elements
    if line.strip() == "$Elements":
        MeshFormat_key = "Elements"
        MeshFormat[MeshFormat_key] = [index]
    elif line.strip() == "$EndElements":
        MeshFormat[MeshFormat_key] +=[index]
del MeshFormat_key, MeshFormat_value   # not need anymore
print(MeshFormat)
# Loop in PhysicalNames
PhysicalNames_List = []
class PhysicalEntity(object):
    def __init__(self, dim, tag, name):
        self.dim = dim      #Dimension : 0=point, 1=curve, 2=surface, 3=volume
        self.tag = tag
        self.name = name
for i in xrange(MeshFormat["PhysicalNames"][0]+2, MeshFormat["PhysicalNames"][1]):
    line = list_f1[i].split(" ")
    PhysicalNames_List += [PhysicalEntity(int(line[0]), int(line[1]), line[2].strip())]

# Loop in Entities
Entities_List = []
count_idx = list_f1[ MeshFormat["Entities"][0]+1 ].strip().split(" ")  # numbers of Points, Curves, Surfaces, Volumes
count_idx = map(int,count_idx)
count_start = MeshFormat["Entities"][0]+2
class Entity_point(object):
    def __init__(self, tag, coord, numPhysicals, *args):    # *args = physicalTag
        self.dim = 0    # point
        self.tag = tag
        self.coord = coord  # tuple of coordinates boxMinX,Y,Z to boxMaxX,Y,Z
        self.numPhysicals = numPhysicals
        if self.numPhysicals == 0: self.physicalTag = (("None"),)   # if not defined
        else: self.physicalTag = args  # if have
for i in xrange(count_start,count_start + count_idx[0]):
    line = list_f1[i].strip().split(" ")
    line = map(float,line)
    coordinates = tuple(line[j] for j in xrange(1,7))
    lengL = len(line)
    if lengL ==8: Entities_List += [ Entity_point(int(line[0]), coordinates, int(line[7])) ]
    else:             Entities_List += [ Entity_point(int(line[0]), coordinates, int(line[7]), tuple(j for j in map(int,line[8:])) )]

count_start = count_start+count_idx[0]
class Entity_curve(object):
    def __init__(self, tag, coord, numPhysicals, numBoundingPoints, tagPoint, *args ):  # *args = physicalTag
        self.dim = 1    # curve
        self.tag = tag
        self.coord = coord
        self.numPhysicals = numPhysicals
        if self.numPhysicals == 0: self.physicalTag = (("None"),)   # if not defined
        else: self.physicalTag = args  # if have, tuple of integers
        self.numBoundingPoints = numBoundingPoints
        self.tagPoint = tagPoint    # tuple of integers
for i in xrange(count_start, count_start + count_idx[1]):
    line = list_f1[i].strip().split(" ")
    line = map(float,line)
    coordinates = tuple(line[j] for j in xrange(1,7))
    lengL = len(line)
    if line[7] == 0 :  # that means numPhysicals = 0
        Entities_List += [Entity_curve(int(line[0]), coordinates, int(line[7]), int(line[8]), tuple(j for j in map(int,line[9:])) )]
    else:  # that means numPhysicals >= 1
        Entities_List += [Entity_curve(int(line[0]), coordinates, int(line[7]), int(line[7+int(line[7])+1]), tuple(j for j in map(int,line[7+int(line[7])+2:])), tuple(k for k in map(int,line[(7+1):(7+int(line[7])+1)])) )]

count_start = count_start + count_idx[1]
class Entity_surface(object):
    def __init__(self, tag, coord, numPhysicals, numBoundingCurves, tagCurve, *args ):  # *args = physicalTag
        self.dim = 2    # curve
        self.tag = tag
        self.coord = coord
        self.numPhysicals = numPhysicals
        if self.numPhysicals == 0: self.physicalTag = (("None"),)   # if not defined
        else: self.physicalTag = args  # if have, tuple of integers
        self.numBoundingCurves = numBoundingCurves
        self.tagCurve = tagCurve    # tuple of integers
for i in xrange(count_start, count_start + count_idx[2]):
    line = list_f1[i].strip().split(" ")
    line = map(float,line)
    coordinates = tuple(line[j] for j in xrange(1,7))
    lengL = len(line)
    if line[7] == 0 :  # that means numPhysicals = 0
        Entities_List += [Entity_surface(int(line[0]), coordinates, int(line[7]), int(line[8]), tuple(j for j in map(int,line[9:])) )]
    else:  # that means numPhysicals >= 1
        Entities_List += [Entity_surface(int(line[0]), coordinates, int(line[7]), int(line[7+int(line[7])+1]), tuple(j for j in map(int,line[7+int(line[7])+2:])), tuple(k for k in map(int,line[(7+1):(7+int(line[7])+1)])) )]

# Loop in Nodes
NodeList = []
count_start = MeshFormat["Nodes"][0]+2
count_end = MeshFormat["Nodes"][1]
class Node(object):
    def __init__(self,tagEntity, dimEntity, tag, XYZ):
        self.tagEntity = int(tagEntity)
        self.dimEntity = int(dimEntity)
        self.tag = int(tag)
        self.XYZ = XYZ
NodeEntities = []     # this list specifies the number of lines that contain NodeEntities (not Node coord)
count = count_start
while count < count_end:
    line = map(int,list_f1[count].strip().split(" "))
    numNode = line[len(line) - 1]
    count_ = count + numNode +1
    NodeEntities += [ [count, count_, line[0], line[1]] ]   # relative line in file, end line, tagEntity, dimEntity
    count = count_
#for i in NodeEntities: print i
for counter in NodeEntities:
    for j in xrange(counter[0]+1,counter[1]):
        line = map(float,list_f1[j].strip().split(" "))
        XYZ = line[1:]
        NodeList += [Node(counter[2], counter[3], int(line[0]), XYZ)]

# Loop in Elements
ElList = []
count_start = MeshFormat["Elements"][0]+2
count_end = MeshFormat["Elements"][1]
class Element(object):
    def __init__(self, tagEntity, dimEntity, typeEle, tag, InzList):
        self.tagEntity = int(tagEntity)
        self.dimEntity = int(dimEntity)
        self.typeEle = int(typeEle)
        self.tag = int(tag)
        self.InzList = InzList
ElemEntities = []     # this list specifies the number of lines that contain ElemEntities (not Node coord)
count = count_start
while count < count_end:
    line = map(int,list_f1[count].strip().split(" "))
    numElem = line[len(line) - 1]
    count_ = count + numElem +1
    ElemEntities += [ [count, count_, line[0], line[1], line[2]] ]   # relative line in file, end line, tagEntity, dimEntity, typeElem
    count = count_
for counter in ElemEntities:
    for j in xrange(counter[0]+1,counter[1]):
        line = map(int,list_f1[j].strip().split(" "))
        InzList = line[1:]
        ElList += [Element(counter[2], counter[3], counter[4], line[0], InzList)]

f1.close()
# Sort Elements by PhysicalNames
del XYZ, count, count_, count_end, count_start, count_idx, counter, i,j, lengL, line, numElem, numNode, InzList, coordinates,  index
ElemOut = {}
for physicalname in PhysicalNames_List:
    ID, ElOut, EntiTag = [], [], []  # preallocate blank list
    Name = physicalname.name.strip()
    ElemOut[Name] = [ID, ElOut]  # preallocate blank lists
    ID += [physicalname.tag, physicalname.dim, EntiTag]
    #build ID list which contains the entities matching with physicalName.tag+dim
    for Enti in Entities_List:
        if Enti.numPhysicals <> 0:
            leng_ = len(Enti.physicalTag[0])
            for i in xrange(leng_):  # Enti.physicalTag maybe have more than one value
                if Enti.physicalTag[0][i] == physicalname.tag and Enti.dim == physicalname.dim:
                    EntiTag += [Enti.tag]
    #build ElOut list which contains elements satisfing the conditions in ID list
    for Elem in ElList:
        if Elem.dimEntity == ID[1]:
            for entiTag_ in ID[2]:
                if Elem.tagEntity == entiTag_: ElOut += [Elem]
    ElemOut[Name] = [ID, ElOut]

# Print to screen
for key_ in ElemOut.keys():
    print("\nPhysicalName:%s" %(key_))
    for Elem in ElemOut[key_][1]:
        print("ElemTag:%s TypeEle:%s InzList:%s" %(Elem.tag, Elem.typeEle,Elem.InzList))
del ElOut,Enti,EntiTag,ID,Name,entiTag_,i,key_,leng_,physicalname   # delete variables not-used anymore

# Write to text file

with open("C:/Users/regga/Desktop/slab_out.txt", "w") as outF:
    outF.write("*HEADING\n")
    outF.write("nonlinear slab analysis\n")
    # write NODE section
    outF.write("*NODE\n")
    for NODE in NodeList:
        line = "%s, %s, %s, %s\n" %(NODE.tag,NODE.XYZ[0],NODE.XYZ[1],NODE.XYZ[2])
        outF.write(line)
    outF.write("\n******************************************************************\n") #end section
    # write Elem section
with open("C:/Users/regga/Desktop/slab_out.txt", "a") as outF:
    for key_ in ElemOut.keys():
        typeElem_ = ElemOut[key_][1][0].typeEle
        outF.write("*ELEMENT, TYPE=%s, ELSET=%s\n" %(typeElem_,key_))
        for Elem in ElemOut[key_][1]:
            line_p1 = "%s, " %(Elem.tag)
            outF.write(line_p1)
            print(*Elem.InzList, sep=', ', end='\n', file=outF)
        outF.write("\n******************************************************************\n")

# Sort nodes in elements to a boundary list
def BoundaryGenerator(ElList, prescribedDof, value, file=None):
    """
    :param ElList: format : [list of Elem objects that have attributes: Elem.tag, Elem.typeEle, Elem.InzList]
    :param prescribedDof:
    :param value:
    :return: print out list of nodes, corresponding prescribed Dof, corresponding value
    """
    NodeList = []
    for elem in ElList:
        for node in elem.InzList:
            if not node in NodeList:
                NodeList += [node]
                for i in xrange(len(prescribedDof)):
                    if file == None:
                        print('%4d, %s, %s, %s' % (node, prescribedDof[i],prescribedDof[i], value))  # print out to console
                    else:
                        print('%4d, %s, %s, %s' % (node, prescribedDof[i],prescribedDof[i], value), file=file)  # print out to output file


""" To add boundary condition corresponding to PhysicalName, manually add/edit the script below
"""
if not __name__=="__main__":
    outputfile = open("C:/Users/regga/Desktop/slab_out.txt", "a")
    #outputfile = None

    n = 1
    targDict = ElemOut[ElemOut.keys()[n]]
    if outputfile ==None: print('\nBoundaryList of PhysicalName=%s' % ElemOut.keys()[n])
    else: print('\nBoundaryList of PhysicalName=%s' % ElemOut.keys()[n], file=outputfile)
    BoundaryGenerator(ElList=targDict[1], prescribedDof=(3,4), value=0.0, file=outputfile)

    n = 3
    targDict = ElemOut[ElemOut.keys()[n]]
    if outputfile ==None: print('\nBoundaryList of PhysicalName=%s' % ElemOut.keys()[n])
    else: print('\nBoundaryList of PhysicalName=%s' % ElemOut.keys()[n], file=outputfile)
    BoundaryGenerator(ElList=targDict[1], prescribedDof=(3,5), value=0.0, file=outputfile)

