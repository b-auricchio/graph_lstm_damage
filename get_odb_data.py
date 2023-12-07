import numpy as np
from odbAccess import *
from textRepr import *
import argparse
import os

# parser = argparse.ArgumentParser(
#                     prog='.odb data extractor',
#                     description='Extracts element data from .odb file and exports to csv for multiple frames.'
#                 )

# parser.add_argument('root')
# parser.add_argument('-w', dest='wrinkle_amp', help='wrinkle amp of .odb') 

# args = parser.parse_args()
# odb=openOdb(args.root + '.odb')

## INPUTS
filename = 'wrinkle_amp_0_2/singleQIRVEnew.odb'
wrinkle_angle = 0.2

################

odb=openOdb(filename)
assembly = odb.rootAssembly
numNodes = 0
nel = 0
         
stepdata = odb.steps

stepname = []
for s1 in stepdata.keys():
    stepname.append(s1)



statenames = ['SDV9','SDV8','SDV7','SDV4']
numframes = len(odb.steps[stepname[0]].frames)
for i in range(numframes-8, numframes):
    elements = []

    frame = odb.steps[stepname[0]].frames[i]
    statedict = dict()
    
    for fieldname in frame.fieldOutputs.keys():
        print(fieldname)

    for var in statenames:
    # Reading state variables
        field = frame.fieldOutputs[var].getSubset(position=CENTROID)
        statedict[var] = field.values

    nel = len(statedict[statenames[0]])
    print("Num elements:", nel)

    filename = 'data_{angle}_T{t}.csv'.format(angle=wrinkle_angle, t=i)
    folder = 'wrinkle_{angle}'.format(angle=wrinkle_angle)
    isExist = os.path.exists(folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(folder)
    f = open(os.path.join(folder, filename),"w")
    f.write('element,type,sdv9,sdv8,sdv7,sdv4\n')
    for i in range(nel):
        f.write('{element},{type},{data1},{data2},{data3},{data4}\n'.format(element=statedict[statenames[0]][i].elementLabel,
                                                                            type=statedict[statenames[0]][i].baseElementType,
                                                                    data1 = statedict[statenames[0]][i].data,
                                                                    data2 = statedict[statenames[1]][i].data,
                                                                    data3 = statedict[statenames[2]][i].data,
                                                                    data4 = statedict[statenames[3]][i].data))
    f.close()

# Get coordinates of centroids
# n = 0
# xyz = np.zeros((numNodes*2,3),dtype=float)
# conn = np.zeros(8,dtype = int)
# cent = np.zeros(3,dtype=float)
#
# for name, instance in assembly.instances.items():
#     ne = len(instance.elements)
#     for node in instance.nodes:
#         n1 = node.label
#         xyz[n1,0] = node.coordinates[0]
#         xyz[n1,1] = node.coordinates[1]
#         xyz[n1,2] = node.coordinates[2]
#     for element in instance.elements:
#         cent = np.zeros(3,dtype=float)
#         for j in range(8):
#             cent[0] =cent[0] + 0.125*xyz[conn[j],0]
#             cent[1] =cent[1] + 0.125*xyz[conn[j],1]
#             cent[2] =cent[2] + 0.125*xyz[conn[j],2]
#             numNodes = max(numNodes,conn[j])
#         statedict['centroid'] = cent

