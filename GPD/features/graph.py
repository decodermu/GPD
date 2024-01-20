import numpy as np
from numpy.core.fromnumeric import size
import torch
import mdtraj as md
import time
import networkx as nx
import sys
import os

currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)


def quaternion(v1,v2):
    w = torch.dot(v1,v2)+torch.norm(v1)*torch.norm(v2)
    u = torch.cross(v1,v2)
    w = torch.unsqueeze(w,dim=-1)
    q = torch.cat([w,u],dim=-1)
    q = q /torch.norm(q)
    return q

def compute_rotation_movment(traj,top,length=400):
    distances = torch.zeros((length,length),dtype=float)
    quternions = torch.zeros((length,length,4),dtype=float)
    movement = torch.zeros((length,length,3),dtype=float)

    CAs_index = top.select("backbone and name CA")
    Ns_index = top.select("backbone and name N")
    Cs_index = top.select("backbone and name C")
    CAs_np = traj.xyz[0,CAs_index,]
    Ns_np = traj.xyz[0,Ns_index,]
    Cs_np = traj.xyz[0,Cs_index,]
    CAs = torch.from_numpy(CAs_np)
    Ns = torch.from_numpy(Ns_np)
    Cs = torch.from_numpy(Cs_np)

    CA_N = Ns - CAs
    CA_C = Cs - CAs

    dircts = torch.cross(CA_C,CA_N)

    real_length = len(dircts)
    for i in range(0,real_length):
        for j in range(0,real_length):
            if(i==j):
                quternions_tmp = torch.tensor([1.0,0.0,0.0,0.0])
                movement_tmp = torch.tensor([0.0,0.0,0.0])
                quternions[i,j] = quternions_tmp
            else:
                quternions_tmp = quaternion(dircts[i],dircts[j])
                movement_tmp = CAs[i] - CAs[j]
                distances[i,j] = torch.norm(movement_tmp)
                movement[i,j] = movement_tmp/distances[i,j]
                quternions[i,j] = quternions_tmp

    return distances,movement,quternions


def compute_shortestpath_centerilty(distances,length=400):
    shape = distances.shape
    weight = torch.where(distances!=0.0,1/distances,0.0)
    graph_matrix = torch.where(weight>0.8333333 ,1.0,0.0).numpy() #CA距离大于12A
    # graph = nx.from_numpy_matrix(graph_matrix)
    graph = nx.from_numpy_array(graph_matrix)
    '''import matplotlib.pyplot as plt
    subax1 = plt.subplot(121)
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()'''
    centerity = np.array(list(nx.centrality.betweenness_centrality(graph).values()))
    shortest_path_length = np.empty(shape=shape)
    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            try:
                shortest_path_length[i][j] = nx.shortest_path_length(graph,source=i,target=j)
            except:
                shortest_path_length[i][j] = 0.0
    return shortest_path_length,centerity

if __name__ == "__main__":
    start = time.time()

    dir_name = "./cath-dataset-nonredundant-S40-v4_3_0.pdb/"
    files = os.listdir(dir_name)
    i = 0
    count = 0
    errors = 0
    distance_value = np.load("distance_value.npy")
    distance_value = distance_value.astype("float64")
    distance_value = torch.from_numpy(distance_value)
    path_length = np.empty(shape=(30868,400,400))
    centerity = np.empty(shape=(30868,400))
    for distance in distance_value:
        try:
            path_length[i], centerity[i] = compute_shortestpath_centerilty(distances=distance,length=400)
            i = i+1
            count += 1
        except:
            print(i)
            i += 1
            errors += 1
    print(count)
    print(errors)
    np.save("path_length.npy",path_length)
    np.save("centerity.npy",centerity)
    end = time.time()

    print(end-start)
