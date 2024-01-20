import numpy as np
import torch
import mdtraj as md
import math
import time
import random
import sys
import os

currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)

def DSSP_trans(DSSP_tmp,length):
    DSSP_dir = {
        'H':1.0,
        'G':2.0,
        'I':3.0,
        'E':4.0,
        'B':5.0,
        'T':6.0,
        'S':7.0,
        ' ':0.0,
        'NA':0.0
    }
    DSSP = np.zeros(shape=length,dtype=np.float16)
    length_min = min(length,len(DSSP_tmp))
    for i in range(0,length_min):
        DSSP[i] = DSSP_dir[DSSP_tmp[i]]
    return DSSP


def compute_phipsi_DSSP(top,t,length=400):
    atom = top.select("backbone and name CA")
    real_length = len(atom)

    #生成phipsi
    phipsi = np.zeros((length,2),dtype=np.float16)
    phi_tmp = md.compute_phi(t)
    psi_tmp = md.compute_psi(t)
    phipsi[0,1] = psi_tmp[1][0][0]
    for i in range(1,length):
        try:
            phipsi[i,0] = phi_tmp[1][0][i-1]
            phipsi[i,1] = psi_tmp[1][0][i]
        except:
            pass
    phipsi[length-1,1] = 0.0

    #计算DSSP
    DSSP_tmp = md.compute_dssp(t,simplified=False)
    DSSP = DSSP_trans(DSSP_tmp=DSSP_tmp[0],length=length)

    #生成mask(false不mask，true要mask)
    mask = np.zeros(shape=length,dtype=bool)
    for i in range(real_length,length):
        mask[i] = True

    # return phipsi,DSSP, mask
    return phipsi, DSSP, mask

seq_dir = {
    "ALA":1.0,
    "ARG":2.0,
    "ASN":3.0,
    "ASP":4.0,
    "CYS":5.0,
    "GLN":6.0,
    "GLU":7.0,
    "GLY":8.0,
    "HIS":9.0,
    "ILE":10.0,
    "LEU":11.0,
    "LYS":12.0,
    "MET":13.0,
    "PHE":14.0,
    "PRO":15.0,
    "SER":16.0,
    "THR":17.0,
    "TRP":18.0,
    "TYR":19.0,
    "VAL":20.0,
    "TPO":17.0,
    "SEP":16.0,
    "PTR":19.0,
    "0":0.0
}

seq_re_dir = {
    1:'A',
    2:'R',
    3:'N',
    4:'D',
    5:'C',
    6:'Q',
    7:'E',
    8:'G',
    9:'H',
    10:'I',
    11:'L',
    12:'K',
    13:'M',
    14:'F',
    15:'P',
    16:'S',
    17:'T',
    18:'W',
    19:'Y',
    20:'V'
    }

def get_seq(top,length=400):
    seq = np.zeros(shape=length,dtype=np.float16)
    i = 0
    for chain in top.chains:
        for residue in chain.residues:
            if(i==length):
                break
            if(residue.name in seq_dir.keys()):
                seq[i] = seq_dir[residue.name]
                i = i+1
            else:
                seq[i] = 21.0
                i = i+1
    while(i<length):
        seq[i] = 0.0
        i = i+1
    #print(label)
    return seq

def mask_seq(seq,length=400,mask_rate=1.0):
    seq_mask = np.zeros(shape=length,dtype=np.float16)
    for i in range(0,length):
        if(random.random()<mask_rate):
            seq_mask[i] = 0.0
        else:
            seq_mask[i] = seq[i]
    return seq_mask


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
    gas_dirct = torch.cross(CA_C,CA_N)
    Orientation = torch.cat([CA_C.unsqueeze(-1),CA_N.unsqueeze(-1),gas_dirct.unsqueeze(-1)],dim=-1)

    real_length = len(CA_N)
    for i in range(0,real_length):
        for j in range(0,real_length):
            if(i==j):
                quternions_tmp = torch.tensor([1.0,0.0,0.0,0.0])
                movement_tmp = torch.tensor([0.0,0.0,0.0])
                quternions[i,j] = quternions_tmp
            else:
                quternions_tmp1 = quaternion(CA_N[i],CA_N[j])
                quternions_tmp2 = quaternion(CA_C[i],CA_C[j])
                movement_tmp = CAs[i] - CAs[j]
                distances_tmp = torch.norm(movement_tmp)
                distances[i,j] = distances_tmp
                movement_tmp = movement_tmp/distances_tmp
                movement[i,j] = torch.mm(movement_tmp.unsqueeze(0),Orientation[i]).squeeze()
                quternions[i,j] = (quternions_tmp1 + quternions_tmp2) / 2

    return distances,movement,quternions

if __name__ == "__main__":
    start = time.time()

    dir_name = "./cath-dataset-nonredundant-S40-v4_3_0.pdb/"
    files = os.listdir(dir_name)
    i = 0
    count = 0
    errors = 0
    distance_value = np.empty(shape=(30868,400,400),dtype=np.float16)
    movement_vector = np.empty(shape=(30868,400,400,3),dtype=np.float16)
    quater_number = np.empty(shape=(30868,400,400,4),dtype=np.float16)
    phipsi = np.empty(shape=(30868,400,2),dtype=np.float16)
    DSSP = np.empty(shape=(30868,400),dtype=np.float16)
    mask = np.empty(shape=(30868,400),dtype=bool)
    seq = np.empty(shape=(30868,400),dtype=np.float16)
    for file in files:
        try:
            name = dir_name+file
            top = md.load(name).topology
            t = md.load(name)
            distance_value[i], movement_vector[i], quater_number[i] = compute_rotation_movment(traj=t,top=top,length=400)
            phipsi[i], DSSP[i], mask[i] = compute_phipsi_DSSP(top=top,t=t,length=400)
            seq[i] = get_seq(top=top,length=400)
            i = i+1
            count += 1
        except:
            #i += 1
            errors += 1
            print(file)
    print(count)
    print(errors)
    np.save("distance_value.npy",distance_value)
    np.save("movement_vector.npy",movement_vector)
    np.save("quater_number.npy",quater_number)
    np.save("phipsi.npy",phipsi)
    np.save("DSSP.npy",DSSP)
    np.save("mask.npy",mask)
    np.save("seq.npy",seq)
    end = time.time()

    print(end-start)
