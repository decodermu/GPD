import sys
import os
import random
import numpy as np

currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)

from GPD.GPD import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Graphomer_20220111().to(device)
    #导入参数
    optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
    use_gpu = torch.cuda.is_available()
    
    #Train_loader, Test_loader = loaddataset()
    subset = "train"
    seq_name = "../data/seq.npy"
    seq_mask_name = "../data/seq.npy"
    phipsi_name = "../data/phipsi.npy"
    DSSP_name = "../data/DSSP.npy"
    distance_name = "../data/distance_value.npy"
    movement_name = "../data/movement_vector.npy"
    quate_name = "../data/quater_number.npy"
    path_length_name = "../data/path_length.npy"
    centerity_name = "../data/centerity.npy"
    mask_name = "../data/mask.npy"

    seqs_np = np.load(seq_name).astype("long")
    seqs_mask_np = np.load(seq_mask_name).astype("long")
    phipsi_np = np.load(phipsi_name).astype("float32")
    DSSP_np = np.load(DSSP_name).astype("long")
    distance_np = np.load(distance_name).astype("float32")
    movement_np = np.load(movement_name).astype("float32")
    quate_np = np.load(quate_name).astype("float32")
    path_length_np = np.load(path_length_name).astype("float32")
    centerity_np = np.load(centerity_name).astype("float32")
    mask_np = np.load(mask_name).astype("bool")
    #根据distances生成mask
    seqs = torch.from_numpy(seqs_np)
    seqs_mask = torch.from_numpy(seqs_mask_np)
    phipsi = torch.from_numpy(phipsi_np)
    DSSP = torch.from_numpy(DSSP_np)
    path_length = torch.from_numpy(path_length_np)
    centerity = torch.from_numpy(centerity_np)
    distance = torch.from_numpy(distance_np)
    movement = torch.from_numpy(movement_np)
    quate = torch.from_numpy(quate_np)
    mask = torch.from_numpy(mask_np)
    Total_set = torch.utils.data.TensorDataset(seqs,seqs_mask,phipsi,DSSP,centerity,distance,path_length,movement,quate,mask)
    Train_set , Test_set = torch.utils.data.random_split(Total_set,[29868,1000])
    Train_loader = torch.utils.data.DataLoader(Train_set,batch_size=64)
    Test_loader = torch.utils.data.DataLoader(Test_set,batch_size=64)
    print("finish loading dataset")

    num_epoch = 200
    best_acc = 0.0
    best_epoch = 0
    best_model = model

    for epoch in range(1,num_epoch+1):
        if(use_gpu):
            model = model.cuda()
            loss_func = loss_func.cuda()
        step = 0
        for seqs,seqs_mask,phipsi,DSSP,centerity,distance,path_length,movement,quate,mask in Train_loader:
            step += 1
            rand_seed = torch.rand(size=(seqs.shape[0],seqs.shape[1],1))
            seqs_zeros = torch.zeros(size=seqs.shape,dtype=torch.long)
            seqs_rand = torch.rand(size=seqs.shape)
            mask_percentage = random.random()
            seqs_mask = torch.where(seqs_rand>mask_percentage,seqs,seqs_zeros)
            if(use_gpu):
                seqs = seqs.cuda()
                seqs_mask = seqs_mask.cuda()
                phipsi = phipsi.cuda()
                DSSP = DSSP.cuda()
                centerity = centerity.cuda()
                distance = distance.cuda()
                path_length = path_length.cuda()
                movement = movement.cuda()
                quate = quate.cuda()
                mask = mask.cuda()
                rand_seed = rand_seed.cuda()
            distance = torch.unsqueeze(distance,dim=-1) #(30966,400,400) -> (30966,400,400,1)
            path_length = torch.unsqueeze(path_length,dim=-1) #as top
            src_mask = torch.cat([distance*0,distance,path_length,movement,quate],dim=-1) #(30966,400,400,10)
            output = model(phipsi=phipsi,DSSP=DSSP,centerity=centerity,rand_seed=rand_seed,tgt=seqs_mask,src_mask=src_mask,padding_mask=mask)
            #print(output[0][0])
            output = output[0:].view(-1,output.shape[-1])
            label = seqs[0:].view(-1)
            loss = loss_func(output,label)
            predict = output.max(1)
            acc = accuracy(predict=predict,label=label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(mask_percentage,end="\t")
            #print(loss,end='\t')
            #print(acc,end="\t")
            #print(step,end='\t')
            #print(epoch)
        acc_test = 0.0
        i = 0
        with torch.no_grad():
            for seqs,seqs_mask,phipsi,DSSP,centerity,distance,path_length,movement,quate,mask in Test_loader:
                step += 1
                rand_seed = torch.rand(size=(seqs.shape[0],seqs.shape[1],1))
                seqs_zeros = torch.zeros(size=seqs.shape,dtype=torch.long)
                seqs_rand = torch.rand(size=seqs.shape)
                seqs_mask = torch.where(seqs_rand>0.0,seqs,seqs_zeros)
                if(use_gpu):
                    seqs = seqs.cuda()
                    seqs_mask = seqs_mask.cuda()
                    phipsi = phipsi.cuda()
                    DSSP = DSSP.cuda()
                    centerity = centerity.cuda()
                    distance = distance.cuda()
                    path_length = path_length.cuda()
                    movement = movement.cuda()
                    quate = quate.cuda()
                    mask = mask.cuda()
                    rand_seed = rand_seed.cuda()
                distance = torch.unsqueeze(distance,dim=-1) #(30966,400,400) -> (30966,400,400,1)
                path_length = torch.unsqueeze(path_length,dim=-1) #as top
                src_mask = torch.cat([distance*0,distance,path_length,movement,quate],dim=-1) #(30966,400,400,10)
                output = model(phipsi=phipsi,DSSP=DSSP,centerity=centerity,rand_seed=rand_seed,tgt=seqs_mask,src_mask=src_mask,padding_mask=mask)
                output = output[0:].view(-1,output.shape[-1])
                label = seqs[0:].view(-1)
                predict = output.max(1)
                acc_test += accuracy(predict=predict,label=label)
                i += 1
            acc_test = acc_test / i
        #print("Test acc:",end="\t")
        #print(acc_test,end="\t")
        #print("epoch:",end="\t")
        #print(epoch)
        print("epoch\t%.0f\tmask_percentage\t%.3f\tloss\t%.3f\tacc\t%.3f\tTest_acc\t%.3f" % (epoch, mask_percentage, loss, acc, acc_test))
        if(acc_test>best_acc):
            best_model = model
            best_epoch = epoch
            best_acc = acc_test
            torch.save(best_model.state_dict(),"./tmp.pkl")
    print("best epoch:",end=" ")
    print(best_epoch)
    print("acc:",end=" ")
    print(best_acc)
    torch.save(best_model.state_dict(),"./20220607_random_1.pkl")
