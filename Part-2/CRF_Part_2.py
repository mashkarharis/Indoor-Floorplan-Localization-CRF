#!/usr/bin/env python
# coding: utf-8

# # Build Model

# ### Import Libraries

# In[1]:


import pandas as pd 
from matplotlib import image
from matplotlib import pyplot as plt
from math import cos, asin, sqrt, pi, atan2, log
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


# ### Loading Prepared Data

# 1. Loading Data CSV

# In[2]:


dataset = pd.read_csv("part_1_output/final_prepared_dataset.csv") 
datasetdf=pd.DataFrame(dataset)
datasetdf.head()


# 2. Load Floorplan Graph Dataset

# In[3]:


graph = pd.read_csv("part_1_output/floor_plan_graph.csv") 
graphdf=pd.DataFrame(graph)
graphdf.head()


# In[4]:


graph_dict=graphdf.T.to_dict().values()
graph_dict


# 3. Floorplan Graph Image

# In[5]:


floorplan = image.imread('part_1_output/floor_plan_graph.png')
plt.figure(figsize = (10,10))
plt.imshow(floorplan)
plt.show()


# ### Prepare Train Test Set

# Split inputs and targets

# In[6]:


inputsdf=datasetdf.iloc[:,0:42]
inputsdf.head()


# In[7]:


targetsdf=datasetdf.iloc[:,42:]
targetsdf.head()


# In[8]:


inputs=inputsdf.to_numpy()
print(inputs.shape)
inputs


# In[9]:


targets=targetsdf.to_numpy()
print(targets.shape)
targets


# Convert To Tensor

# In[10]:


inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)


# In[11]:


ds = TensorDataset(inputs, targets)
ds[0:3]


# Split Train Test Set Randomly

# In[12]:


# Random split
train_set_size = int(len(ds) * 0.8)
test_set_size = len(ds) - train_set_size
train_set, test_set = data.random_split(ds, [train_set_size, test_set_size])


# Train DataLoader

# In[13]:


batch_size = 5
train_dl = DataLoader(train_set, batch_size, shuffle=True)
len(train_dl)


# Test DataLoader

# In[14]:


batch_size = 5
test_dl = DataLoader(test_set, batch_size, shuffle=True)
len(test_dl)


# ### Define CRF Functions

# Nearest State

# In[15]:


def calcNearestState(x,y):
    currentNearestStateID=None
    minDistance=float('inf')
    
    for points in graph_dict:
        
        stateid=points["nodeid"]
        val1=points['y_dir_pixels']/2.5-y
        val1*=val1
        val2=points['x_dir_pixels']/2.5-x
        val2*=val2
        val3=val1+val2
        distanceToState=val3
        
        if distanceToState<=minDistance:
            minDistance=distanceToState
            currentNearestStateID=stateid
    return currentNearestStateID


# Connection Checking Function

# In[16]:


def isConnected(state_id_t,state_id_t_minus_1):
    state_t=graphdf[graphdf["nodeid"]==state_id_t]
    state_t_minus_1=graphdf[graphdf["nodeid"]==state_id_t_minus_1]
    gid1,gid2=state_t["connected_graph_id"].values[0],state_t_minus_1["connected_graph_id"].values[0]
    if gid1==gid2:
        return 1
    else:
        return 0


# Function 1

# In[17]:


def f1(velocity_value_at_t,std_of_velocity_values,state_id_t,state_id_t_minus_1):
    state_t=graphdf[graphdf["nodeid"]==state_id_t]
    state_t_minus_1=graphdf[graphdf["nodeid"]==state_id_t_minus_1]
    x2,y2=state_t["x_dir_pixels"].values[0],state_t["y_dir_pixels"].values[0]
    x1,y1=state_t_minus_1["x_dir_pixels"].values[0],state_t_minus_1["y_dir_pixels"].values[0]
    
    euc_dis=sqrt((y2-y1)**2+(x2-x1)**2)
    
    val= log(1/(std_of_velocity_values*sqrt(2*pi)))-((velocity_value_at_t-euc_dis)**2/(2*(std_of_velocity_values**2)))
    return val*isConnected(state_id_t,state_id_t_minus_1)
    


# Function 2

# In[18]:


def f2(velocity_angle_at_t,std_of_velocity_angles,state_id_t,state_id_t_minus_1):
    state_t=graphdf[graphdf["nodeid"]==state_id_t]
    state_t_minus_1=graphdf[graphdf["nodeid"]==state_id_t_minus_1]
    x2,y2=state_t["x_dir_pixels"].values[0],state_t["y_dir_pixels"].values[0]
    x1,y1=state_t_minus_1["x_dir_pixels"].values[0],state_t_minus_1["y_dir_pixels"].values[0]
    
    orientation=atan2(y2-y1,x2-x1)
    
    val= log(1/(std_of_velocity_angles*sqrt(2*pi)))-((velocity_angle_at_t-orientation)**2/(2*(std_of_velocity_angles**2)))
    return val*isConnected(state_id_t,state_id_t_minus_1)


# Combine Both

# In[19]:


def F(velocity_value_at_t,velocity_angle_at_t,std_of_velocity_values,std_of_velocity_angles,state_id_t,state_id_t_minus_1,w1,w2):
    
    return w1*f1(velocity_value_at_t,std_of_velocity_values,state_id_t,state_id_t_minus_1)+w2*f2(velocity_angle_at_t,std_of_velocity_angles,state_id_t,state_id_t_minus_1)
    


# Calculate std

# In[20]:


def std(list1):
    return torch.std(list1)


# ### Define Viterbi Algorithms

# #### Viterbi Recursion

# In[21]:


def ViterbiRecursion(starting_state,lstate,velocity_value_at_t,velocity_angle_at_t,std_of_velocity_values,std_of_velocity_angles,w1,w2,i,cache):
    
    if i==0:
        return F(velocity_value_at_t,velocity_angle_at_t,std_of_velocity_values,std_of_velocity_angles,lstate,starting_state,w1,w2)
    else:
        maxval=-float('inf')
        argmax=None
        for j in graphdf["nodeid"]:
            key="delta-i,l="+str(i-1)+","+str(j)
            if key in cache.keys():
                val1 = cache[key]
            else:
                val1 = ViterbiRecursion(starting_state,j,velocity_value_at_t,velocity_angle_at_t,std_of_velocity_values,std_of_velocity_angles,w1,w2,i-1,cache)
            val2 = F(velocity_value_at_t,velocity_angle_at_t,std_of_velocity_values,std_of_velocity_angles,lstate,j,w1,w2)
            val=val1+val2
            if(val>maxval):
                maxval=val
        cache["delta-i,l="+str(i)+","+str(lstate)]=maxval
        return maxval
        
    


# ### Define Model

# Detailed Model Diagram

# ![diagram.png](attachment:diagram.png)

# Layer 1

# In[22]:


#ViterbiRecursion(starting_state,lstate,velocity_value_at_t,velocity_angle_at_t,std_of_velocity_values,std_of_velocity_angles,w1,w2,i,cache):
   
class Layer1(torch.nn.Module):
    def __init__(self):
        
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(()))
        self.w2 = torch.nn.Parameter(torch.randn(()))
        self.deltas=[]
        self.gammas=[]
        

    def forward(self, x):
        startx=x[:,0]
        starty=x[:,1]
        v_values=x[:,2:22]
        v_angles=x[:,22:42]
        startingState=[]
        for iteration in range(len(startx)):
            startingState.append(calcNearestState(startx[iteration],starty[iteration]))
        startingState=torch.tensor(startingState)
        std_v_values=std(v_values)
        std_v_angles=std(v_angles)
        
        for datano in range(5):
            cache={}
            single_argmax=[]
            for t in range(1,21):
                maxval=-float('inf')
                argmax=None
                for state in range(30):
                    val = ViterbiRecursion(startingState[datano].item(),state,v_values[datano][t-1],v_angles[datano][t-1],std_v_values,std_v_angles,self.w1,self.w2,t,cache) 
                    if val>maxval:
                        maxval=val
                        argmax=state
                single_argmax.append(argmax)
            print(single_argmax)
            self.gammas.append(single_argmax)
            
        return self.gammas


# In[23]:


model = torch.nn.Sequential(
    Layer1()
)


# In[24]:


list(model.parameters())


# ### Training

# In[25]:


loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# In[26]:


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


# In[ ]:


fit(100, model, loss_fn, opt, train_dl)


# In[27]:


for xb,yb in train_dl:
    pred = model(xb)
    print(pred)
    break


# In[ ]:




