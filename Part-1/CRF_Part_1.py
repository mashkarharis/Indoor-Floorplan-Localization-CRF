#!/usr/bin/env python
# coding: utf-8

# # Inertial + Floorplan Localization Using CRF

# ## Paper

# This implementation is based on the researches done on the following papers,

# [1]	Z. Xiao, H. Wen, A. Markham, και N. Trigoni, ‘Lightweight map matching for indoor localisation using conditional random fields’, στο IPSN-14 proceedings of the 13th international symposium on information processing in sensor networks, 2014, σσ. 131–142.
# 
# [2]	J. Zhang, M. Ren, P. Wang, J. Meng, και Y. Mu, ‘Indoor localization based on VIO system and three-dimensional map matching’, Sensors, τ. 20, τχ. 10, σ. 2790, 2020.
# 

# Note that due to unavailability of exact dataset used for above researchers, I had to use following dataset and convert that accordingly.

# [3] S. Herath, S. Irandoust, B. Chen, Y. Qian, P. Kim, και Y. Furukawa, ‘Fusion-DHL: WiFi, IMU, and Floorplan Fusion for Dense History of Locations in Indoor Environments’, στο 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, σσ. 5677–5683.

# ## Theory

# In this notebook, I am going to implement indoor localization mechanism using Linear Chain Conditional Random Fields. By using this model we can predict location of a user when starting position, IMU observations (velocity vectors) and floorplan of the building is given.

# Inputs,
# 
# * Stating Position - (Meters In X direction(TopLeft|LeftRight), Meters In Y direction(TopLeft|TopBottom))
# * Sequence of Velocity Vectors Captured In Small Time Range (20 seconds) : Velocity_Values(ms-1), Velocity_Angles(radian)
# * Graph of Floorplan
# 

# ### Overall Architecture

# Here is the overall system architecture

# ![image.png](attachment:image.png)

# The input is a velocity vector observed using IMU data Z = {Z0,...,ZT }, and the task is to predict a sequence of states S = {S0,...,ST } given input Z.

# ### Viterbi Algorithm

# We use Viterbi algorithm, which can dynamically solve the optimal state points sequence that is most likely to produce the currently given observation value sequence. The solution steps of Viterbi algorithm are as follows:

# (1) Initialization: Compute the non-normalized probability of the first position for all states, where m is the number of states.

# ![image.png](attachment:image.png)

# (2) Recursion: Iterate through each state from front to back, find the maximum value of the non-normalized probability of each state l = 1, 2, · · · , m at position i = 2, 3, · · · , n, and record the state sequence label Ψi(l) with the highest probability.

# ![image.png](attachment:image.png)

# (3) When i = n, we obtain the maximum value of the non-normalized probability and the terminal
# of the optimal state points sequence

# ![image.png](attachment:image.png)

# (4) Calculate the final state points output sequence

# ![image.png](attachment:image.png)

# (5) Finally, the optimal sequence of state points is as follows:

# ![image.png](attachment:image.png)

# ### Defined F and W

# We can use w and F(y, x) to represent the weight vector and the global state transfer function vector.

# ![image-2.png](attachment:image-2.png)

# where I(Yt−1, Yt) is an indicator function equal to 1 when states Yt−1 and Yt are connected and 0 otherwise.

# We use two functions f1 anf f2

# ![image.png](attachment:image.png)

# where xdt is the Euclidean distance between two consecutive observations, d(yt−1, yt) is the Euclidean distance between two consecutive state points, and σ2d is the variance of the distance in the observation data.

# ![image.png](attachment:image.png)

# where xθt is the orientation of two consecutive observations, θ(yt−1, yt) is the orientation between two consecutive state points, and σ2θ is the variance of the orientation in the observation data.

# ## Load Libraries

# In[1]:


import pandas as pd 
from matplotlib import image
from matplotlib import pyplot as plt
from math import cos, asin, sqrt, pi, atan2


# ## Data Preprocessing

# Here are units we use,
# 
# * location coordinates (x,y) : In pixels
# * angle : radian
# * distance - meter
# * velocity - meter per second

# ### Prepare Floorplan

# For this we use Lougheed Floorplan
# 
# 2.5 Pixels = 1 m

# ![lougheed_00.png](attachment:lougheed_00.png)

# Here are the constrains,
# 
# Coordinate Center - (TOP LEFT, LR-> X+, TB -> Y+)
# 
# * Top Left (TL) - (0,0) in pixels, (49.252463,-122.897553) in Lat,Lon
# * Bottom Left (BL) - (0,776) in pixels, (49.249681,-122.897553) in Lat,Lon
# * Top Right (TR) - (887,0) in pixels, (49.252463,-122.892735) in Lat,Lon
# * Bottom Right (BR) - (887,776) in pixels, (49.249681,-122.892735) in Lat,Lon

# Now Let's load data we have. For this we use Fusion Location Provider's Data. (Which is in file FLP)

# Load Data Set 1

# In[2]:


dataset1 = pd.read_csv("dataset1.csv") 
df1=pd.DataFrame(dataset1)
df1.head()


# Load Data Set 2

# In[3]:


dataset2 = pd.read_csv("dataset2.csv") 
df2=pd.DataFrame(dataset2)
df2.head()


# Concat Two Datasets

# In[4]:


df_merged=pd.concat([df1,df2])
df_merged.reset_index(inplace=True, drop=True)
df_merged.head()


# Get Dataframe Informations

# In[5]:


print(df1.shape[0])
print(df2.shape[0])
print(df_merged.shape[0])


# Select Only required data

# In[6]:


sub_df=df_merged[['Column1','Column2','Column3']]
sub_df = sub_df.rename(columns={'Column1': 'TimeStamp', 'Column2': 'Latitude','Column3':'Longitude'})
print(sub_df.shape[0])
sub_df.head()


# However we can't deal with Latitude and Longitude, We have to convert it to pixels or meters.

# In[7]:


# Calculate X direction,
# X direction -- Longitude
# X's Plus direction = Longitude's Plus Direction

# Calculate Y direction,
# Y direction -- Latitude
# Y's Plus direction = Latitude's Negative Direction

X_0_in_longitude=-122.897553
Y_0_in_latitude=49.252463
pixelspermeter=2.5
number_of_node_in_graph=30

# Distance between Two Lat,Lon


def distanceLatLonInMeters(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742000 * asin(sqrt(a))

x_dir_pixels=[]
y_dir_pixels=[]
x_dir_meters=[]
y_dir_meters=[]

for tuple in sub_df.itertuples():
    meters_in_x_direction=abs(distanceLatLonInMeters(Y_0_in_latitude,X_0_in_longitude,Y_0_in_latitude,tuple[3]))
    meters_in_y_direction=abs(distanceLatLonInMeters(Y_0_in_latitude,X_0_in_longitude,tuple[2],X_0_in_longitude))
    
    pixels_in_x_direction=round(meters_in_x_direction*2.5)
    pixels_in_y_direction=round(meters_in_y_direction*2.5)
    
    x_dir_pixels.append(pixels_in_x_direction)
    y_dir_pixels.append(pixels_in_y_direction)
    
    x_dir_meters.append(meters_in_x_direction)
    y_dir_meters.append(meters_in_y_direction)
    
    ##print(pixels_in_x_direction,pixels_in_y_direction,meters_in_x_direction,meters_in_y_direction)
    
updated_df=sub_df.copy()
updated_df["Pixels In X Direction"] = x_dir_pixels
updated_df["Pixels In Y Direction"] = y_dir_pixels
updated_df["Meters In X Direction"] = x_dir_meters
updated_df["Meters In Y Direction"] = y_dir_meters


# Let's see updated dataframe

# In[8]:


updated_df.head()


# Load Image and Mark Visited Areas

# In[9]:


floorplan0 = image.imread('lougheed_00 - Copy.png')
plt.imshow(floorplan0)
for row in updated_df.itertuples():
    plt.plot(row[4],row[5] , marker=',', color="red")
plt.savefig('visited_areas.png', bbox_inches='tight')
plt.show()


# Now We can see that the area where measurements are taken

# Now let's select some points to create graphs. In here we should **create several connected graph and give them unique graph ids**. However, the map we selected contains only one connected graph.

# In[10]:


graph=[{'nodeid': 0, 'x_dir_pixels': 392, 'y_dir_pixels': 285, 'connected_graph_id': 'G1'}, {'nodeid': 1, 'x_dir_pixels': 283, 'y_dir_pixels': 353, 'connected_graph_id': 'G1'}, {'nodeid': 2, 'x_dir_pixels': 445, 'y_dir_pixels': 357, 'connected_graph_id': 'G1'}, {'nodeid': 3, 'x_dir_pixels': 447, 'y_dir_pixels': 442, 'connected_graph_id': 'G1'}, {'nodeid': 4, 'x_dir_pixels': 280, 'y_dir_pixels': 270, 'connected_graph_id': 'G1'}, {'nodeid': 5, 'x_dir_pixels': 308, 'y_dir_pixels': 228, 'connected_graph_id': 'G1'}, {'nodeid': 6, 'x_dir_pixels': 283, 'y_dir_pixels': 415, 'connected_graph_id': 'G1'}, {'nodeid': 7, 'x_dir_pixels': 260, 'y_dir_pixels': 550, 'connected_graph_id': 'G1'}, {'nodeid': 8, 'x_dir_pixels': 265, 'y_dir_pixels': 493, 'connected_graph_id': 'G1'}, {'nodeid': 9, 'x_dir_pixels': 269, 'y_dir_pixels': 436, 'connected_graph_id': 'G1'}, {'nodeid': 10, 'x_dir_pixels': 261, 'y_dir_pixels': 513, 'connected_graph_id': 'G1'}, {'nodeid': 11, 'x_dir_pixels': 258, 'y_dir_pixels': 470, 'connected_graph_id': 'G1'}, {'nodeid': 12, 'x_dir_pixels': 390, 'y_dir_pixels': 500, 'connected_graph_id': 'G1'}, {'nodeid': 13, 'x_dir_pixels': 341, 'y_dir_pixels': 500, 'connected_graph_id': 'G1'}, {'nodeid': 14, 'x_dir_pixels': 263, 'y_dir_pixels': 449, 'connected_graph_id': 'G1'}, {'nodeid': 15, 'x_dir_pixels': 266, 'y_dir_pixels': 366, 'connected_graph_id': 'G1'}, {'nodeid': 16, 'x_dir_pixels': 275, 'y_dir_pixels': 319, 'connected_graph_id': 'G1'}, {'nodeid': 17, 'x_dir_pixels': 282, 'y_dir_pixels': 463, 'connected_graph_id': 'G1'}, {'nodeid': 18, 'x_dir_pixels': 282, 'y_dir_pixels': 306, 'connected_graph_id': 'G1'}, {'nodeid': 19, 'x_dir_pixels': 440, 'y_dir_pixels': 500, 'connected_graph_id': 'G1'}, {'nodeid': 20, 'x_dir_pixels': 286, 'y_dir_pixels': 331, 'connected_graph_id': 'G1'}, {'nodeid': 21, 'x_dir_pixels': 283, 'y_dir_pixels': 332, 'connected_graph_id': 'G1'}, {'nodeid': 22, 'x_dir_pixels': 285, 'y_dir_pixels': 369, 'connected_graph_id': 'G1'}, {'nodeid': 23, 'x_dir_pixels': 230, 'y_dir_pixels': 540, 'connected_graph_id': 'G1'}, {'nodeid': 24, 'x_dir_pixels': 438, 'y_dir_pixels': 323, 'connected_graph_id': 'G1'}, {'nodeid': 25, 'x_dir_pixels': 289, 'y_dir_pixels': 390, 'connected_graph_id': 'G1'}, {'nodeid': 26, 'x_dir_pixels': 329, 'y_dir_pixels': 240, 'connected_graph_id': 'G1'}, {'nodeid': 27, 'x_dir_pixels': 200, 'y_dir_pixels': 550, 'connected_graph_id': 'G1'}, {'nodeid': 28, 'x_dir_pixels': 371, 'y_dir_pixels': 265, 'connected_graph_id': 'G1'}, {'nodeid': 29, 'x_dir_pixels': 282, 'y_dir_pixels': 242, 'connected_graph_id': 'G1'}]
graphtable=pd.DataFrame(graph)
graphtable


# Let's draw graph,

# In[11]:


floorplan1 = image.imread('lougheed_00 - Copy.png')
plt.figure(figsize = (10,10))
plt.imshow(floorplan1)
for row in graph:
    plt.plot(row["x_dir_pixels"],row["y_dir_pixels"] , marker='.', color="red")
plt.savefig('floor_plan_graph.png', bbox_inches='tight')
plt.show()


# Now we have successfully created a graph for the floor plan

# ### Prepare DataSet

# Let's revisit updated_df

# In[12]:


updated_df


# Denfine Helper Functions

# In[13]:


def calcVelocityVal(prev_row,row):
    distance=sqrt((row[6]-prev_row[6])**2+(row[7]-prev_row[7])**2)
    timediff=row[1]-prev_row[1]
    return distance/timediff

def calcVelocityAngle(prev_row,row):
    return atan2(row[7]-prev_row[7], row[6]-prev_row[6])

def calcNearestState(row):
    
    currentNearestStateID=None
    minDistance=float('inf')
    
    for points in graph:
        
        stateid=points["nodeid"]
        distanceToState=sqrt((points['y_dir_pixels']-row[5])**2+(points['x_dir_pixels']-row[4])**2)
        
        if distanceToState<=minDistance:
            minDistance=distanceToState
            currentNearestStateID=stateid
    return currentNearestStateID
    


# Now let's calculate velocities and nearest states

# In[14]:


velocity_value=["N/A"]
velocity_angle=["N/A"]
nearest_state=["N/A"]
isValid=[]
prev_row=None

for row in updated_df.itertuples():
   
    # Process First Value
    if row[0]==0:
        prev_row=row
        isValid.append(0)
        continue
        
    timediff=row[1]-prev_row[1]
    if timediff>=2:
        isValid.append(0)
    else:
        isValid.append(1)
    
        
    velocity_value.append(calcVelocityVal(prev_row,row))
    velocity_angle.append(calcVelocityAngle(prev_row,row))
    nearest_state.append(calcNearestState(row))
    prev_row=row
    
updated_df["Velocity_Value"]=velocity_value
updated_df["Velocity_Angle"]=velocity_angle
updated_df["Nearest_State"]=nearest_state
updated_df["isValid"]=isValid


# Let's check new dataframe

# In[15]:


updated_df


# Now lets created filtered dataframe which only contains, required columns and valid values

# In[16]:


filtered_df=updated_df.copy()
filtered_df=filtered_df[filtered_df['isValid'] == 1]
filtered_df=filtered_df[["TimeStamp","Meters In X Direction","Meters In Y Direction","Velocity_Value","Velocity_Angle","Nearest_State"]]
filtered_df.reset_index(inplace=True, drop=True)
filtered_df.head()


# Let's check first index of second data set

# In[17]:


print(filtered_df.loc[873:877])


# In[18]:


first_index_of_second_set=875


# Now Let's create a dataset

# In[19]:


data=[]
for row in filtered_df.loc[19:first_index_of_second_set-1].itertuples():
    datadict={}
    start_x = filtered_df.loc[row[0]-19][1]
    start_y = filtered_df.loc[row[0]-19][2]
    datadict["startX"]=start_x
    datadict["startY"]=start_y
    i=0
    for subrow in filtered_df.loc[row[0]-19:row[0]].itertuples():
        datadict["velocity_value_"+str(i+1)]=subrow[4]
        i+=1
    j=0
    for subrow in filtered_df.loc[row[0]-19:row[0]].itertuples():
        datadict["velocity_angle_"+str(j+1)]=subrow[5]
        j+=1
    k=0
    for subrow in filtered_df.loc[row[0]-19:row[0]].itertuples():
        datadict["state_"+str(k+1)]=subrow[6]
        k+=1
    data.append(datadict)

prepared_dataset1=pd.DataFrame(data)
prepared_dataset1


# In[20]:


data=[]
for row in filtered_df.loc[first_index_of_second_set:].itertuples():
    datadict={}
    start_x = filtered_df.loc[row[0]-19][1]
    start_y = filtered_df.loc[row[0]-19][2]
    datadict["startX"]=start_x
    datadict["startY"]=start_y
    i=0
    for subrow in filtered_df.loc[row[0]-19:row[0]].itertuples():
        datadict["velocity_value_"+str(i+1)]=subrow[4]
        i+=1
    j=0
    for subrow in filtered_df.loc[row[0]-19:row[0]].itertuples():
        datadict["velocity_angle_"+str(j+1)]=subrow[5]
        j+=1
    k=0
    for subrow in filtered_df.loc[row[0]-19:row[0]].itertuples():
        datadict["state_"+str(k+1)]=subrow[6]
        k+=1
    data.append(datadict)

prepared_dataset2=pd.DataFrame(data)
prepared_dataset2


# In[21]:


final_prepared_dataset=pd.concat([prepared_dataset1,prepared_dataset2])
final_prepared_dataset.reset_index(inplace=True, drop=True)
final_prepared_dataset


# ### Save Outcomes

# Now let's save out results

# In[22]:


graphtable.to_csv("floor_plan_graph.csv", encoding='utf-8', index=False)
final_prepared_dataset.to_csv("final_prepared_dataset.csv", encoding='utf-8', index=False)

