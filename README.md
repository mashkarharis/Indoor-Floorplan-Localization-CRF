# Inertial + Floorplan Localization Using CRF

## Paper

This implementation is based on the researches done on the following papers,

[1]	Z. Xiao, H. Wen, A. Markham, και N. Trigoni, ‘Lightweight map matching for indoor localisation using conditional random fields’, στο IPSN-14 proceedings of the 13th international symposium on information processing in sensor networks, 2014, σσ. 131–142.

[2]	J. Zhang, M. Ren, P. Wang, J. Meng, και Y. Mu, ‘Indoor localization based on VIO system and three-dimensional map matching’, Sensors, τ. 20, τχ. 10, σ. 2790, 2020.


Note that due to unavailability of exact dataset used for above researchers, I had to use following dataset and convert that accordingly.

[3] S. Herath, S. Irandoust, B. Chen, Y. Qian, P. Kim, και Y. Furukawa, ‘Fusion-DHL: WiFi, IMU, and Floorplan Fusion for Dense History of Locations in Indoor Environments’, στο 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, σσ. 5677–5683.

## Theory

In this notebook, I am going to implement indoor localization mechanism using Linear Chain Conditional Random Fields. By using this model we can predict location of a user when starting position, IMU observations (velocity vectors) and floorplan of the building is given.

Inputs,

* Stating Position - (Meters In X direction(TopLeft|LeftRight), Meters In Y direction(TopLeft|TopBottom))
* Sequence of Velocity Vectors Captured In Small Time Range (20 seconds) : Velocity_Values(ms-1), Velocity_Angles(radian)
* Graph of Floorplan


### Overall Architecture

Here is the overall system architecture

![image](https://user-images.githubusercontent.com/54017081/190659807-904a3aba-bc21-4145-ae31-8e01536976b8.png)

The input is a velocity vector observed using IMU data Z = {Z0,...,ZT }, and the task is to predict a sequence of states S = {S0,...,ST } given input Z.

### Viterbi Algorithm

We use Viterbi algorithm, which can dynamically solve the optimal state points sequence that is most likely to produce the currently given observation value sequence. The solution steps of Viterbi algorithm are as follows:

(1) Initialization: Compute the non-normalized probability of the first position for all states, where m is the number of states.

![image](https://user-images.githubusercontent.com/54017081/190659872-eb1e00ea-ee51-4e78-bf14-33c3365da137.png)

(2) Recursion: Iterate through each state from front to back, find the maximum value of the non-normalized probability of each state l = 1, 2, · · · , m at position i = 2, 3, · · · , n, and record the state sequence label Ψi(l) with the highest probability.

![image](https://user-images.githubusercontent.com/54017081/190659929-0b4b951b-4f1e-4f66-804c-4d0a20eadeff.png)

(3) When i = n, we obtain the maximum value of the non-normalized probability and the terminal
of the optimal state points sequence

![image](https://user-images.githubusercontent.com/54017081/190659964-b82b2e3b-0a93-4099-ab70-c16b8f3749ab.png)

(4) Calculate the final state points output sequence

![image](https://user-images.githubusercontent.com/54017081/190660022-3b971ae0-3c53-4acf-a473-59f4f3e5d7d7.png)

(5) Finally, the optimal sequence of state points is as follows:

![image](https://user-images.githubusercontent.com/54017081/190660055-807b1135-1070-45af-8866-52f0be6a788f.png)

### Defined F and W

We can use w and F(y, x) to represent the weight vector and the global state transfer function vector.

![image](https://user-images.githubusercontent.com/54017081/190660094-26364f07-3514-4d14-81c5-c771bd551f63.png)

where I(Yt−1, Yt) is an indicator function equal to 1 when states Yt−1 and Yt are connected and 0 otherwise.

We use two functions f1 anf f2

![image](https://user-images.githubusercontent.com/54017081/190660144-8d3d22e0-9c0f-4914-b85d-e0f2e7205210.png)

where xdt is the Euclidean distance between two consecutive observations, d(yt−1, yt) is the Euclidean distance between two consecutive state points, and σ2d is the variance of the distance in the observation data.

![image](https://user-images.githubusercontent.com/54017081/190660209-0f444d20-85b6-4d58-a501-02e0775f8a7b.png)

where xθt is the orientation of two consecutive observations, θ(yt−1, yt) is the orientation between two consecutive state points, and σ2θ is the variance of the orientation in the observation data.
