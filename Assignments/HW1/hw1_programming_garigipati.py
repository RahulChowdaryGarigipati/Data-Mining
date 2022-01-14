### Q1(1) Euclidean Distance
import math
p1 = (1, 2, 4, 5)
p2 = (4, 3, 2, 1)
euclidean_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
print("Euclidean distance from p1 to p2: ",euclidean_distance)

### Q1(1) Manhattan Distance
import math
p1 = (1, 2, 4, 5)
p2 = (4, 3, 2, 1)
manhattan_distance = sum(abs(a-b) for a,b in zip(p1,p2))
print("Manhattan distance from p1 to p2: ",manhattan_distance)

### Q1(1) Cosine Similarity
import math
p1 = (1, 2, 4, 5)
p2 = (4, 3, 2, 1)
dot = sum(a*b for a, b in zip(p1, p2))
norm_a = sum(a*a for a in p1) ** 0.5
norm_b = sum(b*b for b in p2) ** 0.5
cosine_similarity = dot / (norm_a*norm_b)
print('Cosine Similarity between p1 and p2 is:', cosine_similarity)

### Q1(2) Simple Matching Coefficient (SMC)
def SMC(p3,p4):
    similarity = 0.0
    total = 0.0
    
    
    for i in p3:
        for j in p4:
            if p3[i]==0 and p4[j]==0:
                total +=1
            elif p3[i] == 1 and p4[j] == 1:
                total +=1
            elif p3[i] == 1 and p4[j] == 0:
                similarity +=1
            elif p3[i] == 0 and p4[j] == 1:
                similarity +=1
    
    total += similarity
    count = similarity/total
    return count
p3 = [1,1,1,0,0]
p4 = [1,0,0,0,1]
print('Simple Matching Coefficient between p3 and p4 is:',SMC(p3,p4))

### Q1(2) Jaccard Coefficient
def jaccard(p3,p4):
    similarity = 0.0
    total =0.0
    zero_zero=0.0
    
    for i in p3:
        for j in p4:
            if p3[i]==0 and p4[j]==0:
                total +=1
            elif p3[i] == 1 and p4[j] == 1:
                total +=1
            elif p3[i] == 1 and p4[j] == 0:
                similarity +=1
            elif p3[i] == 0 and p4[j] == 1:
                zero_zero +=1
    
    total += similarity
    count = (similarity)/(total)-(zero_zero)
    return count
p3 = [1,1,1,0,0]
p4 = [1,0,0,0,1]

count=(jaccard(p3,p4))
print('Jaccard Coefficient between p3 and p4 is:',count)

### Q1(2) Hamming Distance
def hammingDist(p3, p4):
    i = 0
    count = 0
 
    while(i < len(p3)):
        if(p3[i] != p4[i]):
            count += 1
        i += 1
    return count

p3 = [1 ,  1  , 1 ,  0  , 0]
p4 = [1 ,  0  , 0 ,  0  , 1]
 
print('Hamming Distance Between p3 and p4 is:',hammingDist(p3, p4))

### Q1(3) Dynamic Time Warping (DTW)
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
x = np.array([1, 2, 4, 5])
y = np.array([1, 1, 1, 0, 0])
distance, path = fastdtw(x, y, dist=euclidean)
print('Dynamic Time Warping Distance between p1 and p3 is: ', distance)
print('Dynamic Time Warping Path between p1 and p3 is:',path)

### Q2(1) Random Sampling
import pandas as pd
data = pd.read_csv (r'C:\Users\rahul\Desktop\Data Mining\HW1\BreastCancerCoimbra_imbalanced.csv')
df = pd.DataFrame(data)
x  = df.sample(n=10)
print(x)

### Q2(2) Stratified sampling
import pandas as pd
from scipy.stats import mode
data = pd.read_csv (r'C:\Users\rahul\Desktop\Data Mining\HW1\BreastCancerCoimbra_imbalanced.csv')
df = pd.DataFrame(data)
x  = mode(df.sample(n=10))
print(df)