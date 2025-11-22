# %% [markdown]
# # K-MEANS ALGORITHM

# %% [markdown]
# This exercise consists of three parts. Finish the first part to get a mark of 3.0; the first two parts for 4.0. Complete all three parts to get 5.0. <br>
# Advanced* and optional - means it is optional and will not affect the grade.

# %% [markdown]
# ## Part 1

# %%
### SOME IMPORTS
import common as cm
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# 1.1) Complete the following distance function

# %%
import math
# Computes a Euclidean distance between points A and B (these are vectors, i.e., A[0], A[1], ....)
def getEuclideanDistance(A, B):
    return math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    
### TEST
print(getEuclideanDistance([0.0, 0.0], [0.0, 1.0]))
print(getEuclideanDistance([0.0, 0.0], [1.0, 1.0]))

# %% [markdown]
# 1.2) Get test data set & display (data = matrix n x m, n = the number of objects, m = the number of attributes)

# %%
data = cm.getTestDataSet()
cm.displayDataSet(plt, data) #plt = plot package; see the imports above

# %% [markdown]
# **How many different clusters (groups) do you see here?**

# %%
4

# %% [markdown]
# The data for this exercise was generated artificially. You can run the below code to see the "true" group assignment.

# %%
data = cm.getTestDataSet()
assignments = cm.getTestAssignments() ### GET "TRUE" GROUP ASSIGNMENT
cm.displayDataSet(plt, data, assignments = assignments)

# %% [markdown]
# 1.3) K-Means implementation: Firstly, we need to construct K "centroids". Each centroid represents one group. For simplicity, initially assume that the centroids are randomly selected from the data set (i.e., clone/copy some K random points from data set). Check numpy.random package. Important: each centroid should be unique (no repetitions, consider a "shuffle" approach). Finish the bolow code.

# %%
### return a vector of centroids (vectors) [[x1, y1], ..., [xk, yk]]
def getCentroids(K, data):
    rng = np.random.default_rng()
    centroids = rng.choice(data, K, False)
    return centroids

print(getCentroids(2, data))

# %% [markdown]
# 1.4) Get acquainted with some parameters: 
# - DATA - test data set, a vector of  n 2d points: [[x1, y1], ..., [xn, yn]], loaded from common.py
# - M - the number of attributes/dimensions; M = 2 for this exercise,
# - K - expected number of groups,
# - CENTROIDS - initial K centroids; CENTROIDS =  [[x1, y1],...,[xk, yk]],
# - ASSIGNMENTS - data structure representing group assignments; ASSIGNMENTS = [[idx1_1,...,],....,[idx1_K,....]], i.e., i-th element is a vector of indexes of corresponding data points in DATA, being assigned to i-th group. For instance if |DATA| = 3, K = 2, and ASSIGNMENTS = [[0, 2], [1]], it means that DATA[0] and DATA[2] points are assigned to the first group, while DATA[1] point is assigned to the second group.

# %% [markdown]
# 1.5) Finish the below function. It should perform a single step of K-Means algorithm:
# 
# a) Firstly, construct new group assignments. For this reason, iterate over all data points. For each (i-th) point, verify its distance to each (k-th) centroid. Check for which centroid the distance is the smallest and update NEW_ASSIGNMENTS adequately (NEW_ASSIGNMENTS[k-th centroid].append(i-th index/data point)).
# 
# Important: NO_CHANGE boolean variable should be set to False if the assignments have changed from the previous iteration to the current one (NEW_ASSIGNMENTS != (OLD) ASSIGNMENTS). 
# 
# b) Update centroids (NEW_CENTROIDS), i.e., compute centers of masses of data points belonging to different groups. 
# 
# c) Return NO_CHANGE, NEW_ASSIGNMENTS, NEW_CENTROIDS.

# %%
def doKMeansStep(DATA, M, K, CENTROIDS, ASSIGNMENTS):    
    NO_CHANGE = True
    if ASSIGNMENTS is None: NO_CHANGE = False
    #TODO
    
    ### CONSTRUCT NEW ASSIGNMENTS
    NEW_ASSIGNMENTS = [[] for k in range(K)]
    for i in range(len(DATA)):
        minDistacnce = math.inf
        assigment = 0
        for c in range(len(CENTROIDS)):
            distance = getEuclideanDistance(DATA[i], CENTROIDS[c])
            if minDistacnce > distance:
                minDistacnce = distance
                assigment = c
        NEW_ASSIGNMENTS[assigment].append(i)

    if ASSIGNMENTS is not None:
        # NO_CHANGE  = ASSIGNMENTS == NEW_ASSIGNMENTS
        NO_CHANGE  = np.array_equal(ASSIGNMENTS, NEW_ASSIGNMENTS)

    ### CONSTRUCT NEW CENTROIDS
    NEW_CENTROIDS = []
    for groupIndex in range(K):
        atributesSums = [0] * M
        for pointIndex in NEW_ASSIGNMENTS[groupIndex]:
            for a in range(M):
                atributesSums[a] += DATA[pointIndex][a]
        centroid = []
        for a in range(M):
            centroid.append(atributesSums[a]/ len(NEW_ASSIGNMENTS[groupIndex]))
        NEW_CENTROIDS.append(centroid)

    
    return NO_CHANGE, NEW_CENTROIDS, NEW_ASSIGNMENTS 

# %% [markdown]
# 1.6) The below code performs 1 iteration of K-Menas algorithm for K=2 and the test data set. Check the results (notice that centroids are marked with squares).

# %%
DATA = cm.getTestDataSet()
CENTROIDS = getCentroids(2, DATA)
NO_CHANGE, CENTROIDS, ASSIGNMENTS = doKMeansStep(DATA, 2, 2, CENTROIDS.copy(), None)
cm.displayDataSet(plt, DATA, assignments = ASSIGNMENTS, centroids = CENTROIDS)

# %% [markdown]
# 1.7) Complete the below piece of code. The doKMeans function should perform 100 steps of K-Means algorithm. However, the loop should be stopped when the NO_CHANGE variable = True. It that is so, **print the information on after how many iterations the process has stopped**. Lastly, use cm.displayDataSet to depict the final groups. 
# 
# (Advanced*): make an animation showing the steps of the K-means algorithm. It is easier to do in jupyter notebook.
# https://matplotlib.org/3.3.2/api/animation_api.html
# http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/

# %%
def doKMeans(DATA, CENTROIDS, K = 2, M = 2, display = True):
    ASSIGNMENTS = [[] for i in range(K)] 
    steps = 0 
    for i in range(100):
        NO_CHANGE, CENTROIDS, ASSIGNMENTS = doKMeansStep(DATA, M, K, CENTROIDS.copy(), None)
        steps = i+1
        if NO_CHANGE:
            break
    print('Compleated after: ', steps, ' steps') 
    return DATA, CENTROIDS, ASSIGNMENTS   

DATA = cm.getTestDataSet()
CENTROIDS = getCentroids(2, DATA)
DATA, CENTROIDS, ASSIGNMENTS = doKMeans(DATA, CENTROIDS, K = 2)
cm.displayDataSet(plt, DATA, assignments = ASSIGNMENTS, centroids = CENTROIDS)

# %% [markdown]
# 1.8) Run the below piece of code and observe the results. Which K seems to be the best choice?

# %%
DATA = cm.getTestDataSet()
for k in range(2, 11):
    CENTROIDS = getCentroids(k, DATA)
    DATA, NEW_CENTROIDS, NEW_ASSIGNMENTS = doKMeans(DATA, CENTROIDS, K = k)
    cm.displayDataSet(plt, DATA, assignments = NEW_ASSIGNMENTS, centroids = NEW_CENTROIDS)

# k=4 is the best choice


# %% [markdown]
# ## Part 2

# %% [markdown]
# The quality of final group assignment can be assessed in various ways. In this exercise, you are asked to compute a total (sum) distance between data points and their cluster centroids for different values of K. Obviously, it is expected that such indicator will always decrese with the increase of K. But, obviously, $K=\infty$ is not the best option. However, there exsits some threshold K' such that for each K'' > K' the decrease will not be significant. This threshold is called an "elbow" and its corresponding K value is considered satisfactory. Firstly, complete the below function. It should compute the total (sum) distance between data points and their cluster centroids. Secondly, compute the the total distances for final clusters for for $K\in [2, 10]$. Then, plot the results. Use cm.displayResults(plt, results), where results takes the follwoing form: [[2, result for K = 2], [3, result for K = 3], ..., [10, result for K = 10]]. **Find the "elbow"**.

# %%
def getTotalDistance(DATA, CENTROIDS, ASSIGNMENTS):
    #TODO
    return 0.0

# %%
### PERFORM THE ANALYSIS HERE

# %% [markdown]
# # Part 3 - a small case study

# %% [markdown]
# In this exercise you are asked to use the K-Means algorithm to cluster some data provided in cm.getCaseDataSet() (important note, this data was generated artificially - this is not real-world data). This data contains information on 250 persons. Each person is characterized with the following attributes:
# 
# a) Age <br>
# b) Salary (zł) <br> 
# c) Health index (0-100; 0 = worst health, 100 = best health) <br>
# d) Time spent in school or work (hours) <br>
# e) Time spent on sport activities (hours) <br>
# 
# Follow these steps:
# 1. Load the data. 
# 2. Check the min and max values for each attribute and consider normalization. 
# 3. Run K-Means algorithm (use the pieces of code you completed in previous exercises) for different K. 
# 4. Identify the best K using the performance indicator introduced in Part 2. 
# 5. For the best K - analyze constructed clusters. 
#     * Compute basic stats (mean, max, min and standard deviation) attribute values within each cluster. You should use centroids constructed in the final iteration. 
#     * (optional) Generate distribution plots. 
#     * (optional) Calculate the internal consistency of clusters by calculating basic stats (mean, max, min and standard deviation) of distance between each pair of objects in each cluster and visualize this data.
#     * (optional) Compute basic stats (mean, max, min and standard deviation) of distance between each pair of clusters and visualize this data.
#     * You can print these values / show pandas DataFrame / visualize them with matplotlib boxplot (optional). 
#     * Values should be presented in a non-normalized version.
# 6. Using the above, briefly describe each cluster.

# %%
DATA = cm.getCaseDataSet()

# %%
### CONSIDER NORMALIZATION HERE

# %%
 def doKMeans_CaseStudy(DATA, K = 2, M = 5):
    CENTROIDS = getCentroids(K, DATA) # GET CENTOIDS
    ASSIGNMENTS = [[] for i in range(K)] # 
    ### TODO
    return DATA, CENTROIDS, ASSIGNMENTS 

DATA_N, CENTROIDS, ASSIGNMENTS = doKMeans_CaseStudy(DATA_N, K = 2)

# %%
### DO THE ANALYSIS HERE (FIND ELBOW)

# %% [markdown]
# **Characterize the data in clusters generated by K-means run for suitably adjusted K**

# %%
### DISPLAY - SUMMARIZE - STATS FOR THE BEST K FOUND IN THE PREVIOUS STEP

# %%



