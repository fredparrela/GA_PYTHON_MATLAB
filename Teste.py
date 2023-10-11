import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import GA_lib as GA
import copy
from  GA_lib import data_sampled
from  GA_lib import dag_true
import bnlearn as bn
from pgmpy.estimators import ParameterEstimator, BicScore
from pgmpy.models import BayesianNetwork
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import classification_report



# Read the CSV file into a DataFrame
df = dag_true
# Define the desired topological order
topological_order =list(data_sampled.columns)

# Create a list of unique variables (nodes)
nodes = np.unique(df[['Variable 1', 'Variable 2']].values)

# Initialize an empty adjacency matrix filled with zeros
num_nodes = len(nodes)
adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

# Iterate through the DataFrame and update the adjacency matrix based on dependencies
for _, row in df.iterrows():
    var1 = row['Variable 1']
    var2 = row['Variable 2']
    dependency = row['Dependency']

    # Find the indices of var1 and var2 in the topological order list
    index1 = topological_order.index(var1)
    index2 = topological_order.index(var2)

    # Set the corresponding entry in the adjacency matrix to 1
    if dependency == '->':
        adjacency_matrix[index1, index2] = 1

#print("Adjacency Matrix (in topological order):")
#print(adjacency_matrix)

def otimizador_func(obj):
    obj.evaluate_cost()
    obj_old_cost=obj.cost
    # Deep copy (new copies of all objects referenced within obj)
    obj_old_deep = copy.deepcopy(obj)
    #obj_old=obj
    obj.mutate()
    obj.evaluate_cost()
    i=0
    while(i<obj.adjacency_matrix.shape[0]):


        #print("obj_old_cost :", obj_old_cost)
        #print("obj_old_deep.cost :", obj_old_deep.cost)
        #print("obj.cost   :", obj.cost)

        if(obj.cost>obj_old_cost):
            obj_old_deep = copy.deepcopy(obj)
            obj_old_cost=obj_old_deep.cost
            #print("Entrei")
        else:
            obj=copy.deepcopy(obj_old_deep)
            obj.mutate()
            obj.evaluate_cost()
            #print("nao entrei")
        i=i+1
        
        



    #print("obj_old_deep:", obj_old_deep.cost)
    #print("custo atual :", obj.cost)
    return obj_old_deep
n=adjacency_matrix.shape[0]
mask=np.triu(np.ones([n,n]), k=1)
rows, cols = np.where(mask)
index = np.array([rows, cols]).T


### GA body 
best_fit=[]
max_interations=5000
stability_threshold=500
prev_best_cost=0
j=0
pc=0.8
pm=0.5
po=0.05
n_pais=8
stable_count=1
initial_pop=300
#######################################################################################################################################
#Creating initial population

# Define the probabilities vector
n=adjacency_matrix.shape[0]
vector_length = int(n*(n-1)*0.5)
probabilities = [1/vector_length, (vector_length-2)/vector_length, 1/vector_length]
Pop_size = np.array(range(initial_pop))
##

object_list = []
for i in range(initial_pop):
    vector = np.random.choice([-1, 0, 1], size=vector_length, p=probabilities)
    obj=GA.BayesianNetworkIndividual.from_bit_representation(vector,data_sampled,index=index)
    cost=obj.evaluate_cost()
    #print(cost)
    object_list.append(obj)

sorted_people = sorted(object_list, key=lambda obj:obj.cost,reverse = True)

while(True):

    #for j in range(max_interations):
    # parents selection
    parents=[]
    for i in range(n_pais):
        # selection by tourment assuming and order vector just return the smallest index drawed
        selected_torneio=np.random.choice(Pop_size, size=1, replace=False)
        parents.append(sorted_people[selected_torneio[0]])

    # Crossover
    filhos=[]
    for i in  range(0, n_pais, int(2)):
        if(pc>np.random.rand()):
        #parents[i]=otimizador_func(parents[i])
            f1,f2 =parents[i].crossover(parents[i+1])
            filhos.extend([f1,f2])
        else:
            filhos.extend([parents[i],parents[i+1]])

    # Mutation & cost evaluation
    for i in range(n_pais):
        if(pm>np.random.rand()):
            filhos[i].mutate()
        filhos[i].evaluate_cost()

    if(po>np.random.rand()):
        inx=np.random.choice(n_pais,1)[0]
        filhos[inx]=otimizador_func(filhos[inx])

    # Sorting acording to cost function 
    sorted_people.extend(filhos)
    sorted_people = sorted(sorted_people, key=lambda obj:obj.cost,reverse = True)
    po=po*1.009
    for _ in range(n_pais):
        sorted_people.pop()
        best_fit.append(sorted_people[0].cost)

    # Check for stability
    #print("prev_best_cost", prev_best_cost)
    #print("best_fit[j]", best_fit[j])
    if best_fit[j] == prev_best_cost:
        stable_count += 1
    #print("stable_count", stable_count)

    else:
        stable_count =0

    if stable_count >= stability_threshold:
        print(f"Stopping criterion met: Cost has been stable for {stability_threshold} generations.")
        break

    if j >= max_interations:
        print("Maximum number of iterations reached.")
        break

    #print("Stable count", stable_count)
    #print("prev_best_cost ", prev_best_cost)
    prev_best_cost=copy.deepcopy(sorted_people[0].cost)
    #prev_best_cost=(sorted_people[0].cost)
    j=j+1


interacoes = np.arange(0, len(best_fit))
# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(interacoes, best_fit, label='best_fit', color='red')

ground=GA.BayesianNetworkIndividual(adjacency_matrix,data_sampled)
ground.evaluate_cost()


GA.plot_digraph(sorted_people[0].adjacency_matrix,data_sampled.columns)