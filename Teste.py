import numpy as np
import GA_lib as GA
import pandas as pd

n=8

matrix=np.random.randint(0,2, size=(n, n))
matrix*np.eye(n)

matrix=np.ones_like(matrix) - np.eye(matrix.shape[0])

data=pd.read_csv("ASIA_DATA.csv")

obs=5000
data_sampled=data.sample(obs)
data_sampled=data_sampled.astype('category')
GA.plot_digraph(matrix,np.array(data_sampled.columns))

bayesian_network_individual = GA.BayesianNetworkIndividual(matrix, data_sampled)

repaired_matrix=GA.repair_dag(bayesian_network_individual).toarray()
GA.plot_digraph(repaired_matrix,np.array(data_sampled.columns))