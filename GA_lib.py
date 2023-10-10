import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import warnings
from sympy import primerange

data=pd.read_csv("ASIA_DATA.csv")
dag_true=pd.read_csv("DAGtrue_ASIA.csv")
column_indices = np.random.permutation(data.columns)
N_samples=5000
data_sampled=data.sample(N_samples)
data_sampled=data_sampled.astype('category')
data_sampled = data_sampled[column_indices]



my_dict_global={}
prime_list = list(primerange(1, 7920))  # The range [1, 7920] contains the first 1000 primes



def my_dic_inti (data):
    # Generate the first 1000 prime numbers
    my_dict = {}
    for index,content in enumerate(data.columns):
        #print(index, content)
        my_dict[content] = index
    return my_dict

my_dict=my_dic_inti(data_sampled)


def state_count(data: pd.DataFrame, adjacency_matrix: np.ndarray, node: int, node_names: np.ndarray) -> pd.DataFrame:

    """
    Compute the states conditionally according to the DAG represented by the adjacency matrix.

    Args:
    
    adjacency_matrix: A adjacenty matrix.
    names:The coluns, nodes name. 
    data: Data frame ( All types must be converted into categorical)
    node: The colum/ node which the states will be computed 

    Returns:
     Pandas data frame containing the states conditionally counted according to the adjacency matrix

    """
    parents=np.where(adjacency_matrix[:, node] == 1)[0]
    selected_node_names = node_names[parents]
    group_filtered=[selected_node_names,node_names [node]]
    nested_array=group_filtered
    flattened_array = [item for sublist in nested_array for item in (sublist if isinstance(sublist, np.ndarray) else [sublist])]
    group_counts = data.groupby(flattened_array,observed=False).size().reset_index(name='Count')
    #group_counts['Count']=group_counts['Count']+1
    return group_counts

def BIC_score(data: pd.DataFrame, adjacency_matrix: np.ndarray, node_names: np.ndarray) -> int:
    """
    Compute the Bic score of DAG given the data adjacency matrix and nodes_name.

    Args:
    adjacency_matrix: A adjacenty matrix.
    names:The coluns, nodes name. 
    data: Data frame ( All types must be converted into categorical)
    Returns:
    BIC scored of the DAG
    """
    #print(my_dict_global)
    n=np.shape(adjacency_matrix)[0]
    obs=data.shape[0]
    Bn_size=0
    acc=0
    for node in range(n):
      #print(acc)
      bic=0
      states=state_count(data,adjacency_matrix,node,node_names)
      #print(list(states.columns))
      teste=np.array(states.columns[:-1])
      key=1
      #print(teste)
      if(len(teste)==1):
        key=prime_list[my_dict[teste[0]]+2*n]
      else:
        aux=teste[:-1]
      #np.random.shuffle(aux)
        for _ ,name in enumerate(aux):
            #print(name)
            key=key*prime_list[my_dict[name]]
        key=key*prime_list[my_dict[teste[-1]]+2*n]
      #print(key)
      #print(states)
      A=len(states.iloc[:,-2].unique())
      B=len(states)
      Bn_size=(B/A)*(A-1)+Bn_size
      if key not in  my_dict_global:
        count=0
        count2=0
        for i in range(int(B/A)):
            for z in range(int(A)):
                #print(states.loc[count,'Count'])
            
                if states.loc[count2:A+count2-1,'Count'].sum()==0:
                    bic =0+bic
                    #print("entrei")
                else:
                    if(states.loc[count,'Count']==0):
                        bic=bic+0
                    #print("entrei 2")
                    else:
                        bic=bic + states.loc[count,'Count']*np.log(states.loc[count,'Count']/states.loc[count2:A+count2-1,'Count'].sum())
                        #print("BIC:",bic)
                count=count+1
            count2=count2+A
        #print("bic",bic)
        my_dict_global[key]=bic
        #print("dic",my_dict_global[key])
      
      acc=acc+ my_dict_global[key]
    #print(Bn_size)
      

    return acc -0.5*np.log(obs)*Bn_size




def plot_digraph(adjacency_matrix: np.ndarray,names: np.ndarray):
    """
    Plot a graph givin its adjacency matrix .

    Args:
    adjacency_matrix: A adjacenty matrix.
    names:The coluns, nodes name. 

    Returns:
    void
    """
    # Create a DiGraph
    networkx_graph = nx.DiGraph()
    # Get the number of nodes based on the adjacency matrix shape
    num_nodes = adjacency_matrix.shape[0]
    # Add nodes to the graph
    networkx_graph.add_nodes_from(range(num_nodes))

    # Iterate through the adjacency matrix and add edges to the graph
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] == 1:
                networkx_graph.add_edge(i, j)
            
    # Print the resulting graph
    nodes_name_mapping = dict(zip(networkx_graph.nodes(),names))
    new_G = nx.relabel_nodes(networkx_graph, nodes_name_mapping, copy=True)
    pos = nx.nx_pydot.graphviz_layout(new_G, prog="dot")
    #pos=nx.nx_agraph.graphviz_layout(networkx_graph)
    fig, ax = plt.subplots(figsize=(9,6))
    nx.draw(new_G, pos, with_labels=True, node_size=1000, font_size=20, node_color='skyblue', font_color='black',width=0.5,
        connectionstyle='arc3,rad=0.3',arrows=True)
    plt.show()




def dag_to_bit(F1: np.ndarray) ->  np.ndarray:
  """
  Converts a adjacenty matrix to a bit representation.

  Args:
    F1: A adjacenty matrix.

  Returns:
    A bit representation of theadjacenty matrix.
  """
  n = F1.shape[0]
  bit_represent=np.array([])
  F = np.triu(F1, 1) + -1*np.tril(F1, -1).T
  #print(F)
  for i in range(n - 1):

    bit_represent=np.concatenate((bit_represent,  F[i, i + 1:n]))

  return bit_represent




def bit_to_dag(bit_represent: np.ndarray,index : np.ndarray )->  np.ndarray:
  """Converts a bit representation of a directed acyclic graph (DAG) to a DAG matrix.

  Args:
    bit_represent: A 1D NumPy array containing the bit representation of the DAG.
    n: The number of nodes in the DAG.
    index: 

  Returns:
    A 2D NumPy array containing the DAG matrix.
  """
  n=int((1+np.sqrt(1+4*1*len(bit_represent)*2))/2)

  matrix=np.zeros([n,n])

  for i in range(len(bit_represent)):
        matrix[index[i,0]][index[i,1]]=bit_represent[i]

  dag = 1 * ((matrix > 0) + (matrix.T < 0))

  # Return the DAG matrix.
  return dag







def is_dag(matrix :np.ndarray)->(bool, nx.DiGraph):
  """Converts an matrix to a NetworkX graph object and checks if the graph is a DAG.

  Args:
    matrix: A NumPy array representing the upper triangular matrix.

  Returns:
    A boolean value indicating whether or not the graph is a DAG.
    The nx.graph representation of the DAG
  """
  #print(matrix)
    # Check the dimension of the matrix
  if matrix.shape[0] != matrix.shape[1]:
    raise ValueError("The matrix must be square.")
  
  # Create a NetworkX graph object.
  G = nx.DiGraph()

  # Add edges to the graph object.
  for i in range(matrix.shape[0]):
    G.add_node(i)
    for j in range(matrix.shape[1]):
      if matrix[i, j] == 1:
        G.add_edge(i, j)


  
  # Check if the NetworkX graph is a DAG.
  is_dag_0 = nx.is_directed_acyclic_graph(G)

  return is_dag_0,G




class BayesianNetworkIndividual:
  """A class to represent Bayesian network individuals as objects."""

  def __init__(self, adjacency_matrix, pandas_dataframe):
    """Initializes a new BayesianNetworkIndividual object.

    Args:
      adjacency_matrix: A NumPy array representing the adjacency matrix of the Bayesian network.
      pandas_dataframe: A Pandas DataFrame containing the data to evaluate the Bayesian network on.
    """
    flag,nx_dag =is_dag(adjacency_matrix)
    #if not flag :
      #raise ValueError("The adjacency matrix is not a DAG. The Bayesian network is not be valid.")
      #adjacency_matrix=repair_dag(self).toarray()
      #warnings.warn("The adjacency matrix is not a DAG. The repair operator was called.")
      #warnings.warn("The adjacency matrix is not a DAG. The Bayesian network may not be valid.")

    n=adjacency_matrix.shape[0]
    mask=np.triu(np.ones([n,n]), k=1)
    rows, cols = np.where(mask)
    _Index = np.array([rows, cols]).T


    self.adjacency_matrix = adjacency_matrix
    self.pandas_dataframe = pandas_dataframe
    self.DAG_nx=nx_dag
    self.bit_representation =dag_to_bit(self.adjacency_matrix)
    self.Index=_Index

    if not flag :
      #warnings.warn("The adjacency matrix is not a DAG. The repair operator was called.")
      adjacency_matrix=repair_dag(self).toarray()
      self.adjacency_matrix = adjacency_matrix
      self.bit_representation =dag_to_bit(self.adjacency_matrix)
      flag,nx_dag =is_dag(adjacency_matrix)
      self.DAG_nx=nx_dag
      



  @classmethod
  def from_bit_representation(cls, bit_representation: np.ndarray, pandas_dataframe: pd.DataFrame,index : np.ndarray):
        """Alternative constructor based on bit_representation.
        Args:
            bit_representation: A NumPy array representing the bit representation of the Bayesian network.
            pandas_dataframe: A Pandas DataFrame containing the data to evaluate the Bayesian network on.

        Returns:
            A new BayesianNetworkIndividual object created from the bit_representation.
        """
        adjacency_matrix = bit_to_dag(bit_representation,index)
        return cls(adjacency_matrix, pandas_dataframe)

  def evaluate_cost(self):
    """Evaluates the Bayesian network on the given data.

    Returns:
      A flot value of the BIC to this individuo.
    """
    names=np.array(self.pandas_dataframe.columns)
    #print(self.adjacency_matrix)
    self.cost=BIC_score(self.pandas_dataframe, self.adjacency_matrix,names)
    
    return self.cost

  def crossover(self, other_individual):
    """Crosses over the current individual with the given other individual.

    Args:
      other_individual: Another BayesianNetworkIndividual object.

    Returns:
      A new BayesianNetworkIndividual object that is the result of crossing over the current individual with the given other individual.
    """
    
    AA=self.bit_representation
    BB=other_individual.bit_representation

    # Assuming you have the arrays AA and BB defined
    # You can replace these with your actual data

    # Generate two random points in the array indices
    point = np.random.permutation(len(AA))[:2]


    point = np.sort(point)
    
    # Create aux1 and aux2 based on the points
    #if(np.random.rand()>0.5):
    aux1 = np.concatenate((AA[:point[0]], BB[point[0]:point[1]], AA[point[1]:]))
    #else:
    aux2 = np.concatenate((BB[:point[0]], AA[point[0]:point[1]], BB[point[1]:]))

    new_individual1=BayesianNetworkIndividual.from_bit_representation(aux1,self.pandas_dataframe,self.Index)
    new_individual2=BayesianNetworkIndividual.from_bit_representation(aux2,self.pandas_dataframe,self.Index)


    return new_individual1,new_individual2

  def mutate(self):
    """Mutates the current individual.

    Returns:
      A new BayesianNetworkIndividual object that is the result of mutating the current individual.
    """
    aux=self.bit_representation
    coin = np.random.random()
    array = []
    if coin < 1/3:
        array = np.where(aux == 0)[0]
    elif coin < 2/3:
        array = np.where(aux == 1)[0]
    elif coin > 2/3:
        array = np.where(aux == -1)[0]
    #print(array)

    if np.random.random() > 0.5:
        if len(array) > 0:
            mu = np.random.randint(0, len(array))
            if(mu)>len(array)/4:
              mu=int(np.floor(len(array)/4))
            #print(mu)
        #print(mu)
            if len(array) > mu:
                if coin < 1/3:
                    if(mu>3):
                        mu = 3
                        #print(coin < 1/3)
                bit3 = np.random.choice(array, size=mu, replace=False)
            else:
                bit3 = [np.random.randint(1, len(aux) - 1)]    
        else:
            if len(array) == 0:
                bit3 = [np.random.randint(1, len(aux) - 1)]
            else:
                bit3 = [np.random.choice(array)]

    else:
            bit3 = [np.random.randint(0, len(aux) - 1)]


        

    #print('Bits: ',bit3)
    #print('conteudo:', aux[bit3])
      
    for bit in bit3:
        if aux[bit] == 0:
            old = 0
            if np.random.random() > 0.5:
                aux[bit] = 1
            else:
                aux[bit] = -1
        elif aux[bit] == 1:
            old = 1
            if np.random.random() > 0.5:
                aux[bit] = 0
            else:
                aux[bit] = -1
        else:
            old = -1
            if np.random.random() > 0.5:
                aux[bit] = 0
            else:
                aux[bit] = 1

        # Converting aux to a DAG and checking if it's valid
        ajd_m = bit_to_dag(aux, self.Index)
        flag, dag =is_dag(ajd_m)
        #print(flag)
        if not flag:
            aux[bit] = 0

    #print('Bits depois: ',bit3)
    #print('conteudo depois :', aux[bit3])
    ajd_m = bit_to_dag(aux, self.Index)
    flag, dag =is_dag(ajd_m)
    if not flag:
      print('NOT A DAG')
    #self=self.from_bit_representation(aux,self.pandas_dataframe,self.Index)
    self.DAG_nx=dag
    self.bit_representation=aux
    self.adjacency_matrix=ajd_m



  def mutate2(self):
    """Mutates the current individual.

    Returns:
      A new BayesianNetworkIndividual object that is the result of mutating the current individual.
    """
    aux=self.bit_representation

        # Define the number of elements to sample (n)
    mu = np.random.choice(len(aux), 1, replace=False)

    if(mu)>len(aux)/4:
      mu=int(np.floor(len(aux)/4))
    # Use np.random.choice() to randomly select 'n' indices
    sampled_indices = np.random.choice(len(aux), mu, replace=False)

    # Get the elements at the sampled indices
    #sampled_elements = aux[sampled_indices]

    bit3=sampled_indices
        

    print('Bits mutate 2: ',bit3)
    print('conteudo mutate 2:', aux[bit3])
      
    for bit in bit3:
        if aux[bit] == 0:
            old = 0
            if np.random.random() > 0.5:
                aux[bit] = 1
            else:
                aux[bit] = -1
        elif aux[bit] == 1:
            old = 1
            if np.random.random() > 0.5:
                aux[bit] = 0
            else:
                aux[bit] = -1
        else:
            old = -1
            if np.random.random() > 0.5:
                aux[bit] = 0
            else:
                aux[bit] = 1

        # Converting aux to a DAG and checking if it's valid
        ajd_m = bit_to_dag(aux, self.Index)
        flag, dag =is_dag(ajd_m)
        #print(flag)
        if not flag:
            aux[bit] = old
        print('Bits depois mutate 2: ',bit3)
        print('conteudo depois mutate 2 :', aux[bit3])
        self.bit_representation=aux














def repair_dag(BayesianNetworkIndividual)-> scipy.sparse._csr.csr_array:
  """
  Repairs a DAG by removing edges until it is acyclic.

  Args:
    BayesianNetwork: A BayesianNetwork class .

  Returns:
    The ajdancency matrix of the repaired dag.
  """
  #improve
  
    # Convert the binary adjacency matrix to a networkx DiGraph.
  G = BayesianNetworkIndividual.DAG_nx
    # While the DAG is not acyclic, remove a random edge.
  while not nx.is_directed_acyclic_graph(G):
      # Find all cycles in the DAG.
      #len_cycles=len(sorted(nx.simple_cycles(G)))
      cycles=sorted(nx.simple_cycles(G), key=lambda x: len(x))
      #print(cycles)
      #print("\n")
      k =0
      cycle = cycles[k]
      len_cycle=cycle
      #print(len_cycle)
      if len(len_cycle)<2:
          # Remove the edge from the cycle.
          G.remove_edge(cycle[0],cycle[0])
          #print('<2')
          #print((cycle[0],cycle[0]))
          
      else:
          if len(len_cycle)==2:
              # Remove the edge from the cycle.
              G.remove_edge(cycle[0],cycle[1])
              #print('=2')
              #print(cycle[0],cycle[1])
              
          else:
              #G.remove_edge(cycle[-1],cycle[0])
              #k = np.random.randint(0, len(len_cycle)-1)
              
              #j=k+1
              #while j == k:
              #    j = np.random.randint(0, len(len_cycle))
              #print('>2')
              
              # Remove the edge from the cycle.
              #print(G.has_edge(cycle[k], cycle[j]))
              #print((cycle))
              #if G.has_edge(cycle[-1], cycle[0]):
                #print(cycle[j],cycle[k])
              G.remove_edge(cycle[-1],cycle[0])
              #G.remove_edges_from(cycle)
              #print((cycle))

      
  
  
    
  # Get the adjacency matrix from the graph object.
  A = nx.adjacency_matrix(G,list(range(len(G))))

  # Convert the SciPy sparse matrix to a NumPy array.
  #A = np.array(A.toarray())

  return A
