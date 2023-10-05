import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import warnings


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
    n=np.shape(adjacency_matrix)[0]
    obs=data.shape[0]
    Bn_size=0
    bic=0
    for node in range(n):
        states=state_count(data,adjacency_matrix,node,node_names)
        A=len(states.iloc[:,-2].unique())
        B=len(states)
        Bn_size=(B/A)*(A-1)+Bn_size
        #acc=0
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
                        bic=bic+states.loc[count,'Count']*np.log(states.loc[count,'Count']/states.loc[count2:A+count2-1,'Count'].sum())
                count=count+1
            count2=count2+A

    return bic -0.5*np.log(obs)*Bn_size




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
    for j in range(matrix.shape[1]):
      if matrix[i, j] == 1:
        G.add_edge(i, j)


  
  # Check if the NetworkX graph is a DAG.
  is_dag = nx.is_directed_acyclic_graph(G)

  return is_dag,G




class BayesianNetworkIndividual:
  """A class to represent Bayesian network individuals as objects."""

  def __init__(self, adjacency_matrix, pandas_dataframe):
    """Initializes a new BayesianNetworkIndividual object.

    Args:
      adjacency_matrix: A NumPy array representing the adjacency matrix of the Bayesian network.
      pandas_dataframe: A Pandas DataFrame containing the data to evaluate the Bayesian network on.
    """
    flag,nx_dag =is_dag(adjacency_matrix)
    if not flag :
      #raise ValueError("The adjacency matrix is not a DAG. The Bayesian network is not be valid.")
      warnings.warn("The adjacency matrix is not a DAG. The Bayesian network may not be valid.")
    #pandas_dataframe.columns

    #nodes_name_mapping = dict(zip( list(range(adjacency_matrix.shape[0])), pandas_dataframe.columns))
    #nodes_dic = {}
    #for node, node_name in zip(list(range(adjacency_matrix.shape[0])), pandas_dataframe.columns):
    #  nodes_dic[node] = {"labels": node_name}
    #print(nodes_dic)
    #my_dict = {0: 'asia', 1: 'tub', 2: 'smoke', 3: 'lung', 4: 'bronc', 5: 'either', 6: 'xray', 7: 'dysp'}
    #nx.set_node_attributes(nx_dag, nodes_dic)
    self.adjacency_matrix = adjacency_matrix
    self.pandas_dataframe = pandas_dataframe
    self.DAG_nx=nx_dag
    self.bit_representation =dag_to_bit(self.adjacency_matrix)

  def evaluate_cost(self):
    """Evaluates the Bayesian network on the given data.

    Returns:
      A flot value of the BIC to this individuo.
    """
    names=np.array(self.pandas_dataframe.columns)
    #print(self.adjacency_matrix)
    
    return BIC_score(self.pandas_dataframe, self.adjacency_matrix,names)

  def crossover(self, other_individual):
    """Crosses over the current individual with the given other individual.

    Args:
      other_individual: Another BayesianNetworkIndividual object.

    Returns:
      A new BayesianNetworkIndividual object that is the result of crossing over the current individual with the given other individual.
    """

    new_individual = BayesianNetworkIndividual(
        crossover_adjacency_matrix(self.adjacency_matrix, other_individual.adjacency_matrix),
        self.pandas_dataframe.copy())
    return new_individual

  def mutate(self):
    """Mutates the current individual.

    Returns:
      A new BayesianNetworkIndividual object that is the result of mutating the current individual.
    """

    new_individual = BayesianNetworkIndividual(
        mutate_adjacency_matrix(self.adjacency_matrix), self.pandas_dataframe.copy())
    return new_individual













def repair_dag(BayesianNetworkIndividual)-> scipy.sparse._csr.csr_array:
  """
  Repairs a DAG by removing edges until it is acyclic.

  Args:
    BayesianNetwork: A BayesianNetwork class .

  Returns:
    The ajdancency matrix of the repaired dag.
  """

  
    # Convert the binary adjacency matrix to a networkx DiGraph.
  G = BayesianNetworkIndividual.DAG_nx
    # While the DAG is not acyclic, remove a random edge.
  while not nx.is_directed_acyclic_graph(G):
      # Find all cycles in the DAG.
      len_cycles=len(sorted(nx.simple_cycles(G)))
      cycles=list(nx.simple_cycles(G))
      k = np.random.randint(0, (len_cycles)) 
      cycle = cycles[k]
      len_cycle=cycle
      print(len_cycle)
      if len(len_cycle)<2:
          # Remove the edge from the cycle.
          G.remove_edge(cycle[0],cycle[0])
          print('<2')
          print((cycle[0],cycle[0]))
          
      else:
          if len(len_cycle)==2:
              # Remove the edge from the cycle.
              G.remove_edge(cycle[0],cycle[1])
              print('=2')
              print(cycle[0],cycle[1])
              
          else:
              
              k = np.random.randint(0, len(len_cycle))
              
              j=k
              while j == k:
                  j = np.random.randint(0, len(len_cycle))
              print('>2')
              
              # Remove the edge from the cycle.
              if G.has_edge(cycle[j], cycle[k]):
                print(cycle[j],cycle[k])
                G.remove_edge(cycle[j],cycle[k])

      
  
  
    
  # Get the adjacency matrix from the graph object.
  A = nx.adjacency_matrix(G,list(range(len(G))))

  # Convert the SciPy sparse matrix to a NumPy array.
  #A = np.array(A.toarray())

  return A
