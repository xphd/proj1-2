import numpy as np
import keras
from keras.layers import Layer

def sample_random_walks(graph, num_walks, walk_length, p, q):
    """
    Sampling random walks from a graph. Each walk is a sequence of graph nodes obtained from a random walk.    
    Input: 
        graph: the graph object of networkx.Graph  
        num_walks: the number of random walks 
        walk_length: the length of random walks
        p: the p parameter. See the node2vec paper 
        q: the q parameter. See the node2vec paper
    Return: 
        walks: an numpy array with shape (num_walks, walk_length)
    """  
    
    walks = None

    ########

    # write your code here
    walks = np.zeros((num_walks, walk_length))
    num_nodes = graph.number_of_nodes()    
    aliasTables = _preprocess(graph, p, q)     
    for i in range(num_walks):
        start_node = np.random.randint(0, num_nodes) # randomly pick a node as start of each walk
        walk = _node2vecWalk(graph, start_node, walk_length, aliasTables)
        walks[i] = walk  
    ########

    return walks 

###
def _node2vecWalk(G, start_node, walk_length, aliasTables):
    walk = [start_node]
    for walk_iter in range(walk_length - 1):
        curr = walk[-1]
        curr_neighbors = list(G.neighbors(curr))
        s = start_node
        if len(walk) == 1: # if only one node in the walk, randomly pick one of its neighbors 
            n = len(curr_neighbors)
            i = np.random.randint(0, n)
            s = curr_neighbors[i]
        else:
            prev = walk[-2]        
            s = _aliasSample(prev, curr,curr_neighbors, aliasTables)            
        walk.append(s)    
    return walk
###

###
def _aliasSample(prev, curr, curr_neighbors, aliasTables):
    aliasTable = aliasTables[(prev, curr)]
    prob = aliasTable["prob"]
    alias = aliasTable["alias"]
    n = len(curr_neighbors)
    die_roll = np.random.randint(0, n)
    coin_toss = np.random.random()
    
    sample_node = curr
    if coin_toss < prob[die_roll]:
        sample_node = curr_neighbors[die_roll]
    else:
        sample_node = curr_neighbors[alias[die_roll]]
    return sample_node
###
    
###
def _preprocess(graph, p, q):
    '''
    Preprocess the graph with parameters p and q, to generate alias tables. Alias tables is a dict, where each key is an edge (t, v) and the value is the corresponding alias table which is used to sample next node when v is the current node and t is the previous node. Note that alias tables have both (t, v) and (v, t) as keys.
    input:
        graph: the graph object of networkx.Graph 
        p: the p parameter. See the node2vec paper 
        q: the q parameter. See the node2vec paper
    return:
        aliasTables
    '''
    edges = graph.edges
    aliasTables = {}
    for edge in edges:   
        t = edge[0]
        v = edge[1]
        aliasTables[(t, v)] = _getAliasTable(graph, p, q, t, v)
        aliasTables[(v, t)] = _getAliasTable(graph, p, q, v, t) 
    return aliasTables      
###

###
def _getAliasTable(graph, p, q, t, v):
    '''
    Get alias table for edge (t, v), where t is the previous node and v is the current node.
    input:
        graph: the graph object of networkx.Graph 
        p: the p parameter. See the node2vec paper 
        q: the q parameter. See the node2vec paper
        t: previous node
        v: current node
    return:
        aliasTable: alias table, a dict, for sampling
    '''
    unnormalized_transition_probabilities = []    
    v_neighbors = graph.neighbors(v)
    for x in v_neighbors:
        alpha = 0
        if x == t: # d_tx = 0
            alpha = 1 / p 
        elif graph.has_edge(x,t): # d_tx = 1
            alpha = 1
        else: # d_tx = 2
            alpha = 1 / q
        unnormalized_transition_probabilities.append(alpha)
    Z = sum(unnormalized_transition_probabilities) # normalizing constant
    normalized_transition_probabilities = [p / Z for p in unnormalized_transition_probabilities]
    aliasTable = _generateAliasTable(normalized_transition_probabilities)
    return aliasTable
###

###
def _generateAliasTable(probabilities):
    # Reference
    # http://www.keithschwarz.com/darts-dice-coins/
    # https://zhuanlan.zhihu.com/p/54867139
    '''
    Create alias table from input probability distribution
    Input:
        probabilities: probability distribution for a random variable. It's an array.
    Return:
        aliasTable: alias table, a dict, for sampling
    '''
    n = len(probabilities)
    p_times_n = [p * n for p in probabilities]
    prob, alias = [0] * n, [0] * n
    less, greater = [], []
    for i, p in enumerate(p_times_n):
        if p < 1:
            less.append(i)
        else:
            greater.append(i)
            
    while less and greater:
        less_idx, greater_idx = less.pop(), greater.pop()
        prob[less_idx] = p_times_n[less_idx]
        alias[less_idx] = greater_idx
        p_times_n[greater_idx] = p_times_n[greater_idx] - (1 - p_times_n[less_idx])
        if p_times_n[greater_idx] < 1.0:
            less.append(greater_idx)
        else:
            greater.append(greater_idx)

    while greater:
        greater_idx = greater.pop()
        prob[greater_idx] = 1
    while less:
        less_idx = less.pop()
        prob[less_idx] = 1
    aliasTable = {"prob": prob, "alias": alias}
    return aliasTable
###
    
    

def collect_skip_gram_pairs(walks, context_size, num_nodes, num_negative_samples):
    """
    Generate positive node pairs from random walks, and also generate random negative pairs 
    Input: 
        walks: numpy.array with shape (num_walks, walk_length). Each row is a random walk with each entry being a graph node 
        context_size: integer, the maximum number of nodes considered before or after the current node
        num_nodes: integer, the size of the original graph generating random walks. (num_nodes - 1) is the largest node index.  
        num_negative_samples: integer, the number of negative nodes to be sampled to get negative pairs.  
    Return: 
        pairs: an numpy array with shape (num_pairs, 3) and type integer. Each row contains a pair of nodes and the label of the pair. 
               If the pair is generated from two nearby nodes in a random walk, then the label is 1. If the pair is generated 
               from a node in the random walk and a random node, then the pair has label 0. The number of pairs is a little less than 
               walks.shape[0] * walks.shape[1] * context_size * 2. In general, each node in `walks` generates `2 * context_size` pairs. 
               But if the node is at the beginning or the end of a walk, it generates less pairs.  
    """

    pairs = None


    ######################################

    assert(walks.shape[1] >= (2 * context_size + 1))


    # write your code here
    pairs_list = []
    walk_length = walks.shape[1]  
    c = context_size 
    # generate positive node pairs    
    for walk in walks:
        for i in range(walk_length):        
            center = walk[i]    
            c_range = [*range(-c, 0), *range(1, c + 1)]
            for j in c_range:           
                pos_idx = i + j
                if pos_idx > -1 and pos_idx < walk_length:               
                    pair = (center, walk[pos_idx], 1) 
                    pairs_list.append(pair)  
                
    # generate negative node pairs
#     neg_pairs = []
#     c = context_size
    for walk in walks: 
        for i in range(walk_length): 
            c_range = [*range(-c, 0), *range(1, c + 1)]
            valid_index = []
            for j in c_range: 
                idx = i + j
                if idx > -1 and idx < walk_length:               
                    valid_index.append(idx)
            random_index = np.random.choice(valid_index, len(valid_index))
            for index in random_index:
                pair_0 = walk[index]
                p = _getProbability(walks, num_nodes)
                for pair_1 in np.random.choice(num_nodes, num_negative_samples, p):
                    neg_pair = (pair_0, pair_1, 0)                      
                    pairs_list.append(neg_pair) 
    # convert list to np.array
    pairs = np.array(pairs_list)

    ######################################

    return pairs

###
def _getProbability(walks, num_nodes):
    '''
    Create alias table from input probability distribution
    Input:
        walks: an numpy array with shape (num_walks, walk_length)
        num_nodes: an integer, the number of nodes in the graph.
    Return:
        p: a list, the probability of the categorical distribution over all nodes (words) in the graph (vocabulary). This probability for each node is proportional to the frequency of nodes (words) in random_walks (corpus) raised to power of 3/4 (check section 2.2 of [2])
    '''
    counts = np.zeros(num_nodes) # initialize
    for walk in walks:
        for v in walk:
            counts[v] = counts[v] + 1
    counts_powered = counts ** (3/4)
    Z = sum(counts_powered) 
    normalized_p = counts_powered / Z
    p = normalized_p.tolist()
    return p
###



class EmbLayer(Layer):
    """
    A keras layer. The layer takes the input of node pairs, looks up embedding vectors for nodes in pairs, and compute 
    the inner product of the vectors in each node pair. 
    """

    def __init__(self, num_nodes=None, emb_dim=None, init_emb=None, output_vecs=None, **kwargs):
        """
        Initialization of the layer. You should provide two ways to initialize the layer. In the first way, provide the shape, 
        (num_nodes, emb_dim), of the embedding matrix. Then the embedding matrix will be initialized internally. In the second 
        way, provide an initial embedding matrix such the embedding matrix will be initialized by the provided matrix. 
        Input: 
            num_nodes: integer, the number of nodes in the graph. Must be provided if init_emb == None. 
            emb_dim: integer, the embedding dimension. Must be provided if init_emb == None. 
            init_emb: numpy.array with size (num_nodes, emb_dim). If this argument is provided, the embedding matrix must be 
                      initialized by this argument, then `num_nodes` or `emb_dim` should have no effect. 
            output_vecs: numpy.array with size (num_nodes, emb_dim). If init_emb is not None, then this argument needs to be 
                         provided, otherwise this argument is neglected.
        Return: 
            
        """

        ######################################### 
        # Write your code here        
        if init_emb is not None:
            self.init_emb = init_emb
            if output_vecs is None:
                print("Error! output_vecs must be proviede when init_emb is provided!")
            else:
                self.output_vecs = output_vecs
                print("Success!")
        else: # init_emb is None
            if num_nodes is None or emb_dim is None:
                print("Error! Both num_nodes and emb_dim must be provided when init_emb is not.")
            else :
                self.init_emb = (np.random.random(size=(num_nodes, emb_dim)) - 0.5) * 2
                self.output_vecs = (np.random.random(size=(num_nodes, emb_dim)) - 0.5) * 2
                print("Success!")
#         print(self.init_emb)
        ######################################### 

        super(EmbLayer, self).__init__(**kwargs) # Be sure to call this at the end 

    def build(self, input_shape):
        """
        Build the keras layer. You should allocate the embedding matrix (also called weights or kernel) as a optimization 
        variable in this function.
        Input:
            input_shape: it should be (num_pairs, 3). It has no use in deciding the shape of the embedding matrix. 
        """

        ######################################### 
        # Write your code here
        model = Keras.models.Sequential()
        model.add(Dense())
        ######################################### 

        super(EmbLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, pairs):
        """
        Here we define the computation (graph) of the layer. Given a pair of nodes, look up the two embedding vectors, 
        take the inner product of the two vectors, and convert it to a probability by the sigmoid function
        Input: 
            pairs: keras tensor with shape (batch_size, 2). Each row is a pair of nodes
        Return: 
            prob: keras tensor with shape (batch_size, ). The probabilities of pair labels being 1 

        """

        ######################################### 
        # Write your code here
        print("EmbLayer.call is called")
        logits = sigmoid(np.sum(self.init_emb[pairs[0]] * self.output_vecs[pairs[1]], axis=1))             
#         logits = 1

        input_shape = slef.init_emb.shape
        self.build(input_shape)
        ######################################### 

        return logits 

    def compute_output_shape(self, input_shape):
        return (input_shape[0],1)


def node2vec(graph, num_walks, walk_length, p, q, context_size, num_negative_samples, emb_dim, num_epochs):
    """
    The node2vec algorithm. 
    Input: 
        graph: the graph object of networkx.Graph  
        num_walks: the number of random walks 
        walk_length: the length of random walks
        p: the p parameter. See the node2vec paper 
        q: the q parameter. See the node2vec paper
        context_size: integer, the maximum number of nodes considered before or after the current node
        num_negative_samples: integer, the number of negative nodes to be sampled to get negative pairs.  
        emb_dim: integer, the embedding dimension 
        num_epochs: integer, number of training epochs
    Return: 
        node_emb: an numpy array with shape (num_nodes, emb_dim)
    """
 
    node_emb = np.random.random(size=(graph.number_of_nodes(), emb_dim))

    ############################################################
    # Write your code here
    ############################################################


    return node_emb 

