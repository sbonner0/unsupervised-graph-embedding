import networkx as nx
import random
import tensorflow as tf
from os import walk
from os.path import splitext
from os.path import join
from multiprocessing import Process
import multiprocessing


def read_graph(dataset_location, weighted=False, directed=False):
    """Reads the input network in networkx """

    if weighted:
        G = nx.read_edgelist(dataset_location, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(dataset_location, nodetype=int, data=False)

    # Provide sequential node labelling
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_undirected()

    return G

def generate_random_graphs(num_nodes, num_edges, gen_meth):
    """Generate random graphs: 1 for ER, 2 for BA"""

    if gen_meth == 1:
        temp_graph = nx.erdos_renyi_graph(num_nodes, 0.3)

    elif gen_meth == 2:
        temp_graph = nx.barabasi_albert_graph(num_nodes, num_edges)

    else:
        raise ValueError('Invalid model selection')

    print("Graph Generation Complete")

    return temp_graph

def generate_deg_seq_graph():
    """Generate a graph with a set degree sequence"""

    temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10 = ([] for i in range(10))

    for i in range(5000):
        temp1.append(1)
        temp2.append(5)
        temp3.append(10)
        temp4.append(15)
        temp5.append(20)
        temp6.append(25)
        temp7.append(30)
        temp8.append(35)
        temp9.append(40)
        temp10.append(45)

    temp = temp1+temp2+temp3+temp4+temp5+temp6+temp7+temp8+temp9+temp10
    random.shuffle(temp)

    G = nx.configuration_model(temp, create_using=nx.Graph())

    return G

def extract_graph_features(graph):
    """Extract features and save in graph object"""

    # Extract features for each vertex
    deg = dict(nx.degree(graph))
    deg_cent = nx.degree_centrality(graph)
    clu = nx.clustering(graph)
    tr = nx.triangles(graph)
    pr = nx.pagerank_scipy(graph, alpha=0.75)
    eig = nx.eigenvector_centrality_numpy(graph)
    bet = nx.betweenness_centrality(graph, k=100)

    # Save features back into the graph object
    nx.set_node_attributes(graph, deg, 'DEG')
    nx.set_node_attributes(graph, deg_cent, 'DC')
    nx.set_node_attributes(graph, clu, 'CLU')
    nx.set_node_attributes(graph, tr, 'TR')
    nx.set_node_attributes(graph, pr, 'PR')
    nx.set_node_attributes(graph, eig, 'EC')
    nx.set_node_attributes(graph, bet, 'BC')

    print('Feature Extraction Complete')

    return graph

def save_graph(graph, save_path):
    """Save a graph to disk"""
    # Using Pickle for now, h5py seems not to accept python objects

    nx.write_gpickle(graph, save_path)

    return 0

def clean_dataset(indir):
    """Generate clean datasets where all vertices have sequential ordering"""

    fileList = list()

    # Loop through all the files in the input dir and add to list
    for root, dirs, files in walk(indir):
        for f in files:
            if splitext(f)[1].lower() == ".txt" and "auto" not in splitext(f)[0].lower() and "10" not in splitext(f)[0].lower():
                fileList.append(root+f)

    # Loop through and load the graphs and save the cleaned graph's back to disk
    for f in fileList:
        clean_graph = read_graph(f)
        nx.write_edgelist(clean_graph, f, data=False)

def main(infile, outfile):

    graph = read_graph(infile)
    feat_graph = extract_graph_features(graph)
    save_graph(feat_graph, outfile)

if __name__ == "__main__":

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('indir', '../data', 'Input Directory (Will read all .txt files).')
    flags.DEFINE_string('outdir', '../data/features/', 'Output Directory.')

    # Setup the parallel env
    num_cores = multiprocessing.cpu_count()
    fileList = list()

    # Code to run separately to clean the datasets
    #clean_dataset(FLAGS.indir)

    # Loop through all the files in the input dir and add to list
    for root, dirs, files in walk(FLAGS.indir):
        for f in files:
            if splitext(f)[1].lower() == ".txt" and "auto" not in splitext(f)[0].lower() and "10" not in splitext(f)[0].lower():
                fileList.append(root+f)


    # Run the processing on the files   
    for f in fileList:

        main(f, join(FLAGS.outdir, f.strip('.txt') + '.ini.gz'))