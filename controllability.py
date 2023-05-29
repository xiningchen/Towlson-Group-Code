import networkx as nx
import scipy.linalg as la
import numpy as np
import control

def modal_control(G, stabilize=True):
    """
    # Modal controllability based on code from Basset
    # https://complexsystemsupenn.com/codedata
    :param G: numpy array connectome
    :param stabilize: stabilize by ensuring largest e-value is 1.
    :return: modal controllability for each node
    """
    A = nx.to_numpy_array(G)    # Order in G.nodes()
    if stabilize:
        # Matrix norm
        normA = la.svdvals(A)[0] + 1
        A = A / normA
    # Schur decomp - stability
    T, U = la.schur(A, output='real')
    eVals = np.diag(T)
    N = len(eVals)
    phi = np.empty([N, ])
    for i in range(0, N):
        phi[i] = np.dot(U[i, :]**2, (1 - eVals**2))

    nodeList = G.nodes()
    modalCtrbDict = {}
    for i, n in enumerate(nodeList):
        modalCtrbDict[n] = phi[i]
    return modalCtrbDict


def avg_control(G, stabilize=True):
    """
    Average controllability based on code from Basset
    https://complexsystemsupenn.com/codedata
    :param G: numpy array connectome
    :param stabilize: stabilize by ensuring largest e-value is 1.
    :return: average controllability for each node
    """
    A = nx.to_numpy_array(G)
    if stabilize:
        normA = la.svdvals(A)[0] + 1
        A = A / normA
    T, U = la.schur(A, output='real')
    eVals = np.diag(T)
    eVals.shape = [len(A[0,:]), 1]
    midMat = (U**2).transpose()          # ******** is U guaranteed to be real?
    P = np.tile(1-eVals**2, (1, len(A[0,:])))
    res = sum(midMat/P)     # row vector

    nodeList = G.nodes()
    ac = {}
    for i, n in enumerate(nodeList):
        ac[n] = res[i]
    return ac


def avg_control2(G):
    """
    Average controllability based on new_ave_control.m - from Emma Towlson.
    https://python-control.readthedocs.io/en/latest/control.html#system-creation
    :param G:
    :return:
    """
    A = nx.to_numpy_array(G)
    normA = la.svdvals(A)[0] + 1
    A = A / normA
    n = A.shape[0]
    nodeList = G.nodes()
    ac = {}
    # Calculate for each node i
    for i, node in enumerate(nodeList):
        Bi = np.zeros((n, 1))
        Bi[i] = 1
        sys = control.ss(A, Bi, np.identity(n), [])     # make a discrete time system
        Wc = control.gram(sys, 'c')                     # calculate the controllability grammian
        ac[node] = np.trace(Wc)                            # calculate trace
    return ac