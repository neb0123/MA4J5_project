import networkx as nx
import numpy as np
import collections
import random
import copy
import matplotlib.pyplot as plt

# First define the agents.
def nice_agent(M,Y,s):
    return (0,0)

def cheat_agent(M,Y,s):
    return (1,0)

def copy_agent(M,Y,s):
    if len(Y) == 0:
        return (0,0)
    else: 
        return (Y[-1],0) 

def grudge_agent(M,Y,s):
    if len(Y) == 0:
        return (0,0)
    elif s == 1:
        return (1,1)
    elif s == 0:
        if Y[-1] == 1:
            return (1,1)
        else:
            return (0,0)
    else:
        return (0,0)
    
def detective_agent(M,Y,s):
    if len(Y) == 0:
        return (0,1)
    elif s == 1:
        return (1,2)
    elif s == 2:
        return (0,3)
    elif s == 3:
        return (0,4)
    elif s == 6 or (1 in Y):
        return (copy_agent(M,Y,s)[0],6)
    else:
        return (1,0)
    
def random_agent(M,Y,s):
    r = np.random.ranf()
    if r < 0.5:
        return (0,0)
    else:
        return (1,0)

# Then we define 1 repeated game between 2 players P1 and P2.
# The payoff matrices are given by M1 and M2 and 'rounds' gives the # of rounds.
def play(M1,M2,P1,P2,rounds):
    P1_total = 0
    P2_total = 0
    P1_moves = []
    P2_moves = []
    P1_state = 0
    P2_state = 0
    for r in range(rounds):
        p11, p12 = P1(P1_moves,P2_moves,P1_state)
        p21, p22 = P2(P2_moves,P1_moves,P2_state)
        P1_move = p11
        P2_move = p21
        P1_total += M1[P1_move][P2_move]
        P2_total += M2[P2_move][P1_move]
        P1_state = p12
        P2_state = p22
        P1_moves.append(P1_move)
        P2_moves.append(P2_move)
    return P1_total,P2_total

# Define the first iteration of the full graph.
def tournament_first_round(P,M1,M2,rounds,m,type_of_graph):
    T = []
    for p in P:
        for i in range(P[p]):
            T.append(p)
            
    R = [0 for i in range(len(T))]
    
    G = make_graph(len(T),m,type_of_graph)
    E = [e for e in G.edges]
    
    for (i,j) in E:
        result = play(M1,M2,T[i],T[j],rounds)
        R[i] += result[0]
        R[j] += result[1]

    return T,R

# Here we define a procedure for making the game graph.
def make_graph(n,m,type_of_graph):
    if type_of_graph == nx.barabasi_albert_graph:
        G = type_of_graph(n,m)
        ran = range(n)
        random.shuffle(ran)
        G = nx.relabel_nodes(G,dict(zip(G.nodes(),ran)))
        return G
    elif type_of_graph == nx.watts_strogatz_graph:
        G = type_of_graph(n,m,0.03)
        ran = range(n)
        random.shuffle(ran)
        G = nx.relabel_nodes(G,dict(zip(G.nodes(),ran)))
        return G
    elif type_of_graph == nx.complete_graph or type_of_graph == nx.cycle_graph:
        G = type_of_graph(n)
        ran = range(n)
        random.shuffle(ran)
        G = nx.relabel_nodes(G,dict(zip(G.nodes(),ran)))
        return G

# Define a general iteration similar to the first iteration but without graph creation.
def tournament(T,M1,M2,rounds,m,type_of_graph):        
    R = [0 for i in range(len(T))]
    
    G = make_graph(len(T),m,type_of_graph)
    E = [e for e in G.edges]
    
    for (i,j) in E:
        result = play(M1,M2,T[i],T[j],rounds)
        R[i] += result[0]
        R[j] += result[1]

    return T,R

# This section makes the transition between iterations. i.e. birth/death of agents.
def next_round(T, R, deletions):
    if 2*deletions > len(T):
        raise ValueError('Deletes and additions overlap! Use more agents.')
    
    d = deletions
    while d != 0:
        mins = [i for i in range(len(R)) if R[i] == min(R)]
        l = len(mins)            
        if l >= d:
            S = np.random.choice(mins,d,replace=False)
            T = np.delete(T, S)
            R = np.delete(R, S)
            d -= d
        else:
            T = np.delete(T, mins)
            R = np.delete(R, mins)
            d -= l

    
    d = deletions
    T_tmp = copy.copy(T)
    R_tmp = copy.copy(R)
    while d != 0:
        maxs = [i for i in range(len(R_tmp)) if R_tmp[i] == max(R_tmp)]
        l = len(maxs)
        if l >= d:
            M = [T_tmp[i] for i in np.random.choice(maxs,d,replace=False)]
            T = np.concatenate([T,M])
            R_tmp = np.delete(R_tmp,maxs)
            d -= d
        else :
            M = [T_tmp[i] for i in maxs]
            T = np.concatenate([T,M])
            T_tmp = np.delete(T_tmp,maxs)
            R_tmp = np.delete(R_tmp,maxs)
            d -= l
    
    return T

# Put all the definitions together into one function.
def multiple_tournament(P,M1,M2,rounds,deletions,total_tournaments,m,type_of_graph):
    ALL_T = []
    T,R = tournament_first_round(P,M1,M2,rounds,m,type_of_graph)
    
    ALL_T.append(copy.copy(T))
    T = next_round(T, R, deletions)
    
    for i in range(total_tournaments-1):
        T,R = tournament(T,M1,M2,rounds,m,type_of_graph)
        ALL_T.append(copy.copy(T))
        T = next_round(T, R, deletions)
        
    return ALL_T

# A plotting function to plot # agents over iterations.
def plot_PD(A,D,iterations):
    copy_G=[]
    cheat_G=[]
    nice_G=[]
    detective_G=[]
    random_G=[]
    grudge_G=[]

    for i in range(iterations):    
        copy_G.append(sum([1 for a in A[i] if a == copy_agent]))
        cheat_G.append(sum([1 for a in A[i] if a == cheat_agent]))
        nice_G.append(sum([1 for a in A[i] if a == nice_agent]))  
        grudge_G.append(sum([1 for a in A[i] if a == grudge_agent]))
        detective_G.append(sum([1 for a in A[i] if a == detective_agent]))
        random_G.append(sum([1 for a in A[i] if a == random_agent]))

    who_plot = {copy_agent:0,cheat_agent:1,nice_agent:2,detective_agent:3,grudge_agent:4,random_agent:5}
    I = []
    for player in D:
        I.append(who_plot[player])

    if 0 in I:
        plt.plot(range(iterations), copy_G, label="Copy")
    if 1 in I:
        plt.plot(range(iterations), cheat_G, label="Cheat")
    if 2 in I:
        plt.plot(range(iterations), nice_G, label="Nice")
    if 3 in I:
        plt.plot(range(iterations), detective_G, label="Detective")
    if 4 in I:
        plt.plot(range(iterations), grudge_G, label="Grudge")
    if 5 in I:
        plt.plot(range(iterations), random_G, label="Random")
    plt.legend()