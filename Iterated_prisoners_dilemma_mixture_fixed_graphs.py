import networkx as nx
import numpy as np
import collections
import random
import copy
import matplotlib.pyplot as plt
import math
import seaborn

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
    
# Here we also define mixture agents:

# Takes as input a random number r in (0,1] and a state-tuple 
# [a,b,c,...] with a+b+c+.. = 1. it returns an integer from
# 1 to n, representing the agent it picked.
def which_agent(r, state):
    A = [nice_agent, cheat_agent, copy_agent, grudge_agent, detective_agent, random_agent]
    total = 0
    for i in range(len(state)):
        total += state[i]
        if r < total:
            agent = i
            break
    return A[agent]
    
# A mixture agent will have a state = ([a,b,c,..],s), an n-tuple
# where each entry represents the probability it is agent n.
# 0 - nice, 1 - cheat, 2 - copy, 3 - grudge, 4 - detective,
# 5 - random and then change their state to who they are. s is the
# the state of who they are impersonating.
def mix_agent(M,Y,s):
    if len(Y) == 0:
        r = np.random.ranf()
        who = which_agent(r,s[0])
        return (who(M,Y,s)[0],(who,who(M,Y,s)[1]))
    else:
        return (s[0](M,Y,s[1])[0],(s[0],s[0](M,Y,s[1])[1]))

# Then we define 1 repeated game between 2 players P1 and P2.
# The payoff matrices are given by M1 and M2 and 'rounds' gives the # of rounds.
def play(M1,M2,P1,P2,rounds,s1,s2):
    P1_total = 0
    P2_total = 0
    P1_moves = []
    P2_moves = []
    P1_state = s1
    P2_state = s2
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
def tournament_first_round(P,M1,M2,rounds,m,type_of_graph,draw_graph):
    T = []
    for p in P:
        for i in range(P[p]):
            T.append(p)
            
    R = [0 for i in range(len(T))]
    
    G = make_graph(len(T),m,type_of_graph)
    E = [e for e in G.edges]
    
    for (i,j) in E:
        result = play(M1,M2,T[i],T[j],rounds,0,0)
        R[i] += result[0]
        R[j] += result[1]

    if draw_graph[0]:
        plt.figure(figsize=draw_graph[1])
        cMap = []
        for node in G:
            if T[node]==nice_agent:
                cMap.append('green')
            elif T[node]==cheat_agent:
                cMap.append('yellow')
            elif T[node]==copy_agent:
                cMap.append('skyblue')
            elif T[node]==mix_agent:
                cMap.append('red')
            elif T[node]==random_agent:
                cMap.append('white')
            elif T[node]==grudge_agent:
                cMap.append('grey')
            elif T[node]==detective_agent:
                cMap.append('brown')
        nx.draw(G,pos=nx.layout.spectral_layout(G),node_color=cMap,labels=dict(zip(range(len(R)),R)),font_size=15,node_size=600)
        
    return T,R,G

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
    
# Return m unique elements from seq.
def _random_subset(seq,m):
    targets=set()
    while len(targets)<m:
        x=random.choice(seq)
        targets.add(x)
    return targets

# Performs an update to the graph based on preferential attatchment.
def update_graph_BA(G_tmp,D_tmp,m):
    G = G_tmp.copy()
    D = copy.copy(D_tmp)
    
    # Remove all nodes v in D
    for v in D:
        G.remove_node(v)

    for v in D:
        E_repeat = [i for e in G.edges for i in e] # Add nodes based on preferential attatchment
        G.add_node(v) # Re-include node v
        E_new = [(v,u) for u in _random_subset(E_repeat,m)] 
        G.add_edges_from(E_new) # Add random m edges based on pref.attatchment.
        
    return G

# Define a general iteration similar to the first iteration but without graph creation.       
def tournament(T,M1,M2,rounds,m,SS,G,type_of_graph,draw_graph):     
    R = [0 for i in range(len(T))]

    E = [e for e in G.edges]
    
    for (i,j) in E:
        
        # set states = 0
        s1, s2 = 0, 0
        # If we have mix agents change starting state
        if SS[i] != 0:
            s1 = SS[i]
        if SS[j] != 0:
            s2 = SS[j]
        
        result = play(M1,M2,T[i],T[j],rounds,(s1,0),(s2,0))
        R[i] += result[0]
        R[j] += result[1]
        
    if draw_graph[0]:
        plt.figure(figsize=draw_graph[1])
        cMap = []
        for node in G:
            if T[node]==nice_agent:
                cMap.append('green')
            elif T[node]==cheat_agent:
                cMap.append('yellow')
            elif T[node]==copy_agent:
                cMap.append('skyblue')
            elif T[node]==mix_agent:
                cMap.append('red')
            elif T[node]==random_agent:
                cMap.append('white')
            elif T[node]==grudge_agent:
                cMap.append('grey')
            elif T[node]==detective_agent:
                cMap.append('brown')
        nx.draw(G,pos=nx.layout.spectral_layout(G),node_color=cMap,labels=dict(zip(range(len(R)),R)),font_size=15,node_size=600)

    return T,R

# This section makes the transition between iterations. i.e. birth/death of agents.
def next_round(T, R, deletions,SS,G,m,update):
    if 2*deletions > len(T):
        raise ValueError('Deletes and additions overlap! Use more agents.')
         
    SS_best = [] 
    
    # D is the set of indexs of T/R that we will replace (Delete) with new agents.
    D = np.array([])
    
    d = deletions
    T_tmp = copy.copy(T)
    R_tmp = copy.copy(R)
    while d != 0:
        mins = [i for i in range(len(R_tmp)) if R_tmp[i] == min(R_tmp)]
        M = max(R)+1 # The biggest element to turn mins into after used.
        l = len(mins)            
        if l >= d:
            S = np.random.choice(mins,d,replace=False)
            D = np.concatenate([D,S])
            d -= d
        else:
            D = np.concatenate([D,mins])
            for i in mins:
                R_tmp[i] = M
            d -= l
    
    # Convert floats to ints:
    D = [int(d) for d in D]
    # Give D a shuffle:
    random.shuffle(D)
    
    # Here we collect an array Best, of the top 'deletion' number of people:
    Best = np.array([])
    d = deletions
    T_tmp = copy.copy(T)
    R_tmp = copy.copy(R)
    while d != 0:
        maxs = [i for i in range(len(R_tmp)) if R_tmp[i] == max(R_tmp)]
        l = len(maxs)
        if l >= d:
            X = np.random.choice(maxs,d,replace=False)
            M = []
            for i in X:
                if T_tmp[i]!=mix_agent:
                    M.append(T_tmp[i])
                else:
                    SS_best.append(i)
            Best = np.concatenate([Best,M])
            d -= d
        else :
            M = [T_tmp[i] for i in maxs if T_tmp[i]!=mix_agent]
            k=0
            for i in maxs:
                if T_tmp[i]==mix_agent:
                    SS_best.append(i)
                    k+=1
            
            Best = np.concatenate([Best,M])
            for i in maxs:
                T_tmp[i] = 0
            for i in maxs:
                R_tmp[i] = min(R_tmp)
            d -= l
            
    # Add in 'deletion' lots of new mix players at the indexs fiven by D:
    for i in D:
        T[i] = mix_agent
    
    # Calculate the amount contributed by best mix agents.
    Total = [0,0,0,0,0,0]
    
    for i in SS_best:
        for j in range(len(Total)):
                    Total[j] += SS[i][j]
                
    # Now create the mix array Mix of probabilities based on Best:
    Mix = [0,0,0,0,0,0]
    Mix[0] += (float(sum([1 for a in Best if a == nice_agent]))+Total[0]) / deletions
    Mix[1] += (float(sum([1 for a in Best if a == cheat_agent]))+Total[1]) / deletions
    Mix[2] += (float(sum([1 for a in Best if a == copy_agent]))+Total[2]) / deletions
    Mix[3] += (float(sum([1 for a in Best if a == grudge_agent]))+Total[3]) / deletions
    Mix[4] += (float(sum([1 for a in Best if a == detective_agent]))+Total[4]) / deletions
    Mix[5] += (float(sum([1 for a in Best if a == random_agent]))+Total[5]) / deletions
    
    for i in D:
        SS[i] = Mix
    
    # Finally we update the graph 
    if update == 0:
        None
    elif update == 1:
        G = update_graph_BA(G,D,m)
    
    return T,SS,G

# Put all the definitions together into one function.
def multiple_tournament(P,M1,M2,rounds,deletions,total_tournaments,m,type_of_graph,draw_graph,update):
    ALL_T = []
    T,R,G = tournament_first_round(P,M1,M2,rounds,m,type_of_graph,draw_graph)
    SS = [0 for i in range(len(T))]
    
    ALL_T.append(copy.copy(T))
    T,M,G = next_round(T, R, deletions,SS,G,m,update) # Next round array of players and mix array
    
    for i in range(total_tournaments-1):
        T,R = tournament(T,M1,M2,rounds,m,SS,G,type_of_graph,draw_graph)
        ALL_T.append(copy.copy(T))
        T,SS,G = next_round(T, R, deletions,SS,G,m,update) # Next round array of players and mix array
        
    return ALL_T, SS

# A plotting function to plot # agents over iterations.
def plot_PD(A,D,iterations):
    copy_G=[]
    cheat_G=[]
    nice_G=[]
    detective_G=[]
    random_G=[]
    grudge_G=[]
    mix_G=[]

    for i in range(iterations):    
        copy_G.append(sum([1 for a in A[i] if a == copy_agent]))
        cheat_G.append(sum([1 for a in A[i] if a == cheat_agent]))
        nice_G.append(sum([1 for a in A[i] if a == nice_agent]))  
        grudge_G.append(sum([1 for a in A[i] if a == grudge_agent]))
        detective_G.append(sum([1 for a in A[i] if a == detective_agent]))
        random_G.append(sum([1 for a in A[i] if a == random_agent]))
        mix_G.append(sum([1 for a in A[i] if a == mix_agent]))

    who_plot = {copy_agent:0,cheat_agent:1,nice_agent:2,detective_agent:3,grudge_agent:4,random_agent:5}
    I = []
    for player in D:
        I.append(who_plot[player])

    plt.figure()
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
    plt.plot(range(iterations), mix_G, label="Mix")
    plt.legend()

# A plotting function to plot the "mixture barcode"
def plot_barcode(SS):
    plt.figure()
    SS0 = [i for i in SS if i!=0]
    seaborn.heatmap(np.array(SS0),cmap='binary')
