#Comment in case we dont need to import the packages
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random 
#import sys #I hope we can use the sys package, because otherwise this doesnt work


def Get_costs(ID, C_behav):     
    if (ID==45):
        #        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope,  ca, thal
        Costs =   [0,   0, 71,       20,    5,  30,      73,      10,    75,      72,    77, 100,   40] #ID = 45
            
        if (C_behav=='simple'):
            Bundles = []
        elif (C_behav=='complex'):
            Bundles = [0,0,  1,       2,    3,   2,       1,       4,     1,       1,     1,   5,    2]
        else: 
            print('you spelled something wrong in C_behav')
            
        return Costs, Bundles
    
    #do the other IDs later      
        
##################################################################################################################

def Model_model(Data_vals, Data_cols, y, alpha, ID=-1, Budget = 0, M_behav = 'same', C_behav = 'simple', Un_norm = np.array([]) , focus = 0 ):
    ''' Creates the MIP, optimizes it and gives you the resulting ruleset.
    ----------------
    Data_vals: Array with X data values
    
    Data_cols: Column names for X
    
    y:       List with obj data
    
    alpha:   Maximum number of bounds allowed
    
    ID:      The database id
    
    Budget:  Maximum budget allowed
    
    M_behav: same       --> M is computed at the begining (making it very large) and all Ms are the same
             diff       --> M is computed per feature depending on a_j (option disabled)
             
    c_behav: simple     --> Each feature has independent costs
             complex    --> Some features get discounted if selected together
             
    Un_norm: array outputed by the data prep function if you decide to normalize data; needed to turn it back to original scale
    
    focus:   Changes the MIPfocus, from 0 to 3, each modifies the branch&cut strategy for Gurobi. 
    
    -----------------
    Outputs: 
    The model variables that encode the bounds (a, b, s), the rectangles (r), and maybe more if you want
    The ruleset as a code-ready string 
    The time it took and the objective it reached'''
    
    
    #define some parameters
    N = len(Data_vals)    #number of points
    d = len(Data_cols)    #number of features
    m = 2*alpha +1        #small m for the inequality-related indicators
    if (Budget != 0):
        if (ID == -1):
            Costs = [1]*d
        else:
            Costs, Bundles = Get_costs(ID, C_behav)
            print(Costs)

    if (M_behav == 'same'):
        M= 2*Data_vals.max() + 2      
    
    Data_vals = np.asfarray(Data_vals) #Change all columns to float and find the precision of each column   
     
    e = []
    for i in range(len(Data_cols)):
        buffer = []
        for j in range(len(Data_vals)):
            buffer += [len(str(Data_vals[j][i]).split('.')[1])]
        e+=[10**-(max(buffer)+1)]
   
        
    #separate indices of y= 1s and 0s 
    Y_pos_index = np.array([i for i in range(len(y))])[[(i==1) for i in y]]
    Y_neg_index = np.array([i for i in range(len(y))])[[(i==0) for i in y]]
    
    #create the model
    Model_1 = gp.Model('model1')

    b = Model_1.addMVar((alpha), lb=0, ub=[float('inf')]*alpha, vtype=GRB.CONTINUOUS, name='b')    #b_(bnd)
    a = Model_1.addMVar((alpha,d), vtype=GRB.BINARY, name='a')                                     #a_(bnd, dim)
    s = Model_1.addMVar((alpha), vtype=GRB.BINARY, name='s')                                       #s_(bnd)
    z = Model_1.addMVar((N, alpha), vtype=GRB.BINARY, name='z')                                    #z_(point, bnd)
    r = Model_1.addMVar((alpha, alpha), vtype=GRB.BINARY, name='r')                                #r_(rec, bnd)
    w = Model_1.addMVar((N, alpha), vtype=GRB.BINARY, name='w')                                    #w_(point, rec)
    h = Model_1.addMVar((N, alpha, alpha), vtype=GRB.BINARY, name='h')                             #h_(point, rec, bnd)
    g = Model_1.addMVar((alpha), vtype=GRB.BINARY, name='g')                                       #g_(rec)
    c = Model_1.addMVar((N), vtype=GRB.BINARY, name='c')                                           #c_(point)

    
    #Model_1.addConstrs( (b[j] == 0.5 for j in range(alpha)), name='false_b_con')

    
    Model_1.addConstrs( (sum(a[j]) == 1 for j in range(alpha)), name='a_con')

    Model_1.addConstrs( (sum([r[l][j] for l in range(alpha)]) <= 1 for j in range(alpha)), name='r_con1')

    Model_1.addConstrs( (r[l][j]       >= h[i][l][j]           for i in range(N) for j in range(alpha) for l in range(alpha)), name='h_con1')
    Model_1.addConstrs( (z[i][j]       >= h[i][l][j]           for i in range(N) for j in range(alpha) for l in range(alpha)), name='h_con2')
    Model_1.addConstrs( (h[i][l][j] +1 >= r[l][j] + z[i][j]    for i in range(N) for j in range(alpha) for l in range(alpha)), name='h_con3')

    Model_1.addConstrs( (sum(r[l]) <= g[l]*alpha   for l in range(alpha)), name='g_conr')
    Model_1.addConstrs( (sum(r[l]) >= g[l]         for l in range(alpha)), name='g_conl')

    Model_1.addConstrs( (g[l] >= w[i][l]           for i in Y_neg_index for l in range(alpha)), name='w_con1')
    Model_1.addConstrs( (g[l] >= 1-w[i][l]         for i in Y_pos_index for l in range(alpha)), name='w_con1')
    
    Model_1.addConstrs( (sum(r[l])-sum(h[i][l]) >= w[i][l]           - m*(1-g[l])   for i in Y_pos_index for l in range(alpha)), name='w_con2l')
    Model_1.addConstrs( (sum(r[l])-sum(h[i][l]) <= w[i][l]*alpha     + m*(1-g[l])   for i in Y_pos_index for l in range(alpha)), name='w_con2r')
    Model_1.addConstrs( (sum(h[i][l])           >= (1-w[i][l])       - m*(1-g[l])   for i in Y_neg_index for l in range(alpha)), name='w_con3l')
    Model_1.addConstrs( (sum(h[i][l])           <= (1-w[i][l])*alpha + m*(1-g[l])   for i in Y_neg_index for l in range(alpha)), name='w_con3r')
    
    Model_1.addConstrs( (alpha - sum(w[i]) <= alpha*(1-c[i])   for i in Y_pos_index ), name='c_con1r')
    Model_1.addConstrs( (alpha - sum(w[i]) >= 1-c[i]           for i in Y_pos_index ), name='c_con1l')
    Model_1.addConstrs( (sum(w[i])         <= alpha*c[i]       for i in Y_neg_index ), name='c_con2r')
    Model_1.addConstrs( (sum(w[i])         >= c[i]             for i in Y_neg_index ), name='c_con2l')
    
    if (M_behav=='same'):
        #                     b   -  ...M(1-z)... -  ...2M(s)... +          ........e........           <=         ............ax............
        Model_1.addConstrs( (b[j] - M*(1-z[i][j]) - 2*M*(s[j])   + sum(a[j][k]*e[k] for k in range(d))  <= sum(a[j][k]*Data_vals[i][k] for k in range(d))        for i in Y_neg_index for j in range(alpha)), name='z_con1l')
        Model_1.addConstrs( (b[j] - M*z[i][j]     - 2*M*(1-s[j])                                        <= sum(a[j][k]*Data_vals[i][k] for k in range(d))        for i in Y_neg_index for j in range(alpha)), name='z_con2l')
        Model_1.addConstrs( (b[j] - M*z[i][j]     - 2*M*(s[j])   + sum(a[j][k]*e[k] for k in range(d))  <= sum(a[j][k]*Data_vals[i][k] for k in range(d))        for i in Y_pos_index for j in range(alpha)), name='z_con3l')
        Model_1.addConstrs( (b[j] - M*(1-z[i][j]) - 2*M*(1-s[j])                                        <= sum(a[j][k]*Data_vals[i][k] for k in range(d))        for i in Y_pos_index for j in range(alpha)), name='z_con4l')
        
        Model_1.addConstrs( (sum(a[j][k]*Data_vals[i][k] for k in range(d)) <= b[j] + M*z[i][j]     + 2*M*(s[j])                                                 for i in Y_neg_index for j in range(alpha)), name='z_con1r')
        Model_1.addConstrs( (sum(a[j][k]*Data_vals[i][k] for k in range(d)) <= b[j] + M*(1-z[i][j]) + 2*M*(1-s[j]) - sum(a[j][k]*e[k] for k in range(d))         for i in Y_neg_index for j in range(alpha)), name='z_con2r')
        Model_1.addConstrs( (sum(a[j][k]*Data_vals[i][k] for k in range(d)) <= b[j] + M*(1-z[i][j]) + 2*M*(s[j])                                                 for i in Y_pos_index for j in range(alpha)), name='z_con3r')
        Model_1.addConstrs( (sum(a[j][k]*Data_vals[i][k] for k in range(d)) <= b[j] + M*z[i][j]     + 2*M*(1-s[j]) - sum(a[j][k]*e[k] for k in range(d))         for i in Y_pos_index for j in range(alpha)), name='z_con4r')
    
    if (Budget != 0):
        q = Model_1.addMVar((d), vtype=GRB.BINARY, name='q')                                       #q_(dim)
        Model_1.addConstrs( (    sum([a[j][k] for j in range(alpha)])      <= alpha*q[k]       for k in range(d) ), name='q_con1r')
        Model_1.addConstrs( (    sum([a[j][k] for j in range(alpha)])      >= q[k]             for k in range(d) ), name='q_con1l')
        
        if (C_behav=='simple'):
            print('simple budget constrs ativated')
            Model_1.addConstr(  sum( q[k]*Costs[k] for k in range(d)) <= Budget , name='Budget_con')
        
        elif (C_behav=='complex'):
            print('complex budget constrs ativated')
            num_bundl = max(Bundles)+1                  #number of distinct bundles
            Index = np.array([i for i in range(d)])     #Column indices
            bundl_sets = []                             #Will store the column indices that belong to the same bundle
            gammas = []                                 #will store the shared cost of each bundle set
            
            for beta in range(num_bundl): #separates the tests in bundles
                bundl_sets += [list(Index[[Bundles[i] == beta for i in range(d)]])]

            for sett in bundl_sets: #gets the common cost and updates the Costs to reflect only the extra charge
                costs_in_set = []
                for ind in sett:
                    costs_in_set += [Costs[ind]]
                gamma = min(costs_in_set)
                gammas += [gamma]
                for ind in sett:
                    Costs[ind] -= gamma
                    
            beta = Model_1.addMVar((num_bundl), vtype=GRB.BINARY, name='beta')                  #beta_(#OfBundles)
            
            Model_1.addConstrs( (    sum([q[k] for k in bundl_sets[set_ind]])      <= len(bundl_sets[set_ind])*beta[set_ind]       for set_ind in range(num_bundl) ), name='beta_con1r')
            Model_1.addConstrs( (    sum([q[k] for k in bundl_sets[set_ind]])      >= beta[set_ind]                                for set_ind in range(num_bundl) ), name='beta_con1l')
            Model_1.addConstr(  sum( q[k]*Costs[k] for k in range(d)) + sum(beta[s]*gammas[s] for s in range(num_bundl)) <= Budget , name='Budget_con')
        else:
            print('something went wrong in the budget constraints')
            

    
    #Parameters:
    Model_1.Params.MIPFocus = focus
    Model_1.params.MIPGap = 0.10
    #Model_1.Params.timelimit = 7200.0
    Model_1.Params.DisplayInterval = 120
    #Model_1.Params.OutputFlag = 0
    
    Model_1.setObjective(sum(1-c[i] for i in range(N)), GRB.MAXIMIZE)
    
    print('About to start optimizing: N'+str(len(Data_vals))+'_A'+str(alpha)+'_B'+str(Budget)+' ' +C_behav +'_F'+str(focus))
    #compute the solution
    Model_1.optimize()
    #Model_1.write("debug2.lp")
    print('Done optimizing: N'+str(len(Data_vals))+'_A'+str(alpha)+'_B'+str(Budget)+' ' +C_behav +'_F'+str(focus))
    #print the score and the ruleset
    print('---SOLUTION---')
    print('Objective: %g' % Model_1.ObjVal)
    
    
    
    print('---Ruleset---')
    
    or_cntr = 0
    and_cntrs = [0]*alpha
    
    for l in range(alpha):              #On every rectangle
        if g[l].X>0.5:                  #check if non-empty
            or_cntr += 1                #count how many
            for j in range(alpha):      #On each of those rectangles 
                if r[l][j].X>0.5:       #count how many bounds
                    and_cntrs[l] += 1
            
    ruleset_list = '' 
    for l in range(alpha):
        if g[l].X>0.5:                  #on every not empty rectangle
            ruleset_list += '(' 
            for j in range(alpha):
                if r[l][j].X>0.5:       #on every bound where r=1 in this rectangle  
                    for k in range(d):
                        if a[j][k].X >0.5:       #find feature
                            feat = Data_cols[k]
                            
                    #select bound
                    if (not Un_norm.size == 0):                                             #if we need to un-normalize
                        bd = b[j].X*Un_norm[0][k] + Un_norm[1][k]
                    else:
                        bd = b[j].X
                        
                    if s[j].X>0.5:                                                     #find direction
                        print(str(feat) + ' >= ' + str(bd) + ' & ')                    #print inequality
                        ruleset_list += '(' + str(feat) + ' >= ' + str(bd) + ')'
                    elif s[j].X<0.5:
                        print(str(feat) + ' <= ' + str(bd) + ' & ')
                        ruleset_list += '(' + str(feat) + ' <= ' + str(bd) + ')'
                        
                    if (and_cntrs[l]-1 > 0):
                        ruleset_list += '&'
                        and_cntrs[l] += -1
                    
            print('OR')
            ruleset_list += ')'
            if (or_cntr-1 > 0):
                ruleset_list += '  |  '
                or_cntr += -1
      
    #Model_1.write("debug2.lp")
    
    vvars = []
    for v in Model_1.getVars():
        if str(v.varname)[0] in ['a', 'b', 's', 'r', 'z']:
            vvars += (v.varname, v.X)

    name_of_file = '_N'+str(len(Data_vals))+'_A'+str(alpha)+'_B'+str(Budget)+' ' +C_behav +'_F'+str(focus)+'.npy'
    
    with open('Results/Ruleset'+ name_of_file, 'wb') as f:
        np.save(f, ruleset_list)
    with open('Results/TimeObj'+ name_of_file, 'wb') as f:
        np.save(f, np.array([Model_1.Runtime, Model_1.ObjVal]) )
    with open('Results/Vars'+ name_of_file, 'wb') as f:
        np.save(f, vvars)

##################################################################################################################

def Random_data_gen(length, dim):
    '''length = number of points; dim = number of features'''

    points = np.random.rand(length,dim)
    for i in range(length):
        for j in range(dim):
            points[i][j] = round(points[i][j], 3)
    val = np.random.randint(2, size=(length))
    return points, val

##################################################################################################################

#Read sys inputs
#Read sys inputs
#length = int(sys.argv[1])  #-1 for ID=45, any other number for randomly generated data of that size
#alpha =  int(sys.argv[2])  #number of inequs
#Budgt =  int(sys.argv[3])  #0 to disable budget constraints, any other number to use them
#B_behav= sys.argv[4]       #'simple' or 'complex' for both behaviours
#Focs =   int(sys.argv[5])  #from 0 to 4

length = 60
#alpha =  1
Budgt =  0
B_behav= 'simple'     
Focs =   3 
'''
if (length==-1): #If selected heard disease data, do it with inputed variables
    IDD=45
    with open('Data/Data_vals_id_'+str(IDD)+'.npy', 'rb') as f:
        Data_vals = np.load(f,allow_pickle=True)
    with open('Data/Data_cols_id_'+str(IDD)+'.npy', 'rb') as f:
        Data_cols = np.load(f,allow_pickle=True)
    with open('Data/y_id_'+str(IDD)+'.npy', 'rb') as f:
        y = np.load(f,allow_pickle=True)
    with open('Data/un_norm_vals_id_'+str(IDD)+'.npy', 'rb') as f:
        un_norm_vals = np.load(f,allow_pickle=True)
    
    Model_model(Data_vals, Data_cols, y, alpha, ID=IDD, Un_norm = un_norm_vals ,Budget = Budgt, C_behav = B_behav,  focus = Focs )

else: #if we want random data, (to compare MIP focuses) then we need to use the same data for all 4 tests, sooooo we have to do them back to back
    IDD=-1
    Data_vals, y = Random_data_gen(length, 3)
    Data_cols = ['x'*(i+1) for i in range(3)]
    un_norm_vals = np.array([])

    Model_model(Data_vals, Data_cols, y, alpha, ID=IDD, Un_norm = un_norm_vals ,Budget = Budgt, C_behav = B_behav,  focus = 0 )
    Model_model(Data_vals, Data_cols, y, alpha, ID=IDD, Un_norm = un_norm_vals ,Budget = Budgt, C_behav = B_behav,  focus = 1 )
    Model_model(Data_vals, Data_cols, y, alpha, ID=IDD, Un_norm = un_norm_vals ,Budget = Budgt, C_behav = B_behav,  focus = 2 )
    Model_model(Data_vals, Data_cols, y, alpha, ID=IDD, Un_norm = un_norm_vals ,Budget = Budgt, C_behav = B_behav,  focus = 3 )
'''


Data_vals, y = Random_data_gen(length, 10)
Data_cols = ['x'*(i+1) for i in range(10)]
for alpha in range(1,11):
    Model_model(Data_vals, Data_cols, y, alpha,  focus = 3 )


