# Implemetation of Paper named Tabu Search Implemetation of Simulatneous Scheduling of machine & AGV, IJPR 2014.
# Paper Author: Yan Zheng, Yujie Xiao & Yoonho Seo 
# Written by duyeon Kim 2021.04.30 
from instance_JSSPMH import getBenchmarkInstance
from utils import tabuList, DAG
import numpy as np 
import networkx as nx 
import copy
import random 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from collections import deque

class tabuSearch:
    def __init__(self, idx_layout=1, idx_job=1, num_AGV=2, ro=5, N_iter_fac=200):     
        self.idx_layout = idx_layout
        self.idx_job = idx_job   
        self.machine_alloc, self.processing_time, self.layout, self.num_AGV = getBenchmarkInstance(idx_layout, idx_job, num_AGV) # import instance of JSSPMH (Ulusoy 1995)
        self.jobNum = len(self.machine_alloc)
        self.machNum = len(self.layout) - 1 
        self.operation_count = len(np.concatenate(self.machine_alloc, axis=0))
        self.operation_id_matrix = self.genOperation_id_Matrix()
        self.K = self.operation_count # Tabu list size 
        self.ro = ro # penalty factor 
        self.Niter = N_iter_fac * self.operation_count # maximum number of iterations 
        self.tabu_list = tabuList(self.K) # tabu list
        
    # return operation id 2darray
    # operation id is (i,j): job i's jth operation. 
    def genOperation_id_Matrix(self): 
        operation_id_matrix = []
        for job in range(self.jobNum):
            job_list = []
            for op in range(len(self.machine_alloc[job])):
                job_list.append((job,op))
            operation_id_matrix.append(job_list)
        return operation_id_matrix

    # generate initial feasible pi(solution)
    def genInit_PI(self):
        pi_s = self.genInit_PI_S()
        pi_v = self.genInit_PI_V(pi_s)
        pi_init = [pi_s, pi_v]

        return pi_init

    # generate initial feasible operation pi 
    # To create a feasible pi_s, topological sort and random priority assignment (RPA) methods are adopted
    def genInit_PI_S(self):
        self.dag = DAG(self.operation_count) # directed acyclic graph 
        sort_TPA = self.adjust_Topological_Prior_Assignment() #  topological sort 
        sort_RPA = self.adjust_Random_Prior_Assignment() # random priority assignment (RPA)

        op_priority_dict = {opId:{'TPA':p, 'RPA':sort_RPA.pop()} for p, opId in enumerate(sort_TPA)}    
        sorted_op_priority_dict = sorted(op_priority_dict.items(), key=lambda x: (x[1]['TPA'], x[1]['RPA']))
        pi_s = [i[0] for i in sorted_op_priority_dict]

        return pi_s

    # generate initial feasible agv pi 
    # AGV assignment conforms to the nearest vehicle rule.
    def genInit_PI_V(self, pi_s):
        pi_v = [] 
        agv_loc_dict = {i:0 for i in range(self.num_AGV)}
        for i in range(self.num_AGV): 
            pi_v.append(i)
            agv_loc_dict[i] = self.machine_alloc[pi_s[i][0]][pi_s[i][1]]
        
        for idx, op in enumerate(pi_s[2:]):
            last_op = True if op[1] == len(self.machine_alloc[op[0]])-1 else False 
            if last_op: 
                pi_v.append(-1)
            else: 
                mach = self.machine_alloc[op[0]][op[1]]
                agv_distance_list = {}
                for agv_id, agv_loc in agv_loc_dict.items():
                    dist = self.layout[agv_loc][mach]
                    agv_distance_list[agv_id] = dist 
                
                if agv_distance_list[0] > agv_distance_list[1]:
                    near_agv = 1 
                else: 
                    near_agv = 0 
                pi_v.append(near_agv)
                agv_loc_dict[near_agv] = mach 

        return pi_v

    #  topological sort 
    def adjust_Topological_Prior_Assignment(self):
        for each_job in self.operation_id_matrix:
            for idx, j in enumerate(each_job):
                if idx == len(each_job)-1: 
                    self.dag.graph[j] = []
                else:    
                    self.dag.addEdge(j, each_job[idx+1])

        return self.dag.topologicalSort()

    # random priority assignment (RPA)
    def adjust_Random_Prior_Assignment(self):
        priority_list = [i for i in range(self.operation_count)]
        np.random.shuffle(priority_list)
        return priority_list

    # operation sequence-fixed neighbourhood generation (OSF) -> Changing pi_v
    def generate_OSF(self, pi):
        pi_s = pi[0]
        pi_v = pi[1]
        pi_neighbors_list = []
        for idx,val in enumerate(pi_v):
            neighbor_pi_v = copy.deepcopy(pi_v)
            """
            # AGV num >= 3 Case:
            other_AGVs = [i for i in range(self.num_AGV) if i!=val]
            random_select_AGV = np.random.choice(other_AGVs)
            self.pi_v[idx] = random_select_AGV # reallocate 
            """
            if val != -1:
                if val == 0: 
                    neighbor_pi_v[idx] = 1
                else: 
                    neighbor_pi_v[idx] = 0 

                pi_neighbors_list.append([pi_s, neighbor_pi_v])
        return pi_neighbors_list

    # vehicle assignment-fixed neighbourhood generation (VAF) -> Changing pi_s 
    def generate_VAF(self, pi):
        pi_s = pi[0]
        pi_v = pi[1]
        pi_neighbors_list = [] 
        # insertion method 
        for h, val in enumerate(pi_s):
            new_h = np.random.choice([i for i in range(len(pi_s)) if i!=h])
            
            neighbor_pi_s = copy.deepcopy(pi_s)
            neighbor_pi_v = copy.deepcopy(pi_v)

            opId = pi_s[h] 
            vId = pi_v[h]

            del neighbor_pi_s[h]
            del neighbor_pi_v[h]

            neighbor_pi_s.insert(new_h,opId) 
            neighbor_pi_v.insert(new_h,vId)

            neighbor_pi_s, neighbor_pi_v = self.takeRepairMechanism(h, new_h, opId, neighbor_pi_s, neighbor_pi_v)
            neighbor_pi = [neighbor_pi_s, neighbor_pi_v]
            pi_neighbors_list.append(neighbor_pi)

        return pi_neighbors_list

    def takeRepairMechanism(self, h, new_h, opId, pi_s, pi_v):
        # Step 1. Among the operations which are located between the original position and 
        # new position of the inserted operation i, find all operations belonging to J(i).
        same_job_op_list = [i for i in pi_s[h:new_h] if i[0] == opId[0]]
        # Step 2. Randomly find a valid position for each of those operations of J(i) (found in Step1) 
        # and insert them into the new positions one by one.
        # TODO: update also pi_v 
        if same_job_op_list:
            for op in same_job_op_list:
                v = pi_v[pi_s.index(op)]
                checked_pos_list = []
                while self.checkInfeasible(pi_s, op): 
                    h = pi_s.index(op) 
                    checked_pos_list.append(h)
                    random_pos = np.random.choice([idx for idx, val in enumerate(pi_s) if idx not in checked_pos_list])
                    checked_pos_list.append(random_pos)
                    del pi_s[h]
                    del pi_v[h]
                    pi_s.insert(random_pos, op)
                    pi_v.insert(random_pos, v)

        return pi_s, pi_v

    def checkInfeasible(self, pi_s, op):
        # check precedence constraint of opeartion(op) at pi_s 
        job_op_top_sort_list = [i for i in self.dag.topologicalSort() if i[0] == op[0]]
        job_op_pi_s_sort_list = [i for i in pi_s if i[0] == op[0]]
        idx_top = job_op_top_sort_list.index(op)
        #idx_pi = job_op_pi_s_sort_list.index(job_op_top_sort_list[0])
        idx_pi = job_op_pi_s_sort_list.index(op)
        job_op_pi_s_sort_filtered_list = [i for i in job_op_pi_s_sort_list[:idx_pi+1] if i[1]<=op[1]]
        if job_op_top_sort_list[:idx_top+1] != job_op_pi_s_sort_filtered_list:
            return True 
        return False 

    # run the tabu search 
    def solve(self):
        # step 1. 
        #lmf initialize 
        lmf = {v:{} for v in range(self.num_AGV)} 
        for v in range(self.num_AGV):
            for job in self.operation_id_matrix: 
                for op in job:
                    jobId = op[0]
                    if op[1] != len(self.machine_alloc[jobId])-1:
                        if op not in lmf[v].keys(): 
                            lmf[v][op] = 0
                        
        # step 2. create an initial feasible solution. calculate its makespan. 
        pi_init= self.genInit_PI()
        z_init = self.get_objective(pi_init)
        pi_curr = pi_init
        pi_best = pi_curr
        z_curr = z_init
        z_best = z_curr

        n_iter = 0 

        # step 3: generate the neighbour solutions 
        while n_iter < self.Niter: 
            if n_iter % 20 == 0:
                print("iter={}".format(n_iter))
            # 3.1 generate neighbour solutions of pi_curr by OSF. 
            neighbours_OSF_dict = {idx: {'sol':sol, 'z':0} for idx, sol in enumerate(self.generate_OSF(pi_curr))}
            # 3.1.1 calculate z_neigh of each neighbour solution. 
            for idx, val in neighbours_OSF_dict.items():
                pi_neigh = val['sol']
                z_neigh = self.get_objective(pi_neigh)
                neighbours_OSF_dict[idx]['z'] = z_neigh 

            # For neighbours with z_neigh > z_curr, set z_neigh = z_neigh + ro * lmf(v,i)
            for idx, val in neighbours_OSF_dict.items():
                pi_neigh = val['sol']
                z_neigh = val['z']
                if z_neigh > z_curr: 
                    move_idx = [idx for idx, v in enumerate(zip(pi_curr[1], pi_neigh[1])) if v[0]!=v[1]][0]
                    v = pi_neigh[1][move_idx] # vehicle 
                    i = pi_neigh[0][move_idx] # operation

                    z_neigh = z_neigh + self.ro * lmf[v][i]
                    val['z'] = z_neigh

            # 3.1.2 find pi_neigh*.
            pi_neigh_star = sorted(neighbours_OSF_dict.items(), key=lambda x: x[1]['z'])[0][1]['sol']
            z_neigh_star = self.get_objective(pi_neigh_star)

            move_idx = [idx for idx, v in enumerate(zip(pi_curr[1], pi_neigh_star[1])) if v[0]!=v[1]][0]
            v_star = pi_neigh_star[1][move_idx]
            i_star = pi_neigh_star[0][move_idx]
            
            # update the LMF 
            lmf[v_star][i_star] = lmf[v_star][i_star] + 1 

            pi_curr = pi_neigh_star
            z_curr = z_neigh_star
            
            if z_curr < z_best:
                pi_best = pi_curr
                z_best = z_curr
            
            # 3.2 generate neighbour solutions of pi_curr by VAF. 
            # 3.2.1 calculate z_neighbour of each neighbour solution 
            neighbours_VAF_dict = {idx: {'sol':sol, 'z':0} for idx, sol in enumerate(self.generate_VAF(pi_curr))}
            for idx, val in neighbours_VAF_dict.items():
                pi_neigh = val['sol']
                z_neigh = self.get_objective(pi_neigh)
                neighbours_VAF_dict[idx]['z'] = z_neigh

            # Neighbours which are not tabu solutions and tabu neighbours with 
            # objective function value Zneigh < Zbest consist of the admissible neighbour set.
            neighbours_admissible_list = []
            for idx, val in neighbours_VAF_dict.items():
                pi_neigh = val['sol']
                z_neigh = val['z']

                pi_s = pi_neigh[0]
                if  pi_s not in self.tabu_list: 
                    neighbours_admissible_list.append([idx, pi_neigh, z_neigh])
                if pi_s in self.tabu_list and z_neigh < z_best: 
                    neighbours_admissible_list.append([idx, pi_neigh, z_neigh])

            # 3.2.2 find pi_neigh_star in the admissible neighbour set. 
            _, pi_neigh_star, z_neigh_star = sorted(neighbours_admissible_list, key=lambda x: x[2])[0]
            pi_curr = pi_neigh_star
            z_curr = z_neigh_star
            if z_curr < z_best: 
                pi_best = pi_curr
                z_best = z_curr

            pi_s = pi_neigh_star[0]
            self.tabu_list.append(pi_s)
            
            n_iter = n_iter + 1 
        
        print("solved makespan is {}".format(z_best))
        return pi_best, z_best

    # return the objective(makespan) of schedule. 
    def get_objective(self, pi):
        machine_schedule, agv_schedule = self.get_schedule(pi)
        makespan = max([i[-1][-1] for i in machine_schedule.values()])

        return makespan

    # return the schedule of each resource(Machines, AGVs)
    def get_schedule(self, pi):
        pi_s = pi[0]
        pi_v = pi[1]
        job_info = {i:{'p':p,'m':m,'pf':9999,'lf':9999} for i,p,m in zip(sorted(pi_s), 
                                                                        np.concatenate(self.processing_time, axis=0),
                                                                        np.concatenate(self.machine_alloc, axis=0))}

        machine_schedule = {i:[] for i in range(1, len(self.layout))}
        agv_schedule = {i:[] for i in range(self.num_AGV)}

        # decode 
        for op, agv in zip(pi_s, pi_v):
            mach = job_info[op]['m']
            first_op = True if op[1] == 0 else False
            last_op = True if agv == -1 else False 

            if first_op:
                job_ready_t = 0 
            else:
                job_ready_t = job_info[op[0],op[1]-1]['lf']
            
            if not machine_schedule[mach]:
                mach_ready_t = 0 
            else: 
                mach_ready_t = machine_schedule[mach][-1][-1]
            
            # schedule: Process, Empty Trip, Load Trip 
            process_st = max(job_ready_t, mach_ready_t)
            process_et = process_st + job_info[op]['p'] 
            job_info[op]['pf'] = process_et
            
            if not last_op: 
                if not agv_schedule[agv]:  # agv location: LU stocker 
                    agv_loc = mach 
                    emptyTrip_st = 0
                else: 
                    agv_loc = agv_schedule[agv][-1][2]
                    emptyTrip_st = agv_schedule[agv][-1][-1]
                emptyTrip_et = emptyTrip_st + self.layout[agv_loc][mach]

                # schedule load trip task 
                next_mach = job_info[op[0],op[1]+1]['m'] 
                move_st = max(process_et, emptyTrip_et)
                move_et = move_st + self.layout[mach][next_mach]
                job_info[op]['lf'] = move_et

                agv_schedule[agv].append([(-1,-1), agv_loc, mach, emptyTrip_st, emptyTrip_et])
                agv_schedule[agv].append([op, mach, next_mach, move_st, move_et])
            machine_schedule[mach].append([op, process_st, process_et]) 

        
        return machine_schedule, agv_schedule 

    # return and save the gantt chart of schedule
    def draw_Gantt(self, z, machine_schedule, agv_schedule):
        import matplotlib.colors as mcolors
        color_dict = np.random.choice(list(mcolors.CSS4_COLORS.keys()),self.jobNum)
        job_color = {i:color_dict[i] for i in range(self.jobNum)}
        job_color[-1] = 'lightgrey'
        
        fig, gnt = plt.subplots(figsize=(30,10))
        
        gnt.set_title('gantt chart, layout={}, jobset={}, num_AGV={}, makespan={}'.format(self.idx_layout, self.idx_job, self.num_AGV, z))
        gnt.set_xlabel('time')
        gnt.set_ylabel('resources')

        gnt.set_xticks([i for i in range(100)])
        plt.xticks(rotation=45)
        gnt.set_yticks([25, 45, 65, 85, 105, 125])
        gnt.set_yticklabels(['agv1','agv2','m1', 'm2', 'm3', 'm4'])
        gnt.grid(True)

        for k,v in machine_schedule.items():
            gnt.broken_barh([(i[1], i[2]-i[1]) for i in v], (k*20+40, 9), facecolors=[job_color[i[0][0]] for i in v])
            
        for k,v in agv_schedule.items():
            gnt.broken_barh([(i[3], i[4]-i[3]) for i in v], ((k+1)*20, 9), facecolors=[job_color[i[0][0]] for i in v])
            for i in v:
                if i[1] == i[2]:
                    gnt.text(x=(i[3]+i[4])/2, y=k*20+25, s='*wait')
                else:
                    gnt.text(x=(i[3]+i[4])/2, y=k*20+25, s='{}->{}'.format(i[1], i[2]))
            
            
        patch_list = [] 
        for i in range(self.jobNum):
            patch = mpatches.Patch(color=job_color[i], label='job_'+str(i+1))
            patch_list.append(patch)

        patch = mpatches.Patch(color=job_color[-1], label='empty trip')
        patch_list.append(patch)

        gnt.legend(handles=patch_list)
        fig.savefig('./gantt.png')
    
    # check wheter last operation of each job is unassigned with AGV(=-1)
    def checkLastOpFeasible(self, pi):
        pi_s = pi[0]
        pi_v = pi[1] 

        last_op_set = [i[-1] for i in self.operation_id_matrix]
        
        for op in last_op_set:
            if pi_v[pi_s.index(op)] != -1:
                print("infeasible assn, op: {}".format(op))
        print("checked")
     

if __name__ == '__main__':
    # define instance condition 
    layout_id = 1 
    jobset_id = 1
    agv_num = 2 
    # make tabu search object
    ts = tabuSearch(layout_id,jobset_id,agv_num)

    # solve. returnt the solution and makespan(z)
    pi, z = ts.solve()
    machine_schedule, agv_schedule  = ts.get_schedule(pi)

    # result gantt chart 
    ts.draw_Gantt(z, machine_schedule, agv_schedule)

