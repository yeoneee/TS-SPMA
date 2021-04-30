import numpy as np 
import random 

def getBenchmarkInstance(indexLayout, indexJobset, agv_num):
    # [from][to], [0][0]: L/U -> L/U
    if indexLayout == 1:
        layout = [
            [0, 6, 8, 10, 12], 
            [12, 0, 6, 8, 10], 
            [10, 6, 0, 6, 8],
            [8, 8, 6, 0, 6], 
            [6, 10, 8, 6, 0]
        ]
    elif indexLayout == 2:
        layout = [
            [0, 4, 6, 8, 6],
            [6, 0, 2, 4, 2],
            [8, 12, 0, 2, 4],
            [6, 10, 12, 0, 2],
            [4, 8, 10, 12, 0]
        ]

    elif indexLayout == 3:
        layout = [
            [0,2,4,10,12],
            [12,0,2,8,10],
            [10,12,0,6,8],
            [4,6,8,0,2],
            [2,4,6,12,0]
        ]
    elif indexLayout == 4:
        layout = [
            [0,4,8,10,14],
            [18,0,4,6,10],
            [20,14,0,8,6],
            [12,8,6,0,6],
            [14,14,12,6,0]
        ]

    # [jobIndex][machineOperIndex][0=machineIndex, 1=procTime]
    if indexJobset == 1:
        jobset = [
            [[1,8], [2,16],[4,12]], 
            [[1,20],[3,10],[2,18]], 
            [[3,12],[4,8], [1,15]],
            [[4,14],[2,18]],
            [[3,10],[1,15]]
        ]
    elif indexJobset == 2:
        jobset = [
            [[1,10],[4,18]], 
            [[2,10],[4,18]], 
            [[1,10],[3,20]], 
            [[2,10],[3,15],[4,12]], 
            [[1,10],[2,15],[4,12]], 
            [[1,10],[2,15],[3,12]]
            ]
    elif indexJobset == 3:
        jobset = [
            [[1,16],[3,15]], 
            [[2,18],[4,15]], 
            [[1,20],[2,10]], 
            [[3,15],[4,10]], 
            [[1,8], [2,10],[3,15],[4,17]], 
            [[2,10],[3,15],[4,8],[1,15]]]

    elif indexJobset == 4:
        jobset = [
            [[4,11],[1,10],[2,7]], 
            [[3,12],[2,10],[4,8]], 
            [[2,7], [3,10],[1,9], [3,8]], 
            [[2,7], [4,8], [1,12],[2,6]], 
            [[1,9], [2,7], [4,8], [2,10], [3,8]]
            ]
    elif indexJobset == 5:
        jobset = [
            [[1,6], [2,12],[4,9]], 
            [[1,18],[3,6], [2,15]], 
            [[3,9], [4,3], [1,12]], 
            [[4,6], [2,15]], 
            [[3,3], [1,9]]
            ]
    elif indexJobset == 6:
        jobset = [
            [[1,9], [2,11],[4,7]], 
            [[1,19],[2,20],[4,13]], 
            [[2,14],[3,20],[4,9]], 
            [[2,14],[3,20],[4,9]], 
            [[1,11],[3,16],[4,8]], 
            [[1,10],[3,12],[4,10]]
            ]
    elif indexJobset == 7:
        jobset = [
            [[1,6], [4,6]], 
            [[2,11],[4,9]], 
            [[2,9], [4,7]], 
            [[3,16],[4,7]], 
            [[1,9], [3,18]], 
            [[2,13],[3,19],[4,6]], 
            [[1,10],[2,9], [3,13]],
            [[1,11],[2,9], [4,8]]
        ]

    elif indexJobset == 8:
        jobset = [
            [[2,12],[3,21],[4,11]], 
            [[2,12],[3,21],[4,11]], 
            [[2,12],[3,21],[4,11]], 
            [[2,12],[3,21],[4,11]], 
            [[1,10],[2,14],[3,18],[4,9]], 
            [[1,10],[2,14],[3,18],[4,9]]]

    elif indexJobset == 9:
        jobset = [
            [[3,9], [1,12],[2,9], [4,6]], 
            [[3,16],[2,11],[4,9]], 
            [[1,21],[2,18],[4,7]], 
            [[2,20],[3,22],[4,11]], 
            [[3,14],[1,16],[2,13],[4,9]]]
            
    elif indexJobset == 10:
        jobset = [
            [[1,11],[3,19],[2,16],[4,13]], 
            [[2,21],[3,16],[4,14]], 
            [[3,8], [2,10],[1,14],[4,9]], 
            [[2,13],[3,20],[4,10]], 
            [[1,9], [3,16],[4,18]], 
            [[2,19],[1,21],[3,11],[4,15]]]

    machine_alloc_mat, processing_time_mat = seperateMat(jobset)
    return [np.array(machine_alloc_mat), np.array(processing_time_mat), np.array(layout), agv_num]

    machine_alloc_mat = [] 
    processing_time_mat = []
    for i in instance:
        each_job_malloc = [] 
        each_job_process = []
        for j in i:
            each_job_malloc.append(j[0])
            each_job_process.append(j[1])

        machine_alloc_mat.append(each_job_malloc)
        processing_time_mat.append(each_job_process)
        
    return machine_alloc_mat, processing_time_mat
    

def seperateMat(instance):
    machine_alloc_mat = [] 
    processing_time_mat = []
    for i in instance:
        each_job_malloc = [] 
        each_job_process = []
        for j in i:
            each_job_malloc.append(j[0])
            each_job_process.append(j[1])

        machine_alloc_mat.append(each_job_malloc)
        processing_time_mat.append(each_job_process)
        
    return machine_alloc_mat, processing_time_mat