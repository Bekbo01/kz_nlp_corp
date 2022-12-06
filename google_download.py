from bisect import bisect
import copy
import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
import math
import random


def func(x):
    # return math.sin(x[0]) * math.cos(x[1]) * (1. / (abs(x[0]) + 1))
    return (-20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20)

def func2(x):
    # return -1.0*(math.sin(x[0]) * math.cos(x[1]) * (1. / (abs(x[0]) + 1)))
    return -1*(-20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20)
    

def nelder_mead(x_start, step=0.1, no_improve_thr=1e-12, no_improve_break=10, max_iter=1000,
               alpha=1, gamma=2, rho=0.5, sigma=0.5, verbose=False):
    dim = len(x_start)
    prev_best = func(x_start)
    no_improve = 0
    res = [[x_start, prev_best]]
    
    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = func(x)
        res.append([x, score])
    
    iters = 0
    while True:
        res.sort(key=lambda x: x[1])
        best = res[0][1]
        
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1
        
        if verbose:
            print('best', best)
            
        if best < prev_best - no_improve_thr:
            no_improve = 0
            prev_best = best
        else:
            no_improve += 1
        
        if no_improve >= no_improve_break:
            return res[0]
        
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)
        
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = func(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue
        
        if rscore < res[0][1]:
            xe = x0 + gamma*(xr - res[-1][0])# x0)
            escore = func(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue
        
        xc = x0 + rho * (x0 - res[-1][0])
        cscore = func(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue
    
        x1 = res[0][0]
        nres = list()
        for tup in res:
            redx = x1 + sigma * (tup[0] -x1)
            score = func(redx)
            nres.append([redx, score])

        res = nres
    return res


def test_func(x, y, z):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20


def genetic_algo(features, x_start, norm_factor, no_improve_thr=10e-6, num_solutions=20,
                no_improve_break=10, max_iter=1000, mutation_thr=0.15, alpha=0.5):
    # features pandas dataframe
    dim = len(x_start)
    no_improve = 0

    NUMBER_OF_SELECTED_SOLUTIONS = num_solutions/2

    solutions_dict = {i: np.random.normal(1, 0.05, dim) for i in range(num_solutions)}

    for i in range(max_iter):

        solutions_scores = np.zeros(num_solutions)
        for k in range(num_solutions):
            solution = solutions_dict[k]
            results = test_func(features, solution, norm_factor)
            solutions_scores[k] = results
        best = solutions_scores.mean()

        if i == 0:
            prev_best = best
        
        if best < prev_best - no_improve_thr:
            no_improve = 0
            prev_best = best
        else:
            no_improve += 1
        
        if no_improve >= no_improve_break:
            ind = np.argpartition(solutions_scores, 0)
            solutions_dict_new = solutions_dict

            solutions_dict[0] = solutions_dict_new[ind[0]]
            return solutions_dict[0], solutions_scores[ind[0]]
        

        ind = np.argpartition(solutions_scores, -NUMBER_OF_SELECTED_SOLUTIONS)[:NUMBER_OF_SELECTED_SOLUTIONS]
        solutions_dict_new = solutions_dict
        for a in range(NUMBER_OF_SELECTED_SOLUTIONS):
            solutions_dict[a] = solutions_dict_new[ind[a]]
        
        for b in range(NUMBER_OF_SELECTED_SOLUTIONS, num_solutions, 2):
            solutions_dict[b] = alpha * solutions_dict[b-NUMBER_OF_SELECTED_SOLUTIONS] + (1-alpha) * solutions_dict[b-NUMBER_OF_SELECTED_SOLUTIONS+1]
            solutions_dict[b+1] = alpha * solutions_dict[b-NUMBER_OF_SELECTED_SOLUTIONS+1] + (1-alpha) * solutions_dict[b-NUMBER_OF_SELECTED_SOLUTIONS]

        for c in range(num_solutions):
            mutation_random = np.random.uniform(0, 1)
            mutation_index = np.random.randint(0, dim-1)
            if mutation_random < mutation_thr:
                solutions_dict[c][mutation_index] += np.random.normal(0, 0.15, 1)
    
    ind = np.argpartition(solutions_scores, 0)
    solutions_dict_new = solutions_dict
    solutions_dict[0] = solutions_dict_new[ind[0]]
    return solutions_dict[0], solutions_scores[ind[0]]


def my_dump_algo(x_extrimum, y_extrimum, number_of_grid=10, number_of_select_per_grid=5, 
                max_iteration=1000, callback_iteration=10, number_top_point=5,
                no_improve_thr=1e-6):
    no_improve = 0
    best_score = 1e6
    best_point = None

    for curr_iteration in range(1, max_iteration+1):

        x_grids = np.linspace(x_extrimum[0], x_extrimum[1], number_of_grid+1)
        y_grids = np.linspace(y_extrimum[0], y_extrimum[1], number_of_grid+1)

        points = []
        for i in range(1, number_of_grid+1):
            x = np.random.uniform(x_grids[i-1], x_grids[i], number_of_select_per_grid)
            y = np.random.uniform(y_grids[i-1], y_grids[i], number_of_select_per_grid)
            for a in zip(x,y):
                points.append(a)
        
        scores = []
        for j in range(number_of_select_per_grid + number_of_grid):
            scores.append(func(points[j]))
        top_scores = sorted(scores)[:number_top_point]
        top_points = [points[scores.index(k)] for k in top_scores]
        x_extrimum = sorted([l[0] for l in top_points])[::number_top_point-1]
        y_extrimum = sorted([l[1] for l in top_points])[::number_top_point-1]

        if top_scores[0] < best_score - no_improve_thr:
            best_score = top_scores[0]
            best_point = top_points[0]
            no_improve = 0
        else:
            no_improve +=1
        
        if no_improve >= callback_iteration:
            break
    
    return [best_point, best_score]


def swarm_algo(W=5, c1=0.5, c2=0.9, target=0, n_iterations=500, target_error=0.00008,
                n_particles=5):
    
    iteration = 0

    return 1


def objective(x):

    return x[0]**2 + x[1]**2


def genetic_algorithm(func, bounds=[[-5.0, 5.0], [-5.0, 5.0]], n_iter=100, 
                    n_bits=16, n_pop=100, r_cross=0.9, verbose=False):
    r_mut = 1.0 / (float(n_bits) * len(bounds))
    pop = [np.random.randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]

    def decode(bounds, n_bits, bitstring):
        decoded = list()
        largest = 2**n_bits
        for i in range(len(bounds)):
            start, end = i*n_bits, (i*n_bits)+n_bits
            substring = bitstring[start:end]
            chars = ''.join([str(s) for s in substring])
            integer = int(chars, 2)
            values = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
            decoded.append(values)
        return decoded
    
    def selection(pop, scores, k=3):
        selection_idx = np.random.randint(len(pop))
        for idx in np.random.randint(0, len(pop), k-1):
            if scores[idx] < scores[selection_idx]:
                selection_idx = idx
        return pop[selection_idx]

    def crossover(p1, p2, r_cross):
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < r_cross:
            pt = np.random.randint(1, len(p1)-2)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    def mutation(bistring, r_mut):
        for i in range(len(bistring)):
            if np.random.rand() < r_mut:
                bistring[i] = 1 - bistring[i]

    best, best_eval = 0, func(decode(bounds, n_bits, pop[0]))
    for gen in range(n_iter):
        decoded = [decode(bounds, n_bits, p) for p in pop]
        scores = [func(d) for d in decoded]
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                if verbose:
                    print(f'{gen} gen {decoded[i]}, {scores[i]}')
        selected = [selection(pop, scores) for _ in range(n_pop)]
        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
        pop = children
    return [decode(bounds, n_bits, best), best_eval]


from bees_algorithm import BeesAlgorithm

if __name__ == "__main__":
    
    # new section to great opportunity
    print('Geneti algo')
    start = time.time()
    print(genetic_algorithm(func=func))
    print(time.time()-start)

    def get_pt(r_min=-5.0, r_max=5.0):
        return r_min + np.random.rand(2) * (r_max - r_min)

    # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    print('Bees algo')
    start = time.time()
    res2_list = []
    for i in range(100):
        alg = BeesAlgorithm(func2, [-5.0, -5.0], [5.0, 5.0])
        alg.performFullOptimisation(max_iteration=100)
        best = alg.best_solution
        res2_list.append([best.score]) # best.values,
    #print(res2_list)
    print(np.min(res2_list))
    print(time.time()-start)
    # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    # start = time.time()
    # print(nelder_mead(get_pt(), max_iter=100))
    # print(time.time()-start)

    # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    print('Nelder mead algo')
    start = time.time()
    res1_list = []
    for i in range(100):
        pt = get_pt()
        #res = my_dump_algo((-5.0, 5.0), (-5.0, 5.0))
        res = nelder_mead(pt, max_iter=100)
        #res = minimize(func, pt, method='nelder-mead')
        res1_list.append(res[-1])
    #print(res1_list)
    print(np.min(res1_list))
    print(time.time()-start)

    # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    print('Nelder mead algo (parallel)')
    print('Start multy ')
    start = time.time()
    pool = mp.Pool(10)
    processes = []
    for i in range(10000):
        pt = get_pt()
        j = pool.apply_async(nelder_mead, args=(pt,))
        # j = pool.apply_async(minimize, args=(func, pt,), kwds={'method': 'nelder-mead'})
        processes.append(j)

    # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    res_list = []
    for step, r in enumerate(processes):
        #print(step, r)
        res = r.get()
        #print(res)
        res_list.append(res)
    pool.close()
    pool.join()
    res_list = sorted(res_list, key=lambda a: a[1])
    print(res_list[0])
    print(time.time()-start)
    
