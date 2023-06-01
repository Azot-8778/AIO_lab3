import numpy as np

def crossover(x, y):
    point = np.random.randint(1, len(x) - 1)
    newx = np.concatenate([x[:point], y[point:]])
    newy = np.concatenate([y[:point], x[point:]])
    
    return newx, newy

def mutate(x, p):
    newx = x.copy()
    for i in range(len(x)):
        if np.random.uniform() <= p:
            newx[i] = (x[i] + 1) % 2
            
    return x

def fitness(x, c):
    return x @ c

def correct(x, w, maxW):
    weight = x @ w
    while weight > maxW:
        x[np.random.choice(x.nonzero()[0])] = 0
        weight = x @ w
        
def generate(n, w, maxW):
    chromo = []
    for x in range(n):
        if np.random.uniform() <= 0.5:
            chromo.append(1)
        else:
            chromo.append(0)
    x = np.array(chromo)
    correct(x, w, maxW)
    return x
    
def genetic_solver(w, c, maxW, popsize, max_iter, mutation_proba=0.05):
    population = [generate(len(w), w, maxW) for _ in range(popsize)]

    for i in range(max_iter):
        fitnesses = [fitness(x, c) for x in population]
        fsum = np.sum(fitnesses)
        proba = [fitness(x, c) / fsum for x in population]
        idx1 = np.random.choice([i for i in range(len(fitnesses))], p=proba)
        idx2 = np.random.choice([i for i in range(len(fitnesses))], p=proba)
        newx, newy = crossover(population[idx1], population[idx2])
        population[idx1] = newx
        population[idx2] = newy
        population[idx1] = mutate(population[idx1], mutation_proba)
        population[idx2] = mutate(population[idx2], mutation_proba)
        correct(population[idx1], w, maxW)
        correct(population[idx2], w, maxW)
        
    fitnesses = [fitness(x, c) for x in population]
    idx1 = np.argmax(fitnesses)
    ans = population[idx1]
    
    return ans


w = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
c = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])
maxW = 165

ans = genetic_solver(w, c, maxW, 1000, 250) 
print({i for i in range(len(ans)) if ans[i] == 1}, 'ans cost:', fitness(ans, c), 'ans w:', ans @ w)
