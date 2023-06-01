from random import randint, shuffle
from numpy.random import randint, random
from random import shuffle
from tqdm import tqdm

class TravellingSalesman:
    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix
        self.n = len(matrix)
        self.opt = [i for i in range(self.n)]
        self.opt_target = sum(self.matrix[i][(i + 1) % self.n] for i in range(self.n))

    def calculate_fitness(self):
        self.target = [0 for _ in range(self.n_samples)]
        for i in range(self.n_samples):
            for j in range(self.n):
                start = self.genes[i][j]
                end = self.genes[i][(j + 1) % self.n]
                self.target[i] += self.matrix[start][end]
            if self.target[i] < self.opt_target:
                self.opt_target = self.target[i]
                self.opt = self.genes[i]

    def selection(self, K: int) -> int:
        indexes = [i for i in range(self.n_samples)]
        shuffle(indexes)
        indexes = indexes[:K]
        min_i = indexes[0]
        for i in indexes:
            if self.target[i] < self.target[min_i]:
                min_i = i
        self.target[min_i] = float('inf')
        return min_i

    def mutation(self, x: list[int]) -> list[int]:
        for i in range(self.n):
            if random() < self.mutation_proba:
                j = randint(0, self.n)
                while i == j:
                    j = randint(0, self.n)
                x[i], x[j] = x[j], x[i]
        return x

    def permutation_swap(self, x: list[int], y: list[int], p1: int, p2: int) -> list[int]:
        used = set()
        x_new = [0 for _ in range(self.n)]
        for i in range(p1, p2 + 1):
            x_new[i] = y[i]
            used.add(y[i])
        x_cycle = x[p2 + 1:self.n] + x[0:p2 + 1]
        x_cycle = [i for i in x_cycle if i not in used]
        i = p2 + 1
        while i % self.n != p1:
            x_new[i % self.n] = x_cycle[i - p2 - 1]
            i += 1
        return x_new

    def crossover(self, x: list[int], y: list[int]) -> tuple[list[int], list[int]]:
        p1, p2 = randint(0, self.n, 2)
        if p1 > p2:
            p1, p2 = p2, p1
        x_new = self.permutation_swap(x, y, p1, p2)
        y_new = self.permutation_swap(y, x, p1, p2)
        return x_new, y_new

    def solve(self, n_samples: int, max_iters: int, n_cross: int,
              mutation_proba: float = 0.01) -> tuple[list[int], int]:
        self.n_samples = n_samples
        self.mutation_proba = mutation_proba
        self.genes = []
        for _ in range(n_samples):
            permutation = [i for i in range(self.n)]
            shuffle(permutation)
            self.genes.append(permutation)
        for _ in tqdm(range(max_iters)):
            self.calculate_fitness()
            for _ in range(n_cross):
                i = self.selection(self.n_samples // 4)
                j = self.selection(self.n_samples // 4)
                self.genes[i], self.genes[j] = self.crossover(self.genes[i], self.genes[j])
                self.genes[i] = self.mutation(self.genes[i])
                self.genes[j] = self.mutation(self.genes[j])
        self.calculate_fitness()
        return self.opt, self.opt_target

matrix = [[  0., 633., 257.,  91., 412., 150.,  80., 134., 259., 505., 353.,
         324.,  70., 211., 268., 246., 121.],
        [633.,   0., 390., 661., 227., 488., 572., 530., 555., 289., 282.,
         638., 567., 466., 420., 745., 518.],
        [257., 390.,   0., 228., 169., 112., 196., 154., 372., 262., 110.,
         437., 191.,  74.,  53., 472., 142.],
        [ 91., 661., 228.,   0., 383., 120.,  77., 105., 175., 476., 324.,
         240.,  27., 182., 239., 237.,  84.],
        [412., 227., 169., 383.,   0., 267., 351., 309., 338., 196.,  61.,
         421., 346., 243., 199., 528., 297.],
        [150., 488., 112., 120., 267.,   0.,  63.,  34., 264., 360., 208.,
         329.,  83., 105., 123., 364.,  35.],
        [ 80., 572., 196.,  77., 351.,  63.,   0.,  29., 232., 444., 292.,
         297.,  47., 150., 207., 332.,  29.],
        [134., 530., 154., 105., 309.,  34.,  29.,   0., 249., 402., 250.,
         314.,  68., 108., 165., 349.,  36.],
        [259., 555., 372., 175., 338., 264., 232., 249.,   0., 495., 352.,
          95., 189., 326., 383., 202., 236.],
        [505., 289., 262., 476., 196., 360., 444., 402., 495.,   0., 154.,
         578., 439., 336., 240., 685., 390.],
        [353., 282., 110., 324.,  61., 208., 292., 250., 352., 154.,   0.,
         435., 287., 184., 140., 542., 238.],
        [324., 638., 437., 240., 421., 329., 297., 314.,  95., 578., 435.,
           0., 254., 391., 448., 157., 301.],
        [ 70., 567., 191.,  27., 346.,  83.,  47.,  68., 189., 439., 287.,
         254.,   0., 145., 202., 289.,  55.],
        [211., 466.,  74., 182., 243., 105., 150., 108., 326., 336., 184.,
         391., 145.,   0.,  57., 426.,  96.],
        [268., 420.,  53., 239., 199., 123., 207., 165., 383., 240., 140.,
         448., 202.,  57.,   0., 483., 153.],
        [246., 745., 472., 237., 528., 364., 332., 349., 202., 685., 542.,
         157., 289., 426., 483.,   0., 336.],
        [121., 518., 142.,  84., 297.,  35.,  29.,  36., 236., 390., 238.,
         301.,  55.,  96., 153., 336.,   0.]]

a = TravellingSalesman(matrix)
print(a.solve(100,10000, 100))

