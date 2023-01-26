import random
from itertools import product


def ks_exhaustive(s_list, v_list, S):
    m = len(s_list)
    opt_i = 0
    selection_list = list(product((1, 0), repeat=m))
    for selection in selection_list:
        result = v_kS(selection, s_list, v_list, S, m)
        if result and result > opt_i:
            opt_i = result
    return opt_i


def v_kS(selection, s_list, v_list, S, m):
    tmp_S = 0
    tmp_k = 0
    for i in range(m):
        if selection[i] == 1:
            tmp_k += v_list[i]
            tmp_S += s_list[i]
    if tmp_S <= S:
        return tmp_S
    return None


class GeneticAlgorithm():

    def __init__(self, w, v, W):
        self.w = w
        self.v = v
        self.W = W
        self.m = len(v)
        self.generation = []  # store genes
        self.mutate = 0.1  # mutation chance
        self.survival = 0.5  # survival chance
        self.n_childs = 64  # population size
        self.n_generations = 1000  # number of generations
        self.best_score = (None, -1)

    def populate(self):
        for j in range(self.n_childs):
            self.generation += [[random.randint(0, 1)
                                 for i in range(self.m)]]

    def fitness(self, genes):
        genes_W, genes_score = 0, 0
        for i in range(self.m):
            if genes[i] == 1:
                genes_W += self.w[i]
                genes_score += self.v[i]
            if genes_W > self.W:
                return -1
        return genes_score

    def mutation(self, genes):
        for i in range(self.m):
            if random.uniform(0, 1) <= self.mutate:
                genes[i] = 1 - genes[i]
        return genes

    def crossover(self, genes_1, genes_2):
        j = random.randint(1, (self.m - 1))
        if random.randint(0, 1) == 0:
            return genes_1[:j] + genes_2[j:]
        return genes_2[:j] + genes_1[j:]

    def selection(self):
        evaluation = []
        for genes in self.generation:
            evaluation += [(genes, self.fitness(genes))]
        evaluation.sort(key=lambda w: w[1], reverse=True)
        if evaluation[0][1] > self.best_score[1]:
            self.best_score = evaluation[0]
        evaluation = [g[0] for g in evaluation]
        self.generation = evaluation[:round(self.n_childs * self.survival)]

    def evolution(self):
        self.populate()  # init generation

        for l in range(self.n_generations):
            self.selection()  # surviving of the fittest
            n_parents = len(self.generation)
            for _ in range(self.n_childs - n_parents):
                parent_1 = self.generation[random.randint(0, n_parents - 1)]
                parent_2 = self.generation[random.randint(0, n_parents - 1)]
                self.generation += [self.crossover(parent_1, parent_2)]

            self.generation = [self.mutation(gene) for gene in self.generation]
        return self.generation


if __name__ == '__main__':
    # bug atm
    s_list, v_list, S = ([5, 5, 4, 5, 5], [9, 9, 1, 9, 9], 4)
    print('ks_exhaustive: ', ks_exhaustive(s_list, v_list, S))
    gens = GeneticAlgorithm(s_list, v_list, S)
    gens.evolution()
    print('GeneticAlgorithm: ', gens.best_score)
