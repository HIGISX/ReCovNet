import random
import copy
import numpy as np
import time
from matplotlib import pyplot as plt
import torch
from matplotlib.patches import Circle

class Chromosome:
    """
    Class Chromosome represents one chromosome which consists of genetic code and value of
    fitness function.
    Genetic code represents potential solution to problem - the list of locations that are selected
    as medians.
    """

    def __init__(self, content, fitness):
        self.content = content
        self.fitness = fitness

    def __str__(self): return "%s f=%d" % (self.content, self.fitness)

    def __repr__(self): return "%s f=%d" % (self.content, self.fitness)


class GeneticAlgorithm:

    def __init__(self, n, m, p, cost_matrix, r, demand):

        self.time = None
        # self.num_facilities = num_facilities
        self.user_num = n
        self.fac_num = m
        self.p = p
        self.r = r
        self.cost_matrix = cost_matrix
        self.demand = demand
        self.iterations = 200  # Maximal number of iterations
        self.current_iteration = 0
        self.generation_size = 50  # Number of individuals in one generation
        self.reproduction_size = 20  # Number of individuals for reproduction

        self.mutation_prob = 0.1  # Mutation probability

        self.top_chromosome = None  # Chromosome that represents solution of optimization process

    def mutation(self, chromosome):
        """
        Applies mutation over chromosome with probability self.mutation_prob
        In this process, a randomly selected median is replaced with a randomly selected demand point.
        """

        mp = random.random()
        if mp < self.mutation_prob:
            # index of randomly selected median:
            i = random.randint(0, len(chromosome) - 1)
            # demand points without current medians:
            demand_points = [element for element in range(0, len(self.cost_matrix)) if element not in chromosome]
            # replace selected median with randomly selected demand point:
            chromosome[i] = random.choice(demand_points)

        return chromosome

    def crossover(self, parent1, parent2):

        identical_elements = [element for element in parent1 if element in parent2]

        # If the two parents are equal to each other, one of the parents is reproduced unaltered for the next generation
        # and the other parent is deleted, to avoid that duplicate individuals be inserted into the population.
        if len(identical_elements) == len(parent1):
            return parent1, None

        exchange_vector_for_parent1 = [element for element in parent1 if element not in identical_elements]
        exchange_vector_for_parent2 = [element for element in parent2 if element not in identical_elements]

        c = random.randint(0, len(exchange_vector_for_parent1) - 1)

        for i in range(c):
            exchange_vector_for_parent1[i], exchange_vector_for_parent2[i] = exchange_vector_for_parent2[i], \
                                                                             exchange_vector_for_parent1[i]

        child1 = identical_elements + exchange_vector_for_parent1
        child2 = identical_elements + exchange_vector_for_parent2

        return child1, child2

    def fitness(self, chromosome):
        """ Calculates fitness of given chromosome """
        dist_p = self.cost_matrix[chromosome]
        mask = dist_p < self.r
        dist_p[mask] = 1
        dist_p[~mask] = 0
        backup_cover = np.max(dist_p, axis=0)
        cover = np.matmul(backup_cover, self.demand)
        no_cover = np.sum(self.demand) - cover
        return no_cover

        # no_cover = self.fac_num
        # for i in range(self.fac_num):
        #     for j in chromosome:
        #         if self.cost_matrix[i, j] > self.r:
        #             continue
        #         else:
        #             no_cover -= 1
        #             break
        # return no_cover

    def initial_random_population(self):
        """
        Creates initial population by generating self.generation_size random individuals.
        Each individual is created by randomly choosing p facilities to be medians.
        """

        init_population = []
        for k in range(self.generation_size):
            rand_medians = []
            facilities = list(range(self.fac_num))
            for i in range(self.p):
                rand_median = random.choice(facilities)
                rand_medians.append(rand_median)
                facilities.remove(rand_median)
            init_population.append(rand_medians)

        init_population = [Chromosome(content, self.fitness(content)) for content in init_population]
        self.top_chromosome = min(init_population, key=lambda chromo: chromo.fitness)
        print("Current top solution: %s" % self.top_chromosome)
        return init_population

    def selection(self, chromosomes):
        """Ranking-based selection method"""

        # Chromosomes are sorted ascending by their fitness value
        chromosomes.sort(key=lambda x: x.fitness)
        L = self.reproduction_size
        selected_chromosomes = []

        for i in range(self.reproduction_size):
            j = L - np.floor((-1 + np.sqrt(1 + 4 * random.uniform(0, 1) * (L ** 2 + L))) / 2)
            selected_chromosomes.append(chromosomes[int(j)])
        return selected_chromosomes

    def create_generation(self, for_reproduction):
        """
        Creates new generation from individuals that are chosen for reproduction,
        by applying crossover and mutation operators.
        Size of the new generation is same as the size of previous.
        """
        new_generation = []

        while len(new_generation) < self.generation_size:
            parents = random.sample(for_reproduction, 2)
            child1, child2 = self.crossover(parents[0].content, parents[1].content)

            self.mutation(child1)
            new_generation.append(Chromosome(child1, self.fitness(child1)))

            if child2 is not None and len(new_generation) < self.generation_size:
                self.mutation(child2)
                new_generation.append(Chromosome(child2, self.fitness(child2)))

        return new_generation

    def optimize(self):

        start_time = time.time()

        chromosomes = self.initial_random_population()

        while self.current_iteration < self.iterations:
            # From current population choose individuals for reproduction
            for_reproduction = self.selection(chromosomes)

            # Create new generation from individuals that are chosen for reproduction
            chromosomes = self.create_generation(for_reproduction)

            self.current_iteration += 1

            chromosome_with_min_fitness = min(chromosomes, key=lambda chromo: chromo.fitness)
            if chromosome_with_min_fitness.fitness < self.top_chromosome.fitness:
                self.top_chromosome = chromosome_with_min_fitness

        end_time = time.time()
        self.time = end_time - start_time
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)

        print()
        print("Final top solution: %s" % self.top_chromosome)
        print('Time: {:0>2}:{:0>2}:{:05.4f}'.format(int(hours), int(minutes), seconds))


def display_points_with_mclp(users, facilities, centers, radius, demand):
    ax = plt.gca()
    # plt.xlim(-0.3, 1.1)
    # plt.ylim(-0.2, 1.3)
    plt.scatter(users[:, 0], users[:, 1], c='black', s=10*demand, label='Users')
    plt.scatter(facilities[:, 0], facilities[:, 1], c='blue', label='Candidate Facilities')
    for i in centers:
        plt.scatter(facilities[i][0], facilities[i][1], c='red', s=50, marker='*')
        circle = Circle(xy=(facilities[i][0], facilities[i][1]), radius=radius, color='black', fill=False, lw=2)
        ax.add_artist(circle)
    plt.scatter(facilities[i][0], facilities[i][1], c='red', marker='*', s=50, label='Centers')


if __name__ == '__main__':
    torch.manual_seed(1234)
    radius = 0.15
    n_users = 200
    n_facilities = 100
    n_centers = 15
    users = [(random.random(), random.random()) for i in range(n_users)]
    demand = np.random.randint(1, 10, size=n_users)
    facilities = [(random.random(), random.random()) for i in range(n_facilities)]
    users, facilities = np.array(users), np.array(facilities)
    distance = np.sum((facilities[:, np.newaxis, :] - users[np.newaxis, :, :]) ** 2, axis=-1) ** 0.5
    start_time = time.time()
    genetic = GeneticAlgorithm(n_users, n_facilities, n_centers, distance, radius, demand)
    genetic.optimize()
    obj = np.sum(demand) - genetic.top_chromosome.fitness
    centers = genetic.top_chromosome.content
    time = genetic.time


    print("The Set of centers are: %s" % centers)
    print("The objective is: %s" % str(round(obj)))

    fig = plt.figure(figsize=(8, 8))
    name = 'MCLP_OLD (P=' + str(n_centers) + ',N=' + str(n_users) + ',M=' + str(
        n_facilities) + ')\nThe objective of BCLP is ' + str(round(obj))
    plt.title(name, font='Times New Roman', fontsize=18)
    display_points_with_mclp(users, facilities, centers, radius, demand)
    plt.legend(loc='best', prop='Times New Roman', fontsize=18)
    plt.show()
