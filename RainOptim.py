from numpy import exp, sqrt, sum
from numpy.random import uniform
from copy import deepcopy
from mealpy.root import Root


class Rain(Root):

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True, epoch=750, pop_size=100):
        Root.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        # Init pop and calculate fitness
        pop = [self.create_solution() for _ in range(self.pop_size)]

        # Find the pathfinder
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        gbest_present = deepcopy(g_best)

        for epoch in range(self.epoch):
            alpha, beta = uniform(1, 2, 2)
            A = uniform(self.lb, self.ub) * exp(-2 * (epoch + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self.amend_position_faster(temp)
            fit = self.get_fitness_position(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for i in range(1, self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                pos_new = deepcopy(pop[i][self.ID_POS])

                t1 = beta * uniform() * (gbest_present[self.ID_POS] - temp)
                for k in range(1, self.pop_size):
                    dist = sqrt(sum((pop[k][self.ID_POS] - temp)**2)) / self.problem_size
                    t2 = alpha * uniform() * (pop[k][self.ID_POS] - temp)
                    ## First stabilize the distance
                    t3 = uniform() * (1 - (epoch + 1) * 1.0 / self.epoch) * (dist / (self.ub - self.lb))
                    pos_new += t2 + t3
                ## Second stabilize the population size
                pos_new = (pos_new + t1) / self.pop_size

                ## Update members
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]

                ### Batch size idea
                ## Update the best position found so far (current pathfinder)
                if i % self.batch_size:
                    pop, gbest_present = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, gbest_present)
            self.loss_train.append(gbest_present[self.ID_FIT])
            # if self.verbose:
            #     print("> Epoch: {}, Best fit: {}".format(epoch + 1, gbest_present[self.ID_FIT]))
        self.solution = gbest_present
        return gbest_present[self.ID_FIT]
