#!/usr/bin/env python
from cosc343world import Creature, World
import numpy as np
import random
import operator
import time

# You can change this number to specify how many generations creatures are going to evolve over...
numGenerations = 250

# You can change this number to specify how many turns in simulation of the world for given generation
numTurns = 100

# You can change this number to change the percept format.  You have three choice - format 1, 2 and 3 (described in
# the assignment 2 pdf document)
perceptFormat = 1

# You can change this number to change the world size
gridSize = 24

# You can set this mode to True to have same initial conditions for each simulation in each generation.  Good
# for development, when you want to have some determinism in how the world runs from generating to generation.
repeatableMode = True

# This is a class implementing you creature a.k.a MyCreature.  It extends the basic Creature, which provides the
# basic functionality of the creature for the world simulation.  Your job is to implement the AgentFunction
# that controls creature's behaviour by producing actions in response to percepts.
class MyCreature(Creature):

    # Initialisation function.  This is where you creature
    # should be initialised with a chromosome in random state.  You need to decide the format of your
    # chromosome and the model that it's going to give rise to
    #
    # Input: numPercepts - the size of percepts list that creature will receive in each turn
    #        numActions - the size of actions list that creature must create on each turn
    def __init__(self, numPercepts, numActions):
        # Place your initialisation code here.  Ideally this should set up the creature's chromosome
        # and set it to some random state.

        # Generate the monster part of the chromosome.
        self.monster_move_away = [round(random.random(), 2)]
        self.monster_move_closer = [round(random.random(), 2)]
        self.creature_move_away = [round(random.random(), 2)]
        self.creature_move_closer = [round(random.random(), 2)]
        self.food_move_away = [round(random.random(), 2)]
        self.food_move_closer = [round(random.random(), 2)]
        self.eat = [round(random.random(), 2)]
        self.random = [round(random.random(), 2)]

        # Generate cross over intercept probabilities.
        self.crossover = np.random.random(7)
        self.crossover /= sum(self.crossover)

        # Chromosome is in format: [mma, mmc, cma, cmc, fma, fmc, eat, random]
        self.mutate = round(random.random(), 2) / 3.0
        self.crossover = self.crossover.tolist()
        self.chromosome = self.monster_move_away + self.monster_move_closer + self.creature_move_away + \
            self.creature_move_closer + self.food_move_away + self.food_move_closer + self.eat + self.random
        self.fitness = 0
        # self.chromosome = [.2, .3, .1, .3, .2, .1, .1, .1]

        # Do not remove this line at the end.  It calls constructors
        # of the parent classes.
        Creature.__init__(self)

    # This is the implementation of the agent function that is called on every turn, giving your
    # creature a chance to perform an action.  You need to implement a model here, that takes its parameters
    # from the chromosome and it produces a set of actions from provided percepts
    #
    # Input: percepts - a list of percepts
    #        numAction - the size of the actions list that needs to be returned
    def AgentFunction(self, percepts, numActions):

        # At the moment the actions is a list of random numbers.  You need to
        # replace this with some model that maps percepts to actions.  The model
        # should be parametrised by the chromosome
        actions = [0] * numActions
        # percepts = [1,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0]
        # print(percepts)

        # Set movement actions
        for p_index, percept in enumerate(percepts[:9]):
            if percept:
                for c_index, chromosome in enumerate(self.chromosome[:2]):
                    if c_index % 2 == 0:
                        actions[8 - p_index] += percept * chromosome
                    else:
                        actions[p_index] += percept * chromosome

        for p_index, percept in enumerate(percepts[9:18]):
            if percept:
                for c_index, chromosome in enumerate(self.chromosome[2:4]):
                    if c_index % 2 == 0:
                        actions[8 - p_index] += percept * chromosome
                    else:
                        actions[p_index] += percept * chromosome

        for p_index, percept in enumerate(percepts[18:]):
            if percept:
                for c_index, chromosome in enumerate(self.chromosome[4:6]):
                    if c_index % 2 != 0:
                        actions[8 - p_index] += percept * chromosome
                    else:
                        actions[p_index] += percept * chromosome

        # Set food and random actions
        # actions[9] = sum(percepts[18:])/len(percepts[18:]) # Or should it be the count of things that are non zero?
        # playing with divisor = 9
        divisor = 9
        if percepts[22] > 1:
            actions[9] += 0.5 + self.chromosome[6]
            if percepts[22] == 2:
                actions[9] += 0.5

        actions[10] += (((len(percepts) - np.count_nonzero(percepts)) / 27)/9) + self.chromosome[7]
        # print(actions)
        # print(self.chromosome)
        # exit()
        return actions


# This function is called after every simulation, passing a list of the old population of creatures, whose fitness
# you need to evaluate and whose chromosomes you can use to create new creatures.
#
# Input: old_population - list of objects of MyCreature type that participated in the last simulation.  You
#                         can query the state of the creatures by using some built-in methods as well as any methods
#                         you decide to add to MyCreature class.  The length of the list is the size of
#                         the population.  You need to generate a new population of the same size.  Creatures from
#                         old population can be used in the new population - simulation will reset them to starting
#                         state.
#
# Returns: a list of MyCreature objects of the same length as the old_population.
def newPopulation(old_population):
    global numTurns

    nSurvivors = 0
    avgLifeTime = 0
    fitnessScore = 0

    # For each individual you can extract the following information left over
    # from evaluation to let you figure out how well individual did in the
    # simulation of the world: whether the creature is dead or not, how much
    # energy did the creature have a the end of simulation (0 if dead), tick number
    # of creature's death (if dead).  You should use this information to build
    # a fitness function, score for how the individual did
    for individual in old_population:

        # You can read the creature's energy at the end of the simulation.  It will be 0 if creature is dead
        energy = individual.getEnergy()

        # This method tells you if the creature died during the simulation
        dead = individual.isDead()

        # If the creature is dead, you can get its time of death (in turns)
        if dead:
            timeOfDeath = individual.timeOfDeath()
            avgLifeTime += timeOfDeath

            individual.fitness += timeOfDeath

            # If they died but their energy isn't zero, and its higher than the number of turns they lasted, or
            # its higher than initial
            if energy > timeOfDeath or energy > 50:
                individual.fitness += 75
        else:
            nSurvivors += 1
            avgLifeTime += numTurns

            individual.fitness += energy

            # Large bonus for surviving
            individual.fitness += 100

        fitnessScore += individual.fitness

        # print(individual.chromosome, individual.fitness)

    # Here are some statistics, which you may or may not find useful
    avgLifeTime = float(avgLifeTime)/float(len(population))
    fitnessScore = float(fitnessScore)/float(len(population))
    print("Simulation stats:")
    print("  Survivors    : %d out of %d" % (nSurvivors, len(population)))
    print("  Avg life time: %.1f turns" % avgLifeTime)
    print("  Avg fitness: %.1f" % fitnessScore)

    # The information gathered above should allow you to build a fitness function that evaluates fitness of
    # every creature.  You should show the average fitness, but also use the fitness for selecting parents and
    # creating new creatures.

    # Based on the fitness you should select individuals for reproduction and create a
    # new population.  At the moment this is not done, and the same population with the same number
    # of individuals
    def tournamentSelect(n):
        fittest = {}
        for individual in n:
            fittest[individual] = individual.fitness

        sorted_fitness = sorted(fittest.items(), key=lambda x: x[1])

        # print(sorted_fitness[0][0].fitness, sorted_fitness[1][0].fitness, \
        #   sorted_fitness[len(sorted_fitness) - 2][0].fitness, sorted_fitness[len(sorted_fitness) - 1][0].fitness)

        return [sorted_fitness[len(sorted_fitness) - 1][0], sorted_fitness[len(sorted_fitness) - 2][0]]

    def decision(probability):
        return random.random() < probability

    new_population = []
    while len(new_population) < len(old_population):
        winner, winner2 = tournamentSelect(random.sample(old_population, int(len(old_population) / 4)))

        # print(winner.fitness, winner2.fitness)
        # print(winner.chromosome, winner2.chromosome)

        average_mutate = (winner.mutate + winner2.mutate) / 2.0

        average_crossover = []
        for i in range(len(winner.crossover)):
            average_crossover.append(((winner.crossover[i] + winner2.crossover[i]) / 2.0))

        #Work out crossover
        for i in range(len(average_crossover)):
            if i == len(average_crossover) - 1:
                if decision(average_crossover[i]):
                    crossover = i
                    break
                crossover = random.randint(0, 7)
                break
            if decision(average_crossover[i]):
                crossover = i
                break
        #crossover = int(len(winner.chromosome) / 2)

        new_chromosome = winner.chromosome[:crossover] + winner2.chromosome[crossover:len(winner2.chromosome)]


        if decision(average_mutate):
            new_chromosome[random.randint(0, len(new_chromosome) - 1)] = round(random.random(), 2)

        new_individual = MyCreature(numCreaturePercepts, numCreatureActions)
        new_individual.chromosome = new_chromosome
        new_individual.crossover = average_crossover
        new_individual.mutate = average_mutate

        # print(winner.chromosome[6:], winner2.chromosome[6:], "\n", new_chromosome[6:])
        # print(new_individual.chromosome, new_individual.crossover, new_individual.mutate)
        new_population.append(new_individual)

    # print(old_population[0].chromosome, new_population[0].chromosome)

    return new_population

# Create the world. Representation type chooses the type of percept representation (there are three types to chose from)
# gridSize specifies the size of the world, repeatable parameter allows you to run the simulation in exactly same way.
w = World(representationType=perceptFormat, gridSize=gridSize, repeatable=repeatableMode)

#Get the number of creatures in the world
numCreatures = w.maxNumCreatures()

#Get the number of creature percepts
numCreaturePercepts = w.numCreaturePercepts()

#Get the number of creature actions
numCreatureActions = w.numCreatureActions()

# Create a list of initial creatures - instantiations of the MyCreature class that you implemented
population = list()
for i in range(numCreatures):
   c = MyCreature(numCreaturePercepts, numCreatureActions)
   population.append(c)

# Pass the first population to the world simulator
w.setNextGeneration(population)

# Runs the simulation to evalute the first population
w.evaluate(numTurns)

# Show visualisation of initial creature behaviour
w.show_simulation(titleStr='Initial population', speed='fast')

for i in range(numGenerations):
    print("\nGeneration %d:" % (i+1))

    # Create a new population from the old one
    population = newPopulation(population)

    # Pass the new population to the world simulator
    w.setNextGeneration(population)

    # Run the simulation again to evalute the next population
    w.evaluate(numTurns)

    # Show visualisation of final generation
    if i==numGenerations-1:
        for pop in population:
            print ("MMA: " + str(pop.chromosome[0]) + \
                   " MMC: " + str(pop.chromosome[1]) + \
                   " CMA: " + str(pop.chromosome[2]) + \
                   " CMC: " + str(pop.chromosome[3]) + \
                   " FMA: " + str(pop.chromosome[4]) + \
                   " FMC: " + str(pop.chromosome[5]) + \
                   " EAT: " + str(pop.chromosome[6]) + \
                   " RAN: " + str(pop.chromosome[7]))
        w.show_simulation(titleStr='Final population', speed='slow')


