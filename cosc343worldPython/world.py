#!/usr/bin/env python
from cosc343world import Creature, World
import numpy as np
import random
import matplotlib.pyplot as plt
import statistics

# You can change this number to specify how many generations creatures are going to evolve over...
numGenerations = 500

# You can change this number to specify how many turns in simulation of the world for given generation
numTurns = 100
nSurvivors = 0

# You can change this number to change the percept format.  You have three choice - format 1, 2 and 3 (described in
# the assignment 2 pdf document)
perceptFormat = 1
incest = 0

# You can change this number to change the world size
gridSize = 40

# You can set this mode to True to have same initial conditions for each simulation in each generation.  Good
# for development, when you want to have some determinism in how the world runs from generating to generation.
repeatableMode = True

archive = [[] for _ in range(14)]

rElitism = random.randint(0, 10)
rCrossover = random.randint(2, 5)
rCheckIncest = random.randint(2, 10)
rCheckStdDev = random.randint(2, 6)
rStdDev = round(random.uniform(0.01, 0.5), 2)
rIncreaseChance = random.randint(1, 3)

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
        self.monster_move_diag = [round(random.random(), 2)]
        self.creature_move_away = [round(random.random(), 2)]
        self.creature_move_closer = [round(random.random(), 2)]
        self.creature_move_diag = [round(random.random(), 2)]
        self.food_move_away = [round(random.random(), 2)]
        self.food_move_closer = [round(random.random(), 2)]
        self.food_move_diag = [round(random.random(), 2)]
        self.eat = [round(random.uniform(0.01, 0.5), 2)]
        self.random = [round(random.random(), 2)]

        # Generate cross over intercept probabilities.
        self.crossover = np.random.random(8)

        # Chromosome is in format: [mma, mmc, cma, cmc, fma, fmc, eat, random]
        self.crossover = self.crossover.tolist()
        self.chromosome = self.monster_move_away + self.monster_move_closer + self.monster_move_diag + \
                          self.creature_move_away + self.creature_move_closer + self.creature_move_diag + \
                          self.food_move_away + self.food_move_closer + self.food_move_diag + \
                          self.eat + self.random
        self.fitness = 0

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

        # Set movement actions
        for p_index, percept in enumerate(percepts[:9]):
            if percept and percepts[8 - p_index]:
                if p_index in [0, 1, 5, 6]:
                    lower = p_index + 2
                    upper = (8 - p_index) - 2
                elif p_index in [2, 3, 7, 8]:
                    lower = p_index - 2
                    upper = (8 - p_index) + 2

                if percepts[lower]:
                    actions[upper] += percept * self.chromosome[2]
                elif percepts[upper]:
                    actions[lower] += percept * self.chromosome[2]
                else:
                    actions[lower] += percept * self.chromosome[2]
                    actions[upper] += percept * self.chromosome[2]
            elif percept:
                actions[8 - p_index] += percept * self.chromosome[0]
                actions[p_index] += percept * self.chromosome[1]

        for p_index, percept in enumerate(percepts[9:18]):
            # if percept:
            #     actions[8 - p_index] += percept * self.chromosome[3]
            #     actions[p_index] += percept * self.chromosome[4]
            if percept and percepts[8 - p_index]:
                if p_index in [0, 1, 5, 6]:
                    lower = p_index + 2
                    upper = (8 - p_index) - 2
                elif p_index in [2, 3, 7, 8]:
                    lower = p_index - 2
                    upper = (8 - p_index) + 2

                if percepts[lower]:
                    actions[upper] += percept * self.chromosome[5]
                elif percepts[upper]:
                    actions[lower] += percept * self.chromosome[5]
                else:
                    actions[lower] += percept * self.chromosome[5]
                    actions[upper] += percept * self.chromosome[5]
            elif percept:
                actions[8 - p_index] += percept * self.chromosome[3]
                actions[p_index] += percept * self.chromosome[4]

        for p_index, percept in enumerate(percepts[18:]):
            # if percept:
            #     actions[8 - p_index] += percept * self.chromosome[5]
            #     actions[p_index] += percept * self.chromosome[6]
            if percept and percepts[8 - p_index]:
                if p_index in [0, 1, 5, 6]:
                    lower = p_index + 2
                    upper = (8 - p_index) - 2
                elif p_index in [2, 3, 7, 8]:
                    lower = p_index - 2
                    upper = (8 - p_index) + 2

                if percepts[lower]:
                    actions[upper] += percept * self.chromosome[8]
                elif percepts[upper]:
                    actions[lower] += percept * self.chromosome[8]
                else:
                    actions[lower] += percept * self.chromosome[8]
                    actions[upper] += percept * self.chromosome[8]
            elif percept:
                actions[8 - p_index] += percept * self.chromosome[6]
                actions[p_index] += percept * self.chromosome[7]

        # Set food and random actions
        # if percepts[22] == 2:
        #     actions[9] += self.chromosome[7] #* percepts[22]
        if percepts[22]:
            actions[9] += self.chromosome[9] * percepts[22]

        actions[10] += ((1 - (np.count_nonzero(percepts) / 27)) / 5) + self.chromosome[10]
        # actions[10] += ((len(percepts) - np.count_nonzero(percepts)) / 27) / 4 + self.chromosome[7]

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
    global archive
    global nSurvivors
    global incest
    global rElitism
    global rCrossover
    global rCheckIncest
    global rCheckStdDev
    global rStdDev
    global rIncreaseChance

    nSurvivors = 0
    avgLifeTime = 0
    fitnessScore = 0

    # For each individual you can extract the following information left over
    # from evaluation to let you figure out how well individual did in the
    # simulation of the world: whether the creature is dead or not, how much
    # energy did the creature have a the end of simulation (0 if dead), tick number
    # of creature's death (if dead).  You should use this information to build
    # a fitness function, score for how the individual did
    def fitness(individual):
        dead = individual.isDead()
        tod = individual.timeOfDeath()
        energy = individual.getEnergy()

        fitness = 0

        # Give each chromosome base line
        if not dead:
            fitness += 50 + (energy * 2)
        elif tod < 50:
            fitness += tod

        # Chromosomes that eat
        if energy > (50 - tod):
            fitness += (energy - (50 - tod))

        # Chromosomes that move closer to food
        # if tod > 50:
        #     fitness += tod - 50

        return fitness

    for individual in old_population:

        # You can read the creature's energy at the end of the simulation.  It will be 0 if creature is dead
        energy = individual.getEnergy()

        # This method tells you if the creature died during the simulation
        dead = individual.isDead()

        # If the creature is dead, you can get its time of death (in turns)
        if dead:
            timeOfDeath = individual.timeOfDeath()
            avgLifeTime += timeOfDeath
        else:
            nSurvivors += 1
            avgLifeTime += numTurns

        individual.fitness = fitness(individual)
        fitnessScore += individual.fitness

    # Here are some statistics, which you may or may not find useful
    avgLifeTime = float(avgLifeTime)/float(len(population))
    fitnessScore = float(fitnessScore)/float(len(population))
    archive[11].append(nSurvivors)
    archive[12].append(fitnessScore)
    #print("Simulation stats:")
    print("  Survivors    : %d out of %d (%d percent)" % (nSurvivors, len(population), float(nSurvivors)/len(population) * 100))
    #print("  Avg life time: %.1f turns" % avgLifeTime)
    #print("  Avg fitness: %.1f" % fitnessScore)

    # The information gathered above should allow you to build a fitness function that evaluates fitness of
    # every creature.  You should show the average fitness, but also use the fitness for selecting parents and
    # creating new creatures.

    # Based on the fitness you should select individuals for reproduction and create a
    # new population.  At the moment this is not done, and the same population with the same number
    # of individuals
    def elitism(old_population, n):
        sorted_fitness = sorted(old_population, key=lambda individual: individual.fitness)

        elite = []
        for e in sorted_fitness[len(sorted_fitness) - n:]:
            elite_individual = MyCreature(numCreaturePercepts, numCreatureActions)
            elite_individual.chromosome = e.chromosome
            elite_individual.crossover = e.crossover
            elite.append(elite_individual)

        return elite

    def tournament_select(n):
        sorted_fitness = sorted(n, key=lambda individual: individual.fitness)
        return sorted_fitness[-1]

    def multi_crossover(parent1, parent2, n, id):
        global incest
        if parent1.chromosome == parent2.chromosome:
            incest += 1
            if id % 5 == 0:
                parent2.chromosome = np.random.random(11).tolist()

        # Calculate the average crossover of the two parents
        average_crossover = []
        for i in range(len(parent1.crossover)):
            average_crossover.append(((parent1.crossover[i] + parent2.crossover[i]) / 2.0))

        # Select crossover
        crossovers = []
        while len(crossovers) != n:
            i = random.randint(0, len(average_crossover) - 1)
            while i != len(average_crossover):
                # Loop back to start of array
                if i == len(average_crossover):
                    i = 0

                # Check if random value is valid
                if random.random() < average_crossover[i]:
                    if i not in crossovers:
                        crossovers.append(i)
                        break
                else:
                    i += 1
        crossovers = sorted(crossovers)

        new_chromosome = []
        for index, crossover in enumerate(crossovers):
            if index == 0:
                new_chromosome += (parent1.chromosome[:crossover])
            elif index == len(crossovers) - 1:
                new_chromosome += (parent2.chromosome[crossovers[index - 1]:])
            else:
                if index % 2 == 0:
                    new_chromosome += (parent1.chromosome[crossovers[index - 1]: crossover])
                else:
                    new_chromosome += (parent2.chromosome[crossovers[index - 1]: crossover])

        return [new_chromosome, average_crossover]

    def mutate_chance(chromosome, chance):
        rand = random.randint(0, 100)

        if rand < chance:
            i = random.randint(0, len(chromosome) - 1)
            mutate = round(random.random(), 2)
            chromosome[i] = mutate

        return chromosome

    def standard_deviation(population):
        chromosomes = [[] for _ in range(11)]

        for individual in population:
            for index, val in enumerate(individual.chromosome):
                chromosomes[index].append(val)

        standard_devs = []
        for c in chromosomes:
            standard_devs.append(statistics.stdev(c))

        for std_dev in standard_devs:
            if std_dev < 0.005:
                return True

        return False

    def the_purge(pop):
        fitness_scores = []
        for individual in pop:
            fitness_scores.append(individual.fitness)

        if statistics.stdev(fitness_scores) < 0.2:
            for individual in random.sample(old_population, int(len(pop)/7)):
                individual.chromosome = np.random.random(11).tolist()

        return pop

    # Social cleansing (THE PURGE)
    old_population = the_purge(old_population)

    # Perform elitism
    new_population = elitism(old_population, 7)
    sums = [0] * 11
    chance = 5
    incest = 0
    while len(new_population) < len(old_population):
        # Select new parents via tournament selection
        winner1 = tournament_select(random.sample(old_population, int(len(old_population) / 7)))
        winner2 = tournament_select(random.sample(old_population, int(len(old_population) / 7)))

        # Ensure parents aren't the exact same (one person can't make a child)
        while winner1 == winner2:
            winner2 = tournament_select(random.sample(old_population, int(len(old_population) / 7)))

        # Crossover parents
        child_chromosome, child_crossover = multi_crossover(winner1, winner2, 4, len(new_population))

        # Mutate child chromosome
        if len(new_population) % 4 == 0 and len(new_population) != 0:
            # Check SD
            if standard_deviation(new_population):
                chance += 2
        child_chromosome = mutate_chance(child_chromosome, chance)

        # Create new child
        new_individual = MyCreature(numCreaturePercepts, numCreatureActions)
        new_individual.chromosome = child_chromosome
        new_individual.crossover = child_crossover

        # Add new child to new population
        new_population.append(new_individual)

        for index, val in enumerate(new_individual.chromosome):
            sums[index] += val

    averages = [0] * 11
    for index, val in enumerate(sums):
        averages[index] = round(val/len(new_population), 2)

    for index, val in enumerate(averages):
        archive[index].append(val)

    # print(archive)

    print("MMA:\t" + str(averages[0]) + "\n" +
          "MMC:\t" + str(averages[1]) + "\n" +
          "MMD:\t" + str(averages[2]) + "\n" +
          "CMA:\t" + str(averages[3]) + "\n" +
          "CMC:\t" + str(averages[4]) + "\n" +
          "CMD:\t" + str(averages[5]) + "\n" +
          "FMA:\t" + str(averages[6]) + "\n" +
          "FMC:\t" + str(averages[7]) + "\n" +
          "FMD:\t" + str(averages[8]) + "\n" +
          "EAT:\t" + str(averages[9]) + "\n" +
          "RAN:\t" + str(averages[10]))
    #
    # for i in new_population[-5:]:
    #     print(i.chromosome)

    archive[13].append(incest)
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
#w.show_simulation(titleStr='Initial population', speed='normal')

for i in range(numGenerations):
    print("\nGeneration %d:" % (i+1))

    # Create a new population from the old one
    population = newPopulation(population)

    # Pass the new population to the world simulator
    w.setNextGeneration(population)

    # Run the simulation again to evalute the next population
    w.evaluate(numTurns)

    # Show visualisation of final generation
    # if i==numGenerations-1:
    #     w.show_simulation(titleStr='Final population', speed='slow')

# plt.plot(archive[0], color='blue', label="MMA")
# plt.plot(archive[1], color='green', label="MMC")
# plt.plot(archive[2], color='blue', label="CMA")
# plt.plot(archive[3], color='green', label="CMC")
# plt.plot(archive[4], color='blue', label="FMA")
# plt.plot(archive[5], color='green', label="FMC")
# plt.plot(archive[6], color='blue', label="EAT")
# plt.plot(archive[7], color='green', label="RAND")
plt.plot(archive[11], color='green', label="Survivors")
plt.plot(archive[12], color='red', label="Fitness")
plt.plot(archive[13], color='blue', label="Incest")
plt.ylabel('Generational average')
plt.title('\n' + str(nSurvivors) + ' survivors - ' + str(int(float(nSurvivors)/len(population) * 100)) + ' percent\nOn average ' + str(int(np.mean(archive[11]))) + ' creatures survived')
plt.legend(loc='upper left')
plt.savefig('report/images/morning_all_half_food_rand_test.png')
plt.show()