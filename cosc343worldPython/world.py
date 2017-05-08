#!/usr/bin/env python
from cosc343world import Creature, World
import numpy as np
import random
import time

# You can change this number to specify how many generations creatures are going to evolve over...
numGenerations = 500

# You can change this number to specify how many turns in simulation of the world for given generation
numTurns = 100

# You can change this number to change the percept format.  You have three choice - format 1, 2 and 3 (described in
# the assignment 2 pdf document)
perceptFormat=1

# You can change this number to chnage the world size
gridSize=24

# You can set this mode to True to have same initial conditions for each simulation in each generation.  Good
# for development, when you want to have some determinism in how the world runs from generatin to generation.
repeatableMode=False

# This is a class implementing you creature a.k.a MyCreature.  It extends the basic Creature, which provides the
# basic functionality of the creature for the world simulation.  Your job is to implement the AgentFunction
# that controls creature's behavoiur by producing actions in respons to percepts.
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
        self.north_west = [round(random.random(), 2)]
        self.north = [round(random.random(), 2)]
        self.north_east = [round(random.random(), 2)]
        self.east = [round(random.random(), 2)]
        self.south_east = [round(random.random(), 2)]
        self.south = [round(random.random(), 2)]
        self.south_west = [round(random.random(), 2)]
        self.west = [round(random.random(), 2)]
        self.stay = [round(random.random(), 2)]
        self.eat = [round(random.random(), 2)]
        self.random = [round(random.random(), 2)]

        # Generate cross over intercept probabilities.
        self.crossover = np.random.random(11)

        # Chromosome is in format: [nw, n, ne, e, se, s, sw, w, stay, eat, random]
        self.mutate = [round(random.random(), 2)]
        self.crossover /= sum(self.crossover)
        self.chromosome = self.north_west + self.north + self.north_east + self.west + self.stay + self.east + \
            self.south_west + self.south + self.south_east + self.eat + self.random

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
        # actions = np.random.uniform(0, 1, size=numActions)
        actions = self.chromosome
        # actions = [None] * numActions

        # Get the list containing the location of each type around our creature.
        monsters = [i + 1 for i, x in enumerate(percepts[0:9]) if x == 1]
        creatures = [i + 1 for i, x in enumerate(percepts[9:18]) if x == 1]
        food = [i + 1 for i, x in enumerate(percepts[18:27]) if x >= 1]

        for monster in monsters:
            actions[monster - 1] -= round(random.random(), 2)

        for creature in creatures:
            actions[creature - 1] -= round(random.random(), 2)

        for meal in food:
            if meal == 4:
                actions[9] += 2000
            else:
                actions[meal - 1] += round(random.random(), 2)

        if not monsters and not creatures and not food:
            actions[10] += round(random.random(), 2)

        return actions#.tolist()


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
    # energy did the creature have a the end of simualation (0 if dead), tick number
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
        else:
            nSurvivors += 1
            avgLifeTime += numTurns

    # Here are some statistics, which you may or may not find useful
    avgLifeTime = float(avgLifeTime)/float(len(population))
    print("Simulation stats:")
    print("  Survivors    : %d out of %d" % (nSurvivors, len(population)))
    print("  Avg life time: %.1f turns" % avgLifeTime)

    # The information gathered above should allow you to build a fitness function that evaluates fitness of
    # every creature.  You should show the average fitness, but also use the fitness for selecting parents and
    # creating new creatures.


    # Based on the fitness you should select individuals for reproduction and create a
    # new population.  At the moment this is not done, and the same population with the same number
    # of individuals
    new_population = old_population

    return new_population

# Create the world.  Representaiton type choses the type of percept representation (there are three types to chose from);
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
w.show_simulation(titleStr='Initial population', speed='normal')

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
        w.show_simulation(titleStr='Final population', speed='normal')


