#!/usr/bin/env python
from cosc343worldcc import _cCreature, _cWorld
import numpy as np
import time
import sys

# This is a creature class that your EvolvingCreature needs to inherit from
# This class wraps the _cCreature class which was implemented in C.
class Creature(_cCreature):

    # Your child class must override this method, where the
    # mapping of percepts to actions is implemented
    def AgentFunction(self, percepts, nActions):
        print("Your EvolvingCreature needs to override the AgentFunction method!")
        sys.exit(-1)

    # Agent function evoked from the simulation of the world implemented in C.
    # This method translates the percepts to python list, and translates back
    # the list representing the actions into C format.
    def internal_AgentFunction(self):

        # Get the number of percepts and actions
        nPercepts = self.numPercepts()
        nActions = self.numActions()

        # Create lists of percepts
        percepts = np.zeros((nPercepts))
        for i in range(nPercepts):
            percepts[i] = self.getPercept(i)

        # Execute the AgentFunction method that needs to be implemented
        # by the EvolvingCreature.  Pass in the list of percepts and
        # specify the number of actions expected.
        actions = self.AgentFunction(percepts, nActions)

        if not isinstance(actions, list) or len(actions) != nActions:
            print("Error!  Expecting the actions returned from the AgentFunction to be a list of %d numbers." % nActions)

        # Translate actions and feed it back to the engine
        for i in range(nActions):
            self.setAction(i, actions[i])

# Wrapper class for _cWorld which implements the engine for the simulation
class World(_cWorld):

   # Initialisation wrapper with some defaults for representation type, grid size
   # and repeatability setting.
   def __init__(self, representationType=0, gridSize=24, repeatable=False):
      self.ph = None
      super().__init__(representationType, gridSize, repeatable)

   # Feed the next generation of creatures to the simulation
   #
   # Input: population - a list of creatures for simulation
   def setNextGeneration(self, population):
      self.resetCreatures()
      for i in range(len(population)):
         self.addCreature(population[i])

   # Animation of the simulation
   #
   # Input: titleStr - title string of the simulation
   #        speed - of the simulation: can be 'slow', 'normal' or 'fast'
   def show_simulation(self, titleStr = "", speed='normal'):
      import pygame
      gridSize = self.gridSize()
      left_frame = 100

      pygame.init()

      size = width, height = 720, 480
      WHITE = (255, 255, 255)
      BLACK = 0, 0, 0

      if speed == "normal":
          frameTurns = 20
          nSteps = 10
      elif speed == "fast":
          frameTurns = 1
          nSteps = 5
      elif speed == "slow":
          frameTurns = 40
          nSteps = 10

      screen = pygame.display.set_mode(size)

      unit = int(np.min([width-left_frame, height])/gridSize)

      im_strawbs = [pygame.image.load('images/strawberry-green.png'),
                    pygame.image.load('images/strawberry-red.png')
                   ]

      im_creatures = [pygame.image.load('images/smiley_happy.png'),
                      pygame.image.load('images/smiley_hungry.png'),
                      pygame.image.load('images/smiley_sick.png')
                     ]

      for i in range(len(im_strawbs)):
          im_strawbs[i] = pygame.transform.scale(im_strawbs[i], (unit, unit))

      for i in range(len(im_creatures)):
          im_creatures[i] = pygame.transform.scale(im_creatures[i], (unit, unit))

      im_monster = pygame.transform.scale(pygame.image.load("images/monster.png"), (unit, unit))

      nTurns = self.vis_numTurns()
      stepDiff = 1.0/float(nSteps)

      nFood = self.vis_num(0)
      nCreatures = self.vis_num(1)
      nMonsters = self.vis_num(2)

      nBodies = [nFood, nCreatures, nMonsters]

      halfSteps = int(np.floor(nSteps/2))

      for t in range(1, nTurns + 1):

          pygame.display.set_caption("%s (turn %d)" % (titleStr, t))

          for k in range(nSteps):

              for event in pygame.event.get():
                  if event.type == pygame.QUIT: sys.exit()


              screen.fill(WHITE)

              for i in range(gridSize + 1):
                 pygame.draw.line(screen, BLACK, [left_frame, i*unit], [left_frame+(gridSize*unit), i*unit])
                 pygame.draw.line(screen, BLACK, [left_frame+(i*unit), 0], [left_frame+(i*unit), gridSize * unit])

              for type in range(3):
                  for i in range(nBodies[type]):
                      x = self.vis(type, 0, i, t)
                      y = self.vis(type, 1, i, t)
                      s = self.vis(type, 2, i, t)

                      xprev = self.vis(type, 0, i, t-1)
                      yprev = self.vis(type, 1, i, t-1)

                      xshift = xprev-x
                      if np.abs(xshift)<=1:
                          xdiff = (x - xprev) * k * stepDiff
                      elif k <= halfSteps:
                          xdiff = np.sign(xshift) * k * stepDiff
                      else:
                          xdiff = -np.sign(xshift) * k * stepDiff
                          xprev = x

                      yshift = yprev - y
                      if np.abs(yshift) <= 1:
                          ydiff = (y - yprev) * k * stepDiff
                      elif k <= halfSteps:
                          ydiff = np.sign(yshift) * k * stepDiff
                      else:
                          ydiff = -np.sign(yshift) * k * stepDiff
                          yprev = y

                      if type==0:
                          if s >= 0 and s <= 1:
                              obj_loc = pygame.Rect(left_frame + (x * unit), y * unit, unit, unit)
                              obj_im = im_strawbs[s]
                              screen.blit(obj_im, obj_loc)

                      elif type==1:
                          if s > 0:
                              obj_im = im_creatures[s-1]
                              obj_loc = pygame.Rect(left_frame + (xprev + xdiff) * unit, (yprev + ydiff) * unit, unit,
                                                    unit)
                              screen.blit(obj_im, obj_loc)


                      elif type==2:
                          obj_loc = pygame.Rect(left_frame+(xprev + xdiff) * unit, (yprev + ydiff) * unit, unit, unit)
                          screen.blit(im_monster, obj_loc)

              pygame.display.flip()
              pygame.time.delay(frameTurns)
      pygame.display.quit()
      pygame.quit()
