Building MPC ULG project
==============================


# the goal

the goal is to practice statistical inference and later to learn model predictive control.

# the project

## what is given

a data set of a house temperature behaviour given time, heater command.

## what is expected

EVERYTHING IS PIPELINED FROM THE START

### building thermal response

a building response model should be proposed:

- base model
- model with single room or all the rooms
- the sensor themselves have a reading model

### statistical inference model

- write a full statistical inference pipeline with model, optimizer, bayesian modeling

### vizualisation

- propose a simple vizualisation (I think the phase plot is good)
- set up a little dashboard

### model predictive control

- build a little simulator
- given the price of fuel and a metric of confort, use pyzo to optimize the mpc given a 24h weather/occupancy etc forecast
- try to think about uncertainty
- compare the concumption to a binary command

# How this project should be done

## code

This project should not be some notebooks, but a good quality code with documentation, modularity,

## teaching

This project should leave a how-to manual on organizing a project: which classes to create, are decorators of any use, is cookiecutter useful, how to copy code from tutorials, how to keep trace of the project structure, how prototyping and mlops differ, how to use vscode to code faster.

In a nutshell, which questions should the modeler asks himself and how and how long time to do it.

## nice to have

- get rid of the data and build a data generator with a model (different from the later infered)

# what was done/tried

- the source code is in the [/old folder](old/src)
- my first attempt lead to big classes either [with arrays](old/src/models/model_On_Array2.py) or [with networkX graphs]([old/src/models/Model_On_G.py])

