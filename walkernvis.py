
import gym
import random
import copy
import numpy as np
import scipy.special
import logging
from statistics import mean, median
from collections import Counter
import math
import sys
import os
import neat
from neat import Agent, Genome, Gene, Node, Species
import vis


from matplotlib import pyplot as plt
from matplotlib import style
exceeded = False
agent_name = "Walker"

def createdir(name):
	try:
		if not os.path.exists(name):
			os.makedirs(name)
		pass
	except OSError:
		print("Error While creating a directory")
		pass

createdir(agent_name)
createdir(agent_name+"/agents")
createdir(agent_name+"/agents/fittest")
createdir(agent_name+"/agents/fittest/genomes")
createdir(agent_name+"/logs")
createdir(agent_name+"/graphs")
createdir(agent_name+"/dump")

gen_iterations = 500
gen_timesteps = 400
max_generations = 5000


generation = []

# environment specific hyperparameters
stagnation_rate = 15
mating_rate = 0.001
crossover_rate = 0.75



env = gym.make('BipedalWalker-v2')

i_shape = 14
o_shape = 4

style.use('ggplot')
plt.title('Max Fitness in Each Generation')
plt.xlabel('Generations')
plt.ylabel('Fitness')

fitness_plots = []

logger = logging.getLogger('myapp')
hdlr = logging.FileHandler(agent_name+'/logs/applog.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

genelog = logging.getLogger('genes')
hdlr = logging.FileHandler(agent_name+'/logs/genes.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
genelog.addHandler(hdlr)
genelog.setLevel(logging.INFO)

wLog = logging.getLogger('wlog')
hdlr = logging.FileHandler(agent_name+'/logs/warns.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
wLog.addHandler(hdlr)
wLog.setLevel(logging.INFO)

mLog = logging.getLogger('mlog')
hdlr = logging.FileHandler(agent_name+'/logs/mutate.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
mLog.addHandler(hdlr)
mLog.setLevel(logging.INFO)


	


next_generation = []
prev_generation = []
def calculate_fitness():
	# find each agents fitness values
	pass


def calculateAverageFitness(generation):
	for species in generation:
		total = 0
		for organism in species.organisms:
			total += organism.global_fitness
		species.repFitness = total/species.population


def totalAverageFitness(generation):
	total = 0
	for species in generation:
		total += species.repFitness
	return total

reproduction_count = 0

def speciate(org):
	global next_generation
	for x in next_generation:
		if Species.share(org.genome, x.organisms[0].genome) == 1:
			x.add(org)
			wLog.info('Organism added to existing species - ' + str(x.number))
			return x.number
	s = Species()
	s.add(org)
	wLog.info('Organism added to new species - ' + str(s.number))
	next_generation.append(s)
	
	return s.number


def extinction():
	global next_generation
	# min_fitness = -200
	totals = []
	for x in next_generation:
		totals.append(x.repFitness)
	m = median(totals)
	for i, x in enumerate(next_generation):
		if x.repFitness < m:
			next_generation.pop(i)

def removeStagnantSpecies():
	global next_generation
	for i, x in enumerate(next_generation):
		if x.generationsSinceImrpovement >= stagnation_rate:
			next_generation.pop(i)


def updateFitness():

	pass
def updateStagnationInformation():
	global next_generation
	for i, x in enumerate(next_generation):
		if(x.repFitness >= x.lastFitness):
			x.lastFitness = x.repFitness
			x.generationsSinceImrpovement = 0
		else:
			x.generationsSinceImrpovement += 1

def newGeneration():
	
	global reproduction_count
	reproduction_count = 0
	global next_generation
	# if len(next_generation) > 1:
	# 	if len(next_generation) < 10:
	# 		neat.species_threshold -= 0.3
	# 	elif len(next_generation) >= 10:
	# 		neat.species_threshold +=0.3
	children = []
	calculateAverageFitness(next_generation)
	updateStagnationInformation()
	removeStagnantSpecies()
	calculateAverageFitness(next_generation)
	totalFitness = totalAverageFitness(next_generation)
	for x in next_generation:
		genelog.info("[Species : "+str(x.number) +" Organisms: "+str(len(x.organisms))+" ]")
		x.sort()
		if(len(x.organisms) > 10):
			x.removeUnfit()
		genelog.info("[Trim Step- Organisms Survived: "+str(len(x.organisms))+" ]")
		breedCount = int((x.repFitness / totalFitness) * gen_iterations) - 1
		for i in range(breedCount):
			if random.random() < crossover_rate and len(x.organisms) > 1:
				genelog.info("[ORGANISM]")
				xx = random.randrange(0, len(x.organisms))
				xy = random.randrange(0, len(x.organisms))
				while xx == xy:
					xx = random.randrange(0, len(x.organisms))
					xy = random.randrange(0, len(x.organisms))

				if x.organisms[xy].global_fitness > x.organisms[xx].global_fitness:
						temp = xx
						xx = xy
						xy = temp
				childGenome = Genome.crossover(
									x.organisms[xx].genome, x.organisms[xy].genome)
				reproduction_count += 1
				# apply random chance of further mutation
				childGenome.mutate()
				childGenome.mutate_topology()
				childOrganism = Agent(i_shape, o_shape, childGenome,agent_name)
				# TODO: random enable disable genes
				children.append(childOrganism)
			else:
				xx = random.randrange(0, len(x.organisms))
				childGenome = copy.deepcopy(x.organisms[xx].genome)
				childGenome.mutate()
				childOrganism = Agent(i_shape, o_shape, childGenome,agent_name)
				children.append(childOrganism)
				reproduction_count += 1

	if random.random() < mating_rate and len(next_generation) > 1:
		print("Interspecies Breeding")
		xx = x.organisms[0]
		xy = next_generation[random.randrange(0, len(next_generation))].organisms[0]
		while xx == xy:
			xy = next_generation[random.randrange(0, len(next_generation))].organisms[0]
		childGenome = Genome.crossover(xx.genome, xy.genome)
		reproduction_count += 1
		childGenome.mutate()
		childGenome.mutate_topology()
		childOrganism = Agent(i_shape, o_shape, childGenome, agent_name)
		children.append(childOrganism)

	for species in next_generation:
		species.reduce()
	
	for organism in children:
		speciate(organism)
	global exceeded
	# if len(next_generation) >= 10 and exceeded == False:
	# 	neat.species_threshold += 0.3
	# 	exceeded = True

	



def generateInitialPopulation():
	global next_generation
	next_generation = []
	children = []
	a = Agent(i_shape, o_shape,False,agent_name)
	a.genome.initcon()
	for _ in range(gen_iterations):
		b = copy.deepcopy(a)

		b.genome.mutate_topology()
		b.genome.mutate()
		children.append(b)
		# s.add(b)
	for organism in children:
		speciate(organism)
	input("Population Generated Continue?...")

def printGeneration(x):
	print("Current Generation", str(x))
	print("Number of Species: ", str(len(next_generation)))
	norg = 0
	
	for species in next_generation:
		print("species ", str(species.number), " Organisms: ",str(len(species.organisms)))
		print(species.repFitness)
		norg += len(species.organisms)
	print("Organisms: ", str(norg))
	print("Breed Count: ", str(reproduction_count))

def findFittest():
	global next_generation
	fittest = 0
	s = next_generation[0]
	max_fitness = -200
	for species in next_generation:
		for organism in species.organisms:
			if organism.global_fitness > max_fitness:
				fittest = organism
				max_fitness = organism.global_fitness
				s = species
	return fittest,s


def showFittest(gen, save = False):
	
	organism,sp = findFittest()
	prev_status = []
	score = 0

	global fitness_plots
	fitness_plots.append([gen, organism.global_fitness])
	np.save(agent_name+'/dump/plots.npy', fitness_plots)
	prev_status = env.reset()
	prev_status = prev_status[:-10]
	if save:
		neat.save_object(organism, agent_name+"/agents/fittest/Fittest.pkl")
		neat.save_object(organism.genome,agent_name+"/agents/fittest/genomes/genome.pkl")
	vis.reset()
	for timestep in range(gen_timesteps):
		env.render()
		
		action = env.action_space.sample()
		action = organism.genome.predict(prev_status)
		
		vis.draw(organism.genome)
		status, reward, done, info = env.step(action)
		score += reward
		prev_status = status[:-10]
		if done:
				break
	print("Fittest Organism in Generation (Species "+ str(sp.number) + " )score ", str(score))
	print("Number of nodes: ", str(len(organism.genome.nodes)))
	print("Number of Genes: ", str(len(organism.genome.genes)))
	organism.genome.display_gene()

	pass

def showOrganism(org):
	
	organism = org

	prev_status = []
	score = 0
	prev_status = env.reset()
	prev_status = prev_status[:-10]
	for timestep in range(1500):
		env.render()
		action = env.action_space.sample()
		action = organism.genome.predict(prev_status)
		status, reward, done, info = env.step(action)
		score += reward
		prev_status = status[:-10]
		if done:
				break
	print("Fittest Organism in Generation: score ", str(score))
	print("Number of nodes: ", str(len(organism.genome.nodes)))
	print("Number of Genes: ", str(len(organism.genome.genes)))
	organism.genome.display_gene()
	pass



def run_generation(x,display=False):
	counter = 0
	if x == 0:
		# first generation
		logger.info('Started New Training Session')
		logger.info('Running Generation ' + str(x))
		generateInitialPopulation()
		printGeneration(x)
	else:
		# other generations
		logger.info('Running Generation ' + str(x))
		newGeneration()
		printGeneration(x)
	for i,species in enumerate(next_generation):
		for j,organism in enumerate(species.organisms):
			prev_status = []
			score = 0
			prev_status = env.reset()
			prev_status = prev_status[:-10]
			for timestep in range(gen_timesteps):
				if display:
					env.render()
				action = env.action_space.sample()
				action = organism.genome.predict(prev_status)
				status, reward, done, info = env.step(action)
				score += reward
				prev_status = status[:-10]
				if done:
					counter += 1
					break
			organism.global_fitness = score + 200
			vis.draw_status(x,counter,organism.global_fitness,i,len(next_generation),organism.genome)
			vis.reset()
	showFittest(x,True)


for i in range(max_generations):
	run_generation(i)
env.close()

o = neat.load_model()
showOrganism(o)

