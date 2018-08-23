# Python Implementation of the Neuro Evolution of Augmenting Topologies Algorithm, (k stanely)
# By Elvin Dsouza
# 16/04/2018

import random
import copy
import numpy as np
import logging
from statistics import mean, median
from collections import Counter
import math
try:
	import cPickle as pickle
except ModuleNotFoundError:
	import pickle


current_session = "Anonymous"

# Hyper Parameters
# Alter based on the generation size

# mutation rates for various different types of mutations
# rate that a node gets added splitting an existing connection
mutation_add_rate = 0.03
# rate that a link gets added
mutation_link_rate = 0.05
# rate of altering a links weights by a gaussian distribution
mutation_weight_rate = 0.8
mutation_refresh_rate = 0.1

stagnation_rate = 15

# chance of breeding happening over just mutation which is 1 - crossover_chance
crossover_rate = 0.75

perturbance_rate = 0.1

# interspecies mating rate
mating_rate = 0.001

# threshold value for distance between two species
species_threshold = 1.35

# constants to decide on which component affects compatibility
coeff_excess = 1.0
coeff_disjoint = 1.0
coeff_weights = 0.3


# Activation Functions

# Sigmoid activation Function

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# TODO: implement relu instead of sigmoid
def relu(x):
	if x > 0:
		return x
	else:
		return 0.01*x

# utility function to create a directory
def createdir(name):
	try:
		if not os.path.exists(name):
			os.makedirs(name)
		pass
	except OSError:
		print("Error While creating a directory")
		pass


# logging stubs
logger = logging.getLogger('myapp')
genelog = logging.getLogger('genes')
wLog = logging.getLogger('wlog')
mLog = logging.getLogger('mlog')




class Gene:
	
	def __init__(self, input_node, output_node ,weight, enabled,innovation_number):
		self.input_node = input_node
		self.output_node = output_node
		self.weight = weight
		self.enabled = enabled
		self.innovation_number = innovation_number
		pass


	def display(self):
		print("-----------------------------------")
		print("Gene | Innovation Number: ",str(self.innovation_number))
		print("In: ",str(self.input_node),"  Out: ", str(self.output_node))
		print("Weight: ",str(self.weight))
		print("enabled: ",str(self.enabled))

	def match(self, bgenes):
		if self.innovation_number == bgenes.innovation_number:
			return True
		else:
			return False

	def disjoint(self, genes):
		found = False
		for gene in genes:
			if self.innovation_number == gene.innovation_number:
				found = True
		if found:
			return False
		else:
			return True


class Node:

	def __init__(self, node_number, ntype):
		self.node_number = node_number
		self.ntype = ntype
		self.activated = 0
		self.output_value = 0.0
		pass


	def display(self):
		print("-----------------------------------")
		print("Node Number: ",str(self.node_number))
		print("Node Type: ",str(self.ntype),"  Activated: ", str(self.activated))
		print("Output Value: ",str(self.output_value))

		pass



	def connected(self,b,genes):
		for gene in genes:
			if gene.input_node == b.node_number:
				if gene.output_node == self.node_number:
					return True

			if gene.output_node == b.node_number:
				if gene.input_node == self.node_number:
					return True


		return False

		


class Genome:
	innovation = 0
	def __init__(self, num_inputs, num_outputs, species_no = -1):
	
		self.nno = 1
		self.visited = []
		self.species_no = species_no
		self.genes = []
		self.fitness = 0.0
		self.nodes = []
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		for x in range(num_inputs):
			self.nodes.append(Node(self.nno,0))
			self.nno += 1
		for x in range(num_outputs):
			self.nodes.append(Node(self.nno,2))
			# for node in self.nodes:
			# 	if node.ntype == 0:
			# 		Genome.innovation+= 1
			# 		self.genes.append(Gene(node.node_number,self.nno,random.uniform(0,1),1,Genome.innovation))
			# print("current innovation_number" + str(local_innovation))
			self.nno += 1
		pass
	def initrandom(self, num = 5):
		for _ in range(num):
			x1 = self.random_node(True, False)
			x2 = self.random_node(False,True)
			while x1 == x2:
				x1 = self.random_node(True, False)
				x2 = self.random_node(False,True)
			Genome.innovation+= 1
			self.genes.append(Gene(x1,x2,0.0,1,Genome.innovation))



	def initcon(self):
		for inode in self.nodes:
			if inode.ntype == 0:
				for onode in self.nodes:
					if onode.ntype == 2:
						Genome.innovation+= 1
						self.genes.append(Gene(inode.node_number,onode.node_number,0.0,1,Genome.innovation))


	
	
	def split_nodes(self, nodea, nodeb):
		for gene in self.genes:
			if gene.input_node == nodea.node_number:
				if gene.output_node == nodeb.node_number:
					gene.enabled = 0
					node = Node(self.nno, 1)
					Genome.innovation += 1
					split_gene_a = Gene(gene.input_node, self.nno, 1, 1, Genome.innovation)
					Genome.innovation += 1
					split_gene_b = Gene(self.nno, gene.output_node,
										gene.weight, 1, Genome.innovation)
					self.genes.append(split_gene_a)
					self.genes.append(split_gene_b)
					self.nodes.append(node)
					self.nno += 1


	def join_nodes(self, nodea, nodeb):
		
		if nodea.connected(nodeb, self.genes):
			wLog.warning("Both Nodes are Conected")
			return False
		if nodea.ntype == 2 and nodeb.ntype == 2:
			wLog.warning("Node A and Node B Both are output")
			return False
		if nodea.ntype == 2 and nodeb.ntype == 1:
			wLog.warning("Node A is output and node B is input")
			return False
			
		Genome.innovation +=1
		joint = Gene(nodea.node_number, nodeb.node_number,
					 random.uniform(-1, 1), 1, Genome.innovation)
		self.genes.append(joint)
		genelog.info("New Gene Created, Innovation Number: "+ str(Genome.innovation))
		self.visited = []
		if self.getPrevious(nodeb) == True:
			self.genes.pop()
			genelog.info("Gene Having Innovation Number " +str(Genome.innovation) + "Deleted (Loop)")
			Genome.innovation -=1 
			return False
		else:
			genelog.info("[LINK MUTATION]Connection created between " + str(nodea.node_number) + " and " + str(nodeb.node_number))
			return True
		

	def testSet1(self):
		self.split_nodes(self.nodes[0], self.nodes[4])
		self.split_nodes(self.nodes[1], self.nodes[4])
		self.split_nodes(self.nodes[0], self.nodes[6])
		Genome.innovation += 1
		self.genes.append(Gene(7, 8, random.uniform(-1, 1), 1, Genome.innovation))
		# Genome.innovation += 1
		# self.genes.append(Gene(8, 9, random.uniform(0, 1), 1, Genome.innovation))

	
	def getPrevious(self, node):
		if(node.ntype == 0):
			return False
		if len(self.visited) > 0:
			for x in self.visited:
				if x == node.node_number:
					return True	

		self.visited.append(node.node_number)
		output = False
		for gene in self.genes:
			if gene.enabled == 1:
				if node.node_number == gene.output_node:
					output = self.getPrevious(self.nodes[gene.input_node-1])
		return output

	def feedforward(self,node):
		connections = []
		output = 0
		if node.ntype == 0:
			return node.output_value

		for gene in self.genes:
			if gene.output_node == node.node_number and gene.enabled == True:
				connections.append([gene.input_node - 1, gene.weight])

		if len(connections) == 0:
			return 0
		else:
			for link in connections:
				out = 0
				if self.nodes[link[0]].ntype == 0:
					out = self.nodes[link[0]].output_value
				else:
					out = self.feedforward(self.nodes[link[0]])
				output += (out * link[1])
			return relu(output)

	def predict(self, inputs):
		# print("predict")
		for inode in self.nodes:
			if inode.ntype == 0:
				inode.output_value = inputs[inode.node_number-1]
		outputs = []
		for onode in self.nodes:
			if onode.ntype == 2:
				self.visited = []
				if self.getPrevious(onode) == True: #kill the species if it has a loop
					self.fitness-=9999
					genelog.warning("Organism unable to continue")
					outputs = [0.0,0.0,0.0,0.0]
				else:
					outputs.append(self.feedforward(onode))
		return outputs

	def mutate(self):
		for gene in self.genes:
			if random.random() < perturbance_rate:
				gene.weight += random.gauss(0,perturbance_rate)
				genelog.warning(str(gene.innovation_number)+" - Mutated to weight"+str(gene.weight))
			elif random.random() < perturbance_rate:
				gene.weight =random.uniform(-1,1)
				genelog.warning(str(gene.innovation_number) +" -Random Mutated to weight"+str(gene.weight))
			
		pass

	def random_node(self , ipIncluded = False,  opIncluded = False):
		pos = -1
		pos = random.randrange(0, len(self.nodes))
		node = self.nodes[pos]
		if ipIncluded == False and opIncluded == False: # only hidden
			while node.ntype == 0 or node.ntype == 2:
				pos = random.randrange(0, len(self.nodes))
				node = self.nodes[pos]
		elif ipIncluded == True and opIncluded == False:  # all excluding output
			# input("outside loop"+str(node.node_number))
			while node.ntype == 2:
				pos = random.randrange(0, len(self.nodes))
				node = self.nodes[pos]
		elif ipIncluded == False and opIncluded == True:  # all excluding input
			while node.ntype == 0:
				pos = random.randrange(0, len(self.nodes))
				node = self.nodes[pos]
		return pos


	def mutate_topology(self):
		# add node mutation
		if random.random() < mutation_add_rate:
				nodea = self.nodes[self.random_node(True, False)] # the first node can be an input but not an output
				nodeb = self.nodes[self.random_node(False, True)] # the second node can be an output but not an input
				self.split_nodes(nodea,nodeb) #split the two nodes this method auto creates the genes
				genelog.info("[SPLIT MUTATION] Connection created between " +str(nodea.node_number) + " and " + str(nodeb.node_number))
		if random.random() < mutation_link_rate:
			# if self.nno != (self.num_inputs + self.num_outputs + 1): 
				# the first node can be an input but not an output
				nodea = self.nodes[self.random_node(True, False)]
				# the second node can be an output but not an input
				nodeb = self.nodes[self.random_node(False, True)]
				result = self.join_nodes(nodea, nodeb)
				counter = 0
				while result == False:
					if counter < self.nno * 2:
						nodea = self.nodes[self.random_node(True, False)]
						nodeb = self.nodes[self.random_node(False, True)]
						result = self.join_nodes(nodea, nodeb)
						counter+=1
					else:
						
						break
	

	# TODO:assume excess also as disjoint, will simplify computation and reduce one loop

	def calculateSpeciesDelta(self, xgenes):

		bgenes = []
		sgenes = []
		weight_differences = []
		num_excess = 0
		num_disjoint = 0
		excess_component = 0
		disjoint_component = 0
		weight_component = 0
		sdelta = 0.0
		N = 0

		# TODO: figure out how to sort out when one organism has no genes

		if len(self.genes) == 0 and len(xgenes) == 0:
			return 0
		elif len(self.genes) >= len(xgenes):
			N = len(self.genes)
			bgenes = self.genes
			sgenes = xgenes
		else:
			N = len(xgenes)
			bgenes = xgenes
			sgenes = self.genes
		total = 0
		for bgene in bgenes:
			# total += abs(bgene.weight)
			for sgene in sgenes:
				if bgene.innovation_number == sgene.innovation_number:
					weight_differences.append(abs(bgene.weight - sgene.weight))
					# logger.info("Weight difference "+ str(abs(bgene.weight - sgene.weight)))
		num_disjoint = Genome.ndisjoint(bgenes,sgenes)
		# print(num_disjoint)
		if N < 20:
			N = 1
		# excess_component = coeff_excess * num_excess
		disjoint_component = coeff_disjoint * num_disjoint / N
		# logger.info("Coeff="+str(coeff_disjoint)+" "+str(num_disjoint)+" "+ str(N))
		
		if len(weight_differences) > 0:
			weight_component = coeff_weights * mean(weight_differences)
		else:
			weight_component = coeff_weights * 0.1

		sdelta = disjoint_component + weight_component
		logger.info("Delta: "+str(sdelta)+ " " + str(disjoint_component) +" "+ str(weight_component))

		return sdelta

	def sortGenes(self):
		for passnum in range(len(self.genes)-1, 0, -1):
			for i in range(passnum):
				if self.genes[i].innovation_number > self.genes[i+1].innovation_number:
					temp = self.genes[i]
					self.genes[i] = self.genes[i+1]
					self.genes[i+1] = temp
		pass


	@staticmethod
	def same(agenes, bgenes):
		outgenes = []
		for i, ag in enumerate(agenes):
			for j, bg in enumerate(bgenes):
				if(i!=j):
					if ag.innovation_number == bg.innovation_number:
						choice = random.randrange(0, 2)
						if choice == 1:
							outgenes.append(ag)
						elif choice == 0:
							outgenes.append(bg)
		return outgenes
						
	@staticmethod
	def ndisjoint(agenes,bgenes):
		l = Genome.disjoint(agenes,bgenes)
		return len(l)

	@staticmethod
	def disjoint(agenes,bgenes):
		outgenes = []
		for i, ag in enumerate(agenes):
			for j, bg in enumerate(bgenes):
				if(i != j):
					if ag.innovation_number == bg.innovation_number:
						choice = random.randrange(0, 2)
						if choice == 1:
							outgenes.append(ag)
						elif choice == 0:
							outgenes.append(bg)
						agenes.pop(i)
		for ag in agenes:
			outgenes.append(ag)
		return outgenes
		
	@staticmethod
	def crossover(parentA, parentB):
		# assuming parent A is the fitter parent
		child = copy.deepcopy(parentA)
		child.visited = []
		child.genes = []
		dgenes = Genome.disjoint(parentA.genes, parentB.genes)
		child.genes.extend(dgenes)
		child.sortGenes()
		return child


	def display(self):
		for node in self.nodes:
			node.display()
		for gene in self.genes:
			gene.display()
	
	def display_gene(self):
		print("nodes")
		for node in self.nodes:
			print(str(node.node_number)+"\t", end="")
		print("\n\n")
		for gene in self.genes:
			print(str(gene.innovation_number)+"\t", end="")
		print("\n")
		for gene in self.genes:
			print(str(gene.input_node)+ " -> "+ str(gene.output_node)+"\t", end="")
		print("\n")
		for gene in self.genes:
			if gene.enabled == True:
				print(str(1)+" "+str(gene.weight)+"\t", end="")
		print("\n")


class Agent:
	def __init__(self, input_shape, output_shape, genome = False, name='Anonymous'):
		global current_session
		global mlog
		global wLog
		global genelog
		global logger
		
		self.score = 0
		self.fitness = 0.0
		self.global_fitness = 0.0
		current_session = name
		if not genome:
			self.genome = Genome(input_shape, output_shape)
			hdlr = logging.FileHandler(name+'/logs/applog.log')
			formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
			hdlr.setFormatter(formatter)
			logger.addHandler(hdlr)
			logger.setLevel(logging.INFO)

			hdlr = logging.FileHandler(name+'/logs/genes.log')
			formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
			hdlr.setFormatter(formatter)
			genelog.addHandler(hdlr)
			genelog.setLevel(logging.INFO)

			hdlr = logging.FileHandler(name+'/logs/warns.log')
			formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
			hdlr.setFormatter(formatter)
			wLog.addHandler(hdlr)
			wLog.setLevel(logging.INFO)

			hdlr = logging.FileHandler(name+'/logs/mutate.log')
			formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
			hdlr.setFormatter(formatter)
			mLog.addHandler(hdlr)
			mLog.setLevel(logging.INFO)
		else:
			self.genome = copy.deepcopy(genome)


		# logging
		
		pass

class Species:
	speciesNumber = 0
	def __init__(self):
		self.organisms = []
		self.population = 0
		self.generationsSinceImrpovement = 0
		self.lastFitness = -99999
		Species.speciesNumber += 1
		self.number = Species.speciesNumber
		self.repFitness = 0.0
		pass
	
	def add(self, agent):
		self.organisms.append(agent)
		self.population+=1
	
	def sort(self):
		for passnum in range(len(self.organisms)-1, 0, -1):
			for i in range(passnum):
				if self.organisms[i].fitness < self.organisms[i+1].fitness:
					temp = self.organisms[i]
					self.organisms[i] = self.organisms[i+1]
					self.organisms[i+1] = temp
		pass
	
	def removeUnfit(self):
		
		if len(self.organisms) > 5:
			self.sort()
			# print("Called")
			unfit = int(len(self.organisms)/2)
			# print(unfit)
			self.organisms = self.organisms[:-unfit]
	
	def reduce(self):
		temp =  self.organisms[0]
		self.organisms = []
		self.organisms.append(temp)
		self.population = 1

	@staticmethod
	def share(a,b):
		
		d = a.calculateSpeciesDelta(b.genes)
		if d > species_threshold:
			wLog.info('Organism doesnt share - '+ str(d))
			return 0
		else:
			wLog.info('Organism shares - '+str(d))
			return 1

	def print(self):
		print("---------------------------------------------------")
		print("Species Number: ", str(self.number))
		print("Fitness: ", str(self.repFitness))
		print("Number of organisms (Population): ", len(self.organisms))
		print("---------------------------------------------------")


def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_model():
	# filename = input("Enter the filename")
	filename = 'Walker/agents/good/wtf?/Fittest.pkl'
	with open(filename, 'rb') as inp:
		walker = pickle.load(inp)
		# showFittest(0)
		walker.genome.display_gene()
		# env.close()
	return walker


def load_walk():
	# filename = input("Enter the filename")
	filename = 'Walker/agents/interesting/firstleap/Fittest.pkl'
	with open(filename, 'rb') as inp:
		walker = pickle.load(inp)
		# showFittest(0)
		walker.genome.display_gene()
		# env.close()
	return walker