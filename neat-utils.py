from neat import Agent, Genome, Gene, Node, Species

def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_model():
	filename = input("Enter the filename")
	with open(filename, 'rb') as inp:
		walker = pickle.load(inp)
		# showFittest()
		walker.genome.display_gene()
		env.close()
	return walker

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
		breedCount = int(x.repFitness / totalFitness * gen_iterations) - 1
		for i in range(breedCount):
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
				childOrganism = Agent(i_shape, o_shape, childGenome)
				children.append(childOrganism)
			elif random.random() < crossover_rate and len(x.organisms) > 1:
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
				childOrganism = Agent(i_shape, o_shape, childGenome)
				# TODO: random enable disable genes
				children.append(childOrganism)
			else:
				xx = random.randrange(0, len(x.organisms))
				childGenome = copy.deepcopy(x.organisms[xx].genome)
				childGenome.mutate()
				childOrganism = Agent(i_shape, o_shape, childGenome)
				children.append(childOrganism)
				reproduction_count += 1
	# print(len(children))
	# input("children")
	for species in next_generation:
		species.reduce()
	for organism in children:
		speciate(organism)

	for species in next_generation:
		species.print()
	



def generateInitialPopulation():
	global next_generation
	next_generation = []
	children = []
	a = Agent(i_shape, o_shape)
	for _ in range(gen_iterations):
		b = copy.deepcopy(a)
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
	np.save('plots.npy', fitness_plots)
	prev_status = env.reset()
	if save:
		save_object(organism, "walkers/Fittest.pkl")
		save_object(organism.genome,"walkers/genome.pkl")
	for timestep in range(1000):
		env.render()
		
		action = env.action_space.sample()
		action = np.argmax(organism.genome.predict(prev_status))
		
		status, reward, done, info = env.step(action)
		score += reward
		prev_status = status
		if done:
				break
	print("Fittest Organism in Generation (Species "+ str(sp.number) + " )score ", str(score))
	print("Number of nodes: ", str(len(organism.genome.nodes)))
	print("Number of Genes: ", str(len(organism.genome.genes)))
	organism.genome.display_gene()

	pass
