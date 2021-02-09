import random
import threading
import os
import shutil
import sys
import time

from super_tux_env import SuperTuxEnv
from game_model import GameModel
from deap import base, creator, tools


def parse_arguments(argv):
    models = 2
    num_generations = 1
    crossover_prob = 0.5
    mutation_prob = 0.1

    for i in range(len(argv)):
        if argv[i] == "--models":
            i += 1
            if i > len(argv):
                print("Need to specify number of models for --models")
            else:
                models = int(argv[i])
                
    for i in range(len(argv)):
        if argv[i] == "--gens":
            i += 1
            if i > len(argv):
                print("Need to specify number of generations for --gens")
            else:
                num_generations = int(argv[i])
                
    for i in range(len(argv)):
        if argv[i] == "--cx-prob":
            i += 1
            if i > len(argv):
                print("Need to specify probability of crossover for --cx-prob")
            else:
                crossover_prob = int(argv[i])
                
    for i in range(len(argv)):
        if argv[i] == "--mut-prob":
            i += 1
            if i > len(argv):
                print("Need to specify probability of mutation for --mut-prob")
            else:
                mutation_prob = int(argv[i])
                
    return models, num_generations, crossover_prob, mutation_prob


def main():
    models, num_generations, crossover_prob, mutation_prob = parse_arguments(sys.argv)
    
    proc_name, num_envs, port_numbers, load_model, load_path, train_model, \
    tensorboard_logs_path, render, steps, eval_freq = GameModel.parse_arguments(sys.argv)

    if not train_model:
        model = GameModel(SuperTuxEnv)
        model.test()
    else:
        # Create directory for temporary files
        try:
            os.mkdir("tmp")
        except FileExistsError:
            pass

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", GameModel, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        def make_individual():
            return creator.Individual(SuperTuxEnv)

        def clean_population(population):
            for ind in population:
                ind.close()

        toolbox.register("population", tools.initRepeat, list, make_individual)

        print("-----------------------------------------")
        print("Deep Reinforcement Learning Settings")
        print("-----------------------------------------")
        print(f"Models training for {proc_name}")
        print(f"Total timesteps: {steps}")
        print(f"Evaluation Frequency: {eval_freq}")
        print(f"Environments: {num_envs}")
        print(f"Rendering: {'On' if render else 'Off'}")
        print("-----------------------------------------", end="\n\n")

        # Initialize population
        print("Initializing models population...", end="\r")
        pop = toolbox.population(n=models)
        print("Initializing models population - Done", end="\n\n")

        threads = list()

        for generation in range(num_generations):
            print(f"########## Generation {generation} ##########")

            # Train models
            print("Training models...", end="\r")
            for individual in pop:
                thread = threading.Thread(target=GameModel.train, args=(individual,))
                thread.start()
                threads.append(thread)
                time.sleep(5)

            for thread in threads:
                thread.join()

            threads.clear()
            print("Training models - Done")

            # Evaluate population
            print(f"Evaluating models... (0/{len(pop)})", end="\r")
            fitnesses = map(GameModel.evaluate, pop)
            evaluated_models = 0
            for individual, fitness in zip(pop, fitnesses):
                individual.fitness.values = fitness
                evaluated_models += 1
                print(f"Evaluating models... ({evaluated_models}/{len(pop)})", end="\r")
            print("Evaluating models - Done     ")

            # Select individuals to offspring
            print("Selecting models to offspring...", end="\r")
            offspring = toolbox.select(pop, len(pop))
            offspring_params_floats = list()
            for child in offspring:
                child_data, child_params = child.get_params()
                offspring_params_floats.append(GameModel.convert_params_to_floats(child_params))

            print("Selecting models to offspring - Done")

            # Mate even indices children with uneven indices children
            print("Crossing-over models in offspring...", end="\r")
            for child1, child2 in zip(offspring_params_floats[::2], offspring_params_floats[1::2]):
                if random.random() < crossover_prob:
                    toolbox.mate(child1, child2)
            print("Crossing-over models in offspring - Done")

            # Mutate
            print("Mutating selected models in offspring...", end="\r")
            for mutant in offspring_params_floats:
                if random.random() < mutation_prob:
                    toolbox.mutate(mutant)
            print("Mutating selected models in offspring - Done")

            # Replace population with offspring
            print("Replacing models population with offspring...", end="\r")
            for i in range(len(pop)):
                data, params = pop[i].get_params()
                pop[i].set_params(data, GameModel.convert_floats_to_params(offspring_params_floats[i], params))
                del pop[i].fitness.values
            print("Replacing models population with offspring - Done")

            print(f"########## Generation finished ##########", end="\n\n")

        # Close models and clean up their resources
        print(f"Closing models and cleaning resources...", end="\r")
        clean_population(pop)
        shutil.rmtree("tmp", ignore_errors=True)
        print(f"Closing models and cleaning resources - Done")


main()
