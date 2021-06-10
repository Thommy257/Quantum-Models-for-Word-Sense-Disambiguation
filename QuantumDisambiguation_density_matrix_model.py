import json
import datetime, os

from math import pi
from random import uniform
from time import time

import numpy as np
from discopy import CircuitFunctor, Cup, Diagram, Id, Ty, Word, qubit
from discopy.grammar import brute_force
from discopy.quantum import CX, Circuit, CRz, H, Ket, Rx, Rz, Ry, X, sqrt, C, SWAP, CRy
from pytket.extensions.qiskit import AerBackend, tk_to_qiskit

from noisyopt_batches import minimizeSPSA_batches
from words import ambiguous_verbs
from words import nouns as nouns
from words import sentences as sentences
# from words import noun_vectors
# load noun vectors
with open("new_noun_vectors.json", 'r') as fp:
    noun_vectors = json.load(fp)


# Ansätze for 2-qubit states
n_verb_params = 6
def verb_ansatz(p):
    return Ket(0,0) >> \
        Rx(p[0]) @ Rx(p[1]) >> \
        Ry(p[2]) @ Ry(p[3]) >> \
        Rz(p[4]) @ Rz(p[5]) >> \
        CX >> SWAP >> CX >> SWAP 
        
def noun_ansatz(arr):
    a1 = np.linalg.norm(arr[0:2])
    a2 = np.linalg.norm(arr[2:])
    phi1 = np.arccos(a1)/np.pi

    # fix issues with rotations
    rot1 = arr[0:2]/a1
    phi2_cos = np.arccos(rot1[0])/np.pi
    phi2_sin = np.arcsin(rot1[1])/np.pi
    if not np.sign(phi2_cos) == np.sign(phi2_sin):
        phi2_cos *= -1
    rot2 = arr[2: ]/a2
    phi3_cos = np.arccos(rot2[0])/np.pi
    phi3_sin = np.arcsin(rot2[1])/np.pi
    if not np.sign(phi3_cos) == np.sign(phi3_sin):
        phi3_cos *= -1

    return Ket(0,0) >> Ry(phi1) @ Circuit.id(1) >> CRy(phi3_cos) >> X @ Circuit.id(1) >> CRy(phi2_cos) >> X @ Circuit.id(1)


def prepare_dataset():
    dataset = []
    nouns, verbs = set(), set()
    for sentence in sentences:
        _s = sentence.split(" ")
        if _s[0] not in ambiguous_verbs:
            dataset.append([_s[0],_s[1],int(_s[2][1])])
            verbs.add(_s[0])
            nouns.add(_s[1])

    nouns = list(nouns)
    verbs = list(verbs)

    return dataset, nouns, verbs


def parse_dataset(dataset, vocab, grammar):

    parsing = dict()
    start = time()
    for entry in dataset:
        sentence = (entry[0], entry[1])
        # make graph parsings
        verb = vocab.get(entry[0])
        obj = vocab.get(entry[1])
        diagram = verb @ obj >> grammar
        parsing.update({sentence: diagram})

    return parsing


def prepare_init_params(verbs, nouns):
    
    params_nouns = {noun: {"p": noun_vectors[noun], "learn": False} for noun in nouns}
    params_verbs = {verb: {"p": [uniform(0, 1) for i in range(n_verb_params)], "learn": True} for verb in verbs}

    init_params = {**params_nouns, **params_verbs}


    return init_params

def _calc_opti_nouns(dataset):
    optimal_params_nouns = {}
    for entry in dataset:
        verb, noun = entry[0], entry[1]
        if verb not in ambiguous_verbs and entry[2]==1:
            optimal_params_nouns[noun] = params_unamb_verbs[verb]


def evaluate(params, batch):

    backend=AerBackend()
    loss = []
    for point in batch:
        circ = verb_ansatz(params[point[0]]["p"]) >> noun_ansatz(params[point[1]]["p"]).dagger()
        res = Circuit.eval(
                circ,
                backend=backend,
                n_shots=2**10,
                seed=0,
                compilation=backend.default_compilation_pass(2))
        loss.append((point[2]-np.abs(res.array)[0])**2 )
    return sum(loss)/len(loss)


def compare(params, dataset):

    backend=AerBackend()
    result = {}
    for point in dataset:
        circ = verb_ansatz(params[point[0]]["p"]) >> noun_ansatz(params[point[1]]["p"]).dagger()
        res = Circuit.eval(
                circ,
                backend=backend,
                n_shots=2**10,
                seed=0,
                compilation=backend.default_compilation_pass(2))
        result[point[0] + " " + point[1] + "."] = [point[2], np.abs(res.array)[0]]
    return result


def fit(init_params, dataset, batch_size, niter=100):

    # retrieve #batch_size sentences from dataset without amb_verbs
    def batch_generator():
        while True:
            ind = np.random.randint(len(dataset),size=(batch_size))
            yield [dataset[i] for i in ind]
    
    result = minimizeSPSA_batches(
        evaluate, 
        init_params,
        batch_generator(),
        dataset,
        param_stats={
            "nouns": 0,
            "amb_verbs": n_verb_params},
        bounds=[0,1],
        niter=niter,
        paired=False,
        a=0.2,
        c=0.1
        )

    return result


def main():

    folder = './parameters_full_noun_vec/' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(folder)

    batch_size = 4

    # define DisCo Types
    s, v = Ty('s'), Ty('v')
    dataset, nouns, verbs = prepare_dataset()

    # map different words to DisCo Categories
    noun_voc = {noun: Word(noun, v.r @ s) for noun in nouns}
    verb_voc = {verb: Word(verb, v) for verb in verbs}
    vocab = {**noun_voc, **verb_voc}

    # define grammar rule
    grammar = Cup(v, v.r) @ Id(s)

    parsing = parse_dataset(dataset, vocab, grammar)

    init_params = prepare_init_params(verbs, nouns)

    # fit nouns first
    result = fit(init_params, dataset, batch_size, niter=250)
    final_loss = evaluate(result.x, dataset)
    print("Final MSE: ", result.fun)
   

    fiel_path_results = os.path.join(folder,'params.json')
    with open(fiel_path_results, 'w') as fp:
        json.dump(result.x, fp)

    fiel_path_evo = os.path.join(folder,'evo.json')
    with open(fiel_path_evo, 'w') as fp:
        json.dump(result.evo, fp)

    fiel_path_loss = os.path.join(folder,'final_loss.json')
    with open(fiel_path_loss, 'w') as fp:
        json.dump(final_loss, fp)

    comp = compare(result.x, dataset)

    fiel_path_comp = os.path.join(folder,'results.json')
    with open(fiel_path_comp, 'w') as fp:
        json.dump(comp, fp)

    print("finished!")


if __name__ == "__main__":
    main()



