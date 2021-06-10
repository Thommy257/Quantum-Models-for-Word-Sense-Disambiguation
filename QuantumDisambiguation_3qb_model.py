import json
import datetime, os

from math import pi
from random import uniform
from time import time

import numpy as np
from discopy import CircuitFunctor, Cup, Diagram, Id, Ty, Word, qubit
from discopy.grammar import brute_force
from discopy.quantum import CX, Circuit, CRz, H, Ket, Rx, Rz, Ry, X, sqrt, C, SWAP
from pytket.extensions.qiskit import AerBackend, tk_to_qiskit

from noisyopt_batches import minimizeSPSA_batches
from words import ambiguous_verbs
from words import nouns as nouns
from words import sentences as sentences

# Ansätze for 3-qubit states
n_amb_params = 9
def amb_verb_ansatz(p):

    ## U3 ansazt
    # return Ket(0,0,0) >> \
    #     Rz(p[0]) @ Rx(p[1]) @ Circuit.id(1) >> \
    #     Rx(p[2]) @ Rz(p[3]) @ Circuit.id(1) >> \
    #     CX @ Circuit.id(1) >> \
    #     Rx(p[4]) @ Rz(p[5]) @ Rx(p[6]) >> \
    #     Rz(p[7]) @ Rx(p[8]) @ Rz(p[9]) >> \
    #     Circuit.id(1) @ CX >> \
    #     Circuit.id(1) @ Rx(p[10]) @ Rz(p[11]) >> \
    #     Circuit.id(1) @ Rz(p[12]) @ Rx(p[13]) >> \
    #     Circuit.id(1) @ Circuit.id(1) @ Circuit.id(1) @ sqrt(2)
    
    # Strongly Entangling Layer ansatz
    return Ket(0,0,0) >> \
        Rx(p[0]) @ Rx(p[1]) @ Rx(p[2]) >> \
        Ry(p[3]) @ Ry(p[4]) @ Ry(p[5]) >> \
        Rz(p[6]) @ Rz(p[7]) @ Rz(p[8]) >> \
        CX  @ Circuit.id(1) >> \
        Circuit.id(1) @ CX >> \
        Circuit.id(1) @ SWAP >> \
        SWAP @ Circuit.id(1) >> \
        CX @ Circuit.id(1) >> \
        SWAP @ Circuit.id(1) >> \
        Circuit.id(1) @ SWAP
        
    

n_unamb_ansatz = 3
def unamb_verb_ansatz(p):
    return Ket(p[0],p[1],p[2])

n_noun_params = 3
def noun_ansatz(p):
    return Ket(0,0,0) >> \
    Ry(p[0]) @ Ry(p[1]) @ Ry(p[2])


def prepare_dataset():
    dataset = []
    nouns, verbs = set(), set()
    for sentence in sentences:
        _s = sentence.split(" ")
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


def prepare_init_params(unamb_verbs, nouns, learn_verbs = False):
    # prepare intitial params
    binaries = [list(bin(i)[2:]) for i, verb in enumerate(unamb_verbs)]
    int_binaries = [[0]*(n_unamb_ansatz - len(bi)) + [int(b) for b in bi] for bi in binaries]

    n_nouns = len(nouns)
    n_amb_verbs = len(ambiguous_verbs)
    n_unamb_verbs = len(unamb_verbs)
    
    params_unamb_verbs = {verb: {"p": int_binaries[i], "learn": False} for i, verb in enumerate(unamb_verbs)}
    params_nouns = {noun: {"p": [uniform(0, 1) for i in range(n_noun_params)], "learn": True} for noun in nouns}
    params_amb_verbs = {verb: {"p": [uniform(0, 1) for i in range(n_amb_params)], "learn": learn_verbs} for verb in ambiguous_verbs}

    init_params = {**params_unamb_verbs, **params_nouns, **params_amb_verbs}


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
        if point[0] in ambiguous_verbs:
            circ = amb_verb_ansatz(params[point[0]]["p"]) >> noun_ansatz(params[point[1]]["p"]).dagger()
        else:
            circ = unamb_verb_ansatz(params[point[0]]["p"]) >> noun_ansatz(params[point[1]]["p"]).dagger()

        res = Circuit.eval(
                circ,
                backend=backend,
                n_shots=2**10,
                seed=0,
                compilation=backend.default_compilation_pass(2))
        loss.append( (point[2]-np.abs(res.array)[0])**2 )
    return sum(loss)/len(loss)


def compare(params, dataset):

    backend=AerBackend()
    result = {}
    for point in dataset:
        if point[0] in ambiguous_verbs:
            circ = amb_verb_ansatz(params[point[0]]["p"]) >> noun_ansatz(params[point[1]]["p"]).dagger()
        else:
            circ = unamb_verb_ansatz(params[point[0]]["p"]) >> noun_ansatz(params[point[1]]["p"]).dagger()

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
            "nouns":n_noun_params,
            "amb_verbs": n_amb_params},
        bounds=[0,1],
        niter=niter,
        paired=False,
        a=0.2,
        c=0.1
        )

    return result


def main():

    folder = './wxperiments/3_qb_model_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(folder)

    batch_size = 4

    # define DisCo Types
    s, v = Ty('s'), Ty('v')
    dataset, nouns, verbs = prepare_dataset()
    unamb_verbs = [verb for verb in verbs if verb not in ambiguous_verbs]

    # map different words to DisCo Categories
    noun_voc = {noun: Word(noun, v.r @ s) for noun in nouns}
    verb_voc = {verb: Word(verb, v) for verb in verbs}
    vocab = {**noun_voc, **verb_voc}

    # define grammar rule
    grammar = Cup(v, v.r) @ Id(s)

    parsing = parse_dataset(dataset, vocab, grammar)

    init_params = prepare_init_params(unamb_verbs, nouns, learn_verbs=False)

    # fit nouns first
    result_n = fit(init_params, dataset, batch_size, niter=250)
    # print("Final MSE: ", result_n.fun)

    fiel_path_noun_results = os.path.join(folder,'params_noun_fit.json')
    with open(fiel_path_noun_results, 'w') as fp:
        json.dump(result_n.x, fp)

    fiel_path_noun_evo = os.path.join(folder,'evo_nouns.json')
    with open(fiel_path_noun_evo, 'w') as fp:
        json.dump(result_n.evo, fp)

    # fit verbs after nouns
    params_after_noun_fit = {key:{"p":result_n.x[key]["p"], "learn": True if key in ambiguous_verbs else False} for key in result_n.x}
    reduced_dataset_amb = [point for point in dataset if point[0] in ambiguous_verbs]
    result = fit(params_after_noun_fit, reduced_dataset_amb, batch_size, niter=500)

    final_loss = evaluate(result.x, dataset)
    print("Final MSE:", final_loss)

    fiel_path_results = os.path.join(folder,'params.json')
    with open(fiel_path_results, 'w') as fp:
        json.dump(result.x, fp)

    fiel_path_verbs_evo = os.path.join(folder,'evo.json')
    with open(fiel_path_verbs_evo, 'w') as fp:
        json.dump(result.evo, fp)

    fiel_path_loss = os.path.join(folder,'final_loss.json')
    with open(fiel_path_loss, 'w') as fp:
        json.dump(final_loss, fp)

    params_after_verb_fit = {key:{"p":result.x[key]["p"], "learn": False} for key in result.x}
    comp = compare(params_after_verb_fit, dataset)

    fiel_path_comp = os.path.join(folder,'results.json')
    with open(fiel_path_comp, 'w') as fp:
        json.dump(comp, fp)

    print("finished!")


if __name__ == "__main__":
    main()



