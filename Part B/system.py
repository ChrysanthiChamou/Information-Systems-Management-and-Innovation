from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.log.util import log as utils
from pm4py.objects.log.obj import EventLog
import matplotlib.pyplot as plt
import pandas as pd


import os
import random

os.environ["PATH"] += os.pathsep + r"C:\Program Files\windows_10_cmake_Release_Graphviz-14.1.2-win32\Graphviz-14.1.2-win32\bin"


log = xes_importer.apply("BPI Challenge 2017.xes")

subset_size = 75
subset_list = random.sample(log, subset_size)
subset_log = EventLog(subset_list)

print("Number of traces:", len(subset_log))

total_events = sum(len(trace) for trace in subset_log)
print("Total number of events:", total_events)

#Structure of trace & event

print("\n===== STRUCTURE =====")
print("Trace attributes:", list(subset_log[0].attributes.keys()))
print("Event attributes:", list(subset_log[0][0].keys()))


# 📌 Unique activities

activities = utils.get_event_labels(subset_log, "concept:name")
print("\nUnique activities:", activities)
print("Number of unique activities:", len(activities))


#Print events

print("\n===== EVENTS =====")
for trace in subset_log:
    case_id = trace.attributes.get("concept:name", "UNKNOWN")
    for event in trace:
        print(case_id,
              event.get("concept:name", "UNKNOWN"),
              event.get("time:timestamp", "UNKNOWN"))


#Alpha Miner

net_a, initial_marking_a, final_marking_a = alpha_miner.apply(subset_log)

gviz = pn_visualizer.apply(net_a, initial_marking_a, final_marking_a)
pn_visualizer.save(gviz, "Alpha_miner.png")


#Heuristics Miner

net_h, initial_marking_h, final_marking_h = heuristics_miner.apply(subset_log)

gviz = pn_visualizer.apply(net_h, initial_marking_h, final_marking_h)
pn_visualizer.save(gviz, "Heuristics_miner.png")


#Inductive Miner

tree = inductive_miner.apply(subset_log)

net_i, initial_marking_i, final_marking_i = pt_converter.apply(
    tree,
    variant=pt_converter.Variants.TO_PETRI_NET
)

gviz = pn_visualizer.apply(net_i, initial_marking_i, final_marking_i)
pn_visualizer.save(gviz, "Inductive_miner.png")


#Conformance checking (Inductive model)

print("\n===== ALIGNMENTS (Inductive) =====")
aligned_traces = alignments.apply_log(subset_log, net_i, initial_marking_i, final_marking_i)
print(aligned_traces[:5])   # δείγμα

print("\n===== TOKEN REPLAY (Inductive) =====")
replayed_traces = token_replay.apply(subset_log, net_i, initial_marking_i, final_marking_i)
print(replayed_traces[:5])

#Evaluation for each model

models = {
    "Alpha": (net_a, initial_marking_a, final_marking_a),
    "Heuristics": (net_h, initial_marking_h, final_marking_h),
    "Inductive": (net_i, initial_marking_i, final_marking_i)
}

results = []

for name, (net, initial_marking, final_marking) in models.items():

    fitness = replay_fitness.apply(subset_log, net, initial_marking, final_marking)["averageFitness"]
    precision = precision_evaluator.apply(subset_log, net, initial_marking, final_marking)
    generalization = generalization_evaluator.apply(subset_log, net, initial_marking, final_marking)
    simplicity = simplicity_evaluator.apply(net)

    results.append([name, fitness, precision, generalization, simplicity])

#Evaluation table

print("\n===== EVALUATION TABLE =====")
print("Algorithm\tFitness\tPrecision\tGeneralization\tSimplicity")

for r in results:
    print(f"{r[0]}\t\t{r[1]:.3f}\t{r[2]:.3f}\t\t{r[3]:.3f}\t\t{r[4]:.3f}")

res_df = pd.DataFrame(
    results,
    columns=["Model", "Fitness", "Precision", "Generalization", "Simplicity"]
)

res_df.set_index("Model")[["Fitness","Precision","Generalization","Simplicity"]].plot(kind="bar")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("metrics_comparison.png")
plt.show()