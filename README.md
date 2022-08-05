# TemporalMemoryResearch

A repository containing research aiming to understand the fundamental mechanisms of Numenta's temporal memory algorithm.

## Jupic

This repository contains a Julia implementation of Numenta's temporal memory algorithm, as described in pseudo-code [here](https://numenta.com/assets/pdf/temporal-memory-algorithm/Temporal-Memory-Algorithm-Details.pdf). Some minor departures or ambiguities are described in the comments.

To make use of this implementation clone this repo or download [jupic.jl](Scripts/jupic.jl). Include the file in the first line of your script via

```
include("/path/to/jupic.jl")
```

In this script, you can then initialize a temporal memory model as follows:
```
num_cols = 512
cells_per_col = 32

TempMem(
    num_cols,
    cells_per_col;

    # Optional args
    activation_threshold=15,
    initial_permanence=0.3,
    connected_permanence=0.5,
    learning_threshold=12,
    learning_enabled=true,
    permanence_increment=0.05,
    permanence_decrement=0.001,
    predicted_decrement=0.05,
    synapse_sample_size=50,
    initial_segments_per_cell=0
)
```

The temporal memory model accepts indexes of active columns as input. To train on a sequence of characters, we need to associate each character with a random subset of columns. The `encode` function provides an easy way to do this:

```
encoding_size = 16
sequence = repeat("xyz", 100)

# A dictionary assosicating each unqiue character to 16 random columns
encodings = encode(sequence, num_cols, encoding_size)
```

To train the temporal memory model, simply expose it to the characters in order.

```
epochs = 10
for i in 1:epochs
    for char in sequence
        update!(tm, encodings[char])
    end
end
```

Use the function `predicted_columns(tm)` to check what the algorithm expects to see next.

If training is successful,
```
length(intersect(predicted_columns(tm), encodings['x'])) == encoding_size
```

## Notebooks

This repository also contains jupyter notebooks that document our exploration of the temporal memory algorithm. They are as follows:

1. [LearningJuliaHTM.ipynb](Notebooks/LearningJuliaHTM.ipynb) Exploring the Julia package [HierarchicalTemporalMemory](https://github.com/Oblynx/HierarchicalTemporalMemory.jl) which was hard to use because its unicode variable names didn't render.
1. [GhosalHTM.ipynb](Notebooks/GhosalHTM.ipynb) Learning about Dipak Ghosal's Python implementation of the temporal memory algorithm.
1. [TMAsSimpleRandomProj.ipynb](Notebooks/TMAsSimpleRandomProj.ipynb) Investigating the idea that the temporal memory algorithm is essentially a random projection with optimized decision thresholds.
1. [TMThreshExplore.ipynb](Notebooks/TMThreshExplore.ipynb) Understanding the distributions and how the temporal memory network might perform a random projection.
1. [TMThreshModel.ipynb](Notebooks/TMThreshModel.ipynb) Creation of a model that focuses directly on random projection and attempts to optimize the threshold rather than the synapse permanences.
1. [TMThresholdOpt.ipynb](Notebooks/TMThresholdOpt.ipynb) Optimizing the thresholding algorithm for maximum speed.
1. [TMThreshTrain.ipynb](Notebooks/TMThreshTrain.ipynb) Training the threshold algorithm and failing.
1. [TMThreshTrainLonger.ipynb](Notebooks/TMThreshTrainLonger.ipynb) Training the threshold algorithm for longer and still failing. Deciding that this algorithm doesn't work.
1. [JuliaHTMRunTests.ipynb](Notebooks/JuliaHTMRunTests.ipynb) Attempting to understand [HierarchicalTemporalMemory](https://github.com/Oblynx/HierarchicalTemporalMemory.jl) by taking apart and running its test suite. 

1. [JuliaNumentaTemporalMemory.ipynb](Notebooks/JuliaNumentaTemporalMemory.ipynb) Implementing the Nupic pseudo code myself.
1. [GhosalHTMExperiments.ipynb](Notebooks/GhosalHTMExperiments.ipynb) Learning how Dipak Ghosal's code performs synapse reconnections.
1. [HigherOrderSeq.ipynb](Notebooks/HigherOrderSeq.ipynb) Studying how the temporal memory algorithm captures context.
1. [HigherOrderSeq2.ipynb](Notebooks/HigherOrderSeq2.ipynb) Hypothesizing that the temporal memory model is like a massive markov chain with many possible states corresponding to each character.
1. [RepresentationCapacity.ipynb](Notebooks/RepresentationCapacity.ipynb) Trying to find the limit to the number of representation for a character that the algorithm can store. Did not converge and the number of synapses continued to grow throughout training.
