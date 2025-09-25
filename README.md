# FLEX-C
Forest of Local EXpert Classifiers (FLEX-C) is a semi-supervised framework for fault detection and identification designed to be robust when fault data is scarce and normal operation data is potentially contaminated. 

# Implementation Notice
FLEX-C models subclass the scikit-learn `IsolationForest` to inherit its optimized ensemble implementation. They can be used as a standard Isolation Forest with the usual `fit` and `predict` methods.  

- Use `inject_knowledge` to supervisedly train the model after the unsupervised `fit` initialization.  
- Use `predict_labels` to obtain a classification output.

A complete usage example is provided in the `publication_experiments` folder.

