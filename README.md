# xmodmap package

Package for estimating geometric and functional mappings between multiple modalities of data. Important applications include mapping tissue scale atlases to micron and submicron scale cellular and transcriptional data. 

Current Modeling Assumptions:
Data: particles, each with position and feature value (discrete feature space)
Geometric Mapping: rigid+scale and non-rigid deformation (diffeomorphism)
Functional Mapping: single distribution over target feature space associated to each feature value in source space 

Directories:
io: datatypes, reading and saving
optimizer: optimizing functions, default is pytorch LBFGS
deformation: deformation models and functions
distances: fixed point costs including varifold distance (matching term), regularizer for functional mapping (kl divergence), and regularizer for support estimation parameter 
models: selected parameters and mappings to estimate

Dependencies:
pytorch
pykeops
numpy
scipy
matplotlib

Examples (../examples/):
2D to 2D geometric mapping only (example_2D_single_celltype.py)
3D to 3D geometric mapping only (example_3D_single_BEIALE_MTL.py)
2D to 2D geometric mapping and functional mapping (example_2D_cross_ratToMouseAtlas.py)
3D to 3D geometric mapping and functional mapping (example_3D_cross_B2ToB5.py)

