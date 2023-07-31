# xmodmap package

Package for estimating geometric and functional mappings between multiple modalities of data. Important applications include mapping tissue scale atlases to micron and submicron scale cellular and transcriptional data. 

## Current Modeling Assumptions

- **Data:** particles, each with position and feature value (discrete feature space)
- **Geometric Mapping:** rigid+scale and non-rigid deformation (diffeomorphism)
- **Functional Mapping:** single distribution over target feature space associated to each feature value in source space 

## Project structure

- **xmodmap.io:** datatypes, reading and saving
- **xmodmap.optimizer:** optimizing functions, default is pytorch LBFGS
- **xmodmap.deformation:** deformation models and functions
- **xmodmap.distances:** fixed point costs including varifold distance (matching term), regularizer for functional mapping (kl divergence), and regularizer for support estimation parameter 
- **xmodmap.models:** selected parameters and mappings to estimate

- **examples:** examples of running the code
- **tests:** unit tests

## Quick start

Geometric Mapping Only example: 

- 2D to 2D
```
python examples/example_2D_single_celltype.py
```
- 3D to 3D
```
python examples/example_3D_single_BEIALE_MTL.py
```

Geometric and Functional Mappings
- 2D to 2D
```
python examples/example_2D_cross_ratToMouseAtlas.py
```

- 3D to 3D
```
python examples/example_3D_cross_B2ToB5.py
```

# Authors

- Kaitlin Stouffer (kstouff4@jhmi.edu)
- Alain Trouvé (alain.trouve@ens-paris-saclay.fr)
- Benjamin Charlier (benjamin.charlier@umontpellier.fr)

# License

[MIT License](https://www.mit.edu/~amini/LICENSE.md)