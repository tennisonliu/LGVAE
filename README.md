# Latent-Graph VAE: Modelling Latent Graph Interactions to Generate Drug Targets

PyTorch Implemenation of Latent-Graph VAE. For more information, see our poster:
[LGVAE DRUG DISCOVERY Presentation](graphs/final_graphs/BNN_Poster.pdf)

## Code Structure
Defining/training/evaluation of individual models are actioned through experiment scripts e.g. `LGVAE-M.py`, the main entry point for molecule LGVAE model. The script imports layers and networks defined in `models.py` and data loading utilities from `data_utils.py`. Raw data is downloaded in `data/` and trained models are subsequently saved in `saved_models/`.

Experiment scripts include:
* `LGVAE-M.py`,
* `LGVAE-P.py`: LGVAE training for protein graphs,
* `prediction_model.py`: GNN training to predict molecule's drug-like properties,
* `actor_critic_generation.py`: LGAIN training to model interaction between protein and molecule latent graphs.

## Authors
* Tennison Liu
* Liying Qiu
* Mingze Yao

## TODO: 
* Refactor `reg_task.py`, `class_task` into base and derived classes / sort out inheritance.