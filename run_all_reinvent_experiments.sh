#!/bin/bash
set -e
batch_size=100

steps=150
lr=0.0001
seed=1234
scoring=clogp
for filter in NoFilter IdenticalMurckoScaffold CompoundSimilarity IdenticalTopologicalScaffold ScaffoldSimilarity; do
        python ~/projects/reinvent-memory/reinforce_model.py --scaffold-filter ${filter} --minscore 0.6  --minsimilarity 0.6 --name "clogP ${filter}" --scoring  ${scoring} --range "2.0-3.0" --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2_transfered_highclogP/prior.epoch20.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed}
done

steps=500

lr=0.0005
seed=1234
scoring=activity
target=DRD2
for filter in NoFilter IdenticalMurckoScaffold CompoundSimilarity IdenticalTopologicalScaffold ScaffoldSimilarity; do
    python ~/projects/reinvent-memory/reinforce_model.py --scaffold-filter ${filter} --minscore 0.6  --minsimilarity 0.6 --name "${target} ${filter}" --scoring  ${scoring} --clf-path  ~/projects/reinvent-classifiers/${target}_final.pkl --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2/prior.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed}
done

lr=0.0001
seed=1234
scoring=activity
target=HTR1A
for filter in NoFilter IdenticalMurckoScaffold CompoundSimilarity IdenticalTopologicalScaffold ScaffoldSimilarity; do
        python ~/projects/reinvent-memory/reinforce_model.py --scaffold-filter ${filter} --minscore 0.6  --minsimilarity 0.6 --name "${target} ${filter}" --scoring  ${scoring} --clf-path  ~/projects/reinvent-classifiers/${target}_final.pkl --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2/prior.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed}
done

#sudo shutdown
