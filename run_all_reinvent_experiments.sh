#!/bin/bash
set -e
batch_size=100

steps=500

lr=0.0005
seed=1234
scoring=activity
target=DRD2
minsimilarity=0.6
bucket_size=25
temperature=1.0
experience_replay=False
outputmode=binary
python=`which python`
rm -f jobs
rm -f jobs.sorted

batch_size=100

steps=150
lr=0.0001
seed=1234
scoring=clogp
for filter in NoFilter IdenticalMurckoScaffold CompoundSimilarity IdenticalTopologicalScaffold ScaffoldSimilarity; do
    echo ${python} ~/projects/reinvent-memory/reinforce_model.py --scaffold-filter ${filter} --minscore 0.6  --minsimilarity 0.6 --name "clogP ${filter}" --scoring  ${scoring} --range "2.0-3.0" --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2_transfered_highclogP/prior.epoch20.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed} >> jobs
done

steps=500

lr=0.0005
seed=1234
scoring=activity
target=DRD2
for filter in NoFilter IdenticalMurckoScaffold CompoundSimilarity IdenticalTopologicalScaffold ScaffoldSimilarity; do
    echo ${python} ~/projects/reinvent-memory/reinforce_model.py --scaffold-filter ${filter} --minscore 0.6  --minsimilarity 0.6 --name "${target} ${filter}" --scoring  ${scoring} --clf-path  ~/projects/reinvent-classifiers/${target}_final.pkl --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2/prior.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed} >> jobs
done

lr=0.0001
seed=1234
scoring=activity
target=HTR1A
for filter in NoFilter IdenticalMurckoScaffold CompoundSimilarity IdenticalTopologicalScaffold ScaffoldSimilarity; do
    echo ${python} ~/projects/reinvent-memory/reinforce_model.py --scaffold-filter ${filter} --minscore 0.6  --minsimilarity 0.6 --name "${target} ${filter}" --scoring  ${scoring} --clf-path  ~/projects/reinvent-classifiers/${target}_final.pkl --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2/prior.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed} >> jobs
done




lr=0.0005
seed=1234
scoring=activity
target=DRD2
minsimilarity=0.6
bucket_size=25
temperature=1.0
experience_replay=False
outputmode=binary
for experience_replay in True False; do
    for filter in IdenticalMurckoScaffold CompoundSimilarity IdenticalTopologicalScaffold ScaffoldSimilarity; do
        for bucket_size in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75; do
            echo ${python} ~/projects/reinvent-memory/reinforce_model.py --temperature ${temperature} --experience ${experience_replay} --scaffold-filter ${filter} --minscore 0.6  --minsimilarity ${minsimilarity} --nbmax ${bucket_size} --outputmode ${outputmode} --name '"'${target} ${filter} ${minsimilarity} ${bucket_size} ${outputmode} ${temperature} ${experience_replay}'"' --scoring  ${scoring} --clf-path  ~/projects/reinvent-classifiers/${target}_final.pkl --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2/prior.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed} >> jobs
        done
        bucket_size=25
        for outputmode in binary sigmoid linear; do
            echo ${python} ~/projects/reinvent-memory/reinforce_model.py --temperature ${temperature} --experience ${experience_replay} --scaffold-filter ${filter} --minscore 0.6  --minsimilarity ${minsimilarity} --nbmax ${bucket_size} --outputmode ${outputmode} --name '"'${target} ${filter} ${minsimilarity} ${bucket_size} ${outputmode} ${temperature} ${experience_replay}'"' --scoring  ${scoring} --clf-path  ~/projects/reinvent-classifiers/${target}_final.pkl --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2/prior.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed} >> jobs
        done 
        outputmode=binary
    done
    for filter in CompoundSimilarity ScaffoldSimilarity; do
        for minsimilarity in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
            echo ${python} ~/projects/reinvent-memory/reinforce_model.py --temperature ${temperature} --experience ${experience_replay} --scaffold-filter ${filter} --minscore 0.6  --minsimilarity ${minsimilarity} --nbmax ${bucket_size} --outputmode ${outputmode} --name '"'${target} ${filter} ${minsimilarity} ${bucket_size} ${outputmode} ${temperature} ${experience_replay}'"' --scoring  ${scoring} --clf-path  ~/projects/reinvent-classifiers/${target}_final.pkl --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2/prior.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed} >> jobs
        done
        minsimilarity=0.6
    done
done

for experience_replay in True False; do
    for filter in IdenticalMurckoScaffold CompoundSimilarity IdenticalTopologicalScaffold ScaffoldSimilarity; do
        bucket_size=25
        for outputmode in binary sigmoid linear; do
            echo ${python} ~/projects/reinvent-memory/reinforce_model.py --temperature ${temperature} --experience ${experience_replay} --scaffold-filter ${filter} --minscore 0.6  --minsimilarity ${minsimilarity} --nbmax ${bucket_size} --outputmode ${outputmode} --name '"'${target} ${filter} ${minsimilarity} ${bucket_size} ${outputmode} ${temperature} ${experience_replay}'"' --scoring  ${scoring} --clf-path  ~/projects/reinvent-classifiers/${target}_final.pkl --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2/prior.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed} >> jobs
        done 
        outputmode=binary
    done
done


filter=NoFilter
for temperature in 1.0 1.25 1.5 1.75 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0; do
    echo ${python} ~/projects/reinvent-memory/reinforce_model.py --temperature ${temperature} --experience ${experience_replay} --scaffold-filter ${filter} --minscore 0.6  --minsimilarity ${minsimilarity} --nbmax ${bucket_size} --outputmode ${outputmode} --name '"'${target} ${filter} ${minsimilarity} ${bucket_size} ${outputmode} ${temperature} ${experience_replay}'"' --scoring  ${scoring} --clf-path  ~/projects/reinvent-classifiers/${target}_final.pkl --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2/prior.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed} >> jobs
done 
temperature=1.0
for experience_replay in False True; do
    echo ${python} ~/projects/reinvent-memory/reinforce_model.py --temperature ${temperature} --experience ${experience_replay} --scaffold-filter ${filter} --minscore 0.6  --minsimilarity ${minsimilarity} --nbmax ${bucket_size} --outputmode ${outputmode} --name '"'${target} ${filter} ${minsimilarity} ${bucket_size} ${outputmode} ${temperature} ${experience_replay}'"' --scoring  ${scoring} --clf-path  ~/projects/reinvent-classifiers/${target}_final.pkl --prior ~/projects/reinvent-memory/priors/ChEMBL_withoutDRD2/prior.ckpt --steps ${steps} --batch-size ${batch_size} --lr ${lr} --seed ${seed} >> jobs
done 


experience_replay=False
sort -u jobs > jobs.sorted
rm -f jobs


echo "submit all jobs listed in 'jobs.sorted' to a cluster. If you want to run it locally run 'bash jobs.sorted' "
