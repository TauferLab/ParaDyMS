#!/bin/bash

# Generates the dynamic graph datasets
# ex: ./GenDynGraph data/wikivote.edgelist data/Ins100 1000 100 0

#SBATCH -J gendyn           # Job name
#SBATCH -o gendyn.o%j       # Name of stdout output file
#SBATCH -e gendyn.e%j       # Name of stderr error file
#SBATCH -p vm-small          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for OpenMP)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for OpenMP)
#SBATCH -t 23:59:59        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=aliyk@tacc.utexas.edu
#SBATCH -A CCR23044       # Project/Allocation name (req'd if you have more than 1)


input_root=$WORK
output_root=$WORK
seed=42

# compile program just in case
g++ generate_graph.cpp -o GenDynGraph -std=c++17


# declare -a urls=("https://statml.com/download/data_7z/heter/livejournal.7z" "https://statml.com/download/data_7z/road/road-road-usa.7z" "https://nrvis.com/download/data/soc/soc-orkut.zip" "https://statml.com/download/data_7z/soc/soc-flickr.7z" "https://nrvis.com/download/data/massive/soc-friendster.zip" "https://snap.stanford.edu/data/wiki-Vote.txt.gz")
# declare -a graphs=( "livejournal.edgelist" "roadusa.edgelist" "orkut.edgelist" "flicker.edgelist" "friendster.edgelist" "wikivote.edgelist")
declare -a graphs=( "livejournal.edgelist" "roadusa.edgelist" "orkut.edgelist" "flickr.edgelist" "wikivote.edgelist" )
declare -a no_changed_edges=( 10000 20000 50000 100000 1000000 )
declare -a size_names=( "10K" "20K" "50K" "100K" "1M" )
declare -a percent_insertion=( 0 25 50 75 100 )

# get length of an array
len_a1=${#graphs[@]}
len_a2=${#no_changed_edges[@]}
len_a3=${#percent_insertion[@]}

# use for loop to read all values and indexes
for (( i=0; i<${len_a1}; i++ ));
do
    for (( j=0; j<${len_a2}; j++ ));
    do
        for (( k=0; k<${len_a3}; k++ ));
        do
            output_dir="${output_root}/dynamic_graphs/${size_names[$j]}/${percent_insertion[$k]}"
            mkdir -p ${output_dir}
            echo "\nProcessing ($i, $j, $k): ${graphs[$i]} num_updated_edges = ${size_names[$j]} ins_percent = ${percent_insertion[$k]} seed = ${seed}"
            ./GenDynGraph ${input_root}/${graphs[$i]} ${output_dir} ${no_changed_edges[$j]} ${percent_insertion[$k]} ${seed}
            # ./cE-undirected /work/08434/apandey/ls6/SCC-new/Datasets/OriginalReady/${1} ${n} ${no_changed_edges[$i]} ${percent_insertion[$j]} > /work/08434/apandey/ls6/SCC-new/Datasets/ChangedEdges/${1}_ce/${1}_${size_names[$i]}_${percent_insertion[$j]}
        done
    done
    #echo "index: $i, value: ${array[$i]}"
done
