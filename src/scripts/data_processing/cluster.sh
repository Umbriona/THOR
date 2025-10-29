#!/bin/bash

FASTA_FILE="training_set.fasta"



CLU_DIR="../clusters"

CLUSTERS=(70 60 50 40)


echo "running clustering script"
for CLU_P in ${CLUSTERS[@]};
do
echo ${CLU_P}
mkdir -p ${CLU_DIR}/clu_${CLU_P}
time mmseqs easy-cluster ${FASTA_FILE} ${CLU_DIR}/clu_${CLU_P}/ tmp --min-seq-id "0.${CLU_P}" -c 0.8 --cov-mode 1

mkdir -p ${CLU_DIR}/clu_${CLU_P}/top_100_${CLU_P}

cat  ${CLU_DIR}/clu_${CLU_P}/_cluster.tsv | cut -f 1 | uniq -c | sort -n | tail -n 100 | rev | cut -f 1 -d " " | rev > ${CLU_DIR}/clu_${CLU_P}/top_100_clust.txt

for CLU in `cat ${CLU_DIR}/clu_${CLU_P}/top_100_clust.txt`
do echo -e "col1\tcol2" > ${CLU_DIR}/clu_${CLU_P}/top_100_${CLU_P}/cluster_${CLU}.tsv
cat ${CLU_DIR}/clu_${CLU_P}/_cluster.tsv | grep "^${CLU}" >> ${CLU_DIR}/clu_${CLU_P}/top_100_${CLU_P}/cluster_${CLU}.tsv
done

done