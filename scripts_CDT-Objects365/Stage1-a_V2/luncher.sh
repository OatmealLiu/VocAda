# stable
for i in {1..48}; do
  sbatch scripts_CDT-Objects365/Stage1-a_V2/a_chunk_llava_captioning.sh $i
done

for i in {49..100}; do
  sbatch scripts_CDT-Objects365/Stage1-a_V2/a_chunk_llava_captioning.sh $i
done


# random launched
for i in {49..100}; do
  sbatch scripts_CDT-Objects365/Stage1-a_V2/be_a_chunk_llava_captioning.sh $i
done


for i in {30..48}; do
  sbatch scripts_CDT-Objects365/Stage1-a_V2/be_a_chunk_llava_captioning.sh $i
done


sbatch scripts_CDT-Objects365/Stage1-a_V2/a_chunk_llava_captioning.sh 58
sbatch scripts_CDT-Objects365/Stage1-a_V2/a_chunk_llava_captioning.sh 61

