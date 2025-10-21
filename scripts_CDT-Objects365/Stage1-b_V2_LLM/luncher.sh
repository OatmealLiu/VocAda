# stable
for i in {1..47}; do
  sbatch scripts_CDT-Objects365/Stage1-b_V2_LLM/b2_V2_llm_proposing_100_chunks.sh $i
done

for i in {48..100}; do
  sbatch scripts_CDT-Objects365/Stage1-b_V2_LLM/be_b2_V2_llm_proposing_100_chunks.sh $i
done


for i in {30..47}; do
  sbatch scripts_CDT-Objects365/Stage1-b_V2_LLM/be_b2_V2_llm_proposing_100_chunks.sh $i
done



sbatch scripts_CDT-Objects365/Stage1-b_V2_LLM/be_b2_V2_llm_proposing_100_chunks.sh 28
sbatch scripts_CDT-Objects365/Stage1-b_V2_LLM/be_b2_V2_llm_proposing_100_chunks.sh 29
