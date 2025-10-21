folder_path="scripts_OV-COCO/Detic_SHiNe"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/Detic_CLIP+Name+SHiNe"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/Detic_LLM_inList+SHiNe"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done