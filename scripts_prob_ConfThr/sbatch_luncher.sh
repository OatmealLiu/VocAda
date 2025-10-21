folder_path="scripts_prob_ConfThr/Detic_COCO"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_prob_ConfThr/Detic_LVIS_R50"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_prob_ConfThr/Detic_LVIS_SwinB"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_prob_ConfThr/Detic_Objects365"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done

