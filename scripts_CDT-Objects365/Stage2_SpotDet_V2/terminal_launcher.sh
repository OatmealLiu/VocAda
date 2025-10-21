

folder_path="scripts_CDT-Objects365/Stage2_SpotDet_V2/Detic_CLIP+Name+Caption"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_CDT-Objects365/Stage2_SpotDet_V2/Detic_CLIP+Name+Caption+Image"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_CDT-Objects365/Stage2_SpotDet_V2/Detic_CLIP+Name+Image"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_CDT-Objects365/Stage2_SpotDet_V2/Detic_Tag_CloseSet"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_CDT-Objects365/Stage2_SpotDet_V2/Detic_Tag_OpenSet"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done

