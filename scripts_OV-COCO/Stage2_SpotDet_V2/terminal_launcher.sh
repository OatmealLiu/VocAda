folder_path="scripts_OV-COCO/Stage1-b_V2_Eval-Proposal"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sh "$file"
    fi
done

folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/Detic_LLM_inList"
for file in "$folder_path"/coco*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/NoSyno_Detic_CLIP+Name"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/NoSyno_Detic_CLIP+Name+Caption"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/NoSyno_Detic_CLIP+Name+Caption+Image"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/NoSyno_Detic_CLIP+Name+Image"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/Detic_Tag_CloseSet"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/Detic_Tag_OpenSet"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done


folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/Detic_CLIP+Name+Descr"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done

folder_path="scripts_OV-COCO/Stage2_SpotDet_V2/Detic_LLM_inList+Descr"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done