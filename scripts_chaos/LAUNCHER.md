################################
# OV-COCO-8
################################

[//]: #(CoDet)
sh scripts_chaos/stage2/scripts_OV-COCO8/CoDet_coco8_R50.sh "1"

[//]: #(Detic)
sh scripts_chaos/stage2/scripts_OV-COCO8/Detic_coco8_R50_BoxSup.sh "1"
sh scripts_chaos/stage2/scripts_OV-COCO8/Detic_coco8_R50_Caption-Image.sh "1"

################################
# OV-COCO
################################

[//]: #(CoDet)
- [x] sh scripts_chaos/stage2/scripts_OV-COCO/CoDet/coco_rn50_codet.sh "4" 

[//]: #(Detic)
- [x] sh scripts_chaos/stage2/scripts_OV-COCO/Detic/coco_ovod_BoxSup_CLIP_R50_1x.sh "4"
- [x] [Best] sh scripts_chaos/stage2/scripts_OV-COCO/Detic/coco_ovod_Detic_CLIP_Caption-image_R50_1x.sh "4"
- [x] sh scripts_chaos/stage2/scripts_OV-COCO/Detic/coco_ovod_Detic_CLIP_caption_R50_1x.sh "4"
- [x] sh scripts_chaos/stage2/scripts_OV-COCO/Detic/coco_ovod_Detic_CLIP_image_R50_1x.sh "4"



################################
# OV-LVIS
################################

[//]: #(CoDet)
- [x] sh scripts_chaos/stage2/scripts_OV-LVIS/CoDet/lvis_rn50_codet.sh "1"
- [x] [Best]  sh scripts_chaos/stage2/scripts_OV-LVIS/CoDet/lvis_swinB_codet.sh "1"

[//]: #(Detic)
- [x] sh scripts_chaos/stage2/scripts_OV-LVIS/Detic/lvis_ovod_BoxSup_C2_Lbase_CLIP_R50_640_4x.sh "1"
- [x] sh scripts_chaos/stage2/scripts_OV-LVIS/Detic/lvis_ovod_Detic_C2_CCimg_R50_640_4x.sh "1"
- [x] [Best] sh scripts_chaos/stage2/scripts_OV-LVIS/Detic/lvis_ovod_Detic_C2_IN-L_R50_640_4x.sh "1"

- [x] sh scripts_chaos/stage2/scripts_OV-LVIS/Detic/lvis_ovod_BoxSup_C2_Lbase_CLIP_SwinB_4x.sh "1"
- [x] [Best] sh scripts_chaos/stage2/scripts_OV-LVIS/Detic/lvis_ovod_Detic_C2_IN-L_SwinB_4x.sh "1"