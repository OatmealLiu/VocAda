#!/bin/bash

CFG_PATH="./configs_detic/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

CLASSIFIER_NAME="o365_clip_a+cnamefix.npy"

CUDA_VISIBLE_DEVICES=1,2,3  python train_net_detic.py \
        --num-gpus 3 \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('objects365_v2_val_spotdet_gt',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(365,)" \
        MODEL.MASK_ON False


[04/30 02:22:46 d2.evaluation.coco_evaluation]: Preparing results for COCO format ...
[04/30 02:22:48 d2.evaluation.coco_evaluation]: Saving results to ./output/Detic/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size/inference_objects365_v2_val_spotdet_gt/coco_instances_results.json
[04/30 02:23:28 d2.evaluation.coco_evaluation]: Evaluating predictions with unofficial COCO API...
Loading and preparing results...
DONE (t=37.66s)
creating index...
index created!
[04/30 02:24:11 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[04/30 02:34:22 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 610.69 seconds.
[04/30 02:34:38 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[04/30 02:35:41 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 62.38 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.493
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.174
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.354
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.236
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.580
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.604
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.698
[04/30 02:35:41 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 34.520 | 49.348 | 37.004 | 17.357 | 35.446 | 48.867 |
[04/30 02:35:46 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 16.561 | Sneakers                        | 37.444 | Chair                 | 42.610 |
| Other Shoes         | 11.479 | Hat                             | 51.058 | Car                   | 19.730 |
| Lamp                | 27.140 | Glasses                         | 34.135 | Bottle                | 34.412 |
| Desk                | 26.911 | Cup                             | 46.551 | Street Lights         | 9.343  |
| Cabinet/shelf       | 16.847 | Handbag/Satchel                 | 22.265 | Bracelet              | 28.011 |
| Plate               | 61.175 | Picture/Frame                   | 16.440 | Helmet                | 44.167 |
| Book                | 16.893 | Gloves                          | 41.710 | Storage box           | 25.628 |
| Boat                | 26.927 | Leather Shoes                   | 20.956 | Flower                | 10.351 |
| Bench               | 26.782 | Potted Plant                    | 0.648  | Bowl/Basin            | 53.139 |
| Flag                | 49.340 | Pillow                          | 57.077 | Boots                 | 44.071 |
| Vase                | 36.316 | Microphone                      | 19.684 | Necklace              | 35.633 |
| Ring                | 18.729 | SUV                             | 24.984 | Wine Glass            | 65.477 |
| Belt                | 38.253 | Monitor/TV                      | 55.966 | Backpack              | 40.284 |
| Umbrella            | 33.804 | Traffic Light                   | 36.200 | Speaker               | 43.163 |
| Watch               | 50.131 | Tie                             | 29.780 | Trash bin Can         | 55.309 |
| Slippers            | 31.005 | Bicycle                         | 48.686 | Stool                 | 46.624 |
| Barrel/bucket       | 40.361 | Van                             | 22.075 | Couch                 | 54.119 |
| Sandals             | 32.260 | Basket                          | 44.221 | Drum                  | 33.052 |
| Pen/Pencil          | 29.725 | Bus                             | 41.896 | Wild Bird             | 31.489 |
| High Heels          | 22.913 | Motorcycle                      | 36.848 | Guitar                | 51.982 |
| Carpet              | 42.029 | Cell Phone                      | 47.926 | Bread                 | 25.511 |
| Camera              | 37.537 | Canned                          | 32.994 | Truck                 | 23.832 |
| Traffic cone        | 46.388 | Cymbal                          | 38.826 | Lifesaver             | 7.184  |
| Towel               | 58.899 | Stuffed Toy                     | 39.611 | Candle                | 36.581 |
| Sailboat            | 15.941 | Laptop                          | 74.492 | Awning                | 34.425 |
| Bed                 | 54.972 | Faucet                          | 38.273 | Tent                  | 15.582 |
| Horse               | 52.976 | Mirror                          | 40.118 | Power outlet          | 31.526 |
| Sink                | 37.061 | Apple                           | 37.143 | Air Conditioner       | 31.220 |
| Knife               | 50.617 | Hockey Stick                    | 43.151 | Paddle                | 28.555 |
| Pickup Truck        | 45.871 | Fork                            | 59.413 | Traffic Sign          | 10.595 |
| Ballon              | 60.538 | Tripod                          | 8.684  | Dog                   | 63.256 |
| Spoon               | 49.461 | Clock                           | 63.329 | Pot                   | 40.197 |
| Cow                 | 43.054 | Cake                            | 25.578 | Dining Table          | 59.024 |
| Sheep               | 46.076 | Hanger                          | 3.163  | Blackboard/Whiteboard | 33.484 |
| Napkin              | 45.653 | Other Fish                      | 40.426 | Orange/Tangerine      | 23.879 |
| Toiletry            | 28.862 | Keyboard                        | 60.264 | Tomato                | 58.598 |
| Lantern             | 39.937 | Machinery Vehicle               | 13.021 | Fan                   | 44.321 |
| Green Vegetables    | 0.628  | Banana                          | 9.903  | Baseball Glove        | 52.343 |
| Airplane            | 61.706 | Mouse                           | 54.762 | Train                 | 28.357 |
| Pumpkin             | 61.823 | Soccer                          | 34.165 | Skiboard              | 7.749  |
| Luggage             | 35.807 | Nightstand                      | 37.233 | Teapot                | 39.112 |
| Telephone           | 34.377 | Trolley                         | 26.876 | Head Phone            | 38.330 |
| Sports Car          | 61.332 | Stop Sign                       | 44.357 | Dessert               | 26.539 |
| Scooter             | 36.798 | Stroller                        | 48.039 | Crane                 | 2.475  |
| Remote              | 50.138 | Refrigerator                    | 74.220 | Oven                  | 30.728 |
| Lemon               | 45.659 | Duck                            | 53.505 | Baseball Bat          | 54.388 |
| Surveillance Camera | 12.509 | Cat                             | 68.784 | Jug                   | 31.895 |
| Broccoli            | 49.326 | Piano                           | 28.534 | Pizza                 | 63.369 |
| Elephant            | 72.379 | Skateboard                      | 37.298 | Surfboard             | 52.346 |
| Gun                 | 25.723 | Skating and Skiing shoes        | 25.729 | Gas stove             | 22.915 |
| Donut               | 58.167 | Bow Tie                         | 30.353 | Carrot                | 46.857 |
| Toilet              | 76.522 | Kite                            | 51.517 | Strawberry            | 48.345 |
| Other Balls         | 49.133 | Shovel                          | 18.813 | Pepper                | 24.207 |
| Computer Box        | 7.377  | Toilet Paper                    | 50.859 | Cleaning Products     | 31.254 |
| Chopsticks          | 41.451 | Microwave                       | 70.864 | Pigeon                | 56.501 |
| Baseball            | 46.131 | Cutting/chopping Board          | 54.202 | Coffee Table          | 56.516 |
| Side Table          | 26.985 | Scissors                        | 46.293 | Marker                | 23.883 |
| Pie                 | 24.459 | Ladder                          | 41.845 | Snowboard             | 55.215 |
| Cookies             | 28.310 | Radiator                        | 52.952 | Fire Hydrant          | 55.395 |
| Basketball          | 44.797 | Zebra                           | 67.691 | Grape                 | 2.545  |
| Giraffe             | 69.256 | Potato                          | 39.854 | Sausage               | 33.918 |
| Tricycle            | 23.157 | Violin                          | 23.386 | Egg                   | 70.154 |
| Fire Extinguisher   | 51.742 | Candy                           | 4.600  | Fire Truck            | 55.323 |
| Billards            | 36.330 | Converter                       | 1.956  | Bathtub               | 60.268 |
| Wheelchair          | 56.299 | Golf Club                       | 36.879 | Briefcase             | 43.585 |
| Cucumber            | 45.784 | Cigar/Cigarette                 | 22.033 | Paint Brush           | 13.847 |
| Pear                | 22.595 | Heavy Truck                     | 33.606 | Hamburger             | 38.524 |
| Extractor           | 7.397  | Extension Cord                  | 3.651  | Tong                  | 15.504 |
| Tennis Racket       | 66.783 | Folder                          | 18.034 | American Football     | 32.577 |
| earphone            | 6.337  | Mask                            | 27.608 | Kettle                | 56.019 |
| Tennis              | 30.722 | Ship                            | 35.578 | Swing                 | 0.297  |
| Coffee Machine      | 59.180 | Slide                           | 34.116 | Carriage              | 8.298  |
| Onion               | 39.262 | Green beans                     | 14.171 | Projector             | 39.262 |
| Frisbee             | 69.949 | Washing Machine/Drying Machine  | 37.949 | Chicken               | 55.481 |
| Printer             | 59.662 | Watermelon                      | 48.366 | Saxophone             | 40.500 |
| Tissue              | 2.933  | Toothbrush                      | 40.643 | Ice cream             | 16.263 |
| Hot air balloon     | 74.142 | Cello                           | 15.148 | French Fries          | 0.272  |
| Scale               | 17.721 | Trophy                          | 32.250 | Cabbage               | 10.485 |
| Hot dog             | 13.824 | Blender                         | 54.146 | Peach                 | 42.703 |
| Rice                | 0.930  | Wallet/Purse                    | 47.347 | Volleyball            | 58.882 |
| Deer                | 57.178 | Goose                           | 51.825 | Tape                  | 31.395 |
| Tablet              | 22.591 | Cosmetics                       | 26.638 | Trumpet               | 17.366 |
| Pineapple           | 31.171 | Golf Ball                       | 40.428 | Ambulance             | 75.052 |
| Parking meter       | 36.250 | Mango                           | 3.344  | Key                   | 19.486 |
| Hurdle              | 1.431  | Fishing Rod                     | 30.242 | Medal                 | 9.798  |
| Flute               | 30.263 | Brush                           | 21.302 | Penguin               | 60.571 |
| Megaphone           | 22.099 | Corn                            | 35.517 | Lettuce               | 13.782 |
| Garlic              | 41.542 | Swan                            | 51.782 | Helicopter            | 47.144 |
| Green Onion         | 21.078 | Sandwich                        | 47.961 | Nuts                  | 7.602  |
| Speed Limit Sign    | 30.217 | Induction Cooker                | 18.569 | Broom                 | 24.523 |
| Trombone            | 17.970 | Plum                            | 6.823  | Rickshaw              | 25.248 |
| Goldfish            | 38.135 | Kiwi fruit                      | 35.853 | Router/modem          | 25.560 |
| Poker Card          | 24.217 | Toaster                         | 63.323 | Shrimp                | 39.802 |
| Sushi               | 54.044 | Cheese                          | 27.177 | Notepaper             | 14.214 |
| Cherry              | 24.104 | Pliers                          | 28.720 | CD                    | 12.508 |
| Pasta               | 0.992  | Hammer                          | 21.186 | Cue                   | 2.731  |
| Avocado             | 44.077 | Hami melon                      | 7.636  | Flask                 | 34.664 |
| Mushroom            | 42.555 | Screwdriver                     | 24.318 | Soap                  | 30.258 |
| Recorder            | 6.311  | Bear                            | 62.870 | Eggplant              | 34.506 |
| Board Eraser        | 18.436 | Coconut                         | 40.032 | Tape Measure/ Ruler   | 19.095 |
| Pig                 | 57.163 | Showerhead                      | 26.226 | Globe                 | 46.158 |
| Chips               | 0.781  | Steak                           | 49.480 | Crosswalk Sign        | 15.653 |
| Stapler             | 40.352 | Camel                           | 58.708 | Formula 1             | 35.927 |
| Pomegranate         | 5.995  | Dishwasher                      | 71.487 | Crab                  | 13.276 |
| Hoverboard          | 11.459 | Meatball                        | 53.873 | Rice Cooker           | 49.859 |
| Tuba                | 24.751 | Calculator                      | 57.658 | Papaya                | 28.860 |
| Antelope            | 42.116 | Parrot                          | 46.000 | Seal                  | 30.195 |
| Butterfly           | 53.963 | Dumbbell                        | 6.477  | Donkey                | 43.492 |
| Lion                | 36.001 | Urinal                          | 67.553 | Dolphin               | 47.576 |
| Electric Drill      | 28.615 | Hair Dryer                      | 22.787 | Egg tart              | 30.868 |
| Jellyfish           | 25.707 | Treadmill                       | 21.170 | Lighter               | 26.273 |
| Grapefruit          | 3.870  | Game board                      | 45.866 | Mop                   | 17.095 |
| Radish              | 1.745  | Baozi                           | 22.295 | Target                | 13.009 |
| French              | 1.058  | Spring Rolls                    | 60.360 | Monkey                | 46.457 |
| Rabbit              | 42.273 | Pencil Case                     | 34.724 | Yak                   | 68.258 |
| Red Cabbage         | 9.348  | Binoculars                      | 24.623 | Asparagus             | 38.847 |
| Barbell             | 5.129  | Scallop                         | 27.791 | Noddles               | 5.600  |
| Comb                | 35.656 | Dumpling                        | 12.330 | Oyster                | 50.233 |
| Table Tennis paddle | 17.081 | Cosmetics Brush/Eyeliner Pencil | 33.474 | Chainsaw              | 7.999  |
| Eraser              | 22.365 | Lobster                         | 9.401  | Durian                | 10.563 |
| Okra                | 1.802  | Lipstick                        | 24.096 | Cosmetics Mirror      | 11.580 |
| Curling             | 48.755 | Table Tennis                    | 4.121  |                       |        |
[04/30 02:37:00 detectron2]: Evaluation results for objects365_v2_val_spotdet_gt in csv format:
[04/30 02:37:00 d2.evaluation.testing]: copypaste: Task: bbox
[04/30 02:37:00 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/30 02:37:00 d2.evaluation.testing]: copypaste: 34.5197,49.3476,37.0040,17.3571,35.4455,48.8667