#!/bin/bash

CFG_PATH="./configs_detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

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


4/30 02:22:02 d2.evaluation.coco_evaluation]: Saving results to ./output/Detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size/inference_objects365_v2_val_spotdet_gt/coco_instances_results.json
[04/30 02:22:45 d2.evaluation.coco_evaluation]: Evaluating predictions with unofficial COCO API...
Loading and preparing results...
DONE (t=48.60s)
creating index...
index created!
[04/30 02:23:40 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[04/30 02:33:44 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 604.52 seconds.
[04/30 02:34:02 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[04/30 02:35:09 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 67.40 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.503
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.377
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.611
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.706
[04/30 02:35:09 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 35.177 | 50.299 | 37.677 | 17.348 | 35.776 | 49.003 |
[04/30 02:35:14 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 16.824 | Sneakers                        | 38.638 | Chair                 | 40.911 |
| Other Shoes         | 10.330 | Hat                             | 50.453 | Car                   | 22.972 |
| Lamp                | 27.549 | Glasses                         | 38.142 | Bottle                | 34.250 |
| Desk                | 27.537 | Cup                             | 46.742 | Street Lights         | 10.330 |
| Cabinet/shelf       | 17.910 | Handbag/Satchel                 | 22.512 | Bracelet              | 28.037 |
| Plate               | 60.082 | Picture/Frame                   | 25.324 | Helmet                | 44.510 |
| Book                | 16.844 | Gloves                          | 40.829 | Storage box           | 24.112 |
| Boat                | 27.731 | Leather Shoes                   | 20.180 | Flower                | 15.119 |
| Bench               | 25.019 | Potted Plant                    | 0.778  | Bowl/Basin            | 52.327 |
| Flag                | 48.901 | Pillow                          | 55.942 | Boots                 | 42.924 |
| Vase                | 36.942 | Microphone                      | 17.977 | Necklace              | 36.002 |
| Ring                | 18.451 | SUV                             | 21.625 | Wine Glass            | 64.747 |
| Belt                | 37.610 | Monitor/TV                      | 63.353 | Backpack              | 39.743 |
| Umbrella            | 32.323 | Traffic Light                   | 36.422 | Speaker               | 45.060 |
| Watch               | 49.790 | Tie                             | 36.120 | Trash bin Can         | 54.653 |
| Slippers            | 28.964 | Bicycle                         | 47.315 | Stool                 | 46.209 |
| Barrel/bucket       | 40.088 | Van                             | 22.938 | Couch                 | 55.697 |
| Sandals             | 33.238 | Basket                          | 44.048 | Drum                  | 31.881 |
| Pen/Pencil          | 28.365 | Bus                             | 44.121 | Wild Bird             | 31.420 |
| High Heels          | 24.886 | Motorcycle                      | 34.798 | Guitar                | 50.096 |
| Carpet              | 39.513 | Cell Phone                      | 48.769 | Bread                 | 25.952 |
| Camera              | 37.006 | Canned                          | 33.361 | Truck                 | 23.411 |
| Traffic cone        | 46.346 | Cymbal                          | 36.071 | Lifesaver             | 8.811  |
| Towel               | 58.028 | Stuffed Toy                     | 43.529 | Candle                | 36.121 |
| Sailboat            | 20.466 | Laptop                          | 73.591 | Awning                | 34.428 |
| Bed                 | 52.601 | Faucet                          | 37.476 | Tent                  | 17.337 |
| Horse               | 52.601 | Mirror                          | 40.458 | Power outlet          | 40.578 |
| Sink                | 36.901 | Apple                           | 36.618 | Air Conditioner       | 30.444 |
| Knife               | 48.879 | Hockey Stick                    | 45.097 | Paddle                | 25.667 |
| Pickup Truck        | 43.939 | Fork                            | 58.076 | Traffic Sign          | 12.955 |
| Ballon              | 59.161 | Tripod                          | 9.234  | Dog                   | 62.530 |
| Spoon               | 47.689 | Clock                           | 63.991 | Pot                   | 38.385 |
| Cow                 | 42.937 | Cake                            | 24.662 | Dining Table          | 57.972 |
| Sheep               | 46.836 | Hanger                          | 3.986  | Blackboard/Whiteboard | 34.764 |
| Napkin              | 44.203 | Other Fish                      | 40.252 | Orange/Tangerine      | 29.685 |
| Toiletry            | 32.498 | Keyboard                        | 61.421 | Tomato                | 57.498 |
| Lantern             | 39.835 | Machinery Vehicle               | 17.737 | Fan                   | 43.342 |
| Green Vegetables    | 0.906  | Banana                          | 10.256 | Baseball Glove        | 51.438 |
| Airplane            | 61.420 | Mouse                           | 56.342 | Train                 | 41.098 |
| Pumpkin             | 62.582 | Soccer                          | 33.896 | Skiboard              | 10.680 |
| Luggage             | 38.995 | Nightstand                      | 38.564 | Teapot                | 39.607 |
| Telephone           | 32.395 | Trolley                         | 29.693 | Head Phone            | 38.670 |
| Sports Car          | 64.360 | Stop Sign                       | 46.558 | Dessert               | 26.269 |
| Scooter             | 35.785 | Stroller                        | 44.601 | Crane                 | 8.549  |
| Remote              | 51.881 | Refrigerator                    | 73.757 | Oven                  | 31.163 |
| Lemon               | 47.186 | Duck                            | 54.247 | Baseball Bat          | 55.854 |
| Surveillance Camera | 12.270 | Cat                             | 68.137 | Jug                   | 31.844 |
| Broccoli            | 49.804 | Piano                           | 27.462 | Pizza                 | 62.411 |
| Elephant            | 71.597 | Skateboard                      | 34.176 | Surfboard             | 51.915 |
| Gun                 | 23.775 | Skating and Skiing shoes        | 28.245 | Gas stove             | 20.809 |
| Donut               | 59.572 | Bow Tie                         | 29.018 | Carrot                | 45.923 |
| Toilet              | 76.347 | Kite                            | 51.068 | Strawberry            | 46.158 |
| Other Balls         | 52.263 | Shovel                          | 18.616 | Pepper                | 36.827 |
| Computer Box        | 11.484 | Toilet Paper                    | 54.171 | Cleaning Products     | 31.458 |
| Chopsticks          | 41.307 | Microwave                       | 72.920 | Pigeon                | 56.122 |
| Baseball            | 45.287 | Cutting/chopping Board          | 53.460 | Coffee Table          | 53.773 |
| Side Table          | 25.280 | Scissors                        | 44.962 | Marker                | 22.410 |
| Pie                 | 28.773 | Ladder                          | 42.624 | Snowboard             | 53.587 |
| Cookies             | 29.961 | Radiator                        | 52.287 | Fire Hydrant          | 54.103 |
| Basketball          | 42.916 | Zebra                           | 67.290 | Grape                 | 1.522  |
| Giraffe             | 69.341 | Potato                          | 36.319 | Sausage               | 34.839 |
| Tricycle            | 20.252 | Violin                          | 18.377 | Egg                   | 69.518 |
| Fire Extinguisher   | 49.199 | Candy                           | 5.763  | Fire Truck            | 53.146 |
| Billards            | 31.395 | Converter                       | 4.274  | Bathtub               | 59.780 |
| Wheelchair          | 54.139 | Golf Club                       | 35.615 | Briefcase             | 43.216 |
| Cucumber            | 42.746 | Cigar/Cigarette                 | 18.736 | Paint Brush           | 15.621 |
| Pear                | 18.775 | Heavy Truck                     | 33.631 | Hamburger             | 37.498 |
| Extractor           | 14.688 | Extension Cord                  | 5.714  | Tong                  | 12.690 |
| Tennis Racket       | 67.879 | Folder                          | 15.126 | American Football     | 27.368 |
| earphone            | 5.853  | Mask                            | 26.757 | Kettle                | 53.596 |
| Tennis              | 27.599 | Ship                            | 44.849 | Swing                 | 0.851  |
| Coffee Machine      | 56.897 | Slide                           | 33.077 | Carriage              | 8.669  |
| Onion               | 31.893 | Green beans                     | 15.079 | Projector             | 36.178 |
| Frisbee             | 67.044 | Washing Machine/Drying Machine  | 33.770 | Chicken               | 55.915 |
| Printer             | 57.812 | Watermelon                      | 43.060 | Saxophone             | 38.228 |
| Tissue              | 3.321  | Toothbrush                      | 39.763 | Ice cream             | 14.874 |
| Hot air balloon     | 70.920 | Cello                           | 28.055 | French Fries          | 0.390  |
| Scale               | 14.617 | Trophy                          | 32.479 | Cabbage               | 16.745 |
| Hot dog             | 16.576 | Blender                         | 56.329 | Peach                 | 35.902 |
| Rice                | 7.155  | Wallet/Purse                    | 43.073 | Volleyball            | 51.080 |
| Deer                | 56.809 | Goose                           | 50.911 | Tape                  | 29.799 |
| Tablet              | 27.089 | Cosmetics                       | 29.400 | Trumpet               | 22.892 |
| Pineapple           | 30.784 | Golf Ball                       | 45.344 | Ambulance             | 72.896 |
| Parking meter       | 36.774 | Mango                           | 4.089  | Key                   | 17.116 |
| Hurdle              | 1.708  | Fishing Rod                     | 27.863 | Medal                 | 10.875 |
| Flute               | 32.471 | Brush                           | 20.761 | Penguin               | 58.663 |
| Megaphone           | 21.262 | Corn                            | 39.100 | Lettuce               | 15.724 |
| Garlic              | 31.599 | Swan                            | 57.540 | Helicopter            | 48.600 |
| Green Onion         | 15.856 | Sandwich                        | 47.642 | Nuts                  | 15.723 |
| Speed Limit Sign    | 24.615 | Induction Cooker                | 20.767 | Broom                 | 24.432 |
| Trombone            | 14.295 | Plum                            | 9.643  | Rickshaw              | 25.171 |
| Goldfish            | 39.278 | Kiwi fruit                      | 34.283 | Router/modem          | 21.235 |
| Poker Card          | 41.973 | Toaster                         | 60.726 | Shrimp                | 41.764 |
| Sushi               | 49.891 | Cheese                          | 38.633 | Notepaper             | 12.574 |
| Cherry              | 25.304 | Pliers                          | 24.728 | CD                    | 22.319 |
| Pasta               | 3.098  | Hammer                          | 16.484 | Cue                   | 9.148  |
| Avocado             | 39.053 | Hami melon                      | 8.758  | Flask                 | 34.479 |
| Mushroom            | 42.427 | Screwdriver                     | 21.166 | Soap                  | 29.401 |
| Recorder            | 18.532 | Bear                            | 63.362 | Eggplant              | 31.786 |
| Board Eraser        | 13.323 | Coconut                         | 40.283 | Tape Measure/ Ruler   | 19.430 |
| Pig                 | 56.214 | Showerhead                      | 26.401 | Globe                 | 46.729 |
| Chips               | 1.174  | Steak                           | 51.945 | Crosswalk Sign        | 14.463 |
| Stapler             | 35.093 | Camel                           | 55.820 | Formula 1             | 44.153 |
| Pomegranate         | 13.786 | Dishwasher                      | 71.395 | Crab                  | 15.625 |
| Hoverboard          | 15.741 | Meatball                        | 56.982 | Rice Cooker           | 48.882 |
| Tuba                | 26.342 | Calculator                      | 54.650 | Papaya                | 21.940 |
| Antelope            | 48.521 | Parrot                          | 46.852 | Seal                  | 50.140 |
| Butterfly           | 52.095 | Dumbbell                        | 5.711  | Donkey                | 58.956 |
| Lion                | 33.118 | Urinal                          | 68.982 | Dolphin               | 46.469 |
| Electric Drill      | 27.107 | Hair Dryer                      | 21.949 | Egg tart              | 33.663 |
| Jellyfish           | 35.789 | Treadmill                       | 25.571 | Lighter               | 26.642 |
| Grapefruit          | 5.930  | Game board                      | 46.935 | Mop                   | 14.333 |
| Radish              | 2.397  | Baozi                           | 50.035 | Target                | 12.385 |
| French              | 5.018  | Spring Rolls                    | 58.763 | Monkey                | 49.959 |
| Rabbit              | 42.686 | Pencil Case                     | 24.070 | Yak                   | 70.425 |
| Red Cabbage         | 24.648 | Binoculars                      | 23.353 | Asparagus             | 33.192 |
| Barbell             | 3.497  | Scallop                         | 38.471 | Noddles               | 6.655  |
| Comb                | 43.871 | Dumpling                        | 34.371 | Oyster                | 60.038 |
| Table Tennis paddle | 21.210 | Cosmetics Brush/Eyeliner Pencil | 33.668 | Chainsaw              | 11.510 |
| Eraser              | 16.401 | Lobster                         | 14.802 | Durian                | 10.637 |
| Okra                | 11.041 | Lipstick                        | 32.377 | Cosmetics Mirror      | 8.720  |
| Curling             | 67.609 | Table Tennis                    | 2.840  |                       |        |
[04/30 02:36:17 detectron2]: Evaluation results for objects365_v2_val_spotdet_gt in csv format:
[04/30 02:36:17 d2.evaluation.testing]: copypaste: Task: bbox
[04/30 02:36:17 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/30 02:36:17 d2.evaluation.testing]: copypaste: 35.1765,50.2993,37.6770,17.3480,35.7762,49.0032