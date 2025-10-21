#!/bin/bash

CFG_PATH="./configs_detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml"
MODEL_PATH="./models/Detic_CDT/BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth"

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



[1, 3, 133, 8, 10, 13, 15, 19, 160, 34, 38, 171, 300, 45, 55, 317, 206, 208, 81, 223, 107, 124]
[70, 13, 77, 302, 80, 82, 24, 61, 31]
[103, 10, 11, 108, 238, 16, 85, 183, 217]
[242]
[208, 1, 42, 60]
[1, 237, 46, 14, 53, 21, 182, 217]
[1, 33, 354, 7, 8, 104, 14, 111, 16, 48, 52, 21, 53]
[65, 10, 236, 81, 21]
[1, 66, 4, 199, 136, 105, 43, 75, 14, 46, 83, 180, 21, 53, 28, 30]
[167, 11, 123, 16, 280, 27, 94]
[1, 2, 5, 104, 232, 234, 43, 108, 112, 49, 113, 83, 21, 182, 23, 346, 159]
[96, 164, 134, 140, 45, 13, 21, 214, 311, 27, 61, 158]
[1, 259, 6, 173, 15, 49]
[129, 196, 260, 108, 172, 16, 112, 83, 180]
[216, 1, 18, 79]
[16, 129, 188]
[27, 99, 11, 261]
[1, 2, 131, 5, 8, 14, 142, 21, 30, 159, 289, 290, 180, 182, 310, 56, 83, 84, 217, 218, 348, 105, 237, 112]
[192, 161, 70, 77, 302, 80, 82, 21, 154]
[118, 35, 5, 6]
[1, 109, 5, 14]
{}
{}
[04/30 02:21:42 d2.evaluation.coco_evaluation]: Preparing results for COCO format ...
[04/30 02:21:45 d2.evaluation.coco_evaluation]: Saving results to ./output/Detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x/inference_objects365_v2_val_spotdet_gt/coco_instances_results.json
[04/30 02:22:23 d2.evaluation.coco_evaluation]: Evaluating predictions with unofficial COCO API...
Loading and preparing results...
DONE (t=44.42s)
creating index...
index created!
[04/30 02:23:12 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[04/30 02:32:38 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 565.80 seconds.
[04/30 02:32:49 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[04/30 02:33:39 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 50.42 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.340
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.486
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.235
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.605
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697
[04/30 02:33:39 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 33.984 | 48.644 | 36.450 | 17.335 | 34.916 | 47.670 |
[04/30 02:33:42 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 17.609 | Sneakers                        | 35.957 | Chair                 | 42.690 |
| Other Shoes         | 10.531 | Hat                             | 51.049 | Car                   | 20.432 |
| Lamp                | 27.648 | Glasses                         | 35.448 | Bottle                | 34.434 |
| Desk                | 27.149 | Cup                             | 45.912 | Street Lights         | 9.488  |
| Cabinet/shelf       | 18.861 | Handbag/Satchel                 | 21.413 | Bracelet              | 27.829 |
| Plate               | 61.679 | Picture/Frame                   | 18.623 | Helmet                | 44.220 |
| Book                | 17.191 | Gloves                          | 42.486 | Storage box           | 25.757 |
| Boat                | 27.555 | Leather Shoes                   | 19.985 | Flower                | 9.988  |
| Bench               | 25.727 | Potted Plant                    | 1.229  | Bowl/Basin            | 52.383 |
| Flag                | 49.501 | Pillow                          | 57.156 | Boots                 | 44.241 |
| Vase                | 36.419 | Microphone                      | 19.085 | Necklace              | 35.826 |
| Ring                | 17.988 | SUV                             | 20.566 | Wine Glass            | 65.087 |
| Belt                | 38.373 | Monitor/TV                      | 56.515 | Backpack              | 40.159 |
| Umbrella            | 33.817 | Traffic Light                   | 35.534 | Speaker               | 44.491 |
| Watch               | 50.145 | Tie                             | 29.749 | Trash bin Can         | 55.422 |
| Slippers            | 30.191 | Bicycle                         | 49.263 | Stool                 | 46.397 |
| Barrel/bucket       | 40.263 | Van                             | 18.734 | Couch                 | 56.395 |
| Sandals             | 31.741 | Basket                          | 44.396 | Drum                  | 31.873 |
| Pen/Pencil          | 28.728 | Bus                             | 42.225 | Wild Bird             | 31.387 |
| High Heels          | 24.298 | Motorcycle                      | 36.687 | Guitar                | 49.792 |
| Carpet              | 41.367 | Cell Phone                      | 48.676 | Bread                 | 25.480 |
| Camera              | 38.146 | Canned                          | 34.903 | Truck                 | 23.766 |
| Traffic cone        | 46.217 | Cymbal                          | 39.119 | Lifesaver             | 8.127  |
| Towel               | 58.965 | Stuffed Toy                     | 39.583 | Candle                | 36.055 |
| Sailboat            | 15.504 | Laptop                          | 74.469 | Awning                | 34.805 |
| Bed                 | 54.212 | Faucet                          | 38.487 | Tent                  | 15.390 |
| Horse               | 53.346 | Mirror                          | 40.871 | Power outlet          | 28.254 |
| Sink                | 36.801 | Apple                           | 36.708 | Air Conditioner       | 30.272 |
| Knife               | 51.058 | Hockey Stick                    | 39.562 | Paddle                | 26.876 |
| Pickup Truck        | 45.540 | Fork                            | 59.366 | Traffic Sign          | 11.755 |
| Ballon              | 60.750 | Tripod                          | 10.170 | Dog                   | 63.465 |
| Spoon               | 49.641 | Clock                           | 64.236 | Pot                   | 39.866 |
| Cow                 | 43.154 | Cake                            | 25.015 | Dining Table          | 60.166 |
| Sheep               | 45.635 | Hanger                          | 3.108  | Blackboard/Whiteboard | 36.741 |
| Napkin              | 46.694 | Other Fish                      | 37.916 | Orange/Tangerine      | 20.792 |
| Toiletry            | 29.695 | Keyboard                        | 61.852 | Tomato                | 58.164 |
| Lantern             | 40.052 | Machinery Vehicle               | 16.587 | Fan                   | 44.600 |
| Green Vegetables    | 1.010  | Banana                          | 9.936  | Baseball Glove        | 49.957 |
| Airplane            | 62.667 | Mouse                           | 53.176 | Train                 | 28.604 |
| Pumpkin             | 62.112 | Soccer                          | 33.039 | Skiboard              | 8.668  |
| Luggage             | 37.472 | Nightstand                      | 37.465 | Teapot                | 40.588 |
| Telephone           | 33.096 | Trolley                         | 31.310 | Head Phone            | 38.501 |
| Sports Car          | 59.684 | Stop Sign                       | 45.514 | Dessert               | 24.735 |
| Scooter             | 36.676 | Stroller                        | 47.325 | Crane                 | 2.535  |
| Remote              | 49.677 | Refrigerator                    | 74.775 | Oven                  | 32.129 |
| Lemon               | 45.127 | Duck                            | 54.008 | Baseball Bat          | 53.766 |
| Surveillance Camera | 13.493 | Cat                             | 69.122 | Jug                   | 30.183 |
| Broccoli            | 50.095 | Piano                           | 30.190 | Pizza                 | 63.894 |
| Elephant            | 72.339 | Skateboard                      | 36.323 | Surfboard             | 52.576 |
| Gun                 | 21.812 | Skating and Skiing shoes        | 32.025 | Gas stove             | 22.971 |
| Donut               | 59.898 | Bow Tie                         | 29.252 | Carrot                | 45.353 |
| Toilet              | 76.441 | Kite                            | 51.961 | Strawberry            | 47.869 |
| Other Balls         | 49.850 | Shovel                          | 19.039 | Pepper                | 33.488 |
| Computer Box        | 6.311  | Toilet Paper                    | 52.323 | Cleaning Products     | 32.868 |
| Chopsticks          | 41.089 | Microwave                       | 71.252 | Pigeon                | 56.052 |
| Baseball            | 45.876 | Cutting/chopping Board          | 55.232 | Coffee Table          | 56.763 |
| Side Table          | 27.847 | Scissors                        | 46.782 | Marker                | 22.978 |
| Pie                 | 25.155 | Ladder                          | 42.182 | Snowboard             | 55.587 |
| Cookies             | 28.108 | Radiator                        | 53.597 | Fire Hydrant          | 54.594 |
| Basketball          | 43.726 | Zebra                           | 67.928 | Grape                 | 1.690  |
| Giraffe             | 69.363 | Potato                          | 37.201 | Sausage               | 36.942 |
| Tricycle            | 21.867 | Violin                          | 18.567 | Egg                   | 70.399 |
| Fire Extinguisher   | 50.500 | Candy                           | 5.343  | Fire Truck            | 56.046 |
| Billards            | 32.756 | Converter                       | 1.664  | Bathtub               | 59.901 |
| Wheelchair          | 56.935 | Golf Club                       | 37.924 | Briefcase             | 42.614 |
| Cucumber            | 45.132 | Cigar/Cigarette                 | 19.582 | Paint Brush           | 12.971 |
| Pear                | 19.627 | Heavy Truck                     | 36.126 | Hamburger             | 40.484 |
| Extractor           | 4.924  | Extension Cord                  | 3.546  | Tong                  | 11.516 |
| Tennis Racket       | 67.502 | Folder                          | 16.733 | American Football     | 28.661 |
| earphone            | 5.911  | Mask                            | 28.932 | Kettle                | 54.223 |
| Tennis              | 29.560 | Ship                            | 35.351 | Swing                 | 0.572  |
| Coffee Machine      | 59.370 | Slide                           | 33.129 | Carriage              | 8.512  |
| Onion               | 36.571 | Green beans                     | 13.116 | Projector             | 39.632 |
| Frisbee             | 69.641 | Washing Machine/Drying Machine  | 36.243 | Chicken               | 54.484 |
| Printer             | 59.872 | Watermelon                      | 45.176 | Saxophone             | 34.716 |
| Tissue              | 2.967  | Toothbrush                      | 40.423 | Ice cream             | 11.918 |
| Hot air balloon     | 73.087 | Cello                           | 16.399 | French Fries          | 0.341  |
| Scale               | 13.620 | Trophy                          | 33.736 | Cabbage               | 13.571 |
| Hot dog             | 12.686 | Blender                         | 54.009 | Peach                 | 36.945 |
| Rice                | 0.631  | Wallet/Purse                    | 45.233 | Volleyball            | 45.722 |
| Deer                | 56.079 | Goose                           | 49.971 | Tape                  | 30.442 |
| Tablet              | 19.230 | Cosmetics                       | 23.340 | Trumpet               | 21.517 |
| Pineapple           | 31.560 | Golf Ball                       | 41.024 | Ambulance             | 70.809 |
| Parking meter       | 36.016 | Mango                           | 3.244  | Key                   | 16.789 |
| Hurdle              | 1.658  | Fishing Rod                     | 28.086 | Medal                 | 10.533 |
| Flute               | 24.086 | Brush                           | 19.684 | Penguin               | 57.327 |
| Megaphone           | 18.672 | Corn                            | 36.774 | Lettuce               | 15.773 |
| Garlic              | 34.644 | Swan                            | 52.235 | Helicopter            | 48.770 |
| Green Onion         | 20.084 | Sandwich                        | 47.330 | Nuts                  | 7.465  |
| Speed Limit Sign    | 30.020 | Induction Cooker                | 21.632 | Broom                 | 24.218 |
| Trombone            | 14.601 | Plum                            | 3.863  | Rickshaw              | 24.303 |
| Goldfish            | 34.286 | Kiwi fruit                      | 34.056 | Router/modem          | 21.002 |
| Poker Card          | 38.768 | Toaster                         | 62.276 | Shrimp                | 40.798 |
| Sushi               | 51.125 | Cheese                          | 24.860 | Notepaper             | 13.139 |
| Cherry              | 22.551 | Pliers                          | 24.623 | CD                    | 9.761  |
| Pasta               | 1.611  | Hammer                          | 16.509 | Cue                   | 3.245  |
| Avocado             | 41.284 | Hami melon                      | 7.327  | Flask                 | 30.841 |
| Mushroom            | 41.743 | Screwdriver                     | 19.851 | Soap                  | 28.176 |
| Recorder            | 6.746  | Bear                            | 63.175 | Eggplant              | 31.116 |
| Board Eraser        | 18.391 | Coconut                         | 36.595 | Tape Measure/ Ruler   | 19.759 |
| Pig                 | 51.289 | Showerhead                      | 26.336 | Globe                 | 49.033 |
| Chips               | 1.237  | Steak                           | 49.780 | Crosswalk Sign        | 15.744 |
| Stapler             | 36.877 | Camel                           | 46.527 | Formula 1             | 19.344 |
| Pomegranate         | 6.973  | Dishwasher                      | 71.976 | Crab                  | 13.930 |
| Hoverboard          | 10.567 | Meatball                        | 52.815 | Rice Cooker           | 47.593 |
| Tuba                | 20.817 | Calculator                      | 57.062 | Papaya                | 16.166 |
| Antelope            | 46.709 | Parrot                          | 45.860 | Seal                  | 29.411 |
| Butterfly           | 54.116 | Dumbbell                        | 4.441  | Donkey                | 43.895 |
| Lion                | 28.155 | Urinal                          | 68.751 | Dolphin               | 42.624 |
| Electric Drill      | 27.860 | Hair Dryer                      | 21.982 | Egg tart              | 33.806 |
| Jellyfish           | 14.757 | Treadmill                       | 20.196 | Lighter               | 25.163 |
| Grapefruit          | 3.345  | Game board                      | 48.177 | Mop                   | 16.337 |
| Radish              | 1.495  | Baozi                           | 29.727 | Target                | 11.463 |
| French              | 4.318  | Spring Rolls                    | 52.520 | Monkey                | 44.828 |
| Rabbit              | 36.208 | Pencil Case                     | 23.091 | Yak                   | 68.795 |
| Red Cabbage         | 12.301 | Binoculars                      | 20.399 | Asparagus             | 37.216 |
| Barbell             | 2.980  | Scallop                         | 28.860 | Noddles               | 6.921  |
| Comb                | 37.605 | Dumpling                        | 10.864 | Oyster                | 48.739 |
| Table Tennis paddle | 19.410 | Cosmetics Brush/Eyeliner Pencil | 34.466 | Chainsaw              | 9.891  |
| Eraser              | 21.543 | Lobster                         | 10.429 | Durian                | 8.318  |
| Okra                | 3.356  | Lipstick                        | 23.288 | Cosmetics Mirror      | 11.260 |
| Curling             | 66.642 | Table Tennis                    | 3.798  |                       |        |
[04/30 02:34:39 detectron2]: Evaluation results for objects365_v2_val_spotdet_gt in csv format:
[04/30 02:34:39 d2.evaluation.testing]: copypaste: Task: bbox
[04/30 02:34:39 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/30 02:34:39 d2.evaluation.testing]: copypaste: 33.9843,48.6442,36.4504,17.3352,34.9160,47.6705