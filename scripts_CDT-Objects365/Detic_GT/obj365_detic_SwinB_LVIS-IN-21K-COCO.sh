#!/bin/bash

CFG_PATH="./configs_detic/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

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



[04/30 02:22:42 d2.evaluation.coco_evaluation]: Preparing results for COCO format ...
[04/30 02:22:45 d2.evaluation.coco_evaluation]: Saving results to ./output/Detic/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size/inference_objects365_v2_val_spotdet_gt/coco_instances_results.json
[04/30 02:23:35 d2.evaluation.coco_evaluation]: Evaluating predictions with unofficial COCO API...
Loading and preparing results...
DONE (t=53.29s)
creating index...
index created!
[04/30 02:24:34 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[04/30 02:35:08 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 633.94 seconds.
[04/30 02:35:26 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[04/30 02:36:27 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 61.04 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.180
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.602
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.725
[04/30 02:36:27 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 36.537 | 51.647 | 39.188 | 18.050 | 37.073 | 50.496 |
[04/30 02:36:31 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 63.415 | Sneakers                        | 37.921 | Chair                 | 51.036 |
| Other Shoes         | 10.518 | Hat                             | 51.076 | Car                   | 31.373 |
| Lamp                | 27.475 | Glasses                         | 37.549 | Bottle                | 40.028 |
| Desk                | 30.932 | Cup                             | 56.343 | Street Lights         | 10.273 |
| Cabinet/shelf       | 17.759 | Handbag/Satchel                 | 27.186 | Bracelet              | 27.112 |
| Plate               | 57.427 | Picture/Frame                   | 24.354 | Helmet                | 43.875 |
| Book                | 25.994 | Gloves                          | 42.577 | Storage box           | 23.933 |
| Boat                | 28.506 | Leather Shoes                   | 18.660 | Flower                | 17.834 |
| Bench               | 29.497 | Potted Plant                    | 41.096 | Bowl/Basin            | 58.833 |
| Flag                | 48.398 | Pillow                          | 57.550 | Boots                 | 43.709 |
| Vase                | 39.487 | Microphone                      | 17.430 | Necklace              | 34.595 |
| Ring                | 17.531 | SUV                             | 25.058 | Wine Glass            | 68.893 |
| Belt                | 37.157 | Monitor/TV                      | 60.403 | Backpack              | 41.866 |
| Umbrella            | 33.604 | Traffic Light                   | 38.503 | Speaker               | 45.322 |
| Watch               | 49.334 | Tie                             | 35.641 | Trash bin Can         | 55.944 |
| Slippers            | 30.166 | Bicycle                         | 50.451 | Stool                 | 49.640 |
| Barrel/bucket       | 40.625 | Van                             | 24.861 | Couch                 | 59.393 |
| Sandals             | 33.607 | Basket                          | 44.023 | Drum                  | 33.644 |
| Pen/Pencil          | 27.960 | Bus                             | 48.539 | Wild Bird             | 31.318 |
| High Heels          | 25.540 | Motorcycle                      | 37.294 | Guitar                | 47.892 |
| Carpet              | 42.050 | Cell Phone                      | 52.431 | Bread                 | 26.542 |
| Camera              | 36.977 | Canned                          | 37.284 | Truck                 | 28.864 |
| Traffic cone        | 46.595 | Cymbal                          | 37.062 | Lifesaver             | 15.114 |
| Towel               | 58.955 | Stuffed Toy                     | 43.974 | Candle                | 35.093 |
| Sailboat            | 26.122 | Laptop                          | 77.053 | Awning                | 35.638 |
| Bed                 | 61.363 | Faucet                          | 37.954 | Tent                  | 16.293 |
| Horse               | 56.728 | Mirror                          | 41.178 | Power outlet          | 39.659 |
| Sink                | 44.446 | Apple                           | 36.960 | Air Conditioner       | 30.568 |
| Knife               | 53.517 | Hockey Stick                    | 44.132 | Paddle                | 23.326 |
| Pickup Truck        | 47.696 | Fork                            | 61.881 | Traffic Sign          | 11.625 |
| Ballon              | 60.278 | Tripod                          | 8.936  | Dog                   | 67.307 |
| Spoon               | 51.286 | Clock                           | 62.529 | Pot                   | 39.850 |
| Cow                 | 48.274 | Cake                            | 27.785 | Dining Table          | 75.067 |
| Sheep               | 49.535 | Hanger                          | 4.955  | Blackboard/Whiteboard | 35.783 |
| Napkin              | 43.477 | Other Fish                      | 41.562 | Orange/Tangerine      | 24.747 |
| Toiletry            | 33.011 | Keyboard                        | 66.631 | Tomato                | 57.629 |
| Lantern             | 38.573 | Machinery Vehicle               | 22.149 | Fan                   | 44.224 |
| Green Vegetables    | 1.159  | Banana                          | 14.292 | Baseball Glove        | 53.878 |
| Airplane            | 61.167 | Mouse                           | 58.729 | Train                 | 44.160 |
| Pumpkin             | 63.340 | Soccer                          | 33.030 | Skiboard              | 12.235 |
| Luggage             | 42.446 | Nightstand                      | 43.020 | Teapot                | 41.200 |
| Telephone           | 32.586 | Trolley                         | 32.533 | Head Phone            | 37.561 |
| Sports Car          | 69.853 | Stop Sign                       | 50.331 | Dessert               | 32.376 |
| Scooter             | 36.382 | Stroller                        | 47.394 | Crane                 | 7.245  |
| Remote              | 52.540 | Refrigerator                    | 75.909 | Oven                  | 27.983 |
| Lemon               | 46.308 | Duck                            | 53.569 | Baseball Bat          | 56.783 |
| Surveillance Camera | 12.392 | Cat                             | 73.389 | Jug                   | 33.054 |
| Broccoli            | 51.614 | Piano                           | 27.471 | Pizza                 | 66.141 |
| Elephant            | 73.426 | Skateboard                      | 35.026 | Surfboard             | 53.481 |
| Gun                 | 24.235 | Skating and Skiing shoes        | 30.569 | Gas stove             | 22.080 |
| Donut               | 60.861 | Bow Tie                         | 28.471 | Carrot                | 47.675 |
| Toilet              | 78.206 | Kite                            | 52.591 | Strawberry            | 46.842 |
| Other Balls         | 51.210 | Shovel                          | 18.488 | Pepper                | 37.420 |
| Computer Box        | 12.697 | Toilet Paper                    | 54.207 | Cleaning Products     | 28.430 |
| Chopsticks          | 41.183 | Microwave                       | 72.703 | Pigeon                | 57.890 |
| Baseball            | 45.637 | Cutting/chopping Board          | 52.349 | Coffee Table          | 57.650 |
| Side Table          | 31.177 | Scissors                        | 47.647 | Marker                | 22.606 |
| Pie                 | 31.573 | Ladder                          | 42.198 | Snowboard             | 53.484 |
| Cookies             | 28.226 | Radiator                        | 52.362 | Fire Hydrant          | 56.234 |
| Basketball          | 40.591 | Zebra                           | 67.900 | Grape                 | 2.476  |
| Giraffe             | 70.300 | Potato                          | 37.545 | Sausage               | 50.712 |
| Tricycle            | 23.423 | Violin                          | 23.548 | Egg                   | 70.646 |
| Fire Extinguisher   | 48.960 | Candy                           | 5.765  | Fire Truck            | 55.823 |
| Billards            | 30.650 | Converter                       | 3.074  | Bathtub               | 60.650 |
| Wheelchair          | 58.076 | Golf Club                       | 35.881 | Briefcase             | 46.863 |
| Cucumber            | 44.224 | Cigar/Cigarette                 | 20.294 | Paint Brush           | 15.063 |
| Pear                | 21.558 | Heavy Truck                     | 38.161 | Hamburger             | 44.377 |
| Extractor           | 19.020 | Extension Cord                  | 6.394  | Tong                  | 11.147 |
| Tennis Racket       | 67.337 | Folder                          | 14.716 | American Football     | 25.091 |
| earphone            | 5.639  | Mask                            | 25.900 | Kettle                | 55.117 |
| Tennis              | 21.926 | Ship                            | 48.207 | Swing                 | 0.564  |
| Coffee Machine      | 56.421 | Slide                           | 33.987 | Carriage              | 8.014  |
| Onion               | 33.039 | Green beans                     | 15.984 | Projector             | 35.386 |
| Frisbee             | 69.637 | Washing Machine/Drying Machine  | 37.827 | Chicken               | 56.219 |
| Printer             | 58.896 | Watermelon                      | 43.573 | Saxophone             | 35.067 |
| Tissue              | 4.000  | Toothbrush                      | 40.764 | Ice cream             | 15.355 |
| Hot air balloon     | 72.754 | Cello                           | 16.812 | French Fries          | 0.756  |
| Scale               | 11.471 | Trophy                          | 31.971 | Cabbage               | 16.751 |
| Hot dog             | 45.446 | Blender                         | 57.122 | Peach                 | 37.323 |
| Rice                | 6.007  | Wallet/Purse                    | 44.346 | Volleyball            | 45.501 |
| Deer                | 57.451 | Goose                           | 50.834 | Tape                  | 29.945 |
| Tablet              | 23.849 | Cosmetics                       | 25.572 | Trumpet               | 23.087 |
| Pineapple           | 31.349 | Golf Ball                       | 45.138 | Ambulance             | 67.648 |
| Parking meter       | 23.061 | Mango                           | 5.224  | Key                   | 17.664 |
| Hurdle              | 1.620  | Fishing Rod                     | 27.191 | Medal                 | 11.795 |
| Flute               | 30.012 | Brush                           | 21.953 | Penguin               | 58.345 |
| Megaphone           | 20.475 | Corn                            | 38.984 | Lettuce               | 13.485 |
| Garlic              | 29.212 | Swan                            | 58.976 | Helicopter            | 46.325 |
| Green Onion         | 16.526 | Sandwich                        | 53.778 | Nuts                  | 17.329 |
| Speed Limit Sign    | 25.908 | Induction Cooker                | 23.551 | Broom                 | 23.046 |
| Trombone            | 9.354  | Plum                            | 10.060 | Rickshaw              | 25.051 |
| Goldfish            | 37.635 | Kiwi fruit                      | 35.636 | Router/modem          | 23.128 |
| Poker Card          | 42.035 | Toaster                         | 63.493 | Shrimp                | 41.290 |
| Sushi               | 50.623 | Cheese                          | 44.749 | Notepaper             | 13.807 |
| Cherry              | 24.572 | Pliers                          | 26.520 | CD                    | 20.173 |
| Pasta               | 4.269  | Hammer                          | 19.446 | Cue                   | 7.133  |
| Avocado             | 41.616 | Hami melon                      | 7.489  | Flask                 | 37.625 |
| Mushroom            | 42.196 | Screwdriver                     | 22.699 | Soap                  | 30.658 |
| Recorder            | 16.938 | Bear                            | 64.907 | Eggplant              | 30.710 |
| Board Eraser        | 12.305 | Coconut                         | 34.676 | Tape Measure/ Ruler   | 22.823 |
| Pig                 | 57.865 | Showerhead                      | 27.369 | Globe                 | 47.562 |
| Chips               | 1.440  | Steak                           | 50.829 | Crosswalk Sign        | 15.920 |
| Stapler             | 34.573 | Camel                           | 58.620 | Formula 1             | 43.999 |
| Pomegranate         | 14.343 | Dishwasher                      | 72.081 | Crab                  | 20.099 |
| Hoverboard          | 12.442 | Meatball                        | 55.700 | Rice Cooker           | 47.395 |
| Tuba                | 22.714 | Calculator                      | 56.526 | Papaya                | 20.099 |
| Antelope            | 47.112 | Parrot                          | 45.008 | Seal                  | 49.269 |
| Butterfly           | 54.727 | Dumbbell                        | 3.527  | Donkey                | 54.034 |
| Lion                | 36.590 | Urinal                          | 69.836 | Dolphin               | 48.873 |
| Electric Drill      | 27.861 | Hair Dryer                      | 23.174 | Egg tart              | 44.849 |
| Jellyfish           | 37.610 | Treadmill                       | 29.500 | Lighter               | 28.661 |
| Grapefruit          | 6.632  | Game board                      | 45.060 | Mop                   | 12.823 |
| Radish              | 2.800  | Baozi                           | 52.433 | Target                | 17.240 |
| French              | 2.925  | Spring Rolls                    | 53.323 | Monkey                | 51.289 |
| Rabbit              | 45.265 | Pencil Case                     | 30.137 | Yak                   | 70.263 |
| Red Cabbage         | 20.654 | Binoculars                      | 25.163 | Asparagus             | 34.884 |
| Barbell             | 1.599  | Scallop                         | 41.276 | Noddles               | 6.270  |
| Comb                | 44.772 | Dumpling                        | 35.972 | Oyster                | 60.377 |
| Table Tennis paddle | 17.507 | Cosmetics Brush/Eyeliner Pencil | 31.502 | Chainsaw              | 13.979 |
| Eraser              | 20.063 | Lobster                         | 11.377 | Durian                | 42.148 |
| Okra                | 8.830  | Lipstick                        | 32.992 | Cosmetics Mirror      | 12.733 |
| Curling             | 69.964 | Table Tennis                    | 2.977  |                       |        |
[04/30 02:37:37 detectron2]: Evaluation results for objects365_v2_val_spotdet_gt in csv format:
[04/30 02:37:37 d2.evaluation.testing]: copypaste: Task: bbox
[04/30 02:37:37 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/30 02:37:37 d2.evaluation.testing]: copypaste: 36.5369,51.6473,39.1880,18.0500,37.0730,50.4961
