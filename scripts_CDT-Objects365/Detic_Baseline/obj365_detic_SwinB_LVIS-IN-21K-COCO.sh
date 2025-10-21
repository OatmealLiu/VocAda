#!/bin/bash


CFG_PATH="./configs_detic/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

CLASSIFIER_NAME="o365_clip_a+cnamefix.npy"

CUDA_VISIBLE_DEVICES=1,2,3  python train_net_detic.py \
        --num-gpus 3 \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('objects365_v2_val',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(365,)" \
        MODEL.MASK_ON False


[04/30 11:51:51 d2.evaluation.coco_evaluation]: Saving results to ./output/Detic/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size/inference_objects365_v2_val/coco_instances_results.json
[04/30 11:53:17 d2.evaluation.coco_evaluation]: Evaluating predictions with unofficial COCO API...
Loading and preparing results...
DONE (t=99.26s)
creating index...
index created!
[04/30 11:55:09 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[04/30 12:11:49 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 1000.26 seconds.
[04/30 12:12:20 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[04/30 12:15:02 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 161.60 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.294
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.234
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.091
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.539
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.668
[04/30 12:15:02 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 21.635 | 29.416 | 23.404 | 9.056 | 21.376 | 31.858 |
[04/30 12:15:05 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 61.468 | Sneakers                        | 18.142 | Chair                 | 38.995 |
| Other Shoes         | 2.186  | Hat                             | 40.025 | Car                   | 19.052 |
| Lamp                | 21.727 | Glasses                         | 29.293 | Bottle                | 24.611 |
| Desk                | 21.864 | Cup                             | 45.261 | Street Lights         | 5.366  |
| Cabinet/shelf       | 15.712 | Handbag/Satchel                 | 18.001 | Bracelet              | 19.903 |
| Plate               | 51.661 | Picture/Frame                   | 7.491  | Helmet                | 33.481 |
| Book                | 15.129 | Gloves                          | 26.413 | Storage box           | 9.951  |
| Boat                | 23.376 | Leather Shoes                   | 1.941  | Flower                | 3.380  |
| Bench               | 19.157 | Potted Plant                    | 29.226 | Bowl/Basin            | 45.778 |
| Flag                | 26.838 | Pillow                          | 46.672 | Boots                 | 27.804 |
| Vase                | 21.073 | Microphone                      | 15.254 | Necklace              | 25.394 |
| Ring                | 14.606 | SUV                             | 7.498  | Wine Glass            | 60.888 |
| Belt                | 26.523 | Monitor/TV                      | 48.385 | Backpack              | 24.883 |
| Umbrella            | 24.794 | Traffic Light                   | 34.500 | Speaker               | 34.074 |
| Watch               | 39.609 | Tie                             | 20.504 | Trash bin Can         | 41.346 |
| Slippers            | 9.010  | Bicycle                         | 43.754 | Stool                 | 36.897 |
| Barrel/bucket       | 26.808 | Van                             | 9.279  | Couch                 | 45.022 |
| Sandals             | 11.680 | Basket                          | 32.807 | Drum                  | 30.801 |
| Pen/Pencil          | 21.084 | Bus                             | 36.250 | Wild Bird             | 11.224 |
| High Heels          | 5.814  | Motorcycle                      | 24.862 | Guitar                | 43.640 |
| Carpet              | 31.099 | Cell Phone                      | 40.245 | Bread                 | 13.769 |
| Camera              | 26.591 | Canned                          | 19.550 | Truck                 | 11.842 |
| Traffic cone        | 39.871 | Cymbal                          | 35.019 | Lifesaver             | 7.546  |
| Towel               | 48.975 | Stuffed Toy                     | 25.349 | Candle                | 19.217 |
| Sailboat            | 16.977 | Laptop                          | 71.872 | Awning                | 17.522 |
| Bed                 | 54.324 | Faucet                          | 36.704 | Tent                  | 6.288  |
| Horse               | 44.871 | Mirror                          | 34.217 | Power outlet          | 21.357 |
| Sink                | 42.600 | Apple                           | 25.386 | Air Conditioner       | 18.991 |
| Knife               | 46.579 | Hockey Stick                    | 29.822 | Paddle                | 17.255 |
| Pickup Truck        | 25.313 | Fork                            | 58.374 | Traffic Sign          | 2.251  |
| Ballon              | 25.408 | Tripod                          | 5.331  | Dog                   | 57.386 |
| Spoon               | 40.663 | Clock                           | 49.298 | Pot                   | 31.536 |
| Cow                 | 18.964 | Cake                            | 11.482 | Dining Table          | 18.723 |
| Sheep               | 28.177 | Hanger                          | 3.968  | Blackboard/Whiteboard | 18.901 |
| Napkin              | 21.235 | Other Fish                      | 32.309 | Orange/Tangerine      | 7.675  |
| Toiletry            | 22.540 | Keyboard                        | 61.823 | Tomato                | 49.300 |
| Lantern             | 32.229 | Machinery Vehicle               | 12.160 | Fan                   | 36.432 |
| Green Vegetables    | 0.284  | Banana                          | 13.061 | Baseball Glove        | 37.909 |
| Airplane            | 58.570 | Mouse                           | 53.828 | Train                 | 35.580 |
| Pumpkin             | 53.148 | Soccer                          | 9.264  | Skiboard              | 1.390  |
| Luggage             | 22.707 | Nightstand                      | 16.803 | Teapot                | 21.818 |
| Telephone           | 27.145 | Trolley                         | 16.815 | Head Phone            | 28.180 |
| Sports Car          | 47.191 | Stop Sign                       | 39.025 | Dessert               | 11.292 |
| Scooter             | 14.272 | Stroller                        | 21.699 | Crane                 | 1.932  |
| Remote              | 36.927 | Refrigerator                    | 66.825 | Oven                  | 23.035 |
| Lemon               | 30.348 | Duck                            | 39.730 | Baseball Bat          | 30.586 |
| Surveillance Camera | 1.877  | Cat                             | 66.646 | Jug                   | 7.641  |
| Broccoli            | 45.839 | Piano                           | 21.369 | Pizza                 | 51.610 |
| Elephant            | 70.116 | Skateboard                      | 11.607 | Surfboard             | 45.420 |
| Gun                 | 17.767 | Skating and Skiing shoes        | 22.101 | Gas stove             | 16.675 |
| Donut               | 51.821 | Bow Tie                         | 21.192 | Carrot                | 31.417 |
| Toilet              | 74.675 | Kite                            | 33.905 | Strawberry            | 37.955 |
| Other Balls         | 7.732  | Shovel                          | 6.071  | Pepper                | 23.812 |
| Computer Box        | 1.370  | Toilet Paper                    | 34.754 | Cleaning Products     | 10.987 |
| Chopsticks          | 28.108 | Microwave                       | 62.227 | Pigeon                | 50.437 |
| Baseball            | 24.843 | Cutting/chopping Board          | 34.349 | Coffee Table          | 17.484 |
| Side Table          | 4.073  | Scissors                        | 41.822 | Marker                | 11.000 |
| Pie                 | 6.251  | Ladder                          | 24.065 | Snowboard             | 40.901 |
| Cookies             | 14.562 | Radiator                        | 40.528 | Fire Hydrant          | 37.744 |
| Basketball          | 15.455 | Zebra                           | 65.799 | Grape                 | 2.068  |
| Giraffe             | 67.629 | Potato                          | 14.191 | Sausage               | 30.492 |
| Tricycle            | 6.417  | Violin                          | 12.724 | Egg                   | 61.591 |
| Fire Extinguisher   | 38.828 | Candy                           | 1.241  | Fire Truck            | 33.473 |
| Billards            | 11.055 | Converter                       | 0.149  | Bathtub               | 54.958 |
| Wheelchair          | 38.957 | Golf Club                       | 28.949 | Briefcase             | 5.833  |
| Cucumber            | 25.888 | Cigar/Cigarette                 | 7.943  | Paint Brush           | 2.397  |
| Pear                | 12.836 | Heavy Truck                     | 10.345 | Hamburger             | 17.179 |
| Extractor           | 2.475  | Extension Cord                  | 1.006  | Tong                  | 0.141  |
| Tennis Racket       | 55.107 | Folder                          | 1.899  | American Football     | 5.207  |
| earphone            | 0.955  | Mask                            | 10.532 | Kettle                | 28.829 |
| Tennis              | 4.513  | Ship                            | 37.635 | Swing                 | 0.268  |
| Coffee Machine      | 35.402 | Slide                           | 32.178 | Carriage              | 5.152  |
| Onion               | 15.182 | Green beans                     | 7.057  | Projector             | 20.765 |
| Frisbee             | 49.386 | Washing Machine/Drying Machine  | 32.126 | Chicken               | 44.323 |
| Printer             | 49.879 | Watermelon                      | 32.309 | Saxophone             | 28.481 |
| Tissue              | 0.592  | Toothbrush                      | 35.113 | Ice cream             | 4.469  |
| Hot air balloon     | 28.721 | Cello                           | 8.707  | French Fries          | 0.088  |
| Scale               | 5.218  | Trophy                          | 18.227 | Cabbage               | 9.574  |
| Hot dog             | 8.188  | Blender                         | 44.017 | Peach                 | 23.386 |
| Rice                | 2.845  | Wallet/Purse                    | 25.606 | Volleyball            | 15.642 |
| Deer                | 43.696 | Goose                           | 15.137 | Tape                  | 17.595 |
| Tablet              | 2.646  | Cosmetics                       | 3.018  | Trumpet               | 12.503 |
| Pineapple           | 22.278 | Golf Ball                       | 13.872 | Ambulance             | 30.655 |
| Parking meter       | 18.186 | Mango                           | 0.647  | Key                   | 12.030 |
| Hurdle              | 0.017  | Fishing Rod                     | 14.402 | Medal                 | 3.722  |
| Flute               | 14.979 | Brush                           | 6.932  | Penguin               | 50.961 |
| Megaphone           | 4.203  | Corn                            | 11.431 | Lettuce               | 1.818  |
| Garlic              | 11.998 | Swan                            | 42.164 | Helicopter            | 38.502 |
| Green Onion         | 2.374  | Sandwich                        | 15.820 | Nuts                  | 0.669  |
| Speed Limit Sign    | 10.082 | Induction Cooker                | 4.219  | Broom                 | 12.273 |
| Trombone            | 3.329  | Plum                            | 0.743  | Rickshaw              | 2.518  |
| Goldfish            | 9.304  | Kiwi fruit                      | 24.126 | Router/modem          | 9.477  |
| Poker Card          | 16.353 | Toaster                         | 46.318 | Shrimp                | 17.881 |
| Sushi               | 37.638 | Cheese                          | 22.039 | Notepaper             | 1.953  |
| Cherry              | 9.056  | Pliers                          | 15.716 | CD                    | 6.803  |
| Pasta               | 0.670  | Hammer                          | 11.988 | Cue                   | 2.888  |
| Avocado             | 21.421 | Hami melon                      | 1.296  | Flask                 | 0.737  |
| Mushroom            | 16.253 | Screwdriver                     | 9.906  | Soap                  | 17.778 |
| Recorder            | 0.240  | Bear                            | 39.698 | Eggplant              | 14.640 |
| Board Eraser        | 0.640  | Coconut                         | 8.481  | Tape Measure/ Ruler   | 11.787 |
| Pig                 | 35.140 | Showerhead                      | 17.055 | Globe                 | 23.793 |
| Chips               | 0.395  | Steak                           | 21.030 | Crosswalk Sign        | 1.924  |
| Stapler             | 13.327 | Camel                           | 37.388 | Formula 1             | 6.309  |
| Pomegranate         | 0.673  | Dishwasher                      | 32.451 | Crab                  | 11.042 |
| Hoverboard          | 0.102  | Meatball                        | 25.102 | Rice Cooker           | 13.715 |
| Tuba                | 9.778  | Calculator                      | 35.083 | Papaya                | 8.477  |
| Antelope            | 9.119  | Parrot                          | 22.619 | Seal                  | 28.035 |
| Butterfly           | 35.123 | Dumbbell                        | 2.198  | Donkey                | 12.830 |
| Lion                | 6.008  | Urinal                          | 60.681 | Dolphin               | 17.679 |
| Electric Drill      | 13.937 | Hair Dryer                      | 14.229 | Egg tart              | 6.063  |
| Jellyfish           | 21.277 | Treadmill                       | 26.297 | Lighter               | 4.353  |
| Grapefruit          | 0.284  | Game board                      | 15.415 | Mop                   | 3.006  |
| Radish              | 0.284  | Baozi                           | 1.166  | Target                | 0.747  |
| French              | 0.000  | Spring Rolls                    | 16.884 | Monkey                | 35.961 |
| Rabbit              | 23.425 | Pencil Case                     | 7.298  | Yak                   | 26.804 |
| Red Cabbage         | 3.481  | Binoculars                      | 2.397  | Asparagus             | 6.682  |
| Barbell             | 0.746  | Scallop                         | 17.790 | Noddles               | 0.371  |
| Comb                | 15.431 | Dumpling                        | 0.955  | Oyster                | 39.002 |
| Table Tennis paddle | 0.290  | Cosmetics Brush/Eyeliner Pencil | 1.226  | Chainsaw              | 3.153  |
| Eraser              | 1.758  | Lobster                         | 3.941  | Durian                | 3.751  |
| Okra                | 0.309  | Lipstick                        | 5.164  | Cosmetics Mirror      | 0.524  |
| Curling             | 1.486  | Table Tennis                    | 0.001  |                       |        |
[04/30 12:16:39 detectron2]: Evaluation results for objects365_v2_val in csv format:
[04/30 12:16:39 d2.evaluation.testing]: copypaste: Task: bbox
[04/30 12:16:39 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/30 12:16:39 d2.evaluation.testing]: copypaste: 21.6347,29.4159,23.4036,9.0556,21.3757,31.8583