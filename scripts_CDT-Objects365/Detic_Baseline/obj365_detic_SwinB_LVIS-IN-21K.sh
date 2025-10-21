#!/bin/bash


CFG_PATH="./configs_detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

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


[04/30 11:49:34 d2.evaluation.coco_evaluation]: Preparing results for COCO format ...
[04/30 11:49:40 d2.evaluation.coco_evaluation]: Saving results to ./output/Detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size/inference_objects365_v2_val/coco_instances_results.json
[04/30 11:51:04 d2.evaluation.coco_evaluation]: Evaluating predictions with unofficial COCO API...
Loading and preparing results...
DONE (t=98.25s)
creating index...
index created!
[04/30 11:52:55 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[04/30 12:09:03 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 967.94 seconds.
[04/30 12:09:22 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[04/30 12:11:36 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 134.42 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.295
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.090
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.548
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.653
[04/30 12:11:36 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 21.447 | 29.491 | 23.214 | 8.981 | 21.301 | 31.786 |
[04/30 12:11:39 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 14.182 | Sneakers                        | 20.602 | Chair                 | 27.786 |
| Other Shoes         | 1.846  | Hat                             | 39.109 | Car                   | 13.991 |
| Lamp                | 21.188 | Glasses                         | 31.677 | Bottle                | 18.639 |
| Desk                | 20.281 | Cup                             | 33.154 | Street Lights         | 5.076  |
| Cabinet/shelf       | 15.868 | Handbag/Satchel                 | 14.401 | Bracelet              | 20.538 |
| Plate               | 54.709 | Picture/Frame                   | 8.140  | Helmet                | 35.534 |
| Book                | 8.590  | Gloves                          | 25.867 | Storage box           | 11.274 |
| Boat                | 23.150 | Leather Shoes                   | 2.137  | Flower                | 2.912  |
| Bench               | 15.331 | Potted Plant                    | 0.270  | Bowl/Basin            | 37.724 |
| Flag                | 28.830 | Pillow                          | 44.864 | Boots                 | 27.779 |
| Vase                | 17.373 | Microphone                      | 15.986 | Necklace              | 27.403 |
| Ring                | 15.397 | SUV                             | 6.378  | Wine Glass            | 56.236 |
| Belt                | 27.676 | Monitor/TV                      | 52.880 | Backpack              | 23.215 |
| Umbrella            | 22.912 | Traffic Light                   | 32.330 | Speaker               | 35.066 |
| Watch               | 39.902 | Tie                             | 19.595 | Trash bin Can         | 39.802 |
| Slippers            | 8.786  | Bicycle                         | 39.467 | Stool                 | 34.562 |
| Barrel/bucket       | 26.513 | Van                             | 8.310  | Couch                 | 42.967 |
| Sandals             | 10.443 | Basket                          | 33.330 | Drum                  | 30.344 |
| Pen/Pencil          | 22.073 | Bus                             | 32.697 | Wild Bird             | 12.985 |
| High Heels          | 5.954  | Motorcycle                      | 21.979 | Guitar                | 46.642 |
| Carpet              | 26.834 | Cell Phone                      | 36.175 | Bread                 | 13.369 |
| Camera              | 27.586 | Canned                          | 15.119 | Truck                 | 10.955 |
| Traffic cone        | 40.272 | Cymbal                          | 34.319 | Lifesaver             | 0.214  |
| Towel               | 47.922 | Stuffed Toy                     | 26.908 | Candle                | 20.126 |
| Sailboat            | 14.062 | Laptop                          | 67.795 | Awning                | 16.606 |
| Bed                 | 46.886 | Faucet                          | 36.241 | Tent                  | 5.737  |
| Horse               | 38.060 | Mirror                          | 33.424 | Power outlet          | 21.957 |
| Sink                | 35.188 | Apple                           | 24.986 | Air Conditioner       | 18.319 |
| Knife               | 42.119 | Hockey Stick                    | 36.792 | Paddle                | 20.181 |
| Pickup Truck        | 25.240 | Fork                            | 54.258 | Traffic Sign          | 2.746  |
| Ballon              | 25.985 | Tripod                          | 6.142  | Dog                   | 49.938 |
| Spoon               | 38.732 | Clock                           | 46.814 | Pot                   | 30.490 |
| Cow                 | 17.655 | Cake                            | 12.064 | Dining Table          | 13.046 |
| Sheep               | 23.403 | Hanger                          | 3.085  | Blackboard/Whiteboard | 17.588 |
| Napkin              | 22.172 | Other Fish                      | 32.295 | Orange/Tangerine      | 10.201 |
| Toiletry            | 23.344 | Keyboard                        | 54.443 | Tomato                | 49.285 |
| Lantern             | 33.491 | Machinery Vehicle               | 9.449  | Fan                   | 35.551 |
| Green Vegetables    | 0.187  | Banana                          | 9.525  | Baseball Glove        | 36.729 |
| Airplane            | 59.248 | Mouse                           | 51.841 | Train                 | 37.304 |
| Pumpkin             | 52.668 | Soccer                          | 10.515 | Skiboard              | 1.592  |
| Luggage             | 21.078 | Nightstand                      | 14.421 | Teapot                | 20.004 |
| Telephone           | 26.883 | Trolley                         | 15.368 | Head Phone            | 30.584 |
| Sports Car          | 50.989 | Stop Sign                       | 34.283 | Dessert               | 8.988  |
| Scooter             | 19.507 | Stroller                        | 28.240 | Crane                 | 1.664  |
| Remote              | 40.873 | Refrigerator                    | 65.389 | Oven                  | 26.876 |
| Lemon               | 32.102 | Duck                            | 43.708 | Baseball Bat          | 36.537 |
| Surveillance Camera | 1.918  | Cat                             | 59.552 | Jug                   | 7.405  |
| Broccoli            | 45.852 | Piano                           | 22.131 | Pizza                 | 48.152 |
| Elephant            | 68.389 | Skateboard                      | 13.933 | Surfboard             | 44.252 |
| Gun                 | 17.103 | Skating and Skiing shoes        | 18.363 | Gas stove             | 15.420 |
| Donut               | 49.903 | Bow Tie                         | 24.464 | Carrot                | 30.471 |
| Toilet              | 73.037 | Kite                            | 27.998 | Strawberry            | 37.330 |
| Other Balls         | 10.882 | Shovel                          | 6.645  | Pepper                | 24.768 |
| Computer Box        | 1.128  | Toilet Paper                    | 37.095 | Cleaning Products     | 13.023 |
| Chopsticks          | 28.836 | Microwave                       | 63.531 | Pigeon                | 49.243 |
| Baseball            | 27.331 | Cutting/chopping Board          | 37.049 | Coffee Table          | 14.277 |
| Side Table          | 2.415  | Scissors                        | 39.715 | Marker                | 10.949 |
| Pie                 | 6.007  | Ladder                          | 25.027 | Snowboard             | 40.308 |
| Cookies             | 16.081 | Radiator                        | 40.991 | Fire Hydrant          | 36.139 |
| Basketball          | 22.193 | Zebra                           | 65.010 | Grape                 | 1.188  |
| Giraffe             | 66.537 | Potato                          | 14.833 | Sausage               | 16.334 |
| Tricycle            | 5.168  | Violin                          | 9.388  | Egg                   | 60.914 |
| Fire Extinguisher   | 39.580 | Candy                           | 1.915  | Fire Truck            | 35.699 |
| Billards            | 11.468 | Converter                       | 0.514  | Bathtub               | 54.181 |
| Wheelchair          | 35.024 | Golf Club                       | 29.997 | Briefcase             | 4.665  |
| Cucumber            | 24.782 | Cigar/Cigarette                 | 4.626  | Paint Brush           | 2.257  |
| Pear                | 10.501 | Heavy Truck                     | 12.144 | Hamburger             | 15.536 |
| Extractor           | 1.735  | Extension Cord                  | 1.042  | Tong                  | 0.463  |
| Tennis Racket       | 55.901 | Folder                          | 3.247  | American Football     | 8.758  |
| earphone            | 1.012  | Mask                            | 10.223 | Kettle                | 28.382 |
| Tennis              | 15.072 | Ship                            | 35.287 | Swing                 | 0.459  |
| Coffee Machine      | 36.614 | Slide                           | 31.283 | Carriage              | 6.143  |
| Onion               | 16.218 | Green beans                     | 6.575  | Projector             | 18.294 |
| Frisbee             | 47.963 | Washing Machine/Drying Machine  | 28.511 | Chicken               | 46.341 |
| Printer             | 49.178 | Watermelon                      | 33.189 | Saxophone             | 32.747 |
| Tissue              | 0.432  | Toothbrush                      | 34.188 | Ice cream             | 4.996  |
| Hot air balloon     | 31.096 | Cello                           | 19.880 | French Fries          | 0.045  |
| Scale               | 7.850  | Trophy                          | 20.538 | Cabbage               | 10.474 |
| Hot dog             | 1.239  | Blender                         | 43.045 | Peach                 | 18.247 |
| Rice                | 3.610  | Wallet/Purse                    | 25.643 | Volleyball            | 26.166 |
| Deer                | 42.880 | Goose                           | 17.644 | Tape                  | 17.648 |
| Tablet              | 4.228  | Cosmetics                       | 5.687  | Trumpet               | 13.692 |
| Pineapple           | 20.508 | Golf Ball                       | 14.142 | Ambulance             | 54.615 |
| Parking meter       | 29.785 | Mango                           | 0.620  | Key                   | 11.641 |
| Hurdle              | 0.041  | Fishing Rod                     | 19.771 | Medal                 | 3.821  |
| Flute               | 20.690 | Brush                           | 6.568  | Penguin               | 53.954 |
| Megaphone           | 7.008  | Corn                            | 12.821 | Lettuce               | 2.568  |
| Garlic              | 16.984 | Swan                            | 40.933 | Helicopter            | 40.028 |
| Green Onion         | 2.275  | Sandwich                        | 17.504 | Nuts                  | 1.274  |
| Speed Limit Sign    | 13.046 | Induction Cooker                | 4.494  | Broom                 | 14.831 |
| Trombone            | 7.284  | Plum                            | 0.944  | Rickshaw              | 3.131  |
| Goldfish            | 13.206 | Kiwi fruit                      | 23.740 | Router/modem          | 9.061  |
| Poker Card          | 12.336 | Toaster                         | 44.236 | Shrimp                | 19.938 |
| Sushi               | 38.056 | Cheese                          | 14.303 | Notepaper             | 2.423  |
| Cherry              | 9.963  | Pliers                          | 15.347 | CD                    | 8.324  |
| Pasta               | 0.218  | Hammer                          | 10.683 | Cue                   | 4.067  |
| Avocado             | 23.207 | Hami melon                      | 1.946  | Flask                 | 1.845  |
| Mushroom            | 16.279 | Screwdriver                     | 11.782 | Soap                  | 15.131 |
| Recorder            | 0.274  | Bear                            | 38.383 | Eggplant              | 15.057 |
| Board Eraser        | 1.609  | Coconut                         | 20.950 | Tape Measure/ Ruler   | 9.497  |
| Pig                 | 33.000 | Showerhead                      | 16.587 | Globe                 | 30.493 |
| Chips               | 0.452  | Steak                           | 23.237 | Crosswalk Sign        | 1.971  |
| Stapler             | 17.022 | Camel                           | 40.072 | Formula 1             | 8.136  |
| Pomegranate         | 0.741  | Dishwasher                      | 32.278 | Crab                  | 9.952  |
| Hoverboard          | 0.095  | Meatball                        | 25.083 | Rice Cooker           | 19.765 |
| Tuba                | 12.535 | Calculator                      | 32.408 | Papaya                | 9.525  |
| Antelope            | 10.127 | Parrot                          | 30.308 | Seal                  | 34.986 |
| Butterfly           | 34.919 | Dumbbell                        | 4.535  | Donkey                | 25.882 |
| Lion                | 11.310 | Urinal                          | 61.466 | Dolphin               | 23.489 |
| Electric Drill      | 14.248 | Hair Dryer                      | 11.290 | Egg tart              | 4.341  |
| Jellyfish           | 23.065 | Treadmill                       | 23.043 | Lighter               | 5.334  |
| Grapefruit          | 0.172  | Game board                      | 13.911 | Mop                   | 3.581  |
| Radish              | 0.201  | Baozi                           | 2.427  | Target                | 0.171  |
| French              | 0.004  | Spring Rolls                    | 16.113 | Monkey                | 35.323 |
| Rabbit              | 20.289 | Pencil Case                     | 4.762  | Yak                   | 33.018 |
| Red Cabbage         | 6.143  | Binoculars                      | 4.320  | Asparagus             | 3.809  |
| Barbell             | 1.707  | Scallop                         | 15.143 | Noddles               | 0.374  |
| Comb                | 14.564 | Dumpling                        | 1.236  | Oyster                | 39.763 |
| Table Tennis paddle | 0.656  | Cosmetics Brush/Eyeliner Pencil | 2.293  | Chainsaw              | 2.896  |
| Eraser              | 1.493  | Lobster                         | 6.814  | Durian                | 0.404  |
| Okra                | 0.315  | Lipstick                        | 6.424  | Cosmetics Mirror      | 0.090  |
| Curling             | 2.239  | Table Tennis                    | 0.003  |                       |        |
[04/30 12:12:57 detectron2]: Evaluation results for objects365_v2_val in csv format:
[04/30 12:12:57 d2.evaluation.testing]: copypaste: Task: bbox
[04/30 12:12:57 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/30 12:12:57 d2.evaluation.testing]: copypaste: 21.4473,29.4910,23.2137,8.9807,21.3006,31.7864