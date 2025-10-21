#!/bin/bash

CFG_PATH="./configs_detic/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

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



#index created!
#[04/29 20:59:20 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
#[04/29 21:19:14 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 1193.55 seconds.
#[04/29 21:19:39 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
#[04/29 21:22:34 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 174.72 seconds.
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.213
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.293
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.230
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.091
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.213
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.315
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.231
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.475
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.516
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.340
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.534
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.641
#[04/29 21:22:34 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
#|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
#|:------:|:------:|:------:|:-----:|:------:|:------:|
#| 21.279 | 29.293 | 23.014 | 9.077 | 21.283 | 31.514 |
#[04/29 21:22:38 d2.evaluation.coco_evaluation]: Per-category bbox AP:
#| category            | AP     | category                        | AP     | category              | AP     |
#|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
#| Person              | 14.181 | Sneakers                        | 17.959 | Chair                 | 29.852 |
#| Other Shoes         | 2.257  | Hat                             | 39.985 | Car                   | 11.609 |
#| Lamp                | 21.450 | Glasses                         | 25.610 | Bottle                | 18.041 |
#| Desk                | 19.921 | Cup                             | 32.445 | Street Lights         | 4.785  |
#| Cabinet/shelf       | 14.836 | Handbag/Satchel                 | 14.335 | Bracelet              | 20.268 |
#| Plate               | 56.024 | Picture/Frame                   | 2.960  | Helmet                | 35.424 |
#| Book                | 8.501  | Gloves                          | 24.664 | Storage box           | 10.856 |
#| Boat                | 22.111 | Leather Shoes                   | 2.017  | Flower                | 1.224  |
#| Bench               | 16.568 | Potted Plant                    | 0.239  | Bowl/Basin            | 38.701 |
#| Flag                | 28.264 | Pillow                          | 45.931 | Boots                 | 28.192 |
#| Vase                | 17.398 | Microphone                      | 18.074 | Necklace              | 27.002 |
#| Ring                | 15.916 | SUV                             | 9.424  | Wine Glass            | 57.039 |
#| Belt                | 26.372 | Monitor/TV                      | 37.931 | Backpack              | 24.638 |
#| Umbrella            | 23.497 | Traffic Light                   | 31.692 | Speaker               | 31.607 |
#| Watch               | 40.297 | Tie                             | 13.431 | Trash bin Can         | 41.449 |
#| Slippers            | 10.634 | Bicycle                         | 41.688 | Stool                 | 35.308 |
#| Barrel/bucket       | 27.350 | Van                             | 8.042  | Couch                 | 38.923 |
#| Sandals             | 11.326 | Basket                          | 32.780 | Drum                  | 30.488 |
#| Pen/Pencil          | 23.948 | Bus                             | 24.452 | Wild Bird             | 13.341 |
#| High Heels          | 3.998  | Motorcycle                      | 25.218 | Guitar                | 48.371 |
#| Carpet              | 30.985 | Cell Phone                      | 33.218 | Bread                 | 13.909 |
#| Camera              | 28.074 | Canned                          | 13.261 | Truck                 | 11.853 |
#| Traffic cone        | 40.814 | Cymbal                          | 37.771 | Lifesaver             | 0.151  |
#| Towel               | 48.510 | Stuffed Toy                     | 21.201 | Candle                | 20.389 |
#| Sailboat            | 8.486  | Laptop                          | 68.053 | Awning                | 16.941 |
#| Bed                 | 49.338 | Faucet                          | 37.066 | Tent                  | 3.501  |
#| Horse               | 41.491 | Mirror                          | 32.878 | Power outlet          | 11.919 |
#| Sink                | 35.312 | Apple                           | 26.036 | Air Conditioner       | 20.146 |
#| Knife               | 44.166 | Hockey Stick                    | 36.771 | Paddle                | 24.013 |
#| Pickup Truck        | 29.490 | Fork                            | 55.827 | Traffic Sign          | 1.869  |
#| Ballon              | 21.833 | Tripod                          | 5.713  | Dog                   | 49.168 |
#| Spoon               | 40.845 | Clock                           | 43.064 | Pot                   | 32.150 |
#| Cow                 | 17.477 | Cake                            | 12.653 | Dining Table          | 14.544 |
#| Sheep               | 24.392 | Hanger                          | 2.232  | Blackboard/Whiteboard | 18.436 |
#| Napkin              | 22.659 | Other Fish                      | 33.994 | Orange/Tangerine      | 5.506  |
#| Toiletry            | 18.182 | Keyboard                        | 51.874 | Tomato                | 50.676 |
#| Lantern             | 32.489 | Machinery Vehicle               | 4.617  | Fan                   | 37.283 |
#| Green Vegetables    | 0.132  | Banana                          | 9.235  | Baseball Glove        | 37.740 |
#| Airplane            | 59.580 | Mouse                           | 47.233 | Train                 | 16.305 |
#| Pumpkin             | 52.002 | Soccer                          | 13.246 | Skiboard              | 0.760  |
#| Luggage             | 14.196 | Nightstand                      | 13.654 | Teapot                | 22.577 |
#| Telephone           | 28.700 | Trolley                         | 8.583  | Head Phone            | 30.164 |
#| Sports Car          | 37.576 | Stop Sign                       | 31.939 | Dessert               | 8.590  |
#| Scooter             | 19.926 | Stroller                        | 27.605 | Crane                 | 0.121  |
#| Remote              | 35.883 | Refrigerator                    | 66.133 | Oven                  | 26.808 |
#| Lemon               | 30.598 | Duck                            | 43.039 | Baseball Bat          | 38.767 |
#| Surveillance Camera | 2.346  | Cat                             | 59.868 | Jug                   | 6.886  |
#| Broccoli            | 45.527 | Piano                           | 22.857 | Pizza                 | 49.716 |
#| Elephant            | 69.543 | Skateboard                      | 18.857 | Surfboard             | 44.581 |
#| Gun                 | 17.058 | Skating and Skiing shoes        | 12.775 | Gas stove             | 15.886 |
#| Donut               | 47.045 | Bow Tie                         | 25.393 | Carrot                | 31.064 |
#| Toilet              | 72.899 | Kite                            | 42.128 | Strawberry            | 40.454 |
#| Other Balls         | 10.532 | Shovel                          | 8.227  | Pepper                | 6.821  |
#| Computer Box        | 0.724  | Toilet Paper                    | 31.024 | Cleaning Products     | 12.706 |
#| Chopsticks          | 30.907 | Microwave                       | 59.569 | Pigeon                | 50.321 |
#| Baseball            | 29.001 | Cutting/chopping Board          | 37.382 | Coffee Table          | 20.716 |
#| Side Table          | 3.553  | Scissors                        | 41.207 | Marker                | 13.278 |
#| Pie                 | 4.094  | Ladder                          | 23.794 | Snowboard             | 46.915 |
#| Cookies             | 13.755 | Radiator                        | 42.795 | Fire Hydrant          | 37.374 |
#| Basketball          | 27.896 | Zebra                           | 65.601 | Grape                 | 2.243  |
#| Giraffe             | 66.824 | Potato                          | 19.152 | Sausage               | 15.686 |
#| Tricycle            | 7.901  | Violin                          | 19.148 | Egg                   | 61.583 |
#| Fire Extinguisher   | 43.348 | Candy                           | 0.439  | Fire Truck            | 41.704 |
#| Billards            | 10.179 | Converter                       | 0.060  | Bathtub               | 55.086 |
#| Wheelchair          | 48.765 | Golf Club                       | 31.746 | Briefcase             | 4.988  |
#| Cucumber            | 29.945 | Cigar/Cigarette                 | 11.142 | Paint Brush           | 2.750  |
#| Pear                | 15.136 | Heavy Truck                     | 10.482 | Hamburger             | 16.870 |
#| Extractor           | 0.717  | Extension Cord                  | 0.296  | Tong                  | 1.165  |
#| Tennis Racket       | 53.050 | Folder                          | 3.119  | American Football     | 12.290 |
#| earphone            | 1.116  | Mask                            | 11.195 | Kettle                | 33.087 |
#| Tennis              | 17.371 | Ship                            | 20.098 | Swing                 | 0.062  |
#| Coffee Machine      | 43.696 | Slide                           | 31.805 | Carriage              | 5.297  |
#| Onion               | 22.385 | Green beans                     | 6.693  | Projector             | 27.006 |
#| Frisbee             | 56.569 | Washing Machine/Drying Machine  | 31.686 | Chicken               | 45.337 |
#| Printer             | 50.764 | Watermelon                      | 41.220 | Saxophone             | 37.738 |
#| Tissue              | 0.354  | Toothbrush                      | 35.832 | Ice cream             | 5.693  |
#| Hot air balloon     | 65.168 | Cello                           | 8.111  | French Fries          | 0.018  |
#| Scale               | 11.382 | Trophy                          | 19.502 | Cabbage               | 4.720  |
#| Hot dog             | 0.655  | Blender                         | 40.746 | Peach                 | 30.222 |
#| Rice                | 0.088  | Wallet/Purse                    | 28.402 | Volleyball            | 42.296 |
#| Deer                | 43.645 | Goose                           | 18.322 | Tape                  | 17.864 |
#| Tablet              | 3.242  | Cosmetics                       | 2.022  | Trumpet               | 7.803  |
#| Pineapple           | 22.007 | Golf Ball                       | 12.063 | Ambulance             | 64.668 |
#| Parking meter       | 30.961 | Mango                           | 0.396  | Key                   | 13.949 |
#| Hurdle              | 0.452  | Fishing Rod                     | 21.774 | Medal                 | 2.710  |
#| Flute               | 11.418 | Brush                           | 6.077  | Penguin               | 55.762 |
#| Megaphone           | 9.841  | Corn                            | 4.782  | Lettuce               | 2.299  |
#| Garlic              | 27.456 | Swan                            | 22.688 | Helicopter            | 40.916 |
#| Green Onion         | 3.670  | Sandwich                        | 19.024 | Nuts                  | 0.049  |
#| Speed Limit Sign    | 12.438 | Induction Cooker                | 2.456  | Broom                 | 15.939 |
#| Trombone            | 8.538  | Plum                            | 0.367  | Rickshaw              | 3.302  |
#| Goldfish            | 15.193 | Kiwi fruit                      | 24.517 | Router/modem          | 11.244 |
#| Poker Card          | 6.032  | Toaster                         | 48.503 | Shrimp                | 15.244 |
#| Sushi               | 43.192 | Cheese                          | 3.929  | Notepaper             | 2.740  |
#| Cherry              | 11.182 | Pliers                          | 21.572 | CD                    | 1.594  |
#| Pasta               | 0.028  | Hammer                          | 15.441 | Cue                   | 0.005  |
#| Avocado             | 31.014 | Hami melon                      | 1.536  | Flask                 | 0.544  |
#| Mushroom            | 18.668 | Screwdriver                     | 16.732 | Soap                  | 15.101 |
#| Recorder            | 0.026  | Bear                            | 38.729 | Eggplant              | 20.111 |
#| Board Eraser        | 0.910  | Coconut                         | 18.567 | Tape Measure/ Ruler   | 9.847  |
#| Pig                 | 32.989 | Showerhead                      | 17.108 | Globe                 | 33.280 |
#| Chips               | 0.123  | Steak                           | 19.065 | Crosswalk Sign        | 1.839  |
#| Stapler             | 23.841 | Camel                           | 52.391 | Formula 1             | 6.121  |
#| Pomegranate         | 0.272  | Dishwasher                      | 34.909 | Crab                  | 6.559  |
#| Hoverboard          | 0.090  | Meatball                        | 26.524 | Rice Cooker           | 18.932 |
#| Tuba                | 9.803  | Calculator                      | 40.714 | Papaya                | 15.662 |
#| Antelope            | 5.302  | Parrot                          | 27.031 | Seal                  | 1.154  |
#| Butterfly           | 38.940 | Dumbbell                        | 5.881  | Donkey                | 4.757  |
#| Lion                | 16.683 | Urinal                          | 59.217 | Dolphin               | 28.902 |
#| Electric Drill      | 17.096 | Hair Dryer                      | 13.784 | Egg tart              | 3.209  |
#| Jellyfish           | 0.698  | Treadmill                       | 5.743  | Lighter               | 2.815  |
#| Grapefruit          | 0.103  | Game board                      | 12.943 | Mop                   | 8.812  |
#| Radish              | 0.147  | Baozi                           | 0.078  | Target                | 0.263  |
#| French              | 0.001  | Spring Rolls                    | 21.974 | Monkey                | 27.261 |
#| Rabbit              | 23.207 | Pencil Case                     | 11.007 | Yak                   | 11.059 |
#| Red Cabbage         | 1.385  | Binoculars                      | 10.314 | Asparagus             | 9.545  |
#| Barbell             | 4.005  | Scallop                         | 6.439  | Noddles               | 0.138  |
#| Comb                | 5.466  | Dumpling                        | 0.096  | Oyster                | 16.594 |
#| Table Tennis paddle | 1.942  | Cosmetics Brush/Eyeliner Pencil | 2.219  | Chainsaw              | 0.373  |
#| Eraser              | 2.619  | Lobster                         | 1.094  | Durian                | 1.404  |
#| Okra                | 0.046  | Lipstick                        | 1.748  | Cosmetics Mirror      | 0.129  |
#| Curling             | 1.471  | Table Tennis                    | 0.011  |                       |        |
#[04/29 21:24:21 detectron2]: Evaluation results for objects365_v2_val in csv format:
#[04/29 21:24:21 d2.evaluation.testing]: copypaste: Task: bbox
#[04/29 21:24:21 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
#[04/29 21:24:21 d2.evaluation.testing]: copypaste: 21.2789,29.2927,23.0135,9.0765,21.2825,31.5136