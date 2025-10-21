#!/bin/bash


CFG_PATH="./configs_detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml"
MODEL_PATH="./models/Detic_CDT/BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth"

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
#[04/29 20:56:41 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
#[04/29 21:17:55 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 1273.97 seconds.
#[04/29 21:18:30 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
#[04/29 21:22:25 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 234.81 seconds.
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.194
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.266
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.210
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.083
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.195
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.285
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.229
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.471
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.509
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.334
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.527
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
#[04/29 21:22:25 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
#|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
#|:------:|:------:|:------:|:-----:|:------:|:------:|
#| 19.389 | 26.635 | 20.988 | 8.251 | 19.490 | 28.498 |
#[04/29 21:22:31 d2.evaluation.coco_evaluation]: Per-category bbox AP:
#| category            | AP     | category                        | AP     | category              | AP     |
#|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
#| Person              | 14.978 | Sneakers                        | 14.396 | Chair                 | 30.006 |
#| Other Shoes         | 2.078  | Hat                             | 39.365 | Car                   | 12.138 |
#| Lamp                | 21.633 | Glasses                         | 26.158 | Bottle                | 18.395 |
#| Desk                | 20.100 | Cup                             | 31.578 | Street Lights         | 5.108  |
#| Cabinet/shelf       | 16.769 | Handbag/Satchel                 | 13.693 | Bracelet              | 20.844 |
#| Plate               | 56.297 | Picture/Frame                   | 3.447  | Helmet                | 36.052 |
#| Book                | 8.577  | Gloves                          | 25.191 | Storage box           | 11.452 |
#| Boat                | 22.651 | Leather Shoes                   | 2.120  | Flower                | 1.403  |
#| Bench               | 16.167 | Potted Plant                    | 0.873  | Bowl/Basin            | 37.660 |
#| Flag                | 27.842 | Pillow                          | 45.517 | Boots                 | 29.077 |
#| Vase                | 16.973 | Microphone                      | 17.016 | Necklace              | 27.279 |
#| Ring                | 15.200 | SUV                             | 5.878  | Wine Glass            | 56.141 |
#| Belt                | 27.471 | Monitor/TV                      | 38.465 | Backpack              | 23.802 |
#| Umbrella            | 22.380 | Traffic Light                   | 31.352 | Speaker               | 31.549 |
#| Watch               | 40.239 | Tie                             | 10.277 | Trash bin Can         | 42.244 |
#| Slippers            | 9.225  | Bicycle                         | 42.157 | Stool                 | 34.887 |
#| Barrel/bucket       | 27.191 | Van                             | 6.227  | Couch                 | 41.529 |
#| Sandals             | 10.400 | Basket                          | 33.118 | Drum                  | 27.281 |
#| Pen/Pencil          | 22.359 | Bus                             | 24.416 | Wild Bird             | 12.543 |
#| High Heels          | 4.971  | Motorcycle                      | 24.827 | Guitar                | 43.792 |
#| Carpet              | 29.117 | Cell Phone                      | 34.466 | Bread                 | 13.831 |
#| Camera              | 28.207 | Canned                          | 15.143 | Truck                 | 11.053 |
#| Traffic cone        | 40.021 | Cymbal                          | 36.891 | Lifesaver             | 0.275  |
#| Towel               | 48.456 | Stuffed Toy                     | 20.371 | Candle                | 19.640 |
#| Sailboat            | 7.628  | Laptop                          | 67.929 | Awning                | 17.116 |
#| Bed                 | 48.246 | Faucet                          | 37.257 | Tent                  | 3.819  |
#| Horse               | 37.409 | Mirror                          | 33.800 | Power outlet          | 9.847  |
#| Sink                | 35.031 | Apple                           | 24.519 | Air Conditioner       | 18.933 |
#| Knife               | 44.756 | Hockey Stick                    | 25.181 | Paddle                | 21.870 |
#| Pickup Truck        | 28.293 | Fork                            | 56.053 | Traffic Sign          | 2.233  |
#| Ballon              | 23.722 | Tripod                          | 6.486  | Dog                   | 49.852 |
#| Spoon               | 40.817 | Clock                           | 45.893 | Pot                   | 31.027 |
#| Cow                 | 16.116 | Cake                            | 12.823 | Dining Table          | 14.870 |
#| Sheep               | 24.070 | Hanger                          | 1.755  | Blackboard/Whiteboard | 19.163 |
#| Napkin              | 23.552 | Other Fish                      | 28.083 | Orange/Tangerine      | 4.237  |
#| Toiletry            | 19.008 | Keyboard                        | 54.271 | Tomato                | 49.818 |
#| Lantern             | 32.037 | Machinery Vehicle               | 6.763  | Fan                   | 36.972 |
#| Green Vegetables    | 0.206  | Banana                          | 9.242  | Baseball Glove        | 34.198 |
#| Airplane            | 60.617 | Mouse                           | 44.247 | Train                 | 16.518 |
#| Pumpkin             | 52.142 | Soccer                          | 11.545 | Skiboard              | 0.758  |
#| Luggage             | 15.779 | Nightstand                      | 12.058 | Teapot                | 21.600 |
#| Telephone           | 27.246 | Trolley                         | 15.520 | Head Phone            | 28.261 |
#| Sports Car          | 27.113 | Stop Sign                       | 32.628 | Dessert               | 7.749  |
#| Scooter             | 17.542 | Stroller                        | 24.876 | Crane                 | 0.111  |
#| Remote              | 34.299 | Refrigerator                    | 66.655 | Oven                  | 27.794 |
#| Lemon               | 29.354 | Duck                            | 42.726 | Baseball Bat          | 32.484 |
#| Surveillance Camera | 2.423  | Cat                             | 58.655 | Jug                   | 4.901  |
#| Broccoli            | 45.801 | Piano                           | 24.617 | Pizza                 | 50.303 |
#| Elephant            | 68.859 | Skateboard                      | 14.564 | Surfboard             | 44.263 |
#| Gun                 | 7.416  | Skating and Skiing shoes        | 23.164 | Gas stove             | 15.896 |
#| Donut               | 49.218 | Bow Tie                         | 23.716 | Carrot                | 29.565 |
#| Toilet              | 72.875 | Kite                            | 29.061 | Strawberry            | 39.695 |
#| Other Balls         | 9.081  | Shovel                          | 5.843  | Pepper                | 15.879 |
#| Computer Box        | 0.504  | Toilet Paper                    | 30.733 | Cleaning Products     | 13.797 |
#| Chopsticks          | 30.306 | Microwave                       | 60.004 | Pigeon                | 49.162 |
#| Baseball            | 27.190 | Cutting/chopping Board          | 38.234 | Coffee Table          | 17.395 |
#| Side Table          | 3.466  | Scissors                        | 41.317 | Marker                | 11.388 |
#| Pie                 | 4.493  | Ladder                          | 24.675 | Snowboard             | 45.101 |
#| Cookies             | 13.862 | Radiator                        | 42.302 | Fire Hydrant          | 35.605 |
#| Basketball          | 22.446 | Zebra                           | 65.420 | Grape                 | 1.359  |
#| Giraffe             | 67.155 | Potato                          | 17.206 | Sausage               | 17.197 |
#| Tricycle            | 5.293  | Violin                          | 4.674  | Egg                   | 61.136 |
#| Fire Extinguisher   | 40.491 | Candy                           | 1.373  | Fire Truck            | 39.033 |
#| Billards            | 1.040  | Converter                       | 0.033  | Bathtub               | 54.413 |
#| Wheelchair          | 40.054 | Golf Club                       | 23.689 | Briefcase             | 4.324  |
#| Cucumber            | 27.087 | Cigar/Cigarette                 | 7.009  | Paint Brush           | 1.524  |
#| Pear                | 12.019 | Heavy Truck                     | 12.102 | Hamburger             | 14.612 |
#| Extractor           | 0.319  | Extension Cord                  | 0.240  | Tong                  | 0.379  |
#| Tennis Racket       | 53.841 | Folder                          | 2.597  | American Football     | 8.369  |
#| earphone            | 0.954  | Mask                            | 16.963 | Kettle                | 28.231 |
#| Tennis              | 11.056 | Ship                            | 16.739 | Swing                 | 0.100  |
#| Coffee Machine      | 42.511 | Slide                           | 30.730 | Carriage              | 5.569  |
#| Onion               | 19.211 | Green beans                     | 5.662  | Projector             | 22.805 |
#| Frisbee             | 48.700 | Washing Machine/Drying Machine  | 28.446 | Chicken               | 42.659 |
#| Printer             | 50.031 | Watermelon                      | 35.128 | Saxophone             | 26.982 |
#| Tissue              | 0.401  | Toothbrush                      | 35.331 | Ice cream             | 2.666  |
#| Hot air balloon     | 36.342 | Cello                           | 4.857  | French Fries          | 0.025  |
#| Scale               | 5.863  | Trophy                          | 22.419 | Cabbage               | 6.238  |
#| Hot dog             | 0.530  | Blender                         | 39.674 | Peach                 | 21.780 |
#| Rice                | 0.023  | Wallet/Purse                    | 26.082 | Volleyball            | 18.784 |
#| Deer                | 36.724 | Goose                           | 16.446 | Tape                  | 14.658 |
#| Tablet              | 1.500  | Cosmetics                       | 0.748  | Trumpet               | 8.797  |
#| Pineapple           | 20.890 | Golf Ball                       | 5.017  | Ambulance             | 47.944 |
#| Parking meter       | 30.504 | Mango                           | 0.242  | Key                   | 11.013 |
#| Hurdle              | 0.466  | Fishing Rod                     | 15.922 | Medal                 | 3.020  |
#| Flute               | 5.749  | Brush                           | 4.673  | Penguin               | 49.585 |
#| Megaphone           | 3.530  | Corn                            | 4.407  | Lettuce               | 2.286  |
#| Garlic              | 20.026 | Swan                            | 18.672 | Helicopter            | 38.409 |
#| Green Onion         | 3.712  | Sandwich                        | 16.245 | Nuts                  | 0.033  |
#| Speed Limit Sign    | 12.167 | Induction Cooker                | 3.360  | Broom                 | 13.317 |
#| Trombone            | 4.999  | Plum                            | 0.216  | Rickshaw              | 1.852  |
#| Goldfish            | 5.950  | Kiwi fruit                      | 23.673 | Router/modem          | 4.727  |
#| Poker Card          | 5.079  | Toaster                         | 44.689 | Shrimp                | 15.777 |
#| Sushi               | 38.992 | Cheese                          | 2.711  | Notepaper             | 2.021  |
#| Cherry              | 8.052  | Pliers                          | 14.082 | CD                    | 0.706  |
#| Pasta               | 0.037  | Hammer                          | 7.756  | Cue                   | 0.003  |
#| Avocado             | 21.171 | Hami melon                      | 1.237  | Flask                 | 0.379  |
#| Mushroom            | 16.569 | Screwdriver                     | 9.551  | Soap                  | 13.372 |
#| Recorder            | 0.037  | Bear                            | 34.474 | Eggplant              | 15.011 |
#| Board Eraser        | 1.789  | Coconut                         | 12.738 | Tape Measure/ Ruler   | 7.274  |
#| Pig                 | 16.956 | Showerhead                      | 16.775 | Globe                 | 29.973 |
#| Chips               | 0.151  | Steak                           | 18.197 | Crosswalk Sign        | 1.799  |
#| Stapler             | 16.292 | Camel                           | 10.170 | Formula 1             | 0.449  |
#| Pomegranate         | 0.242  | Dishwasher                      | 33.117 | Crab                  | 4.746  |
#| Hoverboard          | 0.061  | Meatball                        | 20.725 | Rice Cooker           | 17.460 |
#| Tuba                | 6.387  | Calculator                      | 37.017 | Papaya                | 5.388  |
#| Antelope            | 7.280  | Parrot                          | 26.440 | Seal                  | 0.805  |
#| Butterfly           | 37.030 | Dumbbell                        | 1.262  | Donkey                | 5.162  |
#| Lion                | 3.392  | Urinal                          | 60.392 | Dolphin               | 10.037 |
#| Electric Drill      | 11.564 | Hair Dryer                      | 10.513 | Egg tart              | 2.310  |
#| Jellyfish           | 0.090  | Treadmill                       | 1.761  | Lighter               | 1.128  |
#| Grapefruit          | 0.066  | Game board                      | 10.172 | Mop                   | 3.262  |
#| Radish              | 0.109  | Baozi                           | 0.135  | Target                | 0.049  |
#| French              | 0.000  | Spring Rolls                    | 10.354 | Monkey                | 22.395 |
#| Rabbit              | 11.792 | Pencil Case                     | 3.752  | Yak                   | 13.551 |
#| Red Cabbage         | 1.227  | Binoculars                      | 3.135  | Asparagus             | 7.134  |
#| Barbell             | 0.433  | Scallop                         | 2.124  | Noddles               | 0.182  |
#| Comb                | 4.979  | Dumpling                        | 0.078  | Oyster                | 13.399 |
#| Table Tennis paddle | 0.124  | Cosmetics Brush/Eyeliner Pencil | 1.769  | Chainsaw              | 0.453  |
#| Eraser              | 1.516  | Lobster                         | 2.022  | Durian                | 0.483  |
#| Okra                | 0.082  | Lipstick                        | 1.032  | Cosmetics Mirror      | 0.561  |
#| Curling             | 0.425  | Table Tennis                    | 0.004  |                       |        |
#[04/29 21:25:08 detectron2]: Evaluation results for objects365_v2_val in csv format:
#[04/29 21:25:08 d2.evaluation.testing]: copypaste: Task: bbox
#[04/29 21:25:08 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
#[04/29 21:25:08 d2.evaluation.testing]: copypaste: 19.3887,26.6349,20.9875,8.2505,19.4900,28.4980