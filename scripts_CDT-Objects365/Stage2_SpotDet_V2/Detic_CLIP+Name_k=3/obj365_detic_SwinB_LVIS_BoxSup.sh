#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_BoxSup_SpotDet_Noun_k=3.out


CFG_PATH="./configs_detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml"
MODEL_PATH="./models/Detic_CDT/BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth"

CLASSIFIER_NAME="o365_clip_a+cnamefix.npy"

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('objects365_v2_val_spotdet_v2_clip_noun_k=3',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(365,)" \
        MODEL.MASK_ON False


index created!
[05/13 14:49:33 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/13 15:05:12 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 938.07 seconds.
[05/13 15:05:40 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/13 15:08:00 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 140.51 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.200
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.199
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.188
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.279
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.474
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.601
[05/13 15:08:00 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 19.958 | 27.375 | 21.588 | 8.358 | 19.863 | 30.008 |
[05/13 15:08:07 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 15.680 | Sneakers                        | 14.141 | Chair                 | 28.595 |
| Other Shoes         | 0.903  | Hat                             | 34.102 | Car                   | 12.331 |
| Lamp                | 19.547 | Glasses                         | 23.452 | Bottle                | 15.957 |
| Desk                | 20.344 | Cup                             | 17.501 | Street Lights         | 4.655  |
| Cabinet/shelf       | 16.540 | Handbag/Satchel                 | 12.738 | Bracelet              | 6.277  |
| Plate               | 56.456 | Picture/Frame                   | 4.517  | Helmet                | 34.309 |
| Book                | 8.651  | Gloves                          | 22.023 | Storage box           | 9.938  |
| Boat                | 22.582 | Leather Shoes                   | 1.141  | Flower                | 2.231  |
| Bench               | 16.422 | Potted Plant                    | 0.888  | Bowl/Basin            | 37.719 |
| Flag                | 23.097 | Pillow                          | 42.239 | Boots                 | 18.615 |
| Vase                | 16.086 | Microphone                      | 17.473 | Necklace              | 11.272 |
| Ring                | 3.730  | SUV                             | 6.025  | Wine Glass            | 55.975 |
| Belt                | 8.078  | Monitor/TV                      | 42.007 | Backpack              | 23.422 |
| Umbrella            | 24.901 | Traffic Light                   | 32.188 | Speaker               | 30.903 |
| Watch               | 31.687 | Tie                             | 14.589 | Trash bin Can         | 37.851 |
| Slippers            | 9.981  | Bicycle                         | 41.400 | Stool                 | 31.604 |
| Barrel/bucket       | 22.581 | Van                             | 5.538  | Couch                 | 43.607 |
| Sandals             | 7.795  | Basket                          | 27.081 | Drum                  | 30.091 |
| Pen/Pencil          | 19.215 | Bus                             | 31.696 | Wild Bird             | 13.333 |
| High Heels          | 5.701  | Motorcycle                      | 24.838 | Guitar                | 44.054 |
| Carpet              | 29.759 | Cell Phone                      | 34.417 | Bread                 | 13.438 |
| Camera              | 26.145 | Canned                          | 14.902 | Truck                 | 11.053 |
| Traffic cone        | 37.180 | Cymbal                          | 37.748 | Lifesaver             | 2.733  |
| Towel               | 48.315 | Stuffed Toy                     | 22.440 | Candle                | 20.718 |
| Sailboat            | 8.109  | Laptop                          | 69.175 | Awning                | 12.492 |
| Bed                 | 48.031 | Faucet                          | 29.492 | Tent                  | 4.128  |
| Horse               | 38.227 | Mirror                          | 35.035 | Power outlet          | 9.294  |
| Sink                | 35.068 | Apple                           | 24.424 | Air Conditioner       | 13.909 |
| Knife               | 41.544 | Hockey Stick                    | 30.015 | Paddle                | 22.576 |
| Pickup Truck        | 27.066 | Fork                            | 54.177 | Traffic Sign          | 2.423  |
| Ballon              | 27.698 | Tripod                          | 5.998  | Dog                   | 50.340 |
| Spoon               | 40.667 | Clock                           | 45.200 | Pot                   | 30.196 |
| Cow                 | 16.625 | Cake                            | 13.269 | Dining Table          | 15.015 |
| Sheep               | 25.692 | Hanger                          | 2.069  | Blackboard/Whiteboard | 19.741 |
| Napkin              | 23.121 | Other Fish                      | 29.311 | Orange/Tangerine      | 4.633  |
| Toiletry            | 19.972 | Keyboard                        | 55.074 | Tomato                | 46.696 |
| Lantern             | 32.718 | Machinery Vehicle               | 6.812  | Fan                   | 16.705 |
| Green Vegetables    | 0.223  | Banana                          | 9.219  | Baseball Glove        | 40.295 |
| Airplane            | 60.781 | Mouse                           | 47.670 | Train                 | 23.669 |
| Pumpkin             | 53.896 | Soccer                          | 12.111 | Skiboard              | 1.689  |
| Luggage             | 20.725 | Nightstand                      | 19.184 | Teapot                | 20.867 |
| Telephone           | 21.279 | Trolley                         | 16.168 | Head Phone            | 28.198 |
| Sports Car          | 28.485 | Stop Sign                       | 31.113 | Dessert               | 7.388  |
| Scooter             | 21.621 | Stroller                        | 26.398 | Crane                 | 1.343  |
| Remote              | 34.324 | Refrigerator                    | 67.450 | Oven                  | 28.033 |
| Lemon               | 29.067 | Duck                            | 42.743 | Baseball Bat          | 38.846 |
| Surveillance Camera | 1.915  | Cat                             | 62.096 | Jug                   | 4.512  |
| Broccoli            | 45.828 | Piano                           | 25.439 | Pizza                 | 53.589 |
| Elephant            | 71.243 | Skateboard                      | 14.755 | Surfboard             | 47.187 |
| Gun                 | 13.852 | Skating and Skiing shoes        | 27.013 | Gas stove             | 16.452 |
| Donut               | 53.156 | Bow Tie                         | 22.632 | Carrot                | 29.829 |
| Toilet              | 73.120 | Kite                            | 45.796 | Strawberry            | 39.031 |
| Other Balls         | 8.632  | Shovel                          | 6.741  | Pepper                | 18.733 |
| Computer Box        | 1.950  | Toilet Paper                    | 30.383 | Cleaning Products     | 14.058 |
| Chopsticks          | 29.228 | Microwave                       | 62.811 | Pigeon                | 43.065 |
| Baseball            | 28.716 | Cutting/chopping Board          | 37.316 | Coffee Table          | 17.467 |
| Side Table          | 3.644  | Scissors                        | 38.257 | Marker                | 11.832 |
| Pie                 | 1.020  | Ladder                          | 22.447 | Snowboard             | 46.294 |
| Cookies             | 13.937 | Radiator                        | 36.737 | Fire Hydrant          | 36.343 |
| Basketball          | 24.645 | Zebra                           | 64.897 | Grape                 | 1.336  |
| Giraffe             | 67.617 | Potato                          | 17.708 | Sausage               | 21.287 |
| Tricycle            | 6.515  | Violin                          | 4.952  | Egg                   | 61.149 |
| Fire Extinguisher   | 34.026 | Candy                           | 1.475  | Fire Truck            | 39.319 |
| Billards            | 23.811 | Converter                       | 0.020  | Bathtub               | 54.912 |
| Wheelchair          | 41.343 | Golf Club                       | 35.243 | Briefcase             | 8.339  |
| Cucumber            | 25.245 | Cigar/Cigarette                 | 12.145 | Paint Brush           | 2.199  |
| Pear                | 12.882 | Heavy Truck                     | 12.304 | Hamburger             | 17.839 |
| Extractor           | 0.181  | Extension Cord                  | 0.532  | Tong                  | 0.118  |
| Tennis Racket       | 54.477 | Folder                          | 3.627  | American Football     | 11.192 |
| earphone            | 0.970  | Mask                            | 16.803 | Kettle                | 22.015 |
| Tennis              | 20.047 | Ship                            | 18.821 | Swing                 | 0.435  |
| Coffee Machine      | 44.054 | Slide                           | 30.613 | Carriage              | 6.225  |
| Onion               | 18.405 | Green beans                     | 4.696  | Projector             | 23.825 |
| Frisbee             | 58.882 | Washing Machine/Drying Machine  | 30.203 | Chicken               | 43.470 |
| Printer             | 48.212 | Watermelon                      | 35.207 | Saxophone             | 28.428 |
| Tissue              | 0.346  | Toothbrush                      | 35.321 | Ice cream             | 4.166  |
| Hot air balloon     | 48.506 | Cello                           | 5.565  | French Fries          | 0.025  |
| Scale               | 5.195  | Trophy                          | 26.540 | Cabbage               | 7.073  |
| Hot dog             | 0.916  | Blender                         | 39.368 | Peach                 | 21.136 |
| Rice                | 0.138  | Wallet/Purse                    | 26.387 | Volleyball            | 28.374 |
| Deer                | 40.177 | Goose                           | 16.493 | Tape                  | 13.896 |
| Tablet              | 3.518  | Cosmetics                       | 1.321  | Trumpet               | 9.714  |
| Pineapple           | 17.865 | Golf Ball                       | 6.468  | Ambulance             | 55.381 |
| Parking meter       | 29.102 | Mango                           | 0.292  | Key                   | 10.325 |
| Hurdle              | 0.474  | Fishing Rod                     | 17.636 | Medal                 | 4.810  |
| Flute               | 10.138 | Brush                           | 3.340  | Penguin               | 53.200 |
| Megaphone           | 3.794  | Corn                            | 5.383  | Lettuce               | 2.287  |
| Garlic              | 19.741 | Swan                            | 24.705 | Helicopter            | 37.922 |
| Green Onion         | 3.585  | Sandwich                        | 8.878  | Nuts                  | 1.788  |
| Speed Limit Sign    | 11.686 | Induction Cooker                | 3.762  | Broom                 | 11.978 |
| Trombone            | 5.540  | Plum                            | 0.143  | Rickshaw              | 0.938  |
| Goldfish            | 6.281  | Kiwi fruit                      | 22.441 | Router/modem          | 5.063  |
| Poker Card          | 9.909  | Toaster                         | 44.918 | Shrimp                | 17.796 |
| Sushi               | 42.048 | Cheese                          | 5.511  | Notepaper             | 2.295  |
| Cherry              | 8.050  | Pliers                          | 13.406 | CD                    | 1.213  |
| Pasta               | 0.138  | Hammer                          | 6.895  | Cue                   | 0.138  |
| Avocado             | 24.434 | Hami melon                      | 0.305  | Flask                 | 0.295  |
| Mushroom            | 17.129 | Screwdriver                     | 8.772  | Soap                  | 13.076 |
| Recorder            | 0.060  | Bear                            | 42.836 | Eggplant              | 15.614 |
| Board Eraser        | 4.190  | Coconut                         | 23.789 | Tape Measure/ Ruler   | 5.747  |
| Pig                 | 29.487 | Showerhead                      | 17.224 | Globe                 | 32.242 |
| Chips               | 0.215  | Steak                           | 19.184 | Crosswalk Sign        | 1.902  |
| Stapler             | 10.762 | Camel                           | 38.645 | Formula 1             | 1.463  |
| Pomegranate         | 0.656  | Dishwasher                      | 32.529 | Crab                  | 5.277  |
| Hoverboard          | 0.278  | Meatball                        | 22.668 | Rice Cooker           | 10.618 |
| Tuba                | 7.414  | Calculator                      | 33.680 | Papaya                | 2.004  |
| Antelope            | 8.225  | Parrot                          | 32.596 | Seal                  | 4.666  |
| Butterfly           | 40.725 | Dumbbell                        | 1.776  | Donkey                | 10.258 |
| Lion                | 5.693  | Urinal                          | 60.875 | Dolphin               | 10.600 |
| Electric Drill      | 11.415 | Hair Dryer                      | 12.186 | Egg tart              | 4.684  |
| Jellyfish           | 1.420  | Treadmill                       | 7.381  | Lighter               | 0.858  |
| Grapefruit          | 0.117  | Game board                      | 10.942 | Mop                   | 1.701  |
| Radish              | 0.101  | Baozi                           | 1.501  | Target                | 0.085  |
| French              | 0.000  | Spring Rolls                    | 13.635 | Monkey                | 27.517 |
| Rabbit              | 15.175 | Pencil Case                     | 3.354  | Yak                   | 7.864  |
| Red Cabbage         | 2.768  | Binoculars                      | 4.553  | Asparagus             | 9.101  |
| Barbell             | 0.453  | Scallop                         | 4.780  | Noddles               | 0.968  |
| Comb                | 7.109  | Dumpling                        | 0.875  | Oyster                | 16.490 |
| Table Tennis paddle | 0.659  | Cosmetics Brush/Eyeliner Pencil | 2.472  | Chainsaw              | 0.732  |
| Eraser              | 2.144  | Lobster                         | 2.792  | Durian                | 5.300  |
| Okra                | 0.000  | Lipstick                        | 3.048  | Cosmetics Mirror      | 0.591  |
| Curling             | 38.084 | Table Tennis                    | 0.035  |                       |        |
[05/13 15:09:39 detectron2]: Evaluation results for objects365_v2_val_spotdet_v2_clip_noun_k=3 in csv format:
[05/13 15:09:39 d2.evaluation.testing]: copypaste: Task: bbox
[05/13 15:09:39 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/13 15:09:39 d2.evaluation.testing]: copypaste: 19.9578,27.3746,21.5884,8.3578,19.8628,30.0084