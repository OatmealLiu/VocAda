#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_IN-L_SpotDet_Noun_k=3.out


CFG_PATH="./configs_detic/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

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
[05/13 14:49:34 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/13 15:04:43 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 908.43 seconds.
[05/13 15:05:04 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/13 15:07:03 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 119.34 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.293
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.230
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.189
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.281
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.603
[05/13 15:07:03 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 21.298 | 29.273 | 23.007 | 8.865 | 21.103 | 32.238 |
[05/13 15:07:06 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 14.799 | Sneakers                        | 16.774 | Chair                 | 28.564 |
| Other Shoes         | 1.074  | Hat                             | 34.234 | Car                   | 11.744 |
| Lamp                | 19.493 | Glasses                         | 22.688 | Bottle                | 15.505 |
| Desk                | 20.108 | Cup                             | 17.839 | Street Lights         | 4.272  |
| Cabinet/shelf       | 14.645 | Handbag/Satchel                 | 13.429 | Bracelet              | 6.236  |
| Plate               | 56.072 | Picture/Frame                   | 3.786  | Helmet                | 33.800 |
| Book                | 8.573  | Gloves                          | 21.495 | Storage box           | 9.324  |
| Boat                | 22.018 | Leather Shoes                   | 0.462  | Flower                | 2.253  |
| Bench               | 16.573 | Potted Plant                    | 0.265  | Bowl/Basin            | 38.636 |
| Flag                | 24.144 | Pillow                          | 42.453 | Boots                 | 18.505 |
| Vase                | 16.073 | Microphone                      | 18.312 | Necklace              | 11.491 |
| Ring                | 3.820  | SUV                             | 9.499  | Wine Glass            | 56.770 |
| Belt                | 7.754  | Monitor/TV                      | 41.243 | Backpack              | 24.253 |
| Umbrella            | 25.192 | Traffic Light                   | 32.555 | Speaker               | 30.128 |
| Watch               | 31.692 | Tie                             | 16.650 | Trash bin Can         | 37.111 |
| Slippers            | 10.084 | Bicycle                         | 40.780 | Stool                 | 31.871 |
| Barrel/bucket       | 22.870 | Van                             | 6.615  | Couch                 | 41.191 |
| Sandals             | 8.542  | Basket                          | 26.834 | Drum                  | 31.532 |
| Pen/Pencil          | 20.518 | Bus                             | 31.435 | Wild Bird             | 13.631 |
| High Heels          | 5.206  | Motorcycle                      | 25.186 | Guitar                | 48.455 |
| Carpet              | 31.308 | Cell Phone                      | 33.557 | Bread                 | 13.526 |
| Camera              | 26.071 | Canned                          | 13.473 | Truck                 | 11.487 |
| Traffic cone        | 37.475 | Cymbal                          | 37.828 | Lifesaver             | 1.968  |
| Towel               | 48.192 | Stuffed Toy                     | 23.056 | Candle                | 21.517 |
| Sailboat            | 8.648  | Laptop                          | 69.186 | Awning                | 12.626 |
| Bed                 | 48.795 | Faucet                          | 29.395 | Tent                  | 4.687  |
| Horse               | 42.304 | Mirror                          | 34.247 | Power outlet          | 10.660 |
| Sink                | 35.274 | Apple                           | 25.893 | Air Conditioner       | 14.888 |
| Knife               | 41.017 | Hockey Stick                    | 39.588 | Paddle                | 23.970 |
| Pickup Truck        | 27.757 | Fork                            | 54.033 | Traffic Sign          | 2.041  |
| Ballon              | 25.448 | Tripod                          | 5.274  | Dog                   | 49.606 |
| Spoon               | 40.742 | Clock                           | 43.115 | Pot                   | 30.891 |
| Cow                 | 17.910 | Cake                            | 13.098 | Dining Table          | 14.689 |
| Sheep               | 25.898 | Hanger                          | 2.399  | Blackboard/Whiteboard | 18.640 |
| Napkin              | 22.223 | Other Fish                      | 34.550 | Orange/Tangerine      | 5.991  |
| Toiletry            | 19.203 | Keyboard                        | 53.137 | Tomato                | 47.414 |
| Lantern             | 32.091 | Machinery Vehicle               | 4.693  | Fan                   | 16.475 |
| Green Vegetables    | 0.147  | Banana                          | 9.205  | Baseball Glove        | 42.367 |
| Airplane            | 59.720 | Mouse                           | 49.391 | Train                 | 23.458 |
| Pumpkin             | 53.355 | Soccer                          | 13.636 | Skiboard              | 1.684  |
| Luggage             | 19.375 | Nightstand                      | 20.142 | Teapot                | 22.134 |
| Telephone           | 22.301 | Trolley                         | 12.250 | Head Phone            | 29.510 |
| Sports Car          | 38.821 | Stop Sign                       | 30.623 | Dessert               | 8.086  |
| Scooter             | 22.733 | Stroller                        | 31.040 | Crane                 | 1.337  |
| Remote              | 35.892 | Refrigerator                    | 66.844 | Oven                  | 26.991 |
| Lemon               | 30.285 | Duck                            | 42.899 | Baseball Bat          | 42.734 |
| Surveillance Camera | 1.884  | Cat                             | 62.477 | Jug                   | 5.579  |
| Broccoli            | 45.340 | Piano                           | 23.724 | Pizza                 | 53.280 |
| Elephant            | 71.172 | Skateboard                      | 18.962 | Surfboard             | 46.943 |
| Gun                 | 20.305 | Skating and Skiing shoes        | 18.176 | Gas stove             | 16.436 |
| Donut               | 51.047 | Bow Tie                         | 23.560 | Carrot                | 31.267 |
| Toilet              | 73.174 | Kite                            | 48.763 | Strawberry            | 39.872 |
| Other Balls         | 10.049 | Shovel                          | 8.600  | Pepper                | 9.529  |
| Computer Box        | 2.442  | Toilet Paper                    | 30.583 | Cleaning Products     | 12.924 |
| Chopsticks          | 29.759 | Microwave                       | 62.297 | Pigeon                | 43.460 |
| Baseball            | 29.884 | Cutting/chopping Board          | 36.664 | Coffee Table          | 20.779 |
| Side Table          | 3.716  | Scissors                        | 38.029 | Marker                | 13.696 |
| Pie                 | 0.945  | Ladder                          | 21.764 | Snowboard             | 47.395 |
| Cookies             | 13.821 | Radiator                        | 36.932 | Fire Hydrant          | 36.232 |
| Basketball          | 29.783 | Zebra                           | 64.862 | Grape                 | 2.225  |
| Giraffe             | 67.274 | Potato                          | 19.458 | Sausage               | 19.724 |
| Tricycle            | 9.090  | Violin                          | 19.324 | Egg                   | 61.195 |
| Fire Extinguisher   | 35.838 | Candy                           | 0.533  | Fire Truck            | 41.932 |
| Billards            | 30.743 | Converter                       | 0.026  | Bathtub               | 55.511 |
| Wheelchair          | 49.221 | Golf Club                       | 34.859 | Briefcase             | 9.839  |
| Cucumber            | 27.553 | Cigar/Cigarette                 | 15.236 | Paint Brush           | 3.602  |
| Pear                | 15.017 | Heavy Truck                     | 10.679 | Hamburger             | 18.793 |
| Extractor           | 0.268  | Extension Cord                  | 0.546  | Tong                  | 0.065  |
| Tennis Racket       | 53.475 | Folder                          | 4.045  | American Football     | 15.725 |
| earphone            | 1.068  | Mask                            | 12.080 | Kettle                | 25.148 |
| Tennis              | 19.462 | Ship                            | 21.290 | Swing                 | 0.194  |
| Coffee Machine      | 44.177 | Slide                           | 31.653 | Carriage              | 5.967  |
| Onion               | 21.418 | Green beans                     | 6.053  | Projector             | 26.282 |
| Frisbee             | 62.336 | Washing Machine/Drying Machine  | 33.052 | Chicken               | 46.432 |
| Printer             | 48.550 | Watermelon                      | 39.215 | Saxophone             | 37.835 |
| Tissue              | 0.283  | Toothbrush                      | 35.680 | Ice cream             | 7.109  |
| Hot air balloon     | 68.246 | Cello                           | 8.464  | French Fries          | 0.031  |
| Scale               | 8.118  | Trophy                          | 23.787 | Cabbage               | 5.213  |
| Hot dog             | 0.952  | Blender                         | 39.516 | Peach                 | 28.461 |
| Rice                | 0.215  | Wallet/Purse                    | 28.560 | Volleyball            | 48.037 |
| Deer                | 46.240 | Goose                           | 18.361 | Tape                  | 15.400 |
| Tablet              | 5.517  | Cosmetics                       | 2.561  | Trumpet               | 8.692  |
| Pineapple           | 18.566 | Golf Ball                       | 14.934 | Ambulance             | 66.727 |
| Parking meter       | 30.166 | Mango                           | 0.456  | Key                   | 12.933 |
| Hurdle              | 0.454  | Fishing Rod                     | 23.291 | Medal                 | 4.635  |
| Flute               | 17.576 | Brush                           | 5.216  | Penguin               | 57.815 |
| Megaphone           | 6.880  | Corn                            | 5.622  | Lettuce               | 2.311  |
| Garlic              | 28.095 | Swan                            | 26.717 | Helicopter            | 40.341 |
| Green Onion         | 3.863  | Sandwich                        | 9.223  | Nuts                  | 2.892  |
| Speed Limit Sign    | 11.736 | Induction Cooker                | 3.498  | Broom                 | 12.789 |
| Trombone            | 8.709  | Plum                            | 0.289  | Rickshaw              | 1.423  |
| Goldfish            | 15.678 | Kiwi fruit                      | 22.887 | Router/modem          | 11.345 |
| Poker Card          | 9.542  | Toaster                         | 48.314 | Shrimp                | 17.128 |
| Sushi               | 45.813 | Cheese                          | 6.395  | Notepaper             | 3.071  |
| Cherry              | 11.156 | Pliers                          | 18.650 | CD                    | 2.419  |
| Pasta               | 0.102  | Hammer                          | 11.451 | Cue                   | 0.116  |
| Avocado             | 30.584 | Hami melon                      | 0.369  | Flask                 | 0.564  |
| Mushroom            | 19.144 | Screwdriver                     | 13.477 | Soap                  | 14.739 |
| Recorder            | 0.043  | Bear                            | 45.455 | Eggplant              | 18.729 |
| Board Eraser        | 2.549  | Coconut                         | 28.841 | Tape Measure/ Ruler   | 7.536  |
| Pig                 | 41.644 | Showerhead                      | 17.358 | Globe                 | 32.389 |
| Chips               | 0.160  | Steak                           | 20.047 | Crosswalk Sign        | 1.969  |
| Stapler             | 15.497 | Camel                           | 55.483 | Formula 1             | 9.290  |
| Pomegranate         | 1.442  | Dishwasher                      | 34.318 | Crab                  | 6.659  |
| Hoverboard          | 0.310  | Meatball                        | 28.008 | Rice Cooker           | 11.307 |
| Tuba                | 10.347 | Calculator                      | 36.279 | Papaya                | 7.207  |
| Antelope            | 6.833  | Parrot                          | 31.987 | Seal                  | 9.800  |
| Butterfly           | 42.301 | Dumbbell                        | 5.903  | Donkey                | 10.625 |
| Lion                | 18.485 | Urinal                          | 59.554 | Dolphin               | 28.848 |
| Electric Drill      | 15.661 | Hair Dryer                      | 14.272 | Egg tart              | 4.037  |
| Jellyfish           | 5.153  | Treadmill                       | 11.694 | Lighter               | 3.015  |
| Grapefruit          | 0.145  | Game board                      | 13.924 | Mop                   | 5.368  |
| Radish              | 0.109  | Baozi                           | 1.025  | Target                | 0.280  |
| French              | 0.000  | Spring Rolls                    | 22.111 | Monkey                | 33.311 |
| Rabbit              | 24.797 | Pencil Case                     | 8.782  | Yak                   | 7.241  |
| Red Cabbage         | 1.217  | Binoculars                      | 10.600 | Asparagus             | 13.076 |
| Barbell             | 4.053  | Scallop                         | 9.489  | Noddles               | 0.638  |
| Comb                | 6.417  | Dumpling                        | 0.894  | Oyster                | 19.640 |
| Table Tennis paddle | 3.971  | Cosmetics Brush/Eyeliner Pencil | 3.057  | Chainsaw              | 0.531  |
| Eraser              | 4.277  | Lobster                         | 1.615  | Durian                | 5.826  |
| Okra                | 0.000  | Lipstick                        | 3.679  | Cosmetics Mirror      | 0.147  |
| Curling             | 14.822 | Table Tennis                    | 0.103  |                       |        |
[05/13 15:08:40 detectron2]: Evaluation results for objects365_v2_val_spotdet_v2_clip_noun_k=3 in csv format:
[05/13 15:08:40 d2.evaluation.testing]: copypaste: Task: bbox
[05/13 15:08:40 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/13 15:08:40 d2.evaluation.testing]: copypaste: 21.2981,29.2728,23.0069,8.8647,21.1033,32.2381