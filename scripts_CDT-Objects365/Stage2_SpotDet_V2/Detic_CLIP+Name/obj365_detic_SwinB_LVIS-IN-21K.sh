#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_IN-21k_SpotDet_Noun.out


CFG_PATH="./configs_detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

CLASSIFIER_NAME="o365_clip_a+cnamefix.npy"

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('objects365_v2_val_spotdet_v2_clip_noun',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(365,)" \
        MODEL.MASK_ON False


index created!
[05/13 14:59:07 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/13 15:14:47 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 940.21 seconds.
[05/13 15:15:20 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/13 15:17:41 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 141.59 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.210
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.288
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.226
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.086
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.203
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.162
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.409
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.540
[05/13 15:17:41 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 20.983 | 28.839 | 22.642 | 8.646 | 20.269 | 31.763 |
[05/13 15:17:50 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 15.209 | Sneakers                        | 16.273 | Chair                 | 23.456 |
| Other Shoes         | 0.176  | Hat                             | 32.107 | Car                   | 13.231 |
| Lamp                | 17.263 | Glasses                         | 25.987 | Bottle                | 13.080 |
| Desk                | 16.076 | Cup                             | 0.990  | Street Lights         | 3.999  |
| Cabinet/shelf       | 15.189 | Handbag/Satchel                 | 12.916 | Bracelet              | 4.893  |
| Plate               | 53.933 | Picture/Frame                   | 8.509  | Helmet                | 35.142 |
| Book                | 8.669  | Gloves                          | 20.837 | Storage box           | 8.309  |
| Boat                | 0.990  | Leather Shoes                   | 0.401  | Flower                | 3.971  |
| Bench               | 15.658 | Potted Plant                    | 0.333  | Bowl/Basin            | 37.482 |
| Flag                | 22.983 | Pillow                          | 39.087 | Boots                 | 18.063 |
| Vase                | 16.397 | Microphone                      | 16.540 | Necklace              | 9.976  |
| Ring                | 3.295  | SUV                             | 3.345  | Wine Glass            | 55.310 |
| Belt                | 4.966  | Monitor/TV                      | 55.117 | Backpack              | 22.120 |
| Umbrella            | 25.635 | Traffic Light                   | 33.562 | Speaker               | 30.603 |
| Watch               | 17.243 | Tie                             | 18.804 | Trash bin Can         | 32.239 |
| Slippers            | 1.328  | Bicycle                         | 39.211 | Stool                 | 25.945 |
| Barrel/bucket       | 18.146 | Van                             | 7.650  | Couch                 | 42.754 |
| Sandals             | 5.632  | Basket                          | 23.674 | Drum                  | 12.454 |
| Pen/Pencil          | 16.813 | Bus                             | 35.061 | Wild Bird             | 13.586 |
| High Heels          | 7.929  | Motorcycle                      | 19.867 | Guitar                | 47.753 |
| Carpet              | 27.718 | Cell Phone                      | 34.899 | Bread                 | 11.720 |
| Camera              | 12.254 | Canned                          | 13.587 | Truck                 | 0.919  |
| Traffic cone        | 36.563 | Cymbal                          | 34.897 | Lifesaver             | 3.235  |
| Towel               | 47.040 | Stuffed Toy                     | 27.685 | Candle                | 20.639 |
| Sailboat            | 12.888 | Laptop                          | 68.193 | Awning                | 12.179 |
| Bed                 | 46.208 | Faucet                          | 25.549 | Tent                  | 8.926  |
| Horse               | 42.901 | Mirror                          | 28.298 | Power outlet          | 16.430 |
| Sink                | 34.954 | Apple                           | 25.080 | Air Conditioner       | 10.633 |
| Knife               | 37.679 | Hockey Stick                    | 41.987 | Paddle                | 20.479 |
| Pickup Truck        | 17.523 | Fork                            | 51.177 | Traffic Sign          | 3.052  |
| Ballon              | 1.395  | Tripod                          | 4.382  | Dog                   | 51.625 |
| Spoon               | 37.284 | Clock                           | 46.818 | Pot                   | 26.420 |
| Cow                 | 19.020 | Cake                            | 10.569 | Dining Table          | 13.526 |
| Sheep               | 27.955 | Hanger                          | 3.049  | Blackboard/Whiteboard | 18.472 |
| Napkin              | 21.524 | Other Fish                      | 33.220 | Orange/Tangerine      | 11.177 |
| Toiletry            | 16.553 | Keyboard                        | 54.738 | Tomato                | 28.545 |
| Lantern             | 30.451 | Machinery Vehicle               | 9.503  | Fan                   | 1.465  |
| Green Vegetables    | 0.199  | Banana                          | 9.406  | Baseball Glove        | 46.442 |
| Airplane            | 59.535 | Mouse                           | 51.024 | Train                 | 39.393 |
| Pumpkin             | 54.043 | Soccer                          | 11.306 | Skiboard              | 6.215  |
| Luggage             | 28.945 | Nightstand                      | 26.468 | Teapot                | 17.709 |
| Telephone           | 10.096 | Trolley                         | 16.426 | Head Phone            | 28.070 |
| Sports Car          | 51.644 | Stop Sign                       | 26.002 | Dessert               | 4.106  |
| Scooter             | 15.650 | Stroller                        | 32.979 | Crane                 | 5.370  |
| Remote              | 40.594 | Refrigerator                    | 65.910 | Oven                  | 26.875 |
| Lemon               | 31.262 | Duck                            | 42.470 | Baseball Bat          | 47.728 |
| Surveillance Camera | 1.035  | Cat                             | 62.957 | Jug                   | 5.041  |
| Broccoli            | 44.315 | Piano                           | 17.634 | Pizza                 | 50.458 |
| Elephant            | 70.411 | Skateboard                      | 14.325 | Surfboard             | 47.818 |
| Gun                 | 20.066 | Skating and Skiing shoes        | 20.341 | Gas stove             | 15.441 |
| Donut               | 53.177 | Bow Tie                         | 19.667 | Carrot                | 31.148 |
| Toilet              | 73.327 | Kite                            | 47.812 | Strawberry            | 37.230 |
| Other Balls         | 9.971  | Shovel                          | 5.562  | Pepper                | 24.238 |
| Computer Box        | 6.198  | Toilet Paper                    | 34.243 | Cleaning Products     | 12.439 |
| Chopsticks          | 27.681 | Microwave                       | 60.969 | Pigeon                | 43.000 |
| Baseball            | 38.275 | Cutting/chopping Board          | 35.505 | Coffee Table          | 17.097 |
| Side Table          | 3.478  | Scissors                        | 36.551 | Marker                | 10.832 |
| Pie                 | 0.660  | Ladder                          | 22.342 | Snowboard             | 41.502 |
| Cookies             | 16.193 | Radiator                        | 34.832 | Fire Hydrant          | 36.061 |
| Basketball          | 27.446 | Zebra                           | 64.504 | Grape                 | 1.059  |
| Giraffe             | 67.413 | Potato                          | 14.493 | Sausage               | 18.772 |
| Tricycle            | 5.252  | Violin                          | 9.974  | Egg                   | 54.927 |
| Fire Extinguisher   | 32.222 | Candy                           | 2.372  | Fire Truck            | 38.460 |
| Billards            | 28.203 | Converter                       | 0.090  | Bathtub               | 52.595 |
| Wheelchair          | 42.245 | Golf Club                       | 33.622 | Briefcase             | 7.150  |
| Cucumber            | 22.004 | Cigar/Cigarette                 | 10.502 | Paint Brush           | 3.736  |
| Pear                | 10.351 | Heavy Truck                     | 12.527 | Hamburger             | 20.280 |
| Extractor           | 0.023  | Extension Cord                  | 1.312  | Tong                  | 0.026  |
| Tennis Racket       | 56.684 | Folder                          | 5.285  | American Football     | 13.116 |
| earphone            | 0.892  | Mask                            | 15.970 | Kettle                | 2.267  |
| Tennis              | 0.000  | Ship                            | 35.528 | Swing                 | 0.704  |
| Coffee Machine      | 35.475 | Slide                           | 30.308 | Carriage              | 6.286  |
| Onion               | 14.549 | Green beans                     | 4.695  | Projector             | 21.962 |
| Frisbee             | 60.426 | Washing Machine/Drying Machine  | 30.397 | Chicken               | 49.607 |
| Printer             | 46.371 | Watermelon                      | 33.879 | Saxophone             | 32.891 |
| Tissue              | 0.062  | Toothbrush                      | 34.383 | Ice cream             | 6.781  |
| Hot air balloon     | 48.423 | Cello                           | 19.715 | French Fries          | 0.098  |
| Scale               | 5.843  | Trophy                          | 27.595 | Cabbage               | 7.323  |
| Hot dog             | 2.133  | Blender                         | 38.237 | Peach                 | 18.964 |
| Rice                | 4.731  | Wallet/Purse                    | 25.763 | Volleyball            | 42.432 |
| Deer                | 45.376 | Goose                           | 17.772 | Tape                  | 14.103 |
| Tablet              | 15.199 | Cosmetics                       | 6.380  | Trumpet               | 13.744 |
| Pineapple           | 17.697 | Golf Ball                       | 21.504 | Ambulance             | 63.674 |
| Parking meter       | 29.180 | Mango                           | 0.688  | Key                   | 10.756 |
| Hurdle              | 0.090  | Fishing Rod                     | 22.827 | Medal                 | 7.148  |
| Flute               | 23.164 | Brush                           | 0.629  | Penguin               | 57.144 |
| Megaphone           | 5.749  | Corn                            | 12.601 | Lettuce               | 2.372  |
| Garlic              | 13.859 | Swan                            | 44.487 | Helicopter            | 39.772 |
| Green Onion         | 2.729  | Sandwich                        | 0.000  | Nuts                  | 6.408  |
| Speed Limit Sign    | 9.565  | Induction Cooker                | 2.396  | Broom                 | 11.832 |
| Trombone            | 7.397  | Plum                            | 0.691  | Rickshaw              | 1.079  |
| Goldfish            | 15.882 | Kiwi fruit                      | 21.554 | Router/modem          | 8.238  |
| Poker Card          | 22.117 | Toaster                         | 38.528 | Shrimp                | 21.531 |
| Sushi               | 42.563 | Cheese                          | 12.633 | Notepaper             | 2.857  |
| Cherry              | 11.073 | Pliers                          | 14.442 | CD                    | 10.598 |
| Pasta               | 1.388  | Hammer                          | 6.968  | Cue                   | 5.408  |
| Avocado             | 25.548 | Hami melon                      | 1.090  | Flask                 | 1.846  |
| Mushroom            | 17.396 | Screwdriver                     | 8.877  | Soap                  | 12.378 |
| Recorder            | 0.092  | Bear                            | 50.406 | Eggplant              | 14.284 |
| Board Eraser        | 2.409  | Coconut                         | 32.182 | Tape Measure/ Ruler   | 7.594  |
| Pig                 | 40.906 | Showerhead                      | 16.820 | Globe                 | 32.368 |
| Chips               | 0.593  | Steak                           | 23.795 | Crosswalk Sign        | 2.997  |
| Stapler             | 11.778 | Camel                           | 52.274 | Formula 1             | 9.892  |
| Pomegranate         | 2.737  | Dishwasher                      | 27.957 | Crab                  | 11.253 |
| Hoverboard          | 6.956  | Meatball                        | 28.818 | Rice Cooker           | 10.036 |
| Tuba                | 12.453 | Calculator                      | 31.534 | Papaya                | 3.191  |
| Antelope            | 12.060 | Parrot                          | 35.823 | Seal                  | 45.498 |
| Butterfly           | 41.596 | Dumbbell                        | 4.885  | Donkey                | 45.563 |
| Lion                | 17.855 | Urinal                          | 60.948 | Dolphin               | 25.983 |
| Electric Drill      | 12.997 | Hair Dryer                      | 11.787 | Egg tart              | 4.340  |
| Jellyfish           | 32.969 | Treadmill                       | 23.549 | Lighter               | 3.710  |
| Grapefruit          | 0.091  | Game board                      | 21.467 | Mop                   | 1.445  |
| Radish              | 0.127  | Baozi                           | 5.806  | Target                | 0.184  |
| French              | 0.000  | Spring Rolls                    | 17.832 | Monkey                | 41.821 |
| Rabbit              | 24.451 | Pencil Case                     | 2.974  | Yak                   | 6.350  |
| Red Cabbage         | 7.284  | Binoculars                      | 7.207  | Asparagus             | 5.892  |
| Barbell             | 1.722  | Scallop                         | 9.525  | Noddles               | 1.260  |
| Comb                | 8.054  | Dumpling                        | 6.903  | Oyster                | 40.778 |
| Table Tennis paddle | 5.169  | Cosmetics Brush/Eyeliner Pencil | 6.348  | Chainsaw              | 3.439  |
| Eraser              | 0.987  | Lobster                         | 8.892  | Durian                | 10.003 |
| Okra                | 0.000  | Lipstick                        | 7.738  | Cosmetics Mirror      | 0.111  |
| Curling             | 62.521 | Table Tennis                    | 0.439  |                       |        |
[05/13 15:19:28 detectron2]: Evaluation results for objects365_v2_val_spotdet_v2_clip_noun in csv format:
[05/13 15:19:28 d2.evaluation.testing]: copypaste: Task: bbox
[05/13 15:19:28 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/13 15:19:28 d2.evaluation.testing]: copypaste: 20.9831,28.8388,22.6422,8.6464,20.2686,31.7628