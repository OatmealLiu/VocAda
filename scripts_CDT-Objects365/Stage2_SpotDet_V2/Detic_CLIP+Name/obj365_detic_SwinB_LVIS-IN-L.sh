#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_IN-L_SpotDet_Noun.out


CFG_PATH="./configs_detic/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

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
[05/13 14:57:12 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/13 15:12:12 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 900.13 seconds.
[05/13 15:12:35 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/13 15:14:18 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 102.81 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.206
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.283
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.222
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.085
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.201
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.161
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.536
[05/13 15:14:18 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 20.612 | 28.326 | 22.238 | 8.514 | 20.109 | 31.624 |
[05/13 15:14:22 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 14.928 | Sneakers                        | 14.339 | Chair                 | 25.186 |
| Other Shoes         | 0.231  | Hat                             | 32.246 | Car                   | 10.973 |
| Lamp                | 17.204 | Glasses                         | 22.754 | Bottle                | 12.552 |
| Desk                | 15.910 | Cup                             | 0.990  | Street Lights         | 3.809  |
| Cabinet/shelf       | 14.185 | Handbag/Satchel                 | 12.954 | Bracelet              | 4.538  |
| Plate               | 55.027 | Picture/Frame                   | 3.913  | Helmet                | 34.759 |
| Book                | 8.581  | Gloves                          | 19.823 | Storage box           | 7.799  |
| Boat                | 0.990  | Leather Shoes                   | 0.404  | Flower                | 2.187  |
| Bench               | 16.740 | Potted Plant                    | 0.273  | Bowl/Basin            | 38.365 |
| Flag                | 22.803 | Pillow                          | 39.700 | Boots                 | 18.279 |
| Vase                | 16.095 | Microphone                      | 18.340 | Necklace              | 10.113 |
| Ring                | 3.322  | SUV                             | 4.605  | Wine Glass            | 55.895 |
| Belt                | 4.718  | Monitor/TV                      | 43.677 | Backpack              | 23.173 |
| Umbrella            | 26.379 | Traffic Light                   | 33.148 | Speaker               | 29.005 |
| Watch               | 17.554 | Tie                             | 13.653 | Trash bin Can         | 33.684 |
| Slippers            | 1.387  | Bicycle                         | 40.883 | Stool                 | 26.174 |
| Barrel/bucket       | 18.712 | Van                             | 6.746  | Couch                 | 40.813 |
| Sandals             | 6.386  | Basket                          | 22.933 | Drum                  | 12.568 |
| Pen/Pencil          | 17.695 | Bus                             | 32.244 | Wild Bird             | 13.732 |
| High Heels          | 7.421  | Motorcycle                      | 22.241 | Guitar                | 49.377 |
| Carpet              | 31.549 | Cell Phone                      | 33.165 | Bread                 | 11.763 |
| Camera              | 12.740 | Canned                          | 12.601 | Truck                 | 0.990  |
| Traffic cone        | 36.590 | Cymbal                          | 37.793 | Lifesaver             | 1.829  |
| Towel               | 47.457 | Stuffed Toy                     | 23.174 | Candle                | 20.809 |
| Sailboat            | 9.500  | Laptop                          | 68.929 | Awning                | 12.719 |
| Bed                 | 48.354 | Faucet                          | 26.114 | Tent                  | 6.609  |
| Horse               | 43.887 | Mirror                          | 27.927 | Power outlet          | 10.846 |
| Sink                | 35.046 | Apple                           | 25.996 | Air Conditioner       | 11.636 |
| Knife               | 39.266 | Hockey Stick                    | 41.281 | Paddle                | 22.184 |
| Pickup Truck        | 19.277 | Fork                            | 52.426 | Traffic Sign          | 2.107  |
| Ballon              | 0.630  | Tripod                          | 4.383  | Dog                   | 51.159 |
| Spoon               | 39.072 | Clock                           | 44.223 | Pot                   | 27.515 |
| Cow                 | 18.974 | Cake                            | 11.111 | Dining Table          | 14.872 |
| Sheep               | 28.381 | Hanger                          | 2.364  | Blackboard/Whiteboard | 18.825 |
| Napkin              | 21.971 | Other Fish                      | 35.028 | Orange/Tangerine      | 6.620  |
| Toiletry            | 14.031 | Keyboard                        | 53.386 | Tomato                | 29.492 |
| Lantern             | 30.380 | Machinery Vehicle               | 4.704  | Fan                   | 1.561  |
| Green Vegetables    | 0.151  | Banana                          | 9.086  | Baseball Glove        | 45.647 |
| Airplane            | 59.911 | Mouse                           | 49.103 | Train                 | 25.155 |
| Pumpkin             | 53.452 | Soccer                          | 13.150 | Skiboard              | 4.207  |
| Luggage             | 25.132 | Nightstand                      | 25.300 | Teapot                | 20.245 |
| Telephone           | 10.873 | Trolley                         | 12.820 | Head Phone            | 27.767 |
| Sports Car          | 39.887 | Stop Sign                       | 25.187 | Dessert               | 3.503  |
| Scooter             | 16.414 | Stroller                        | 33.842 | Crane                 | 1.417  |
| Remote              | 35.712 | Refrigerator                    | 66.419 | Oven                  | 26.800 |
| Lemon               | 29.614 | Duck                            | 41.772 | Baseball Bat          | 48.376 |
| Surveillance Camera | 1.362  | Cat                             | 63.285 | Jug                   | 4.213  |
| Broccoli            | 44.072 | Piano                           | 17.872 | Pizza                 | 51.877 |
| Elephant            | 71.183 | Skateboard                      | 19.247 | Surfboard             | 47.582 |
| Gun                 | 21.637 | Skating and Skiing shoes        | 17.699 | Gas stove             | 16.367 |
| Donut               | 51.374 | Bow Tie                         | 19.729 | Carrot                | 31.654 |
| Toilet              | 73.258 | Kite                            | 49.429 | Strawberry            | 39.730 |
| Other Balls         | 9.889  | Shovel                          | 6.599  | Pepper                | 9.228  |
| Computer Box        | 3.799  | Toilet Paper                    | 29.254 | Cleaning Products     | 11.588 |
| Chopsticks          | 28.943 | Microwave                       | 58.598 | Pigeon                | 43.502 |
| Baseball            | 38.580 | Cutting/chopping Board          | 35.790 | Coffee Table          | 21.947 |
| Side Table          | 4.243  | Scissors                        | 37.412 | Marker                | 12.586 |
| Pie                 | 0.990  | Ladder                          | 21.586 | Snowboard             | 44.741 |
| Cookies             | 13.826 | Radiator                        | 35.820 | Fire Hydrant          | 36.707 |
| Basketball          | 31.418 | Zebra                           | 64.890 | Grape                 | 2.106  |
| Giraffe             | 67.413 | Potato                          | 17.646 | Sausage               | 17.688 |
| Tricycle            | 5.801  | Violin                          | 19.587 | Egg                   | 55.392 |
| Fire Extinguisher   | 34.734 | Candy                           | 0.785  | Fire Truck            | 43.544 |
| Billards            | 33.112 | Converter                       | 0.023  | Bathtub               | 53.323 |
| Wheelchair          | 50.429 | Golf Club                       | 34.856 | Briefcase             | 8.140  |
| Cucumber            | 26.350 | Cigar/Cigarette                 | 15.046 | Paint Brush           | 4.672  |
| Pear                | 12.908 | Heavy Truck                     | 10.994 | Hamburger             | 21.095 |
| Extractor           | 0.021  | Extension Cord                  | 0.678  | Tong                  | 0.025  |
| Tennis Racket       | 53.834 | Folder                          | 5.691  | American Football     | 15.036 |
| earphone            | 0.960  | Mask                            | 14.891 | Kettle                | 2.399  |
| Tennis              | 0.000  | Ship                            | 21.771 | Swing                 | 0.214  |
| Coffee Machine      | 39.998 | Slide                           | 30.770 | Carriage              | 5.704  |
| Onion               | 20.372 | Green beans                     | 5.281  | Projector             | 25.718 |
| Frisbee             | 65.044 | Washing Machine/Drying Machine  | 34.295 | Chicken               | 48.769 |
| Printer             | 47.935 | Watermelon                      | 39.286 | Saxophone             | 36.908 |
| Tissue              | 0.084  | Toothbrush                      | 35.533 | Ice cream             | 7.528  |
| Hot air balloon     | 69.311 | Cello                           | 9.216  | French Fries          | 0.055  |
| Scale               | 7.237  | Trophy                          | 26.697 | Cabbage               | 4.384  |
| Hot dog             | 1.376  | Blender                         | 36.432 | Peach                 | 28.505 |
| Rice                | 0.255  | Wallet/Purse                    | 28.121 | Volleyball            | 50.590 |
| Deer                | 46.807 | Goose                           | 18.443 | Tape                  | 13.447 |
| Tablet              | 11.973 | Cosmetics                       | 2.826  | Trumpet               | 9.143  |
| Pineapple           | 18.574 | Golf Ball                       | 21.068 | Ambulance             | 67.908 |
| Parking meter       | 29.806 | Mango                           | 0.430  | Key                   | 11.840 |
| Hurdle              | 0.462  | Fishing Rod                     | 25.625 | Medal                 | 5.676  |
| Flute               | 18.877 | Brush                           | 1.261  | Penguin               | 58.837 |
| Megaphone           | 5.802  | Corn                            | 5.803  | Lettuce               | 2.163  |
| Garlic              | 19.268 | Swan                            | 37.563 | Helicopter            | 40.399 |
| Green Onion         | 3.974  | Sandwich                        | 0.000  | Nuts                  | 3.163  |
| Speed Limit Sign    | 10.916 | Induction Cooker                | 1.820  | Broom                 | 11.711 |
| Trombone            | 9.182  | Plum                            | 0.427  | Rickshaw              | 0.659  |
| Goldfish            | 16.689 | Kiwi fruit                      | 21.971 | Router/modem          | 9.547  |
| Poker Card          | 11.773 | Toaster                         | 42.232 | Shrimp                | 17.435 |
| Sushi               | 47.196 | Cheese                          | 4.867  | Notepaper             | 2.830  |
| Cherry              | 11.666 | Pliers                          | 18.164 | CD                    | 3.855  |
| Pasta               | 0.199  | Hammer                          | 9.760  | Cue                   | 0.949  |
| Avocado             | 29.967 | Hami melon                      | 0.263  | Flask                 | 1.177  |
| Mushroom            | 19.760 | Screwdriver                     | 10.723 | Soap                  | 12.417 |
| Recorder            | 0.033  | Bear                            | 50.557 | Eggplant              | 16.816 |
| Board Eraser        | 2.098  | Coconut                         | 29.780 | Tape Measure/ Ruler   | 7.678  |
| Pig                 | 42.392 | Showerhead                      | 16.940 | Globe                 | 32.474 |
| Chips               | 0.215  | Steak                           | 19.335 | Crosswalk Sign        | 2.727  |
| Stapler             | 14.501 | Camel                           | 55.693 | Formula 1             | 11.325 |
| Pomegranate         | 0.516  | Dishwasher                      | 30.169 | Crab                  | 8.169  |
| Hoverboard          | 4.078  | Meatball                        | 29.245 | Rice Cooker           | 9.269  |
| Tuba                | 10.973 | Calculator                      | 34.532 | Papaya                | 7.143  |
| Antelope            | 10.481 | Parrot                          | 34.876 | Seal                  | 23.179 |
| Butterfly           | 44.467 | Dumbbell                        | 5.908  | Donkey                | 30.112 |
| Lion                | 22.520 | Urinal                          | 59.242 | Dolphin               | 31.491 |
| Electric Drill      | 15.270 | Hair Dryer                      | 12.262 | Egg tart              | 4.330  |
| Jellyfish           | 9.258  | Treadmill                       | 16.421 | Lighter               | 2.333  |
| Grapefruit          | 0.075  | Game board                      | 18.517 | Mop                   | 2.822  |
| Radish              | 0.105  | Baozi                           | 0.406  | Target                | 0.278  |
| French              | 0.000  | Spring Rolls                    | 22.370 | Monkey                | 36.809 |
| Rabbit              | 25.513 | Pencil Case                     | 5.590  | Yak                   | 5.294  |
| Red Cabbage         | 0.781  | Binoculars                      | 8.885  | Asparagus             | 13.974 |
| Barbell             | 4.153  | Scallop                         | 5.529  | Noddles               | 1.123  |
| Comb                | 5.283  | Dumpling                        | 2.117  | Oyster                | 22.420 |
| Table Tennis paddle | 6.908  | Cosmetics Brush/Eyeliner Pencil | 7.352  | Chainsaw              | 0.960  |
| Eraser              | 2.025  | Lobster                         | 3.362  | Durian                | 7.196  |
| Okra                | 0.000  | Lipstick                        | 5.030  | Cosmetics Mirror      | 0.154  |
| Curling             | 29.915 | Table Tennis                    | 1.135  |                       |        |
[05/13 15:15:50 detectron2]: Evaluation results for objects365_v2_val_spotdet_v2_clip_noun in csv format:
[05/13 15:15:50 d2.evaluation.testing]: copypaste: Task: bbox
[05/13 15:15:50 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/13 15:15:50 d2.evaluation.testing]: copypaste: 20.6125,28.3263,22.2378,8.5140,20.1088,31.6243