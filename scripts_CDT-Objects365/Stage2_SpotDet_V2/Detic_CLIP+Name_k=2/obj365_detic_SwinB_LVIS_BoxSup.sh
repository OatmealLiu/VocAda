#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_BoxSup_SpotDet_Noun_k=2.out


CFG_PATH="./configs_detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml"
MODEL_PATH="./models/Detic_CDT/BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth"

CLASSIFIER_NAME="o365_clip_a+cnamefix.npy"

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('objects365_v2_val_spotdet_v2_clip_noun_k=2',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(365,)" \
        MODEL.MASK_ON False


index created!
[05/13 14:48:02 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/13 15:01:57 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 835.07 seconds.
[05/13 15:02:17 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/13 15:04:06 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 108.99 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.200
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.083
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.199
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.179
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.582
[05/13 15:04:06 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 20.005 | 27.447 | 21.633 | 8.342 | 19.872 | 30.209 |
[05/13 15:04:11 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 15.740 | Sneakers                        | 13.505 | Chair                 | 28.198 |
| Other Shoes         | 0.729  | Hat                             | 33.972 | Car                   | 12.322 |
| Lamp                | 19.559 | Glasses                         | 23.648 | Bottle                | 15.139 |
| Desk                | 17.616 | Cup                             | 14.401 | Street Lights         | 4.584  |
| Cabinet/shelf       | 16.453 | Handbag/Satchel                 | 12.779 | Bracelet              | 5.142  |
| Plate               | 56.273 | Picture/Frame                   | 4.595  | Helmet                | 34.325 |
| Book                | 8.660  | Gloves                          | 21.629 | Storage box           | 9.663  |
| Boat                | 3.973  | Leather Shoes                   | 0.776  | Flower                | 2.293  |
| Bench               | 16.412 | Potted Plant                    | 0.897  | Bowl/Basin            | 37.394 |
| Flag                | 21.812 | Pillow                          | 41.022 | Boots                 | 18.607 |
| Vase                | 16.101 | Microphone                      | 17.549 | Necklace              | 10.548 |
| Ring                | 3.476  | SUV                             | 3.310  | Wine Glass            | 56.052 |
| Belt                | 5.704  | Monitor/TV                      | 42.693 | Backpack              | 22.983 |
| Umbrella            | 24.827 | Traffic Light                   | 32.525 | Speaker               | 30.781 |
| Watch               | 28.942 | Tie                             | 15.345 | Trash bin Can         | 36.536 |
| Slippers            | 6.161  | Bicycle                         | 41.571 | Stool                 | 27.786 |
| Barrel/bucket       | 20.894 | Van                             | 5.982  | Couch                 | 43.556 |
| Sandals             | 6.624  | Basket                          | 25.915 | Drum                  | 30.152 |
| Pen/Pencil          | 19.005 | Bus                             | 32.335 | Wild Bird             | 13.459 |
| High Heels          | 6.690  | Motorcycle                      | 24.670 | Guitar                | 46.261 |
| Carpet              | 29.874 | Cell Phone                      | 33.955 | Bread                 | 13.303 |
| Camera              | 26.220 | Canned                          | 14.871 | Truck                 | 9.072  |
| Traffic cone        | 37.229 | Cymbal                          | 37.814 | Lifesaver             | 2.588  |
| Towel               | 47.940 | Stuffed Toy                     | 22.345 | Candle                | 20.400 |
| Sailboat            | 8.119  | Laptop                          | 69.133 | Awning                | 12.614 |
| Bed                 | 47.883 | Faucet                          | 27.879 | Tent                  | 4.600  |
| Horse               | 40.251 | Mirror                          | 35.131 | Power outlet          | 9.372  |
| Sink                | 35.032 | Apple                           | 24.520 | Air Conditioner       | 13.029 |
| Knife               | 40.083 | Hockey Stick                    | 31.746 | Paddle                | 22.806 |
| Pickup Truck        | 25.180 | Fork                            | 54.044 | Traffic Sign          | 2.459  |
| Ballon              | 27.754 | Tripod                          | 5.623  | Dog                   | 50.596 |
| Spoon               | 40.296 | Clock                           | 45.321 | Pot                   | 28.932 |
| Cow                 | 17.123 | Cake                            | 13.464 | Dining Table          | 15.063 |
| Sheep               | 27.915 | Hanger                          | 2.107  | Blackboard/Whiteboard | 19.865 |
| Napkin              | 23.207 | Other Fish                      | 29.608 | Orange/Tangerine      | 4.880  |
| Toiletry            | 19.191 | Keyboard                        | 55.129 | Tomato                | 46.360 |
| Lantern             | 32.715 | Machinery Vehicle               | 6.824  | Fan                   | 13.491 |
| Green Vegetables    | 0.225  | Banana                          | 9.214  | Baseball Glove        | 40.529 |
| Airplane            | 60.769 | Mouse                           | 47.785 | Train                 | 24.743 |
| Pumpkin             | 53.999 | Soccer                          | 12.080 | Skiboard              | 3.159  |
| Luggage             | 24.859 | Nightstand                      | 22.463 | Teapot                | 20.763 |
| Telephone           | 20.892 | Trolley                         | 16.685 | Head Phone            | 27.698 |
| Sports Car          | 28.942 | Stop Sign                       | 30.961 | Dessert               | 5.380  |
| Scooter             | 15.862 | Stroller                        | 28.740 | Crane                 | 1.416  |
| Remote              | 34.325 | Refrigerator                    | 67.352 | Oven                  | 28.198 |
| Lemon               | 28.660 | Duck                            | 42.762 | Baseball Bat          | 41.349 |
| Surveillance Camera | 1.144  | Cat                             | 62.480 | Jug                   | 4.443  |
| Broccoli            | 45.830 | Piano                           | 25.541 | Pizza                 | 53.875 |
| Elephant            | 71.316 | Skateboard                      | 14.803 | Surfboard             | 47.835 |
| Gun                 | 14.642 | Skating and Skiing shoes        | 27.752 | Gas stove             | 16.364 |
| Donut               | 53.290 | Bow Tie                         | 22.595 | Carrot                | 30.081 |
| Toilet              | 73.104 | Kite                            | 49.034 | Strawberry            | 39.177 |
| Other Balls         | 8.906  | Shovel                          | 5.275  | Pepper                | 18.856 |
| Computer Box        | 2.260  | Toilet Paper                    | 30.190 | Cleaning Products     | 13.758 |
| Chopsticks          | 29.189 | Microwave                       | 62.182 | Pigeon                | 43.102 |
| Baseball            | 31.811 | Cutting/chopping Board          | 37.176 | Coffee Table          | 17.542 |
| Side Table          | 3.818  | Scissors                        | 38.054 | Marker                | 11.608 |
| Pie                 | 1.130  | Ladder                          | 22.341 | Snowboard             | 44.790 |
| Cookies             | 13.946 | Radiator                        | 36.545 | Fire Hydrant          | 36.668 |
| Basketball          | 26.072 | Zebra                           | 65.001 | Grape                 | 1.338  |
| Giraffe             | 67.647 | Potato                          | 18.416 | Sausage               | 21.905 |
| Tricycle            | 6.865  | Violin                          | 4.990  | Egg                   | 59.885 |
| Fire Extinguisher   | 34.088 | Candy                           | 1.496  | Fire Truck            | 39.348 |
| Billards            | 26.072 | Converter                       | 0.013  | Bathtub               | 54.900 |
| Wheelchair          | 42.230 | Golf Club                       | 35.882 | Briefcase             | 11.341 |
| Cucumber            | 25.522 | Cigar/Cigarette                 | 12.433 | Paint Brush           | 2.686  |
| Pear                | 11.526 | Heavy Truck                     | 12.660 | Hamburger             | 18.987 |
| Extractor           | 0.150  | Extension Cord                  | 0.558  | Tong                  | 0.116  |
| Tennis Racket       | 54.369 | Folder                          | 4.210  | American Football     | 13.160 |
| earphone            | 0.941  | Mask                            | 17.229 | Kettle                | 18.559 |
| Tennis              | 19.739 | Ship                            | 19.096 | Swing                 | 0.451  |
| Coffee Machine      | 43.169 | Slide                           | 30.683 | Carriage              | 6.265  |
| Onion               | 18.320 | Green beans                     | 4.697  | Projector             | 23.700 |
| Frisbee             | 60.463 | Washing Machine/Drying Machine  | 30.841 | Chicken               | 44.847 |
| Printer             | 47.980 | Watermelon                      | 35.541 | Saxophone             | 29.458 |
| Tissue              | 0.283  | Toothbrush                      | 35.241 | Ice cream             | 4.332  |
| Hot air balloon     | 52.190 | Cello                           | 5.856  | French Fries          | 0.034  |
| Scale               | 5.150  | Trophy                          | 27.334 | Cabbage               | 6.298  |
| Hot dog             | 0.953  | Blender                         | 37.404 | Peach                 | 21.249 |
| Rice                | 0.177  | Wallet/Purse                    | 26.646 | Volleyball            | 30.877 |
| Deer                | 40.674 | Goose                           | 16.512 | Tape                  | 13.574 |
| Tablet              | 7.113  | Cosmetics                       | 1.465  | Trumpet               | 11.901 |
| Pineapple           | 17.798 | Golf Ball                       | 7.153  | Ambulance             | 57.851 |
| Parking meter       | 29.216 | Mango                           | 0.260  | Key                   | 9.750  |
| Hurdle              | 0.936  | Fishing Rod                     | 19.044 | Medal                 | 5.148  |
| Flute               | 10.758 | Brush                           | 3.258  | Penguin               | 53.357 |
| Megaphone           | 3.260  | Corn                            | 5.523  | Lettuce               | 2.131  |
| Garlic              | 20.798 | Swan                            | 30.078 | Helicopter            | 38.234 |
| Green Onion         | 3.580  | Sandwich                        | 0.062  | Nuts                  | 1.957  |
| Speed Limit Sign    | 10.428 | Induction Cooker                | 3.424  | Broom                 | 11.564 |
| Trombone            | 5.561  | Plum                            | 0.146  | Rickshaw              | 1.107  |
| Goldfish            | 6.266  | Kiwi fruit                      | 22.910 | Router/modem          | 5.288  |
| Poker Card          | 12.155 | Toaster                         | 42.314 | Shrimp                | 18.027 |
| Sushi               | 42.722 | Cheese                          | 5.809  | Notepaper             | 2.453  |
| Cherry              | 8.065  | Pliers                          | 13.677 | CD                    | 1.409  |
| Pasta               | 0.163  | Hammer                          | 6.254  | Cue                   | 0.153  |
| Avocado             | 24.511 | Hami melon                      | 0.258  | Flask                 | 0.318  |
| Mushroom            | 17.253 | Screwdriver                     | 6.985  | Soap                  | 11.580 |
| Recorder            | 0.077  | Bear                            | 45.881 | Eggplant              | 14.627 |
| Board Eraser        | 3.884  | Coconut                         | 24.554 | Tape Measure/ Ruler   | 5.782  |
| Pig                 | 31.267 | Showerhead                      | 17.272 | Globe                 | 33.210 |
| Chips               | 0.256  | Steak                           | 19.233 | Crosswalk Sign        | 2.179  |
| Stapler             | 10.626 | Camel                           | 39.884 | Formula 1             | 2.084  |
| Pomegranate         | 0.521  | Dishwasher                      | 31.465 | Crab                  | 6.196  |
| Hoverboard          | 0.288  | Meatball                        | 23.841 | Rice Cooker           | 10.233 |
| Tuba                | 8.610  | Calculator                      | 32.651 | Papaya                | 1.919  |
| Antelope            | 8.179  | Parrot                          | 35.008 | Seal                  | 6.867  |
| Butterfly           | 41.759 | Dumbbell                        | 2.104  | Donkey                | 12.561 |
| Lion                | 7.646  | Urinal                          | 61.314 | Dolphin               | 11.081 |
| Electric Drill      | 11.926 | Hair Dryer                      | 12.193 | Egg tart              | 4.875  |
| Jellyfish           | 2.120  | Treadmill                       | 10.099 | Lighter               | 0.744  |
| Grapefruit          | 0.125  | Game board                      | 12.305 | Mop                   | 1.486  |
| Radish              | 0.100  | Baozi                           | 1.192  | Target                | 0.083  |
| French              | 0.000  | Spring Rolls                    | 13.688 | Monkey                | 31.569 |
| Rabbit              | 15.759 | Pencil Case                     | 1.929  | Yak                   | 7.518  |
| Red Cabbage         | 2.375  | Binoculars                      | 4.614  | Asparagus             | 10.055 |
| Barbell             | 0.556  | Scallop                         | 6.087  | Noddles               | 0.996  |
| Comb                | 8.047  | Dumpling                        | 1.827  | Oyster                | 17.099 |
| Table Tennis paddle | 1.259  | Cosmetics Brush/Eyeliner Pencil | 2.549  | Chainsaw              | 0.764  |
| Eraser              | 2.352  | Lobster                         | 2.883  | Durian                | 6.063  |
| Okra                | 0.000  | Lipstick                        | 3.427  | Cosmetics Mirror      | 0.588  |
| Curling             | 39.664 | Table Tennis                    | 0.044  |                       |        |
[05/13 15:05:23 detectron2]: Evaluation results for objects365_v2_val_spotdet_v2_clip_noun_k=2 in csv format:
[05/13 15:05:23 d2.evaluation.testing]: copypaste: Task: bbox
[05/13 15:05:23 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/13 15:05:23 d2.evaluation.testing]: copypaste: 20.0051,27.4475,21.6328,8.3417,19.8717,30.2091