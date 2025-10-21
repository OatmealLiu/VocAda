#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_IN-21k-COCO_SpotDet_Noun_k=2.out


CFG_PATH="./configs_detic/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

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
[05/13 14:48:19 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/13 15:02:23 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 844.12 seconds.
[05/13 15:02:39 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/13 15:04:22 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 102.87 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.220
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.298
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.090
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.185
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.409
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.603
[05/13 15:04:22 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 21.965 | 29.841 | 23.722 | 9.012 | 21.463 | 32.971 |
[05/13 15:04:25 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 61.774 | Sneakers                        | 16.348 | Chair                 | 36.224 |
| Other Shoes         | 0.667  | Hat                             | 33.984 | Car                   | 19.144 |
| Lamp                | 19.768 | Glasses                         | 25.339 | Bottle                | 19.914 |
| Desk                | 18.940 | Cup                             | 17.129 | Street Lights         | 4.695  |
| Cabinet/shelf       | 15.474 | Handbag/Satchel                 | 16.238 | Bracelet              | 5.106  |
| Plate               | 51.983 | Picture/Frame                   | 7.919  | Helmet                | 32.394 |
| Book                | 15.158 | Gloves                          | 22.294 | Storage box           | 8.393  |
| Boat                | 4.002  | Leather Shoes                   | 0.607  | Flower                | 4.757  |
| Bench               | 19.206 | Potted Plant                    | 26.815 | Bowl/Basin            | 42.978 |
| Flag                | 21.459 | Pillow                          | 41.810 | Boots                 | 18.019 |
| Vase                | 19.363 | Microphone                      | 15.997 | Necklace              | 10.004 |
| Ring                | 3.466  | SUV                             | 4.488  | Wine Glass            | 60.514 |
| Belt                | 5.622  | Monitor/TV                      | 50.330 | Backpack              | 23.922 |
| Umbrella            | 26.236 | Traffic Light                   | 35.413 | Speaker               | 31.756 |
| Watch               | 28.572 | Tie                             | 22.995 | Trash bin Can         | 35.510 |
| Slippers            | 5.899  | Bicycle                         | 42.899 | Stool                 | 29.136 |
| Barrel/bucket       | 20.693 | Van                             | 8.192  | Couch                 | 46.309 |
| Sandals             | 7.244  | Basket                          | 25.935 | Drum                  | 31.972 |
| Pen/Pencil          | 18.020 | Bus                             | 38.523 | Wild Bird             | 11.354 |
| High Heels          | 7.233  | Motorcycle                      | 24.616 | Guitar                | 45.419 |
| Carpet              | 31.345 | Cell Phone                      | 38.826 | Bread                 | 13.445 |
| Camera              | 25.493 | Canned                          | 17.848 | Truck                 | 9.842  |
| Traffic cone        | 37.095 | Cymbal                          | 35.821 | Lifesaver             | 6.808  |
| Towel               | 48.029 | Stuffed Toy                     | 26.079 | Candle                | 20.580 |
| Sailboat            | 17.273 | Laptop                          | 72.193 | Awning                | 12.725 |
| Bed                 | 53.097 | Faucet                          | 27.521 | Tent                  | 7.046  |
| Horse               | 46.250 | Mirror                          | 35.537 | Power outlet          | 16.454 |
| Sink                | 42.184 | Apple                           | 25.469 | Air Conditioner       | 13.296 |
| Knife               | 41.706 | Hockey Stick                    | 36.361 | Paddle                | 19.259 |
| Pickup Truck        | 22.261 | Fork                            | 56.314 | Traffic Sign          | 2.481  |
| Ballon              | 28.630 | Tripod                          | 5.092  | Dog                   | 57.012 |
| Spoon               | 40.219 | Clock                           | 47.617 | Pot                   | 28.904 |
| Cow                 | 19.827 | Cake                            | 12.465 | Dining Table          | 18.831 |
| Sheep               | 30.836 | Hanger                          | 3.723  | Blackboard/Whiteboard | 19.317 |
| Napkin              | 20.864 | Other Fish                      | 33.204 | Orange/Tangerine      | 8.357  |
| Toiletry            | 22.199 | Keyboard                        | 61.669 | Tomato                | 45.895 |
| Lantern             | 31.385 | Machinery Vehicle               | 12.224 | Fan                   | 13.311 |
| Green Vegetables    | 0.298  | Banana                          | 13.045 | Baseball Glove        | 41.996 |
| Airplane            | 58.758 | Mouse                           | 53.412 | Train                 | 41.364 |
| Pumpkin             | 55.008 | Soccer                          | 11.153 | Skiboard              | 4.925  |
| Luggage             | 29.599 | Nightstand                      | 26.484 | Teapot                | 21.156 |
| Telephone           | 20.913 | Trolley                         | 17.707 | Head Phone            | 27.245 |
| Sports Car          | 47.539 | Stop Sign                       | 36.956 | Dessert               | 7.828  |
| Scooter             | 14.548 | Stroller                        | 27.166 | Crane                 | 4.647  |
| Remote              | 36.946 | Refrigerator                    | 67.207 | Oven                  | 23.389 |
| Lemon               | 29.800 | Duck                            | 40.217 | Baseball Bat          | 41.520 |
| Surveillance Camera | 1.137  | Cat                             | 68.384 | Jug                   | 5.683  |
| Broccoli            | 46.252 | Piano                           | 22.600 | Pizza                 | 55.581 |
| Elephant            | 72.228 | Skateboard                      | 11.877 | Surfboard             | 48.833 |
| Gun                 | 20.115 | Skating and Skiing shoes        | 26.807 | Gas stove             | 16.704 |
| Donut               | 54.764 | Bow Tie                         | 20.868 | Carrot                | 31.790 |
| Toilet              | 74.795 | Kite                            | 49.701 | Strawberry            | 37.510 |
| Other Balls         | 7.332  | Shovel                          | 5.946  | Pepper                | 24.845 |
| Computer Box        | 5.123  | Toilet Paper                    | 33.060 | Cleaning Products     | 10.880 |
| Chopsticks          | 27.741 | Microwave                       | 63.126 | Pigeon                | 44.040 |
| Baseball            | 29.024 | Cutting/chopping Board          | 34.424 | Coffee Table          | 17.610 |
| Side Table          | 4.389  | Scissors                        | 38.419 | Marker                | 11.348 |
| Pie                 | 0.943  | Ladder                          | 21.965 | Snowboard             | 41.897 |
| Cookies             | 14.667 | Radiator                        | 35.279 | Fire Hydrant          | 36.165 |
| Basketball          | 21.193 | Zebra                           | 65.081 | Grape                 | 2.057  |
| Giraffe             | 68.222 | Potato                          | 16.212 | Sausage               | 33.746 |
| Tricycle            | 7.494  | Violin                          | 13.141 | Egg                   | 60.033 |
| Fire Extinguisher   | 32.961 | Candy                           | 1.358  | Fire Truck            | 34.291 |
| Billards            | 26.830 | Converter                       | 0.054  | Bathtub               | 55.381 |
| Wheelchair          | 42.951 | Golf Club                       | 33.952 | Briefcase             | 13.635 |
| Cucumber            | 24.551 | Cigar/Cigarette                 | 13.532 | Paint Brush           | 3.541  |
| Pear                | 12.023 | Heavy Truck                     | 10.784 | Hamburger             | 21.124 |
| Extractor           | 0.515  | Extension Cord                  | 1.338  | Tong                  | 0.018  |
| Tennis Racket       | 55.582 | Folder                          | 3.407  | American Football     | 9.579  |
| earphone            | 0.935  | Mask                            | 12.393 | Kettle                | 19.125 |
| Tennis              | 13.929 | Ship                            | 37.945 | Swing                 | 0.422  |
| Coffee Machine      | 36.968 | Slide                           | 31.980 | Carriage              | 5.680  |
| Onion               | 14.670 | Green beans                     | 5.613  | Projector             | 21.543 |
| Frisbee             | 58.956 | Washing Machine/Drying Machine  | 33.296 | Chicken               | 46.906 |
| Printer             | 47.390 | Watermelon                      | 33.418 | Saxophone             | 29.424 |
| Tissue              | 0.174  | Toothbrush                      | 35.159 | Ice cream             | 6.387  |
| Hot air balloon     | 42.493 | Cello                           | 8.455  | French Fries          | 0.218  |
| Scale               | 4.453  | Trophy                          | 24.308 | Cabbage               | 8.902  |
| Hot dog             | 8.291  | Blender                         | 40.009 | Peach                 | 22.922 |
| Rice                | 3.657  | Wallet/Purse                    | 25.795 | Volleyball            | 29.314 |
| Deer                | 46.773 | Goose                           | 15.190 | Tape                  | 16.539 |
| Tablet              | 10.481 | Cosmetics                       | 3.944  | Trumpet               | 13.963 |
| Pineapple           | 19.126 | Golf Ball                       | 19.511 | Ambulance             | 46.040 |
| Parking meter       | 17.877 | Mango                           | 0.980  | Key                   | 11.368 |
| Hurdle              | 0.032  | Fishing Rod                     | 16.917 | Medal                 | 6.179  |
| Flute               | 19.027 | Brush                           | 5.106  | Penguin               | 54.588 |
| Megaphone           | 4.600  | Corn                            | 11.239 | Lettuce               | 1.734  |
| Garlic              | 15.105 | Swan                            | 43.585 | Helicopter            | 38.209 |
| Green Onion         | 2.591  | Sandwich                        | 0.088  | Nuts                  | 7.195  |
| Speed Limit Sign    | 8.593  | Induction Cooker                | 4.215  | Broom                 | 11.397 |
| Trombone            | 3.411  | Plum                            | 0.636  | Rickshaw              | 1.244  |
| Goldfish            | 9.609  | Kiwi fruit                      | 23.189 | Router/modem          | 9.831  |
| Poker Card          | 24.713 | Toaster                         | 43.658 | Shrimp                | 19.615 |
| Sushi               | 40.901 | Cheese                          | 22.679 | Notepaper             | 2.055  |
| Cherry              | 9.181  | Pliers                          | 14.640 | CD                    | 7.366  |
| Pasta               | 1.251  | Hammer                          | 7.958  | Cue                   | 3.290  |
| Avocado             | 25.469 | Hami melon                      | 0.240  | Flask                 | 0.711  |
| Mushroom            | 16.756 | Screwdriver                     | 8.142  | Soap                  | 15.590 |
| Recorder            | 0.377  | Bear                            | 49.429 | Eggplant              | 14.130 |
| Board Eraser        | 2.204  | Coconut                         | 22.332 | Tape Measure/ Ruler   | 8.815  |
| Pig                 | 41.955 | Showerhead                      | 17.571 | Globe                 | 31.599 |
| Chips               | 0.501  | Steak                           | 22.000 | Crosswalk Sign        | 1.987  |
| Stapler             | 9.907  | Camel                           | 53.736 | Formula 1             | 10.482 |
| Pomegranate         | 3.260  | Dishwasher                      | 30.653 | Crab                  | 12.194 |
| Hoverboard          | 0.428  | Meatball                        | 27.464 | Rice Cooker           | 8.102  |
| Tuba                | 10.338 | Calculator                      | 33.574 | Papaya                | 2.589  |
| Antelope            | 9.240  | Parrot                          | 32.770 | Seal                  | 36.495 |
| Butterfly           | 41.969 | Dumbbell                        | 2.660  | Donkey                | 24.385 |
| Lion                | 11.523 | Urinal                          | 61.856 | Dolphin               | 17.486 |
| Electric Drill      | 13.209 | Hair Dryer                      | 14.703 | Egg tart              | 6.963  |
| Jellyfish           | 30.745 | Treadmill                       | 27.806 | Lighter               | 2.668  |
| Grapefruit          | 0.296  | Game board                      | 18.481 | Mop                   | 1.664  |
| Radish              | 0.207  | Baozi                           | 8.140  | Target                | 0.760  |
| French              | 0.000  | Spring Rolls                    | 15.918 | Monkey                | 41.210 |
| Rabbit              | 26.486 | Pencil Case                     | 6.063  | Yak                   | 8.779  |
| Red Cabbage         | 4.574  | Binoculars                      | 2.474  | Asparagus             | 9.030  |
| Barbell             | 0.807  | Scallop                         | 18.537 | Noddles               | 1.061  |
| Comb                | 13.431 | Dumpling                        | 5.422  | Oyster                | 39.398 |
| Table Tennis paddle | 2.448  | Cosmetics Brush/Eyeliner Pencil | 1.685  | Chainsaw              | 3.696  |
| Eraser              | 2.315  | Lobster                         | 4.163  | Durian                | 28.479 |
| Okra                | 0.000  | Lipstick                        | 7.897  | Cosmetics Mirror      | 0.552  |
| Curling             | 51.975 | Table Tennis                    | 0.038  |                       |        |
[05/13 15:05:32 detectron2]: Evaluation results for objects365_v2_val_spotdet_v2_clip_noun_k=2 in csv format:
[05/13 15:05:32 d2.evaluation.testing]: copypaste: Task: bbox
[05/13 15:05:32 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/13 15:05:32 d2.evaluation.testing]: copypaste: 21.9650,29.8412,23.7220,9.0116,21.4634,32.9706