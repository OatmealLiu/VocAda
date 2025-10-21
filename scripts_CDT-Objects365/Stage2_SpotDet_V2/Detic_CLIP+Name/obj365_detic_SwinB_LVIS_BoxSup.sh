#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_BoxSup_SpotDet_Noun.out


CFG_PATH="./configs_detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml"
MODEL_PATH="./models/Detic_CDT/BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth"

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
[05/13 14:58:53 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/13 15:14:00 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 907.29 seconds.
[05/13 15:14:31 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/13 15:16:40 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 129.16 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.197
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.270
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.082
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.192
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.160
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.231
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.534
[05/13 15:16:40 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 19.676 | 26.994 | 21.258 | 8.214 | 19.202 | 29.975 |
[05/13 15:16:47 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 15.810 | Sneakers                        | 12.115 | Chair                 | 25.309 |
| Other Shoes         | 0.228  | Hat                             | 32.217 | Car                   | 11.565 |
| Lamp                | 17.413 | Glasses                         | 23.551 | Bottle                | 13.132 |
| Desk                | 16.140 | Cup                             | 0.990  | Street Lights         | 4.243  |
| Cabinet/shelf       | 16.023 | Handbag/Satchel                 | 12.380 | Bracelet              | 4.515  |
| Plate               | 55.369 | Picture/Frame                   | 4.667  | Helmet                | 35.046 |
| Book                | 8.663  | Gloves                          | 20.376 | Storage box           | 8.620  |
| Boat                | 0.990  | Leather Shoes                   | 0.754  | Flower                | 2.305  |
| Bench               | 16.468 | Potted Plant                    | 0.893  | Bowl/Basin            | 37.378 |
| Flag                | 21.767 | Pillow                          | 39.558 | Boots                 | 18.495 |
| Vase                | 16.090 | Microphone                      | 17.555 | Necklace              | 10.035 |
| Ring                | 3.235  | SUV                             | 3.023  | Wine Glass            | 55.141 |
| Belt                | 4.952  | Monitor/TV                      | 44.563 | Backpack              | 22.523 |
| Umbrella            | 26.661 | Traffic Light                   | 32.664 | Speaker               | 29.958 |
| Watch               | 17.174 | Tie                             | 11.784 | Trash bin Can         | 34.323 |
| Slippers            | 1.380  | Bicycle                         | 41.581 | Stool                 | 26.077 |
| Barrel/bucket       | 18.385 | Van                             | 5.889  | Couch                 | 42.618 |
| Sandals             | 5.845  | Basket                          | 23.205 | Drum                  | 12.108 |
| Pen/Pencil          | 16.775 | Bus                             | 32.740 | Wild Bird             | 13.546 |
| High Heels          | 8.232  | Motorcycle                      | 22.152 | Guitar                | 46.657 |
| Carpet              | 30.038 | Cell Phone                      | 34.007 | Bread                 | 11.679 |
| Camera              | 12.602 | Canned                          | 13.645 | Truck                 | 0.916  |
| Traffic cone        | 36.328 | Cymbal                          | 37.814 | Lifesaver             | 2.649  |
| Towel               | 47.561 | Stuffed Toy                     | 22.813 | Candle                | 20.065 |
| Sailboat            | 9.250  | Laptop                          | 68.996 | Awning                | 12.691 |
| Bed                 | 47.590 | Faucet                          | 26.224 | Tent                  | 5.658  |
| Horse               | 42.156 | Mirror                          | 28.575 | Power outlet          | 9.174  |
| Sink                | 34.826 | Apple                           | 24.638 | Air Conditioner       | 10.874 |
| Knife               | 39.695 | Hockey Stick                    | 35.044 | Paddle                | 21.121 |
| Pickup Truck        | 18.843 | Fork                            | 52.576 | Traffic Sign          | 2.500  |
| Ballon              | 0.564  | Tripod                          | 4.718  | Dog                   | 51.919 |
| Spoon               | 39.000 | Clock                           | 45.765 | Pot                   | 27.270 |
| Cow                 | 17.992 | Cake                            | 11.259 | Dining Table          | 15.222 |
| Sheep               | 28.590 | Hanger                          | 2.138  | Blackboard/Whiteboard | 19.956 |
| Napkin              | 22.858 | Other Fish                      | 30.110 | Orange/Tangerine      | 5.096  |
| Toiletry            | 14.422 | Keyboard                        | 55.170 | Tomato                | 28.949 |
| Lantern             | 30.656 | Machinery Vehicle               | 6.824  | Fan                   | 1.546  |
| Green Vegetables    | 0.224  | Banana                          | 9.111  | Baseball Glove        | 44.090 |
| Airplane            | 60.929 | Mouse                           | 47.414 | Train                 | 25.195 |
| Pumpkin             | 54.080 | Soccer                          | 11.685 | Skiboard              | 4.605  |
| Luggage             | 26.546 | Nightstand                      | 24.239 | Teapot                | 18.956 |
| Telephone           | 10.431 | Trolley                         | 16.934 | Head Phone            | 27.323 |
| Sports Car          | 29.892 | Stop Sign                       | 25.585 | Dessert               | 3.206  |
| Scooter             | 15.758 | Stroller                        | 33.438 | Crane                 | 1.438  |
| Remote              | 34.178 | Refrigerator                    | 67.097 | Oven                  | 27.805 |
| Lemon               | 28.515 | Duck                            | 41.555 | Baseball Bat          | 45.367 |
| Surveillance Camera | 1.062  | Cat                             | 63.320 | Jug                   | 3.956  |
| Broccoli            | 44.564 | Piano                           | 19.471 | Pizza                 | 52.183 |
| Elephant            | 71.185 | Skateboard                      | 15.014 | Surfboard             | 48.226 |
| Gun                 | 17.469 | Skating and Skiing shoes        | 23.813 | Gas stove             | 16.362 |
| Donut               | 53.215 | Bow Tie                         | 19.431 | Carrot                | 30.157 |
| Toilet              | 73.199 | Kite                            | 49.052 | Strawberry            | 38.950 |
| Other Balls         | 8.362  | Shovel                          | 5.152  | Pepper                | 18.155 |
| Computer Box        | 3.046  | Toilet Paper                    | 29.403 | Cleaning Products     | 12.643 |
| Chopsticks          | 28.700 | Microwave                       | 59.075 | Pigeon                | 43.114 |
| Baseball            | 39.458 | Cutting/chopping Board          | 36.161 | Coffee Table          | 19.753 |
| Side Table          | 4.406  | Scissors                        | 37.680 | Marker                | 11.230 |
| Pie                 | 0.990  | Ladder                          | 22.218 | Snowboard             | 44.836 |
| Cookies             | 13.940 | Radiator                        | 35.563 | Fire Hydrant          | 36.714 |
| Basketball          | 28.796 | Zebra                           | 65.040 | Grape                 | 1.258  |
| Giraffe             | 67.664 | Potato                          | 16.107 | Sausage               | 18.805 |
| Tricycle            | 6.098  | Violin                          | 5.504  | Egg                   | 55.340 |
| Fire Extinguisher   | 33.230 | Candy                           | 1.749  | Fire Truck            | 41.522 |
| Billards            | 28.287 | Converter                       | 0.050  | Bathtub               | 52.866 |
| Wheelchair          | 46.198 | Golf Club                       | 35.628 | Briefcase             | 7.665  |
| Cucumber            | 24.531 | Cigar/Cigarette                 | 12.086 | Paint Brush           | 3.150  |
| Pear                | 11.234 | Heavy Truck                     | 13.018 | Hamburger             | 21.028 |
| Extractor           | 0.008  | Extension Cord                  | 0.647  | Tong                  | 0.173  |
| Tennis Racket       | 54.846 | Folder                          | 4.777  | American Football     | 12.567 |
| earphone            | 0.859  | Mask                            | 19.898 | Kettle                | 2.076  |
| Tennis              | 0.000  | Ship                            | 19.382 | Swing                 | 0.472  |
| Coffee Machine      | 39.824 | Slide                           | 29.688 | Carriage              | 6.098  |
| Onion               | 17.463 | Green beans                     | 4.119  | Projector             | 24.268 |
| Frisbee             | 63.306 | Washing Machine/Drying Machine  | 31.929 | Chicken               | 47.740 |
| Printer             | 47.492 | Watermelon                      | 35.637 | Saxophone             | 29.586 |
| Tissue              | 0.064  | Toothbrush                      | 35.274 | Ice cream             | 4.679  |
| Hot air balloon     | 53.445 | Cello                           | 7.800  | French Fries          | 0.055  |
| Scale               | 4.931  | Trophy                          | 29.142 | Cabbage               | 5.189  |
| Hot dog             | 1.338  | Blender                         | 36.400 | Peach                 | 21.492 |
| Rice                | 0.192  | Wallet/Purse                    | 26.474 | Volleyball            | 35.817 |
| Deer                | 41.305 | Goose                           | 16.961 | Tape                  | 12.659 |
| Tablet              | 9.241  | Cosmetics                       | 1.561  | Trumpet               | 11.786 |
| Pineapple           | 17.952 | Golf Ball                       | 10.746 | Ambulance             | 61.064 |
| Parking meter       | 28.873 | Mango                           | 0.251  | Key                   | 9.315  |
| Hurdle              | 0.941  | Fishing Rod                     | 21.708 | Medal                 | 6.406  |
| Flute               | 11.677 | Brush                           | 0.872  | Penguin               | 55.173 |
| Megaphone           | 3.804  | Corn                            | 5.461  | Lettuce               | 2.149  |
| Garlic              | 14.864 | Swan                            | 37.408 | Helicopter            | 40.138 |
| Green Onion         | 3.966  | Sandwich                        | 0.000  | Nuts                  | 2.298  |
| Speed Limit Sign    | 9.953  | Induction Cooker                | 2.236  | Broom                 | 10.806 |
| Trombone            | 5.714  | Plum                            | 0.183  | Rickshaw              | 0.632  |
| Goldfish            | 11.571 | Kiwi fruit                      | 22.037 | Router/modem          | 4.837  |
| Poker Card          | 14.039 | Toaster                         | 39.506 | Shrimp                | 18.184 |
| Sushi               | 43.954 | Cheese                          | 5.112  | Notepaper             | 2.541  |
| Cherry              | 8.832  | Pliers                          | 13.656 | CD                    | 2.323  |
| Pasta               | 0.249  | Hammer                          | 5.846  | Cue                   | 1.293  |
| Avocado             | 24.031 | Hami melon                      | 0.210  | Flask                 | 0.378  |
| Mushroom            | 17.769 | Screwdriver                     | 6.784  | Soap                  | 10.788 |
| Recorder            | 0.031  | Bear                            | 49.711 | Eggplant              | 14.229 |
| Board Eraser        | 3.737  | Coconut                         | 26.285 | Tape Measure/ Ruler   | 6.514  |
| Pig                 | 32.538 | Showerhead                      | 16.939 | Globe                 | 33.464 |
| Chips               | 0.360  | Steak                           | 18.641 | Crosswalk Sign        | 2.773  |
| Stapler             | 10.825 | Camel                           | 41.659 | Formula 1             | 3.089  |
| Pomegranate         | 0.532  | Dishwasher                      | 29.197 | Crab                  | 7.836  |
| Hoverboard          | 4.234  | Meatball                        | 25.281 | Rice Cooker           | 8.963  |
| Tuba                | 8.248  | Calculator                      | 32.870 | Papaya                | 1.982  |
| Antelope            | 12.354 | Parrot                          | 34.822 | Seal                  | 23.896 |
| Butterfly           | 42.705 | Dumbbell                        | 2.636  | Donkey                | 30.500 |
| Lion                | 10.844 | Urinal                          | 61.009 | Dolphin               | 14.702 |
| Electric Drill      | 12.040 | Hair Dryer                      | 10.962 | Egg tart              | 4.224  |
| Jellyfish           | 3.680  | Treadmill                       | 13.677 | Lighter               | 0.660  |
| Grapefruit          | 0.188  | Game board                      | 15.995 | Mop                   | 1.562  |
| Radish              | 0.092  | Baozi                           | 0.587  | Target                | 0.082  |
| French              | 0.000  | Spring Rolls                    | 14.318 | Monkey                | 33.105 |
| Rabbit              | 16.108 | Pencil Case                     | 1.952  | Yak                   | 5.377  |
| Red Cabbage         | 2.996  | Binoculars                      | 5.719  | Asparagus             | 10.203 |
| Barbell             | 0.721  | Scallop                         | 2.954  | Noddles               | 1.706  |
| Comb                | 6.014  | Dumpling                        | 2.021  | Oyster                | 19.083 |
| Table Tennis paddle | 1.875  | Cosmetics Brush/Eyeliner Pencil | 5.948  | Chainsaw              | 1.029  |
| Eraser              | 1.017  | Lobster                         | 4.635  | Durian                | 6.608  |
| Okra                | 0.000  | Lipstick                        | 3.932  | Cosmetics Mirror      | 0.593  |
| Curling             | 60.907 | Table Tennis                    | 0.573  |                       |        |
[05/13 15:18:18 detectron2]: Evaluation results for objects365_v2_val_spotdet_v2_clip_noun in csv format:
[05/13 15:18:18 d2.evaluation.testing]: copypaste: Task: bbox
[05/13 15:18:18 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/13 15:18:18 d2.evaluation.testing]: copypaste: 19.6763,26.9937,21.2582,8.2140,19.2018,29.9749