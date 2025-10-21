#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_IN-21k_SpotDet_Noun_k=2.out


CFG_PATH="./configs_detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

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
[05/13 14:47:54 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/13 15:02:02 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 847.90 seconds.
[05/13 15:02:24 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/13 15:04:34 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 130.04 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.296
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.233
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.181
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.399
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.267
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.589
[05/13 15:04:34 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 21.561 | 29.626 | 23.294 | 8.861 | 21.131 | 32.506 |
[05/13 15:04:38 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 15.092 | Sneakers                        | 18.224 | Chair                 | 26.202 |
| Other Shoes         | 0.647  | Hat                             | 33.786 | Car                   | 14.126 |
| Lamp                | 19.259 | Glasses                         | 26.133 | Bottle                | 15.291 |
| Desk                | 17.602 | Cup                             | 14.695 | Street Lights         | 4.409  |
| Cabinet/shelf       | 15.599 | Handbag/Satchel                 | 13.449 | Bracelet              | 5.572  |
| Plate               | 54.710 | Picture/Frame                   | 8.493  | Helmet                | 34.051 |
| Book                | 8.665  | Gloves                          | 21.976 | Storage box           | 9.450  |
| Boat                | 4.160  | Leather Shoes                   | 0.452  | Flower                | 3.969  |
| Bench               | 15.633 | Potted Plant                    | 0.330  | Bowl/Basin            | 37.440 |
| Flag                | 23.169 | Pillow                          | 40.425 | Boots                 | 18.129 |
| Vase                | 16.371 | Microphone                      | 16.506 | Necklace              | 10.434 |
| Ring                | 3.587  | SUV                             | 3.701  | Wine Glass            | 56.160 |
| Belt                | 5.659  | Monitor/TV                      | 54.516 | Backpack              | 22.406 |
| Umbrella            | 24.911 | Traffic Light                   | 33.410 | Speaker               | 31.744 |
| Watch               | 28.599 | Tie                             | 23.913 | Trash bin Can         | 34.294 |
| Slippers            | 5.910  | Bicycle                         | 38.997 | Stool                 | 27.576 |
| Barrel/bucket       | 20.425 | Van                             | 7.727  | Couch                 | 43.714 |
| Sandals             | 6.357  | Basket                          | 26.416 | Drum                  | 30.512 |
| Pen/Pencil          | 18.970 | Bus                             | 35.528 | Wild Bird             | 13.584 |
| High Heels          | 6.919  | Motorcycle                      | 21.851 | Guitar                | 47.720 |
| Carpet              | 27.602 | Cell Phone                      | 35.156 | Bread                 | 12.983 |
| Camera              | 25.522 | Canned                          | 14.728 | Truck                 | 8.803  |
| Traffic cone        | 37.404 | Cymbal                          | 34.882 | Lifesaver             | 3.177  |
| Towel               | 47.347 | Stuffed Toy                     | 27.755 | Candle                | 21.011 |
| Sailboat            | 14.427 | Laptop                          | 68.379 | Awning                | 12.293 |
| Bed                 | 46.473 | Faucet                          | 27.169 | Tent                  | 8.116  |
| Horse               | 41.278 | Mirror                          | 34.784 | Power outlet          | 16.600 |
| Sink                | 35.163 | Apple                           | 24.948 | Air Conditioner       | 13.023 |
| Knife               | 38.012 | Hockey Stick                    | 40.340 | Paddle                | 22.006 |
| Pickup Truck        | 22.715 | Fork                            | 52.423 | Traffic Sign          | 2.993  |
| Ballon              | 29.793 | Tripod                          | 5.119  | Dog                   | 50.207 |
| Spoon               | 38.347 | Clock                           | 46.427 | Pot                   | 27.924 |
| Cow                 | 18.238 | Cake                            | 12.606 | Dining Table          | 13.265 |
| Sheep               | 27.385 | Hanger                          | 3.131  | Blackboard/Whiteboard | 18.442 |
| Napkin              | 21.842 | Other Fish                      | 32.958 | Orange/Tangerine      | 10.834 |
| Toiletry            | 22.746 | Keyboard                        | 54.765 | Tomato                | 45.883 |
| Lantern             | 32.643 | Machinery Vehicle               | 9.507  | Fan                   | 13.193 |
| Green Vegetables    | 0.201  | Banana                          | 9.520  | Baseball Glove        | 43.201 |
| Airplane            | 59.351 | Mouse                           | 51.503 | Train                 | 39.193 |
| Pumpkin             | 53.951 | Soccer                          | 11.814 | Skiboard              | 4.750  |
| Luggage             | 27.784 | Nightstand                      | 24.307 | Teapot                | 19.425 |
| Telephone           | 20.643 | Trolley                         | 16.126 | Head Phone            | 28.525 |
| Sports Car          | 51.469 | Stop Sign                       | 32.453 | Dessert               | 6.733  |
| Scooter             | 15.861 | Stroller                        | 31.101 | Crane                 | 5.378  |
| Remote              | 40.900 | Refrigerator                    | 65.983 | Oven                  | 27.288 |
| Lemon               | 31.448 | Duck                            | 43.749 | Baseball Bat          | 43.547 |
| Surveillance Camera | 1.081  | Cat                             | 62.399 | Jug                   | 5.617  |
| Broccoli            | 45.559 | Piano                           | 23.123 | Pizza                 | 52.247 |
| Elephant            | 70.539 | Skateboard                      | 14.075 | Surfboard             | 47.529 |
| Gun                 | 19.261 | Skating and Skiing shoes        | 23.539 | Gas stove             | 15.519 |
| Donut               | 53.174 | Bow Tie                         | 22.589 | Carrot                | 31.096 |
| Toilet              | 73.276 | Kite                            | 47.812 | Strawberry            | 36.986 |
| Other Balls         | 10.536 | Shovel                          | 5.828  | Pepper                | 25.497 |
| Computer Box        | 4.623  | Toilet Paper                    | 35.263 | Cleaning Products     | 12.924 |
| Chopsticks          | 28.236 | Microwave                       | 64.303 | Pigeon                | 42.981 |
| Baseball            | 30.578 | Cutting/chopping Board          | 36.268 | Coffee Table          | 14.486 |
| Side Table          | 2.708  | Scissors                        | 36.808 | Marker                | 11.321 |
| Pie                 | 1.614  | Ladder                          | 22.390 | Snowboard             | 40.562 |
| Cookies             | 16.190 | Radiator                        | 35.650 | Fire Hydrant          | 36.004 |
| Basketball          | 25.218 | Zebra                           | 64.476 | Grape                 | 1.171  |
| Giraffe             | 67.337 | Potato                          | 16.953 | Sausage               | 21.064 |
| Tricycle            | 6.611  | Violin                          | 9.567  | Egg                   | 59.353 |
| Fire Extinguisher   | 32.936 | Candy                           | 1.974  | Fire Truck            | 36.111 |
| Billards            | 27.411 | Converter                       | 0.090  | Bathtub               | 54.554 |
| Wheelchair          | 37.450 | Golf Club                       | 33.893 | Briefcase             | 11.616 |
| Cucumber            | 23.373 | Cigar/Cigarette                 | 10.863 | Paint Brush           | 3.275  |
| Pear                | 10.647 | Heavy Truck                     | 12.333 | Hamburger             | 19.241 |
| Extractor           | 0.315  | Extension Cord                  | 1.275  | Tong                  | 0.059  |
| Tennis Racket       | 56.173 | Folder                          | 5.003  | American Football     | 13.313 |
| earphone            | 0.979  | Mask                            | 11.958 | Kettle                | 18.845 |
| Tennis              | 18.255 | Ship                            | 35.384 | Swing                 | 0.689  |
| Coffee Machine      | 38.273 | Slide                           | 31.051 | Carriage              | 6.677  |
| Onion               | 15.473 | Green beans                     | 5.414  | Projector             | 20.755 |
| Frisbee             | 56.955 | Washing Machine/Drying Machine  | 29.709 | Chicken               | 47.532 |
| Printer             | 46.719 | Watermelon                      | 33.823 | Saxophone             | 33.137 |
| Tissue              | 0.178  | Toothbrush                      | 34.292 | Ice cream             | 6.528  |
| Hot air balloon     | 47.338 | Cello                           | 19.315 | French Fries          | 0.063  |
| Scale               | 6.141  | Trophy                          | 25.734 | Cabbage               | 8.778  |
| Hot dog             | 1.655  | Blender                         | 39.166 | Peach                 | 18.604 |
| Rice                | 4.680  | Wallet/Purse                    | 25.983 | Volleyball            | 38.002 |
| Deer                | 45.226 | Goose                           | 17.685 | Tape                  | 15.621 |
| Tablet              | 12.732 | Cosmetics                       | 6.382  | Trumpet               | 14.127 |
| Pineapple           | 17.445 | Golf Ball                       | 16.926 | Ambulance             | 61.317 |
| Parking meter       | 29.428 | Mango                           | 0.721  | Key                   | 11.085 |
| Hurdle              | 0.069  | Fishing Rod                     | 21.494 | Medal                 | 6.034  |
| Flute               | 22.806 | Brush                           | 4.955  | Penguin               | 56.277 |
| Megaphone           | 5.962  | Corn                            | 12.773 | Lettuce               | 2.422  |
| Garlic              | 19.008 | Swan                            | 42.036 | Helicopter            | 39.589 |
| Green Onion         | 2.359  | Sandwich                        | 0.099  | Nuts                  | 6.414  |
| Speed Limit Sign    | 10.251 | Induction Cooker                | 4.345  | Broom                 | 12.570 |
| Trombone            | 7.317  | Plum                            | 0.709  | Rickshaw              | 1.731  |
| Goldfish            | 13.434 | Kiwi fruit                      | 22.437 | Router/modem          | 9.280  |
| Poker Card          | 20.179 | Toaster                         | 41.331 | Shrimp                | 21.565 |
| Sushi               | 41.305 | Cheese                          | 16.581 | Notepaper             | 2.832  |
| Cherry              | 10.116 | Pliers                          | 14.553 | CD                    | 9.444  |
| Pasta               | 0.748  | Hammer                          | 7.437  | Cue                   | 4.810  |
| Avocado             | 25.935 | Hami melon                      | 0.710  | Flask                 | 1.655  |
| Mushroom            | 17.116 | Screwdriver                     | 9.096  | Soap                  | 13.248 |
| Recorder            | 0.389  | Bear                            | 48.347 | Eggplant              | 14.462 |
| Board Eraser        | 2.390  | Coconut                         | 30.553 | Tape Measure/ Ruler   | 7.535  |
| Pig                 | 41.139 | Showerhead                      | 17.123 | Globe                 | 32.705 |
| Chips               | 0.519  | Steak                           | 24.183 | Crosswalk Sign        | 2.413  |
| Stapler             | 11.817 | Camel                           | 52.075 | Formula 1             | 9.581  |
| Pomegranate         | 2.662  | Dishwasher                      | 30.321 | Crab                  | 10.484 |
| Hoverboard          | 0.424  | Meatball                        | 27.089 | Rice Cooker           | 11.326 |
| Tuba                | 13.843 | Calculator                      | 31.144 | Papaya                | 3.070  |
| Antelope            | 9.702  | Parrot                          | 36.740 | Seal                  | 40.781 |
| Butterfly           | 40.507 | Dumbbell                        | 4.814  | Donkey                | 35.575 |
| Lion                | 15.288 | Urinal                          | 61.871 | Dolphin               | 23.410 |
| Electric Drill      | 13.107 | Hair Dryer                      | 12.151 | Egg tart              | 5.203  |
| Jellyfish           | 32.017 | Treadmill                       | 23.506 | Lighter               | 4.264  |
| Grapefruit          | 0.182  | Game board                      | 17.860 | Mop                   | 1.486  |
| Radish              | 0.137  | Baozi                           | 8.038  | Target                | 0.183  |
| French              | 0.000  | Spring Rolls                    | 15.529 | Monkey                | 40.756 |
| Rabbit              | 24.226 | Pencil Case                     | 3.464  | Yak                   | 8.787  |
| Red Cabbage         | 6.872  | Binoculars                      | 5.735  | Asparagus             | 5.716  |
| Barbell             | 1.731  | Scallop                         | 15.723 | Noddles               | 0.951  |
| Comb                | 12.083 | Dumpling                        | 6.179  | Oyster                | 40.309 |
| Table Tennis paddle | 4.236  | Cosmetics Brush/Eyeliner Pencil | 3.148  | Chainsaw              | 3.244  |
| Eraser              | 1.717  | Lobster                         | 7.066  | Durian                | 9.555  |
| Okra                | 0.000  | Lipstick                        | 7.310  | Cosmetics Mirror      | 0.106  |
| Curling             | 50.720 | Table Tennis                    | 0.031  |                       |        |
[05/13 15:05:53 detectron2]: Evaluation results for objects365_v2_val_spotdet_v2_clip_noun_k=2 in csv format:
[05/13 15:05:53 d2.evaluation.testing]: copypaste: Task: bbox
[05/13 15:05:53 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/13 15:05:53 d2.evaluation.testing]: copypaste: 21.5611,29.6263,23.2944,8.8607,21.1313,32.5059