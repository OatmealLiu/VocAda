#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_IN-21k-COCO_SpotDet_Noun_k=3.out


CFG_PATH="./configs_detic/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

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
[05/13 14:49:05 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/13 15:04:57 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 951.95 seconds.
[05/13 15:05:29 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/13 15:07:50 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 141.88 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.220
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.298
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.090
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.329
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.195
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.429
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.480
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.287
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.624
[05/13 15:07:51 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 21.969 | 29.830 | 23.739 | 9.024 | 21.528 | 32.869 |
[05/13 15:07:58 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 61.737 | Sneakers                        | 17.062 | Chair                 | 36.681 |
| Other Shoes         | 1.108  | Hat                             | 34.310 | Car                   | 19.187 |
| Lamp                | 19.794 | Glasses                         | 25.154 | Bottle                | 21.011 |
| Desk                | 22.321 | Cup                             | 21.265 | Street Lights         | 4.786  |
| Cabinet/shelf       | 15.563 | Handbag/Satchel                 | 16.185 | Bracelet              | 5.887  |
| Plate               | 52.101 | Picture/Frame                   | 7.900  | Helmet                | 32.259 |
| Book                | 15.153 | Gloves                          | 22.533 | Storage box           | 8.671  |
| Boat                | 23.134 | Leather Shoes                   | 0.673  | Flower                | 4.794  |
| Bench               | 19.269 | Potted Plant                    | 27.003 | Bowl/Basin            | 43.810 |
| Flag                | 22.653 | Pillow                          | 43.079 | Boots                 | 18.056 |
| Vase                | 19.441 | Microphone                      | 15.798 | Necklace              | 10.621 |
| Ring                | 3.635  | SUV                             | 7.603  | Wine Glass            | 60.523 |
| Belt                | 7.954  | Monitor/TV                      | 49.947 | Backpack              | 24.187 |
| Umbrella            | 26.383 | Traffic Light                   | 35.271 | Speaker               | 32.050 |
| Watch               | 31.225 | Tie                             | 22.493 | Trash bin Can         | 36.871 |
| Slippers            | 8.958  | Bicycle                         | 42.783 | Stool                 | 33.338 |
| Barrel/bucket       | 22.183 | Van                             | 8.037  | Couch                 | 46.473 |
| Sandals             | 8.678  | Basket                          | 27.054 | Drum                  | 31.940 |
| Pen/Pencil          | 18.218 | Bus                             | 38.585 | Wild Bird             | 11.229 |
| High Heels          | 6.361  | Motorcycle                      | 24.747 | Guitar                | 44.017 |
| Carpet              | 31.261 | Cell Phone                      | 39.482 | Bread                 | 13.518 |
| Camera              | 25.354 | Canned                          | 18.219 | Truck                 | 12.325 |
| Traffic cone        | 37.017 | Cymbal                          | 35.745 | Lifesaver             | 7.351  |
| Towel               | 48.588 | Stuffed Toy                     | 26.474 | Candle                | 20.838 |
| Sailboat            | 17.254 | Laptop                          | 72.440 | Awning                | 12.674 |
| Bed                 | 53.300 | Faucet                          | 29.155 | Tent                  | 6.760  |
| Horse               | 44.759 | Mirror                          | 35.425 | Power outlet          | 16.827 |
| Sink                | 42.278 | Apple                           | 25.348 | Air Conditioner       | 14.217 |
| Knife               | 43.139 | Hockey Stick                    | 34.721 | Paddle                | 18.997 |
| Pickup Truck        | 23.989 | Fork                            | 56.463 | Traffic Sign          | 2.452  |
| Ballon              | 28.626 | Tripod                          | 5.362  | Dog                   | 57.156 |
| Spoon               | 40.601 | Clock                           | 47.792 | Pot                   | 30.237 |
| Cow                 | 19.167 | Cake                            | 12.172 | Dining Table          | 18.804 |
| Sheep               | 29.638 | Hanger                          | 3.963  | Blackboard/Whiteboard | 19.243 |
| Napkin              | 20.789 | Other Fish                      | 32.971 | Orange/Tangerine      | 8.140  |
| Toiletry            | 23.170 | Keyboard                        | 61.884 | Tomato                | 46.183 |
| Lantern             | 31.396 | Machinery Vehicle               | 12.207 | Fan                   | 16.400 |
| Green Vegetables    | 0.290  | Banana                          | 13.044 | Baseball Glove        | 41.810 |
| Airplane            | 58.728 | Mouse                           | 53.427 | Train                 | 40.879 |
| Pumpkin             | 54.874 | Soccer                          | 10.788 | Skiboard              | 2.622  |
| Luggage             | 27.378 | Nightstand                      | 22.401 | Teapot                | 21.080 |
| Telephone           | 21.200 | Trolley                         | 17.468 | Head Phone            | 28.143 |
| Sports Car          | 47.420 | Stop Sign                       | 37.275 | Dessert               | 10.740 |
| Scooter             | 19.974 | Stroller                        | 24.204 | Crane                 | 4.595  |
| Remote              | 36.941 | Refrigerator                    | 67.268 | Oven                  | 23.256 |
| Lemon               | 30.219 | Duck                            | 39.954 | Baseball Bat          | 37.491 |
| Surveillance Camera | 1.470  | Cat                             | 68.161 | Jug                   | 5.761  |
| Broccoli            | 46.264 | Piano                           | 22.454 | Pizza                 | 55.063 |
| Elephant            | 72.121 | Skateboard                      | 11.824 | Surfboard             | 48.370 |
| Gun                 | 19.929 | Skating and Skiing shoes        | 26.171 | Gas stove             | 16.853 |
| Donut               | 54.689 | Bow Tie                         | 20.899 | Carrot                | 31.590 |
| Toilet              | 74.861 | Kite                            | 47.272 | Strawberry            | 37.384 |
| Other Balls         | 7.213  | Shovel                          | 7.253  | Pepper                | 24.638 |
| Computer Box        | 4.442  | Toilet Paper                    | 33.474 | Cleaning Products     | 11.112 |
| Chopsticks          | 27.400 | Microwave                       | 64.004 | Pigeon                | 43.915 |
| Baseball            | 27.055 | Cutting/chopping Board          | 34.397 | Coffee Table          | 17.571 |
| Side Table          | 4.271  | Scissors                        | 38.699 | Marker                | 11.560 |
| Pie                 | 0.822  | Ladder                          | 21.927 | Snowboard             | 42.003 |
| Cookies             | 14.648 | Radiator                        | 35.441 | Fire Hydrant          | 35.831 |
| Basketball          | 19.651 | Zebra                           | 65.051 | Grape                 | 2.052  |
| Giraffe             | 68.125 | Potato                          | 15.249 | Sausage               | 33.506 |
| Tricycle            | 7.682  | Violin                          | 13.053 | Egg                   | 61.232 |
| Fire Extinguisher   | 33.225 | Candy                           | 1.326  | Fire Truck            | 34.117 |
| Billards            | 26.021 | Converter                       | 0.042  | Bathtub               | 55.371 |
| Wheelchair          | 42.023 | Golf Club                       | 33.646 | Briefcase             | 10.297 |
| Cucumber            | 24.296 | Cigar/Cigarette                 | 13.313 | Paint Brush           | 3.190  |
| Pear                | 13.442 | Heavy Truck                     | 10.476 | Hamburger             | 19.990 |
| Extractor           | 0.676  | Extension Cord                  | 1.287  | Tong                  | 0.014  |
| Tennis Racket       | 55.602 | Folder                          | 2.599  | American Football     | 7.863  |
| earphone            | 0.949  | Mask                            | 11.642 | Kettle                | 22.641 |
| Tennis              | 13.620 | Ship                            | 37.925 | Swing                 | 0.415  |
| Coffee Machine      | 37.503 | Slide                           | 31.939 | Carriage              | 5.592  |
| Onion               | 14.714 | Green beans                     | 5.576  | Projector             | 21.887 |
| Frisbee             | 57.741 | Washing Machine/Drying Machine  | 33.040 | Chicken               | 45.943 |
| Printer             | 47.735 | Watermelon                      | 32.879 | Saxophone             | 28.742 |
| Tissue              | 0.368  | Toothbrush                      | 35.168 | Ice cream             | 6.163  |
| Hot air balloon     | 38.656 | Cello                           | 8.471  | French Fries          | 0.124  |
| Scale               | 4.480  | Trophy                          | 23.331 | Cabbage               | 9.932  |
| Hot dog             | 8.386  | Blender                         | 42.437 | Peach                 | 22.700 |
| Rice                | 3.542  | Wallet/Purse                    | 25.370 | Volleyball            | 26.291 |
| Deer                | 46.383 | Goose                           | 15.172 | Tape                  | 16.814 |
| Tablet              | 6.586  | Cosmetics                       | 3.761  | Trumpet               | 12.814 |
| Pineapple           | 19.230 | Golf Ball                       | 18.154 | Ambulance             | 41.825 |
| Parking meter       | 17.881 | Mango                           | 0.777  | Key                   | 11.758 |
| Hurdle              | 0.024  | Fishing Rod                     | 16.074 | Medal                 | 5.885  |
| Flute               | 18.419 | Brush                           | 5.316  | Penguin               | 54.454 |
| Megaphone           | 5.247  | Corn                            | 11.204 | Lettuce               | 1.855  |
| Garlic              | 13.313 | Swan                            | 42.501 | Helicopter            | 37.979 |
| Green Onion         | 2.585  | Sandwich                        | 9.419  | Nuts                  | 7.099  |
| Speed Limit Sign    | 9.413  | Induction Cooker                | 4.574  | Broom                 | 11.620 |
| Trombone            | 3.403  | Plum                            | 0.607  | Rickshaw              | 1.191  |
| Goldfish            | 9.630  | Kiwi fruit                      | 22.943 | Router/modem          | 9.849  |
| Poker Card          | 22.791 | Toaster                         | 46.391 | Shrimp                | 19.524 |
| Sushi               | 40.118 | Cheese                          | 22.627 | Notepaper             | 2.130  |
| Cherry              | 9.127  | Pliers                          | 14.035 | CD                    | 7.211  |
| Pasta               | 1.138  | Hammer                          | 8.552  | Cue                   | 3.615  |
| Avocado             | 25.114 | Hami melon                      | 0.277  | Flask                 | 0.620  |
| Mushroom            | 16.446 | Screwdriver                     | 9.291  | Soap                  | 17.230 |
| Recorder            | 0.270  | Bear                            | 47.231 | Eggplant              | 15.479 |
| Board Eraser        | 2.299  | Coconut                         | 21.556 | Tape Measure/ Ruler   | 8.884  |
| Pig                 | 41.759 | Showerhead                      | 17.492 | Globe                 | 30.465 |
| Chips               | 0.454  | Steak                           | 21.858 | Crosswalk Sign        | 1.837  |
| Stapler             | 10.035 | Camel                           | 52.986 | Formula 1             | 9.150  |
| Pomegranate         | 3.562  | Dishwasher                      | 31.859 | Crab                  | 11.066 |
| Hoverboard          | 0.402  | Meatball                        | 27.014 | Rice Cooker           | 8.556  |
| Tuba                | 9.921  | Calculator                      | 33.906 | Papaya                | 2.599  |
| Antelope            | 9.285  | Parrot                          | 29.517 | Seal                  | 34.263 |
| Butterfly           | 40.409 | Dumbbell                        | 2.566  | Donkey                | 20.831 |
| Lion                | 8.635  | Urinal                          | 61.343 | Dolphin               | 17.748 |
| Electric Drill      | 13.373 | Hair Dryer                      | 14.900 | Egg tart              | 6.844  |
| Jellyfish           | 28.637 | Treadmill                       | 27.609 | Lighter               | 3.155  |
| Grapefruit          | 0.309  | Game board                      | 16.773 | Mop                   | 1.887  |
| Radish              | 0.236  | Baozi                           | 8.890  | Target                | 0.759  |
| French              | 0.000  | Spring Rolls                    | 15.914 | Monkey                | 39.255 |
| Rabbit              | 25.984 | Pencil Case                     | 6.919  | Yak                   | 9.763  |
| Red Cabbage         | 3.974  | Binoculars                      | 2.420  | Asparagus             | 8.462  |
| Barbell             | 0.798  | Scallop                         | 18.213 | Noddles               | 0.925  |
| Comb                | 13.603 | Dumpling                        | 3.899  | Oyster                | 39.343 |
| Table Tennis paddle | 1.016  | Cosmetics Brush/Eyeliner Pencil | 1.656  | Chainsaw              | 3.405  |
| Eraser              | 2.035  | Lobster                         | 4.081  | Durian                | 23.829 |
| Okra                | 0.000  | Lipstick                        | 7.477  | Cosmetics Mirror      | 0.550  |
| Curling             | 49.837 | Table Tennis                    | 0.021  |                       |        |
[05/13 15:09:28 detectron2]: Evaluation results for objects365_v2_val_spotdet_v2_clip_noun_k=3 in csv format:
[05/13 15:09:28 d2.evaluation.testing]: copypaste: Task: bbox
[05/13 15:09:28 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/13 15:09:28 d2.evaluation.testing]: copypaste: 21.9688,29.8301,23.7389,9.0239,21.5280,32.8692