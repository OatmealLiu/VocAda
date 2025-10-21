#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_IN-21k-COCO_SpotDet_Noun.out


CFG_PATH="./configs_detic/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

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
[05/13 14:58:07 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/13 15:13:03 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 896.19 seconds.
[05/13 15:13:35 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/13 15:15:53 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 138.08 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.292
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.088
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.207
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.324
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.166
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.416
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.241
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
[05/13 15:15:53 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 21.460 | 29.179 | 23.138 | 8.786 | 20.676 | 32.413 |
[05/13 15:15:59 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category                        | AP     | category              | AP     |
|:--------------------|:-------|:--------------------------------|:-------|:----------------------|:-------|
| Person              | 61.793 | Sneakers                        | 14.617 | Chair                 | 32.698 |
| Other Shoes         | 0.175  | Hat                             | 32.374 | Car                   | 17.956 |
| Lamp                | 17.678 | Glasses                         | 25.362 | Bottle                | 17.283 |
| Desk                | 17.080 | Cup                             | 0.965  | Street Lights         | 4.291  |
| Cabinet/shelf       | 15.089 | Handbag/Satchel                 | 15.342 | Bracelet              | 4.650  |
| Plate               | 51.137 | Picture/Frame                   | 7.917  | Helmet                | 33.818 |
| Book                | 15.157 | Gloves                          | 21.034 | Storage box           | 7.311  |
| Boat                | 0.990  | Leather Shoes                   | 0.555  | Flower                | 4.347  |
| Bench               | 19.134 | Potted Plant                    | 26.183 | Bowl/Basin            | 42.761 |
| Flag                | 21.680 | Pillow                          | 40.365 | Boots                 | 17.908 |
| Vase                | 18.941 | Microphone                      | 16.013 | Necklace              | 9.516  |
| Ring                | 3.175  | SUV                             | 4.346  | Wine Glass            | 59.429 |
| Belt                | 4.848  | Monitor/TV                      | 51.323 | Backpack              | 23.230 |
| Umbrella            | 26.816 | Traffic Light                   | 35.488 | Speaker               | 30.710 |
| Watch               | 17.151 | Tie                             | 18.060 | Trash bin Can         | 33.285 |
| Slippers            | 1.439  | Bicycle                         | 43.002 | Stool                 | 27.193 |
| Barrel/bucket       | 18.356 | Van                             | 8.017  | Couch                 | 45.176 |
| Sandals             | 6.470  | Basket                          | 23.300 | Drum                  | 12.926 |
| Pen/Pencil          | 16.121 | Bus                             | 37.782 | Wild Bird             | 11.387 |
| High Heels          | 8.135  | Motorcycle                      | 21.961 | Guitar                | 45.544 |
| Carpet              | 31.515 | Cell Phone                      | 38.515 | Bread                 | 12.457 |
| Camera              | 12.290 | Canned                          | 16.031 | Truck                 | 0.990  |
| Traffic cone        | 36.283 | Cymbal                          | 35.847 | Lifesaver             | 6.293  |
| Towel               | 47.677 | Stuffed Toy                     | 25.989 | Candle                | 20.168 |
| Sailboat            | 16.908 | Laptop                          | 71.887 | Awning                | 12.637 |
| Bed                 | 52.794 | Faucet                          | 25.874 | Tent                  | 7.529  |
| Horse               | 47.528 | Mirror                          | 29.049 | Power outlet          | 16.280 |
| Sink                | 41.880 | Apple                           | 25.525 | Air Conditioner       | 11.049 |
| Knife               | 41.305 | Hockey Stick                    | 39.365 | Paddle                | 17.519 |
| Pickup Truck        | 17.803 | Fork                            | 54.751 | Traffic Sign          | 2.521  |
| Ballon              | 0.949  | Tripod                          | 4.194  | Dog                   | 57.603 |
| Spoon               | 38.890 | Clock                           | 47.490 | Pot                   | 27.240 |
| Cow                 | 20.296 | Cake                            | 10.500 | Dining Table          | 18.793 |
| Sheep               | 30.906 | Hanger                          | 3.676  | Blackboard/Whiteboard | 19.439 |
| Napkin              | 20.599 | Other Fish                      | 33.510 | Orange/Tangerine      | 8.524  |
| Toiletry            | 16.701 | Keyboard                        | 61.506 | Tomato                | 28.660 |
| Lantern             | 29.325 | Machinery Vehicle               | 12.229 | Fan                   | 1.286  |
| Green Vegetables    | 0.309  | Banana                          | 12.957 | Baseball Glove        | 45.713 |
| Airplane            | 59.016 | Mouse                           | 52.892 | Train                 | 41.691 |
| Pumpkin             | 55.060 | Soccer                          | 11.076 | Skiboard              | 7.292  |
| Luggage             | 30.989 | Nightstand                      | 28.704 | Teapot                | 19.349 |
| Telephone           | 10.416 | Trolley                         | 17.606 | Head Phone            | 26.789 |
| Sports Car          | 47.821 | Stop Sign                       | 28.361 | Dessert               | 4.540  |
| Scooter             | 14.424 | Stroller                        | 32.705 | Crane                 | 4.643  |
| Remote              | 36.801 | Refrigerator                    | 66.878 | Oven                  | 23.139 |
| Lemon               | 29.600 | Duck                            | 39.461 | Baseball Bat          | 47.143 |
| Surveillance Camera | 1.114  | Cat                             | 68.786 | Jug                   | 5.008  |
| Broccoli            | 45.059 | Piano                           | 17.227 | Pizza                 | 53.849 |
| Elephant            | 72.121 | Skateboard                      | 12.164 | Surfboard             | 49.134 |
| Gun                 | 21.123 | Skating and Skiing shoes        | 22.540 | Gas stove             | 16.593 |
| Donut               | 54.783 | Bow Tie                         | 18.227 | Carrot                | 31.938 |
| Toilet              | 74.857 | Kite                            | 49.704 | Strawberry            | 37.677 |
| Other Balls         | 6.966  | Shovel                          | 5.760  | Pepper                | 23.659 |
| Computer Box        | 6.851  | Toilet Paper                    | 32.307 | Cleaning Products     | 10.096 |
| Chopsticks          | 27.413 | Microwave                       | 60.007 | Pigeon                | 44.066 |
| Baseball            | 38.301 | Cutting/chopping Board          | 33.817 | Coffee Table          | 19.604 |
| Side Table          | 4.922  | Scissors                        | 38.079 | Marker                | 11.013 |
| Pie                 | 0.990  | Ladder                          | 21.958 | Snowboard             | 43.264 |
| Cookies             | 14.680 | Radiator                        | 34.627 | Fire Hydrant          | 36.165 |
| Basketball          | 23.730 | Zebra                           | 65.099 | Grape                 | 1.950  |
| Giraffe             | 68.283 | Potato                          | 13.839 | Sausage               | 24.456 |
| Tricycle            | 5.993  | Violin                          | 13.765 | Egg                   | 55.469 |
| Fire Extinguisher   | 32.291 | Candy                           | 1.774  | Fire Truck            | 37.187 |
| Billards            | 27.576 | Converter                       | 0.330  | Bathtub               | 53.451 |
| Wheelchair          | 47.509 | Golf Club                       | 33.666 | Briefcase             | 8.835  |
| Cucumber            | 23.954 | Cigar/Cigarette                 | 13.284 | Paint Brush           | 4.062  |
| Pear                | 11.703 | Heavy Truck                     | 11.282 | Hamburger             | 22.699 |
| Extractor           | 0.033  | Extension Cord                  | 1.442  | Tong                  | 0.015  |
| Tennis Racket       | 56.010 | Folder                          | 3.856  | American Football     | 11.126 |
| earphone            | 0.902  | Mask                            | 15.490 | Kettle                | 2.152  |
| Tennis              | 0.000  | Ship                            | 38.017 | Swing                 | 0.433  |
| Coffee Machine      | 34.019 | Slide                           | 31.062 | Carriage              | 5.529  |
| Onion               | 14.020 | Green beans                     | 4.768  | Projector             | 21.795 |
| Frisbee             | 62.140 | Washing Machine/Drying Machine  | 34.055 | Chicken               | 48.430 |
| Printer             | 47.015 | Watermelon                      | 33.489 | Saxophone             | 29.601 |
| Tissue              | 0.077  | Toothbrush                      | 35.252 | Ice cream             | 6.615  |
| Hot air balloon     | 43.671 | Cello                           | 9.425  | French Fries          | 0.296  |
| Scale               | 4.085  | Trophy                          | 26.468 | Cabbage               | 7.869  |
| Hot dog             | 8.998  | Blender                         | 39.073 | Peach                 | 23.637 |
| Rice                | 3.798  | Wallet/Purse                    | 25.844 | Volleyball            | 36.148 |
| Deer                | 47.094 | Goose                           | 15.632 | Tape                  | 14.688 |
| Tablet              | 12.235 | Cosmetics                       | 3.958  | Trumpet               | 14.190 |
| Pineapple           | 19.157 | Golf Ball                       | 26.404 | Ambulance             | 52.933 |
| Parking meter       | 17.778 | Mango                           | 0.992  | Key                   | 11.094 |
| Hurdle              | 0.043  | Fishing Rod                     | 19.748 | Medal                 | 7.066  |
| Flute               | 19.195 | Brush                           | 1.113  | Penguin               | 56.189 |
| Megaphone           | 4.951  | Corn                            | 11.056 | Lettuce               | 1.726  |
| Garlic              | 10.108 | Swan                            | 45.178 | Helicopter            | 38.816 |
| Green Onion         | 2.915  | Sandwich                        | 0.000  | Nuts                  | 7.718  |
| Speed Limit Sign    | 8.997  | Induction Cooker                | 2.686  | Broom                 | 10.779 |
| Trombone            | 3.514  | Plum                            | 1.191  | Rickshaw              | 0.710  |
| Goldfish            | 13.607 | Kiwi fruit                      | 22.986 | Router/modem          | 8.682  |
| Poker Card          | 25.399 | Toaster                         | 40.730 | Shrimp                | 19.523 |
| Sushi               | 42.098 | Cheese                          | 14.355 | Notepaper             | 2.151  |
| Cherry              | 9.326  | Pliers                          | 14.439 | CD                    | 8.207  |
| Pasta               | 1.409  | Hammer                          | 7.263  | Cue                   | 4.362  |
| Avocado             | 25.116 | Hami melon                      | 0.194  | Flask                 | 0.844  |
| Mushroom            | 17.068 | Screwdriver                     | 7.868  | Soap                  | 14.507 |
| Recorder            | 0.113  | Bear                            | 51.698 | Eggplant              | 14.630 |
| Board Eraser        | 2.260  | Coconut                         | 25.841 | Tape Measure/ Ruler   | 8.889  |
| Pig                 | 41.811 | Showerhead                      | 17.184 | Globe                 | 31.640 |
| Chips               | 0.606  | Steak                           | 21.256 | Crosswalk Sign        | 2.298  |
| Stapler             | 10.387 | Camel                           | 54.273 | Formula 1             | 12.012 |
| Pomegranate         | 3.377  | Dishwasher                      | 28.381 | Crab                  | 13.331 |
| Hoverboard          | 6.638  | Meatball                        | 28.601 | Rice Cooker           | 8.394  |
| Tuba                | 9.717  | Calculator                      | 33.272 | Papaya                | 3.322  |
| Antelope            | 12.153 | Parrot                          | 33.311 | Seal                  | 43.839 |
| Butterfly           | 43.293 | Dumbbell                        | 2.774  | Donkey                | 41.332 |
| Lion                | 15.655 | Urinal                          | 61.270 | Dolphin               | 20.359 |
| Electric Drill      | 13.194 | Hair Dryer                      | 13.297 | Egg tart              | 5.659  |
| Jellyfish           | 32.669 | Treadmill                       | 27.925 | Lighter               | 2.413  |
| Grapefruit          | 0.265  | Game board                      | 23.135 | Mop                   | 0.896  |
| Radish              | 0.191  | Baozi                           | 4.883  | Target                | 0.761  |
| French              | 0.000  | Spring Rolls                    | 17.493 | Monkey                | 42.587 |
| Rabbit              | 26.672 | Pencil Case                     | 5.106  | Yak                   | 6.459  |
| Red Cabbage         | 4.836  | Binoculars                      | 4.137  | Asparagus             | 9.231  |
| Barbell             | 0.811  | Scallop                         | 10.865 | Noddles               | 1.350  |
| Comb                | 7.675  | Dumpling                        | 5.719  | Oyster                | 39.805 |
| Table Tennis paddle | 5.040  | Cosmetics Brush/Eyeliner Pencil | 4.280  | Chainsaw              | 4.020  |
| Eraser              | 1.631  | Lobster                         | 5.155  | Durian                | 28.686 |
| Okra                | 0.000  | Lipstick                        | 8.234  | Cosmetics Mirror      | 0.562  |
| Curling             | 66.818 | Table Tennis                    | 0.531  |                       |        |
[05/13 15:17:39 detectron2]: Evaluation results for objects365_v2_val_spotdet_v2_clip_noun in csv format:
[05/13 15:17:39 d2.evaluation.testing]: copypaste: Task: bbox
[05/13 15:17:39 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/13 15:17:39 d2.evaluation.testing]: copypaste: 21.4601,29.1786,23.1382,8.7860,20.6764,32.4127