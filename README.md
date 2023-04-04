# A Visual Active Search Framework for Geospatial Exploration


This repository contains implementation of our work titled as __A Visual Active Search Framework for Geospatial Exploration__. VAS proposes a reinforcement learning framework to perform Visual Active Search. 

<img src="./figures/framework.png" alt="WAMI_Positives" style="width: 200p;"/>

**PDF**: https://arxiv.org/pdf/2211.15788.pdf

**Authors**: Anindya Sarkar, Michael Lanier, Scott Alfeld, Jiarui Feng, Roman Garnett, Nathan Jacobs, Yevgeniy Vorobeychik.

-------------------------------------------------------------------------------------
## Requirements
**Frameworks**: Our implementation uses **Python3.5** and **PyTorch-v1.4.0** framework.

**Packages**: You should install prerequisites using:
```shell
  pip install -r requirements.txt
```

**Datasets**:



**xView**: You can find the instructions to download images [here](https://challenge.xviewdataset.org/data-format). After downloading the images along with **xView_train.geojson**, you need to run the following script. It will generate a csv file containing the image-path and it's corresponding grid-label sequence. Don't forget to change the directory.

```shell
  python3 Prepare_data.py
```

## Training
**Train the VAS Policy Network**


To train the policy network on different benchmarks including **xView**, **DOTA** dataset:

```shell
  python3 vas_train.py
```

Note that, vas_train.py script is used to train the vas policy with ship as target class from DOTA and 6 * 6 grid structure.
In order to train VAS in different settings as reported in the paper, modify the following:
1. Use the appropriate model class for each settings as defined in utils.py ( for example, in order to train VAS with large vehicle target class from DOTA and with 8 * 8 grid structure, use the model class defined in line 900 to line 950 in utils.py. VAS policy architecture for each setting is also defined in utils.py. We mention the setting name just above the model class definition in each settings. VAS policy architecture for all different settings we consider is defined between line 595 to line 950 in utils.py script inside utils_c folder.
2. Specify the right train/test csv file path as input for that particular setting in "get_datasetVIS" function as defined in utils.py. Provide the path of train csv file in line 381 of utils.py and test csv file in line 384 of utils.py.
3. Provide the appropriate label file for that particular settings in dataloader.py script in the dataset folder. Specifically in line 189 and in line 230.
4. Provide the appropriate value for num_actions in line 6 of constant.py. For example, in case of 6 * 6 grid structure num_actions = 36.


## Evaluate
**Test the VAS Policy Network**

To test the policy network on different benchmarks including **xView**, **DOTA** dataset:

```shell
  python3 vas_test.py
```

In order to test VAS in different settings, follow the exact same modification instructions as mentioned above for the training part.
Note that, the provided code is used to test vas in uniform query cost setting, where, we assign the cost budget in line 57. In order to test VAS in distance based query cost setting, assign the budget cost in line 79 and uncomment the lines from 95 to 103. 

We provide the trained VAS policy model parameters for different settings in the following Google Drive folder. 

**Train the Greedy Selection Policy Network**


To train the greedy selection policy network on different benchmarks including **xView**, **DOTA** dataset:

```shell
  python3 greedy_selection.py
```
To train the greedy classification network on different benchmarks including **xView**, **DOTA** dataset, run:

```shell
  python3 greedy_classification.py
```

For questions or comments, please send an e-mail to **anindyasarkar.ece@gmail.com** or use the issue tab in github.

You can cite our paper as:
```
@article{sarkar2022visual,
  title={A Visual Active Search Framework for Geospatial Exploration},
  author={Sarkar, Anindya and Lanier, Michael and Alfeld, Scott and Garnett, Roman and Jacobs, Nathan and Vorobeychik, Yevgeniy},
  journal={arXiv preprint arXiv:2211.15788},
  year={2022}
}
```
