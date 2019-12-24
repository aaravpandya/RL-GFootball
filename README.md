# Training GFootball environment

Part of my undergraduate core project. Trained GFootball environment on different tasks. Trained on 3_vs_1 first from scratch. Then trained 5_vs_3 on parameters from 3_vs_1 and scratch. While there was no difference in the time taken to achieve more than 95% performance, the transfer learned model learned passing from the previous model while the model trained from scratch opted to dribble. 
The goal of this training was to figure out if there is a way to make RL models learn concepts rather than overfit on the environment.
Also provided is a file [core.py](https://github.com/aaravpandya/RL-GFootball/blob/master/core.py) which implements self-attention layer used in the transformer architecture. This is the same block used in the starcraft paper from deepmind. However, adding this block makes the network very complex and heavy, and will require high computation power and time. We were only able to train on empty goal and with goalkeeper scenarios on this block. Future aim is to train it on 3_vs_1 and see its performance on 5_vs_3 scenarios.
## Training results

| ![3_vs_1](https://github.com/aaravpandya/RL-GFootball/blob/master/gifs/3_v_1.gif) |  
|--|
| 3_vs_1 from scratch | 

| ![5_vs_3_scratch](https://github.com/aaravpandya/RL-GFootball/blob/master/gifs/5_v_3_scratch.gif) | ![5_vs_3_transfer](https://github.com/aaravpandya/RL-GFootball/blob/master/gifs/5_v_3_transfer.gif) |
|--|--|
| 5_vs_3 from scratch | 5_vs_3 transfer learned  |


