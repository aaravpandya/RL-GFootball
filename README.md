
## Build steps (for NYU HPC Greene, your mileage may vary)
1. Get a working singularity container.
2. Git clone and install "https://github.com/aaravpandya/football" (IMPORTANT: Do not use the pip package. This repo is updated with the latest gymnasium api)
3. Install .yml in the repo with conda 

	   conda env create -n ENVNAME --file ENV.yml
4. Fix this issue in your Ray installation. As of writing this README, the commit solving the issue has not been released ( "https://github.com/ray-project/ray/issues/34159")
5. Running PPO version -
            
      5.1 Run train.py with the correct arguments. Its that simple.
6. Running IMPALA version - 

      6.1 Start Ray cluster node with ` ray start --head --port=6379`

      6.2 Using the the IP address and port number in the output, start worker nodes using `ray start --address='<IP Address>:<Port Number>'` . For good results use atleast 6 nodes with 16 cores each. Have atleast 2 GPUs across the cluster.

      6.3 Run train.py with algorithm=IMPALA to start running the training. 
7. The checkpoints are stored in your checkpoint directory. Irrespective of the algorithm, you can run evaluate.py to see your model working in the env. 

      7.1 Run `unset DISPLAY` to run gfootball in HEADLESS mode before running evaluate. If you have display connected, this is not required.

# Training GFootball environment



## Training results

| ![3_vs_1](https://github.com/aaravpandya/RL-GFootball/blob/master/gifs/3_v_1.gif) |  
|--|
| 3_vs_1 from scratch | 

| ![5_vs_3_scratch](https://github.com/aaravpandya/RL-GFootball/blob/master/gifs/5_v_3_scratch.gif) | ![5_vs_3_transfer](https://github.com/aaravpandya/RL-GFootball/blob/master/gifs/5_v_3_transfer.gif) |
|--|--|
| 5_vs_3 from scratch | 5_vs_3 transfer learned  |
