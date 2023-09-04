# ReCovNet
This project is the code for the study: ReCovNet: Reinforcement Learning with Covering Information for Solving Maximal Coverage Billboards Location Problem.

## Quick start
You can run [GA](https://github.com/HIGISX/ReCovNet/blob/master/Billboard_MCLP_GA.ipynb) to quickly implement GA to solve MCBLP.

You can run [GUROBI](https://github.com/HIGISX/ReCovNet/blob/master/Billboard_MCLP_solver.ipynb) to quickly implement GUROBI to solve MCBLP. (Ensure that you have installed Gurobi and obtained a valid license.)

You can run [ReCovNet](https://github.com/HIGISX/ReCovNet/blob/master/Billboards_MCLP_DRL.ipynb) to quickly implement ReCovNet to solve MCBLP. (Dependencies required)

## Dependencies
```
conda create -n MCBLP python==3.7(only supported for python 3.7)
conda activate MCBLP
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install scipy
pip install networkx
pip install tensorboardX
pip install tensorboard
pip install tensorflow
pip install tqdm
```

## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.
