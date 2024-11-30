## updata 10.23
Added a deep learning baseline AM to solve MCBLP.
You can run [AM](https://github.com/HIGISX/ReCovNet/blob/master/Comparative_experiment_with_Attention_Model.ipynb) to quickly implement AM to solve MCBLP.

# ReCovNet
This project is the code for the study: 

Zhong, Y., Wang, S., Liang, H., Wang, Z., Zhang, X., Chen, X., & Su, C. (2024). ReCovNet: Reinforcement learning with covering information for solving maximal coverage billboards location problem. International Journal of Applied Earth Observation and Geoinformation, 128, 103710.[[Full article]](https://www.sciencedirect.com/science/article/pii/S1569843224000645)

Paper reference:
```bash
@article{zhong2024recovnet,
  title={ReCovNet: Reinforcement learning with covering information for solving maximal coverage billboards location problem},
  author={Zhong, Yang and Wang, Shaohua and Liang, Haojian and Wang, Zhenbo and Zhang, Xueyan and Chen, Xi and Su, Cheng},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={128},
  pages={103710},
  year={2024},
  publisher={Elsevier}
}
```

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
