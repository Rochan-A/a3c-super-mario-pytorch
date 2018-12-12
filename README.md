# Reinforcement Learning for Super Mario Bros using A3C on GPU
## Modified to work with [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)

> NOTE:
This is an unofficial fork of the original code published as [a3c-super-mario-pytorch](https://github.com/ArvindSoma/a3c-super-mario-pytorch).

This project is based on the paper [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), with custom training modifications. This project was originally created for the course [Deep Learning for Computer Vision](https://vision.in.tum.de/teaching/ws2017/dl4cv) held at TUM.

<img src="video/mario-level1.gif" width="300" height="270" border="5">

## Prerequisites
- Python3.5+
- PyTorch 0.3.0+
- OpenAI Gym <=0.9.5
- opencv-python
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)

## Getting Started
The Super Mario Bros NES environment has to be set up. We are using [Kautenja's Super Mario Bros](https://github.com/Kautenja/gym-super-mario-bros) implementation for gym with some modifications to run on the current OpenAI Gym version.

## Training and Testing
To train the network from scratch, use the following command
```
python3 train-mario.py --num-processes 8
```

This command requires atleast an 8-Core system with 16GB memory and 6GB GPU memory.
You can reduce the number of processes to run on a personal system, but expect the training time to increase drastically.
```
python3 train-mario.py --num-processes 2 --non-sample 1
```

This command requires atleast a 2-Core system with 4GB memory and 2GB GPU memory.

1 test process is created with remaining train processes. Test stores data in a CSV file inside *save* folder, which can be plotted later

The training process uses random and non-random processes so that it converges faster. By default there are two non-random processes, which can be changed using args.

The random processes behaves exactly like the non-random processes when there is a clear difference in the output probabilities of the network. The non-random training processes exactly mimmic the test output, which helps train the network better.

Custom rewards are used to train the model more efficiently. They can be changed using the info dictionary or by modifying the wrappers file in *common/atari_wrappers.py*

More arguments are mentioned in the file *train-mario.py*.

## Results
After ~20 hours of training on 8 processes (7 Train, 1 Test) the game converges.

<img src="graphs/mario_train.jpeg" width="400" height="270"  border="5">

Custom rewards used:
- Time = -0.1
- Distance = +1 or 0
- Player Status = +/- 5
- Score = 2.5 x [Increase in Score]
- Done = +20 [Game Completed] or -20 [Game Incomplete]

The trained model is saved in *save/trained-models/mario_a3c_params.pkl*. Move it outside, to the *save* folder, to run the trained model.

## Repository References
This project heavily relied on:
* [ArvindSoma/a3c-super-mario-pytorch](https://github.com/ArvindSoma/a3c-super-mario-pytorch)
* [ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)