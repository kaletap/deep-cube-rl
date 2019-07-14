This project is not finished yet.

# TODO:
1. Implement saving and restoring trained neural networks
2. Implement logging
3. Actually train the neural network.

# Solving the Rubik's Cube Without Human Knowledge
https://arxiv.org/abs/1805.07470

In this article Stephen McAleer, Forest Agostinelli, Alexander Shmakov, Pierre Baldi 
describe how they approached a problem of solving a Rubik's Cube by a computer without 
supervision. General idea was to apply methods from reinforcement learning and use 
neural networks as functions approximating value of a given state as well as decision functions what move to make.
Main challenge to overcome was to account for the fact that randomly doing moves would not 
result in a solved cube even after a long time. 
That's why authors trained these networks using what they have named "Autodidactic iteration", that is starting from simple positions
(cube being only a few moves away from solved) and moving to more complicated cases when
the network is already trained.
