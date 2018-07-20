# Awesome Deep Learning Theory [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

### Tutorials and other resources

#### 1. ICML 2018 Tutorial: [Toward Theoretical Understanding of Deep Learning](https://icml.cc/Conferences/2018/Schedule?showEvent=1864)

Material: [Homepage](http://unsupervised.cs.princeton.edu/deeplearningtutorial.html), [Slide](ICML%20materials/deepsurveyICML18final.pptx)

Notes:
  - Going through the role of: convexity, overparameterization, role of depth, unsupervised/GAN, simpler networks (compression)
  - Present many important phenomenons and simple theories.

#### 2. ICML Invited Talk: Max Welling - Intelligence per Kilowatthour
  Materials: [Youtube](https://www.youtube.com/watch?v=avtVbH2rdg0) (not from ICML)
  - Various views, not only on intelligence per Kilowatthour, but also philosophical question about computational world.
  
#### 3. ICML 2018 Invited Talk: Josh Tenenbaum - Building Machines that Learn and Think Like People
  Materials: [paper](https://arxiv.org/pdf/1604.00289.pdf) - 55p, [ICML Video](https://www.youtube.com/watch?v=RB78vRUO6X8)
  - His view on the importance of Probabilistic Programming Language (BayesFlow, ProbTorch, WebPPL...)
  - Symbolic is good for abstract reasoning.
  - The inuitive Physics engine in our mind (even new-born)
  - The hard problem of learning: learning program is hard (difficult loss function)
  - Big question: How to connect Symbolic representations and Neural Nets?

### Papers at ICML 2018

 #### 1. [Learning to Explain: An Information-Theoretic Perspective on Model Interpretation](http://proceedings.mlr.press/v80/chen18j.html)
  Materials: [pdf](http://proceedings.mlr.press/v80/chen18j/chen18j.pdf), [code](https://github.com/Jianbo-Lab/L2X)
 - Guided by **Michael Jordan**
 - Instancewise feature selection as a methodology for model interpretation
 - Learn a function to extract a subset of features that are most informative for each given example
 - Develop an efficient variational approximation to the mutual information
 #### 2. [Meta-Learning by Adjusting Priors Based on Extended PAC-Bayes Theory](http://proceedings.mlr.press/v80/amit18a.html)
  Materials: [pdf](http://proceedings.mlr.press/v80/amit18a/amit18a-supp.pdf), [code](https://github.com/ron-amit/meta-learning-adjusting-priors)
 - Nice combination between theory and practice
 - PAC-Bayes Bound (1999) -> PAC-Bayes Bound for Meta Learning (2014)
 - Stochastic Neural Nets with Posteriors and Priors are factorized Gaussians
   
 #### 3. [Data-Dependent Stability of Stochastic Gradient Descent](http://proceedings.mlr.press/v80/kuzborskij18a.html), 
  Materials: [pdf](http://proceedings.mlr.press/v80/kuzborskij18a/kuzborskij18a.pdf)
 - Work of **[Christoph Lampert](https://scholar.google.com/citations?user=iCf3SwgAAAAJ&hl=en)**, who develped PAC-Bayes for Meta Learning.
 - Establish a data-dependent notion of algorithmic stability for SGD
 - Develop novel generalization bounds
   
 #### 4. [Stability and Generalization of Learning Algorithms that Converge to Global Optima](http://proceedings.mlr.press/v80/charles18a.html)
 Materials: [pdf](http://proceedings.mlr.press/v80/charles18a/charles18a.pdf)
 - A Stable Learning Algorithm: Change in data doesn't affect model too much
 - Stability -> Generalization
 - Establish novel generalization bounds for learning algorithms that converge to global minima.
 - Derive black-box stability results that only depend on the convergence of a learning algorithm and the geometry around the minimizers of the empirical risk function
   
 #### 5. [An Alternative View: When Does SGD Escape Local Minima?](http://proceedings.mlr.press/v80/kleinberg18a.html)
 Materials: [pdf](http://proceedings.mlr.press/v80/kleinberg18a/kleinberg18a.pdf), [supp](http://proceedings.mlr.press/v80/kleinberg18a/kleinberg18a-supp.pdf)
 - Although being commonly viewed as a fast but not accurate version of gradient descent (GD), it always finds better solutions than GD for modern NNets. 
 - To understand this phenomenon, we take an alternative view that SGD is working on the convolved (thus smoothed) version of the loss function.
 - Even if the function f has many bad local minima or saddle points, as long as for every point x, the weighted average of the gradients of its neighborhoods is one point convex with respect to the desired solution x∗, SGD will get close to, and then stay around x∗ with constant probability
   
 #### 6. [Geometry Score: A Method For Comparing Generative Adversarial Networks](http://proceedings.mlr.press/v80/khrulkov18a.html)
   Materials: [pdf](http://proceedings.mlr.press/v80/khrulkov18a/khrulkov18a.pdf), [code](https://github.com/KhrulkovV/geometry-score)
 - Biggest challenges in the research of GANs is assessing the quality of generated samples and detecting various levels of mode collapse
 - Propose a novel measure of performance of a GAN by comparing geometrical properties of the underlying data manifold and the generated one, which provides both qualitative and quantitative means for evaluation.
   
 #### 7. [Compressing Neural Networks using the Variational Information Bottelneck](http://proceedings.mlr.press/v80/dai18d.html)
   Materials: [pdf](http://proceedings.mlr.press/v80/dai18d/dai18d.pdf), [code](https://github.com/zhuchen03/VIBNet)
   - NNets can be compressed to reduce memory and computational requirements, or to increase accuracy by facilitating the use of a larger base architecture.
   - Focus on pruning individual neurons, which can simultaneously trim model size, FLOPs, and run-time memory. 
   - Utilize the information bottleneck principle instantiated via a tractable variational bound
 #### 8. [A Spline Theory of Deep Learning](http://proceedings.mlr.press/v80/balestriero18b.html)
   Materials: [pdf](http://proceedings.mlr.press/v80/balestriero18b/balestriero18b.pdf), [supp](http://proceedings.mlr.press/v80/balestriero18b/balestriero18b-supp.pdf)
   - Build a rigorous bridge between deep networks (DNs) and approximation theory via spline functions and operators.
   - A large class of DNs can be written as a composition of max-affine spline operators (MASOs).
   - Deep Nets are (learned) matched filter banks.
 #### 9. [Neural Networks Should Be Wide Enough to Learn Disconnected Decision Regions](http://proceedings.mlr.press/v80/nguyen18b.html)
   Materials: [pdf](http://proceedings.mlr.press/v80/nguyen18b/nguyen18b.pdf), [supp](http://proceedings.mlr.press/v80/nguyen18b/nguyen18b-supp.pdf)
   - Important role of depth in deep learning has been emphasized
   - Argue that sufficient width of a feedforward network is equally important by answering the simple question under which conditions the decision regions of a neural network are connected.
   - Sufficiently wide hidden layer is necessary to guarantee that the network can produce disconnected decision regions

 #### 10. [Stronger Generalization Bounds for Deep Nets via a Compression Approach](http://proceedings.mlr.press/v80/arora18b.html)
   Materials: [pdf](http://proceedings.mlr.press/v80/arora18b/arora18b.pdf), [supp](http://proceedings.mlr.press/v80/arora18b/arora18b-supp.pdf)
   - Recent works use PAC-Bayes and Margin-based analyses, but do not as yet result in sample complexity bounds better than naive parameter counting
   - Use Noise stability -> Compress -> Better bounds
   - Data-dependent for generalization theory makes a stronger generalization bounds.

### 11. [Invariance of Weight Distributions in Rectified MLPs](http://proceedings.mlr.press/v80/tsuchida18a.html)
  Materials: [pdf](http://proceedings.mlr.press/v80/tsuchida18a/tsuchida18a.pdf), [supp](http://proceedings.mlr.press/v80/tsuchida18a/tsuchida18a-supp.pdf)
  - An interesting approach to analyzing NNets that has received renewed attention is to examine the equivalent kernel of the neural network.
  - Fully connected feedforward network with one hidden layer, a certain weight distribution, an activation function, and an infinite number of neurons can be viewed as a mapping into a Hilbert space.
  - Derive the equivalent kernels of MLPs with ReLU or Leaky ReLU activations for all rotationally-invariant weight distributions

#### 12. [Understanding Generalization and Optimization Performance of Deep CNNs](http://proceedings.mlr.press/v80/zhou18a.html)
  Materials: [pdf](http://proceedings.mlr.press/v80/zhou18a/zhou18a.pdf), [supp](http://proceedings.mlr.press/v80/zhou18a/zhou18a-supp.pdf)

 #### 13. [On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups](http://proceedings.mlr.press/v80/kondor18a.html)
  Materials: [pdf](http://proceedings.mlr.press/v80/kondor18a/kondor18a.pdf), [supp](http://proceedings.mlr.press/v80/kondor18a/kondor18a-supp.pdf)
  - CNN: quivariance: translation
  - Can we generalize to other action gorups: rotation?
  - More general notion of convolution on groups
  - Fourier analysis on groups: spherical CNNs, Steerability and CNNs for manifolds
  - NNets for Graphs: Messasage passing Neural Net.

 #### 14. [Bounding and Counting Linear Regions of Deep Neural Networks](http://proceedings.mlr.press/v80/serra18b.html)
  Materials: [pdf](http://proceedings.mlr.press/v80/serra18b/serra18b.pdf), [supp](http://proceedings.mlr.press/v80/serra18b/serra18b-supp.pdf)
  - Study the number of linear regions, i.e. pieces, that a PWL function represented by a DNN can attain, both theoretically and empirically.
  - Tighter upper and lower bounds for the maximum number of linear regions on rectifier networks.
  - Indicate that a deep rectifier network can only have more linear regions than every shallow counterpart with same number of neurons if that number exceeds the dimension of the input.
  
 #### 15. [A Theoretical Explanation for Perplexing Behaviors of Backpropagation-based Visualizations](http://proceedings.mlr.press/v80/nie18a.html)
  Materials: [pdf](http://proceedings.mlr.press/v80/nie18a/nie18a.pdf), [supp](http://proceedings.mlr.press/v80/nie18a/nie18a-supp.pdf)
  - Backpropagation-based visualizations have been proposed to interpret convolutional neural networks (CNNs), however a theory is missing to justify their behaviors.
  - Develop a theoretical explanation revealing that GBP and DeconvNet are essentially doing (partial) image recovery which is unrelated to the network decisions.

### 16. [The Mechanics of n-Player Differentiable Games](http://proceedings.mlr.press/v80/balduzzi18a.html) - Runner Up
  Materials: [pdf](http://proceedings.mlr.press/v80/balduzzi18a/balduzzi18a.pdf), [supp](http://proceedings.mlr.press/v80/balduzzi18a/balduzzi18a-supp.pdf)
  - Context: Everything in DL (CNN, LSTM, Relu, Resnet...) depends on Gradient Descent to find (good) local minima. This fails in setting as of game for GAN model where there are multiple interacting losses.
  - Problem: This paper analyzes why GD fails in multiple interacting losses games like GAN and propose a way to modify/construct loss to make GD succeed.
  - Approach: 
    - Decompose the second-order dynamics into two components. The first is related to potential games, which reduces to gradient descent on an implicit function; the second relates to Hamiltonian games, a new class of games that obey a conservation law, akin to conservation laws in **classical mechanical systems**. 
    - Propose Symplectic Gradient Adjustment (SGA), a new algorithm for finding stable fixed points in general games.
  - Result: SGA is competitive for finding Local Nash equilibria in GANs.
### Important Researchers
