interesting recent papers:

  * [theory](#theory)  
  * [compute and memory architectures](#compute-and-memory-architectures)  
  * [meta-learning](#meta-learning)  
  * [one-shot learning](#one-shot-learning)  
  * [unsupervised learning](#unsupervised-learning)  
  * [generative models](#generative-models)  
    - [invertible density estimation](#generative-models---invertible-density-estimation)  
    - [generative adversarial networks](#generative-models---generative-adversarial-networks)  
    - [variational autoencoders](#generative-models---variational-autoencoders)  
    - [autoregressive models](#generative-models---autoregressive-models)  
  * [bayesian inference and learning](#bayesian-inference-and-learning)  
  * [reasoning](#reasoning)  
  * [program induction](#program-induction)  
  * [reinforcement learning](#reinforcement-learning---agents)  
    - [agents](#reinforcement-learning---agents)  
    - [exploration and intrinsic motivation](#reinforcement-learning---exploration-and-intrinsic-motivation)  
    - [hierarchical reinforcement learning](#reinforcement-learning---hierarchical-reinforcement-learning)  
    - [simulation and planning](#reinforcement-learning---simulation-and-planning)  
    - [memory](#reinforcement-learning---memory)  
    - [transfer](#reinforcement-learning---transfer)  
    - [imitation](#reinforcement-learning---imitation)  
    - [applications](#reinforcement-learning---applications)  
  * [language grounding](#language-grounding)  
  * [natural language processing](#natural-language-processing)  
  * [information retrieval](#information-retrieval)  
  * [dialog systems](#dialog-systems)  

----
interesting papers:

  - [artificial intelligence](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#interesting-papers)  
  - [knowledge representation and reasoning](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers)  
  - [machine learning](https://github.com/brylevkirill/notes/blob/master/Machine%20Learning.md#interesting-papers)  
  - [deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers)  
  - [reinforcement learning](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers)  
  - [bayesian inference and learning](https://github.com/brylevkirill/notes/blob/master/Bayesian%20Inference%20and%20Learning.txt#interesting-papers)  
  - [probabilistic programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md#interesting-papers)  
  - [natural language processing](https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.txt#interesting-papers)  
  - [information retrieval](https://github.com/brylevkirill/notes/blob/master/Information%20Retrieval.txt#interesting-papers)  
  - [dialog systems](https://github.com/brylevkirill/notes/blob/master/Personal%20Assistants.txt#interesting-papers)  



---
### theory

[Opening the Black Box of Deep Neural Networks via Information](http://arxiv.org/abs/1703.00810) (Tishby)
>	"DNNs with SGD have two phases: error minimization, then representation compression"  
>	"
>	The Information Plane provides a unique visualization of DL  
>	  - Most of the learning time goes to compression  
>	  - Layers are learnt bottom up - and "help" each other  
>	  - Layers converge to special (critical?) points on the IB bound  
>	The advantage of the layers is mostly computational  
>	  - Relaxation times are super-linear (exponential?) in the Entropy gap  
>	  - Hidden layers provide intermediate steps and boost convergence time  
>	  - Hidden layers help in avoiding critical slowing down  
>	"
  - `video` <https://youtube.com/watch?v=bLqJHjXihK8> (Tishby)
  - `video` <https://youtube.com/watch?v=ekUWO_pI2M8> (Tishby)
  - `video` <https://youtu.be/RKvS958AqGY?t=12m7s> (Tishby)
  - `notes` <https://theneuralperspective.com/2017/03/24/opening-the-black-box-of-deep-neural-networks-via-information/>
  - `notes` <https://reddit.com/r/MachineLearning/comments/60fhyb/r_opening_the_black_box_of_deep_neural_networks/df8jsbm/>
  - `press` <https://quantamagazine.org/new-theory-cracks-open-the-black-box-of-deep-learning-20170921>

[Understanding Deep Learning Requires Rethinking Generalization](http://arxiv.org/abs/1611.03530) (Google Brain)
>	"1. The effective capacity of neural networks is large enough for a brute-force memorization of the entire data set.  
>	 2. Even optimization on random labels remains easy. In fact, training time increases only by a small constant factor compared with training on the true labels.  
>	 3. Randomizing labels is solely a data transformation, leaving all other properties of the learning problem unchanged."  
>	"It is likely that learning in the traditional sense still occurs in part, but it appears to be deeply intertwined with massive memorization. Classical approaches are therefore poorly suited for reasoning about why these models generalize well."  
>
>	"Deep Learning networks are just massive associative memory stores! Deep Learning networks are capable of good generalization even when fitting random data. This is indeed strange in that many arguments for the validity of Deep Learning is on the conjecture that ‘natural’ data tends to exists in a very narrow manifold in multi-dimensional space. Random data however does not have that sort of tendency."  
>
>	"Large, unregularized deep nets outperform shallower nets with regularization."  
>	"SOTA models can fit arbitrary label patterns, even on large data-sets like ImageNet."  
>	"Popular models can fit structureless noise."  

>	"In the case of one-pass SGD, where each training point is only visited at most once, the algorithm is essentially optimizing the expected loss directly. Therefore, there is no need to define generalization. However, in practice, unless one has access to infinite training samples, one-pass SGD is rarely used. Instead, it is almost always better to run many passes of SGD over the same training set. In this case, the algorithm is optimizing the empirical loss, and the deviation between the empirical loss and the expected loss (i.e. the generalization error) needs to be controlled. In statistical learning theory, the deviation is typically controlled by restricting the complexity of the hypothesis space. For example, in binary classification, for a hypothesis space with VC-dimension d and n i.i.d. training samples, the generalization error could be upper bounded by O(sqrt(d/n)). In the distribution-free setting, the VC dimension also provide a lower bound for the generalization error. For example, if we are highly over-parameterized, i.e. d >> n, then there is a data distribution under which the generalization error could be arbitrarily bad. This worst case behavior is recently demonstrated by a randomization test on large neural networks that have the full capability of shattering the whole training set. In those experiments, zero-error minimizers for the empirical loss are found by SGD. Since the test performance could be only at the level of chance, the worst possible generalization error is observed. On the other hand, those same networks are found to generalize very well on natural image classification datasets, achieving the state-of-the-art performance on some standard benchmarks. This create a puzzle as our traditional characterization of generalization no longer readily apply in this scenario."

>	"You might assume that if you can fit each point in a random training set you would have bad generalization performance because because if your model class can do this then all of the standard learning theory bounds for generalization error are quite dire. VC theory gives generalization bounds in terms of the maximum number of points where you can achieve zero training error for every possible labelling (they call this "shattering"). Rademacher complexity gives tighter bounds, but they are in terms of the expected error of the model class over uniform random labellings of your data (the expectation is over the randomness in the random labels). If you model class is powerful enough to fit any arbitrary labelling of your data set then both of these theories give no guarantees at all about generalization error. They can't guarantee you will ever make a single correct prediction, even with infinite test samples. Obviously, experience says otherwise. Neural nets tend to generalize pretty well (often surprisingly well) in spite of the dire predictions of learning theory. That's why this result requires "rethinking generalization"; the stuff we know about generalization doesn't explain any of the success we see in practice."

  - `video` <https://facebook.com/iclr.cc/videos/1710657292296663/> (18:25) (Recht)
  - `video` <https://facebook.com/iclr.cc/videos/1710657292296663/> (53:40) (Zhang)
  - `notes` <https://theneuralperspective.com/2017/01/24/understanding-deep-learning-requires-rethinking-generalization/>
  - `notes` <https://blog.acolyer.org/2017/05/11/understanding-deep-learning-requires-re-thinking-generalization/>
  - `notes` <https://reddit.com/r/MachineLearning/comments/6ailoh/r_understanding_deep_learning_requires_rethinking/dhis1hz/>

[Deep Nets Don't Learn via Memorization](https://openreview.net/pdf?id=rJv6ZgHYg) (MILA)
>	"We use empirical methods to argue that deep neural networks do not achieve their performance by memorizing training data, in spite of overlyexpressive model architectures. Instead, they learn a simple available hypothesis that fits the finite data samples. In support of this view, we establish that there are qualitative differences when learning noise vs. natural datasets, showing that: (1) more capacity is needed to fit noise, (2) time to convergence is longer for random labels, but shorter for random inputs, and (3) DNNs trained on real data examples learn simpler functions than when trained with noise data, as measured by the sharpness of the loss function at convergence. Finally, we demonstrate that for appropriately tuned explicit regularization, e.g. dropout, we can degrade DNN training performance on noise datasets without compromising generalization on real data."

[On the Emergence of Invariance and Disentangling in Deep Representations](https://arxiv.org/abs/1706.01350)
>	"We have presented bounds, some of which tight, that connect the amount of information in the weights, the amount of information in the activations, the invariance property of the network, and the geometry of the residual loss."  
>	"This leads to the somewhat surprising result that reducing information stored in the weights about the past (dataset) results in desirable properties of the representation of future data (test datum)."  

>	"We conducted experiments to validate the assumptions underlying these bounds, and found that the results match the qualitative behavior observed on real data and architectures. In particular, the theory predicts a verifiable phase transition between an underfitting and overfitting regime for random labels, and the amount of information in nats needed to cross the transition."  
  - `video` <https://youtube.com/watch?v=BCSoRTMYQcw> (Achille)

[Capacity and Trainability in Recurrent Neural Networks](http://arxiv.org/abs/1611.09913) (Google Brain)
>	"RNNs can store an amount of task information which is linear in the number of parameters, and is approximately 5 bits per parameter.  
>	RNNs can additionally store approximately one real number from their input history per hidden unit."  

[On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)
>	"Deep networks generalise better with smaller batch-size when no other form of regularisation is used. And it may be because SGD biases learning towards flat local minima, rather than sharp local minima."  
>	"Using large batch sizes tends to find sharped minima and generalize worse. We can’t talk about generalization without taking training algorithm into account."  
  - `post` <http://inference.vc/everything-that-works-works-because-its-bayesian-2/>
  - `code` <https://github.com/keskarnitish/large-batch-training>

[The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/abs/1705.08292) (Recht)
>	"Despite the fact that our experimental evidence demonstrates that adaptive methods are not advantageous for machine learning, the Adam algorithm remains incredibly popular. We are not sure exactly as to why, but hope that our step-size tuning suggestions make it easier for practitioners to use standard stochastic gradient methods in their research. In our conversations with other researchers, we have surmised that adaptive gradient methods are particularly popular for training GANs and Q-learning with function approximation. Both of these applications stand out because they are not solving optimization problems. It is possible that the dynamics of Adam are accidentally well matched to these sorts of optimization-free iterative search procedures. It is also possible that carefully tuned stochastic gradient methods may work as well or better in both of these applications."  

[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515) (Hochreiter)
>	"While batch normalization requires explicit normalization, neuron activations of SNNs automatically converge towards zero mean and unit variance. The activation function of SNNs are "scaled exponential linear units" (SELUs), which induce self-normalizing properties. Using the Banach fixed-point theorem, we prove that activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance -- even under the presence of noise and perturbations."  
>	"For activations not close to unit variance, we prove an upper and lower bound on the variance, thus, vanishing and exploding gradients are impossible."  
>
>	"Weights are initialized in such a way that for any unit in a layer with input weights wi Σ wi = 0 and Σ wi^2 = 1."  
>	"selu(x) = λx for x>0 and selu(x) = λ(αe^x − α) for x≤0, where α≈1.6733 and λ≈1.0507"  
  - `video` <https://youtube.com/watch?v=h6eQrkkU9SA> (Hochreiter)
  - `code` <http://github.com/bioinf-jku/SNNs>

[The Shattered Gradients Problem: If resnets are the answer, then what is the question?](https://arxiv.org/abs/1702.08591)
>	"We show that the correlation between gradients in standard feedforward networks decays exponentially with depth resulting in gradients that resemble white noise."  
>	"We present a new “looks linear” (LL) initialization that prevents shattering. Preliminary experiments show the new initialization allows to train very deep networks without the addition of skip-connections."  
>	"In a randomly initialized network, the gradients of deeper layers are increasingly uncorrelated. Shattered gradients play havoc with the optimization methods currently in use and may explain the difficulty in training deep feedforward networks even when effective initialization and batch normalization are employed. Averaging gradients over minibatches becomes analogous to integrating over white noise – there is no clear trend that can be summarized in a single average direction. Shattered gradients can also introduce numerical instabilities, since small differences in the input can lead to large differences in gradients."  
>	"Skip-connections in combination with suitable rescaling reduce shattering. Specifically, we show that the rate at which correlations between gradients decays changes from exponential for feedforward architectures to sublinear for resnets. The analysis uncovers a surprising and unexpected side-effect of batch normalization."  

[Learning Deep ResNet Blocks Sequentially using Boosting Theory](https://arxiv.org/abs/1706.04964) (Schapire)
>	"We construct T weak module classifiers, each contains two of the T layers, such that the combined strong learner is a ResNet."  
>	"We introduce an alternative Deep ResNet training algorithm, which is particularly suitable in non-differentiable architectures."  

[Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730) (Liang)
>	"We use influence functions, a classic technique from robust statistics, to trace a model's prediction through the learning algorithm and back to its training data, thereby identifying training points most responsible for a given prediction."  
>	"We show that even on non-convex and non-differentiable models where the theory breaks down, approximations to influence functions can still provide valuable information."  
>	"On linear models and convolutional neural networks, we demonstrate that influence functions are useful for multiple purposes: understanding model behavior, debugging models, detecting dataset errors, and even creating visually-indistinguishable training-set attacks."  
  - `video` <https://youtube.com/watch?v=0w9fLX_T6tY> (Koh)
  - `code` <https://github.com/kohpangwei/influence-release>



---
### compute and memory architectures

[Hybrid Computing using a Neural Network with Dynamic External Memory](http://www.nature.com.sci-hub.cc/nature/journal/vaop/ncurrent/full/nature20101.html) (DeepMind)
  - `post` <https://deepmind.com/blog/differentiable-neural-computers/>
  - `video` <https://youtube.com/watch?v=steioHoiEms> (Graves)
  - `video` <https://youtube.com/watch?v=PQrlOjj8gAc> (Wayne)
  - `video` <https://youtu.be/otRoAQtc5Dk?t=59m56s> (Polykovskiy)
  - `code` <https://github.com/deepmind/dnc>
  - `code` <https://github.com/yos1up/DNC>
  - `code` <https://github.com/Mostafa-Samir/DNC-tensorflow>
  - `code` <https://github.com/frownyface/dnc>
  - `code` <https://github.com/khaotik/dnc-theano>

[Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes](http://arxiv.org/abs/1610.09027) (DeepMind)  # improved Differentiable Neural Computer

[Dynamic Neural Turing Machine with Soft and Hard Addressing Schemes](http://arxiv.org/abs/1607.00036) (Bengio)

[Hierarchical Memory Networks](http://arxiv.org/abs/1605.07427) (Bengio)

[Learning Efficient Algorithms with Hierarchical Attentive Memory](http://arxiv.org/abs/1602.03218) (DeepMind)
>	"We show that an LSTM network augmented with HAM can learn algorithms for problems like merging, sorting or binary searching from pure input-output examples."  
>	"We also show that HAM can be trained to act like classic data structures: a stack, a FIFO queue and a priority queue."  
>	"Our model may be seen as a special case of Gated Graph Neural Network"  

[Neural Random-Access Machines](http://arxiv.org/abs/1511.06392) (Sutskever)
  - `post` <http://andrew.gibiansky.com/blog/machine-learning/nram-1/> + <http://andrew.gibiansky.com/blog/machine-learning/nram-2/>
  - `code` <https://github.com/gibiansky/experiments/tree/master/nram>

----
[Associative Long Short-Term Memory](http://arxiv.org/abs/1602.03032) (Graves)
  - `video` <http://techtalks.tv/talks/associative-long-short-term-memory/62525/> (Danihelka)
  - `paper` ["Holographic Reduced Representations"](http://www.cogsci.ucsd.edu/~sereno/170/readings/06-Holographic.pdf) by Plate
  - `code` <https://github.com/mohammadpz/Associative_LSTM>

[Using Fast Weights to Attend to the Recent Past](http://arxiv.org/abs/1610.06258) (Hinton)  # alternative to LSTM
>	(Hinton) "It's a different approach to a Neural Turing Machine. It does not require any decisions about where to write stuff or where to read from. Anything that happened recently can automatically be retrieved associatively. Fast associative memory should allow neural network models of sequential human reasoning."  
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Using-Fast-Weights-to-Attend-to-the-Recent-Past> (Ba)
  - `video` <http://www.fields.utoronto.ca/talks/title-tba-337> (Hinton)
  - `video` <https://youtube.com/watch?v=mrj_hyH974o> (Novikov) (in russian)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1610.06258>
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/fast-weight-to-attend.md>
  - `notes` <https://theneuralperspective.com/2016/12/04/implementation-of-using-fast-weights-to-attend-to-the-recent-past/>
  - `notes` <https://reddit.com/r/MachineLearning/comments/58qjiw/research161006258_using_fast_weights_to_attend_to/d92kctk/>
  - `code` <https://github.com/ajarai/fast-weights>
  - `code` <https://github.com/jxwufan/AssociativeRetrieval>

----
[Gradient Episodic Memory for Continuum Learning](https://arxiv.org/abs/1706.08840) (Facebook AI Research)
  - `notes` <http://rayraycano.github.io/data%20science/tech/2017/07/31/A-Paper-a-Day-GEM.html>

[Overcoming Catastrophic Forgetting in Neural Networks](http://arxiv.org/abs/1612.00796) (DeepMind)
>	"The Mixture of Experts Layer is trained using back-propagation. The Gating Network outputs an (artificially made) sparse vector that acts as a chooser of which experts to consult. More than one expert can be consulted at once (although the paper doesn’t give any precision on the optimal number of experts). The Gating Network also decides on output weights for each expert."  
>
>	Huszar:  
>	"on-line sequential (diagonalized) Laplace approximation of Bayesian learning"  
>	"EWC makes sense for any neural network (indeed, any parametric model, really), virtually any task. Doesn't have to be DQN and in fact the paper itself shows examples with way simpler tasks."  
>	"The quadratic penalty/penalties prevent the network from forgetting what it has learnt from previous data - you can think of the quadratic penalty as a summary of the information from the data it has seen so far."  
>	"You can apply it at the level of learning tasks sequentially, or you can even apply it to on-line learning in a single task (in case you can't loop over the same minibatches several time like you do in SGD)."  
  - `paper` <http://www.pnas.org/content/early/2017/03/13/1611835114.abstract>
  - `post` <http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/main.html>
  - `post` <http://inference.vc/comment-on-overcoming-catastrophic-forgetting-in-nns-are-multiple-penalties-needed-2/>
  - `notes` <https://theneuralperspective.com/2017/04/01/overcoming-catastrophic-forgetting-in-neural-networks/>
  - `code` <https://github.com/ariseff/overcoming-catastrophic>

[Improved Multitask Learning Through Synaptic Intelligence](https://arxiv.org/abs/1703.04200)
>	"The regularization penalty is similar to EWC. However, our approach computes the per-synapse consolidation strength in an online fashion, whereas for EWC synaptic importance is computed offline after training on a designated task."  
  - `code` <https://github.com/spiglerg/TF_ContinualLearningViaSynapticIntelligence>

[PathNet: Evolution Channels Gradient Descent in Super Neural Networks](http://arxiv.org/abs/1701.08734) (DeepMind)
  - `code` <https://github.com/jaesik817/pathnet>

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Google Brain)
>	"The MoE with experts shows higher accuracy (or lower perplexity) than the state of the art using only 16% of the training time."  
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/3718d181a0fed5ed806582822ed0dbde530122bf/notes/mixture-experts.md>

----
[Adaptive Computation Time for Recurrent Neural Networks](http://arxiv.org/abs/1603.08983) (Graves)
  - `video` <https://youtu.be/tA8nRlBEVr0?t=1m26s> (Graves)
  - `video` <https://youtu.be/nqiUFc52g78?t=58m45s> (Graves)
  - `post` <http://distill.pub/2016/augmented-rnns/>
  - `notes` <https://www.evernote.com/shard/s189/sh/fd165646-b630-48b7-844c-86ad2f07fcda/c9ab960af967ef847097f21d94b0bff7>
  - `code` <https://github.com/DeNeutoy/act-tensorflow>

[Memory-Efficient Backpropagation Through Time](http://arxiv.org/abs/1606.03401) (Graves)

[Hierarchical Multiscale Recurrent Neural Networks](http://arxiv.org/abs/1609.01704) (Bengio)
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/hm-rnn.md>
  - `notes` <https://medium.com/@jimfleming/notes-on-hierarchical-multiscale-recurrent-neural-networks-7362532f3b64>

[Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences](http://arxiv.org/abs/1610.09513)
>	"If you take an LSTM and add a “time gate” that controls at what frequency to be open to new input and how long to be open each time, you can have different neurons that learn to look at a sequence with different frequencies, create a “wormhole” for gradients, save compute, and do better on long sequences and when you need to process inputs from multiple sensors that are sampled at different rates."  
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Phased-LSTM-Accelerating-Recurrent-Network-Training-for-Long-or-Event-based-Sequences> (Neil)
  - `code` <https://tensorflow.org/api_docs/python/tf/contrib/rnn/PhasedLSTMCell>
  - `code` <https://github.com/dannyneil/public_plstm>

----
[Decoupled Neural Interfaces using Synthetic Gradients](http://arxiv.org/abs/1608.05343) (DeepMind)
>	"At the very least it can allow individual modules to do gradient updates before waiting for the backward pass to reach them. So you could get better GPGPU utilization when the ordinary 'locked' mode of forward-then-backward doesn't always saturate the available compute units.  
>	Put differently, if you consider the dependency DAG of tensor operations, using these DNI things reduces the depth of the parameter gradient nodes (which is the whole point of training) in the DAG. So for example, the gradient update for the layer at the beginning of a n-layer chain goes from depth ~2n to depth ~1, the layer at the end has depth n, which doesn't change. On average, the depth of the gradient computation nodes is about 40% of what it would be normally, for deep networks. So there is a lot more flexibility for scheduling nodes in time and space.  
>	And for coarser-grained parallelism it could allow modules running on different devices to do updates before a final loss gradient is available to be distributed to all the devices. Synchronization still has to happen to update the gradient predictions, but that can happen later, and could even be opportunistic (asynchronous or stochastic)."  
>	"I guess that the synthetic gradients conditioned on the labels and the synthetic layer inputs conditioned on the data work for the same reason why stochastic depth works: during training, at any given layer the networks before and after it can be approximated by simpler, shallower versions. In stochastic depth the approximation is performed by skipping layers, so the whole network is approximated by a shallower version of itself, which changes at each step. In this work, instead, the approximation is performed by separate networks.  
  - `post` <https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/>
  - `video` <https://youtu.be/tA8nRlBEVr0?t=14m40s> + <https://youtube.com/watch?v=-u32TOPGIbQ> (Graves)
  - `video` <https://youtube.com/watch?v=toZprSCCmNI> (Gupta)
  - `post` <https://iamtrask.github.io/2017/03/21/synthetic-gradients/>
  - `notes` <http://cnichkawde.github.io/SyntheticGradients.html>

[Understanding Synthetic Gradients and Decoupled Neural Interfaces](http://arxiv.org/abs/1703.00522) (DeepMind)

[Learning to Communicate with Deep Multi-Agent Reinforcement Learning](http://arxiv.org/abs/1605.06676) (DeepMind)
  - `video` <https://youtu.be/SAcHyzMdbXc?t=19m> (de Freitas)
  - `video` <https://youtube.com/watch?v=xL-GKD49FXs> (Foerster)
  - `video` <http://videolectures.net/deeplearning2016_foerster_learning_communicate/> (Foerster)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.07133>
  - `code` <https://github.com/iassael/learning-to-communicate>

[Learning Multiagent Communication with Backpropagation](http://arxiv.org/abs/1605.07736) (Facebook AI Research)
  - `video` <https://youtu.be/SAcHyzMdbXc?t=19m> (de Freitas)
  - `video` <https://youtu.be/_iVVXWkoEAs?t=30m6s> (Fergus)
  - `code` <https://github.com/facebookresearch/CommNet>



---
### meta-learning

[Learning to Learn by Gradient Descent by Gradient Descent](http://arxiv.org/abs/1606.04474) (DeepMind)
>	"Take some computation where you usually wouldn’t keep around intermediate states, such as a planning computation (say value iteration, where you only keep your most recent estimate of the value function) or stochastic gradient descent (where you only keep around your current best estimate of the parameters). Now keep around those intermediate states as well, perhaps reifying the unrolled computation in a neural net, and take gradients to optimize the entire computation with respect to some loss function. Instances: Value Iteration Networks, Learning to learn by gradient descent by gradient descent."  
  - `video` <https://youtu.be/SAcHyzMdbXc?t=10m24s> (de Freitas)
  - `video` <https://youtu.be/x1kf4Zojtb0?t=1h4m53s> (de Freitas)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/> (27:49) (de Freitas)
  - `notes` <https://theneuralperspective.com/2017/01/04/learning-to-learn-by-gradient-descent-by-gradient-descent/>
  - `notes` <https://blog.acolyer.org/2017/01/04/learning-to-learn-by-gradient-descent-by-gradient-descent/>
  - `post` <https://hackernoon.com/learning-to-learn-by-gradient-descent-by-gradient-descent-4da2273d64f2>
  - `code` <https://github.com/deepmind/learning-to-learn>
  - `code` <https://github.com/ikostrikov/pytorch-meta-optimizer>

[Learned Optimizers that Scale and Generalize](http://arxiv.org/abs/1703.04813) (DeepMind)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/> (42:40) (de Freitas)

[Learning to Learn without Gradient Descent by Gradient Descent](https://arxiv.org/abs/1611.03824) (DeepMind)
>	"Differentiable neural computers as alternatives to parallel Bayesian optimization for hyperparameter tuning of other networks."  
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/> (52:35) (de Freitas)

[Optimization as a Model for Few-Shot Learning](https://openreview.net/forum?id=rJY0-Kcll) (Larochelle)
>	"Using LSTM meta-learner in a few-shot classification setting, where the traditional learner was a convolutional-network-based classifier. In this setting, the whole meta-learning algorithm is decomposed into two parts: the traditional learner’s initial parameters are trained to be suitable for fast gradient-based adaptation; the LSTM meta-learner is trained to be an optimization algorithm adapted for meta-learning tasks."  
>	"few-shot learning by unrolling gradient descent on small training set" 
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255/> (1:26:48) (Ravi)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/> (1:08:08) (de Freitas)

----
[RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](http://arxiv.org/abs/1611.02779) (OpenAI)
>	"MDPs encountered in real world = tiny subset of all MDPs that could be defined"  
>	"How to acquire a good prior for real-world MDPs?"  
>	"How to design algorithms that make use of such prior information?"  
>	"Key idea: learn a fast RL algorithm that make use of such prior information"  
>
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
  - `video` <http://www.fields.utoronto.ca/video-archive/2017/02/2267-16530> (19:00) (Abbeel)
  - `video` <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)
  - `video` <https://youtu.be/BskhUBPRrqE?t=6m28s> (Sutskever)
  - `video` <https://youtu.be/19eNQ1CLt5A?t=7m52s> (Sutskever)
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/RL2-Fast_Reinforcement_Learning_via_Slow_Reinforcement_Learning.md>

[Learning to Reinforcement Learn](http://arxiv.org/abs/1611.05763) (DeepMind)
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
  - `video` <https://youtu.be/Y85Zn50Eczs?t=20m18s> (Botvinick)
  - `video` <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)
  - `post` <https://hackernoon.com/learning-policies-for-learning-policies-meta-reinforcement-learning-rl²-in-tensorflow-b15b592a2ddf> (Juliani)
  - `code` <https://github.com/awjuliani/Meta-RL>

----
[Meta-Learning with Temporal Convolutions](https://arxiv.org/abs/1707.03141) (Abbeel)
>	"Most recent approaches to meta-learning are extensively hand-designed, either using architectures that are specialized to a particular application, or hard-coding algorithmic components that tell the meta-learner how to solve the task. We propose a class of simple and generic meta-learner architectures, based on temporal convolutions, that is domain-agnostic and has no particular strategy or algorithm encoded into it."  
>	"TCML architectures are nothing more than a deep stack of convolutional layers, making them simple, generic, and versatile, and the causal structure allows them to process sequential data in a sophisticated manner. RNNs also have these properties, but traditional architectures can only propagate information through time via their hidden state, and so there are fewer paths for information to flow from past to present. TCMLs do a better job of preserving the temporal structure of the input sequence; the convolutional structure offers more direct, high-bandwidth access to past information, allowing them to perform more sophisticated computation on a fixed temporal segment."  
>	"TCML is closest in spirit to [Santoro et al.](http://arxiv.org/abs/1605.06065); however, our experiments indicate that TCML outperforms such traditional RNN architectures. We can view the TCML architecture as a flavor of RNN that can remember information through the activations of the network rather than through an explicit memory module. Because of its convolutional structure, the TCML better preserves the temporal structure of the inputs it receives, at the expense of only being able to remember information for a fixed amount of time. However, by exponentially increasing the dilation factors of the higher convolutional layers, TCML architectures can tractably store information for long periods of time."  

----
[HyperNetworks](http://arxiv.org/abs/1609.09106) (Google Brain)
>	"Our main result is that hypernetworks can generate non-shared weights for LSTM and achieve near state-of-the-art results on a variety of sequence modelling tasks including character-level language modelling, handwriting generation and neural machine translation, challenging the weight-sharing paradigm for recurrent networks."  
>	"Our results also show that hypernetworks applied to convolutional networks still achieve respectable results for image recognition tasks compared to state-of-the-art baseline models while requiring fewer learnable parameters."  
  - `post` <http://blog.otoro.net/2016/09/28/hyper-networks/>

[Neural Architecture Search with Reinforcement Learning](http://arxiv.org/abs/1611.01578) (Google Brain)
  - `video` <https://youtube.com/watch?v=XDtFXBYpl1w> (Le)
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255/> (1:08:31) (Zoph)
  - `notes` <https://blog.acolyer.org/2017/05/10/neural-architecture-search-with-reinforcement-learning/>

[Designing Neural Network Architectures using Reinforcement Learning](http://arxiv.org/abs/1611.02167)
  - `code` <https://bowenbaker.github.io/metaqnn/>



---
### one-shot learning

[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) (Abbeel, Levine)
>	"Unlike prior methods, the MAML learner’s weights are updated using the gradient, rather than a learned update rule. Our method does not introduce any additional parameters into the learning process and does not require a particular learner model architecture."  
>	"MAML optimizes for a set of parameters such that when a gradient step is taken with respect to a particular task i, the parameters are close to the optimal parameters θi for task i."  
  - <https://sites.google.com/view/maml> (demo)
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=1h1m22s> + <https://youtube.com/watch?v=ID150Tl-MMw&t=1h9m10s> (Abbeel)
  - `video` <https://youtube.com/watch?v=lYU5nq0dAQQ> (Levine)
  - `post` <http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/> (Finn)
  - `code` <https://github.com/cbfinn/maml>

[Optimization as a Model for Few-Shot Learning](https://openreview.net/forum?id=rJY0-Kcll) (Larochelle)
>	"Using LSTM meta-learner in a few-shot classification setting, where the traditional learner was a convolutional-network-based classifier. In this setting, the whole meta-learning algorithm is decomposed into two parts: the traditional learner’s initial parameters are trained to be suitable for fast gradient-based adaptation; the LSTM meta-learner is trained to be an optimization algorithm adapted for meta-learning tasks."  
>	"few-shot learning by unrolling gradient descent on small training set"  
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255/> (1:26:48) (Ravi)

----
[Matching Networks for One Shot Learning](http://arxiv.org/abs/1606.04080) (DeepMind)
>	"Given just a few, or even a single, examples of an unseen class, it is possible to attain high classification accuracy on ImageNet using Matching Networks.  The core architecture is simple and straightforward to train and performant across a range of image and text classification tasks. Matching Networks are trained in the same way as they are tested: by presenting a series of instantaneous one shot learning training tasks, where each instance of the training set is fed into the network in parallel. Matching Networks are then trained to classify correctly over many different input training sets. The effect is to train a network that can classify on a novel data set without the need for a single step of gradient descent."  
  - `poster` <https://pbs.twimg.com/media/Cy7Eyh5WgAAZIw2.jpg:large>
  - `notes` <https://theneuralperspective.com/2017/01/03/matching-networks-for-one-shot-learning/>
  - `notes` <https://blog.acolyer.org/2017/01/03/matching-networks-for-one-shot-learning/>

[Learning to Remember Rare Events](http://arxiv.org/abs/1703.03129) (Google Brain)
>	"We present a large-scale life-long memory module for use in deep learning. The module exploits fast nearest-neighbor algorithms for efficiency and thus scales to large memory sizes. Except for the nearest-neighbor query, the module is fully differentiable and trained end-to-end with no extra supervision. It operates in a life-long manner, i.e., without the need to reset it during training. Our memory module can be easily added to any part of a supervised neural network. The enhanced network gains the ability to remember and do life-long one-shot learning. Our module remembers training examples shown many thousands of steps in the past and it can successfully generalize from them."  
  - `code` <https://github.com/tensorflow/models/tree/master/learning_to_remember_rare_events>

[Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)

----
[One-shot Learning with Memory-Augmented Neural Networks](http://arxiv.org/abs/1605.06065) (DeepMind)
  - `video` <http://techtalks.tv/talks/meta-learning-with-memory-augmented-neural-networks/62523/> + <https://vk.com/wall-44016343_8782> (Santoro)
  - `video` <https://youtube.com/watch?v=qos2CcviAuY> (Bartunov) (in russian)
  - `notes` <http://rylanschaeffer.github.io/content/research/one_shot_learning_with_memory_augmented_nn/main.html>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.06065>
  - `code` <https://github.com/tristandeleu/ntm-one-shot>

----
[Variational Memory Addressing in Generative Models](https://arxiv.org/abs/1709.07116) (DeepMind)
>	"Aiming to augment generative models with external memory, we interpret the output of a memory module with stochastic addressing as a conditional mixture distribution, where a read operation corresponds to sampling a discrete memory address and retrieving the corresponding content from memory. This perspective allows us to apply variational inference to memory addressing, which enables effective training of the memory module by using the target information to guide memory lookups. Stochastic addressing is particularly well-suited for generative models as it naturally encourages multimodality which is a prominent aspect of most high-dimensional datasets. Treating the chosen address as a latent variable also allows us to quantify the amount of information gained with a memory lookup and measure the contribution of the memory module to the generative process. To illustrate the advantages of this approach we incorporate it into a variational autoencoder and apply the resulting model to the task of generative few-shot learning. The intuition behind this architecture is that the memory module can pick a relevant template from memory and the continuous part of the model can concentrate on modeling remaining variations. We demonstrate empirically that our model is able to identify and access the relevant memory contents even with hundreds of unseen Omniglot characters in memory."  

[One-Shot Generalization in Deep Generative Models](http://arxiv.org/abs/1603.05106)
>	"move over DRAW: deepmind's latest has spatial-transform attention and 1-shot generalization"  
  - `video` <http://youtube.com/watch?v=TpmoQ_j3Jv4> (demo)
  - `video` <http://techtalks.tv/talks/one-shot-generalization-in-deep-generative-models/62365/>
  - `video` <https://youtu.be/XpIDCzwNe78?t=43m> (Bartunov)

[Towards a Neural Statistician](http://arxiv.org/abs/1606.02185)
  - `video` <http://techtalks.tv/talks/neural-statistician/63048/> (Edwards)
  - `video` <https://youtu.be/XpIDCzwNe78?t=51m53s> (Bartunov)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1606.02185>

[Fast Adaptation in Generative Models with Generative Matching Networks](http://arxiv.org/abs/1612.02192)
  - `video` <https://youtu.be/XpIDCzwNe78> (Bartunov) ([slides](https://bayesgroup.github.io/bmml_sem/2016/bartunov-oneshot.pdf))
  - `code` <http://github.com/sbos/gmn>

----
[Active One-shot Learning](https://cs.stanford.edu/~woodward/papers/active_one_shot_learning_2016.pdf)
  - `video` <https://youtube.com/watch?v=CzQSQ_0Z-QU> (Woodward)



---
### unsupervised learning

[Independently Controllable Features](https://arxiv.org/abs/1708.01289) (Bengio)
>	"It has been postulated that a good representation is one that disentangles the underlying explanatory factors of variation. However, it remains an open question what kind of training framework could potentially achieve that. Whereas most previous work focuses on the static setting (e.g. with images), we postulate that some of the causal factors could be discovered if the learner is allowed to interact with its environment. The agent can experiment with different actions and observe their effects. We hypothesize that some of these factors correspond to aspects of the environment which are independently controllable, i.e., that there exists a policy and a learnable feature for each such aspect of the environment, such that this policy can yield changes in that feature with minimal changes to other features that explain the statistical variations in the observed data."  
>	"In interactive environments, the temporal dependency between successive observations creates a new opportunity to notice causal structure in data which may not be apparent using only observational studies. In reinforcement learning, several approaches explore mechanisms that push the internal representations of learned models to be “good” in the sense that they provide better control, and control is a particularly important causal relationship between an agent and elements of its environment."  
>	"We propose and explore a more direct mechanism for representation learning, which explicitly links an agent’s control over its environment with its internal feature representations. Specifically, we hypothesize that some of the factors explaining variations in the data correspond to aspects of the world that can be controlled by the agent. For example, an object that could be pushed around or picked up independently of others is an independently controllable aspect of the environment. Our approach therefore aims to jointly discover a set of features (functions of the environment state) and policies (which change the state) such that each policy controls the associated feature while leaving the other features unchanged as much as possible."  
>	"Assume that there are factors of variation underlying the observations coming from an interactive environment that are independently controllable. That is, a controllable factor of variation is one for which there exists a policy which will modify that factor only, and not the others. For example, the object associated with a set of pixels could be acted on independently from other objects, which would explain variations in its pose and scale when we move it around while leaving the others generally unchanged. The object position in this case is a factor of variation. What poses a challenge for discovering and mapping such factors into computed features is the fact that the factors are not explicitly observed. Our goal is for the agent to autonomously discover such factors – which we call independently controllable features – along with policies that control them. While these may seem like strong assumptions about the nature of the environment, we argue that these assumptions are similar to regularizers, and are meant to make a difficult learning problem (that of learning good representations which disentangle underlying factors) better constrained."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/BengioTPPB17#hlarochelle> (Larochelle)

[Discovering Causal Signals in Images](https://arxiv.org/abs/1605.08179) (Facebook AI Research)
>	"First, we take a learning approach to observational causal inference, and build a classifier that achieves state-of-the-art performance on finding the causal direction between pairs of random variables, when given samples from their joint distribution. Second, we use our causal direction finder to effectively distinguish between features of objects and features of their contexts in collections of static images. Our experiments demonstrate the existence of (1) a relation between the direction of causality and the difference between objects and their contexts, and (2) observable causal signals in collections of static images."  
>	"Causal features are those that cause the presence of the object of interest in the image (that is, those features that cause the object’s class label), while anticausal features are those caused by the presence of the object in the image (that is, those features caused by the class label)."  
>	"Paper aims to verify experimentally that the higher-order statistics of image datasets can inform about causal relations. Authors conjecture that object features and anticausal features are closely related and vice-versa context features and causal features are not necessarily related. Context features give the background while object features are what it would be usually inside bounding boxes in an image dataset."  
>	"Better algorithms for causal direction should, in principle, help learning features that generalize better when the data distribution changes. Causality should help with building more robust features by awareness of the generating process of the data."  
  - `video` <https://youtube.com/watch?v=DfJeaa--xO0> (Bottou)
  - `post` <http://giorgiopatrini.org/posts/2017/09/06/in-search-of-the-missing-signals/>

----
[SCAN: Learning Abstract Hierarchical Compositional Visual Concepts](https://arxiv.org/abs/1707.03389) (DeepMind)
>	"We first use the previously published beta-VAE (Higgins et al., 2017a) architecture to learn a disentangled representation of the latent structure of the visual world, before training SCAN to extract abstract concepts grounded in such disentangled visual primitives through fast symbol association."  
  - `post` <https://deepmind.com/blog/imagine-creating-new-visual-concepts-recombining-familiar-ones/>

[beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](http://openreview.net/forum?id=Sy2fzU9gl) (DeepMind)
>	"This paper proposes a modification of the variational ELBO in encourage 'disentangled' representations, and proposes a measure of disentanglement."  
  - <http://tinyurl.com/jgbyzke> (demo)

[Early Visual Concept Learning with Unsupervised Deep Learning](http://arxiv.org/abs/1606.05579) (DeepMind)
  - `code` <https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results>

[Towards Conceptual Compression](http://arxiv.org/abs/1604.08772) (DeepMind)
  - `poster` <https://pbs.twimg.com/media/Cy3pYfWWIAA_C9h.jpg:large>

----
[Generative Temporal Models with Memory](http://arxiv.org/abs/1702.04649) (DeepMind)
>	"A sufficiently powerful temporal model should separate predictable elements of the sequence from unpredictable elements, express uncertainty about those unpredictable elements, and rapidly identify novel elements that may help to predict the future. To create such models, we introduce Generative Temporal Models augmented with external memory systems."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1702.04649>

[Learning Disentangled Representations with Semi-Supervised Deep Generative Models](http://arxiv.org/abs/1706.00400)
>	"Variational autoencoders learn representations of data by jointly training a probabilistic encoder and decoder network. Typically these models encode all features of the data into a single variable. Here we are interested in learning disentangled representations that encode distinct aspects of the data into separate variables. We propose to learn such representations using model architectures that generalize from standard VAEs, employing a general graphical model structure in the encoder and decoder. This allows us to train partially-specified models that make relatively strong assumptions about a subset of interpretable variables and rely on the flexibility of neural networks to learn representations for the remaining variables."  

[Inducing Interpretable Representations with Variational Autoencoders](http://arxiv.org/abs/1611.07492) (Goodman)

[Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders](http://arxiv.org/abs/1611.02648) (Arulkumaran)
  - `post` <http://ruishu.io/2016/12/25/gmvae/>

[Attend, Infer, Repeat: Fast Scene Understanding with Generative Models](http://arxiv.org/abs/1603.08575) (DeepMind)
  - `video` <https://youtube.com/watch?v=4tc84kKdpY4> (demo)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/EslamiHWTKH16> (Larochelle)

----
[Optimizing the Latent Space of Generative Networks](https://arxiv.org/abs/1707.05776) (Facebook AI Research)
  - `post` <https://facebook.com/yann.lecun/posts/10154646915277143>

[Unsupervised Learning by Predicting Noise](https://arxiv.org/abs/1704.05310) (Facebook AI Research)
>	"The authors give a nice analogy: it's a SOM, but instead of mapping a latent vector to each input vector, the convolutional filters are learned in order to map each input vector to a fixed latent vector. In more words: each image is assigned a unique random latent vector as the label, and the mapping from image to label is taught in a supervised manner. Every few epochs, the label assignments are adjusted (but only within batches due to computational cost), so that an image might be assigned a different latent vector label which it is already close to in 'feature space'."
  - `post` <http://inference.vc/unsupervised-learning-by-predicting-noise-an-information-maximization-view-2/>

[Disentangling Factors of Variation in Deep Representations using Adversarial Training](http://arxiv.org/abs/1611.03383) (LeCun)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1611.03383>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=conf%2Fnips%2FMathieuZZRSL16>

----
[Poincare Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039) (Facebook AI Research)
  - `notes` <https://medium.com/towards-data-science/facebook-research-just-published-an-awesome-paper-on-learning-hierarchical-representations-34e3d829ede7>
  - `notes` <https://theneuralperspective.com/2017/06/05/more-on-embeddings-spring-2017/>



---
### generative models

[A Note on the Evaluation of Generative Models](http://arxiv.org/abs/1511.01844)
  - `video` <http://videolectures.net/iclr2016_theis_generative_models/> (Theis)

[On the Quantitative Analysis of Decoder-based Generative Models](http://arxiv.org/abs/1611.04273) (Salakhutdinov)
>	"We propose to use Annealed Importance Sampling for evaluating log-likelihoods for decoder-based models and validate its accuracy using bidirectional Monte Carlo. Using this technique, we analyze the performance of decoder-based models, the effectiveness of existing log-likelihood estimators, the degree of overfitting, and the degree to which these models miss important modes of the data distribution."  
>	"This paper introduces Annealed Importance Sampling to compute tighter lower bounds and upper bounds for any generative model (with a decoder)."  
  - `video` <https://youtu.be/RZOKRFBtSh4?t=54m8s> (Wu)
  - `code` <https://github.com/tonywu95/eval_gen>



---
### generative models - invertible density estimation

[Density Estimation using Real NVP](http://arxiv.org/abs/1605.08803) (Google Brain)
>	"Real-valued Non Volume Preserving transform:"  
>	"- one-pass and exact inference and sampling"  
>	"- explicit learning of a latent representation"  
>	"- tractable log-likelihood"  
>	"- coherent and sharp samples"  
  - <https://laurent-dinh.github.io/2016/07/12/real-nvp-visualization.html> (demo)
  - `video` <https://channel9.msdn.com/events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-1> (08:19) (Dinh)
  - `video` <https://periscope.tv/hugo_larochelle/1ypKdAVmbEpGW> (Dinh)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.08803>
  - `code` <https://github.com/tensorflow/models/tree/master/real_nvp>
  - `code` <https://github.com/taesung89/real-nvp>



---
### generative models - generative adversarial networks

[NIPS 2016 Tutorial: Generative Adversarial Networks](http://arxiv.org/abs/1701.00160) (Goodfellow)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks> (Goodfellow) ([slides](http://iangoodfellow.com/slides/2016-12-04-NIPS.pdf))

----
[On Unifying Deep Generative Models](https://arxiv.org/abs/1706.00550) (Salakhutdinov)
>	"We show that GANs and VAEs are essentially minimizing KL divergences of respective posterior and inference distributions with opposite directions, extending the two learning phases of classic wake-sleep algorithm, respectively. The unified view provides a powerful tool to analyze a diverse set of existing model variants, and enables to exchange ideas across research lines in a principled way. For example, we transfer the importance weighting method in VAE literatures for improved GAN learning, and enhance VAEs with an adversarial mechanism for leveraging generated samples."  

[Variational Approaches for Auto-Encoding Generative Adversarial Networks](https://arxiv.org/abs/1706.04987) (DeepMind)
  - `video` <https://youtu.be/jAI3rBI6poU?t=1h1m33s> (Ulyanov) (in russian)

[Flow-GAN: Bridging Implicit and Prescribed Learning in Generative Models](https://arxiv.org/abs/1705.08868)
>	"generative adversarial network which allows for tractable likelihood evaluation"  
>	"Since it can be trained both adversarially (like a GAN) and in terms of MLE (like a flow model), we can quantitatively evaluate the trade-offs involved. In particular, we also consider a hybrid objective function which involves both types of losses."  
>	"The availability of quantitative metrics allow us to compare to simple baselines which essentially “remember” the training data. Our final results show that naive Gaussian Mixture Models outperforms plain WGAN on both samples quality and log-likelihood for both MNIST and CIFAR-10 which we hope will lead to new directions for both implicit and prescribed learning in generative models."  

[Deep and Hierarchical Implicit Models](http://arxiv.org/abs/1702.08896) (Blei)
>	"We develop likelihood-free variational inference (LFVI). Key to LFVI is specifying a variational family that is also implicit. This matches the model's flexibility and allows for accurate approximation of the posterior. Our work scales up implicit models to sizes previously not possible and advances their modeling design."
  - `post` <http://dustintran.com/blog/deep-and-hierarchical-implicit-models>

[Variational Inference using Implicit Distributions](http://arxiv.org/abs/1702.08235) (Huszar)
>	"This paper provides a unifying review of existing algorithms establishing connections between variational autoencoders, adversarially learned inference, operator VI, GAN-based image reconstruction, and more."  
  - `post` <http://inference.vc/variational-inference-with-implicit-probabilistic-models-part-1-2/>
  - `post` <http://inference.vc/variational-inference-with-implicit-models-part-ii-amortised-inference-2/>
  - `post` <http://inference.vc/variational-inference-using-implicit-models-part-iii-joint-contrastive-inference-ali-and-bigan/>
  - `post` <http://inference.vc/variational-inference-using-implicit-models-part-iv-denoisers-instead-of-discriminators/>

[Learning in Implicit Generative Models](http://arxiv.org/abs/1610.03483) (Mohamed)
  - `video` <https://youtu.be/RZOKRFBtSh4?t=5m37s> (Mohamed)
  - `video` <https://youtu.be/jAI3rBI6poU?t=37m56s> (Ulyanov) (in russian)
  - `post` <https://casmls.github.io/general/2017/05/24/ligm.html>

----
[Optimizing the Latent Space of Generative Networks](https://arxiv.org/abs/1707.05776) (Facebook AI Research)
>	"GAN without discriminator"  

[Good Semi-supervised Learning That Requires a Bad GAN](https://arxiv.org/abs/1705.09783) (Salakhutdinov)

[Bayesian GAN](https://arxiv.org/abs/1705.09558)
>	"In this paper, we present a simple Bayesian formulation for end-to-end unsupervised and semi-supervised learning with generative adversarial networks. Within this framework, we marginalize the posteriors over the weights of the generator and discriminator using stochastic gradient Hamiltonian Monte Carlo. We interpret data samples from the generator, showing exploration across several distinct modes in the generator weights. We also show data and iteration efficient learning of the true distribution. We also demonstrate state of the art semi-supervised learning performance on several benchmarks, including SVHN, MNIST, CIFAR-10, and CelebA. The simplicity of the proposed approach is one of its greatest strengths: inference is straightforward, interpretable, and stable. Indeed all of the experimental results were obtained without feature matching, normalization, or any ad-hoc techniques."

[A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models](https://arxiv.org/abs/1611.03852) (Abbeel, Levine)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-3> (33:17) (Levine)
  - `video` <https://youtu.be/RZOKRFBtSh4?t=10m48s> (Finn)
  - `notes` <http://pemami4911.github.io/paper-summaries/generative-adversarial-networks/2017/02/12/gans-irl-ebm.html>

----
[Do GANs Actually Learn the Distribution? An Empirical Study](https://arxiv.org/abs/1706.08224) (Arora)

[Generalization and Equilibrium in Generative Adversarial Nets](https://arxiv.org/abs/1703.00573) (Arora)
  - `video` <https://youtube.com/watch?v=V7TliSCqOwI> (Arora)
  - `post` <http://www.offconvex.org/2017/03/30/GANs2/> (Arora)

[Towards Principled Methods for Training Generative Adversarial Networks](https://arxiv.org/abs/1701.04862) (Facebook AI Research)

----
[GANs Trained by a Two Time-Scale Update Rule Converge to a Nash Equilibrium](https://arxiv.org/abs/1706.08500) (Hochreiter)
>	"We propose a two time-scale update rule (TTUR) for training GANs with stochastic gradient descent that has an individual learning rate for both the discriminator and the generator."  
>	"For the evaluation of the performance of GANs at image generation, we introduce the "Frechet Inception Distance" (FID) which captures the similarity of generated images to real ones better than the Inception Score."  
>	"In experiments, TTUR improves learning for DCGANs, improved Wasserstein GANs, and BEGANs."  
>	"to the best of our knowledge this is the first convergence proof for GANs"  
  - `video` <https://youtu.be/h6eQrkkU9SA?t=21m6s> (Hochreiter)
  - `code` <https://github.com/bioinf-jku/TTUR>

[How to Train Your DRAGAN](https://arxiv.org/abs/1705.07215)
  - `code` <https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/DRAGAN.py>

[BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717) (Google Brain)
>	"We propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based Generative Adversarial Networks. This method balances the generator and discriminator during training. Additionally, it provides a new approximate convergence measure, fast and stable training and high visual quality. We also derive a way of controlling the trade-off between image diversity and visual quality. We focus on the image generation task, setting a new milestone in visual quality, even at higher resolutions. This is achieved while using a relatively simple model architecture and a standard training procedure."  
>	"- A GAN with a simple yet robust architecture, standard training procedure with fast and stable convergence.  
>	- An equilibrium concept that balances the power of the discriminator against the generator.  
>	- A new way to control the trade-off between image diversity and visual quality.  
>	- An approximate measure of convergence. To our knowledge the only other published measure is from Wasserstein GAN."  
>	"There are still many unexplored avenues. Does the discriminator have to be an auto-encoder? Having pixel-level feedback seems to greatly help convergence, however using an auto-encoder has its drawbacks: what internal embedding size is best for a dataset? When should noise be added to the input and how much? What impact would using other varieties of auto-encoders such Variational Auto-Encoders have?"  
  - <https://pbs.twimg.com/media/C8lYiYbW0AI4_yk.jpg:large> + <https://pbs.twimg.com/media/C8c6T2kXsAAI-BN.jpg> (demo)
  - `notes` <https://blog.heuritech.com/2017/04/11/began-state-of-the-art-generation-of-faces-with-generative-adversarial-networks/>
  - `notes` <https://reddit.com/r/MachineLearning/comments/633jal/r170310717_began_boundary_equilibrium_generative/dfrktje/>
  - `code` <https://github.com/carpedm20/BEGAN-pytorch>
  - `code` <https://github.com/carpedm20/BEGAN-tensorflow>

----
[Wasserstein GAN](https://arxiv.org/abs/1701.07875) (Facebook AI Research)
>	"Paper uses Wasserstein distance instead of Jensen-Shannon divergence to compare distributions."  
>	"Paper gets rid of a few unnecessary logarithms, and clips weights."  
>
>	"Loss curves that actually make sense and reflect sample quality."  
>
>	Authors show how one can have meaningful and stable training process without having to cripple or undertrain the discriminator.  
>	Authors show why original GAN formulations (using KL/JS divergence) are problematic and provide a solution for those problems."  
>
>	"There are two fundamental problems in doing image generation using GANs: 1) model structure 2) optimization instability. This paper makes no claims of improving model structure nor does it have experiments in that direction. To improve on imagenet generation, we need some work in (1) as well."  
>
>	"Authors are not claiming that this directly improves image quality, but offers a host of other benefits like stability, the ability to make drastic architecture changes without loss of functionality, and, most importantly, a loss metric that actually appears to correlate with sample quality. That last one is a pretty big deal."  
>
>	"Using Wasserstein objective reduces instability, but we still lack proof of existence of an equilibrium. Game theory doesn’t help because we need a so-called pure equilibrium, and simple counter-examples such as rock/paper/scissors show that it doesn’t exist in general. Such counterexamples are easily turned into toy GAN scenarios with generator and discriminator having finite capacity, and the game lacks a pure equilibrium."  
>
>	"Our originality is a focus on continuous distributions with low-dimensional support and the idea to parametrize f in order to obtain a fast algorithm."  
  - `video` <https://youtube.com/watch?v=DfJeaa--xO0&t=26m27s> (Bottou)
  - `video` <https://facebook.com/iclr.cc/videos/1710657292296663/> (1:30:02) (Arjowski)
  - `post` <http://www.alexirpan.com/2017/02/22/wasserstein-gan.html>
  - `post` <https://paper.dropbox.com/doc/Wasserstein-GAN-GvU0p2V9ThzdwY3BbhoP7>
  - `post` <http://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/>
  - `code` <https://github.com/martinarjovsky/WassersteinGAN>
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/wasserstein_gan>
  - `code` <https://github.com/shekkizh/WassersteinGAN.tensorflow>
  - `code` <https://github.com/kuleshov/tf-wgan>
  - `code` <https://github.com/blei-lab/edward/blob/master/examples/gan_wasserstein.py>
  - `code` <https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/WassersteinGAN>

[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) (Facebook AI Research)
  - `post` <https://casmls.github.io/general/2017/04/13/gan.html>
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/improved_wasserstein_gan>
  - `code` <https://github.com/igul222/improved_wgan_training>

[The Cramer Distance as a Solution to Biased Wasserstein Gradients](https://arxiv.org/abs/1705.10743) (DeepMind)
  - `code` <https://github.com/jiamings/cramer-gan>

[f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](http://arxiv.org/abs/1606.00709)
>	"Shows how to optimize many different objectives using adversarial training."  
  - `video` <https://youtube.com/watch?v=I1M_jGWp5n0>
  - `video` <https://youtube.com/watch?v=kQ1eEXgGsCU> (Nowozin)
  - `video` <https://youtube.com/watch?v=y7pUN2t5LrA> (Nowozin)
  - `video` <https://youtu.be/jAI3rBI6poU?t=14m31s> (Ulyanov) (in russian)
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/f_gan>

[Improved Generator Objectives for GANs](http://arxiv.org/abs/1612.02780) (Google Brain)
>	"We present a framework to understand GAN training as alternating density ratio estimation and approximate divergence minimization. This provides an interpretation for the mismatched GAN generator and discriminator objectives often used in practice, and explains the problem of poor sample diversity. We also derive a family of generator objectives that target arbitrary f-divergences without minimizing a lower bound, and use them to train generative image models that target either improved sample quality or greater sample diversity."  

[Revisiting Classifier Two-Sample Tests for GAN Evaluation and Causal Discovery](http://arxiv.org/abs/1610.06545) (Facebook AI Research)
  - `video` <https://youtube.com/watch?v=AYlbGWLL4Eo> (Sutherland)

----
[Unrolled Generative Adversarial Networks](http://arxiv.org/abs/1611.02163)
>	"We introduce a method to stabilize GANs by defining the generator objective with respect to an unrolled optimization of the discriminator. This allows training to be adjusted between using the optimal discriminator in the generator's objective, which is ideal but infeasible in practice, and using the current value of the discriminator, which is often unstable and leads to poor solutions. We show how this technique solves the common problem of mode collapse, stabilizes training of GANs with complex recurrent generators, and increases diversity and coverage of the data distribution by the generator."  
  - `video` <https://youtu.be/RZOKRFBtSh4?t=26m16s> (Metz)
  - `code` <https://github.com/poolio/unrolled_gan>

[Improved Techniques for Training GANs](http://arxiv.org/abs/1606.03498)
>	"Our CIFAR-10 samples also look very sharp - Amazon Mechanical Turk workers can distinguish our samples from real data with an error rate of 21.3% (50% would be random guessing)"  
>	"In addition to generating pretty pictures, we introduce an approach for semi-supervised learning with GANs that involves the discriminator producing an additional output indicating the label of the input. This approach allows us to obtain state of the art results on MNIST, SVHN, and CIFAR-10 in settings with very few labeled examples. On MNIST, for example, we achieve 99.14% accuracy with only 10 labeled examples per class with a fully connected neural network — a result that’s very close to the best known results with fully supervised approaches using all 60,000 labeled examples."  
  - `video` <https://youtu.be/RZOKRFBtSh4?t=26m18s> (Metz)
  - `notes` <https://github.com/aleju/papers/blob/master/neural-nets/Improved_Techniques_for_Training_GANs.md>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2FSalimansGZCRC16>
  - `post` <http://inference.vc/understanding-minibatch-discrimination-in-gans/>
  - `code` <https://github.com/openai/improved-gan>

----
[Adversarially Regularized Autoencoders for Generating Discrete Structures](https://arxiv.org/abs/1706.04223) (LeCun)

[Maximum-Likelihood Augmented Discrete Generative Adversarial Networks](http://arxiv.org/abs/1702.07983) (Bengio)

[Boundary-Seeking Generative Adversarial Networks](http://arxiv.org/abs/1702.08431) (Bengio)
>	"This approach can be used to train a generator with discrete output when the generator outputs a parametric conditional distribution. We demonstrate the effectiveness of the proposed algorithm with discrete image data. In contrary to the proposed algorithm, we observe that the recently proposed Gumbel-Softmax technique for re-parametrizing the discrete variables does not work for training a GAN with discrete data."  
  - `post` <http://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/>
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/boundary_seeking_gan>

[GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution](http://arxiv.org/abs/1611.04051)
  - `post` <https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html>

----
[Generative Adversarial Text to Image Synthesis](http://arxiv.org/abs/1605.05396)

[Neural Photo Editing with Introspective Adversarial Networks](http://arxiv.org/abs/1609.07093)
  - `video` <https://youtube.com/watch?v=FDELBFSeqQs> (demo)
  - `code` <https://github.com/ajbrock/Neural-Photo-Editor>

[Conditional Image Synthesis With Auxiliary Classifier GANs](http://arxiv.org/abs/1610.09585) (Google Brain)
  - <https://pbs.twimg.com/media/CwM0BzjVUAAWTn4.jpg:large>
  - `video` <https://youtu.be/RZOKRFBtSh4?t=21m47s> (Odena)
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/auxiliary_classifier_gan>

[Stacked Generative Adversarial Networks](http://arxiv.org/abs/1612.04357)
  - `code` <https://github.com/xunhuang1995/SGAN>

[Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space](http://www.evolvingai.org/files/nguyen2016ppgn__v1.pdf)
  - <https://pbs.twimg.com/media/Czpn0VLVEAA_RpK.jpg:large> (demo)
  - `code` <https://github.com/Evolving-AI-Lab/ppgn>

[Learning from Simulated and Unsupervised Images through Adversarial Training](http://arxiv.org/abs/1612.07828) (Apple)
  - `video` <https://youtube.com/watch?v=P3ayMdNdokg> (Shrivastava)
  - `code` <https://github.com/carpedm20/simulated-unsupervised-tensorflow>

[Unsupervised Pixel-Level Domain Adaptation with Generative Asversarial Networks](http://arxiv.org/abs/1612.05424) (Google Brain)
  - `video` <https://youtube.com/watch?v=VhsTrWPvjcA> (Bousmalis)

[Unsupervised Image-to-Image Translation Networks](http://arxiv.org/abs/1703.00848) (NVIDIA)

[Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192)
  - `code` <https://github.com/SKTBrain/DiscoGAN>

[DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/abs/1704.02510)
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/dual_gan>



---
### generative models - variational autoencoders

[Towards a Deeper Understanding of Variational Autoencoding Models](http://arxiv.org/abs/1702.08658)
>	"We provide a formal explanation for why VAEs generate blurry samples when trained on complex natural images. We show that under some conditions, blurry samples are not caused by the use of a maximum likelihood approach as previously thought, but rather they are caused by an inappropriate choice for the inference distribution. We specifically target this problem by proposing a sequential VAE model, where we gradually augment the the expressiveness of the inference distribution using a process inspired by the recent infusion training process. As a result, we are able to generate sharp samples on the LSUN bedroom dataset, even using 2-norm reconstruction loss in pixel space."  
>
>	"We propose a new explanation of the VAE tendency to ignore the latent code. We show that this problem is specific to the original VAE objective function and does not apply to the more general family of VAE models we propose. We show experimentally that using our more general framework, we achieve comparable sample quality as the original VAE, while at the same time learning meaningful features through the latent code, even when the decoder is a powerful PixelCNN that can by itself model data."  

[Variational Lossy Autoencoder](http://arxiv.org/abs/1611.02731) (OpenAI)

[Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks](http://arxiv.org/abs/1701.04722)
  - `video` <https://youtu.be/y7pUN2t5LrA?t=14m19s> (Nowozin)
  - `video` <https://youtu.be/m80Vp-jz-Io?t=1h28m34s> (Tolstikhin)
  - `post` <http://inference.vc/variational-inference-with-implicit-models-part-ii-amortised-inference-2/>
  - `code` <https://github.com/wiseodd/generative-models/tree/master/VAE/adversarial_vb>
  - `code` <https://gist.github.com/poolio/b71eb943d6537d01f46e7b20e9225149>

----
[The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](http://arxiv.org/abs/1611.00712) (DeepMind)  
[Categorical Reparametrization with Gumbel-Softmax](http://arxiv.org/abs/1611.01144) (Google Brain)
  - `video` <http://youtube.com/watch?v=JFgXEbgcT7g> (Jang)
  - `video` <https://youtu.be/_XRBlhzb31U?t=28m33s> (Figurnov) (in russian)
  - `post` <https://laurent-dinh.github.io/2016/11/21/gumbel-max.html>
  - `post` <https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html>
  - `post` <http://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/>
  - `post` <https://cmaddis.github.io/gumbel-machinery>
  - `post` <https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/>
  - `code` <https://github.com/ericjang/gumbel-softmax/blob/master/gumbel_softmax_vae_v2.ipynb>
  - `code` <https://gist.github.com/gngdb/ef1999ce3a8e0c5cc2ed35f488e19748>
  - `code` <https://github.com/EderSantana/gumbel>

[REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models](http://arxiv.org/abs/1703.07370) (Google Brain + DeepMind)
>	"Learning in models with discrete latent variables is challenging due to high variance gradient estimators. Generally, approaches have relied on control variates to reduce the variance of the REINFORCE estimator. Recent work (Jang et al. 2016; Maddison et al. 2016) has taken a different approach, introducing a continuous relaxation of discrete variables to produce low-variance, but biased, gradient estimates. In this work, we combine the two approaches through a novel control variate that produces low-variance, unbiased gradient estimates."  
  - `code` <https://github.com/tensorflow/models/tree/master/research/rebar>

----
[Stochastic Backpropagation through Mixture Density Distributions](http://arxiv.org/abs/1607.05690) (DeepMind)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1607.05690>

----
[Variational Boosting: Iteratively Refining Posterior Approximations](http://arxiv.org/abs/1611.06585) (Adams)
  - `post` <http://andymiller.github.io/2016/11/23/vb.html>
  - `video` <https://youtu.be/Jh3D8Gi4N0I?t=1h9m52s> (Nekludov) (in russian)

[Improving Variational Inference with Inverse Autoregressive Flow](http://arxiv.org/abs/1606.04934) (OpenAI)
>	"Most VAEs have so far been trained using crude approximate posteriors, where every latent variable is independent. Normalizing Flows have addressed this problem by conditioning each latent variable on the others before it in a chain, but this is computationally inefficient due to the introduced sequential dependencies. The core contribution of this work, termed inverse autoregressive flow (IAF), is a new approach that, unlike previous work, allows us to parallelize the computation of rich approximate posteriors, and make them almost arbitrarily flexible."  
  - `code` <https://github.com/openai/iaf>

[Importance Weighted Autoencoders](http://arxiv.org/abs/1509.00519) (Salakhutdinov)
  - `post` <http://dustintran.com/blog/importance-weighted-autoencoders/>
  - `code` <https://github.com/yburda/iwae>
  - `code` <https://github.com/arahuja/generative-tf>
  - `code` <https://github.com/blei-lab/edward/blob/master/examples/iwvi.py>

----
[Auxiliary Deep Generative Models](http://arxiv.org/abs/1602.05473)
  - `video` <http://techtalks.tv/talks/auxiliary-deep-generative-models/62509/> (Maaløe)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Variational-Inference-Foundations-and-Modern-Methods> (1:23:13) (Mohamed)
  - `code` <https://github.com/larsmaaloee/auxiliary-deep-generative-models>

[Hierarchical Variational Models](https://arxiv.org/pdf/1511.02386) (Blei)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Variational-Inference-Foundations-and-Modern-Methods> (1:23:13) (Mohamed)

----
[Composing Graphical Models with Neural Networks for Structured Representations and Fast Inference](http://arxiv.org/abs/1603.06277)
  - `video` <https://youtube.com/watch?v=btr1poCYIzw>
  - `video` <http://videolectures.net/deeplearning2017_johnson_graphical_models/> (Johnson)
  - `video` <https://youtube.com/watch?v=vnO3w8OgTE8> (Duvenaud)
  - `slides` <http://www.cs.toronto.edu/~duvenaud/courses/csc2541/slides/svae-slides.pdf>
  - `code` <https://github.com/mattjj/svae>

----
[Rejection Sampling Variational Inference](http://arxiv.org/abs/1610.05683)

[The Generalized Reparameterization Gradient](http://arxiv.org/abs/1610.02287)
  - `video` <https://youtu.be/mrj_hyH974o?t=1h23m40s> (Vetrov) (in russian)

[The Variational Fair Autoencoder](http://arxiv.org/abs/1511.00830)
  - `video` <http://videolectures.net/iclr2016_louizos_fair_autoencoder/> (Louizos)

[The Variational Gaussian Process](http://arxiv.org/abs/1511.06499)
  - `video` <http://videolectures.net/iclr2016_tran_variational_gaussian/> (Tran)
  - `code` <http://github.com/blei-lab/edward>

[Stick-Breaking Variational Autoencoders](http://arxiv.org/abs/1605.06197)
>	latent representation with stochastic dimensionality  
  - `code` <https://github.com/enalisnick/stick-breaking_dgms>

----
[Actively Learning What Makes a Discrete Sequence Valid](https://arxiv.org/abs/1708.04465)

[Grammar Variational Autoencoder](http://arxiv.org/abs/1703.01925)
  - `video` <https://youtube.com/watch?v=XkY1z6kCY_s> (Hernandez-Lobato)
  - `video` <https://youtube.com/watch?v=ar4Fm1V65Fw> (Paige)

[Generative Models of Visually Grounded Imagination](https://arxiv.org/abs/1705.10762) (Google)
  - `video` <https://youtu.be/IyP1pxgM_eE?t=1h5m14s> (Murphy)



---
### generative models - autoregressive models

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Google Brain)  # Transformer
>	"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."  
  - `post` <https://research.googleblog.com/2017/08/transformer-novel-neural-network.html>
  - `video` <https://youtu.be/_XRBlhzb31U?t=48m35s> (Figurnov) (in russian)
  - `post` <https://machinethoughts.wordpress.com/2017/09/01/deep-meaning-beyond-thought-vectors/>
  - `notes` <https://medium.com/@sharaf/a-paper-a-day-24-attention-is-all-you-need-26eb2da90a91>
  - `code` <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py>

----
[Pixel Recurrent Neural Networks](http://arxiv.org/abs/1601.06759) (DeepMind)
  - `video` <http://techtalks.tv/talks/pixel-recurrent-neural-networks/62375/> (van den Oord)
  - `post` <https://github.com/tensorflow/magenta/blob/master/magenta/reviews/pixelrnn.md> (Kastner)
  - `notes` <https://evernote.com/shard/s189/sh/fdf61a28-f4b6-491b-bef1-f3e148185b18/aba21367d1b3730d9334ed91d3250848> (Larochelle)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2FOordKK16#shagunsodhani>
  - `code` <https://github.com/carpedm20/pixel-rnn-tensorflow>

[Conditional Image Generation with PixelCNN Decoders](http://arxiv.org/abs/1606.05328) (DeepMind)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-1> (27:26) (van den Oord)
  - `post` <http://sergeiturukin.com/2017/02/22/pixelcnn.html> + <http://sergeiturukin.com/2017/02/24/gated-pixelcnn.html>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1606.05328#shagunsodhani>
  - `code` <https://github.com/openai/pixel-cnn>
  - `code` <https://github.com/anantzoid/Conditional-PixelCNN-decoder>

[PixelCNN++: A PixelCNN Implementation with Discretized Logistic Mixture Likelihood and Other Modifications](https://arxiv.org/abs/1701.05517) (OpenAI)
  - `code` <https://github.com/openai/pixel-cnn>

[Parallel Multiscale Autoregressive Density Estimation](http://arxiv.org/abs/1703.03664) (DeepMind)
>	"O(log N) sampling instead of O(N)"  

----
[WaveNet: A Generative Model for Raw Audio](http://arxiv.org/abs/1609.03499) (DeepMind)
  - `post` <https://deepmind.com/blog/wavenet-generative-model-raw-audio/>
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-1> (42:36) (van den Oord)
  - `code` <https://github.com/tomlepaine/fast-wavenet>
  - `code` <https://github.com/ibab/tensorflow-wavenet>
  - `code` <https://github.com/basveeling/wavenet>

[Neural Machine Translation in Linear Time](http://arxiv.org/abs/1610.10099) (ByteNet) (DeepMind)
>	"Generalizes LSTM seq2seq by preserving the resolution. Dynamic unfolding instead of attention. Linear time computation."  
>
>	"The authors apply a WaveNet-like architecture to the task of Machine Translation. Encoder (Source Network) and Decoder (Target Network) are CNNs that use Dilated Convolutions and they are stacked on top of each other. The Target Network uses Masked Convolutions to ensure that it only relies on information from the past. Crucially, the time complexity of the network is c(|S| + |T|), which is cheaper than that of the common seq2seq attention architecture (|S|*|T|). Through dilated convolutions the network has constant path lengths between [source input -> target output] and [target inputs -> target output] nodes. This allows for efficient propagation of gradients."  
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/nmt-linear-time.md>
  - `code` <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/bytenet.py>

[Language Modeling with Gated Convolutional Networks](http://arxiv.org/abs/1612.08083) (Facebook AI Research)  # outperforming LSTM on language modelling
  - `code` <https://github.com/DingKe/nn_playground/tree/master/gcnn>

----
[Tuning Recurrent Neural Networks with Reinforcement Learning](http://arxiv.org/abs/1611.02796) (Google Brain)
>	"In contrast to relying solely on possibly biased data, our approach allows for encoding high-level domain knowledge into the RNN, providing a general, alternative tool for training sequence models."  
  - `post` <https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning/>
  - `video` <https://www.technologyreview.com/s/604010/google-brain-wants-creative-ai-to-help-humans-make-a-new-kind-of-art/> (10:45) (Eck)
  - `code` <https://github.com/tensorflow/magenta/tree/master/magenta/models/rl_tuner>

[Sequence Tutor: Conservative Fine-tuning of Sequence Generation Models with KL-control](https://research.google.com/pubs/pub46118.html) (Google Brain)
  - `video` <https://1drv.ms/v/s!AhXHs6UCFoepqqZoYlaPyYH53X38TQ> (Jaques)

[Learning to Decode for Future Success](http://arxiv.org/abs/1701.06549) (Stanford)

----
[An Actor-Critic Algorithm for Sequence Prediction](http://arxiv.org/abs/1607.07086) (Bengio)

[Professor Forcing: A New Algorithm for Training Recurrent Networks](http://arxiv.org/abs/1610.09038)
>	"In professor forcing, G is simply an RNN that is trained to predict the next element in a sequence and D a discriminative bi-directional RNN. G is trained to fool D into thinking that the hidden states of G occupy the same state space at training (feeding ground truth inputs to the RNN) and inference time (feeding generated outputs as the next inputs). D, in turn, is trained to tell apart the hidden states of G at training and inference time. At the Nash equilibrium, D cannot tell apart the state spaces any better and G cannot make them any more similar. This is motivated by the problem that RNNs typically diverge to regions of the state space that were never observed during training and which are hence difficult to generalize to."  
  - `video` <https://youtube.com/watch?v=I7UFPBDLDIk>
  - `video` <http://videolectures.net/deeplearning2016_goyal_new_algorithm/> (Goyal)
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/professor-forcing.md>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1610.09038>
  - `code` <https://github.com/anirudh9119/LM_GANS>

[Self-critical Sequence Training for Image Captioning](http://arxiv.org/abs/1612.00563)
>	"REINFORCE with reward normalization but without baseline estimation"  
  - `video` <https://yadi.sk/i/-U5w4NpJ3H5TWD> + <https://yadi.sk/i/W3N7-6is3H5TWN> (Ratnikov) (in russian)

----
[Sequence-to-Sequence Learning as Beam-Search Optimization](http://arxiv.org/abs/1606.02960)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-2> (44:02) (Wiseman)
  - `video` <https://periscope.tv/hugo_larochelle/1eaKbLQXbWdJX> (31:19) (Rush)
  - `notes` <http://shortscience.org/paper?bibtexKey=journals/corr/1606.02960>
  - `notes` <https://medium.com/@sharaf/a-paper-a-day-2-sequence-to-sequence-learning-as-beam-search-optimization-92424b490350>

[Length Bias in Encoder Decoder Models and a Case for Global Conditioning](http://arxiv.org/abs/1606.03402) (Google)  # eliminating beam search

----
[Order Matters: Sequence to Sequence for Sets](http://arxiv.org/abs/1511.06391) (Google Brain)
  - `video` <https://youtube.com/watch?v=uohtFXD_39c&t=56m51s> (Bengio)



---
### bayesian inference and learning

[Stochastic Gradient Descent as Approximate Bayesian Inference](https://arxiv.org/abs/1704.04289) (Blei)
  - `notes` <https://reddit.com/r/MachineLearning/comments/6d7nb1/d_machine_learning_wayr_what_are_you_reading_week/dihh54a/>

----
[What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)
>	"We presented a novel Bayesian deep learning framework to learn a mapping to aleatoric uncertainty from the input data, which is composed on top of epistemic uncertainty models. We derived our framework for both regression and classification applications.  
>	We showed that it is important to model epistemic uncertainty for:  
	- Safety-critical applications, because epistemic uncertainty is required to understand examples which are different from training data  
	- Small datasets where the training data is sparse.  
	"And aleatoric uncertainty is important for:  
	- Large data situations, where epistemic uncertainty is explained away  
	- Real-time applications, because we can form aleatoric models without expensive Monte Carlo samples.  
	"We can actually divide aleatoric into two further sub-categories:  
	- Data-dependant or Heteroscedastic uncertainty is aleatoric uncertainty which depends on the input data and is predicted as a model output.  
	- Task-dependant or Homoscedastic uncertainty is aleatoric uncertainty which is not dependant on the input data. It is not a model output, rather it is a quantity which stays constant for all input data and varies between different tasks. It can therefore be described as task-dependant uncertainty."  
>	"However aleatoric and epistemic uncertainty models are not mutually exclusive. We showed that the combination is able to achieve new state-of-the-art results on depth regression and semantic segmentation benchmarks."  
  - `post` <https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/> (Kendall)

[Uncertainty Decomposition in Bayesian Neural Networks with Latent Variables](https://arxiv.org/abs/1706.08495)
>	"We can identify two distinct forms of uncertainties in the class of models given by BNNs with latent variables. Aleatoric uncertainty captures noise inherent in the observations. On the other hand, epistemic uncertainty accounts for uncertainty in the model. In particular, epistemic uncertainty arises from our lack of knowledge of the values of the synaptic weights in the network, whereas aleatoric uncertainty originates from our lack of knowledge of the value of the latent variables. In the domain of model-based RL the epistemic uncertainty is the source of model bias. When there is high discrepancy between model and real-world dynamics, policy behavior may deteriorate. In analogy to the principle that ”a chain is only as strong as its weakest link” a drastic error in estimating the ground truth MDP at a single transition stepcan render the complete policy useless. In this work we address the decomposition of the uncertainty present in the predictions of BNNs with latent variables into its epistemic and aleatoric components."  
>	"We derive an information-theoretic objective that decomposes the entropy of the predictive distribution of BNNs with latent variables into its epistemic and aleatoric components. By building on that decomposition, we then investigate safe RL using a risk-sensitive criterion which focuses only on risk related to model bias, that is, the risk of the policy performing at test time significantly different from at training time. The proposed criterion quantifies the amount of epistemic uncertainty (model bias risk) in the model’s predictive distribution and ignores any risk stemming from the aleatoric uncertainty."  

[Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](http://arxiv.org/abs/1612.01474) (DeepMind)
>	"Adversarial Training to improve the uncertainty measure of the entropy score of the neural network."  

[Dropout Inference in Bayesian Neural Networks with Alpha-divergences](http://mlg.eng.cam.ac.uk/yarin/PDFs/LiGal2017.pdf)
>	"We demonstrate improved uncertainty estimates and accuracy compared to VI in dropout networks. We study our model’s epistemic uncertainty far away from the data using adversarial images, showing that these can be distinguished from non-adversarial images by examining our model’s uncertainty."

----
[Model Selection in Bayesian Neural Networks via Horseshoe Priors](https://arxiv.org/abs/1705.10388)
>	"Model selection - even choosing the number of nodes - remains an open question. In this work, we apply a horseshoe prior over node pre-activations of a Bayesian neural network, which effectively turns off nodes that do not help explain the data. We demonstrate that our prior prevents the BNN from underfitting even when the number of nodes required is grossly over-estimated. Moreover, this model selection over the number of nodes doesn’t come at the expense of predictive or computational performance; in fact, we learn smaller networks with comparable predictive performance to current approaches."  

[Structured Bayesian Pruning via Log-Normal Multiplicative Noise](https://arxiv.org/abs/1705.07283)
  - `video` <https://youtu.be/jJDVYAxyE3U?t=32m45s> (Molchanov) (in russian)

[Variational Dropout Sparsifies Deep Neural Networks](https://arxiv.org/abs/1701.05369)
>	"Interpretation of Gaussian dropout as performing variational inference in a network with log uniform priors over weights leads to sparsity in weights. This is an interesting approach, wherein sparsity stemsfrom variational optimization instead of the prior."  
  - `video` <https://youtube.com/watch?v=jJDVYAxyE3U> (Molchanov) (in russian)
  - `code` <https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn>

----
[Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798) (DeepMind)
  - `code` <https://github.com/DeNeutoy/bayesian-rnn>

[Sequential Neural Models with Stochastic Layers](http://arxiv.org/abs/1605.07571)
>	"stochastic neural networks:  
>	- allow to learn one-to-many type of mappings  
>	- can be used in structured prediction problems where modeling the internal structure of the output is important  
>	- benefit from stochasticity as regularizer which makes generalization performance potentially better in general"  
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Sequential-Neural-Models-with-Stochastic-Layers> (Fraccaro)
  - `video` <https://youtu.be/mrj_hyH974o?t=32m49s> (in russian)
  - `code` <https://github.com/marcofraccaro/srnn>

[DISCO Nets: DISsimilarity COefficient Networks](http://arxiv.org/abs/1606.02556)
  - `video` <https://youtube.com/watch?v=OogNSKRkoes> 
  - `video` <https://youtube.com/watch?v=LUex45H4YXI> (Bouchacourt)

----
[Deep Probabilistic Programming](http://arxiv.org/abs/1701.03757) (Blei)
  - `code` <http://edwardlib.org/iclr2017>
  - `code` <http://edwardlib.org/zoo>

[Deep Amortized Inference for Probabilistic Programs](http://arxiv.org/abs/1610.05735) (Goodman)

[Inference Compilation and Universal Probabilistic Programming](http://arxiv.org/abs/1610.09900) (Wood)



---
### reasoning

[Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks](https://arxiv.org/abs/1704.08384) (McCallum)
  - `video` <https://youtu.be/lc68_d_DnYs?t=7m28s> (Neelakantan)

[Tracking the World State with Recurrent Entity Networks](https://arxiv.org/abs/1612.03969) (Facebook AI Research)
>	"There's a bunch of memory slots that each can be used to represent a single entity. The first time an entity appears, it's written to a slot. Every time that something happens in the story that corresponds to a change in the state of an entity, the change in the state of that entity is combined with the entity's previous state via a modified GRU update equation and rewritten to the same slot."  
  - `code` <https://github.com/jimfleming/recurrent-entity-networks>

[Key-Value Memory Networks for Directly Reading Documents](http://arxiv.org/abs/1606.03126) (Weston)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1606.03126>
  - `notes` <https://gist.github.com/shagunsodhani/a5e0baa075b4a917c0a69edc575772a8>
  - `code` <https://github.com/facebook/MemNN/blob/master/KVmemnn>

[Gated-Attention Readers for Text Comprehension](http://arxiv.org/abs/1606.01549) (Salakhutdinov)
  - `video` <https://youtube.com/watch?v=ZSDrM-tuOiA> (Salakhutdinov)
  - `notes` <https://theneuralperspective.com/2017/01/19/gated-attention-readers-for-text-comprehension/>

----
[Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision](http://arxiv.org/abs/1611.00020) (Google Brain)
>	"We propose the Manager-Programmer-Computer framework, which integrates neural networks with non-differentiable memory to support abstract, scalable and precise operations through a friendly neural computer interface. Specifically, we introduce a Neural Symbolic Machine, which contains a sequence-to-sequence neural "programmer", and a non-differentiable "computer" that is a Lisp interpreter with code assist."  

[Learning a Natural Language Interface with Neural Programmer](http://arxiv.org/abs/1611.08945)
  - `video` <http://youtu.be/lc68_d_DnYs?t=24m44s> (Neelakantan)
  - `code` <https://github.com/tensorflow/models/tree/master/neural_programmer>

[Neural Enquirer: Learning to Query Tables with Natural Language](http://arxiv.org/abs/1512.00965)
>	"Authors propose a fully distributed neural enquirer, comprising several neuralized execution layers of field attention, row annotation, etc. While the model is not efficient in execution because of intensive matrix/vector operation during neural information processing and lacks explicit interpretation of execution, it can be trained in an end-to-end fashion because all components in the neural enquirer are differentiable."  

----
[End-to-end Differentiable Proving](https://arxiv.org/abs/1705.11040) (Rocktaschel)
>	"We introduce neural networks for end-to-end differentiable theorem proving that operate on dense vector representations of symbols. These neural networks are constructed recursively by taking inspiration from the backward chaining algorithm as used in Prolog. Specifically, we replace symbolic unification with a differentiable computation on vector representations of symbols using a radial basis function kernel, thereby combining symbolic reasoning with learning subsymbolic vector representations. By using gradient descent, the resulting neural network can be trained to infer facts from a given incomplete knowledge base. It learns to (i) place representations of similar symbols in close proximity in a vector space, (ii) make use of such similarities to prove facts, (iii) induce logical rules, and (iv) use provided and induced logical rules for complex multi-hop reasoning. We demonstrate that this architecture outperforms ComplEx, a state-of-the-art neural link prediction model, on four benchmark knowledge bases while at the same time inducing interpretable function-free first-order logic rules."  
  - `slides` <http://aitp-conference.org/2017/slides/Tim_aitp.pdf> (Rocktaschel)
  - <https://soundcloud.com/nlp-highlights/19a> (Rocktaschel)
  - `paper` ["Learning Knowledge Base Inference with Neural Theorem Provers"](http://akbc.ws/2016/papers/14_Paper.pdf) by Rocktaschel and Riedel

[Differentiable Learning of Logical Rules for Knowledge Base Completion](https://arxiv.org/abs/1702.08367) (Cohen)

[TensorLog: A Differentiable Deductive Database](http://arxiv.org/abs/1605.06523) (Cohen)
  - `code` <https://github.com/TeamCohen/TensorLog>

---
[FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) (MILA)
>	"FiLM layer carries out a simple, feature-wise affine transformation on a neural network’s intermediate features, conditioned on an arbitrary input. In the case of visual reasoning, FiLM layers enable a RNN over an input question to influence CNN computation over an image. This process adaptively and radically alters the CNN’s behavior as a function of the input question, allowing the overall model to carry out a variety of reasoning tasks, ranging from counting to comparing. It also enables the CNN to properly localize question-referenced objects."  
>	"Ability to answer image-related questions requires learning a question-dependent, structured reasoning process over images from language. Standard deep learning approaches tend to exploit biases in the data rather than learn this underlying structure, while leading methods learn to visually reason successfully but are hand-crafted for reasoning."  
>	"The crazy thing is that the model does not include anything for reasoning and does not indicate anything about reasoning."  
  - `video` <https://youtu.be/02xIkHowQOk?t=2h44m55s> (Perez)

[A Simple Neural Network Module for Relational Reasoning](https://arxiv.org/abs/1706.01427) (DeepMind)
  - `video` <https://youtube.com/channel/UCIAnkrNn45D0MeYwtVpmbUQ> (demo)
  - `video` <https://youtu.be/02xIkHowQOk?t=2h38m> (Kahou)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1706.01427>
  - `code` <https://github.com/kimhc6028/relational-networks>
  - `code` <https://github.com/gitlimlab/Relation-Network-Tensorflow>
  - `code` <https://github.com/Alan-Lee123/relation-network>

[Inferring and Executing Programs for Visual Reasoning](https://arxiv.org/abs/1705.03633) (Facebook AI Research)
  - `video` <https://youtu.be/02xIkHowQOk?t=2h49m1s> (Perez)
  - `code` <https://github.com/facebookresearch/clevr-iep>

[Learning to Reason: End-to-End Module Networks for Visual Question Answering](https://arxiv.org/abs/1704.05526) (Darrell)
  - `post` <http://bair.berkeley.edu/blog/2017/06/20/learning-to-reason-with-neural-module-networks>
  - `video` <https://youtu.be/ejQNdTdyTBM?t=28m8s> (Kretov) (in russian)

[The More You Know: Using Knowledge Graphs for Image Classification](http://arxiv.org/abs/1612.04844) (Salakhutdinov)  # evolution of Gated Graph Sequence Neural Networks

----
[Visual Interaction Networks](https://arxiv.org/abs/1706.01433) (DeepMind)
>	"a general-purpose model for predicting future physical states from video data"  
>	"The VIN is learnable and can be trained from supervised data sequences which consist of input image frames and target object state values. It can learn to approximate a range of different physical systems which involve interacting entities by implicitly internalizing the rules necessary for simulating their dynamics and interactions."  
  - `video` <https://goo.gl/FD1XX5> + <https://goo.gl/4SSGP0> (demo)

[Interaction Networks for Learning about Objects, Relations and Physics](http://arxiv.org/abs/1612.00222) (DeepMind)
  - `notes` <https://blog.acolyer.org/2017/01/02/interaction-networks-for-learning-about-objects-relations-and-physics/>
  - `code` <https://github.com/jaesik817/Interaction-networks_tensorflow>

[Learning to Perform Physics Experiments via Deep Reinforcement Learning](http://arxiv.org/abs/1611.01843) (DeepMind)
  - `video` <https://youtu.be/SAcHyzMdbXc?t=16m6s> (de Freitas)

[Learning Physical Intuition of Block Towers by Example](http://arxiv.org/abs/1603.01312) (Facebook AI Research)
  - `video` <https://youtu.be/oSAG57plHnI?t=19m48s> (Tenenbaum)



---
### program induction

[Learning Neural Programs To Parse Programs](https://arxiv.org/abs/1706.01284)
>	"We explore a new direction to learn domain-specific programs significantly more complex than previously considered in the literature of learning programs from input-output examples only. In particular, we consider an exemplary problem to learn a program to parse an input satisfying a context-free grammar into its abstract syntax tree. This problem is challenging when the underlying grammar is unknown, and only input-output examples are provided. The program to be synthesized in this problem, i.e., a parser, is more complex than programs consisting of string operations as in many previous work, and thus serves as a good next step challenge to tackle in the domain of learning programs from input-output examples."  
>	"Recent works propose to use sequence-to-sequence models to directly generate parse trees from inputs. However, they often do not generalize well, and our experiments show that their test accuracy is almost 0% on inputs longer than those seen in training."  
>	"This work is the first successful demonstration that reinforcement learning can be applied to train a neural program operating a non-differentiable machine with input-output pairs only, while the learned neural program can fully generalize to longer inputs on a non-trivial task."  
>	"To show that our approach is general and can learn to parse different types of context-free languages using the same architecture and approach, we evaluate it on learning the parsing programs for an imperative language and a functional one, and demonstrate that our approach can successfully learn both of them, and the learned programs can achieve 100% on test set whose inputs are 100x longer than training samples."  
>	"We propose a new approach to learn a hybrid program, a differentiable neural program operating a domain-specific non-differentiable machine, from input-output examples only. Learning such a hybrid program combines the advantage of both differentiable and non-differentiable machines to enable learning more complex programs."  
>	"First, we propose LL machines as an example domain-specific non-differentiable machine to be operated by neural programs, for learning parsers. Intuitively, an LL machine provides a high-level abstraction to regularize the learned programs to be within the space of LL(1) parsers. The instructions provided by an LL machine provide richer semantic information than the primitives considered in previous works, so that the learning algorithm can take advantage of such information to learn more complex programs."  
>	"Second, we propose novel reinforcement learning-based techniques to train a neural program. Specifically, we solve the training problem in two phases: (1) we search for a valid execution trace set for each input-output example; then (2) we search for a set of input-output-trace combinations, so that a neural program can be trained to fit all training examples."  
  - `code` <http://github.com/liuchangacm/neuralparser>

[RobustFill: Neural Program Learning under Noisy I/O](https://arxiv.org/abs/1703.07469) (Microsoft)

[Neuro-Symbolic Program Synthesis](https://arxiv.org/abs/1611.01855) (Microsoft)

[TerpreT: A Probabilistic Programming Language for Program Induction](http://arxiv.org/abs/1608.04428) (Microsoft)
>	"These works raise questions of (a) whether new models can be designed specifically to synthesize interpretable source code that may contain looping and branching structures, and (b) whether searching over program space using techniques developed for training deep neural networks is a useful alternative to the combinatorial search methods used in traditional IPS. In this work, we make several contributions in both of these directions."  
>	"Shows that differentiable interpreter-based program induction is inferior to discrete search-based techniques used by the programming languages community. We are then left with the question of how to make progress on program induction using machine learning techniques."  
  - `video` <https://youtu.be/vzDuVhFMB9Q?t=2m40s> (Gaunt)

[Programming with a Differentiable Forth Interpreter](http://arxiv.org/abs/1605.06640) (Riedel)  # learning details of probabilistic program

[Making Neural Programming Architectures Generalize via Recursion](https://arxiv.org/abs/1704.06611)  # Neural Programmer-Interpreter with recursion
>	"We implement recursion in the Neural Programmer-Interpreter framework on four tasks: grade-school addition, bubble sort, topological sort, and quicksort."  
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255/> (49:59) (Cai)
  - `notes` <https://theneuralperspective.com/2017/03/14/making-neural-programming-architecture-generalize-via-recursion/>

[Adaptive Neural Compilation](http://arxiv.org/abs/1605.07969)



---
### reinforcement learning - agents

[A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) (DeepMind)  # Categorical DQN
>	"The value function gives the expected future discounted return. This ignores variance and multi-modality. Authors argue for modelling the full distribution of the return."  
>	"Distributional Bellman equation"  
>	"It is unclear if method works because of modelling uncertainty over rewards, training network with richer signal (categorical loss) or using distributional Bellman update."  
  - `post` <https://deepmind.com/blog/going-beyond-average-reinforcement-learning/>
  - `video` <https://youtube.com/watch?v=yFBwyPuO2Vg> (demo)
  - `video` <https://vimeo.com/235922311> (Bellemare)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=4m45s> (Mnih)
  - `code` <https://github.com/reinforceio/tensorforce/blob/master/tensorforce/models/categorical_dqn_model.py>
  - `code` <https://github.com/floringogianu/categorical-dqn>

[Multi-step Reinforcement Learning: A Unifying Algorithm](https://arxiv.org/abs/1703.01327) (Sutton)
>	"Currently, there are a multitude of algorithms that can be used to perform TD control, including Sarsa, Q-learning, and Expected Sarsa. These methods are often studied in the one-step case, but they can be extended across multiple time steps to achieve better performance. Each of these algorithms is seemingly distinct, and no one dominates the others for all problems. In this paper, we study a new multi-step action-value algorithm called Q(σ) which unifies and generalizes these existing algorithms, while subsuming them as special cases. A new parameter, σ, is introduced to allow the degree of sampling performed by the algorithm at each step during its backup to be continuously varied, with Sarsa existing at one extreme (full sampling), and Expected Sarsa existing at the other (pure expectation)."  
  - `video` <https://youtube.com/watch?v=MidZJ-oCpRk> (De Asis)

----
["Scalable Trust-region Method for Deep Reinforcement Learning using Kronecker-factored Approximation"](https://arxiv.org/abs/1708.05144)  # ACKTR
  - `video` <https://youtube.com/watch?v=0rrffaYuUi4> (Wu)
  - `video` ["Optimizing Neural Networks using Structured Probabilistic Models of the Gradient Computation"](https://fields.utoronto.ca/video-archive/2017/02/2267-16498 (Grosse)
  - `code` <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr>

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (OpenAI)
  - `post` <https://blog.openai.com/openai-baselines-ppo/> (demo)
  - `post` <https://learningai.io/projects/2017/07/28/ai-gym-workout.html>
  - `code` <https://github.com/openai/baselines/tree/master/baselines/pposgd>
  - `code` <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr>

[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286) (DeepMind)
>	"parallelized Proximal Policy Optimization"  
  - `video` <https://youtube.com/watch?v=hx_bgoTF7bs> (demo)
  - `code` <https://github.com/alexis-jacq/Pytorch-DPPO>

[Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning](https://arxiv.org/abs/1706.00387) (DeepMind)
>	"REINFORCE, TRPO, Q-Prop, DDPG, SVG(0), PGQ, ACER are special limiting cases of IPG"  

[Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/abs/1704.06440) (OpenAI)

[Bridging the Gap Between Value and Policy Reinforcement Learning](http://arxiv.org/abs/1702.08892) (Google Brain)  # PCL
  - `video` <https://youtu.be/fZNyHoXgV7M?t=1h16m17s> (Norouzi)
  - `notes` <https://github.com/ethancaballero/paper-notes/blob/master/Bridging%20the%20Gap%20Between%20Value%20and%20Policy%20Based%20Reinforcement%20Learning.md>
  - `code` <https://github.com/rarilurelo/pcl_keras>
  - `code` <https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/pcl.py>

[Trust-PCL: An Off-Policy Trust Region Method for Continuous Control](https://arxiv.org/abs/1707.01891) (Google Brain)

[Combining policy gradient and Q-learning](http://arxiv.org/abs/1611.01626) (DeepMind)  # PGQ
>	"This connection allows us to estimate the Q-values from the action preferences of the policy, to which we apply Q-learning updates."  
>	"We also establish an equivalency between action-value fitting techniques and actor-critic algorithms, showing that regularized policy gradient techniques can be interpreted as advantage function learning algorithms."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2FODonoghueMKM16>
  - `code` <https://github.com/Fritz449/Asynchronous-RL-agent>
  - `code` <https://github.com/abhishm/PGQ>

[Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](http://arxiv.org/abs/1611.02247) (Lillicrap, Levine)
>	"We present Q-Prop, a policy gradient method that uses a Taylor expansion of the off-policy critic as a control variate. Q-Prop is both sample efficient and stable, and effectively combines the benefits of on-policy and off-policy methods."  
	"- unbiased gradient  
	 - combine PG and AC gradients  
	 - learns critic from off-policy data  
	 - learns policy from on-policy data"  
	"Q-Prop works with smaller batch size than TRPO-GAE  
	Q-Prop is significantly more sample-efficient than TRPO-GAE"  
>	"policy gradient algorithm that is as fast as value estimation"  
>	"take off-policy algorithm and correct it with on-policy algorithm on residuals"  
>	"can be understood as REINFORCE with state-action-dependent baseline with bias correction term instead of unbiased state-dependent baseline"  
  - `video` <https://facebook.com/iclr.cc/videos/1712224178806641/> (1:36:47) (Gu)
  - `video` <https://youtu.be/M6nfipCxQBc?t=16m11s> (Lillicrap)
  - `notes` <http://www.alexirpan.com/rl-derivations/#q-prop>
  - `code` <https://github.com/shaneshixiang/rllabplusplus>

[The Reactor: A Sample-Efficient Actor-Critic Architecture](https://arxiv.org/abs/1704.04651)  # Reactor = Retrace-actor

[Sample Efficient Actor-Critic with Experience Replay](http://arxiv.org/abs/1611.01224) (DeepMind)  # ACER = A3C + Retrace
>	"adds Retrace(λ) off-policy bias correction to A3C"  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2FWangBHMMKF16>
  - `code` <https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/acer.py>

[Safe and Efficient Off-Policy Reinforcement Learning](http://arxiv.org/abs/1606.02647) (DeepMind)  # Retrace
>	"Retrace(λ) is a new strategy to weight a sample for off-policy learning, it provides low-variance, safe and efficient updates."  
>	"Our goal is to design a RL algorithm with two desired properties. Firstly, to use off-policy data, which is important for exploration, when we use memory replay, or observe log-data. Secondly, to use multi-steps returns in order to propagate rewards faster and avoid accumulation of approximation/estimation errors. Both properties are crucial in deep RL. We introduce the “Retrace” algorithm, which uses multi-steps returns and can safely and efficiently utilize any off-policy data."  
>	"open issue: off policy unbiased, low variance estimators for long horizon delayed reward problems"  
  - `video` <https://youtube.com/watch?v=8hK0NnG_DhY&t=25m27s> (Brunskill)

[Q(λ) with Off-Policy Corrections](http://arxiv.org/abs/1602.04951) (DeepMind)
  - `video` <https://youtube.com/watch?v=8hK0NnG_DhY&t=25m27s> (Brunskill)

[Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning](http://arxiv.org/abs/1604.00923) (Brunskill)
  - `video` <https://youtube.com/watch?v=8hK0NnG_DhY&t=15m44s> (Brunskill)

----
[Discrete Sequential Prediction of Continuous Actions for Deep RL](https://arxiv.org/abs/1705.05035) (Google Brain)  # SDQN
>	"We draw inspiration from the recent success of sequence-to-sequence models for structured prediction problems to develop policies over discretized spaces. Central to this method is the realization that complex functions over high dimensional spaces can be modeled by neural networks that use next step prediction. Specifically, we show how Q-values and policies over continuous spaces can be modeled using a next step prediction model over discretized dimensions. With this parameterization, it is possible to both leverage the compositional structure of action spaces during learning, as well as compute maxima over action spaces (approximately). On a simple example task we demonstrate empirically that our method can perform global search, which effectively gets around the local optimization issues that plague DDPG and NAF. We apply the technique to off-policy (Q-learning) methods and show that our method can achieve the state-of-the-art for off-policy methods on several continuous control tasks."  

[Reinforcement Learning in Large Discrete Action Spaces](http://arxiv.org/abs/1512.07679)

[Deep Reinforcement Learning In Parameterized Action Space](http://arxiv.org/abs/1511.04143)
  - `code` <https://github.com/mhauskn/dqn-hfo>

----
[RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](http://arxiv.org/abs/1611.02779) (OpenAI)
>	"MDPs encountered in real world = tiny subset of all MDPs that could be defined"  
>	"How to acquire a good prior for real-world MDPs?"  
>	"How to design algorithms that make use of such prior information?"  
>	"Key idea: learn a fast RL algorithm that make use of such prior information"  
>
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
  - `video` <http://www.fields.utoronto.ca/video-archive/2017/02/2267-16530> (19:00) (Abbeel)
  - `video` <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)
  - `video` <https://youtu.be/BskhUBPRrqE?t=6m28s> (Sutskever)
  - `video` <https://youtu.be/19eNQ1CLt5A?t=7m52s> (Sutskever)
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/RL2-Fast_Reinforcement_Learning_via_Slow_Reinforcement_Learning.md>

[Learning to Reinforcement Learn](http://arxiv.org/abs/1611.05763) (DeepMind)
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
  - `video` <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)
  - `post` <https://hackernoon.com/learning-policies-for-learning-policies-meta-reinforcement-learning-rl²-in-tensorflow-b15b592a2ddf> (Juliani)
  - `code` <https://github.com/awjuliani/Meta-RL>

----
[The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously](https://arxiv.org/abs/1707.03300) (DeepMind)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/> (20:40) (de Freitas)

[Dual Learning for Machine Translation](http://arxiv.org/abs/1611.00179)
>	"In the dual-learning mechanism, we use one agent to represent the model for the primal task and the other agent to represent the model for the dual task, then ask them to teach each other through a reinforcement learning process. Based on the feedback signals generated during this process (e.g., the language model likelihood of the output of a model, and the reconstruction error of the original sentence after the primal and dual translations), we can iteratively update the two models until convergence (e.g., using the policy gradient methods)."  
>	"The basic idea of dual learning is generally applicable: as long as two tasks are in dual form, we can apply the dual-learning mechanism to simultaneously learn both tasks from unlabeled data using reinforcement learning algorithms. Actually, many AI tasks are naturally in dual form, for example, speech recognition versus text to speech, image caption versus image generation, question answering versus question generation (e.g., Jeopardy!), search (matching queries to documents) versus keyword extraction (extracting keywords/queries for documents), so on and so forth."  
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/dual-learning-mt.md>

----
[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](http://arxiv.org/abs/1703.03864) (OpenAI)
>	(Karpathy) "ES is much simpler than RL, and there's no need for backprop, it's highly parallelizable, has fewer hyperparams, needs no value functions."  
>	"In our preliminary experiments we found that using ES to estimate the gradient on the MNIST digit recognition task can be as much as 1,000 times slower than using backpropagation. It is only in RL settings, where one has to estimate the gradient of the expected reward by sampling, where ES becomes competitive."  
  - `post` <https://blog.openai.com/evolution-strategies/>
  - `video` <https://www.technologyreview.com/s/603916/a-new-direction-for-artificial-intelligence/> (Sutskever)
  - `video` <https://youtube.com/watch?v=Rd0UdJFYkqI> (Temirchev) (in russian)
  - `post` <http://inference.vc/evolutionary-strategies-embarrassingly-parallelizable-optimization/> (Huszar)
  - `post` <http://inference.vc/evolution-strategies-variational-optimisation-and-natural-es-2/> (Huszar)
  - `post` <http://davidbarber.github.io/blog/2017/04/03/variational-optimisation/> (Barber)
  - `post` <http://argmin.net/2017/04/03/evolution/> (Recht)
  - `paper` ["Random Gradient-Free Minimization of Convex Functions"](https://mipt.ru/dcam/students/elective/a_5gc1te/RandomGradFree.PDF) by Nesterov
  - `paper` ["Natural Evolution Strategies"](http://jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) by Wierstra et al.
  - `code` <https://github.com/openai/evolution-strategies-starter>
  - `codd` <https://github.com/atgambardella/pytorch-es>

[Learning to Play in a Day: Faster Deep Reinforcement Learning by Optimality Tightening](https://arxiv.org/abs/1611.01606)
>	"We propose a novel training algorithm for reinforcement learning which combines the strength of deep Q-learning with a constrained optimization approach to tighten optimality and encourage faster reward propagation."  
  - `video` <https://yadi.sk/i/yBO0q4mI3GAxYd> (1:10:20) (Fritsler) (in russian)
  - `video` <https://youtu.be/mrj_hyH974o?t=16m13s> (in russian)



---
### reinforcement learning - exploration and intrinsic motivation

[Contextual Decision Processes with Low Bellman Rank are PAC-Learnable](https://arxiv.org/abs/1610.09512) (Microsoft)    # provably correct and sample efficient exploration
>	"This paper studies systematic exploration for reinforcement learning with rich observations and function approximation. We introduce a new model called contextual decision processes, that unifies and generalizes most prior settings. Our first contribution is a complexity measure, the Bellman rank, that we show enables tractable learning of near-optimal behavior in these processes and is naturally small for many well-studied reinforcement learning settings. Our second contribution is a new reinforcement learning algorithm that engages in systematic exploration to learn contextual decision processes with low Bellman rank. Our algorithm provably learns near-optimal behavior with a number of samples that is polynomial in all relevant parameters but independent of the number of unique observations. The approach uses Bellman error minimization with optimistic exploration and provides new insights into efficient exploration for reinforcement learning with function approximation."  
>	"Approximation of value function with function from some class is a powerful practical approach with implicit assumption that true value function is approximately in class.  
>	Even with this assumption:  
>	- no guarantee methods will work  
>	- no bound on how much data needed  
>	- no theory on how to explore in large spaces"  
  - `video` <https://vimeo.com/235929810> (Schapire)

[The Uncertainty Bellman Equation and Exploration](https://arxiv.org/abs/1709.05380) (DeepMind)    # exploration in state-action space guided by uncertainty of value function
>	"We consider uncertainty Bellman equation which connects the uncertainty at any time-step to the expected uncertainties at subsequent time-steps, thereby extending the potential exploratory benefit of a policy beyond individual time-steps. We prove that the unique fixed point of the UBE yields an upper bound on the variance of the estimated value of any fixed policy. This bound can be much tighter than traditional count-based bonuses that compound standard deviation rather than variance. Importantly, and unlike several existing approaches to optimism, this method scales naturally to large systems with complex generalization."  

----
[Count-Based Exploration with Neural Density Models](http://arxiv.org/abs/1703.01310) (DeepMind)    # exploration in state space guided by probability of observation
>	"PixelCNN for exploration, neural alternative to Context Tree Switching"  
  - `video` <http://youtube.com/watch?v=qSfd27AgcEk> (Bellemare)

[#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](http://arxiv.org/abs/1611.04717) (Abbeel)    # exploration in state space guided by probability of observation
>	"The authors encourage exploration by adding a pseudo-reward of the form beta/sqrt(count(state)) for infrequently visited states. State visits are counted using Locality Sensitive Hashing (LSH) based on an environment-specific feature representation like raw pixels or autoencoder representations. The authors show that this simple technique achieves gains in various classic RL control tasks and several games in the ATARI domain. While the algorithm itself is simple there are now several more hyperaprameters to tune: The bonus coefficient beta, the LSH hashing granularity (how many bits to use for hashing) as well as the type of feature representation based on which the hash is computed, which itself may have more parameters. The experiments don't paint a consistent picture and different environments seem to need vastly different hyperparameter settings, which in my opinion will make this technique difficult to use in practice."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1611.04717>

[EX2: Exploration with Exemplar Models for Deep Reinforcement Learning](https://arxiv.org/abs/1703.01260) (Levine)    # exploration in state space guided by probability of observation 
>	"Many of the most effective exploration techniques rely on tabular representations, or on the ability to construct a generative model over states and actions. This paper introduces a novel approach, EX2, which approximates state visitation densities by training an ensemble of discriminators, and assigns reward bonuses to rarely visited states."  

----
[Variational Intrinsic Control](http://arxiv.org/abs/1611.07507) (DeepMind)    # exploration in state space guided by empowerment (number of available actions)
>	"The second scenario is that in which the long-term goal of the agent is to get to a state with a maximal set of available intrinsic options – the objective of empowerment (Salge et al., 2014). This set o
f options consists of those that the agent knows how to use. Note that this is not the theoretical set of all options: it is of no use to the agent that it is possible to do something if it is unable to learn how
 to do it. Thus, to maximize empowerment, the agent needs to simultaneously learn how to control the environment as well – it needs to discover the options available to it. The agent should in fact not aim for st
ates where it has the most control according to its current abilities, but for states where it expects it will achieve the most control after learning. Being able to learn available options is thus fundamental to
 becoming empowered."  
>	"Let us compare this to the commonly used intrinsic motivation objective of maximizing the amount of model-learning progress, measured as the difference in compression of its experience before and after l
earning (Schmidhuber, 1991; 2010; Bellemare et al., 2016; Houthooft et al., 2016). The empowerment objective differs from this in a fundamental manner: the primary goal is not to understand or predict the observa
tions but to control the environment. This is an important point – agents can often control an environment perfectly well without much understanding, as exemplified by canonical model-free reinforcement learning 
algorithms (Sutton & Barto, 1998), where agents only model action-conditioned expected returns. Focusing on such understanding might significantly distract and impair the agent, as such reducing the control it ac
hieves."  

----
[Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363) (Darrell)    # exploration in state space guided by prediction error
>	"Our main contribution is in designing an intrinsic reward signal based on prediction error of the agent’s knowledge about its environment that scales to high-dimensional continuous state spaces like images, bypasses the hard problem of predicting pixels and is unaffected by the unpredictable aspects of the environment that do not affect the agent."  
  - `post` <https://pathak22.github.io/noreward-rl/index.html> (demo)
  - `code` <https://github.com/pathak22/noreward-rl>

[Surprise-Based Intrinsic Motivation for Deep Reinforcement Learning](http://arxiv.org/abs/1703.01732)    # exploration in state-action space guided by prediction error
>	"Authors present two tractable approximations to their framework - one which ignores the stochasticity of the true environmental dynamics, and one which approximates the rate of information gain (somewhat similar to Schmidhuber's formal theory of creativity, fun and intrinsic motivation)."  
>	"Stadie et al. learn deterministic dynamics models by minimizing Euclidean loss - whereas in our work, we learn stochastic dynamics with cross entropy loss - and use L2 prediction errors for intrinsic motivation."  
>	"Our results suggest that surprisal is a viable alternative to VIME in terms of performance, and is highly favorable in terms of computational cost. In VIME, a backwards pass through the dynamics model must be computed for every transition tuple separately to compute the intrinsic rewards, whereas our surprisal bonus only requires forward passes through the dynamics model for intrinsic reward computation. Furthermore, our dynamics model is substantially simpler than the Bayesian neural network dynamics model of VIME. In our speed test, our bonus had a per-iteration speedup of a factor of 3 over VIME."  

----
[RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](http://arxiv.org/abs/1611.02779) (OpenAI)    # learning to explore in state-action space
>	"MDPs encountered in real world = tiny subset of all MDPs that could be defined"  
>	"How to acquire a good prior for real-world MDPs?"  
>	"How to design algorithms that make use of such prior information?"  
>	"Key idea: learn a fast RL algorithm that make use of such prior information"  
>
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy for outer POMDP with no state reset between inner episodes"  
  - `video` <http://www.fields.utoronto.ca/video-archive/2017/02/2267-16530> (19:00) (Abbeel)
  - `video` <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)
  - `video` <https://youtu.be/BskhUBPRrqE?t=6m28s> (Sutskever)
  - `video` <https://youtu.be/19eNQ1CLt5A?t=7m52s> (Sutskever)
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/RL2-Fast_Reinforcement_Learning_via_Slow_Reinforcement_Learning.md>

[Learning to Reinforcement Learn](http://arxiv.org/abs/1611.05763) (DeepMind)    # learning to explore in state-action space
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
  - `video` <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)
  - `post` <https://hackernoon.com/learning-policies-for-learning-policies-meta-reinforcement-learning-rl²-in-tensorflow-b15b592a2ddf> (Juliani)
  - `code` <https://github.com/awjuliani/Meta-RL>

[Exploration Potential](http://arxiv.org/abs/1609.04994)    # learning to explore in state-action space
>	"We introduce exploration potential, a quantity that measures how much a reinforcement learning agent has explored its environment class. In contrast to information gain, exploration potential takes the problem's reward structure into account."  

----
[Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) (DeepMind)    # deeper random exploration
>	"scale of perturbation to parameters is learned along with original objective function"  
  - `code` <https://github.com/Kaixhin/NoisyNet-A3C>
  - `code` <https://github.com/andrewliao11/NoisyNet-DQN>

[Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905) (OpenAI)    # deeper random exploration
  - `post` <https://blog.openai.com/better-exploration-with-parameter-noise/>

----
[UCB and InfoGain Exploration via Q-Ensembles](https://arxiv.org/abs/1706.01502) (OpenAI)    # approximate bayesian exploration

[Deep Exploration via Randomized Value Functions](https://arxiv.org/abs/1703.07608) (Osband)    # approximate bayesian exploration
>	"A very recent thread of work builds on count-based (or upper-confidence-bound-based) exploration schemes that operate with value function learning. These methods maintain a density over the state-action space of pseudo-counts, which represent the quantity of data gathered that is relevant to each state-action pair. Such algorithms may offer a viable approach to deep exploration with generalization. There are, however, some potential drawbacks. One is that a separate representation is required to generalize counts, and it's not clear how to design an effective approach to this. As opposed to the optimal value function, which is fixed by the environment, counts are generated by the agent’s choices, so there is no single target function to learn. Second, the count model generates reward bonuses that distort data used to fit the value function, so the value function representation needs to be designed to not only capture properties of the true optimal value function but also such distorted versions. Finally, these approaches treat uncertainties as uncoupled across state-action pairs, and this can incur a substantial negative impact on statistical efficiency."  
  - `video` <http://youtube.com/watch?v=ck4GixLs4ZQ> (Osband) + [slides](https://docs.google.com/presentation/d/1lis0yBGT-uIXnAsi0vlP3SuWD2svMErJWy_LYtfzMOA/)

----
[The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously](https://arxiv.org/abs/1707.03300) (DeepMind)    # structured exploration in policy space guided by learning additional tasks

[Reinforcement Learning with Unsupervised Auxiliary Tasks](http://arxiv.org/abs/1611.05397) (DeepMind)    # UNREAL, structured exploration in policy space guided by learning additional tasks
>	"Auxiliary tasks:  
>	- pixel changes: learn a policy for maximally changing the pixels in a grid of cells overlaid over the images  
>	- network features: learn a policy for maximally activating units in a specific hidden layer  
>	- reward prediction: predict the next reward given some historical context  
>	- value function replay: value function regression for the base agent with varying window for n-step returns"  
>	"By using these tasks we force the agent to learn about the controllability of its environment and the sorts of sequences which lead to rewards, and all of this shapes the features of the agent."
>	"This approach exploits the multithreading capabilities of standard CPUs. The idea is to execute many instances of our agent in parallel, but using a shared model. This provides a viable alternative to experience replay, since parallelisation also diversifies and decorrelates the data. Our asynchronous actor-critic algorithm, A3C, combines a deep Q-network with a deep policy network for selecting actions. It achieves state-of-the-art results, using a fraction of the training time of DQN and a fraction of the resource consumption of Gorila."  
  - `video` <https://youtube.com/watch?v=Uz-zGYrYEjA> (demo)
  - `video` <https://youtube.com/watch?v=VVLYTqZJrXY> (Jaderberg)
  - `video` <https://facebook.com/iclr.cc/videos/1712224178806641/> (1:15:45) (Jaderberg)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=20m7s> (Mnih)
  - `video` <https://youtube.com/watch?v=-YiMVR3HEuY> (Kilcher)
  - `video` <https://yadi.sk/i/_2_0yqeW3HDbcn> (18:25) (Panin) (in russian)
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/b097e313dc59c956575fb1bf23b64fa8d1d84057/notes/rl-auxiliary-tasks.md>
  - `code` <https://github.com/miyosuda/unreal>

[Learning to Navigate in Complex Environments](http://arxiv.org/abs/1611.03673) (DeepMind)    # structured exploration in policy space guided by learning additional tasks
  - `video` <http://youtu.be/0e_uGa7ic74?t=8m53s> (Hadsell)
  - `video` <http://youtube.com/watch?v=5Rflbx8y7HY> (Mirowski)
  - `notes` <http://pemami4911.github.io/paper-summaries/2016/12/20/learning-to-navigate-in-complex-envs.html>

[Loss is Its Own Reward: Self-Supervision for Reinforcement Learning](http://arxiv.org/abs/1612.07307) (Darrell)    # structured exploration in policy space guided by learning additional tasks

[Feature Control as Intrinsic Motivation for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1705.06769)    # structured exploration in policy space guided by learning additional tasks

[Independently Controllable Features](https://arxiv.org/abs/1708.01289) (Bengio)    # structured exploration in state and policy space guided by learning to control environment
>	"Exploration process could be driven by a notion of controllability, predicting the interestingness of objects in a scene and choosing features and associated policies with which to attempt controlling them – such ideas have only been briefly explored in the literature. How do humans choose with which object to play? We are attracted to objects for which we do not yet know if and how we can control them, and such a process may be critical to learn how the world works."  
>	"It has been postulated that a good representation is one that disentangles the underlying explanatory factors of variation. However, it remains an open question what kind of training framework could potentially achieve that. Whereas most previous work focuses on the static setting (e.g. with images), we postulate that some of the causal factors could be discovered if the learner is allowed to interact with its environment. The agent can experiment with different actions and observe their effects. We hypothesize that some of these factors correspond to aspects of the environment which are independently controllable, i.e., that there exists a policy and a learnable feature for each such aspect of the environment, such that this policy can yield changes in that feature with minimal changes to other features that explain the statistical variations in the observed data."  
>	"In interactive environments, the temporal dependency between successive observations creates a new opportunity to notice causal structure in data which may not be apparent using only observational studies. In reinforcement learning, several approaches explore mechanisms that push the internal representations of learned models to be “good” in the sense that they provide better control, and control is a particularly important causal relationship between an agent and elements of its environment."  
>	"We propose and explore a more direct mechanism for representation learning, which explicitly links an agent’s control over its environment with its internal feature representations. Specifically, we hypothesize that some of the factors explaining variations in the data correspond to aspects of the world that can be controlled by the agent. For example, an object that could be pushed around or picked up independently of others is an independently controllable aspect of the environment. Our approach therefore aims to jointly discover a set of features (functions of the environment state) and policies (which change the state) such that each policy controls the associated feature while leaving the other features unchanged as much as possible."  

----
[Overcoming Exploration in Reinforcement Learning with Demonstrations](https://arxiv.org/abs/1709.10089) (OpenAI)    # structured exploration in policy space guided by learning progress and demonstrations
>	"We use demonstrations to overcome the exploration problem and successfully learn to perform long-horizon, multi-step robotics tasks with continuous control such as stacking blocks with a robot arm."  
>	"Our method, which builds on top of Deep Deterministic Policy Gradients and Hindsight Experience Replay, provides an order of magnitude of speedup over RL on simulated robotics tasks."  
>	"Our method is able to solve tasks not solvable by either RL or behavior cloning alone, and often ends up outperforming the demonstrator policy."  
  - <http://ashvin.me/demoddpg-website/> (demo)

[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (OpenAI)    # structured exploration in policy space guided by learning progress
>	"HER may be seen as a form of implicit curriculum as the goals used for replay naturally shift from ones which are simple to achieve even by a random agent to more difficult ones. However, in contrast to explicit curriculum, HER does not require having any control over the distribution of initial environment states."  
>	"Not only does HER learn with extremely sparse rewards, in our experiments it also performs better with sparse rewards than with shaped ones. These results are indicative of the practical challenges with reward shaping, and that shaped rewards would often constitute a compromise on the metric we truly care about (such as binary success/failure)."  
  - <https://sites.google.com/site/hindsightexperiencereplay/> (demo)
  - `paper` ["Universal Value Function Approximators"](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#schaul-horgan-gregor-silver---universal-value-function-approximators) by Schaul et al.

[Reverse Curriculum Generation for Reinforcement Learning](https://arxiv.org/abs/1707.05300) (OpenAI)    # structured exploration in policy space guided by learning progress
>	"Many tasks require to reach a desired configuration (goal) from everywhere."  
>	"Challenging for current RL: inherently sparse rewards, most start positions get 0 reward."  
>
>	"Solve the task in reverse, first training from positions closer to the goal and then bootstrap this knowledge to solve from further."  
>	"Sample more start states from where you succeed sometimes but not always (for best efficiency)."  

[Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183) (OpenAI)    # structured exploration in policy space guided by learning progress

[Automated Curriculum Learning for Neural Networks](https://arxiv.org/abs/1704.03003) (DeepMind)    # structured exploration in policy space guided by learning progress
>	"We focus on variants of prediction gain, and also introduce a novel class of progress signals which we refer to as complexity gain. Derived from minimum description length principles, complexity gain equates acquisition of knowledge with an increase in effective information encoded in the network weights."  
>	"VIME uses a reward signal that is closely related to variational complexity gain. The difference is that while VIME measures the KL between the posterior before and after a step in parameter space, we consider the change in KL between the posterior and prior induced by the step. Therefore, while VIME looks for any change to the posterior, we focus only on changes that alter the divergence from the prior. Further research will be needed to assess the relative merits of the two signals."  
>	"For maximum likelihood training, we found prediction gain to be the most consistent signal, while for variational inference training, gradient variational complexity gain performed best. Importantly, both are instantaneous, in the sense that they can be evaluated using only the samples used for training."  
  - `video` <https://youtu.be/-u32TOPGIbQ?t=2m43s> (Graves)

[Automatic Goal Generation for Reinforcement Learning Agents](https://arxiv.org/abs/1705.06366) (Abbeel)    # structured exploration in policy space guided by learning progress
>	"self-play between the policy and a task-setter in order to automatically generate goal states which are on the border of what the current policy can achieve"  

[Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play](http://arxiv.org/abs/1703.05407) (Facebook AI Research)    # structured exploration in policy space guided by learning progress
  - `video` <https://youtube.com/watch?v=EHHiFwStqaA> (demo)
  - `video` <https://youtube.com/watch?v=X1O21ziUqUY> (Fergus)
  - `video` <https://youtube.com/watch?v=pSJC9hNt3Uw> (Szlam)
  - `post` <http://giorgiopatrini.org/posts/2017/09/06/in-search-of-the-missing-signals/>

[Towards Information-Seeking Agents](http://arxiv.org/abs/1612.02605) (Maluuba)    # structured exploration in policy space guided by learning progress
  - `video` <https://youtube.com/watch?v=3bSquT1zqj8> (demo)

[Learning to Perform Physics Experiments via Deep Reinforcement Learning](http://arxiv.org/abs/1611.01843) (DeepMind)    # structured exploration in policy space guided by learning progress
>	"By letting our agents conduct physical experiments in an interactive simulated environment, they learn to manipulate objects and observe the consequences to infer hidden object properties."  
>	"By systematically manipulating the problem difficulty and the cost incurred by the agent for performing experiments, we found that agents learn different strategies that balance the cost of gathering information against the cost of making mistakes in different situations."  
  - `video` <https://youtu.be/SAcHyzMdbXc?t=16m6s> (de Freitas)



---
### reinforcement learning - hierarchical reinforcement learning

[Variational Intrinsic Control](http://arxiv.org/abs/1611.07507) (DeepMind)

[A Laplacian Framework for Option Discovery in Reinforcement Learning](https://arxiv.org/abs/1703.00956) (Bowling)
>	"Our algorithm can be seen as a bottom-up approach, in which we construct options before the agent observes any informative reward. These options are composed to generate the desired policy. Options discovered this way tend to be independent of an agent’s intention, and are potentially useful in many different tasks. Moreover, such options can also be seen as being useful for exploration by allowing agents to commit to a behavior for an extended period of time."  
  - `video` <https://youtube.com/watch?v=2BVicx4CDWA> (demo)
  - `video` <https://vimeo.com/220484541> (Machado)

[Strategic Attentive Writer for Learning Macro-Actions](http://arxiv.org/abs/1606.04695) (DeepMind)
>	"method that learns to initialize and update a plan, but which does not use a model and instead directly maps new observations to plan updates"
  - `video` <https://youtube.com/watch?v=niMOdSu3yio> (demo)
  - `notes` <https://blog.acolyer.org/2017/01/06/strategic-attentive-writer-for-learning-macro-actions/>
  - `post` <http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-3-synergistic-and-modular-action/> (Mohamed)

[The Option-Critic Architecture](http://arxiv.org/abs/1609.05140) (Precup)
  - `video` <https://youtube.com/watch?v=8r_EoYnPjGk> (Bacon)
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=56m55s> (Abbeel)

[Options Discovery with Budgeted Reinforcement Learning](https://arxiv.org/abs/1611.06824)

----
[Modular Multitask Reinforcement Learning with Policy Sketches](http://arxiv.org/abs/1611.01796) (Levine)
  - `video` <https://youtube.com/watch?v=NRIcDEB64x8> (Andreas)
  - `code` <https://github.com/jacobandreas/psketch>

[Stochastic Neural Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1704.03012) (Abbeel)
>	"Our SNN hierarchical approach outperforms state-of-the-art intrinsic motivation results like VIME (Houthooft et al., 2016)."  
  - `video` <https://youtube.com/playlist?list=PLEbdzN4PXRGVB8NsPffxsBSOCcWFBMQx3> (demo)
  - `code` <https://github.com/florensacc/snn4hrl>

[FeUdal Networks for Hierarchical Reinforcement Learning](http://arxiv.org/abs/1703.01161) (DeepMind)
>	"Agent with two level hierarchy: manager and worker."  
>	"Manager does not act in environment directly, sets goals for worker and gets rewarded for setting good goals with true reward."  
>	"Worker acts in environment and gets rewarded for achieving goals by manager - this is potentially much richer learning signal."  
>	"Key problems: how to represent goals and determine when they've been achieved."  
  - `video` <https://youtube.com/watch?v=0e_uGa7ic74&t=29m20s> (demo)
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=57m51s> (Abbeel)
  - `video` <https://youtube.com/watch?v=bsuvM1jO-4w&t=46m31s> (Mnih)
  - `video` <https://youtube.com/watch?v=0e_uGa7ic74&t=20m10s> (Hadsell)
  - `code` <https://github.com/dmakian/feudal_networks>

[A Deep Hierarchical Approach to Lifelong Learning in Minecraft](http://arxiv.org/abs/1604.07255)
  - `video` <https://youtube.com/watch?v=RwjfE4kc6j8> (demo)

----
[Principled Option Learning in Markov Decision Processes](https://arxiv.org/abs/1609.05524) (Tishby)
>	"We suggest a mathematical characterization of good sets of options using tools from information theory. This characterization enables us to find conditions for a set of options to be optimal and an algorithm that outputs a useful set of options and illustrate the proposed algorithm in simulation."  



---
### reinforcement learning - simulation and planning

[Recurrent Environment Simulators](https://arxiv.org/abs/1704.02254) (DeepMind)
>	"We improve on previous environment simulators from high-dimensional pixel observations by introducing recurrent neural networks that are able to make temporally and spatially coherent predictions for hundreds of time-steps into the future. "  
>	"We address the issue of computationally inefficiency with a model that does not need to generate a high-dimensional image at each time-step."  
  - `video` <https://drive.google.com/file/d/0B_L2b7VHvBW2NEQ1djNjU25tWUE/view> (TORCS demo)
  - `video` <https://drive.google.com/file/d/0B_L2b7VHvBW2UjMwWVRoM3lTeFU/view> (TORCS demo)
  - `video` <https://drive.google.com/file/d/0B_L2b7VHvBW2UWl5YUtSMXdUbnc/view> (3D Maze demo)

[Prediction and Control with Temporal Segment Models](https://arxiv.org/abs/1703.04070) (Abbeel)
>	"variational autoencoder to learn the distribution over future state trajectories conditioned on past states, past actions, and planned future actions"  
>	"latent action prior, another variational autoencoder that models a prior over action segments, and showed how it can be used to perform control using actions from the same distribution as a dynamics model’s training data"  

[Counterfactual Control for Free from Generative Models](https://arxiv.org/abs/1702.06676)
>	"generative model learning the joint distribution between actions and future states can be used to automatically infer a control scheme for any desired reward function, which may be altered on the fly without retraining the model"  
>	"problem of action selection is reduced to one of gradient descent on the latent space of the generative model, with the model itself providing the means of evaluating outcomes and finding the gradient, much like how the reward network in Deep Q-Networks provides gradient information for the action generator"  

[Learning Multimodal Transition Dynamics for Model-Based Reinforcement Learning](https://arxiv.org/abs/1705.00470)
>	"So why is model-based RL not the standard approach? Model-based RL consists of two steps: 1) transition function estimation through supervised learning, and 2) (sample-based) planning over the learned model. Each step has a particular challenging aspect. For this work we focus on a key challenge of the first step: stochasticity in the transition dynamics. Stochasticity is an inherent property of many environments, and increases in real-world settings due to sensor noise. Transition dynamics usually combine both deterministic aspects (such as the falling trajectory of an object due to gravity) and stochastic elements (such as the behaviour of another car on the road). Our goal is to learn to jointly predict these. Note that stochasticity has many forms, both homoscedastic versus heteroscedastic, and unimodal versus multimodal. In this work we specifically focus on multimodal stochasticity, as this should theoretically pose the largest challenge."  
>	"We focus on deep generative models as they can approximate complex distributions and scale to high-dimensional domains. For model-based RL we have additional requirements, as we are ultimately interested in using the model for sample-based planning. This usually requires sampling a lot of traces, so we require models that are 1) easy to sample from, 2) ideally allow planning at an abstract level. Implicit density models, like Generative Adverserial Networks lack a clear probabilistic objective function, which was the focus of this work. Among the explicit density models, there are two categories. Change of variable formula models, like Real NVP, have the drawback that the latent space dimension must equal the observation space. Fully visible belief nets like pixelCNN, which factorize the likelihood in an auto-regressive fashion, hold state-of-the-art likelihood results. However, they have the drawback that sampling is a sequential operation (e.g. pixel-by-pixel, which is computationally expensive), and they do not allow for latent level planning either. Therefore, most suitable for model-based RL seem approximate density models, most noteworthy the Variational Auto-Encoder framework. These models can estimate stochasticity at a latent level, allow for latent planning, are easy to sample from, and have a clear probabilistic interpretation."  
>	"An important challenge is planning under uncertainty. RL initially provides correlated data from a limited part of state-space. When planning over this model, we should not extrapolate too much, nor trust our model to early with limited data. Note that ‘uncertainty’ (due to limited data) is fundamentally different from the ‘stochasticity’ (true probabilistic nature of the domain) discussed in this paper."  
  - `code` <http://github.com/tmoer/multimodal_varinf>

[Learning and Policy Search in Stochastic Dynamic Systems with Bayesian Neural Networks](https://arxiv.org/abs/1605.07127)
>	"Monte-Carlo model-based policy gradient technique in continuous stochastic systems"  
>	"Proposed approach enables automatic identification of arbitrary stochastic patterns such as multimodality and heteroskedasticity, without having to manually incorporate these into the model."  
>	"We have extended Bayesian neural network with addition of a random input noise source z. This enables principled Bayesian inference over complex stochastic functions. We have also presented an algorithm that uses random roll-outs and stochastic optimization for learning a parameterized policy in a batch scenario. Our BNNs with random inputs have allowed us to solve a challenging benchmark problem where model-based approaches usually fail."  
>	"For safety, we believe having uncertainty over the underlying stochastic functions will allow us to optimize policies by focusing on worst case results instead of on average performance. For exploration, having uncertainty on the stochastic functions will be useful for efficient data collection."  
>	"The optimal policy can be significantly affected by the noise present in the state transitions. This is illustrated by the drunken spider story, in which a spider has two possible paths to go home: either by crossing the bridge or by walking around the lake. In the absence of noise, the bridge option is prefered since it is shorter. However, after heavily drinking alcohol, the spider’s movements may randomly deviate left or right. Since the bridge is narrow, and spiders do not like swimming, the prefered trajectory is now to walk around the lake. The previous example shows how noise can significantly affect optimal control. For example, the optimal policy may change depending on whether the level of noise is high or low. Therefore, we expect to obtain significant improvements in model-based reinforcement learning by capturing with high accuracy any noise patterns present in the state transition data."  
  - `post` <https://medium.com/towards-data-science/bayesian-neural-networks-with-random-inputs-for-model-based-reinforcement-learning-36606a9399b4> (Hernandez-Lobato)
  - `video` <https://youtube.com/watch?v=0H3EkUPENSY> (Hernandez-Lobato)
  - `code` <https://github.com/siemens/policy_search_bb-alpha>

----
[Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/abs/1707.06203) (DeepMind)
>	"interpreting imperfect explicit models"  
  - `video` <https://drive.google.com/drive/folders/0B4tKsKnCCZtQY2tTOThucHVxUTQ> (demo)
  - `post` <https://deepmind.com/blog/agents-imagine-and-plan/>
  - `video` <https://youtube.com/watch?v=agXIYMCICcc> (Kilcher)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=39m27s> (Mnih)

[Learning Model-based Planning from Scratch](https://arxiv.org/abs/1707.06170) (DeepMind)
  - `video` <https://drive.google.com/drive/folders/0B3u8dCFTG5iVaUxzbzRmNldGcU0> (demo)
  - `post` <https://deepmind.com/blog/agents-imagine-and-plan/>
  - `video` <https://youtube.com/watch?v=56GW1IlWgMg> (Kilcher)

[Metacontrol for Adaptive Imagination-Based Optimization](https://arxiv.org/abs/1705.02670) (DeepMind)
>	"Rather than learning a single, fixed policy for solving all instances of a task, we introduce a metacontroller which learns to optimize a sequence of "imagined" internal simulations over predictive models of the world in order to construct a more informed, and more economical, solution. The metacontroller component is a model-free reinforcement learning agent, which decides both how many iterations of the optimization procedure to run, as well as which model to consult on each iteration. The models (which we call "experts") can be state transition models, action-value functions, or any other mechanism that provides information useful for solving the task, and can be learned on-policy or off-policy in parallel with the metacontroller."  
>	"learns an adaptive optimization policy for one-shot decision-making in contextual bandit problems"  

[Value Prediction Network](https://arxiv.org/abs/1707.03497)
>	"extending Predictron model from policy evaluation to optimal control"  
>	"VPN combines model-based RL (i.e., learning the dynamics of an abstract state space sufficient for computing future rewards and values) and model-free RL (i.e., mapping the learned abstract states to rewards and values) in a unified framework. In order to train a VPN, we propose a combination of temporal-difference search (TD search) and n-step Q-learning. In brief, VPNs learn to predict values via Q-learning and rewards via supervised learning. At the same time, VPNs perform lookahead planning to choose actions and compute bootstrapped target Q-values."  
  - `video` <http://videolectures.net/deeplearning2017_singh_reinforcement_learning/> (1:12:46) (Singh)
  - `video` <https://youtu.be/PRQ8-FwDPRE?t=16m> (Holland)

[The Predictron: End-to-End Learning and Planning](https://arxiv.org/abs/1612.08810) (DeepMind)
>	"trains deep network to implicitly plan via iterative rollouts"  
>	"uses implicit environment model which does not capture dynamics"  
>	"only applied to learning Markov reward processes rather than solving Markov decision processes"  
  - `video` <https://youtube.com/watch?v=BeaLdaN2C3Q> (demo)
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=55m9s> (Abbeel)
  - `video` <http://videolectures.net/deeplearning2017_singh_reinforcement_learning/> (1:12:46) (Singh)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=38m6s> (Mnih)
  - `code` <https://github.com/zhongwen/predictron>
  - `code` <https://github.com/muupan/predictron>

[Value Iteration Networks](http://arxiv.org/abs/1602.02867) (Abbeel)
>	"trains deep network to implicitly plan via iterative rollouts"  
>	"uses implicit environment model which does not capture dynamics"  
  - `video` <https://youtu.be/ID150Tl-MMw?t=54m24s> (demo)
  - `video` <https://youtube.com/watch?v=tXBHfbHHlKc> (Tamar) ([slides](http://technion.ac.il/~danielm/icml_slides/Talk7.pdf))
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Value-Iteration-Networks> (Tamar)
  - `video` <http://www.fields.utoronto.ca/video-archive/2017/02/2267-16530> (31:50) (Abbeel)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=38m6s> (Mnih)
  - `notes` <https://github.com/karpathy/paper-notes/blob/master/vin.md>
  - `notes` <https://blog.acolyer.org/2017/02/09/value-iteration-networks/>
  - `code` <https://github.com/avivt/VIN>
  - `code` <https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks>
  - `code` <https://github.com/zuoxingdong/VIN_PyTorch_Visdom>
  - `code` <https://github.com/zuoxingdong/VIN_TensorFlow>
  - `code` <https://github.com/kentsommer/pytorch-value-iteration-networks>
  - `code` <https://github.com/onlytailei/Value-Iteration-Networks-PyTorch>

[Strategic Attentive Writer for Learning Macro-Actions](http://arxiv.org/abs/1606.04695) (DeepMind)
>	"method that learns to initialize and update a plan, but which does not use a model and instead directly maps new observations to plan updates"  
  - `video` <https://youtube.com/watch?v=niMOdSu3yio> (demo)
  - `notes` <https://blog.acolyer.org/2017/01/06/strategic-attentive-writer-for-learning-macro-actions/>
  - `post` <http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-3-synergistic-and-modular-action/> (Mohamed)

----
[Model-Based Planning in Discrete Action Spaces](https://arxiv.org/abs/1705.07177) (LeCun)

[Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439) (Barber)
>	"Planning new policies is performed by tree search, while a deep neural network generalises those plans"  

[Blazing the Trails before Beating the Path: Sample-efficient Monte-Carlo Planning](https://papers.nips.cc/paper/6253-blazing-the-trails-before-beating-the-path-sample-efficient-monte-carlo-planning.pdf) (Munos)
>	"We study the sampling-based planning problem in Markov decision processes (MDPs) that we can access only through a generative model, usually referred to as Monte-Carlo planning."  
>	"Our objective is to return a good estimate of the optimal value function at any state while minimizing the number of calls to the generative model, i.e. the sample complexity."  
>	"TrailBlazer is an adaptive algorithm that exploits possible structures of the MDP by exploring only a subset of states reachable by following near-optimal policies."  
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Blazing-the-trails-before-beating-the-path-Sample-efficient-Monte-Carlo-planning>



---
### reinforcement learning - memory

[Neural Episodic Control](https://arxiv.org/abs/1703.01988) (DeepMind)
>	"Purely parametric approaches are very data inefficient. A hybrid parametric/non-parametric method can be much more efficient."  
>	"NEC represents action-value function as table: slowly changing learned keys + rapidly changing Q-value estimates."  
>
>	"Differentiable memories are used as approximate hash tables, allowing to store and retrieve successful experiences to facilitate rapid learning."  
>
>	"Our agent uses a semi-tabular representation of the value function: a buffer of past experience containing slowly changing state representations and rapidly updated estimates of the value function."  
>
>	"Greedy non-parametric tabular-memory agents like MFEC can outperform model-based agents when data are noisy or scarce.  
>	NEC outperforms MFEC by creating an end-to-end trainable learning system using differentiable neural dictionaries and a convolutional neural network.  
>	A representation of the environment as generated by the mammalian brain's ventral stream can be approximated with random projections, a variational autoencoder, or a convolutional neural network."  
  - `video` <https://youtu.be/bsuvM1jO-4w?t=42m10s> (Mnih)
  - `notes` <http://rylanschaeffer.github.io/content/research/neural_episodic_control/main.html>
  - `code` <https://github.com/EndingCredits/Neural-Episodic-Control>

[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460) (DeepMind)
>	"This might be achieved by a dual system (hippocampus vs neocortex) where information are stored in alternated way such that new nonstationary experience is rapidly encoded in the hippocampus (most flexible region of the brain with the highest amount of plasticity and neurogenesis); long term memory in the cortex is updated in a separate phase where what is updated (both in terms of samples and targets) can be controlled and does not put the system at risk of instabilities."  
  - <https://sites.google.com/site/episodiccontrol/> (demo)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=42m10s> (Mnih)
  - `post` <http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-4-episodic-and-interactive-memory/>
  - `code` <https://github.com/ShibiHe/Model-Free-Episodic-Control>
  - `code` <https://github.com/sudeepraja/Model-Free-Episodic-Control>

[Neural Map: Structured Memory for Deep Reinforcement Learning](https://arxiv.org/abs/1702.08360) (Salakhutdinov)
>	"Memory was given a 2D structure in order to resemble a spatial map to address specific problems such as 2D or 3D navigation"  
>	"Size and computational cost doesn't grow with time horizon of environment"  
  - `video` <https://yadi.sk/i/pMdw-_uI3Gke7Z> (Shvechikov) (in russian)

[A Growing Long-term Episodic and Semantic Memory](http://arxiv.org/abs/1610.06402)

[Memory-based Control with Recurrent Neural Networks](http://arxiv.org/abs/1512.04455) (DeepMind)
  - `video` <https://youtube.com/watch?v=V4_vb1D5NNQ> (demo)



---
### reinforcement learning - transfer

[Generalizing Skills with Semi-Supervised Reinforcement Learning](http://arxiv.org/abs/1612.00429) (Abbeel, Levine)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-3> (39:26) (Levine)

----
[DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475) (DeepMind)
  - `video` <https://youtube.com/watch?v=sZqrWFl0wQ4> (demo)

----
[Mutual Alignment Transfer Learning](https://arxiv.org/abs/1707.07907) (Abbeel)
>	"While the actions performed by the simulator policy can fail to accomplish the task on the robot, the sequence of states visited by the agent in simulation represents its task under limited variation in the system dynamics. We propose MATL, which instead of directly adapting the simulation policy, guides the exploration for both systems towards mutually aligned state distributions via auxiliary rewards. The method employs an adversarial approach to train policies with additional rewards based on confusing a discriminator with respect to the originating system for state sequences visited by the agents. By guiding the target agent on the robot towards states that the potentially more proficient source agent visits in simulation, we can accelerate training. In addition to aligning the robot policy to adapt to progress in simulation, we extend the approach to mutually align both systems which can be beneficial as the agent in simulation will be driven to explore better trajectories from states visited by the real-world policy."  
>	"We demonstrate that auxiliary rewards, which guide the exploration on the target platform, improve performance in environments with sparse rewards and can even guide the agent if only uninformative or no environment rewards at all are given for the target agent."  
>	"In addition to aligning the robot policy to adapt to progress in simulation, the reciprocal alignment of the simulation policy can be beneficial as the agent in simulation will be driven to explore better behaviours from states visited by the robot agent."  
  - <https://sites.google.com/view/matl> (demo)

[Distral: Robust Multitask Reinforcement Learning](https://arxiv.org/abs/1707.04175) (DeepMind)
>	"Two ideas might be worth reemphasizing here. We observe that distillation arises naturally as one half of an optimization procedure when using KL divergences to regularize the output of task models towards a distilled model. The other half corresponds to using the distilled model as a regularizer for training the task models. Another observation is that parameters in deep networks do not typically by themselves have any semantic meaning, so instead of regularizing networks in parameter space, it is worthwhile considering regularizing networks in a more semantically meaningful space, e.g. of policies."  

[Policy Distillation](http://arxiv.org/abs/1511.06295) (DeepMind)
>	"Our new paper uses distillation to consolidate lots of policies into a single deep network. This works remarkably well, and can be applied online, during Q-learning, so that policies are compressed, distilled, and refined whilst being learned. Atari policies are actually improved through distillation and generalize better (with higher scores and lower variance) during novel starting state evaluation."  

[Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning](http://arxiv.org/abs/1511.06342) (Salakhutdinov)
>	"single policy network learning to act in a set of distinct tasks through the guidance of an expert teacher for each task"  
  - `code` <https://github.com/eparisotto/ActorMimic>

----
[Deep Successor Reinforcement Learning](https://arxiv.org/abs/1606.02396)
  - `video` <https://youtube.com/watch?v=OCHwXxSW70o> (Kulkarni)
  - `video` <https://youtube.com/watch?v=kNqXCn7K-BM> (Garipov)
  - `code` <https://github.com/Ardavans/DSR>

[Learning to Act by Predicting the Future](https://arxiv.org/pdf/1611.01779)
>	"application of deep successor reinforcement learning"  
  - `video` <https://youtube.com/watch?v=947bSUtuSQ0> + <https://youtube.com/watch?v=947bSUtuSQ0> (demo)
  - `video` <https://youtube.com/watch?v=buUF5F8UCH8> (Lamb, Ozair)
  - `video` <https://youtube.com/watch?v=Q0ldKJbAwR8> (Dosovitskiy) (in russian)
  - `video` <https://yadi.sk/i/pMdw-_uI3Gke7Z> (1:02:03) (Shvechikov) (in russian)
  - `post` <https://oreilly.com/ideas/reinforcement-learning-for-complex-goals-using-tensorflow>
  - `notes` <https://blog.acolyer.org/2017/05/12/learning-to-act-by-predicting-the-future/>
  - `code` <https://github.com/IntelVCL/DirectFuturePrediction>

[Successor Features for Transfer in Reinforcement Learning](http://arxiv.org/abs/1606.05312) (DeepMind)

----
[Learning and Transfer of Modulated Locomotor Controllers](http://arxiv.org/abs/1610.05182) (DeepMind)
  - `video` <https://youtube.com/watch?v=sboPYvhpraQ> (demo)
  - `video` <https://youtube.com/watch?v=0e_uGa7ic74&t=31m4s> (Hadsell)

[Learning Modular Neural Network Policies for Multi-Task and Multi-Robot Transfer](http://arxiv.org/abs/1609.07088) (Abbeel, Levine)
  - `video` <https://youtube.com/watch?v=n4EgRwzJE1o>
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=56m20s> (Abbeel)

[Learning Invariant Feature Spaces to Transfer Skills with Reinforcement Learning](https://arxiv.org/abs/1703.02949) (Abbeel, Levine)

----
[Progressive Neural Networks](http://arxiv.org/abs/1606.04671) (DeepMind)
  - `video` <https://youtube.com/watch?v=aWAP_CWEtSI> (Hadsell)
  - `video` <http://techtalks.tv/talks/progressive-nets-for-sim-to-real-transfer-learning/63043/> (Hadsell)
  - `notes` <https://blog.acolyer.org/2016/10/11/progressive-neural-networks/>
  - `code` <https://github.com/synpon/prog_nn>



---
### reinforcement learning - imitation

[Inferring The Latent Structure of Human Decision-Making from Raw Visual Inputs](https://arxiv.org/abs/1703.08840)
  - `code` <https://github.com/YunzhuLi/InfoGAIL>

[Robust Imitation of Diverse Behaviors](https://arxiv.org/abs/1707.02747) (DeepMind)
>	"We develop a new version of GAIL that (1) is much more robust than the purely-supervised controller, especially with few demonstrations, and (2) avoids mode collapse, capturing many diverse behaviors when GAIL on its own does not."  
>	"The base of our model is a new type of variational autoencoder on demonstration trajectories that learns semantic policy embeddings, which can be smoothly interpolated with a resulting smooth interpolation of reaching behavior."  
  - `video` <https://youtube.com/watch?v=necs0XfnFno> (demo)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/> (1:16:00) (de Freitas)

[End-to-End Differentiable Adversarial Imitation Learning](http://proceedings.mlr.press/v70/baram17a.html)
>	"Model-free approach does not allow the system to be differentiable, which requires the use of high-variance gradient estimations."  
>	"We show how to use a forward model to make the system fully differentiable, which enables us to train policies using the stochastic gradient of discriminator."  
  - `slides` <http://icri-ci.technion.ac.il/files/2017/05/14-Shie-Mannor-170509.pdf>
  - `code` <https://github.com/itaicaspi/mgail>

[Generative Adversarial Imitation Learning](http://arxiv.org/abs/1606.03476)
>	"Uses a GAN framework to discriminate between teacher and student experience and force the student to behave close to the teacher."  
  - `video` <https://youtube.com/watch?v=bcnCo9RxhB8> (Ermon)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/> (1:09:43) (de Freitas)
  - `code` <https://github.com/openai/imitation>
  - `code` <https://github.com/DanielTakeshi/rl_algorithms/tree/master/il>

----
[One-Shot Visual Imitation Learning via Meta-Learning](https://arxiv.org/abs/1709.04905) (Abbeel, Levine)
  - `video` <https://youtu.be/lYU5nq0dAQQ?t=51m> (Levine)

[One-Shot Imitation Learning](http://arxiv.org/abs/1703.07326) (OpenAI)
  - `video` <http://bit.ly/one-shot-imitation> (demo)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/> (1:03:10) (de Freitas)

[Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards](https://arxiv.org/abs/1707.08817) (DeepMind)
>	"Our work combines imitation learning with learning from task rewards, so that the agent is able to improve upon the demonstrations it has seen."  
>	"Most work on RL in high-dimensional continuous control problems relies on well-tuned shaping rewards both for communicating the goal to the agent as well as easing the exploration problem. While many of these tasks can be defined by a terminal goal state fairly easily, tuning a proper shaping reward that does not lead to degenerate solutions is very difficult. This task only becomes more difficult when you move to multi-stage tasks such as insertion (stage for reaching the insertion point and stage for inserting the object). In this work, we replaced these difficult to tune shaping reward functions with demonstrations of the task from a human demonstrator. This eases the exploration problem without requiring careful tuning of shaping rewards."  
  - `video` <https://youtube.com/watch?v=Vno6FGqhvDc> (demo)

[Learning from Demonstrations for Real World Reinforcement Learning](https://arxiv.org/abs/1704.03732) (DeepMind + OpenAI)
  - `video` <https://youtube.com/playlist?list=PLdjpGm3xcO-0aqVf--sBZHxCKg-RZfa5T> (demo)
  - `code` <https://github.com/reinforceio/tensorforce/blob/master/tensorforce/models/dqfd_model.py>

[Imitation from Observation: Learning to Imitate Behaviors from Raw Video via Context Translation](https://arxiv.org/abs/1707.03374) (Abbeel, Levine)
  - `video` <https://youtube.com/watch?v=kJBRDhInbmU> (demo)

[Third Person Imitation Learning](https://arxiv.org/abs/1703.01703) (OpenAI)
>	"The authors propose a new framework for learning a policy from third-person experience. This is different from standard imitation learning which assumes the same "viewpoint" for teacher and student. The authors build upon Generative Adversarial Imitation Learning, which uses a GAN framework to discriminate between teacher and student experience and force the student to behave close to the teacher. However, when using third-person experience from a different viewpoint the discriminator would simply learn to discriminate between viewpoints instead of behavior and the framework isn't easily applicable. The authors' solution is to add a second discriminator to maximize a domain confusion loss based on the same feature representation. The objective is to learn the same (viewpoint-independent) feature representation for both teacher and student experience while also learning to discriminate between teacher and student observations. In other words, the objective is to maximize domain confusion while minimizing class loss. In practice, this is another discriminator term in the GAN objective. The authors also found that they need to feed observations at time t+n (n=4 in experiments) to signal the direction of movement in the environment."  
  - `video` <http://www.fields.utoronto.ca/video-archive/2017/02/2267-16530> (48:45) (Abbeel)
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/third-person-imitation-learning.md>

[Unsupervised Perceptual Rewards for Imitation Learning](http://arxiv.org/abs/1612.06699) (Levine)
>	"To our knowledge, these are the first results showing that complex robotic manipulation skills can be learned directly and without supervised labels from a video of a human performing the task."  

----
[Deeply AggreVaTeD: Differentiable Imitation Learning for Sequential Prediction](https://arxiv.org/abs/1703.01030)

[Query-Efficient Imitation Learning for End-to-End Autonomous Driving](https://arxiv.org/abs/1605.06450)  # SafeDAgger
  - `video` <https://youtu.be/soZXAH3leeQ?t=15m51s> (Cho)



---
### reinforcement learning - applications

[Deep Reinforcement Learning: An Overview](http://arxiv.org/abs/1701.07274)

[DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker](http://arxiv.org/abs/1701.01724) (Bowling)
  - `paper` <http://science.sciencemag.org/content/early/2017/03/01/science.aam6960>
  - `video` <https://youtube.com/playlist?list=PLX7NnbJAq7PlA2XpynViLOigzWtmr6QVZ> (demo matches)
  - <http://deepstack.ai>
  - <http://twitter.com/DeepStackAI>
  - `video` <https://youtu.be/02xIkHowQOk?t=11m45s> (Bowling)
  - `video` <https://youtube.com/watch?v=qndXrHcV1sM> (Bowling)
  - `video` <http://videolectures.net/aaai2017_bowling_sandholm_poker> (Bowling, Sandholm)
  - `code` <https://github.com/lifrordi/DeepStack-Leduc>

[Neural Combinatorial Optimization with Reinforcement Learning](http://arxiv.org/abs/1611.09940) (Google Brain)
  - `video` <https://youtube.com/watch?v=mxCVgVrUw50> (Bengio)

[Learning Runtime Parameters in Computer Systems with Delayed Experience Injection](http://arxiv.org/abs/1610.09903)

[Ask the Right Questions: Active Question Reformulation with Reinforcement Learning] (Google)
  - `video` <https://youtu.be/soZXAH3leeQ?t=34m43s> (Cho)

[Coarse-to-Fine Question Answering for Long Documents](http://arxiv.org/abs/1611.01839) (Google Research)

[Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning](http://arxiv.org/abs/1603.07954) (Barzilay)
  - `video` <https://youtu.be/k5KWUpqMO2U?t=47m37s> (Narasimhan)
  - `code` <https://github.com/karthikncode/DeepRL-InformationExtraction>

[Teaching Machines to Describe Images via Natural Language Feedback](https://arxiv.org/abs/1706.00130)

[Towards Deep Symbolic Reinforcement Learning](http://arxiv.org/abs/1609.05518)
  - `video` <https://youtube.com/watch?v=HOAVhPy6nrc> (Shanahan)

[Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) (DeepMind + OpenAI)
  - `post` <https://deepmind.com/blog/learning-through-human-feedback/>
  - `post` <https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/>
  - `code` <https://github.com/nottombrown/rl-teacher>

[Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
>	"We evaluate COMA in the testbed of StarCraft unit micromanagement, using a decentralised variant with significant partial observability. COMA significantly improves average performance over other multi-agent actor-critic methods in this setting, and the best performing agents are competitive with state-of-the-art centralised controllers that get access to the full state."  
  - `video` <https://youtube.com/watch?v=3OVvjE5B9LU> (Whiteson)



---
### language grounding

[A Paradigm for Situated and Goal-Driven Language Learning](https://arxiv.org/abs/1610.03585) (OpenAI)

[Natural Language Does Not Emerge ‘Naturally’ in Multi-Agent Dialog](https://arxiv.org/abs/1706.08502)

[Grounded Language Learning in a Simulated 3D World](https://arxiv.org/abs/1706.06551) (DeepMind)
>	"The agent learns simple language by making predictions about the world in which that language occurs, and by discovering which combinations of words, perceptual cues and action decisions result in positive outcomes. Its knowledge is distributed across language, vision and policy networks, and pertains to modifiers, relational concepts and actions, as well as concrete objects. Its semantic representations enable the agent to productively interpret novel word combinations, to apply known relations and modifiers to unfamiliar objects and to re-use knowledge pertinent to the concepts it already has in the process of acquiring new concepts."  
>	"While our simulations focus on language, the outcomes are relevant to machine learning in a more general sense. In particular, the agent exhibits active, multi-modal concept induction, the ability to transfer its learning and apply its knowledge representations in unfamiliar settings, a facility for learning multiple, distinct tasks, and the effective synthesis of unsupervised and reinforcement learning. At the same time, learning in the agent reflects various effects that are characteristic of human development, such as rapidly accelerating rates of vocabulary growth, the ability to learn from both rewarded interactions and predictions about the world, a natural tendency to generalise and re-use semantic knowledge, and improved outcomes when learning is moderated by curricula."  
  - `video` <https://youtube.com/watch?v=wJjdu1bPJ04> (demo)
  - `video` <http://videolectures.net/deeplearning2017_blunsom_language_processing/> (48:54) (Blunsom)

[Programmable Agents](https://arxiv.org/abs/1706.06383) (DeepMind)
  - `video` <https://youtube.com/playlist?list=PLs1LSEoK_daRDnPUB2u7VAXSonlNU7IcV> (demo)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/> (32:57) (de Freitas)
  - `code` <https://github.com/jaesik817/programmable-agents_tensorflow>

[Gated-Attention Architectures for Task-Oriented Language Grounding](https://arxiv.org/abs/1706.07230) (Salakhutdinov)

[Emergent Language in a Multi-Modal, Multi-Step Referential Game](https://arxiv.org/abs/1705.10369)

[Translating Neuralese](https://arxiv.org/abs/1704.06960)
>	"Authors take the vector messages (“neuralese”) passed between two machines trained to perform a collaborative task, and translate them into natural language utterances. To overcome the absence of neuralese-to-English parallel data, authors consider a pair of messages equivalent if they are used in similar scenarios by human and machine agents."  
  - `code` <http://github.com/jacobandreas/neuralese>

[Deal or No Deal? End-to-End Learning for Negotiation Dialogues](https://arxiv.org/abs/1706.05125) (Facebook AI Research)
  - `post` <https://code.facebook.com/posts/1686672014972296>
  - `video` <https://ku.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=e76f464c-6f81-4e31-b942-839312cf0f8c> (Lewis)
  - `code` <https://github.com/facebookresearch/end-to-end-negotiator>

[Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning](https://arxiv.org/abs/1703.06585)
  - `video` <https://youtube.com/watch?v=SztC8VOWwRQ> (demo)
  - `video` <https://youtube.com/watch?v=I9OlorMh7wU> (Das)
  - `video` <http://videolectures.net/deeplearning2017_parikh_batra_deep_rl/> (part 2, 25:47) (Batra)

[A Deep Compositional Framework for Human-like Language Acquisition in Virtual Environment](https://arxiv.org/abs/1703.09831) (Baidu)

[Emergence of Grounded Compositional Language in Multi-Agent Populations](http://arxiv.org/abs/1703.04908) (OpenAI)
>	"Though the agents come up with words that we found to correspond to objects and other agents, as well as actions like 'Look at' or 'Go to', to the agents these words are abstract symbols represented by one-hot vector - we label these one-hot vectors with English words that capture their meaning for the sake of interpretability."  
>
>	"One possible scenario is from goal oriented-dialog systems. Where one agent tries to transmit to another certain API call that it should perform (book restaurant, hotel, whatever). I think these models can make it more data efficient. At the first stage two agents have to communicate and discover their own language, then you can add regularization to make the language look more like natural language and on the final stage, you are adding a small amount of real data (dialog examples specific for your task). I bet that using additional communication loss will make the model more data efficient."  
>
>	"The big outcome to hunt for in this space is a post-gradient descent learning algorithm. Of course you can make agents that play the symbol grounding game, but it's not a very big step from there to compression of data, and from there to compression of 'what you need to know to solve the problem you're about to encounter' - at which point you have a system which can learn by training or learn by receiving messages. It was pretty easy to get stuff like one agent learning a classifier, encoding it in a message, and transmitting it to a second agent who has to use it for zero-shot classification. But it's still single-task specific communication, so there's no benefit to the agent for receiving, say, the messages associated with the previous 100 tasks. The tricky thing is going from there to something more abstract and cumulative, so that you can actually use message generation as an iterative learning mechanism. I think a large part of that difficulty is actually designing the task ensemble, not just the network architecture."  
  - `video` <https://youtube.com/watch?v=liVFy7ZO4OA> (demo)
  - `post` <https://blog.openai.com/learning-to-communicate/>
  - `video` <https://youtu.be/02xIkHowQOk?t=1h17m45s> (Lowe)
  - `video` <http://videos.re-work.co/videos/366-learning-to-communicate> (Lowe)
  - `video` <https://youtube.com/watch?v=f4gKhK8Q6mY&t=22m20s> (Abbeel)
  - `paper` ["A Paradigm for Situated and Goal-Driven Language Learning"](https://arxiv.org/abs/1610.03585)  

[Multi-Agent Cooperation and the Emergence of (Natural) Language](https://arxiv.org/abs/1612.07182) (Facebook AI Research)
  - `video` <https://facebook.com/iclr.cc/videos/1712966538732405/> (Peysakhovich)

[Learning Language Games through Interaction](http://arxiv.org/abs/1606.02447) (Liang)
  - `post` <http://nlp.stanford.edu/blog/interactive-language-learning/>
  - `video` <http://youtube.com/watch?v=PfW4_3tCiw0> (demo, calendar)
  - <http://shrdlurn.sidaw.xyz> (demo, blocks world)
  - `video` <https://youtube.com/watch?v=iuazFltYgCE> (Wang)
  - `video` <https://youtu.be/mhHfnhh-pB4?t=1h5m45s> (Liang)
  - `video` <https://youtu.be/6O5sttckalE?t=40m45s> (Liang)



---
### natural language processing

[Improving Neural Language Models with a Continuous Cache](http://arxiv.org/abs/1612.04426) (Facebook AI Research)  # adaptive softmax

[Learning to Compute Word Embeddings On the Fly](https://arxiv.org/abs/1706.00286)
  - `notes` <https://theneuralperspective.com/2017/06/05/more-on-embeddings-spring-2017/>

----
[Pointer Sentinel Mixture Models](http://arxiv.org/abs/1609.07843) (MetaMind)
>	"The authors combine a standard LSTM softmax with Pointer Networks in a mixture model called Pointer-Sentinel LSTM (PS-LSTM). The pointer networks helps with rare words and long-term dependencies but is unable to refer to words that are not in the input. The opposite is the case for the standard softmax."  
  - `video` <https://youtube.com/watch?v=Ibt8ZpbX3D8> (Merity)
  - `notes` <https://theneuralperspective.com/2016/10/04/pointer-sentinel-mixture-models/>

[Pointing the Unknown Words](http://arxiv.org/abs/1603.08148) (Bengio)

[Machine Comprehension Using Match-LSTM And Answer Pointer](http://arxiv.org/abs/1608.07905)

----
[Towards Universal Paraphrastic Sentence Embeddings](http://arxiv.org/abs/1511.08198)  # outperforming LSTM
  - `video` <http://videolectures.net/iclr2016_wieting_universal_paraphrastic/> (Wieting)

[Order-Embeddings of Images and Language](http://arxiv.org/abs/1511.06361)
  - `video` <http://videolectures.net/iclr2016_vendrov_order_embeddings/> (Vendrov)
  - `code` <https://github.com/ivendrov/order-embedding>
  - `code` <https://github.com/ivendrov/order-embeddings-wordnet>
  - `code` <https://github.com/LeavesBreathe/tensorflow_with_latest_papers/blob/master/partial_ordering_embedding.py>

----
[Bag of Tricks for Efficient Text Classification](http://arxiv.org/abs/1607.01759) (Facebook AI Research)  # fastText
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1607.01759#shagunsodhani>
  - `code` <https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py>

----
[Globally Normalized Transition-Based Neural Networks](http://arxiv.org/abs/1603.06042)  # SyntaxNet, Parsey McParseface
>	"The parser uses a feed forward NN, which is much faster than the RNN usually used for parsing. Also the paper is using a global method to solve the label bias problem. This method can be used for many tasks and indeed in the paper it is used also to shorten sentences by throwing unnecessary words. The label bias problem arises when predicting each label in a sequence using a softmax over all possible label values in each step. This is a local approach but what we are really interested in is a global approach in which the sequence of all labels that appeared in a training example are normalized by all possible sequences. This is intractable so instead a beam search is performed to generate alternative sequences to the training sequence. The search is stopped when the training sequence drops from the beam or ends. The different beams with the training sequence are then used to compute the global loss."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1603.06042>
  - `code` <https://github.com/tensorflow/models/tree/master/syntaxnet>

----
[Semantic Parsing with Semi-Supervised Sequential Autoencoders](http://arxiv.org/abs/1609.09315) (DeepMind)

[Open-Vocabulary Semantic Parsing with both Distributional Statistics and Formal Knowledge](http://arxiv.org/abs/1607.03542) (Gardner)

[Learning a Neural Semantic Parser from User Feedback](https://arxiv.org/abs/1704.08760)
>	"We learn a semantic parser for an academic domain from scratch by deploying an online system using our interactive learning algorithm. After three train-deploy cycles, the system correctly answered 63.51% of user’s questions. To our knowledge, this is the first effort to learn a semantic parser using a live system, and is enabled by our models that can directly parse language to SQL without manual intervention."

----
[Neural Variational Inference for Text Processing](http://arxiv.org/abs/1511.06038) (Blunsom)
  - `post` <http://dustintran.com/blog/neural-variational-inference-for-text-processing/>
  - `code` <https://github.com/carpedm20/variational-text-tensorflow>
  - `code` <https://github.com/cheng6076/NVDM>

[Discovering Discrete Latent Topics with Neural Variational Inference](https://arxiv.org/abs/1706.00359)

[Generating Sentences from a Continuous Space](http://arxiv.org/abs/1511.06349)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1511.06349>
  - `code` <https://github.com/analvikingur/pytorch_RVAE>
  - `code` <https://github.com/cheng6076/Variational-LSTM-Autoencoder>

[A Hybrid Convolutional Variational Autoencoder for Text Generation](http://arxiv.org/pdf/1702.02390)
  - `code` <https://github.com/stas-semeniuta/textvae>

[Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](http://arxiv.org/abs/1702.08139) (Salakhutdinov)

[Toward Controlled Generation of Text](http://arxiv.org/abs/1703.00955) (Salakhutdinov)



---
### information retrieval

[Ask the Right Questions: Active Question Reformulation with Reinforcement Learning](https://arxiv.org/abs/1705.07830) (Google)
  - `video` <https://youtu.be/soZXAH3leeQ?t=40m54s> (Cho)

[Task-Oriented Query Reformulation with Reinforcement Learning](https://arxiv.org/abs/1704.04572)
  - `video` <https://youtu.be/soZXAH3leeQ?t=34m16s> (Cho)
  - `video` <https://ku.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=c933f3da-392f-4aeb-bd09-e766a8ba83aa> (5:03:10) (Nogueira)
  - `code` <https://github.com/nyu-dl/QueryReformulator>

[Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051) (Facebook AI Research)  # DrQA
  - `code` <https://github.com/hitvoice/DrQA>

[R3: Reinforced Reader-Ranker for Open-Domain Question Answering](https://arxiv.org/abs/1709.00023) (IBM)



---
### dialog systems

[How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](http://arxiv.org/abs/1603.08023)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/LiuLSNCP16#shagunsodhani>

[On the Evaluation of Dialogue Systems with Next Utterance Classification](http://arxiv.org/abs/1605.05414)

[Towards an Automatic Turing Test: Learning to Evaluate Dialogue Responses](https://arxiv.org/abs/1708.07149)
  - `video` <https://youtube.com/watch?v=vTgwWobuoFw> (Pineau)

----
[A Deep Reinforcement Learning Chatbot](https://arxiv.org/abs/1709.02349) (Bengio)

----
[Learning from Real Users: Rating Dialogue Success with Neural Networks for Reinforcement Learning in Spoken Dialogue Systems](http://arxiv.org/abs/1508.03386) (Young)

[On-line Active Reward Learning for Policy Optimisation in Spoken Dialogue Systems](http://arxiv.org/abs/1605.07669) (Young)

[Continuously Learning Neural Dialogue Management](http://arxiv.org/abs/1606.02689) (Young)

[Online Sequence-to-Sequence Reinforcement Learning for Open-domain Conversational Agents](http://arxiv.org/abs/1612.03929)

----
[Generative Deep Neural Networks for Dialogue: A Short Review](http://arxiv.org/abs/1611.06216) (Pineau)

[Emulating Human Conversations using Convolutional Neural Network-based IR](http://arxiv.org/abs/1606.07056)

[Two are Better than One: An Ensemble of Retrieval- and Generation-Based Dialog Systems](http://arxiv.org/abs/1610.07149)

[Machine Comprehension by Text-to-Text Neural Question Generation](https://arxiv.org/abs/1705.02012) (Maluuba)
  - `video` <https://youtube.com/watch?v=UIzcIC5RQN8>

----
[Latent Intention Dialogue Models](https://arxiv.org/abs/1705.10229) (Young)
>	"Learning an end-to-end dialogue system is appealing but challenging because of the credit assignment problem. Discrete latent variable dialogue models such as LIDM are attractive because the latent variable can serve as an interface for decomposing the learning of language and the internal dialogue decision-making. This decomposition can effectively help us resolve the credit assignment problem where different learning signals can be applied to different sub-modules to update the parameters. In variational inference for discrete latent variables, the latent distribution is basically updated by the reward from the variational lower bound. While in reinforcement learning, the latent distribution (i.e. policy network) is updated by the rewards from dialogue success and sentence BLEU score. Hence, the latent variable bridges the different learning paradigms such as Bayesian learning and reinforcement learning and brings them together under the same framework. This framework provides a more robust neural network-based approach than previous approaches because it does not depend solely on sequence-to-sequence learning but instead explicitly models the hidden dialogue intentions underlying the user’s utterances and allows the agent to directly learn a dialogue policy through interaction."  

[Hybrid Code Networks: Practical and Efficient End-to-end Dialog Control with Supervised and Reinforcement Learning](https://arxiv.org/abs/1702.03274) (Zweig)
>	"End-to-end methods lack a general mechanism for injecting domain knowledge and constraints. For example, simple operations like sorting a list of database results or updating a dictionary of entities can expressed in a few lines of software, yet may take thousands of dialogs to learn. Moreover, in some practical settings, programmed constraints are essential – for example, a banking dialog system would require that a user is logged in before they can retrieve account information."  
>	"In addition to learning an RNN, HCNs also allow a developer to express domain knowledge via software and action templates."  

[Adversarial Learning for Neural Dialogue Generation](http://arxiv.org/abs/1701.06547) (Jurafsky)
  - `code` <https://github.com/jiweil/Neural-Dialogue-Generation>

[End-to-End Reinforcement Learning of Dialogue Agents for Information Access](http://arxiv.org/abs/1609.00777) (Deng)

[Efficient Exploration for Dialog Policy Learning with Deep BBQ Networks & Replay Buffer Spiking](http://arxiv.org/abs/1608.05081) (Deng)

[Neural Belief Tracker: Data-Driven Dialogue State Tracking](http://arxiv.org/abs/1606.03777) (Young)

[Policy Networks with Two-Stage Training for Dialogue Systems](http://arxiv.org/abs/1606.03152) (Maluuba)
  - `post` <http://www.maluuba.com/blog/2016/11/23/deep-reinforcement-learning-in-dialogue-systems>

----
[Deep Reinforcement Learning for Dialogue Generation](http://arxiv.org/abs/1606.01541) (Jurafsky)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=conf/emnlp/LiMRJGG16>

[End-to-End LSTM-based Dialog Control Optimized with Supervised and Reinforcement Learning](http://arxiv.org/abs/1606.01269) (Zweig)

[Learning End-to-End Goal-Oriented Dialog](http://arxiv.org/abs/1605.07683) (Weston)
  - `video` <https://facebook.com/iclr.cc/videos/1712966538732405/> (27:39) (Boureau)

[A Network-based End-to-End Trainable Task-oriented Dialogue System](http://arxiv.org/abs/1604.04562) (Young)
  - `video` <http://videolectures.net/deeplearning2016_wen_network_based/> (Wen)

[Towards Conversational Recommender Systems](http://kdd.org/kdd2016/subtopic/view/towards-conversational-recommender-systems) (Hoffman)
  - `video` <https://periscope.tv/WiMLworkshop/1vAGRXDbvbkxl> (Christakopoulou)
  - `video` <https://youtube.com/watch?v=nLUfAJqXFUI> (Christakopoulou)

----
[A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue](http://arxiv.org/abs/1701.04024) (Manning)
  - `notes` <https://medium.com/@sharaf/a-paper-a-day-14-a-copy-augmented-sequence-to-sequence-architecture-gives-good-performance-on-44727e880044>

[Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation](http://arxiv.org/abs/1606.00776) (Bengio)

[An Attentional Neural Conversation Model with Improved Specificity](http://arxiv.org/abs/1606.01292) (Zweig)

[A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues](http://arxiv.org/abs/1605.06069) (Bengio)
  - `slides` <http://cs.mcgill.ca/~rlowe1/problem_with_neural_chatbots.pdf> (Lowe)

----
[LSTM-based Mixture-of-Experts for Knowledge-Aware Dialogues](http://arxiv.org/abs/1605.01652)

[Multi-domain Neural Network Language Generation for Spoken Dialogue Systems](http://arxiv.org/abs/1603.01232)

[Sentence Level Recurrent Topic Model: Letting Topics Speak for Themselves](http://arxiv.org/abs/1604.02038)

[Context-aware Natural Language Generation with Recurrent Neural Networks](http://arxiv.org/abs/1611.09900)

[Data Distillation for Controlling Specificity in Dialogue Generation](http://arxiv.org/abs/1702.06703) (Jurafsky)

----
[A Persona-Based Neural Conversation Model](http://arxiv.org/abs/1603.06155)
  - `code` <https://github.com/jiweil/Neural-Dialogue-Generation>

[Conversational Contextual Cues: The Case of Personalization and History for Response Ranking](http://arxiv.org/abs/1606.00372) (Kurzweil)

[A Sequence-to-Sequence Model for User Simulation in Spoken Dialogue Systems](http://arxiv.org/abs/1607.00070) (Maluuba)

----
[Deep Contextual Language Understanding in Spoken Dialogue Systems](http://research.microsoft.com/apps/pubs/default.aspx?id=256085)

[Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning](http://arxiv.org/abs/1603.07954) (Barzilay)
