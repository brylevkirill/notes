interesting recent papers:

  * [theory](#theory)  
  * [compute and memory architectures](#compute-and-memory-architectures)  
  * [meta-learning](#meta-learning)  
  * [one-shot learning](#one-shot-learning)  
  * [unsupervised learning](#unsupervised-learning)  
  * [generative models](#generative-models)  
    - [generative adversarial networks](#generative-models---generative-adversarial-networks)  
    - [variational autoencoders](#generative-models---variational-autoencoders)  
    - [autoregressive models](#generative-models---autoregressive-models)  
  * [probabilistic inference](#probabilistic-inference)  
  * [reasoning](#reasoning)  
  * [program induction](#program-induction)  
  * [reinforcement learning](#reinforcement-learning---algorithms)  
    - [algorithms](#reinforcement-learning---algorithms)  
    - [exploration and intrinsic motivation](#reinforcement-learning---exploration-and-intrinsic-motivation)  
    - [abstractions for states and actions](#reinforcement-learning---abstractions-for-states-and-actions)  
    - [planning](#reinforcement-learning---simulation-and-planning)  
    - [transfer](#reinforcement-learning---transfer)  
    - [imitation](#reinforcement-learning---imitation)  
    - [memory](#reinforcement-learning---memory)  
    - [applications](#reinforcement-learning---applications)  
  * [dialog systems](#dialog-systems)  
  * [natural language processing](#natural-language-processing)  

----
interesting papers:

  - [artificial intelligence](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#interesting-papers)  
  - [knowledge representation and reasoning](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers)  
  - [machine learning](https://github.com/brylevkirill/notes/blob/master/Machine%20Learning.md#interesting-papers)  
  - [deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers)  
  - [reinforcement learning](https://dropbox.com/s/dexryjnmxujdynd/Reinforcement%20Learning.txt#interesting-papers)  
  - [bayesian inference and learning](https://dropbox.com/s/7vlg0vhb51rd6c1/Bayesian%20Inference%20and%20Learning.txt#interesting-papers)  
  - [probabilistic programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md#interesting-papers)  
  - [natural language processing](https://dropbox.com/s/0kw1s9mrrcwct0u/Natural%20Language%20Processing.txt#interesting-papers)  
  - [information retrieval](https://dropbox.com/s/21ugi2p9uy1shvt/Information%20Retrieval.txt#interesting-papers)  
  - [personal assistants](https://dropbox.com/s/0fyarlwcfb8mjdq/Personal%20Assistants.txt#interesting-papers)  



---
### theory

[Understanding Deep Learning Requires Rethinking Generalization](http://arxiv.org/abs/1611.03530) (Google Brain)  
>	"1. The effective capacity of neural networks is large enough for a brute-force memorization of the entire data set.  
>	 2. Even optimization on random labels remains easy. In fact, training time increases only by a small constant factor compared with training on the true labels.  
>	 3. Randomizing labels is solely a data transformation, leaving all other properties of the learning problem unchanged."  
>	"It is likely that learning in the traditional sense still occurs in part, but it appears to be deeply intertwined with massive memorization. Classical approaches are therefore poorly suited for reasoning about why these models generalize well."  
>
>	"Deep Learning networks are just massive associative memory stores! Deep Learning networks are capable of good generalization even when fitting random data. This is indeed strange in that many arguments for the validity of Deep Learning is on the conjecture that ‘natural’ data tends to exists in a very narrow manifold in multi-dimensional space. Random data however does not have that sort of tendency."  

[Opening the Black Box of Deep Neural Networks via Information](http://arxiv.org/abs/1703.00810)  
>	"DNNs with SGD have two phases: error minimization, then representation compression"  

[Capacity and Trainability in Recurrent Neural Networks](http://arxiv.org/abs/1611.09913) (Google Brain)  
>	"RNNs can store an amount of task information which is linear in the number of parameters, and is approximately 5 bits per parameter.  
>	RNNs can additionally store approximately one real number from their input history per hidden unit."  



---
### compute and memory architectures

[Hybrid Computing using a Neural Network with Dynamic External Memory](http://www.nature.com.sci-hub.cc/nature/journal/vaop/ncurrent/full/nature20101.html) (DeepMind)  
  - <https://deepmind.com/blog/differentiable-neural-computers/>  
  - <https://youtube.com/watch?v=steioHoiEms> (Graves)  
  - <https://youtube.com/watch?v=PQrlOjj8gAc> (Wayne) 
  - <https://youtu.be/otRoAQtc5Dk?t=59m56s> (Polykovskiy)  
  - <https://github.com/yos1up/DNC>  
  - <https://github.com/Mostafa-Samir/DNC-tensorflow>  
  - <https://github.com/frownyface/dnc>  
  - <https://github.com/khaotik/dnc-theano>  

[Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes](http://arxiv.org/abs/1610.09027) (DeepMind)  # improved Differentiable Neural Computer  

[Dynamic Neural Turing Machine with Soft and Hard Addressing Schemes](http://arxiv.org/abs/1607.00036) (Bengio)  

[Hierarchical Memory Networks](http://arxiv.org/abs/1605.07427) (Bengio)  

[Learning Efficient Algorithms with Hierarchical Attentive Memory](http://arxiv.org/abs/1602.03218) (DeepMind)  
>	"We show that an LSTM network augmented with HAM can learn algorithms for problems like merging, sorting or binary searching from pure input-output examples."  
>	"We also show that HAM can be trained to act like classic data structures: a stack, a FIFO queue and a priority queue."  
>	"Our model may be seen as a special case of Gated Graph Neural Network"  

[Neural Random-Access Machines](http://arxiv.org/abs/1511.06392) (Sutskever)  

----
[Associative Long Short-Term Memory](http://arxiv.org/abs/1602.03032) (Graves)  
  - <http://techtalks.tv/talks/associative-long-short-term-memory/62525/> (Danihelka)  
  - <http://www.cogsci.ucsd.edu/~sereno/170/readings/06-Holographic.pdf>  
  - <https://github.com/mohammadpz/Associative_LSTM>  

[Using Fast Weights to Attend to the Recent Past](http://arxiv.org/abs/1610.06258) (Hinton)  # alternative to LSTM  
>	(Hinton) "It's a different approach to a Neural Turing Machine. It does not require any decisions about where to write stuff or where to read from. Anything that happened recently can automatically be retrieved associatively. Fast associative memory should allow neural network models of sequential human reasoning."  
  - <https://drive.google.com/file/d/0B8i61jl8OE3XdHRCSkV1VFNqTWc> (Hinton)  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Using-Fast-Weights-to-Attend-to-the-Recent-Past> (Ba)  
  - <http://www.fields.utoronto.ca/talks/title-tba-337> (Hinton)  
  - <https://youtube.com/watch?v=mrj_hyH974o> (Novikov, in russian)  
  - <http://www.shortscience.org/paper?bibtexKey=journals/corr/1610.06258>  
  - <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/fast-weight-to-attend.md>  
  - <https://reddit.com/r/MachineLearning/comments/58qjiw/research161006258_using_fast_weights_to_attend_to/d92kctk/>  
  - <https://theneuralperspective.com/2016/12/04/implementation-of-using-fast-weights-to-attend-to-the-recent-past/>  
  - <https://github.com/ajarai/fast-weights>  
  - <https://github.com/jxwufan/AssociativeRetrieval>  

----
[Overcoming Catastrophic Forgetting in Neural Networks](http://arxiv.org/abs/1612.00796) (DeepMind)  
>	"The Mixture of Experts Layer is trained using back-propagation. The Gating Network outputs an (artificially made) sparse vector that acts as a chooser of which experts to consult. More than one expert can be consulted at once (although the paper doesn’t give any precision on the optimal number of experts). The Gating Network also decides on output weights for each expert."  
>
>	Huszar:  
>	"on-line sequential (diagonalized) Laplace approximation of Bayesian learning"  
>	"EWC makes sense for any neural network (indeed, any parametric model, really), virtually any task. Doesn't have to be DQN and in fact the paper itself shows examples with way simpler tasks."  
>	"The quadratic penalty/penalties prevent the network from forgetting what it has learnt from previous data - you can think of the quadratic penalty as a summary of the information from the data it has seen so far."  
>	"You can apply it at the level of learning tasks sequentially, or you can even apply it to on-line learning in a single task (in case you can't loop over the same minibatches several time like you do in SGD)."  
  - <http://www.pnas.org/content/early/2017/03/13/1611835114.abstract>  
  - <http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/main.html>  
  - <http://inference.vc/comment-on-overcoming-catastrophic-forgetting-in-nns-are-multiple-penalties-needed-2/>  

[PathNet: Evolution Channels Gradient Descent in Super Neural Networks](http://arxiv.org/abs/1701.08734) (DeepMind)  

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](http://openreview.net/forum?id=B1ckMDqlg) (Google Brain)  
>	"The MoE with experts shows higher accuracy (or lower perplexity) than the state of the art using only 16% of the training time."  
  - <https://github.com/dennybritz/deeplearning-papernotes/blob/3718d181a0fed5ed806582822ed0dbde530122bf/notes/mixture-experts.md>  

----
[Adaptive Computation Time for Recurrent Neural Networks](http://arxiv.org/abs/1603.08983) (Graves)  
  - <http://distill.pub/2016/augmented-rnns/>  
  - <https://www.evernote.com/shard/s189/sh/fd165646-b630-48b7-844c-86ad2f07fcda/c9ab960af967ef847097f21d94b0bff7>  
  - <https://github.com/DeNeutoy/act-tensorflow>  

[Memory-Efficient Backpropagation Through Time](http://arxiv.org/abs/1606.03401) (Graves)  

[Hierarchical Multiscale Recurrent Neural Networks](http://arxiv.org/abs/1609.01704) (Bengio)  
  - <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/hm-rnn.md>  
  - <https://medium.com/@jimfleming/notes-on-hierarchical-multiscale-recurrent-neural-networks-7362532f3b64>  

[Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences](http://arxiv.org/abs/1610.09513)  
>	"If you take an LSTM and add a “time gate” that controls at what frequency to be open to new input and how long to be open each time, you can have different neurons that learn to look at a sequence with different frequencies, create a “wormhole” for gradients, save compute, and do better on long sequences and when you need to process inputs from multiple sensors that are sampled at different rates."  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Phased-LSTM-Accelerating-Recurrent-Network-Training-for-Long-or-Event-based-Sequences> (Neil)  
  - <https://github.com/dannyneil/public_plstm>  

----
[Decoupled Neural Interfaces using Synthetic Gradients](http://arxiv.org/abs/1608.05343) (DeepMind)  
>	"At the very least it can allow individual modules to do gradient updates before waiting for the backward pass to reach them. So you could get better GPGPU utilization when the ordinary 'locked' mode of forward-then-backward doesn't always saturate the available compute units.  
>	Put differently, if you consider the dependency DAG of tensor operations, using these DNI things reduces the depth of the parameter gradient nodes (which is the whole point of training) in the DAG. So for example, the gradient update for the layer at the beginning of a n-layer chain goes from depth ~2n to depth ~1, the layer at the end has depth n, which doesn't change. On average, the depth of the gradient computation nodes is about 40% of what it would be normally, for deep networks. So there is a lot more flexibility for scheduling nodes in time and space.  
>	And for coarser-grained parallelism it could allow modules running on different devices to do updates before a final loss gradient is available to be distributed to all the devices. Synchronization still has to happen to update the gradient predictions, but that can happen later, and could even be opportunistic (asynchronous or stochastic)."  
>	"I guess that the synthetic gradients conditioned on the labels and the synthetic layer inputs conditioned on the data work for the same reason why stochastic depth works: during training, at any given layer the networks before and after it can be approximated by simpler, shallower versions. In stochastic depth the approximation is performed by skipping layers, so the whole network is approximated by a shallower version of itself, which changes at each step. In this work, instead, the approximation is performed by separate networks.  
  - <https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/>  
  - <https://iamtrask.github.io/2017/03/21/synthetic-gradients/>  
  - <http://cnichkawde.github.io/SyntheticGradients.html>  

[Understanding Synthetic Gradients and Decoupled Neural Interfaces](http://arxiv.org/abs/1703.00522) (DeepMind)  



---
### meta-learning

[Learning to Learn by Gradient Descent by Gradient Descent](http://arxiv.org/abs/1606.04474) (de Freitas)  
>	"Take some computation where you usually wouldn’t keep around intermediate states, such as a planning computation (say value iteration, where you only keep your most recent estimate of the value function) or stochastic gradient descent (where you only keep around your current best estimate of the parameters). Now keep around those intermediate states as well, perhaps reifying the unrolled computation in a neural net, and take gradients to optimize the entire computation with respect to some loss function. Instances: Value Iteration Networks, Learning to learn by gradient descent by gradient descent."  
  - <https://youtu.be/tPWGGwmgwG0?t=10m50s> (de Freitas)  
  - <https://youtu.be/x1kf4Zojtb0?t=1h4m53s> (de Freitas)  
  - <https://blog.acolyer.org/2017/01/04/learning-to-learn-by-gradient-descent-by-gradient-descent/>  
  - <https://hackernoon.com/learning-to-learn-by-gradient-descent-by-gradient-descent-4da2273d64f2>
  - <https://github.com/deepmind/learning-to-learn>  

[Learning to Learn for Global Optimization of Black Box Functions](http://arxiv.org/abs/1611.03824) (de Freitas)  

[Learned Optimizers that Scale and Generalize](http://arxiv.org/abs/1703.04813) (de Freitas)  

[Optimization as a Model for Few-Shot Learning](https://openreview.net/forum?id=rJY0-Kcll) (Larochelle)  

----
[RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](http://arxiv.org/abs/1611.02779) (OpenAI)  
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
>
>	"future directions:  
>	- better outer-loop algorithms  
>	- scaling RL^2 to 1M games  
>	- model-based RL^2  
>	- curriculum learning / universal RL^2  
>	- RL^2 + one-shot imitation learning  
>	- RL^2 for simulation -> real world transfer"  
  - <https://youtu.be/BskhUBPRrqE?t=6m28s> + <https://youtu.be/19eNQ1CLt5A?t=7m52s> (Sutskever)  
  - <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)  
  - <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/RL2-Fast_Reinforcement_Learning_via_Slow_Reinforcement_Learning.md>  

[Learning to Reinforcement Learn](http://arxiv.org/abs/1611.05763) (DeepMind)  
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
  - <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)  
  - <https://hackernoon.com/learning-policies-for-learning-policies-meta-reinforcement-learning-rl²-in-tensorflow-b15b592a2ddf> (Juliani)  
  - <https://github.com/awjuliani/Meta-RL>  

----
[HyperNetworks](http://arxiv.org/abs/1609.09106) (Google Brain)  
>	"Our main result is that hypernetworks can generate non-shared weights for LSTM and achieve near state-of-the-art results on a variety of sequence modelling tasks including character-level language modelling, handwriting generation and neural machine translation, challenging the weight-sharing paradigm for recurrent networks."  
>	"Our results also show that hypernetworks applied to convolutional networks still achieve respectable results for image recognition tasks compared to state-of-the-art baseline models while requiring fewer learnable parameters."  
  - <http://blog.otoro.net/2016/09/28/hyper-networks/>  

[Neural Architecture Search with Reinforcement Learning](http://arxiv.org/abs/1611.01578) (Google Brain)  
  - <https://youtube.com/watch?v=XDtFXBYpl1w> (Le)  

[Designing Neural Network Architectures using Reinforcement Learning](http://arxiv.org/abs/1611.02167)  



---
### one-shot learning

[Learning to Remember Rare Events](http://arxiv.org/abs/1703.03129) (Google Brain)  
  - <https://github.com/tensorflow/models/tree/master/learning_to_remember_rare_events>  

----
[One-Shot Generalization in Deep Generative Models](http://arxiv.org/abs/1603.05106)  
>	"move over DRAW: deepmind's latest has spatial-transform attention and 1-shot generalization"  
  - <http://youtube.com/watch?v=TpmoQ_j3Jv4> (demo)  
  - <http://techtalks.tv/talks/one-shot-generalization-in-deep-generative-models/62365/>  
  - <https://youtu.be/XpIDCzwNe78?t=43m> (Bartunov)  

[Towards a Neural Statistician](http://arxiv.org/abs/1606.02185)  
  - <http://techtalks.tv/talks/neural-statistician/63048/> (Edwards)  
  - <https://youtu.be/XpIDCzwNe78?t=51m53s> (Bartunov)  
  - <http://www.shortscience.org/paper?bibtexKey=journals/corr/1606.02185>  

[Matching Networks for One Shot Learning](http://arxiv.org/abs/1606.04080) (Vinyals)  
>	"Given just a few, or even a single, examples of an unseen class, it is possible to attain high classification accuracy on ImageNet using Matching Networks.  The core architecture is simple and straightforward to train and performant across a range of image and text classification tasks. Matching Networks are trained in the same way as they are tested: by presenting a series of instantaneous one shot learning training tasks, where each instance of the training set is fed into the network in parallel. Matching Networks are then trained to classify correctly over many different input training sets. The effect is to train a network that can classify on a novel data set without the need for a single step of gradient descent."  
  - <https://pbs.twimg.com/media/Cy7Eyh5WgAAZIw2.jpg:large>  
  - <https://blog.acolyer.org/2017/01/03/matching-networks-for-one-shot-learning/>  

[Fast Adaptation in Generative Models with Generative Matching Networks](http://arxiv.org/abs/1612.02192) (Bartunov)  
  - <https://youtu.be/XpIDCzwNe78> (Bartunov)  
  - <http://github.com/sbos/gmn>  

----
[Active One-shot Learning](https://cs.stanford.edu/~woodward/papers/active_one_shot_learning_2016.pdf)  
  - <https://youtube.com/watch?v=CzQSQ_0Z-QU> (Woodward)  



---
### unsupervised learning

[Early Visual Concept Learning with Unsupervised Deep Learning](http://arxiv.org/abs/1606.05579) (DeepMind)  
  - <https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results>  

[beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](http://openreview.net/forum?id=Sy2fzU9gl) (DeepMind)  
>	"This paper proposes a modification of the variational ELBO in encourage 'disentangled' representations, and proposes a measure of disentanglement."  
  - <http://tinyurl.com/jgbyzke> (demo)  

----
[Towards Conceptual Compression](http://arxiv.org/abs/1604.08772) (DeepMind)  
  - <https://pbs.twimg.com/media/Cy3pYfWWIAA_C9h.jpg:large>  

[Attend, Infer, Repeat: Fast Scene Understanding with Generative Models](http://arxiv.org/abs/1603.08575) (DeepMind)  
>	"The latent variables can be a list or set of vectors."  
>	"Consider the task of clearing a table after dinner. To plan your actions you will need to determine which objects are present, what classes they belong to and where each one is located on the table. In other words, for many interactions with the real world the perception problem goes far beyond just image classification. We would like to build intelligent systems that learn to parse the image of a scene into objects that are arranged in space, have visual and physical properties, and are in functional relationships with each other. And we would like to do so with as little supervision as possible. Starting from this notion our paper presents a framework for efficient inference in structured, generative image models that explicitly reason about objects. We achieve this by performing probabilistic inference using a recurrent neural network that attends to scene elements and processes them one at a time. Crucially, the model itself learns to choose the appropriate number of inference steps. We use this scheme to learn to perform inference in partially specified 2D models (variable-sized variational auto-encoders) and fully specified 3D models (probabilistic renderers). We show that such models learn to identify multiple objects - counting, locating and classifying the elements of a scene - without any supervision, e.g., decomposing 3D images with various numbers of objects in a single forward pass of a neural network."  
  - <https://youtube.com/watch?v=4tc84kKdpY4> (demo)  
  - <http://arkitus.com/attend-infer-repeat/>  
  - <http://www.shortscience.org/paper?bibtexKey=journals/corr/EslamiHWTKH16> (Larochelle)  

----
[Generative Temporal Models with Memory](http://arxiv.org/abs/1702.04649) (DeepMind)  
>	"A sufficiently powerful temporal model should separate predictable elements of the sequence from unpredictable elements, express uncertainty about those unpredictable elements, and rapidly identify novel elements that may help to predict the future. To create such models, we introduce Generative Temporal Models augmented with external memory systems."  
  - <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1702.04649>  

----
[Inducing Interpretable Representations with Variational Autoencoders](http://arxiv.org/abs/1611.07492) (Goodman)  

[Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders](http://arxiv.org/abs/1611.02648) (Arulkumaran)  
  - <http://ruishu.io/2016/12/25/gmvae/>  

[Disentangling Factors of Variation in Deep Representations using Adversarial Training](http://arxiv.org/abs/1611.03383) (LeCun)  
  - <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1611.03383>  
  - <http://www.shortscience.org/paper?bibtexKey=conf%2Fnips%2FMathieuZZRSL16>  

----
[Density Estimation using Real NVP](http://arxiv.org/abs/1605.08803)  
>	"Most interestingly, it is the only powerful generative model I know that combines A) a tractable likelihood, B) an efficient / one-pass sampling procedure and C) the explicit learning of a latent representation."  
  - <http://www-etud.iro.umontreal.ca/~dinhlaur/real_nvp_visual/> (demo)  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Phased-LSTM-Accelerating-Recurrent-Network-Training-for-Long-or-Event-based-Sequences> (08:19, Dinh) + <https://docs.google.com/presentation/d/152NyIZYDRlYuml5DbBONchJYA7AAwlti5gTWW1eXlLM/>  
  - <https://periscope.tv/hugo_larochelle/1ypKdAVmbEpGW> (Dinh)  
  - <http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.08803>  
  - <https://github.com/tensorflow/models/tree/master/real_nvp>  
  - <https://github.com/taesung89/real-nvp>  



---
### generative models

[A Note on the Evaluation of Generative Models](http://arxiv.org/abs/1511.01844)  
  - <http://videolectures.net/iclr2016_theis_generative_models/> (Theis)  
  - <https://pbs.twimg.com/media/CjA02jrWYAECWOZ.jpg:large> ("The generative model on the left gets a better log-likelihood score.")  

[On the Quantitative Analysis of Decoder-based Generative Models](http://arxiv.org/abs/1611.04273) (Salakhutdinov)  
>	"We propose to use Annealed Importance Sampling for evaluating log-likelihoods for decoder-based models and validate its accuracy using bidirectional Monte Carlo. Using this technique, we analyze the performance of decoder-based models, the effectiveness of existing log-likelihood estimators, the degree of overfitting, and the degree to which these models miss important modes of the data distribution."  
>	"This paper introduces Annealed Importance Sampling to compute tighter lower bounds and upper bounds for any generative model (with a decoder)."  
  - <https://github.com/tonywu95/eval_gen>  

[How (not) to train your generative model: schedule sampling, likelihood, adversary](http://arxiv.org/abs/1511.05101) (Huszar)  
  - <http://inference.vc/how-to-train-your-generative-models-why-generative-adversarial-networks-work-so-well-2/>  



---
### generative models - generative adversarial networks

[NIPS 2016 Tutorial: Generative Adversarial Networks](http://arxiv.org/abs/1701.00160) (Goodfellow)  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks> (Goodfellow) + <http://iangoodfellow.com/slides/2016-12-04-NIPS.pdf>  

----
[A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models](https://arxiv.org/abs/1611.03852) (Abbeel, Levine)  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-3> (33:17, Levine)  
  - <https://youtu.be/RZOKRFBtSh4?t=10m48s> (Finn)  
  - <http://pemami4911.github.io/paper-summaries/2017/02/12/gans-irl-ebm.html>  

----
[Learning in Implicit Generative Models](http://arxiv.org/abs/1610.03483) (Mohamed)  

[Variational Inference using Implicit Distributions](http://arxiv.org/abs/1702.08235) (Huszar)  
>	"This paper provides a unifying review of existing algorithms establishing connections between variational autoencoders, adversarially learned inference, operator VI, GAN-based image reconstruction, and more."  
  - <http://inference.vc/variational-inference-with-implicit-probabilistic-models-part-1-2/>  
  - <http://inference.vc/variational-inference-with-implicit-models-part-ii-amortised-inference-2/>  
  - <http://inference.vc/variational-inference-using-implicit-models-part-iii-joint-contrastive-inference-ali-and-bigan/>  
  - <http://inference.vc/variational-inference-using-implicit-models-part-iv-denoisers-instead-of-discriminators/>  

[Deep and Hierarchical Implicit Models](http://arxiv.org/abs/1702.08896) (Blei)
>	"We develop likelihood-free variational inference (LFVI). Key to LFVI is specifying a variational family that is also implicit. This matches the model's flexibility and allows for accurate approximation of the posterior. Our work scales up implicit models to sizes previously not possible and advances their modeling design."
  - <http://dustintran.com/blog/deep-and-hierarchical-implicit-models>  

----
[f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](http://arxiv.org/abs/1606.00709)  
>	"Shows how to optimize many different objectives using adversarial training."  
  - <https://youtube.com/watch?v=I1M_jGWp5n0>  
  - <https://youtube.com/watch?v=kQ1eEXgGsCU> (Nowozin)  
  - <https://github.com/wiseodd/generative-models/tree/master/GAN/f_gan>  

[Improved Generator Objectives for GANs](http://arxiv.org/abs/1612.02780) (Google Brain)  
>	"We present a framework to understand GAN training as alternating density ratio estimation and approximate divergence minimization. This provides an interpretation for the mismatched GAN generator and discriminator objectives often used in practice, and explains the problem of poor sample diversity. We also derive a family of generator objectives that target arbitrary f-divergences without minimizing a lower bound, and use them to train generative image models that target either improved sample quality or greater sample diversity."  

[Revisiting Classifier Two-Sample Tests for GAN Evaluation and Causal Discovery](http://arxiv.org/abs/1610.06545) (Facebook)  

----
[Towards Principled Methods for Training Generative Adversarial Networks](https://arxiv.org/abs/1701.04862) (Facebook)  

[Generalization and Equilibrium in Generative Adversarial Nets](https://arxiv.org/abs/1703.00573)  
  - <https://youtube.com/watch?v=V7TliSCqOwI> (Arora)
  - <http://www.offconvex.org/2017/03/30/GANs2/> (Arora)

----
[Wasserstein GAN](https://arxiv.org/abs/1701.07875) (Facebook)  
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
  - <https://youtube.com/watch?v=DfJeaa--xO0&t=26m27s> (Bottou)
  - <http://www.alexirpan.com/2017/02/22/wasserstein-gan.html>  
  - <https://paper.dropbox.com/doc/Wasserstein-GAN-GvU0p2V9ThzdwY3BbhoP7>  
  - <http://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/>  
  - <https://github.com/martinarjovsky/WassersteinGAN>  
  - <https://github.com/wiseodd/generative-models/tree/master/GAN/wasserstein_gan>  
  - <https://github.com/shekkizh/WassersteinGAN.tensorflow>  
  - <https://github.com/kuleshov/tf-wgan>  
  - <https://github.com/blei-lab/edward/blob/master/examples/gan_wasserstein.py>  
  - <https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/WassersteinGAN>  

[BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717) (Google Brain)  
>	"We propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based Generative Adversarial Networks. This method balances the generator and discriminator during training. Additionally, it provides a new approximate convergence measure, fast and stable training and high visual quality. We also derive a way of controlling the trade-off between image diversity and visual quality. We focus on the image generation task, setting a new milestone in visual quality, even at higher resolutions. This is achieved while using a relatively simple model architecture and a standard training procedure."  
>	"- A GAN with a simple yet robust architecture, standard training procedure with fast and stable convergence.  
>	- An equilibrium concept that balances the power of the discriminator against the generator.  
>	- A new way to control the trade-off between image diversity and visual quality.  
>	- An approximate measure of convergence. To our knowledge the only other published measure is from Wasserstein GAN."  
>	"There are still many unexplored avenues. Does the discriminator have to be an auto-encoder? Having pixel-level feedback seems to greatly help convergence, however using an auto-encoder has its drawbacks: what internal embedding size is best for a dataset? When should noise be added to the input and how much? What impact would using other varieties of auto-encoders such Variational Auto-Encoders have?"  
  - <https://pbs.twimg.com/media/C8lYiYbW0AI4_yk.jpg:large> + <https://pbs.twimg.com/media/C8c6T2kXsAAI-BN.jpg> (demo)  
  - <https://blog.heuritech.com/2017/04/11/began-state-of-the-art-generation-of-faces-with-generative-adversarial-networks/>  
  - <https://reddit.com/r/MachineLearning/comments/633jal/r170310717_began_boundary_equilibrium_generative/dfrktje/>  
  - <https://github.com/carpedm20/BEGAN-tensorflow>  
  - <https://github.com/carpedm20/BEGAN-pytorch>  

----
[Unrolled Generative Adversarial Networks](http://arxiv.org/abs/1611.02163)  
>	"We introduce a method to stabilize GANs by defining the generator objective with respect to an unrolled optimization of the discriminator. This allows training to be adjusted between using the optimal discriminator in the generator's objective, which is ideal but infeasible in practice, and using the current value of the discriminator, which is often unstable and leads to poor solutions. We show how this technique solves the common problem of mode collapse, stabilizes training of GANs with complex recurrent generators, and increases diversity and coverage of the data distribution by the generator."  
  - <https://github.com/poolio/unrolled_gan>  

[Improved Techniques for Training GANs](http://arxiv.org/abs/1606.03498)  
>	"Our CIFAR-10 samples also look very sharp - Amazon Mechanical Turk workers can distinguish our samples from real data with an error rate of 21.3% (50% would be random guessing)"  
>	"In addition to generating pretty pictures, we introduce an approach for semi-supervised learning with GANs that involves the discriminator producing an additional output indicating the label of the input. This approach allows us to obtain state of the art results on MNIST, SVHN, and CIFAR-10 in settings with very few labeled examples. On MNIST, for example, we achieve 99.14% accuracy with only 10 labeled examples per class with a fully connected neural network — a result that’s very close to the best known results with fully supervised approaches using all 60,000 labeled examples."  
  - <https://youtu.be/RZOKRFBtSh4?t=26m18s> (Metz)  
  - <https://github.com/aleju/papers/blob/master/neural-nets/Improved_Techniques_for_Training_GANs.md>  
  - <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2FSalimansGZCRC16>  
  - <http://inference.vc/understanding-minibatch-discrimination-in-gans/>  
  - <https://github.com/openai/improved-gan>  

----
[GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution](http://arxiv.org/abs/1611.04051)  
  - <https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html>  

[Maximum-Likelihood Augmented Discrete Generative Adversarial Networks](http://arxiv.org/abs/1702.07983) (Bengio)  

[Boundary-Seeking Generative Adversarial Networks](http://arxiv.org/abs/1702.08431) (Bengio)  
>	"This approach can be used to train a generator with discrete output when the generator outputs a parametric conditional distribution. We demonstrate the effectiveness of the proposed algorithm with discrete image data. In contrary to the proposed algorithm, we observe that the recently proposed Gumbel-Softmax technique for re-parametrizing the discrete variables does not work for training a GAN with discrete data."  
  - <http://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/>  
  - <https://github.com/wiseodd/generative-models/tree/master/GAN/boundary_seeking_gan>  

----
[Task Specific Adversarial Cost Function](http://arxiv.org/abs/1609.08661)  
  - <https://github.com/ToniCreswell/piGAN>  

[Stacked Generative Adversarial Networks](http://arxiv.org/abs/1612.04357)  
  - <https://github.com/xunhuang1995/SGAN>  

[Generative Multi-Adversarial Networks](http://openreview.net/forum?id=Byk-VI9eg)  

[AdaGAN: Boosting Generative Models](http://arxiv.org/abs/1701.02386)  

[Alternating Back-Propagation for Generator Network](http://www.stat.ucla.edu/~ywu/ABP/doc/arXivABP.pdf)  

----
[Generating Text via Adversarial Training](https://sites.google.com/site/nips2016adversarial/WAT16_paper_20.pdf)  
  - <http://machinedlearnings.com/2017/01/generating-text-via-adversarial-training.html>  

[Learning to Protect Communications with Adversarial Neural Cryptography](http://arxiv.org/abs/1610.06918)  
  - <https://nlml.github.io/neural-networks/adversarial-neural-cryptography/>  
  - <https://blog.acolyer.org/2017/02/10/learning-to-protect-communications-with-adversarial-neural-cryptography/>  

----
[Neural Photo Editing with Introspective Adversarial Networks](http://arxiv.org/abs/1609.07093)  
  - <https://youtube.com/watch?v=FDELBFSeqQs> (demo)  
  - <https://github.com/ajbrock/Neural-Photo-Editor>  

[Generative Adversarial Text to Image Synthesis](http://arxiv.org/abs/1605.05396)  

[Conditional Image Synthesis With Auxiliary Classifier GANs](http://arxiv.org/abs/1610.09585) (Google Brain)  
  - <https://pbs.twimg.com/media/CwM0BzjVUAAWTn4.jpg:large>  
  - <https://youtu.be/RZOKRFBtSh4?t=21m47s> (Odena)  
  - <https://github.com/buriburisuri/ac-gan>  
  - <https://github.com/wiseodd/generative-models/tree/master/GAN/auxiliary_classifier_gan>  

[Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space](http://www.evolvingai.org/files/nguyen2016ppgn__v1.pdf)  
  - <https://pbs.twimg.com/media/Czpn0VLVEAA_RpK.jpg:large>  
  - <https://github.com/Evolving-AI-Lab/ppgn>  

[Sampling Generative Networks: Notes on a Few Effective Techniques](http://arxiv.org/abs/1609.04468)  # smilevector  
  - <https://twitter.com/smilevector> (demo)  
  - <https://github.com/dribnet/plat>  

----
[Learning from Simulated and Unsupervised Images through Adversarial Training](http://arxiv.org/abs/1612.07828) (Apple)  
  - <https://github.com/carpedm20/simulated-unsupervised-tensorflow>  

[Unsupervised Pixel-Level Domain Adaptation with Generative Asversarial Networks](http://arxiv.org/abs/1612.05424) (Google Brain)  

----
[Image-to-Image Translation with Conditional Adversarial Networks](http://arxiv.org/abs/1611.07004)  
  - <https://phillipi.github.io/pix2pix/>  

[Unsupervised Image-to-Image Translation Networks](http://arxiv.org/abs/1703.00848) (NVIDIA)  

[DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/abs/1704.02510)



---
### generative models - variational autoencoders

[Towards a Deeper Understanding of Variational Autoencoding Models](http://arxiv.org/abs/1702.08658)  
>	"We provide a formal explanation for why VAEs generate blurry samples when trained on complex natural images. We show that under some conditions, blurry samples are not caused by the use of a maximum likelihood approach as previously thought, but rather they are caused by an inappropriate choice for the inference distribution. We specifically target this problem by proposing a sequential VAE model, where we gradually augment the the expressiveness of the inference distribution using a process inspired by the recent infusion training process. As a result, we are able to generate sharp samples on the LSUN bedroom dataset, even using 2-norm reconstruction loss in pixel space."  
>
>	"We propose a new explanation of the VAE tendency to ignore the latent code. We show that this problem is specific to the original VAE objective function and does not apply to the more general family of VAE models we propose. We show experimentally that using our more general framework, we achieve comparable sample quality as the original VAE, while at the same time learning meaningful features through the latent code, even when the decoder is a powerful PixelCNN that can by itself model data."  

[Variational Lossy Autoencoder](http://arxiv.org/abs/1611.02731) (OpenAI)  

[Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks](http://arxiv.org/abs/1701.04722)  
  - <https://github.com/wiseodd/generative-models/tree/master/VAE/adversarial_vb>  
  - <https://gist.github.com/poolio/b71eb943d6537d01f46e7b20e9225149>  
  - <http://inference.vc/variational-inference-with-implicit-models-part-ii-amortised-inference-2/>  

----
[Importance Weighted Autoencoders](http://arxiv.org/abs/1509.00519) (Burda)  
  - <http://dustintran.com/blog/importance-weighted-autoencoders/>  
  - <https://github.com/yburda/iwae>  
  - <https://github.com/arahuja/generative-tf>  
  - <https://github.com/blei-lab/edward/blob/master/examples/iwvi.py>  

[Variational Inference for Monte Carlo Objectives](http://arxiv.org/abs/1602.06725)  
  - <http://techtalks.tv/talks/variational-inference-for-monte-carlo-objectives/62507/>  
  - <https://evernote.com/shard/s189/sh/54a9fb88-1a71-4e8a-b0e3-f13480a68b8d/0663de49b93d397f519c7d7f73b6a441>  

[Discrete Variational Autoencoders](http://arxiv.org/abs/1609.02200)  
  - <https://youtube.com/watch?v=c6GukeAkyVs> (Struminsky)  

[The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](http://arxiv.org/abs/1611.00712) (DeepMind)  
  - <http://youtube.com/watch?v=JFgXEbgcT7g> (Jang)  
  - <https://laurent-dinh.github.io/2016/11/22/gumbel-max.html>  
  - <https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html>  
  - <https://gist.github.com/gngdb/ef1999ce3a8e0c5cc2ed35f488e19748>  

[Categorical Reparametrization with Gumbel-Softmax](http://arxiv.org/abs/1611.01144) (Google Brain)  
  - <http://youtube.com/watch?v=JFgXEbgcT7g> (Jang)  
  - <http://blog.evjang.com/2016/11/tutorial-categorical-variational.html>  
  - <https://laurent-dinh.github.io/2016/11/22/gumbel-max.html>  
  - <https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html>  
  - <https://github.com/EderSantana/gumbel>  

[REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models](http://arxiv.org/abs/1703.07370) (Google Brain + DeepMind)  
>	"Learning in models with discrete latent variables is challenging due to high variance gradient estimators. Generally, approaches have relied on control variates to reduce the variance of the REINFORCE estimator. Recent work (Jang et al. 2016, Maddison et al. 2016) has taken a different approach, introducing a continuous relaxation of discrete variables to produce low-variance, but biased, gradient estimates. In this work, we combine the two approaches through a novel control variate that produces low-variance, unbiased gradient estimates."  

[Discrete Variational Autoencoders](http://openreview.net/forum?id=ryMxXPFex) (D-Wave)  

----
[Multi-modal Variational Encoder-Decoders](http://arxiv.org/abs/1612.00377) (Courville)  

[Stochastic Backpropagation through Mixture Density Distributions](http://arxiv.org/abs/1607.05690)  
  - <http://www.shortscience.org/paper?bibtexKey=journals/corr/1607.05690>  

[Variational Boosting: Iteratively Refining Posterior Approximations](http://arxiv.org/abs/1611.06585) (Adams)  
  - <http://andymiller.github.io/2016/11/23/vb.html>  
  - <https://youtu.be/Jh3D8Gi4N0I?t=1h9m52s> (Nekludov, in russian)  

[Improving Variational Inference with Inverse Autoregressive Flow](http://arxiv.org/abs/1606.04934)  
>	"Most VAEs have so far been trained using crude approximate posteriors, where every latent variable is independent. Normalizing Flows have addressed this problem by conditioning each latent variable on the others before it in a chain, but this is computationally inefficient due to the introduced sequential dependencies. The core contribution of this work, termed inverse autoregressive flow (IAF), is a new approach that, unlike previous work, allows us to parallelize the computation of rich approximate posteriors, and make them almost arbitrarily flexible."  
  - <https://github.com/openai/iaf>  

[Normalizing Flows on Riemannian Manifolds](http://arxiv.org/abs/1611.02304) (Mohamed)  
  - <https://pbs.twimg.com/media/CyntnWDXAAA97Ks.jpg>  

----
[Auxiliary Deep Generative Models](http://arxiv.org/abs/1602.05473)  
  - <http://techtalks.tv/talks/auxiliary-deep-generative-models/62509/>  
  - <https://github.com/larsmaaloee/auxiliary-deep-generative-models>  

[Composing graphical models with neural networks for structured representations and fast inference](http://arxiv.org/abs/1603.06277)  
  - <https://youtube.com/watch?v=btr1poCYIzw>  
  - <http://www.cs.toronto.edu/~duvenaud/courses/csc2541/slides/svae-slides.pdf>  
  - <https://github.com/mattjj/svae>  

----
[Rejection Sampling Variational Inference](http://arxiv.org/abs/1610.05683)  

[The Generalized Reparameterization Gradient](http://arxiv.org/abs/1610.02287)  
  - <https://youtu.be/mrj_hyH974o?t=1h23m40s> (Vetrov, in russian)  

[The Variational Fair Autoencoder](http://arxiv.org/abs/1511.00830)  
  - <http://videolectures.net/iclr2016_louizos_fair_autoencoder/> (Louizos)  

[The Variational Gaussian Process](http://arxiv.org/abs/1511.06499)  
  - <http://videolectures.net/iclr2016_tran_variational_gaussian/> (Tran)  
  - <http://github.com/blei-lab/edward>  

[Stick-Breaking Variational Autoencoders  ](http://arxiv.org/abs/1605.06197)# latent representation with stochastic dimensionality  

----
[A Hybrid Convolutional Variational Autoencoder for Text Generation](http://arxiv.org/pdf/1702.02390)  
  - <https://github.com/stas-semeniuta/textvae>  

[Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](http://arxiv.org/abs/1702.08139) (Salakhutdinov)  

[Controllable Text Generation](http://arxiv.org/abs/1703.00955) (Salakhutdinov)  

[Grammar Variational Autoencoder](http://arxiv.org/abs/1703.01925)  



---
### generative models - autoregressive models

[Pixel Recurrent Neural Networks](http://arxiv.org/abs/1601.06759)  
  - <http://techtalks.tv/talks/pixel-recurrent-neural-networks/62375/> (van den Oord)  
  - <https://evernote.com/shard/s189/sh/fdf61a28-f4b6-491b-bef1-f3e148185b18/aba21367d1b3730d9334ed91d3250848> (Larochelle)  
  - <https://github.com/tensorflow/magenta/blob/master/magenta/reviews/pixelrnn.md> (Kastner)  
  - <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2FOordKK16#shagunsodhani>  
  - <https://github.com/zhirongw/pixel-rnn>  
  - <https://github.com/igul222/pixel_rnn>  
  - <https://github.com/carpedm20/pixel-rnn-tensorflow>  
  - <https://github.com/shiretzet/PixelRNN>  

[Conditional Image Generation with PixelCNN Decoders](http://arxiv.org/abs/1606.05328)  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-1> (27:26, van den Oord)  
  - <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1606.05328#shagunsodhani>  
  - <http://sergeiturukin.com/2017/02/22/pixelcnn.html>  
  - <https://github.com/openai/pixel-cnn>  
  - <https://github.com/kundan2510/pixelCNN>  
  - <https://github.com/anantzoid/Conditional-PixelCNN-decoder>  

[PixelCNN++: A PixelCNN Implementation with Discretized Logistic Mixture Likelihood and Other Modifications](https://openreview.net/forum?id=BJrFC6ceg) (OpenAI)  
  - <https://github.com/openai/pixel-cnn>  

[Parallel Multiscale Autoregressive Density Estimation](http://arxiv.org/abs/1703.03664) (DeepMind)  
>	"O(log N) sampling instead of O(N)"  

----
[WaveNet: A Generative Model for Raw Audio](http://arxiv.org/abs/1609.03499) (DeepMind)  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-1> (42:36, van den Oord)  
  - <https://github.com/ibab/tensorflow-wavenet>  
  - <https://github.com/basveeling/wavenet/>  
  - <https://github.com/usernaamee/keras-wavenet>  
  - <https://github.com/tomlepaine/fast-wavenet>  

[Neural Machine Translation in Linear Time](http://arxiv.org/abs/1610.10099) (ByteNet) (DeepMind)  
>	"Generalizes LSTM seq2seq by preserving the resolution. Dynamic unfolding instead of attention. Linear time computation."  
>
>	"The authors apply a WaveNet-like architecture to the task of Machine Translation. Encoder (Source Network) and Decoder (Target Network) are CNNs that use Dilated Convolutions and they are stacked on top of each other. The Target Network uses Masked Convolutions to ensure that it only relies on information from the past. Crucially, the time complexity of the network is c(|S| + |T|), which is cheaper than that of the common seq2seq attention architecture (|S|*|T|). Through dilated convolutions the network has constant path lengths between [source input -> target output] and [target inputs -> target output] nodes. This allows for efficient propagation of gradients."
 - <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/nmt-linear-time.md>  

[Language Modeling with Gated Convolutional Networks](http://arxiv.org/abs/1612.08083) (Facebook)  # outperforming LSTM on language modelling
  - <https://github.com/DingKe/nn_playground/tree/master/gcnn>  

----
[An Actor-Critic Algorithm for Sequence Prediction](http://arxiv.org/abs/1607.07086) (Bengio)  

[Tuning Recurrent Neural Networks with Reinforcement Learning](http://arxiv.org/abs/1611.02796) (Google Brain)  
>	"In contrast to relying solely on possibly biased data, our approach allows for encoding high-level domain knowledge into the RNN, providing a general, alternative tool for training sequence models."  
  - <https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning/>  
  - <https://www.technologyreview.com/s/604010/google-brain-wants-creative-ai-to-help-humans-make-a-new-kind-of-art/> (10:45, Eck)  
  - <https://github.com/tensorflow/magenta/tree/master/magenta/models/rl_tuner>  

[Learning to Decode for Future Success](http://arxiv.org/abs/1701.06549) (Stanford)  

----
[Professor Forcing: A New Algorithm for Training Recurrent Networks](http://arxiv.org/abs/1610.09038)  
>	"In professor forcing, G is simply an RNN that is trained to predict the next element in a sequence and D a discriminative bi-directional RNN. G is trained to fool D into thinking that the hidden states of G occupy the same state space at training (feeding ground truth inputs to the RNN) and inference time (feeding generated outputs as the next inputs). D, in turn, is trained to tell apart the hidden states of G at training and inference time. At the Nash equilibrium, D cannot tell apart the state spaces any better and G cannot make them any more similar. This is motivated by the problem that RNNs typically diverge to regions of the state space that were never observed during training and which are hence difficult to generalize to."  
  - <https://youtube.com/watch?v=I7UFPBDLDIk>  
  - <http://videolectures.net/deeplearning2016_goyal_new_algorithm/> (Goyal)  
  - <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/professor-forcing.md>  
  - <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1610.09038>  

----
[Sequence-to-Sequence Learning as Beam-Search Optimization](http://arxiv.org/abs/1606.02960)  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-2> (Wiseman, 0:44:02)  
  - <http://shortscience.org/paper?bibtexKey=journals/corr/1606.02960>  

[Globally Normalized Transition-Based Neural Networks](http://arxiv.org/abs/1603.06042)  # SyntaxNet, Parsey McParseface
>	"The parser uses a feed forward NN, which is much faster than the RNN usually used for parsing. Also the paper is using a global method to solve the label bias problem. This method can be used for many tasks and indeed in the paper it is used also to shorten sentences by throwing unnecessary words. The label bias problem arises when predicting each label in a sequence using a softmax over all possible label values in each step. This is a local approach but what we are really interested in is a global approach in which the sequence of all labels that appeared in a training example are normalized by all possible sequences. This is intractable so instead a beam search is performed to generate alternative sequences to the training sequence. The search is stopped when the training sequence drops from the beam or ends. The different beams with the training sequence are then used to compute the global loss."  
  - <https://github.com/udibr/notes/raw/master/Talk%20by%20Sasha%20Rush%20-%20Interpreting%2C%20Training%2C%20and%20Distilling%20Seq2Seq%E2%80%A6.pdf>  
  - <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1603.06042>  
  - <https://github.com/tensorflow/models/tree/master/syntaxnet>  

[Length Bias in Encoder Decoder Models and a Case for Global Conditioning](http://arxiv.org/abs/1606.03402) (Google)  

----
[Order Matters: Sequence to Sequence for Sets](http://arxiv.org/abs/1511.06391) (Vinyals)  
  - <https://youtube.com/watch?v=uohtFXD_39c&t=56m51s> (Bengio)  



---
### probabilistic inference

[Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](http://arxiv.org/abs/1612.01474) (DeepMind)  

[Dropout Inference in Bayesian Neural Networks with Alpha-divergences](http://mlg.eng.cam.ac.uk/yarin/PDFs/LiGal2017.pdf)
>	"We demonstrate improved uncertainty estimates and accuracy compared to VI in dropout networks. We study our model’s epistemic uncertainty far away from the data using adversarial images, showing that these can be distinguished from non-adversarial images by examining our model’s uncertainty."

----
[Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798) (DeepMind)  

[Sequential Neural Models with Stochastic Layers](http://arxiv.org/abs/1605.07571)  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Sequential-Neural-Models-with-Stochastic-Layers> (Fraccaro)  
  - <https://youtu.be/mrj_hyH974o?t=32m49s> (in russian)  
  - <https://github.com/marcofraccaro/srnn>  

[DISCO Nets: DISsimilarity COefficient Networks](http://arxiv.org/abs/1606.02556)  
  - <https://youtube.com/watch?v=OogNSKRkoes>   

----
[Deep Probabilistic Programming](http://arxiv.org/abs/1701.03757) (Edward)  
  - <http://edwardlib.org/iclr2017>  
  - <http://edwardlib.org/zoo>  

[Deep Amortized Inference for Probabilistic Programs](http://arxiv.org/abs/1610.05735) (Goodman)  

[Inference Compilation and Universal Probabilistic Programming](http://arxiv.org/abs/1610.09900) (Wood)  

----
[Semantic Parsing to Probabilistic Programs for Situated Question Answering](http://arxiv.org/abs/1606.07046) (AI2)  



---
### reasoning

[Text Understanding with the Attention Sum Reader Network](http://arxiv.org/abs/1603.01547)  

[Key-Value Memory Networks for Directly Reading Documents](http://arxiv.org/abs/1606.03126) (Weston)  
  - <http://www.shortscience.org/paper?bibtexKey=journals/corr/1606.03126>  
  - <https://gist.github.com/shagunsodhani/a5e0baa075b4a917c0a69edc575772a8>  
  - <https://github.com/facebook/MemNN/blob/master/KVmemnn>  

[Tracking the World State with Recurrent Entity Networks](https://openreview.net/forum?id=rJTKKKqeg) (Facebook)  
>	"There's a bunch of memory slots that each can be used to represent a single entity. The first time an entity appears, it's written to a slot. Every time that something happens in the story that corresponds to a change in the state of an entity, the change in the state of that entity is combined with the entity's previous state via a modified GRU update equation and rewritten to the same slot."  
  - <https://github.com/jimfleming/recurrent-entity-networks>  

[Gated-Attention Readers for Text Comprehension](http://arxiv.org/abs/1606.01549) (Salakhutdinov)  
  - <https://youtube.com/watch?v=ZSDrM-tuOiA> (Salakhutdinov)

[Query-Regression Networks for Machine Comprehension](http://arxiv.org/abs/1606.04582) (AI2)  
>	"We show the state-of-the-art results in the three datasets of story-based QA and dialog. We model a story or a dialog as a sequence of state-changing triggers and compute the final answer to the question or the system’s next utterance by recurrently updating (or reducing) the query. QRN is situated between the attention mechanism and RNN, effectively handling time dependency and long-term dependency problems of each technique, respectively. It addresses the long-term dependency problem of most RNNs by simplifying the recurrent update, in which the candidate hidden state (reduced query) does not depend on the previous state. Moreover, QRN can be parallelized and can address the well-known problem of RNN’s vanishing gradients."  
  - <https://github.com/seominjoon/qrn>  

----
[Dynamic Memory Networks for Visual and Textual Question Answering](http://arxiv.org/abs/1603.01417) (Socher)  
  - <https://youtube.com/watch?v=FCtpHt6JEI8>  
  - <https://youtube.com/watch?v=DjPQRLMMAbw>  
  - <http://techtalks.tv/talks/dynamic-memory-networks-for-visual-and-textual-question-answering/62463/>  
  - <https://github.com/therne/dmn-tensorflow>  

[Deep Compositional Question Answering with Neural Module Networks](http://arxiv.org/abs/1511.02799) (Darrell)  
  - <https://youtube.com/watch?v=gDXD3hYfBW8> (Andreas)  
  - <http://research.microsoft.com/apps/video/default.aspx?id=260024> (Darrell, 10:45) 
  - <http://blog.jacobandreas.net/programming-with-nns.html>  
  - <https://github.com/abhshkdz/papers/blob/master/reviews/deep-compositional-question-answering-with-neural-module-networks.md>  
  - <http://github.com/jacobandreas/nmn2>  

[Learning to Compose Neural Networks for Question Answering](http://arxiv.org/abs/1601.01705) (Darrell)  
  - <https://youtube.com/watch?v=gDXD3hYfBW8> (Andreas)  
  - <http://blog.jacobandreas.net/programming-with-nns.html>  
  - <http://github.com/jacobandreas/nmn2>  

[Modeling Relationships in Referential Expressions with Compositional Modular Networks](http://arxiv.org/abs/1611.09978) (Darrell)  

[The More You Know: Using Knowledge Graphs for Image Classification](http://arxiv.org/abs/1612.04844) (Salakhutdinov)  # evolution of Gated Graph Sequence Neural Networks  

----
[Neural Enquirer: Learning to Query Tables with Natural Language](http://arxiv.org/abs/1512.00965)  
>	"Authors propose a fully distributed neural enquirer, comprising several neuralized execution layers of field attention, row annotation, etc. While the model is not efficient in execution because of intensive matrix/vector operation during neural information processing and lacks explicit interpretation of execution, it can be trained in an end-to-end fashion because all components in the neural enquirer are differentiable."  

[Learning a Natural Language Interface with Neural Programmer](http://arxiv.org/abs/1611.08945)  
  - <https://github.com/tensorflow/models/tree/master/neural_programmer>  

[Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision](http://arxiv.org/abs/1611.00020)  (Google Brain)
>	"We propose the Manager-Programmer-Computer framework, which integrates neural networks with non-differentiable memory to support abstract, scalable and precise operations through a friendly neural computer interface. Specifically, we introduce a Neural Symbolic Machine, which contains a sequence-to-sequence neural "programmer", and a non-differentiable "computer" that is a Lisp interpreter with code assist."  

----
[Learning Knowledge Base Inference with Neural Theorem Provers](http://akbc.ws/2016/papers/14_Paper.pdf) (Rocktaschel)  
  - <http://aitp-conference.org/2017/slides/Tim_aitp.pdf> (Rocktaschel)  

[TensorLog: A Differentiable Deductive Database](http://arxiv.org/abs/1605.06523) (Cohen)  
  - <https://github.com/TeamCohen/TensorLog>  

[Differentiable Learning of Logical Rules for Knowledge Base Completion](https://arxiv.org/abs/1702.08367) (Cohen)  

----
[WebNav: A New Large-Scale Task for Natural Language based Sequential Decision Making](http://arxiv.org/abs/1602.02261)  

----
[Learning Physical Intuition of Block Towers by Example](http://arxiv.org/abs/1603.01312) (Fergus)  
  - <https://youtu.be/oSAG57plHnI?t=19m48s> (Tenenbaum)  

[Learning to Perform Physics Experiments via Deep Reinforcement Learning](http://arxiv.org/abs/1611.01843) (DeepMind)  
  - <https://youtu.be/tPWGGwmgwG0?t=16m34s> (de Freitas)  

[Interaction Networks for Learning about Objects, Relations and Physics](http://arxiv.org/abs/1612.00222) (DeepMind)  
  - <https://blog.acolyer.org/2017/01/02/interaction-networks-for-learning-about-objects-relations-and-physics/>  



---
### program induction

[Making Neural Programming Architectures Generalize via Recursion](http://openreview.net/forum?id=BkbY4psgg)  # Neural Programmer-Interpreter with recursion  
>	"We implement recursion in the Neural Programmer-Interpreter framework on four tasks: grade-school addition, bubble sort, topological sort, and quicksort."  

[Adaptive Neural Compilation](http://arxiv.org/abs/1605.07969)  

[Programming with a Differentiable Forth Interpreter](http://arxiv.org/abs/1605.06640) (Riedel)  # learning details of probabilistic program  

[TerpreT: A Probabilistic Programming Language for Program Induction](http://arxiv.org/abs/1608.04428)  
>	"These works raise questions of (a) whether new models can be designed specifically to synthesize interpretable source code that may contain looping and branching structures, and (b) whether searching over program space using techniques developed for training deep neural networks is a useful alternative to the combinatorial search methods used in traditional IPS. In this work, we make several contributions in both of these directions."  
>	"Shows that differentiable interpreter-based program induction is inferior to discrete search-based techniques used by the programming languages community. We are then left with the question of how to make progress on program induction using machine learning techniques."  



---
### reinforcement learning - algorithms

[Benchmarking Deep Reinforcement Learning for Continuous Control](http://arxiv.org/abs/1604.06778) (Abbeel)  
  - <http://techtalks.tv/talks/benchmarking-deep-reinforcement-learning-for-continuous-control/62380/> (Duan)  

----
[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](http://arxiv.org/abs/1703.03864) (OpenAI)  
>	(Karpathy) "ES is much simpler than RL, and there's no need for backprop, it's highly parallelizable, has fewer hyperparams, needs no value functions."  
>	"In our preliminary experiments we found that using ES to estimate the gradient on the MNIST digit recognition task can be as much as 1,000 times slower than using backpropagation. It is only in RL settings, where one has to estimate the gradient of the expected reward by sampling, where ES becomes competitive."  
  - <https://blog.openai.com/evolution-strategies/>  
  - <https://www.technologyreview.com/s/603916/a-new-direction-for-artificial-intelligence/> (Sutskever)  
  - <http://inference.vc/evolutionary-strategies-embarrassingly-parallelizable-optimization/> (Huszar)  
  - <http://inference.vc/evolution-strategies-variational-optimisation-and-natural-es-2/> (Huszar)  
  - <http://davidbarber.github.io/blog/2017/04/03/variational-optimisation/> (Barber)  
  - <http://argmin.net/2017/04/03/evolution/> (Recht)  
  - <https://github.com/openai/evolution-strategies-starter>  
  - <https://github.com/mdibaiee/flappy-es> (demo)  
  - <https://gist.github.com/kashif/5748e199a3bec164a867c9b654e5ffe5>  
  - <https://github.com/atgambardella/pytorch-es>  

----
[RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](http://arxiv.org/abs/1611.02779) (OpenAI)  
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
>
>	"future directions:  
>	- better outer-loop algorithms  
>	- scaling RL^2 to 1M games  
>	- model-based RL^2  
>	- curriculum learning / universal RL^2  
>	- RL^2 + one-shot imitation learning  
>	- RL^2 for simulation -> real world transfer"  
  - <https://youtu.be/BskhUBPRrqE?t=6m28s> + <https://youtu.be/19eNQ1CLt5A?t=7m52s> (Sutskever)  
  - <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)  
  - <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/RL2-Fast_Reinforcement_Learning_via_Slow_Reinforcement_Learning.md>  

[Learning to Reinforcement Learn](http://arxiv.org/abs/1611.05763) (DeepMind)  
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
  - <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)  
  - <https://hackernoon.com/learning-policies-for-learning-policies-meta-reinforcement-learning-rl²-in-tensorflow-b15b592a2ddf> (Juliani)  
  - <https://github.com/awjuliani/Meta-RL>  

----
[Reinforcement Learning with Unsupervised Auxiliary Tasks](http://arxiv.org/abs/1611.05397) (UNREAL, extension of A3C)  
>	"This approach exploits the multithreading capabilities of standard CPUs. The idea is to execute many instances of our agent in parallel, but using a shared model. This provides a viable alternative to experience replay, since parallelisation also diversifies and decorrelates the data. Our asynchronous actor-critic algorithm, A3C, combines a deep Q-network with a deep policy network for selecting actions. It achieves state-of-the-art results, using a fraction of the training time of DQN and a fraction of the resource consumption of Gorila."  
>	"Auxiliary tasks:  
>	- pixel changes: learn a policy for maximally changing the pixels in a grid of cells overlaid over the images  
>	- network features: learn a policy for maximally activating units in a specific hidden layer  
>	- reward prediction: predict the next reward given some historical context  
>	- value function replay: value function regression for the base agent with varying window for n-step returns"  
  - <https://youtube.com/watch?v=Uz-zGYrYEjA> (demo)  
  - <https://github.com/dennybritz/deeplearning-papernotes/blob/b097e313dc59c956575fb1bf23b64fa8d1d84057/notes/rl-auxiliary-tasks.md>  

[Loss is Its Own Reward: Self-Supervision for Reinforcement Learning](http://arxiv.org/abs/1612.07307) (Darrell)  

[Dual Learning for Machine Translation](http://arxiv.org/abs/1611.00179)  
>	"In the dual-learning mechanism, we use one agent to represent the model for the primal task and the other agent to represent the model for the dual task, then ask them to teach each other through a reinforcement learning process. Based on the feedback signals generated during this process (e.g., the language model likelihood of the output of a model, and the reconstruction error of the original sentence after the primal and dual translations), we can iteratively update the two models until convergence (e.g., using the policy gradient methods)."  
>	"The basic idea of dual learning is generally applicable: as long as two tasks are in dual form, we can apply the dual-learning mechanism to simultaneously learn both tasks from unlabeled data using reinforcement learning algorithms. Actually, many AI tasks are naturally in dual form, for example, speech recognition versus text to speech, image caption versus image generation, question answering versus question generation (e.g., Jeopardy!), search (matching queries to documents) versus keyword extraction (extracting keywords/queries for documents), so on and so forth."  
>
>	"The authors finetune an FR -> EN NMT model using a RL-based dual game. 1. Pick a French sentence from a monolingual corpus and translate it to EN. 2. Use an EN language model to get a reward for the translation 3. Translate the translation back into FR using an EN -> FR system. 4. Get a reward based on the consistency between original and reconstructed sentence. Training this architecture using Policy Gradient authors can make efficient use of monolingual data and show that a system trained on only 10% of parallel data and finetuned with monolingual data achieves comparable BLUE scores as a system trained on the full set of parallel data."  
  - <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/dual-learning-mt.md>  

---
[Learning to Play in a Day: Faster Deep Reinforcement Learning by Optimality Tightening](http://openreview.net/forum?id=rJ8Je4clg)  # 10x faster Q-learning  
  - <https://youtu.be/mrj_hyH974o?t=16m13s> (in russian)  

[Sample Efficient Actor-Critic with Experience Replay](http://arxiv.org/abs/1611.01224) (DeepMind)  # ACER  
  - <https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/acer.py>  

[Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](http://arxiv.org/abs/1611.02247) (Lillicrap, Levine)  
>	"We present Q-Prop, a policy gradient method that uses a Taylor expansion of the off-policy critic as a control variate. Q-Prop is both sample efficient and stable, and effectively combines the benefits of on-policy and off-policy methods."  
	"- unbiased gradient  
	 - combine PG and AC gradients  
	 - learns critic from off-policy data  
	 - learns policy from on-policy data"  
	"Q-Prop works with smaller batch size than TRPO-GAE  
	Q-Prop is significantly more sample-efficient than TRPO-GAE"  
>	"policy gradient algorithm that is as fast as value estimation"  
  - <https://youtu.be/M6nfipCxQBc?t=16m11s> (Lillicrap)  

---
[PGQ: Combining policy gradient and Q-learning](http://arxiv.org/abs/1611.01626) (DeepMind)  
>	"This connection allows us to estimate the Q-values from the action preferences of the policy, to which we apply Q-learning updates."  
>	"We also establish an equivalency between action-value fitting techniques and actor-critic algorithms, showing that regularized policy gradient techniques can be interpreted as advantage function learning algorithms."  
  - <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2FODonoghueMKM16>  
  - <https://github.com/Fritz449/Asynchronous-RL-agent>  

[Bridging the Gap Between Value and Policy Reinforcement Learning](http://arxiv.org/abs/1702.08892) (Google Brain)  # PCL  
  - <https://github.com/ethancaballero/paper-notes/blob/master/Bridging%20the%20Gap%20Between%20Value%20and%20Policy%20Based%20Reinforcement%20Learning.md>  
  - <https://github.com/rarilurelo/pcl_keras>  
  - <https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/pcl.py>  

----
[Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783)  
  - <http://youtube.com/watch?v=0xo1Ldx3L5Q> (TORCS demo)  
  - <http://youtube.com/watch?v=nMR5mjCFZCw> (3D Labyrinth demo)  
  - <http://youtube.com/watch?v=Ajjc08-iPx8> (MuJoCo demo)  
  - <http://youtube.com/watch?v=9sx1_u2qVhQ> (Mnih)  
  - <http://techtalks.tv/talks/asynchronous-methods-for-deep-reinforcement-learning/62475/> (Mnih)  
  - <https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2>  
  - <https://github.com/Zeta36/Asynchronous-Methods-for-Deep-Reinforcement-Learning>  
  - <https://github.com/miyosuda/async_deep_reinforce>  
  - <https://github.com/muupan/async-rl>  
  - <https://github.com/yandexdataschool/AgentNet/blob/master/agentnet/learning/a2c_n_step.py>  
  - <https://github.com/coreylynch/async-rl>  
  - <https://github.com/carpedm20/deep-rl-tensorflow/blob/master/agents/async.py>  
  - <https://github.com/danijar/mindpark/blob/master/mindpark/algorithm/a3c.py>  

[Continuous Deep Q-Learning with Model-based Acceleration](http://arxiv.org/abs/1603.00748) (Sutskever)  
  - <http://techtalks.tv/talks/continuous-deep-q-learning-with-model-based-acceleration/62474/>  
  - <https://youtu.be/M6nfipCxQBc?t=10m48s> (Lillicrap)  
  - <https://youtu.be/mrgJ53TIcQc?t=57m> (Seleznev, in russian)  
  - <http://www.bicv.org/?wpdmdl=1940>  
  - <https://github.com/carpedm20/NAF-tensorflow>  
  - <https://github.com/tambetm/gymexperiments>  

----
[Trust Region Policy Optimization](http://arxiv.org/abs/1502.05477) (Schulman, Levine, Jordan, Abbeel)  
  - <https://youtube.com/watch?v=jeid0wIrSn4>  

[High-Dimensional Continuous Control Using Generalized Advantage Estimation](http://arxiv.org/abs/1506.02438) (Schulman)  
  - <https://youtube.com/watch?v=ATvp0Hp7RUI>  

[Gradient Estimation Using Stochastic Computation Graphs](http://arxiv.org/abs/1506.05254) (Schulman)  
  - <http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/> (Abbeel, 01:02:04)  
>	"Can mix and match likelihood ratio and path derivative. If black-box node: might need to place stochastic node in front of it and use likelihood ratio. This includes recurrent neural net policies."  

----
[Q(λ) with Off-Policy Corrections](http://arxiv.org/abs/1602.04951) (DeepMind)  
  - <https://youtube.com/watch?v=8hK0NnG_DhY&t=25m27s> (Brunskill)  

[Safe and Efficient Off-Policy Reinforcement Learning](http://arxiv.org/abs/1606.02647) (DeepMind)  # Retrace  
>	"Retrace(λ) is a new strategy to weight a sample for off-policy learning, it provides low-variance, safe and efficient updates."  
>	"Our goal is to design a RL algorithm with two desired properties. Firstly, to use off-policy data, which is important for exploration, when we use memory replay, or observe log-data. Secondly, to use multi-steps returns in order to propagate rewards faster and avoid accumulation of approximation/estimation errors. Both properties are crucial in deep RL. We introduce the “Retrace” algorithm, which uses multi-steps returns and can safely and efficiently utilize any off-policy data."  
>	"open issue: off policy unbiased, low variance estimators for long horizon delayed reward problems"  
  - <https://youtube.com/watch?v=8hK0NnG_DhY&t=25m27s> (Brunskill)  

[Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning](http://arxiv.org/abs/1604.00923) (Brunskill)  
  - <https://youtube.com/watch?v=8hK0NnG_DhY&t=15m44s> (Brunskill)  

[Multi-step Reinforcement Learning: A Unifying Algorithm](http://arxiv.org/abs/1703.01327) (Sutton)  

----
[Deep Reinforcement Learning In Parameterized Action Space](http://arxiv.org/abs/1511.04143)  
  - <https://github.com/mhauskn/dqn-hfo>  

[Reinforcement Learning in Large Discrete Action Spaces](http://arxiv.org/abs/1512.07679)  



---
### reinforcement learning - exploration and intrinsic motivation

[RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](http://arxiv.org/abs/1611.02779) (OpenAI)  # learning to explore
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
>
>	"future directions:  
>	- better outer-loop algorithms  
>	- scaling RL^2 to 1M games  
>	- model-based RL^2  
>	- curriculum learning / universal RL^2  
>	- RL^2 + one-shot imitation learning  
>	- RL^2 for simulation -> real world transfer"  
  - <https://youtu.be/BskhUBPRrqE?t=6m28s> + <https://youtu.be/19eNQ1CLt5A?t=7m52s> (Sutskever)  
  - <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)  
  - <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/RL2-Fast_Reinforcement_Learning_via_Slow_Reinforcement_Learning.md>  

[Learning to Reinforcement Learn](http://arxiv.org/abs/1611.05763) (DeepMind)  # learning to explore
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
  - <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)  
  - <https://hackernoon.com/learning-policies-for-learning-policies-meta-reinforcement-learning-rl²-in-tensorflow-b15b592a2ddf> (Juliani)  
  - <https://github.com/awjuliani/Meta-RL>  

[Deep Exploration via Randomized Value Functions](https://arxiv.org/abs/1703.07608) (Osband)  
>	"A very recent thread of work builds on count-based (or upper-confidence-bound-based) exploration schemes that operate with value function learning. These methods maintain a density over the state-action space of pseudo-counts, which represent the quantity of data gathered that is relevant to each state-action pair. Such algorithms may offer a viable approach to deep exploration with generalization. There are, however, some potential drawbacks. One is that a separate representation is required to generalize counts, and it's not clear how to design an effective approach to this. As opposed to the optimal value function, which is fixed by the environment, counts are generated by the agent’s choices, so there is no single target function to learn. Second, the count model generates reward bonuses that distort data used to fit the value function, so the value function representation needs to be designed to not only capture properties of the true optimal value function but also such distorted versions. Finally, these approaches treat uncertainties as uncoupled across state-action pairs, and this can incur a substantial negative impact on statistical efficiency."  
  - <http://youtube.com/watch?v=ck4GixLs4ZQ> (Osband) + [slides](https://docs.google.com/presentation/d/1lis0yBGT-uIXnAsi0vlP3SuWD2svMErJWy_LYtfzMOA/)  

----
[Exploration Potential](http://arxiv.org/abs/1609.04994)  
>	"We introduce exploration potential, a quantity that measures how much a reinforcement learning agent has explored its environment class. In contrast to information gain, exploration potential takes the problem's reward structure into account."  

[Surprise-Based Intrinsic Motivation for Deep Reinforcement Learning](http://arxiv.org/abs/1703.01732)  
>	"Authors present two tractable approximations to their framework - one which ignores the stochasticity of the true environmental dynamics, and one which approximates the rate of information gain (somewhat similar to Schmidhuber's formal theory of creativity, fun and intrinsic motivation)."  
>	"Stadie et al. learn deterministic dynamics models by minimizing Euclidean loss - whereas in our work, we learn stochastic dynamics with cross entropy loss - and use L2 prediction errors for intrinsic motivation."  

----
[Count-Based Exploration with Neural Density Models](http://arxiv.org/abs/1703.01310) (DeepMind)  
>	"PixelCNN for exploration, neural alternative to Context Tree Switching"  
  - <http://youtube.com/watch?v=qSfd27AgcEk> (Bellemare)  

[#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](http://arxiv.org/abs/1611.04717) (Abbeel)  
>	"The authors encourage exploration by adding a pseudo-reward of the form beta/sqrt(count(state)) for infrequently visited states. State visits are counted using Locality Sensitive Hashing (LSH) based on an environment-specific feature representation like raw pixels or autoencoder representations. The authors show that this simple technique achieves gains in various classic RL control tasks and several games in the ATARI domain. While the algorithm itself is simple there are now several more hyperaprameters to tune: The bonus coefficient beta, the LSH hashing granularity (how many bits to use for hashing) as well as the type of feature representation based on which the hash is computed, which itself may have more parameters. The experiments don't paint a consistent picture and different environments seem to need vastly different hyperparameter settings, which in my opinion will make this technique difficult to use in practice."  

[EX2: Exploration with Exemplar Models for Deep Reinforcement Learning](https://arxiv.org/abs/1703.01260) (Levine)
>	"Many of the most effective exploration techniques rely on tabular representations, or on the ability to construct a generative model over states and actions. This paper introduces a novel approach, EX2, which approximates state visitation densities by training an ensemble of discriminators, and assigns reward bonuses to rarely visited states."

----
[Variational Intrinsic Control](http://arxiv.org/abs/1611.07507) (DeepMind)  
>	"The second scenario is that in which the long-term goal of the agent is to get to a state with a maximal set of available intrinsic options – the objective of empowerment (Salge et al., 2014). This set of options consists of those that the agent knows how to use. Note that this is not the theoretical set of all options: it is of no use to the agent that it is possible to do something if it is unable to learn how to do it. Thus, to maximize empowerment, the agent needs to simultaneously learn how to control the environment as well – it needs to discover the options available to it. The agent should in fact not aim for states where it has the most control according to its current abilities, but for states where it expects it will achieve the most control after learning. Being able to learn available options is thus fundamental to becoming empowered."  
>	"Let us compare this to the commonly used intrinsic motivation objective of maximizing the amount of model-learning progress, measured as the difference in compression of its experience before and after learning (Schmidhuber, 1991; 2010; Bellemare et al., 2016; Houthooft et al., 2016). The empowerment objective differs from this in a fundamental manner: the primary goal is not to understand or predict the observations but to control the environment. This is an important point – agents can often control an environment perfectly well without much understanding, as exemplified by canonical model-free reinforcement learning algorithms (Sutton & Barto, 1998), where agents only model action-conditioned expected returns. Focusing on such understanding might significantly distract and impair the agent, as such reducing the control it achieves."  

[Towards Information-Seeking Agents](http://arxiv.org/abs/1612.02605) (Maluuba)  
  - <https://youtube.com/watch?v=3bSquT1zqj8> (demo)  

[Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play](http://arxiv.org/abs/1703.05407) (Facebook)  
  - <https://youtube.com/watch?v=EHHiFwStqaA> (demo)  
  - <https://youtube.com/watch?v=X1O21ziUqUY> (Fergus)  



---
### reinforcement learning - abstractions for states and actions

[Variational Intrinsic Control](http://arxiv.org/abs/1611.07507) (DeepMind)  

[A Laplacian Framework for Option Discovery in Reinforcement Learning](https://arxiv.org/abs/1703.00956) (Bowling)
>	"Our algorithm can be seen as a bottom-up approach, in which we construct options before the agent observes any informative reward. Options discovered this way tend to be independent of an agent’s intention, and are potentially useful in many different tasks. Moreover, such options can also be seen as being useful for exploration by allowing agents to commit to a behavior for an extended period of time."  
  - <https://youtube.com/watch?v=2BVicx4CDWA> (demo)  

[Strategic Attentive Writer for Learning Macro-Actions](http://arxiv.org/abs/1606.04695) (DeepMind)  
>	"Learning temporally extended actions and temporal abstraction in general is a long standing problem in reinforcement learning. They facilitate learning by enabling structured exploration and economic computation. In this paper we present a novel deep recurrent neural network architecture that learns to build implicit plans in an end-to-end manner purely by interacting with an environment in a reinforcement learning setting. The network builds an internal plan, which is continuously updated upon observation of the next input from the environment. It can also partition this internal representation into contiguous sub-sequences by learning for how long the plan can be committed to – i.e. followed without replanning. Combining these properties, the proposed model, dubbed STRategic Attentive Writer (STRAW) can learn high-level, temporally abstracted macro-actions of varying lengths that are solely learnt from data without any prior information."  
  - <https://youtube.com/watch?v=niMOdSu3yio> (demo)  
  - <https://blog.acolyer.org/2017/01/06/strategic-attentive-writer-for-learning-macro-actions/>  
  - <http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-3-synergistic-and-modular-action/> (Mohamed)  

[The Option-Critic Architecture](http://arxiv.org/abs/1609.05140) (Precup)  
  - <https://youtube.com/watch?v=8r_EoYnPjGk> (Bacon)  

[Options Discovery with Budgeted Reinforcement Learning](https://arxiv.org/abs/1611.06824)  

----
[Modular Multitask Reinforcement Learning with Policy Sketches](http://arxiv.org/abs/1611.01796) (Levine)  
  - <https://youtube.com/watch?v=NRIcDEB64x8> (Andreas)  
  - <https://github.com/jacobandreas/psketch>  

[Stochastic Neural Networks for Hierarchical Reinforcement Learning](http://openreview.net/forum?id=B1oK8aoxe) (Abbeel)  
>	"Our SNN hierarchical approach outperforms state-of-the-art intrinsic motivation results like VIME (Houthooft et al., 2016)."  
  - <https://youtube.com/playlist?list=PLEbdzN4PXRGVB8NsPffxsBSOCcWFBMQx3> (demo)  

[FeUdal Networks for Hierarchical Reinforcement Learning](http://arxiv.org/abs/1703.01161) (Silver)  

[Learning and Transfer of Modulated Locomotor Controllers](http://arxiv.org/abs/1610.05182) (Silver)  
  - <https://youtube.com/watch?v=sboPYvhpraQ> (demo)  

[A Deep Hierarchical Approach to Lifelong Learning in Minecraft](http://arxiv.org/abs/1604.07255)  
  - <https://youtube.com/watch?v=RwjfE4kc6j8> (demo)  

----
[Principled Option Learning in Markov Decision Processes](https://arxiv.org/abs/1609.05524) (Tishby)  
>	"We suggest a mathematical characterization of good sets of options using tools from information theory. This characterization enables us to find conditions for a set of options to be optimal and an algorithm that outputs a useful set of options and illustrate the proposed algorithm in simulation."  



---
### reinforcement learning - simulation and planning

[Prediction and Control with Temporal Segment Models](https://arxiv.org/abs/1703.04070) (Abbeel)  
>	"variational autoencoder to learn the distribution over future state trajectories conditioned on past states, past actions, and planned future actions"  
>	"latent action prior, another variational autoencoder that models a prior over action segments, and showed how it can be used to perform control using actions from the same distribution as a dynamics model’s training data"  

[Recurrent Environment Simulators](https://arxiv.org/abs/1704.02254) (DeepMind)  
  - <https://sites.google.com/site/resvideos1729/> (demo)  

----
[Metacontrol for Adaptive Imagination-Based Optimization](https://openreview.net/forum?id=Bk8BvDqex) (DeepMind)  
>	"Rather than learning a single, fixed policy for solving all instances of a task, we introduce a metacontroller which learns to optimize a sequence of "imagined" internal simulations over predictive models of the world in order to construct a more informed, and more economical, solution. The metacontroller component is a model-free reinforcement learning agent, which decides both how many iterations of the optimization procedure to run, as well as which model to consult on each iteration. The models (which we call "experts") can be state transition models, action-value functions, or any other mechanism that provides information useful for solving the task, and can be learned on-policy or off-policy in parallel with the metacontroller."  

[The Predictron: End-to-End Learning and Planning](https://openreview.net/forum?id=BkJsCIcgl) (Silver)  
  - <https://youtube.com/watch?v=BeaLdaN2C3Q>  
  - <https://github.com/zhongwen/predictron>  
  - <https://github.com/muupan/predictron>  

[Reinforcement Learning via Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1701.02392)  
>	"solving Markov Decision Processes and Reinforcement Learning problems using Recurrent Convolutional Neural Networks"  
>	"1. Solving Value / Policy Iteration in a standard MDP using Feedforward passes of a Value Iteration RCNN.  
>	2. Representing the Bayes Filter state belief update as feedforward passes of a Belief Propagation RCNN.  
>	3. Learning the State Transition models in a POMDP setting, using backpropagation on the Belief Propagation RCNN.  
>	4. Learning Reward Functions in an Inverse Reinforcement Learning framework from demonstrations, using a QMDP RCNN."  
  - <https://youtube.com/watch?v=gpwA3QNTPOQ> (Shankar)  
  - <https://github.com/tanmayshankar/RCNN_MDP>  

[Value Iteration Networks](http://arxiv.org/abs/1602.02867) (Abbeel)  
  - <https://youtube.com/watch?v=tXBHfbHHlKc> (Tamar) + <http://technion.ac.il/~danielm/icml_slides/Talk7.pdf>  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Value-Iteration-Networks> (Tamar)  
  - <https://github.com/karpathy/paper-notes/blob/master/vin.md>  
  - <https://blog.acolyer.org/2017/02/09/value-iteration-networks/>  
  - <https://github.com/avivt/VIN>  
  - <https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks>  
  - <https://github.com/zuoxingdong/VIN_PyTorch_Visdom>  

----
[Blazing the Trails before Beating the Path: Sample-efficient Monte-Carlo Planning](https://papers.nips.cc/paper/6253-blazing-the-trails-before-beating-the-path-sample-efficient-monte-carlo-planning.pdf) (Munos)  
>	"We study the sampling-based planning problem in Markov decision processes (MDPs) that we can access only through a generative model, usually referred to as Monte-Carlo planning."  
>	"Our objective is to return a good estimate of the optimal value function at any state while minimizing the number of calls to the generative model, i.e. the sample complexity."  
>	"TrailBlazer is an adaptive algorithm that exploits possible structures of the MDP by exploring only a subset of states reachable by following near-optimal policies."  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Blazing-the-trails-before-beating-the-path-Sample-efficient-Monte-Carlo-planning>  



---
### reinforcement learning - transfer

[Generalizing Skills with Semi-Supervised Reinforcement Learning](http://arxiv.org/abs/1612.00429) (Abbeel, Levine)  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-3> (39:26, Levine)  

----
[Successor Features for Transfer in Reinforcement Learning](http://arxiv.org/abs/1606.05312) (Silver)  

[Learning Modular Neural Network Policies for Multi-Task and Multi-Robot Transfer](http://arxiv.org/abs/1609.07088) (Abbeel, Levine)  
  - <https://youtube.com/watch?v=n4EgRwzJE1o>  

[Learning Invariant Feature Spaces to Transfer Skills with Reinforcement Learning](https://openreview.net/pdf?id=Hyq4yhile) (Abbeel, Levine)  

----
[Policy Distillation](http://arxiv.org/abs/1511.06295) (DeepMind)  
>	"Our new paper uses distillation to consolidate lots of policies into a single deep network. This works remarkably well, and can be applied online, during Q-learning, so that policies are compressed, distilled, and refined whilst being learned. Atari policies are actually improved through distillation and generalize better (with higher scores and lower variance) during novel starting state evaluation."  

[Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning](http://arxiv.org/abs/1511.06342) (Salakhutdinov)  
  - <https://github.com/eparisotto/ActorMimic>  

----
[Progressive Neural Networks](http://arxiv.org/abs/1606.04671) (DeepMind)  
  - <https://youtube.com/watch?v=aWAP_CWEtSI> (Hadsell)  
  - <http://techtalks.tv/talks/progressive-nets-for-sim-to-real-transfer-learning/63043/> (Hadsell)  
  - <https://blog.acolyer.org/2016/10/11/progressive-neural-networks/>  
  - <https://github.com/synpon/prog_nn>  



---
### reinforcement learning - imitation

[Deeply AggreVaTeD: Differentiable Imitation Learning for Sequential Prediction](https://arxiv.org/abs/1703.01030)

----
[Generative Adversarial Imitation Learning](http://arxiv.org/abs/1606.03476)  
>	"Uses a GAN framework to discriminate between teacher and student experience and force the student to behave close to the teacher."  
  - <https://github.com/openai/imitation>  
  - <https://github.com/DanielTakeshi/rl_algorithms/tree/master/il>  

[Third Person Imitation Learning](https://arxiv.org/abs/1703.01703) (Abbeel, Sutskever)  
>	"The authors propose a new framework for learning a policy from third-person experience. This is different from standard imitation learning which assumes the same "viewpoint" for teacher and student. The authors build upon Generative Adversarial Imitation Learning, which uses a GAN framework to discriminate between teacher and student experience and force the student to behave close to the teacher. However, when using third-person experience from a different viewpoint the discriminator would simply learn to discriminate between viewpoints instead of behavior and the framework isn't easily applicable. The authors' solution is to add a second discriminator to maximize a domain confusion loss based on the same feature representation. The objective is to learn the same (viewpoint-independent) feature representation for both teacher and student experience while also learning to discriminate between teacher and student observations. In other words, the objective is to maximize domain confusion while minimizing class loss. In practice, this is another discriminator term in the GAN objective. The authors also found that they need to feed observations at time t+n (n=4 in experiments) to signal the direction of movement in the environment."  
  - <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/third-person-imitation-learning.md>  

[Model-based Adversarial Imitation Learning](http://arxiv.org/abs/1612.02179)  

----
[One-Shot Imitation Learning](http://arxiv.org/abs/1703.07326) (OpenAI)  
  - <http://bit.ly/one-shot-imitation>  

[Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](http://arxiv.org/abs/1603.00448) (Abbeel)  
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-3> (22:48, Levine)  

[Unsupervised Perceptual Rewards for Imitation Learning](http://arxiv.org/abs/1612.06699) (Levine)  
>	"To our knowledge, these are the first results showing that complex robotic manipulation skills can be learned directly and without supervised labels from a video of a human performing the task."  



---
### reinforcement learning - memory

[Neural Episodic Control](https://arxiv.org/abs/1703.01988) (DeepMind)  
>	"Our agent uses a semi-tabular representation of the value function: a buffer of past experience containing slowly changing state representations and rapidly updated estimates of the value function."  
>
>	"Greedy non-parametric tabular-memory agents like MFEC can outperform model-based agents when data are noisy or scarce.  
>	NEC outperforms MFEC by creating an end-to-end trainable learning system using differentiable neural dictionaries and a convolutional neural network.  
>	A representation of the environment as generated by the mammalian brain's ventral stream can be approximated with random projections, a variational autoencoder, or a convolutional neural network."  
  - <http://rylanschaeffer.github.io/content/research/neural_episodic_control/main.html>  

[Model-Free Episodic Control](http://arxiv.org/abs/1606.04460) (DeepMind)  
>	"This might be achieved by a dual system (hippocampus vs neocortex <http://wixtedlab.ucsd.edu/publications/Psych%20218/McClellandMcNaughtonOReilly95.pdf> ) where information are stored in alternated way such that new nonstationary experience is rapidly encoded in the hippocampus (most flexible region of the brain with the highest amount of plasticity and neurogenesis); long term memory in the cortex is updated in a separate phase where what is updated (both in terms of samples and targets) can be controlled and does not put the system at risk of instabilities."  
  - <https://sites.google.com/site/episodiccontrol/> (demo)  
  - <http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-4-episodic-and-interactive-memory/>  
  - <https://github.com/ShibiHe/Model-Free-Episodic-Control>  
  - <https://github.com/sudeepraja/Model-Free-Episodic-Control>  

[A Growing Long-term Episodic and Semantic Memory](http://arxiv.org/abs/1610.06402)  

[Memory-based Control with Recurrent Neural Networks](http://arxiv.org/abs/1512.04455) (Silver)  
  - <https://youtube.com/watch?v=V4_vb1D5NNQ> (demo)  



---
### reinforcement learning - applications

[Deep Reinforcement Learning: An Overview](http://arxiv.org/abs/1701.07274)  

[DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker](http://arxiv.org/abs/1701.01724) (Bowling)  
  - <http://science.sciencemag.org/content/early/2017/03/01/science.aam6960>  
  - <http://deepstack.ai>  
  - <http://twitter.com/DeepStackAI>  

[Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning](http://arxiv.org/abs/1603.07954) (Barzilay)  
  - <https://github.com/karthikncode/DeepRL-InformationExtraction>  

[Neural Combinatorial Optimization with Reinforcement Learning](http://arxiv.org/abs/1611.09940) (Google Brain)  

[Learning to Navigate in Complex Environments](http://arxiv.org/abs/1611.03673) (DeepMind)  
  - <http://youtube.com/watch?v=5Rflbx8y7HY> (Mirowski)  
  - <http://pemami4911.github.io/paper-summaries/2016/12/20/learning-to-navigate-in-complex-envs.html>  

[Learning Runtime Parameters in Computer Systems with Delayed Experience Injection](http://arxiv.org/abs/1610.09903)  

[Mastering 2048 with Delayed Temporal Coherence Learning, Multi-State Weight Promotion, Redundant Encoding and Carousel Shaping](http://arxiv.org/abs/1604.05085)  

[Towards Deep Symbolic Reinforcement Learning](http://arxiv.org/abs/1609.05518)  
  - <https://youtube.com/watch?v=HOAVhPy6nrc> (Shanahan)  

[Self-critical Sequence Training for Image Captioning](http://arxiv.org/abs/1612.00563)  # REINFORCE with reward normalization but without baseline estimation  



---
### dialog systems

[How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](http://arxiv.org/abs/1603.08023)  
  - <http://www.shortscience.org/paper?bibtexKey=journals/corr/LiuLSNCP16#shagunsodhani>  

[On the Evaluation of Dialogue Systems with Next Utterance Classification](http://arxiv.org/abs/1605.05414)  

[Towards an Automatic Turing Test: Learning to Evaluate Dialogue Responses](http://openreview.net/forum?id=HJ5PIaseg)  
  - <https://youtube.com/watch?v=vTgwWobuoFw> (Pineau)  

----
[Emergence of Grounded Compositional Language in Multi-Agent Populations](http://arxiv.org/abs/1703.04908) (OpenAI)  
>	"Though the agents come up with words that we found to correspond to objects and other agents, as well as actions like 'Look at' or 'Go to', to the agents these words are abstract symbols represented by one-hot vector - we label these one-hot vectors with English words that capture their meaning for the sake of interpretability."  
>
>	"One possible scenario is from goal oriented-dialog systems. Where one agent tries to transmit to another certain API call that it should perform (book restaurant, hotel, whatever). I think these models can make it more data efficient. At the first stage two agents have to communicate and discover their own language, then you can add regularization to make the language look more like natural language and on the final stage, you are adding a small amount of real data (dialog examples specific for your task). I bet that using additional communication loss will make the model more data efficient."  
>
>	"The big outcome to hunt for in this space is a post-gradient descent learning algorithm. Of course you can make agents that play the symbol grounding game, but it's not a very big step from there to compression of data, and from there to compression of 'what you need to know to solve the problem you're about to encounter' - at which point you have a system which can learn by training or learn by receiving messages. It was pretty easy to get stuff like one agent learning a classifier, encoding it in a message, and transmitting it to a second agent who has to use it for zero-shot classification. But it's still single-task specific communication, so there's no benefit to the agent for receiving, say, the messages associated with the previous 100 tasks. The tricky thing is going from there to something more abstract and cumulative, so that you can actually use message generation as an iterative learning mechanism. I think a large part of that difficulty is actually designing the task ensemble, not just the network architecture."  
  - ["A Paradigm for Situated and Goal-Driven Language Learning"](https://arxiv.org/abs/1610.03585)  
  - <https://blog.openai.com/learning-to-communicate/>  
  - <http://videos.re-work.co/videos/366-learning-to-communicate> (Lowe)  

[Learning to Communicate with Deep Multi-Agent Reinforcement Learning](http://arxiv.org/abs/1605.06676) (de Freitas)  
  - <https://youtube.com/watch?v=cfsYBY4nd1c>  
  - <https://youtube.com/watch?v=xL-GKD49FXs> (Foerster)  
  - <http://videolectures.net/deeplearning2016_foerster_learning_communicate/> (Foerster)  
  - <http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.07133>  
  - <https://github.com/iassael/learning-to-communicate>  

[Learning Multiagent Communication with Backpropagation](http://arxiv.org/abs/1605.07736) (Fergus)  
  - <https://github.com/facebookresearch/CommNet>  

----
[Learning from Real Users: Rating Dialogue Success with Neural Networks for Reinforcement Learning in Spoken Dialogue Systems](http://arxiv.org/abs/1508.03386) (Young)  

[On-line Active Reward Learning for Policy Optimisation in Spoken Dialogue Systems](http://arxiv.org/abs/1605.07669) (Young)  

[Continuously Learning Neural Dialogue Management](http://arxiv.org/abs/1606.02689) (Young)  

[Online Sequence-to-Sequence Reinforcement Learning for Open-domain Conversational Agents](http://arxiv.org/abs/1612.03929)  

----
[Generative Deep Neural Networks for Dialogue: A Short Review](http://arxiv.org/abs/1611.06216) (Pineau)  

[Emulating Human Conversations using Convolutional Neural Network-based IR](http://arxiv.org/abs/1606.07056)  

[Two are Better than One: An Ensemble of Retrieval- and Generation-Based Dialog Systems](http://arxiv.org/abs/1610.07149)  

----
[Hybrid Code Networks: Practical and Efficient End-to-end Dialog Control with Supervised and Reinforcement Learning](https://arxiv.org/abs/1702.03274) (Zweig)  
>	"End-to-end methods lack a general mechanism for injecting domain knowledge and constraints. For example, simple operations like sorting a list of database results or updating a dictionary of entities can expressed in a few lines of software, yet may take thousands of dialogs to learn. Moreover, in some practical settings, programmed constraints are essential – for example, a banking dialog system would require that a user is logged in before they can retrieve account information."  
>	"In addition to learning an RNN, HCNs also allow a developer to express domain knowledge via software and action templates."  

[Deep Reinforcement Learning for Dialogue Generation](http://arxiv.org/abs/1606.01541) (Jurafsky)  

[Adversarial Learning for Neural Dialogue Generation](http://arxiv.org/abs/1701.06547) (Jurafsky)  
  - <https://github.com/jiweil/Neural-Dialogue-Generation>  

[Policy Networks with Two-Stage Training for Dialogue Systems](http://arxiv.org/abs/1606.03152) (Maluuba)  
  - <http://www.maluuba.com/blog/2016/11/23/deep-reinforcement-learning-in-dialogue-systems>  

[End-to-End Reinforcement Learning of Dialogue Agents for Information Access](http://arxiv.org/abs/1609.00777) (Deng)  

[Efficient Exploration for Dialog Policy Learning with Deep BBQ Networks & Replay Buffer Spiking](http://arxiv.org/abs/1608.05081) (Deng)  

[End-to-End LSTM-based Dialog Control Optimized with Supervised and Reinforcement Learning](http://arxiv.org/abs/1606.01269) (Zweig)  

----
[A Network-based End-to-End Trainable Task-oriented Dialogue System](http://arxiv.org/abs/1604.04562) (Young)  
  - <http://videolectures.net/deeplearning2016_wen_network_based/> (Wen)  

[Learning Language Games through Interaction](http://arxiv.org/abs/1606.02447) (Manning)  
  - <https://youtube.com/watch?v=iuazFltYgCE> (Wang)

[Learning End-to-End Goal-Oriented Dialog](http://arxiv.org/abs/1605.07683) (Weston)  

[Neural Belief Tracker: Data-Driven Dialogue State Tracking](http://arxiv.org/abs/1606.03777) (Young)  

[A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue](http://arxiv.org/abs/1701.04024) (Manning)  

[Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation](http://arxiv.org/abs/1606.00776) (Bengio)  

[A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues](http://arxiv.org/abs/1605.06069) (Bengio)  

[An Attentional Neural Conversation Model with Improved Specificity](http://arxiv.org/abs/1606.01292) (Zweig)  

[Towards Conversational Recommender Systems](http://kdd.org/kdd2016/subtopic/view/towards-conversational-recommender-systems) (Hoffman)  
  - <https://periscope.tv/WiMLworkshop/1vAGRXDbvbkxl> (Christakopoulou)  
  - <https://youtube.com/watch?v=nLUfAJqXFUI> (Christakopoulou)  

----
[LSTM-based Mixture-of-Experts for Knowledge-Aware Dialogues](http://arxiv.org/abs/1605.01652)  

[Multi-domain Neural Network Language Generation for Spoken Dialogue Systems](http://arxiv.org/abs/1603.01232)  

[Sentence Level Recurrent Topic Model: Letting Topics Speak for Themselves](http://arxiv.org/abs/1604.02038)  

[Context-aware Natural Language Generation with Recurrent Neural Networks](http://arxiv.org/abs/1611.09900)  

[Data Distillation for Controlling Specificity in Dialogue Generation](http://arxiv.org/abs/1702.06703) (Jurafsky)  

----
[A Persona-Based Neural Conversation Model](http://arxiv.org/abs/1603.06155)  
  - <https://github.com/jiweil/Neural-Dialogue-Generation>  

[Conversational Contextual Cues: The Case of Personalization and History for Response Ranking](http://arxiv.org/abs/1606.00372) (Kurzweil)  

[A Sequence-to-Sequence Model for User Simulation in Spoken Dialogue Systems](http://arxiv.org/abs/1607.00070) (Maluuba)  

----
[Deep Contextual Language Understanding in Spoken Dialogue Systems](http://research.microsoft.com/apps/pubs/default.aspx?id=256085)  

[Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning](http://arxiv.org/abs/1603.07954) (Barzilay)  



---
### natural language processing

[Exploring the Limits of Language Modeling](http://arxiv.org/abs/1602.02410) (Vinyals)  # perplexity increase from 50 to 30  
  - <http://deliprao.com/archives/201>  
  - <https://github.com/tensorflow/models/tree/master/lm_1b>  

[Improving Neural Language Models with a Continuous Cache](http://arxiv.org/abs/1612.04426) (Facebook)  # adaptive softmax  

[Pointer Sentinel Mixture Models](http://arxiv.org/abs/1609.07843) (MetaMind)  
>	"The authors combine a standard LSTM softmax with Pointer Networks in a mixture model called Pointer-Sentinel LSTM (PS-LSTM). The pointer networks helps with rare words and long-term dependencies but is unable to refer to words that are not in the input. The oppoosite is the case for the standard softmax."

----
[Towards Universal Paraphrastic Sentence Embeddings  ](http://arxiv.org/abs/1511.08198)# outperforming LSTM  
  - <http://videolectures.net/iclr2016_wieting_universal_paraphrastic/> (Wieting)  

----
[Bag of Tricks for Efficient Text Classification](http://arxiv.org/abs/1607.01759) (Mikolov)  # fastText  
>	"Train the word-vectors so that they directly predict the target classes, almost as if those target classes were the nearby-context words (that would normally be predicted in classic word2vec). If you squint just right, it's also a little bit like Paragraph Vectors (often aka 'Doc2Vec'), backwards. Instead of an alternate input vector, which combines with all words to help predict words, it's alternate output nodes, to which all words contribute to the prediction."    
>	"As the paper says, people have been using low-rank classifiers for 10+ years (e.g. <http://www.jmlr.org/papers/volume6/ando05a/ando05a.pdf>). Jason Weston of FAIR did a lot of work to popularize these embedding models in more recent times (e.g. <http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf>). This kind of approach is also essentially the same as many feature-based matrix factorization models (here's a survey <https://arxiv.org/abs/1109.2271>).  
>	Use of hierarchical softmax for large label spaces also isn't new (e.g. <https://arxiv.org/abs/1412.7479> or <https://people.cs.umass.edu/~arvind/akbc-hierarchical.pdf>). Other relevant recent work on simple embedding-baselines for text classification: Deep Unordered Composition Rivals Syntactic Methods for Text Classification <https://www.cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf>"  
  - <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1607.01759#shagunsodhani>  
  - <https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py>  

----
[Order-Embeddings of Images and Language](http://arxiv.org/abs/1511.06361)  
  - <http://videolectures.net/iclr2016_vendrov_order_embeddings/> (Vendrov)  
  - <https://github.com/ivendrov/order-embedding>  
  - <https://github.com/ivendrov/order-embeddings-wordnet>  
  - <https://github.com/LeavesBreathe/tensorflow_with_latest_papers/blob/master/partial_ordering_embedding.py>  

----
[Semantic Parsing with Semi-Supervised Sequential Autoencoders](http://arxiv.org/abs/1609.09315) (DeepMind)  # discrete VAE  

[Open-Vocabulary Semantic Parsing with both Distributional Statistics and Formal Knowledge](http://arxiv.org/abs/1607.03542) (Gardner)  

----
[Pointing the Unknown Words](http://arxiv.org/abs/1603.08148) (Bengio)  

[Machine Comprehension Using Match-LSTM And Answer Pointer](http://arxiv.org/abs/1608.07905)  

----
[Neural Variational Inference for Text Processing](http://arxiv.org/abs/1511.06038) (Blunsom)  
  - <http://dustintran.com/blog/neural-variational-inference-for-text-processing/>  
  - <https://github.com/carpedm20/variational-text-tensorflow>  
  - <https://github.com/cheng6076/NVDM>  

[Generating Sentences from a Continuous Space](http://arxiv.org/abs/1511.06349)  
  - <https://github.com/analvikingur/pytorch_RVAE>  
  - <https://github.com/cheng6076/Variational-LSTM-Autoencoder>  




<brylevkirill (at) gmail.com>
