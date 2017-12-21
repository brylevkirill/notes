interesting recent papers:

  * [deep learning theory](#deep-learning-theory)
  * [bayesian deep learning](#bayesian-deep-learning)
  * [compute and memory architectures](#compute-and-memory-architectures)
  * [meta-learning](#meta-learning)
  * [few-shot learning](#few-shot-learning)
  * [unsupervised learning](#unsupervised-learning)
  * [generative models](#generative-models)
    - [invertible density estimation](#generative-models---invertible-density-estimation)
    - [generative adversarial networks](#generative-models---generative-adversarial-networks)
    - [variational autoencoders](#generative-models---variational-autoencoders)
    - [autoregressive models](#generative-models---autoregressive-models)
  * [reinforcement learning](#reinforcement-learning---model-free-methods)
    - [model-free methods](#reinforcement-learning---model-free-methods)
    - [model-based methods](#reinforcement-learning---model-based-methods)
    - [exploration and intrinsic motivation](#reinforcement-learning---exploration-and-intrinsic-motivation)
    - [hierarchical](#reinforcement-learning---hierarchical)
    - [transfer](#reinforcement-learning---transfer)
    - [imitation](#reinforcement-learning---imitation)
    - [multi-agent](#reinforcement-learning---multi-agent)
  * [program induction](#program-induction)
  * [reasoning](#reasoning)
  * [language grounding](#language-grounding)
  * [natural language processing](#natural-language-processing)

----
interesting older papers:

  - [artificial intelligence](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#interesting-papers)
  - [knowledge representation and reasoning](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers)
  - [machine learning](https://github.com/brylevkirill/notes/blob/master/Machine%20Learning.md#interesting-papers)
  - [deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers)
  - [reinforcement learning](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers)
  - [bayesian inference and learning](https://github.com/brylevkirill/notes/blob/master/Bayesian%20Inference%20and%20Learning.md#interesting-papers)
  - [probabilistic programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md#interesting-papers)
  - [natural language processing](https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#interesting-papers)
  - [information retrieval](https://github.com/brylevkirill/notes/blob/master/Information%20Retrieval.md#interesting-papers)



---
### deep learning theory

[interesting older papers](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---theory)

----
#### ["Mathematics of Deep Learning"](https://arxiv.org/abs/1712.04741) Vidal, Bruna, Giryes, Soatto

----
#### ["Understanding Deep Learning Requires Rethinking Generalization"](http://arxiv.org/abs/1611.03530) Zhang, Bengio, Hardt, Recht, Vinyals
  `generalization`
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
  - `post` <http://www.offconvex.org/2017/12/08/generalization1/> (Arora)

#### ["A Closer Look at Memorization in Deep Networks"](https://arxiv.org/abs/1706.05394) Arpit et al.
  `generalization`
>	"We examine the role of memorization in deep learning, drawing connections to capacity, generalization, and adversarial robustness. While deep networks are capable of memorizing noise data, our results suggest that they tend to prioritize learning simple patterns first. In our experiments, we expose qualitative differences in gradient-based optimization of deep neural networks on noise vs. real data. We also demonstrate that for appropriately tuned explicit regularization (e.g., dropout) we can degrade DNN training performance on noise datasets without compromising generalization on real data. Our analysis suggests that the notions of effective capacity which are dataset independent are unlikely to explain the generalization performance of deep networks when trained with gradient based methods because training data itself plays an important role in determining the degree of memorization."  
  - `video` <https://vimeo.com/238241921> (Krueger)

#### ["A Bayesian Perspective on Generalization and Stochastic Gradient Descent"](https://arxiv.org/abs/1710.06451) Smith, Le
  `generalization`
>	"How can we predict if a minimum will generalize to the test set, and why does stochastic gradient descent find minima that generalize well? Our work is inspired by Zhang et al. (2017), who showed deep networks can easily memorize randomly labeled training data, despite generalizing well when shown real labels of the same inputs. We show here that the same phenomenon occurs in small linear models. These observations are explained by evaluating the Bayesian evidence, which penalizes sharp minima but is invariant to model parameterization. We also explore the "generalization gap" between small and large batch training, identifying an optimum batch size which maximizes the test set accuracy."  
>	"The optimum batch size is proportional to the learning rate and the training set size. We verify these predictions empirically."  

#### ["On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"](https://arxiv.org/abs/1609.04836) Keskar, Mudigere, Nocedal, Smelyanskiy, Tang
  `generalization`
>	"Deep networks generalise better with smaller batch-size when no other form of regularisation is used. And it may be because SGD biases learning towards flat local minima, rather than sharp local minima."  
>	"Using large batch sizes tends to find sharped minima and generalize worse. We can’t talk about generalization without taking training algorithm into account."  
  - `post` <http://inference.vc/everything-that-works-works-because-its-bayesian-2/>
  - `code` <https://github.com/keskarnitish/large-batch-training>

#### ["Sharp Minima Can Generalize For Deep Nets"](https://arxiv.org/abs/1703.04933) Dinh, Pascanu, Bengio, Bengio
  `generalization`
  - `video` <https://vimeo.com/237275513> (Dinh)

#### ["mixup: Beyond Empirical Risk Minimization"](https://arxiv.org/abs/1710.09412) Zhang, Cisse, Dauphin, Lopez-Paz
  `generalization`
>	"mixup trains a neural network on convex combinations of pairs of examples and their labels. By doing so, mixup regularizes the neural network to favor simple linear behavior in-between training examples."  
>	"On the one hand, Empirical Risk Minimization allows large neural networks to memorize (instead of generalize from) the training data even in the presence of strong regularization, or in classification problems where the labels are assigned at random. On the other hand, neural networks trained with ERM change their predictions drastically when evaluated on examples just outside the training distribution, also known as adversarial examples. This evidence suggests that ERM is unable to explain or provide generalization on testing distributions that differ only slightly from the training data."  
>	"In Vicinal Risk Minimization, human knowledge is required to describe a vicinity or neighborhood around each example in the training data. Then, additional virtual examples can be drawn from the vicinity distribution of the training examples to enlarge the support of the training distribution. For instance, when performing image classification, it is common to define the vicinity of one image as the set of its horizontal reflections, slight rotations, and mild scalings. mixup extends the training distribution by incorporating the prior knowledge that linear interpolations of feature vectors should lead to linear interpolations of the associated targets."  
  - `post` <http://inference.vc/mixup-data-dependent-data-augmentation/>
  - `code` <https://github.com/leehomyc/mixup_pytorch>

#### ["Opening the Black Box of Deep Neural Networks via Information"](http://arxiv.org/abs/1703.00810) Shwartz-Ziv, Tishby
  `generalization`
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
>
>	"For representation Z, maximizing mutual information with the output while minimizing mutual information with the input."  
  - `video` <https://youtube.com/watch?v=bLqJHjXihK8> (Tishby)
  - `video` <https://youtube.com/watch?v=ekUWO_pI2M8> (Tishby)
  - `video` <https://youtu.be/RKvS958AqGY?t=12m7s> (Tishby)
  - `post` <https://weberna.github.io/jekyll/update/2017/11/08/Information-Bottleneck-Part1.html>
  - `post` <http://inference.vc/representation-learning-and-compression-with-the-information-bottleneck/>
  - `post` <https://medium.com/intuitionmachine/the-peculiar-behavior-of-deep-learning-loss-surfaces-330cb741ec17>
  - `notes` <https://blog.acolyer.org/2017/11/15/opening-the-black-box-of-deep-neural-networks-via-information-part-i/>
  - `notes` <https://blog.acolyer.org/2017/11/16/opening-the-black-box-of-deep-neural-networks-via-information-part-ii/>
  - `notes` <https://theneuralperspective.com/2017/03/24/opening-the-black-box-of-deep-neural-networks-via-information/>
  - `notes` <https://reddit.com/r/MachineLearning/comments/60fhyb/r_opening_the_black_box_of_deep_neural_networks/df8jsbm/>
  - `press` <https://quantamagazine.org/new-theory-cracks-open-the-black-box-of-deep-learning-20170921>

#### ["On the Emergence of Invariance and Disentangling in Deep Representations"](https://arxiv.org/abs/1706.01350) Achille, Soatto
  `generalization`
>	"We have presented bounds, some of which tight, that connect the amount of information in the weights, the amount of information in the activations, the invariance property of the network, and the geometry of the residual loss."  
>	"This leads to the somewhat surprising result that reducing information stored in the weights about the past (dataset) results in desirable properties of the representation of future data (test datum)."  

>	"We conducted experiments to validate the assumptions underlying these bounds, and found that the results match the qualitative behavior observed on real data and architectures. In particular, the theory predicts a verifiable phase transition between an underfitting and overfitting regime for random labels, and the amount of information in nats needed to cross the transition."  
  - `video` <https://youtube.com/watch?v=BCSoRTMYQcw> (Achille)

----
#### ["The Marginal Value of Adaptive Gradient Methods in Machine Learning"](https://arxiv.org/abs/1705.08292) Wilson, Roelofs, Stern, Srebro, Recht
  `optimization`
>	"Despite the fact that our experimental evidence demonstrates that adaptive methods are not advantageous for machine learning, the Adam algorithm remains incredibly popular. We are not sure exactly as to why, but hope that our step-size tuning suggestions make it easier for practitioners to use standard stochastic gradient methods in their research. In our conversations with other researchers, we have surmised that adaptive gradient methods are particularly popular for training GANs and Q-learning with function approximation. Both of these applications stand out because they are not solving optimization problems. It is possible that the dynamics of Adam are accidentally well matched to these sorts of optimization-free iterative search procedures. It is also possible that carefully tuned stochastic gradient methods may work as well or better in both of these applications."  

#### ["First-order Methods Almost Always Avoid Saddle Points"](https://arxiv.org/abs/1710.07406) Lee, Panageas, Piliouras, Simchowitz, Jordan, Recht
  `optimization`
>	"We establish that first-order methods avoid saddle points for almost all initializations. Our results apply to a wide variety of first-order methods, including gradient descent, block coordinate descent, mirror descent and variants thereof. The connecting thread is that such algorithms can be studied from a dynamical systems perspective in which appropriate instantiations of the Stable Manifold Theorem allow for a global stability analysis. Thus, neither access to second-order derivative information nor randomness beyond initialization is necessary to provably avoid saddle points."

#### ["Self-Normalizing Neural Networks"](https://arxiv.org/abs/1706.02515) Klambauer, Unterthiner, Mayr, Hochreiter
  `optimization`
>	"While batch normalization requires explicit normalization, neuron activations of SNNs automatically converge towards zero mean and unit variance. The activation function of SNNs are "scaled exponential linear units" (SELUs), which induce self-normalizing properties. Using the Banach fixed-point theorem, we prove that activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance -- even under the presence of noise and perturbations."  
>	"For activations not close to unit variance, we prove an upper and lower bound on the variance, thus, vanishing and exploding gradients are impossible."  
>
>	"Weights are initialized in such a way that for any unit in a layer with input weights wi Σ wi = 0 and Σ wi^2 = 1."  
>	"selu(x) = λx for x>0 and selu(x) = λ(αe^x − α) for x≤0, where α≈1.6733 and λ≈1.0507"  
  - `video` <https://youtube.com/watch?v=h6eQrkkU9SA> (Hochreiter)
  - `video` <https://facebook.com/nipsfoundation/videos/1555553784535855/> (47:04) (Klambauer)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1706.02515>
  - `notes` <https://bayesgroup.github.io/sufficient-statistics/posts/self-normalizing-neural-networks/> `in russian`
  - `code` <http://github.com/bioinf-jku/SNNs>

#### ["The Shattered Gradients Problem: If resnets are the answer, then what is the question?"](https://arxiv.org/abs/1702.08591) Balduzzi, Frean, Leary, Lewis, Ma, McWilliams
  `optimization`
>	"We show that the correlation between gradients in standard feedforward networks decays exponentially with depth resulting in gradients that resemble white noise."  
>	"We present a new “looks linear” (LL) initialization that prevents shattering. Preliminary experiments show the new initialization allows to train very deep networks without the addition of skip-connections."  
>	"In a randomly initialized network, the gradients of deeper layers are increasingly uncorrelated. Shattered gradients play havoc with the optimization methods currently in use and may explain the difficulty in training deep feedforward networks even when effective initialization and batch normalization are employed. Averaging gradients over minibatches becomes analogous to integrating over white noise – there is no clear trend that can be summarized in a single average direction. Shattered gradients can also introduce numerical instabilities, since small differences in the input can lead to large differences in gradients."  
>	"Skip-connections in combination with suitable rescaling reduce shattering. Specifically, we show that the rate at which correlations between gradients decays changes from exponential for feedforward architectures to sublinear for resnets. The analysis uncovers a surprising and unexpected side-effect of batch normalization."  
  - `video` <https://vimeo.com/237275640> (McWilliams)

----
#### ["Interpretation of Neural Network is Fragile"](https://arxiv.org/abs/1710.10547) Ghorbani, Abid, Zou
>	"In this paper, we show that interpretation of deep learning predictions is extremely fragile in the following sense:  two perceptively indistinguishable inputs with the same predicted label can be assigned very different interpretations. We systematically characterize the fragility of several widely-used feature-importance interpretation methods (saliency maps, relevance propagation, and DeepLIFT) on ImageNet and CIFAR-10. Our experiments show that even small random perturbation can change the feature importance and new systematic perturbations can lead to dramatically different interpretations without changing the label. We extend these results to show that interpretations based on exemplars (e.g. influence functions) are similarly fragile. Our analysis of the geometry of the Hessian matrix gives insight on why fragility could be a fundamental challenge to the current interpretation approaches."  

#### ["Understanding Black-box Predictions via Influence Functions"](https://arxiv.org/abs/1703.04730) Koh, Liang
  `interpretability`
>	"We use influence functions, a classic technique from robust statistics, to trace a model's prediction through the learning algorithm and back to its training data, thereby identifying training points most responsible for a given prediction."  
>	"We show that even on non-convex and non-differentiable models where the theory breaks down, approximations to influence functions can still provide valuable information."  
>	"On linear models and convolutional neural networks, we demonstrate that influence functions are useful for multiple purposes: understanding model behavior, debugging models, detecting dataset errors, and even creating visually-indistinguishable training-set attacks."  
  - `video` <https://youtube.com/watch?v=0w9fLX_T6tY> (Koh)
  - `video` <https://vimeo.com/237274831> (Koh)
  - `video` <https://facebook.com/academics/videos/1633085090076225/> (1:23:28) (Liang)
  - `code` <https://github.com/kohpangwei/influence-release>
  - `code` <https://github.com/darkonhub/darkon>

#### ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599) Guo, Pleiss, Sun, Weinberger
  `interpretability`
>	"Confidence calibration - the problem of predicting probability estimates representative of the true correctness likelihood - is important for classification models in many applications. We discover that modern neural networks, unlike those from a decade ago, are poorly calibrated. Through extensive experiments, we observe that depth, width, weight decay, and Batch Normalization are important factors influencing calibration. We evaluate the performance of various post-processing calibration methods on state-of-the-art architectures with image and document classification datasets. Our analysis and experiments not only offer insights into neural network learning, but also provide a simple and straightforward recipe for practical settings: on most datasets, temperature scaling - a single-parameter variant of Platt Scaling - is surprisingly effective at calibrating predictions."  
>	"A network should provide a calibrated confidence measure in addition to its prediction. In other words, the probability associated with the predicted class label should reflect its ground truth correctness likelihood. Calibrated confidence estimates are also important for model interpretability. Good confidence estimates provide a valuable extra bit of information to establish trustworthiness with the user - especially for neural networks, whose classification decisions are often difficult to interpret. Further, good probability estimates can be used to incorporate neural networks into other probabilistic models."  
  - `video` <https://vimeo.com/238242536> (Pleiss)



---
### bayesian deep learning

[interesting older papers](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---bayesian-deep-learning)  
[interesting older papers - bayesian inference and learning](https://github.com/brylevkirill/notes/blob/master/Bayesian%20Inference%20and%20Learning.md#interesting-papers)  

[interesting recent papers - variational autoencoders](#generative-models---variational-autoencoders)  
[interesting recent papers - unsupervised learning](#unsupervised-learning)  
[interesting recent papers - model-based reinforcement learning](#reinforcement-learning---model-based-methods)  
[interesting recent papers - exploration and intrinsic motivation](#reinforcement-learning---exploration-and-intrinsic-motivation)  

----
#### ["A Bayesian Perspective on Generalization and Stochastic Gradient Descent"](https://arxiv.org/abs/1710.06451) Smith, Le
  `generalization`
>	"How can we predict if a minimum will generalize to the test set, and why does stochastic gradient descent find minima that generalize well? Our work is inspired by Zhang et al. (2017), who showed deep networks can easily memorize randomly labeled training data, despite generalizing well when shown real labels of the same inputs. We show here that the same phenomenon occurs in small linear models. These observations are explained by evaluating the Bayesian evidence, which penalizes sharp minima but is invariant to model parameterization. We also explore the "generalization gap" between small and large batch training, identifying an optimum batch size which maximizes the test set accuracy."  
>	"The optimum batch size is proportional to the learning rate and the training set size. We verify these predictions empirically."  

#### ["Stochastic Gradient Descent Performs Variational Inference, Converges to Limit Cycles for Deep Networks"](https://arxiv.org/abs/1710.11029) Chaudhari, Soatto
  `generalization`
>	"We prove that SGD minimizes an average potential over the posterior distribution of weights along with an entropic regularization term. This potential is however not the original loss function in general. So SGD does perform variational inference, but for a different loss than the one used to compute the gradients. Even more surprisingly, SGD does not even converge in the classical sense: we show that the most likely trajectories of SGD for deep networks do not behave like Brownian motion around critical points. Instead, they resemble closed loops with deterministic components. We prove that such “out-of-equilibrium” behavior is a consequence of the fact that the gradient noise in SGD is highly non-isotropic; the covariance matrix of mini-batch gradients has a rank as small as 1% of its dimension."  
>	"It is widely believed that SGD is an implicit regularizer. This belief stems from its remarkable empirical performance. Our results show that such intuition is very well-placed. Thanks to the special architecture of deep networks where gradient noise is highly non-isotropic, SGD helps itself to a potential Φ with properties that lead to both generalization and acceleration."  

#### ["Stochastic Gradient Descent as Approximate Bayesian Inference"](https://arxiv.org/abs/1704.04289) Mandt, Hoffman, Blei
  `generalization`
  - `notes` <https://reddit.com/r/MachineLearning/comments/6d7nb1/d_machine_learning_wayr_what_are_you_reading_week/dihh54a/>

----
#### ["What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"](https://arxiv.org/abs/1703.04977) Kendall, Gal
  `uncertainty estimation`
>	"We presented a novel Bayesian deep learning framework to learn a mapping to aleatoric uncertainty from the input data, which is composed on top of epistemic uncertainty models. We derived our framework for both regression and classification applications.  
>	We showed that it is important to model epistemic uncertainty for:  
>	- Safety-critical applications, because epistemic uncertainty is required to understand examples which are different from training data  
>	- Small datasets where the training data is sparse.  
>	"And aleatoric uncertainty is important for:  
>	- Large data situations, where epistemic uncertainty is explained away  
>	- Real-time applications, because we can form aleatoric models without expensive Monte Carlo samples.  
>	"We can actually divide aleatoric into two further sub-categories:  
>	- Data-dependant or Heteroscedastic uncertainty is aleatoric uncertainty which depends on the input data and is predicted as a model output.  
>	- Task-dependant or Homoscedastic uncertainty is aleatoric uncertainty which is not dependant on the input data. It is not a model output, rather it is a quantity which stays constant for all input data and varies between different tasks. It can therefore be described as task-dependant uncertainty."  
>	"However aleatoric and epistemic uncertainty models are not mutually exclusive. We showed that the combination is able to achieve new state-of-the-art results on depth regression and semantic segmentation benchmarks."  
  - `post` <https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/> (Kendall)

#### ["Decomposition of Uncertainty for Active Learning and Reliable Reinforcement Learning in Stochastic Systems"](https://arxiv.org/abs/1710.07283) Depeweg, Hernandez-Lobato, Doshi-Velez, Udluft
  `uncertainty estimation`
>	"We have studied a decomposition of predictive uncertainty into its epistemic and aleatoric components when working with Bayesian neural networks with latent variables. This decomposition naturally arises when applying information-theoretic active learning setting, and it also enabled us to derive a novel risk objective for reliable reinforcement learning by decomposing risk into a model bias and noise. In both of these settings, our approch allows us to efficiently learn in the face of sophisticated stochastic functions."  
>	"We investigate safe RL using a risk-sensitive criterion which focuses only on risk related to model bias, that is, the risk of the policy performing at test time significantly different from at training time. The proposed criterion quantifies the amount of epistemic uncertainty (model bias risk) in the model’s predictive distribution and ignores any risk stemming from the aleatoric uncertainty."  
>	"We can identify two distinct forms of uncertainties in the class of models given by BNNs with latent variables. Aleatoric uncertainty captures noise inherent in the observations. On the other hand, epistemic uncertainty accounts for uncertainty in the model. In particular, epistemic uncertainty arises from our lack of knowledge of the values of the synaptic weights in the network, whereas aleatoric uncertainty originates from our lack of knowledge of the value of the latent variables. In the domain of model-based RL the epistemic uncertainty is the source of model bias. When there is high discrepancy between model and real-world dynamics, policy behavior may deteriorate. In analogy to the principle that ”a chain is only as strong as its weakest link” a drastic error in estimating the ground truth MDP at a single transition stepcan render the complete policy useless."  

#### ["A Scalable Laplace Approximation for Neural Networks"](https://openreview.net/forum?id=Skdvd2xAZ)
  `uncertainty estimation`
>	"We leverage recent insights from second-order optimisation for neural networks to construct a Kronecker factored Laplace approximation to the posterior over the weights of a trained network. Our approximation requires no modification of the training procedure, enabling practitioners to estimate the uncertainty of their models currently used in production without having to retrain them. We extensively compare our method to using Dropout and a diagonal Laplace approximation for estimating the uncertainty of a network. We demonstrate that our Kronecker factored method leads to better uncertainty estimates on out-of-distribution data and is more robust to simple adversarial attacks. We illustrate its scalability by applying it to a state-of-the-art convolutional network architecture."  
  - `video` ["Optimizing Neural Networks using Structured Probabilistic Models of the Gradient Computation"](https://www.fields.utoronto.ca/video-archive/2017/02/2267-16498) (Grosse)
  - `video` ["Optimizing NN using Kronecker-factored Approximate Curvature"](https://youtube.com/watch?v=FLV-MLPt3sU) (Kropotov)
  - `post` <https://medium.com/@yaroslavvb/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0>

#### ["Bayesian Uncertainty Estimation for Batch Normalized Deep Networks"](https://openreview.net/forum?id=BJlrSmbAZ) Lewandowski
  `uncertainty estimation`

#### ["Deep and Confident Prediction for Time Series at Uber"](https://arxiv.org/abs/1709.01907) Zhu, Laptev
  `uncertainty estimation`
  - `post` <https://eng.uber.com/neural-networks-uncertainty-estimation/>

#### ["Dropout Inference in Bayesian Neural Networks with Alpha-divergences"](https://arxiv.org/abs/1703.02914) Li, Gal
  `uncertainty estimation`
>	"We demonstrate improved uncertainty estimates and accuracy compared to VI in dropout networks. We study our model’s epistemic uncertainty far away from the data using adversarial images, showing that these can be distinguished from non-adversarial images by examining our model’s uncertainty."  
  - `video` <https://vimeo.com/238221241> (Li)
  - `code` <https://github.com/YingzhenLi/Dropout_BBalpha>

#### ["Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"](http://arxiv.org/abs/1612.01474) Lakshminarayanan, Pritzel, Blundell
  `uncertainty estimation`
>	"Adversarial Training to improve the uncertainty measure of the entropy score of the neural network."  
  - `code` <https://github.com/vvanirudh/deep-ensembles-uncertainty>

----
#### ["Model Selection in Bayesian Neural Networks via Horseshoe Priors"](https://arxiv.org/abs/1705.10388) Ghosh, Doshi-Velez
  `model selection`
>	"Model selection - even choosing the number of nodes - remains an open question. In this work, we apply a horseshoe prior over node pre-activations of a Bayesian neural network, which effectively turns off nodes that do not help explain the data. We demonstrate that our prior prevents the BNN from underfitting even when the number of nodes required is grossly over-estimated. Moreover, this model selection over the number of nodes doesn’t come at the expense of predictive or computational performance; in fact, we learn smaller networks with comparable predictive performance to current approaches."  

#### ["Structured Bayesian Pruning via Log-Normal Multiplicative Noise"](https://arxiv.org/abs/1705.07283) Neklyudov, Molchanov, Ashukha, Vetrov
  `model selection`
  - `video` <https://youtube.com/watch?v=3zEYjw-cB4Y>
  - `video` <https://youtube.com/watch?v=SjYKP8BFhgw> (Nekludov)
  - `video` <https://youtu.be/jJDVYAxyE3U?t=32m45s> (Molchanov) `in russian`
  - `code` <https://github.com/necludov/group-sparsity-sbp>
  - `paper` ["Variational Gaussian Dropout is not Bayesian"](https://arxiv.org/abs/1711.02989) by Hron, Matthews, Ghahramani

#### ["Variational Dropout Sparsifies Deep Neural Networks"](https://arxiv.org/abs/1701.05369) Molchanov, Ashukha, Vetrov
  `model selection`
>	"Interpretation of Gaussian dropout as performing variational inference in a network with log uniform priors over weights leads to sparsity in weights. This is an interesting approach, wherein sparsity stemsfrom variational optimization instead of the prior."  
  - `video` <https://vimeo.com/238221185> (Molchanov)
  - `video` <https://youtube.com/watch?v=jJDVYAxyE3U> (Molchanov) `in russian`
  - `code` <https://github.com/BayesWatch/tf-variational-dropout>
  - `paper` ["Variational Gaussian Dropout is not Bayesian"](https://arxiv.org/abs/1711.02989) by Hron, Matthews, Ghahramani

----
#### ["Implicit Causal Models for Genome-wide Association Studies"](https://arxiv.org/abs/1710.10742) Tran, Blei
  `causal inference`
>	"GAN-style models for identifying causal mutations in a GWAS with adjustment for population-based confounders."  
  - `slides` <http://dustintran.com/talks/Tran_Genomics.pdf>

#### ["Causal Effect Inference with Deep Latent-Variable Models"](https://arxiv.org/abs/1705.08821) Louizos, Shalit, Mooij, Sontag, Zemel, Welling
  `causal inference`
>	"The most important aspect of inferring causal effects from observational data is the handling of confounders, factors that affect both an intervention and its outcome. A carefully designed observational study attempts to measure all important confounders. However, even if one does not have direct access to all confounders, there may exist noisy and uncertain measurement of proxies for confounders. We build on recent advances in latent variable modeling to simultaneously estimate the unknown latent space summarizing the confounders and the causal effect. Our method is based on Variational Autoencoders which follow the causal structure of inference with proxies."  

----
#### ["Generalizing Hamiltonian Monte Carlo with Neural Networks"](https://arxiv.org/abs/1711.09268) Levy, Hoffman, Sohl-Dickstein
  `sampling` `posterior approximation`
>	"We present a general-purpose method to train Markov chain Monte Carlo kernels, parameterized by deep neural networks, that converge and mix quickly to their target distribution. Our method generalizes Hamiltonian Monte Carlo and is trained to maximize expected squared jumped distance, a proxy for mixing speed. We demonstrate large empirical gains on a collection of simple but challenging distributions, for instance achieving a 49x improvement in effective sample size in one case, and mixing when standard HMC makes no measurable progress in a second."  
>	"Hamiltonian Monte Carlo + Real NVP = trainable MCMC sampler that generalizes, and far outperforms, HMC."  
  - `code` <https://github.com/brain-research/l2hmc>

----
#### ["Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference"](https://arxiv.org/abs/1703.09194) Roeder, Wu, Duvenaud
  `variational inference` `posterior approximation`
>	"Intuitively, the reparameterization trick provides more informative gradients by exposing the dependence of sampled latent variables z on variational parameters φ. In contrast, the REINFORCE gradient estimate only depends on the relationship between the density function log qφ(z|x,φ) and its parameters. Surprisingly, even the reparameterized gradient estimate contains the score function — a special case of the REINFORCE gradient estimator. We show that this term can easily be removed, and that doing so gives even lower-variance gradient estimates in many circumstances. In particular, as the variational posterior approaches the true posterior, this gradient estimator approaches zero variance faster, making stochastic gradient-based optimization converge and "stick" to the true variational parameters."  
>	"We present a novel unbiased estimator for the variational evidence lower bound (ELBO) that has zero variance when the variational approximation is exact.  
>	We provide a simple and general implementation of this trick in terms of a single change to the computation graph operated on by standard automatic differentiation packages.  
>	We generalize our gradient estimator to mixture and importance-weighted lower bounds, and discuss extensions to flow-based approximate posteriors. This change takes a single function call using automatic differentiation packages."  
>	"The gain from using our method grows with the complexity of the approximate posterior, making it complementary to the development of non-Gaussian posterior families."  

#### ["Variational Boosting: Iteratively Refining Posterior Approximations"](http://arxiv.org/abs/1611.06585) Miller, Foti, Adams
  `variational inference` `posterior approximation`
  - `post` <http://andymiller.github.io/2016/11/23/vb.html>
  - `video` <https://vimeo.com/240609137> (Miller)
  - `video` <https://youtu.be/Jh3D8Gi4N0I?t=1h9m52s> (Nekludov) `in russian`

#### ["Operator Variational Inference"](https://arxiv.org/abs/1610.09033) Ranganath, Altosaar, Tran, Blei
  `variational inference` `posterior approximation` `OPVI`
>	"Classically, variational inference uses the Kullback-Leibler divergence to define the optimization. Though this divergence has been widely used, the resultant posterior approximation can suffer from undesirable statistical properties. To address this, we reexamine variational inference from its roots as an optimization problem. We use operators, or functions of functions, to design variational objectives. As one example, we design a variational objective with a Langevin-Stein operator. We develop a black box algorithm, operator variational inference (OPVI), for optimizing any operator objective. Importantly, operators enable us to make explicit the statistical and computational tradeoffs for variational inference. We can characterize different properties of variational objectives, such as objectives that admit data subsampling - allowing inference to scale to massive data - as well as objectives that admit variational programs - a rich class of posterior approximations that does not require a tractable density."  
>	"Operator objectives are built from an operator, a family of test functions, and a distance function. We outline the connection between operator objectives and existing divergences such as the KL divergence, and develop a new variational objective using the Langevin-Stein operator. In general, operator objectives produce new ways of posing variational inference. Given an operator objective, we develop a black box algorithm for optimizing it and show which operators allow scalable optimization through data subsampling. Further, unlike the popular evidence lower bound, not all operators explicitly depend on the approximating density. This permits flexible approximating families, called variational programs, where the distributional form is not tractable."  
  - `video` <https://youtu.be/mrj_hyH974o?t=46m37s> (Struminsky) `in russian`

#### ["Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm"](https://arxiv.org/abs/1608.04471) Liu, Wang
  `variational inference` `posterior approximation`
>	"SVGD is a general purpose variational inference algorithm that forms a natural counterpart of gradient descent for optimization. SVGD iteratively transports a set of particles to match with the target distribution, by applying a form of functional gradient descent that minimizes the KL divergence."  
>	"Based on exploiting an interesting connection between Stein discrepancy and KL divergence, we derive a new form of variational inference algorithm that mixes the advantages of variational inference, Monte Carlo, quasi Monte Carlo and gradient descent (for MAP). SVGD provides a new powerful tool for attacking the inference and learning, especially when there is a need for getting diverse outputs to capture the posterior uncertainty in the Bayesian framework."  
>
>	"In the old days we were happy with mean field approximation. Currently we don't. As the model goes more complicated, Bayesian inference needs more accurate yet fast approximations to the exact posterior, and apparently mean-field is not a very good choice. To enhance the power of variational approximations people start to use invertible transformations that warp simple distributions (e.g. factorised Gaussian) to complicated ones that are still differentiable. However it introduces an extra cost: you need to be able to compute the determinant of the Jacobian of that transform in a fast way. Solutions of this include constructing the transform with a sequence of "simple" functions -- simple in the sense that the Jacobian is low-rank or triangular. Authors provide another solution which doesn't need to compute the Jacobian at all."  
>	"Exploits the fact that KL(q||p) = -E[tr{Af(x)}] where Af(x) = f(x)(d log p(x)/dx) + d f(x)/dx for a smooth function f(x) and any continuous density p(x). This is the derivative needed for variational inference, and therefore we can draw samples from an initial distribution q0 and evolve them according to x_t+1 = x_t + A k(x,.) for a kernel k() and after some iterations they'll capture the posterior distribution. It's a similar idea to Normalizing Flows but does not require significant parametric constraints or any inversions."  
  - `post` <http://www.cs.dartmouth.edu/~dartml/project.html?p=vgd>
  - `post` <http://yingzhenli.net/home/blog/?p=536>
  - `code` <https://github.com/DartML/Stein-Variational-Gradient-Descent>

#### ["Bayesian Hypernetworks"](https://arxiv.org/abs/1710.04759) Krueger, Huang, Islam, Turner, Lacoste, Courville
  `variational inference` `posterior approximation` `normalizing flows`
>	"A Bayesian hypernetwork, h, is a neural network which learns to transform a simple noise distribution, p(ϵ)=N(0,I), to a distribution q(θ) = q(h(ϵ)) over the parameters θ of primary neural network. We train q with variational inference, using an invertible h to enable efficient estimation of the variational lower bound on the posterior p(θ|D) via sampling. In contrast to most methods for Bayesian deep learning, Bayesian hypernets can represent a complex multimodal approximate posterior with correlations between parameters, while enabling cheap i.i.d. sampling of q(θ)."  
>	"The key insight for building such a model is the use of an invertible hypernet, which enables Monte Carlo estimation of the entropy term log q(θ) in the variational inference training objective."  
>	"The number of parameters of a DNN scales approximately quadratically in the number of units per layer, so naively parametrizing a large primary net would require an impractically large hypernet. Efficient parametrization of hypernets, however, can actually compress the total size of a network. Factoring a 100x100 weight matrix, W into as W = E100×7 * H7×100 can be viewed as using a simple hypernet (H) to compress the rows of W into 7-dimensional encodings (E)."  
>	"Bayes by Backprop can be can be viewed as a trivial instance of a Bayesian hypernet, where the hypernetwork only performs an element-wise scale and shift of the noise (yielding a factorial Gaussian distribution), which is equivalent to using a hypernet with 0 coupling layers."  
>	"In order to scale BHNs to large primary networks, we use the weight normalization reparametrization: θ = gu; u = v/||v||. We only output the scaling factors g from the hypernet, and learn a maximum likelihood estimate of v. This allows us to overcome the computational limitations of naively-parametrized BHNs, since computation now scales linearly, instead of quadratically, in the number of primary net units. Using this parametrization restricts the family of approximate posteriors, but still allows for a high degree of multimodality and dependence between the parameters. Since q(θ) is now a degenerate distribution (i.e. it is entirely concentrated in a lower-dimensional manifold), we cannot compute KL(q(θ)||p(θ|D)). Instead, we treat g as a random variable, whose distribution induces a distribution (parametrized by u) over θ, and simply compute KL(q(g)||p(g|D)). Since the scale of u is fixed, the scale of g is intuitively meaningful, and we can easily translate commonly-used spherical prior distributions over θ into the priors over g."  
  - `video` <http://videolectures.net/deeplearning2017_krueger_bayesian_networks/> (Krueger)

#### ["Multiplicative Normalizing Flows for Variational Bayesian Neural Networks"](https://arxiv.org/abs/1703.01961) Louizos, Welling
  `variational inference` `posterior approximation` `normalizing flows` `auxiliary variables`
>	"We reinterpret multiplicative noise in neural networks as auxiliary random variables that augment the approximate posterior in a variational setting for Bayesian neural networks. We show that through this interpretation it is both efficient and straightforward to improve the approximation by employing normalizing flows while still allowing for local reparametrizations and a tractable lower bound."  
>	"Authors propose and dismiss Bayesian Hypernetworks due to the issues of scaling to large primary networks. They use a hypernet to generate scaling factors, z on the means µ of a factorial Gaussian distribution. Because z can follow a complicated distribution, this forms a highly flexible approximate posterior: q(θ) = ∫q(θ|z)q(z)dz. This approach also requires to introduce an auxiliary inference network to approximate p(z|θ) in order to estimate the entropy term of the variational lower bound, resulting in lower bound on the variational lower bound."  
  - `code` <https://github.com/AMLab-Amsterdam/MNF_VBNN>

#### ["Improving Variational Inference with Inverse Autoregressive Flow"](http://arxiv.org/abs/1606.04934) Kingma, Salimans, Jozefowicz, Chen, Sutskever, Welling
  `variational inference` `posterior approximation` `normalizing flows` `IAF`
>	"Most VAEs have so far been trained using crude approximate posteriors, where every latent variable is independent. Normalizing Flows have addressed this problem by conditioning each latent variable on the others before it in a chain, but this is computationally inefficient due to the introduced sequential dependencies. Inverse autoregressive flow, unlike previous work, allows us to parallelize the computation of rich approximate posteriors, and make them almost arbitrarily flexible."  
  - `code` <https://github.com/openai/iaf>

#### ["Neural Variational Inference and Learning in Undirected Graphical Models"](https://arxiv.org/abs/1711.02679) Kuleshov, Ermon
  `variational inference` `posterior approximation` `auxiliary variables` `discrete latent variables`
>	"We propose black-box learning and inference algorithms for undirected models that optimize a variational approximation to the log-likelihood of the model. Central to our approach is an upper bound on the log-partition function parametrized by a function q that we express as a flexible neural network."  
>	"Our approach makes generative models with discrete latent variables both more expressive and easier to train."  
>	"Our approach can also be used to train hybrid directed/undirected models using a unified variational framework. Such hybrid models are similar in spirit to deep belief networks. From a statistical point of view, a latent variable prior makes the model more flexible and allows it to better fit the data distribution. Such models may also learn structured feature representations: previous work has shown that undirected modules may learn classes of digits, while lower, directed layers may learn to represent finer variation. Finally, undirected models like the ones we study are loosely inspired by the brain and have been studied from that perspective. In particular, the undirected prior has been previously interpreted as an associative memory module."  
>	"Our work proposes an alternative to sampling-based learning methods; most variational methods for undirected models center on inference. Our approach scales to small and medium-sized datasets, and is most useful within hybrid directed-undirected generative models. It approaches the speed of the Persistent Contrastive Divergence method and offers additional benefits, such as partition function tracking and accelerated sampling. Most importantly, our algorithms are black-box, and do not require knowing the structure of the model to derive gradient or partition function estimators. We anticipate that our methods will be most useful in automated inference systems such as Edward."  
>	"Our approach offers a number of advantages over previous methods. First, it enables training undirected models in a black-box manner, i.e. we do not need to know the structure of the model to compute gradient estimators (e.g., as in Gibbs sampling); rather, our estimators only require evaluating a model’s unnormalized probability. When optimized jointly over q and p, our bound also offers a way to track the partition function during learning. At inference-time, the learned approximating distribution q may be used to speed-up sampling from the undirected model my initializing an MCMC chain (or it may itself provide samples)."  

#### ["Auxiliary Deep Generative Models"](http://arxiv.org/abs/1602.05473) Maaløe, Sønderby, Sønderby, Winther
  `variational inference` `posterior approximation` `auxiliary variables`
>	"Several families of q have been proposed to ensure that the approximating distribution is sufficiently flexible to fit p. This work makes use of a class of distributions q(x,a) = q(x|a)q(a) that contain auxiliary variables a; these are latent variables that make the marginal q(x) multimodal, which in turn enables it to approximate more closely a multimodal target distribution p(x)."  
  - `video` <http://techtalks.tv/talks/auxiliary-deep-generative-models/62509/> (Maaløe)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Variational-Inference-Foundations-and-Modern-Methods> (1:23:13) (Mohamed)
  - `code` <https://github.com/larsmaaloee/auxiliary-deep-generative-models>

#### ["Hierarchical Variational Models"](https://arxiv.org/abs/1511.02386) Ranganath, Tran, Blei
  `variational inference` `posterior approximation` `auxiliary variables`
>	"Several families of q have been proposed to ensure that the approximating distribution is sufficiently flexible to fit p. This work makes use of a class of distributions q(x,a) = q(x|a)q(a) that contain auxiliary variables a; these are latent variables that make the marginal q(x) multimodal, which in turn enables it to approximate more closely a multimodal target distribution p(x)."  
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Variational-Inference-Foundations-and-Modern-Methods> (1:23:13) (Mohamed)

#### ["Variational Inference for Monte Carlo Objectives"](http://arxiv.org/abs/1602.06725) Mnih, Rezende
  `variational inference` `posterior approximation` `discrete latent variables`
  - `video` <http://techtalks.tv/talks/variational-inference-for-monte-carlo-objectives/62507/>
  - `video` <https://youtu.be/_XRBlhzb31U?t=27m16s> (Figurnov) `in russian`
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/MnihR16>

#### ["Importance Weighted Autoencoders"](http://arxiv.org/abs/1509.00519) Burda, Grosse, Salakhutdinov
  `variational inference` `posterior approximation`
>	"As we show empirically, the VAE objective can lead to overly simplified representations which fail to use the network's entire modeling capacity. We present the importance weighted autoencoder, a generative model with the same architecture as the VAE, but which uses a strictly tighter log-likelihood lower bound derived from importance weighting. In the IWAE, the recognition network uses multiple samples to approximate the posterior, giving it increased flexibility to model complex posteriors which do not fit the VAE modeling assumptions."  
  - `post` <http://dustintran.com/blog/importance-weighted-autoencoders/>
  - `post` <https://casmls.github.io/general/2017/04/24/iwae-aae.html>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/BurdaGS15>
  - `code` <https://github.com/yburda/iwae>
  - `code` <https://github.com/arahuja/generative-tf>
  - `code` <https://github.com/blei-lab/edward/blob/master/examples/iwvi.py>

----
#### ["Backpropagation through the Void: Optimizing Control Variates for Black-box Gradient Estimation"](https://arxiv.org/abs/1711.00123) Grathwohl, Choi, Wu, Roeder, Duvenaud
  `variables with discrete distributions` `RELAX`
>	"We generalize REBAR to learn a free-form control variate parameterized by a neural network, giving a lower-variance, unbiased gradient estimator which can be applied to a wider variety of problems with greater flexibility. Most notably, our method is applicable even when no continuous relaxation is available, as in reinforcement learning or black box function optimization. Furthermore, we derive improved variants of popular reinforcement learning methods with unbiased, action-dependent gradient estimates and lower variance."  

#### ["REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models"](http://arxiv.org/abs/1703.07370) Tucker, Mnih, Maddison, Lawson, Sohl-Dickstein
  `variables with discrete distributions` `REBAR`
>	"Learning in models with discrete latent variables is challenging due to high variance gradient estimators. Generally, approaches have relied on control variates to reduce the variance of the REINFORCE estimator. Recent work (Jang et al. 2016; Maddison et al. 2016) has taken a different approach, introducing a continuous relaxation of discrete variables to produce low-variance, but biased, gradient estimates. In this work, we combine the two approaches through a novel control variate that produces low-variance, unbiased gradient estimates. Then, we introduce a novel continuous relaxation and show that the tightness of the relaxation can be adapted online, removing it as a hyperparameter."  
>	"Using continuous relaxation to construct a control variate for functions of discrete random variables. Low-variance estimates of the expectation of the control variate can be computed using the reparameterization trick to produce an unbiased estimator with lower variance than previous methods. Showing how to tune the free parameters of these relaxations to minimize the estimator’s variance during training."  
>	"REBAR gives unbiased gradients with lower variance than REINFORCE - self-tuning and general."  
  - `video` <https://youtube.com/watch?v=QODYgBhv_no>
  - `code` <https://github.com/tensorflow/models/tree/master/research/rebar>

#### ["Stochastic Gradient Estimation With Finite Differences"](http://approximateinference.org/accepted/BuesingEtAl2016.pdf) Buesing, Weber, Mohamed
  `variables with discrete distributions`
>	"If the loss is non-differentiable, as is the case in reinforcement learning, or if the distribution is discrete, as for probabilistic models with discrete latent variables, we have to resort to score-function (SF) gradient estimators. Naive SF estimators have high variance and therefore require sophisticated variance reduction techniques, such as baseline models, to render them effective in practice. Here we show that under certain symmetry and parametric assumptions on the distribution, one can derive unbiased stochastic gradient estimators based on finite differences (FD) of the loss function. These estimators do not require learning baseline models and potentially have less variance. Furthermore, we highlight connections of the FD estimators to simultaneous perturbation sensitivity analysis (SPSA), as well as weak derivative and “straight-through” gradient estimators."

#### ["The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables"](http://arxiv.org/abs/1611.00712) Maddison, Mnih, Teh + ["Categorical Reparametrization with Gumbel-Softmax"](http://arxiv.org/abs/1611.01144) Jang, Gu, Poole
  `variables with discrete distributions`
>	"Continuous reparemetrisation based on the so-called Concrete or Gumbel-softmax distribution, which is a continuous distribution and has a temperature constant that can be annealed during training to converge to a discrete distribution in the limit. In the beginning of training the variance of the gradients is low but biased, and towards the end of training the variance becomes high but unbiased."  
>	"Doesn't close the performance gap of VAEs with continuous latent variables where one can use the Gaussian reparameterisation trick which benefits from much lower variance in the gradients."  
  - `video` <http://youtube.com/watch?v=JFgXEbgcT7g> (Jang)
  - `video` <https://youtu.be/_XRBlhzb31U?t=28m33s> (Figurnov) `in russian`
  - `post` <http://artem.sobolev.name/posts/2017-10-28-stochastic-computation-graphs-discrete-relaxations.html>
  - `post` <https://laurent-dinh.github.io/2016/11/21/gumbel-max.html>
  - `post` <https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html>
  - `post` <http://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/>
  - `post` <https://cmaddis.github.io/gumbel-machinery>
  - `post` <https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/>
  - `code` <https://github.com/ericjang/gumbel-softmax/blob/master/gumbel_softmax_vae_v2.ipynb>
  - `code` <https://gist.github.com/gngdb/ef1999ce3a8e0c5cc2ed35f488e19748>
  - `code` <https://github.com/EderSantana/gumbel>

#### ["Reparameterizing the Birkhoff Polytope for Variational Permutation Inference"](https://arxiv.org/abs/1710.09508) Linderman, Mena, Cooper, Paninski, Cunningham
  `variables with complex discrete distributions`
>	"Many matching, tracking, sorting, and ranking problems require probabilistic reasoning about possible permutations, a set that grows factorially with dimension. Combinatorial optimization algorithms may enable efficient point estimation, but fully Bayesian inference poses a severe challenge in this high-dimensional, discrete space. To surmount this challenge, we start with the usual step of relaxing a discrete set (here, of permutation matrices) to its convex hull, which here is the Birkhoff polytope: the set of all doublystochastic matrices. We then introduce two novel transformations: first, an invertible and differentiable stick-breaking procedure that maps unconstrained space to the Birkhoff polytope; second, a map that rounds points toward the vertices of the polytope. Both transformations include a temperature parameter that, in the limit, concentrates the densities on permutation matrices."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1710.09508>

#### ["Reparameterization Gradients through Acceptance-Rejection Sampling Algorithms"](http://arxiv.org/abs/1610.05683) Naesseth, Ruiz, Linderman, Blei
  `variables with complex distributions`
>	"For many distributions of interest (such as the gamma or Dirichlet), simulation of random variables relies on acceptance-rejection sampling. The discontinuity introduced by the accept-reject step means that standard reparameterization tricks are not applicable. We propose a new method that lets us leverage reparameterization gradients even when variables are outputs of a acceptance-rejection sampling algorithm. Our approach enables reparameterization on a larger class of variational distributions."  
  - `post` <https://casmls.github.io/general/2017/04/25/rsvi.html>
  - `post` <http://artem.sobolev.name/posts/2017-09-10-stochastic-computation-graphs-continuous-case.html>

#### ["The Generalized Reparameterization Gradient"](http://arxiv.org/abs/1610.02287) Ruiz, Titsias, Blei
  `variables with complex distributions`
>	"The reparameterization gradient does not easily apply to commonly used distributions such as beta or gamma without further approximations, and most practical applications of the reparameterization gradient fit Gaussian distributions. We introduce the generalized reparameterization gradient, a method that extends the reparameterization gradient to a wider class of variational distributions. Generalized reparameterizations use invertible transformations of the latent variables which lead to transformed distributions that weakly depend on the variational parameters. This results in new Monte Carlo gradients that combine reparameterization gradients and score function gradients."  
  - `video` <https://youtu.be/mrj_hyH974o?t=1h23m40s> (Vetrov) `in russian`
  - `post` <http://artem.sobolev.name/posts/2017-09-10-stochastic-computation-graphs-continuous-case.html>

#### ["Stochastic Backpropagation through Mixture Density Distributions"](http://arxiv.org/abs/1607.05690) Graves
  `variables with mixture distributions`
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1607.05690>

#### ["Stick-Breaking Variational Autoencoders"](http://arxiv.org/abs/1605.06197) Nalisnick, Smyth
  `variables with stochastic dimensionality`
  - `post` <http://blog.shakirm.com/2015/12/machine-learning-trick-of-the-day-6-tricks-with-sticks/>
  - `code` <https://github.com/enalisnick/stick-breaking_dgms>

----
#### ["Deep Neural Networks as Gaussian Processes"](https://arxiv.org/abs/1711.00165) Lee, Bahri, Novak, Schoenholz, Pennington, Sohl-Dickstein
  `bayesian model`

#### ["Bayesian GAN"](https://arxiv.org/abs/1705.09558) Saatchi, Wilson
  `bayesian model`
>	"In this paper, we present a simple Bayesian formulation for end-to-end unsupervised and semi-supervised learning with generative adversarial networks. Within this framework, we marginalize the posteriors over the weights of the generator and discriminator using stochastic gradient Hamiltonian Monte Carlo. We interpret data samples from the generator, showing exploration across several distinct modes in the generator weights. We also show data and iteration efficient learning of the true distribution. We also demonstrate state of the art semi-supervised learning performance on several benchmarks, including SVHN, MNIST, CIFAR-10, and CelebA. The simplicity of the proposed approach is one of its greatest strengths: inference is straightforward, interpretable, and stable. Indeed all of the experimental results were obtained without feature matching, normalization, or any ad-hoc techniques."
  - `video` <https://youtube.com/watch?v=24A8tWs6aug>
  - `code` <https://github.com/andrewgordonwilson/bayesgan/>

#### ["Bayesian Recurrent Neural Networks"](https://arxiv.org/abs/1704.02798) Fortunato, Blundell, Vinyals
  `bayesian model` `uncertainty estimation` `Bayes by Backprop`
  - `code` <https://github.com/DeNeutoy/bayesian-rnn>
  - `code` <https://github.com/mirceamironenco/BayesianRecurrentNN>
  - `paper` ["Weight Uncertainty in Neural Networks"](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#weight-uncertainty-in-neural-networks-blundell-cornebise-kavukcuoglu-wierstra) by Blundell et al. `summary`

#### ["Sequential Neural Models with Stochastic Layers"](http://arxiv.org/abs/1605.07571) Fraccaro, Sønderby, Paquet, Winther
  `bayesian model`
>	"stochastic neural networks:  
>	- allow to learn one-to-many type of mappings  
>	- can be used in structured prediction problems where modeling the internal structure of the output is important  
>	- benefit from stochasticity as regularizer which makes generalization performance potentially better in general"  
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Sequential-Neural-Models-with-Stochastic-Layers> (Fraccaro)
  - `video` <https://youtu.be/mrj_hyH974o?t=32m49s> (Sokolov) `in russian`
  - `code` <https://github.com/marcofraccaro/srnn>

#### ["DISCO Nets: DISsimilarity COefficient Networks"](http://arxiv.org/abs/1606.02556) Bouchacourt, Kumar, Nowozin
  `bayesian model`
  - `video` <https://youtube.com/watch?v=OogNSKRkoes>
  - `video` <https://youtube.com/watch?v=LUex45H4YXI> (Bouchacourt)
  - `video` <https://youtu.be/xFCuXE1Nb8w?t=34m21s> (Nowozin)

#### ["Composing Graphical Models with Neural Networks for Structured Representations and Fast Inference"](http://arxiv.org/abs/1603.06277) Johnson, Duvenaud, Wiltschko, Datta, Adams
  `bayesian model`
  - `video` <https://youtube.com/watch?v=btr1poCYIzw>
  - `video` <http://videolectures.net/deeplearning2017_johnson_graphical_models/> (Johnson)
  - `video` <https://youtube.com/watch?v=vnO3w8OgTE8> (Duvenaud)
  - `slides` <http://www.cs.toronto.edu/~duvenaud/courses/csc2541/slides/svae-slides.pdf>
  - `notes` <https://casmls.github.io/general/2016/12/11/SVAEandfLDS.html>
  - `code` <https://github.com/mattjj/svae>

#### ["The Variational Gaussian Process"](http://arxiv.org/abs/1511.06499) Tran, Ranganath, Blei
  `bayesian model`
  - `video` <http://videolectures.net/iclr2016_tran_variational_gaussian/> (Tran)
  - `code` <http://github.com/blei-lab/edward>

#### ["Deep Probabilistic Programming"](http://arxiv.org/abs/1701.03757) Tran, Hoffman, Saurous, Brevdo, Murphy, Blei
  `bayesian model`
  - `video` <https://youtube.com/watch?v=PvyVahNl8H8> (Tran) ([slides](http://dustintran.com/talks/Tran_Edward.pdf))
  - `post` <http://dustintran.com/blog/a-quick-update-edward-and-some-motivations/>
  - `code` <http://edwardlib.org/iclr2017>
  - `code` <http://edwardlib.org/zoo>

#### ["Deep Amortized Inference for Probabilistic Programs"](http://arxiv.org/abs/1610.05735) Ritchie, Horsfall, Goodman
  `bayesian model`

#### ["Inference Compilation and Universal Probabilistic Programming"](http://arxiv.org/abs/1610.09900) Le, Baydin, Wood
  `bayesian model`



---
### compute and memory architectures

[interesting older papers](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---architectures)

----
#### ["Hybrid Computing using a Neural Network with Dynamic External Memory"](http://rdcu.be/kXhV) Graves et al.
  `compute and memory` `Differentiable Neural Computer`
  - `post` <https://deepmind.com/blog/differentiable-neural-computers/>
  - `video` <https://youtube.com/watch?v=steioHoiEms> (Graves)
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255/> (9:09) (Graves)
  - `video` <https://youtube.com/watch?v=PQrlOjj8gAc> (Wayne)
  - `video` <https://youtu.be/otRoAQtc5Dk?t=59m56s> (Polykovskiy)
  - `video` <https://youtube.com/watch?v=r5XKzjTFCZQ> (Raval)
  - `code` <https://github.com/deepmind/dnc>
  - `code` <https://github.com/ixaxaar/pytorch-dnc>
  - `code` <https://github.com/Mostafa-Samir/DNC-tensorflow>

#### ["Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes"](http://arxiv.org/abs/1610.09027) Rae, Hunt, Harley, Danihelka, Senior, Wayne, Graves, Lillicrap
  `compute and memory` `Differentiable Neural Computer`

#### ["Using Fast Weights to Attend to the Recent Past"](http://arxiv.org/abs/1610.06258) Ba, Hinton, Mnih, Leibo, Ionescu
  `compute and memory`
>	"Until recently, research on artificial neural networks was largely restricted to systems with only two types of variable: neural activities that represent the current or recent input and weights that learn to capture regularities among inputs, outputs and payoffs. There is no good reason for this restriction. Synapses have dynamics at many different time-scales and this suggests that artificial neural networks might benefit from variables that change slower than activities but much faster than the standard weights. These "fast weights" can be used to store temporary memories of the recent past and they provide a neurally plausible way of implementing the type of attention to the past that has recently proved very helpful in sequence-to-sequence models. By using fast weights we can avoid the need to store copies of neural activity patterns."  
>	(Hinton) "It's a different approach to a Neural Turing Machine. It does not require any decisions about where to write stuff or where to read from. Anything that happened recently can automatically be retrieved associatively. Fast associative memory should allow neural network models of sequential human reasoning."  
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Using-Fast-Weights-to-Attend-to-the-Recent-Past> (Ba)
  - `video` <http://www.fields.utoronto.ca/talks/title-tba-337> (Hinton)
  - `video` <https://youtube.com/watch?v=mrj_hyH974o> (Novikov) `in russian`
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1610.06258>
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/fast-weight-to-attend.md>
  - `notes` <https://theneuralperspective.com/2016/12/04/implementation-of-using-fast-weights-to-attend-to-the-recent-past/>
  - `notes` <https://reddit.com/r/MachineLearning/comments/58qjiw/research161006258_using_fast_weights_to_attend_to/d92kctk/>
  - `code` <https://github.com/ajarai/fast-weights>
  - `code` <https://github.com/jxwufan/AssociativeRetrieval>

#### ["Associative Long Short-Term Memory"](http://arxiv.org/abs/1602.03032) Danihelka, Wayne, Uria, Kalchbrenner, Graves
  `compute and memory`
  - `video` <http://techtalks.tv/talks/associative-long-short-term-memory/62525/> (Danihelka)
  - `paper` ["Holographic Reduced Representations"](http://www.cogsci.ucsd.edu/~sereno/170/readings/06-Holographic.pdf) by Plate
  - `code` <https://github.com/mohammadpz/Associative_LSTM>

#### ["Learning Efficient Algorithms with Hierarchical Attentive Memory"](http://arxiv.org/abs/1602.03218) Andrychowicz, Kurach
  `compute and memory`
>	"It is based on a binary tree with leaves corresponding to memory cells. This allows HAM to perform memory access in O(log n) complexity, which is a significant improvement over the standard attention mechanism that requires O(n) operations, where n is the size of the memory."  
>	"We show that an LSTM network augmented with HAM can learn algorithms for problems like merging, sorting or binary searching from pure input-output examples."  
>	"We also show that HAM can be trained to act like classic data structures: a stack, a FIFO queue and a priority queue."  
>	"Our model may be seen as a special case of Gated Graph Neural Network."  
  - `code` <https://github.com/Smerity/tf-ham>

#### ["Neural Random-Access Machines"](http://arxiv.org/abs/1511.06392) Kurach, Andrychowicz, Sutskever
  `compute and memory`
>	"It can manipulate and dereference pointers to an external variable-size random-access memory."  
  - `post` <http://andrew.gibiansky.com/blog/machine-learning/nram-1/> + <http://andrew.gibiansky.com/blog/machine-learning/nram-2/>
  - `code` <https://github.com/gibiansky/experiments/tree/master/nram>

----
#### ["Neural Episodic Control"](https://arxiv.org/abs/1703.01988) Pritzel, Uria, Srinivasan, Puigdomenech, Vinyals, Hassabis, Wierstra, Blundell
  `episodic memory`
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
  - `video` <https://vimeo.com/238243674> (Pritzel)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=42m10s> (Mnih)
  - `notes` <http://rylanschaeffer.github.io/content/research/neural_episodic_control/main.html>
  - `post` <http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-4-episodic-and-interactive-memory/>
  - `code` <https://github.com/EndingCredits/Neural-Episodic-Control>
  - `code` <https://github.com/NervanaSystems/coach/blob/master/agents/nec_agent.py>

#### ["Model-Free Episodic Control"](http://arxiv.org/abs/1606.04460) Blundell, Uria, Pritzel, Li, Ruderman, Leibo, Rae, Wierstra, Hassabis
  `episodic memory`
>	"This might be achieved by a dual system (hippocampus vs neocortex) where information are stored in alternated way such that new nonstationary experience is rapidly encoded in the hippocampus (most flexible region of the brain with the highest amount of plasticity and neurogenesis); long term memory in the cortex is updated in a separate phase where what is updated (both in terms of samples and targets) can be controlled and does not put the system at risk of instabilities."  
  - <https://sites.google.com/site/episodiccontrol/> (demo)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=42m10s> (Mnih)
  - `post` <http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-4-episodic-and-interactive-memory/>
  - `code` <https://github.com/ShibiHe/Model-Free-Episodic-Control>
  - `code` <https://github.com/sudeepraja/Model-Free-Episodic-Control>

#### ["Learning to Remember Rare Events"](http://arxiv.org/abs/1703.03129) Kaiser, Nachum, Roy, Bengio
  `episodic memory`
>	"We present a large-scale life-long memory module for use in deep learning. The module exploits fast nearest-neighbor algorithms for efficiency and thus scales to large memory sizes. Except for the nearest-neighbor query, the module is fully differentiable and trained end-to-end with no extra supervision. It operates in a life-long manner, i.e., without the need to reset it during training. Our memory module can be easily added to any part of a supervised neural network. The enhanced network gains the ability to remember and do life-long one-shot learning. Our module remembers training examples shown many thousands of steps in the past and it can successfully generalize from them."  
  - `code` <https://github.com/tensorflow/models/tree/master/research/learning_to_remember_rare_events>

#### ["Neural Map: Structured Memory for Deep Reinforcement Learning"](https://arxiv.org/abs/1702.08360) Parisotto, Salakhutdinov
  `episodic memory`
>	"Memory was given a 2D structure in order to resemble a spatial map to address specific problems such as 2D or 3D navigation."  
>	"Size and computational cost doesn't grow with time horizon of environment."  
  - `slides` <http://www.cs.cmu.edu/~rsalakhu/NIPS2017_StructureMemoryForDeepRL.pdf>
  - `video` <https://yadi.sk/i/pMdw-_uI3Gke7Z> (Shvechikov) `in russian`

#### ["A Growing Long-term Episodic and Semantic Memory"](http://arxiv.org/abs/1610.06402) Pickett, Al-Rfou, Shao, Tar
  `episodic memory`
>	"We describe a lifelong learning system that leverages a fast, though non-differentiable, content-addressable memory which can be exploited to encode both a long history of sequential episodic knowledge and semantic knowledge over many episodes for an unbounded number of domains."  

----
#### ["Variational Continual Learning"](https://arxiv.org/abs/1710.10628) Nguyen, Li, Bui, Turner
  `catastrophic forgetting` `continual learning`
>	"The framework can successfully train both deep discriminative models and deep generative models in complex continual learning settings where existing tasks evolve over time and entirely new tasks emerge. Experimental results show that variational continual learning outperforms state-of-the-art continual learning methods on a variety of tasks, avoiding catastrophic forgetting in a fully automatic way."  

#### ["Gradient Episodic Memory for Continuum Learning"](https://arxiv.org/abs/1706.08840) Lopez-Paz, Ranzato
  `catastrophic forgetting` `continual learning`
  - `notes` <http://rayraycano.github.io/data%20science/tech/2017/07/31/A-Paper-a-Day-GEM.html>

#### ["Improved Multitask Learning Through Synaptic Intelligence"](https://arxiv.org/abs/1703.04200) Zenke, Poole, Ganguli
  `catastrophic forgetting` `continual learning`
>	"The regularization penalty is similar to EWC. However, our approach computes the per-synapse consolidation strength in an online fashion, whereas for EWC synaptic importance is computed offline after training on a designated task."  
  - `video` <https://vimeo.com/238242232> (Zenke)
  - `code` <https://github.com/spiglerg/TF_ContinualLearningViaSynapticIntelligence>

#### ["PathNet: Evolution Channels Gradient Descent in Super Neural Networks"](http://arxiv.org/abs/1701.08734) Fernando, Banarse, Blundell, Zwols, Ha, Rusu, Pritzel, Wierstra
  `catastrophic forgetting` `continual learning`
  - `video` <https://youtube.com/watch?v=Wkz4bG_JlcU>
  - `post` <https://medium.com/intuitionmachine/pathnet-a-modular-deep-learning-architecture-for-agi-5302fcf53273>
  - `code` <https://github.com/jaesik817/pathnet>

#### ["Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"](https://arxiv.org/abs/1701.06538) Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean
  `catastrophic forgetting` `continual learning`
>	"The MoE with experts shows higher accuracy (or lower perplexity) than the state of the art using only 16% of the training time."  
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/3718d181a0fed5ed806582822ed0dbde530122bf/notes/mixture-experts.md>

#### ["Overcoming Catastrophic Forgetting in Neural Networks"](http://arxiv.org/abs/1612.00796) Kirkpatrick et al.
  `catastrophic forgetting` `continual learning`
>	"The Mixture of Experts Layer is trained using back-propagation. The Gating Network outputs an (artificially made) sparse vector that acts as a chooser of which experts to consult. More than one expert can be consulted at once (although the paper doesn’t give any precision on the optimal number of experts). The Gating Network also decides on output weights for each expert."  
>
>	Huszar:  
>	"on-line sequential (diagonalized) Laplace approximation of Bayesian learning"  
>	"EWC makes sense for any neural network (indeed, any parametric model, really), virtually any task. Doesn't have to be DQN and in fact the paper itself shows examples with way simpler tasks."  
>	"The quadratic penalty/penalties prevent the network from forgetting what it has learnt from previous data - you can think of the quadratic penalty as a summary of the information from the data it has seen so far."  
>	"You can apply it at the level of learning tasks sequentially, or you can even apply it to on-line learning in a single task (in case you can't loop over the same minibatches several time like you do in SGD)."  
  - `paper` <http://www.pnas.org/content/early/2017/03/13/1611835114.abstract>
  - `post` <https://deepmind.com/blog/enabling-continual-learning-in-neural-networks/>
  - `video` <https://vimeo.com/238221551#t=13m9s> (Hadsell)
  - `post` <http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/main.html>
  - `post` <http://inference.vc/comment-on-overcoming-catastrophic-forgetting-in-nns-are-multiple-penalties-needed-2/>
  - `notes` <https://theneuralperspective.com/2017/04/01/overcoming-catastrophic-forgetting-in-neural-networks/>
  - `code` <https://github.com/ariseff/overcoming-catastrophic>

----
#### ["Hierarchical Multiscale Recurrent Neural Networks"](http://arxiv.org/abs/1609.01704) Chung, Ahn, Bengio
  `compute and memory resources`
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/hm-rnn.md>
  - `notes` <https://medium.com/@jimfleming/notes-on-hierarchical-multiscale-recurrent-neural-networks-7362532f3b64>

#### ["Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences"](http://arxiv.org/abs/1610.09513) Neil, Pfeiffer, Liu
  `compute and memory resources`
>	"If you take an LSTM and add a “time gate” that controls at what frequency to be open to new input and how long to be open each time, you can have different neurons that learn to look at a sequence with different frequencies, create a “wormhole” for gradients, save compute, and do better on long sequences and when you need to process inputs from multiple sensors that are sampled at different rates."  
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Phased-LSTM-Accelerating-Recurrent-Network-Training-for-Long-or-Event-based-Sequences> (Neil)
  - `code` <https://tensorflow.org/api_docs/python/tf/contrib/rnn/PhasedLSTMCell>
  - `code` <https://github.com/dannyneil/public_plstm>

#### ["Memory-Efficient Backpropagation Through Time"](http://arxiv.org/abs/1606.03401) Gruslys, Munos, Danihelka, Lanctot, Graves
  `compute and memory resources`

#### ["Adaptive Computation Time for Recurrent Neural Networks"](http://arxiv.org/abs/1603.08983) Graves
  `compute and memory resources`
  - `video` <https://youtu.be/tA8nRlBEVr0?t=1m26s> (Graves)
  - `video` <https://youtu.be/nqiUFc52g78?t=58m45s> (Graves)
  - `video` <https://vimeo.com/240428387#t=1h28m28s> (Vinyals)
  - `post` <http://distill.pub/2016/augmented-rnns/>
  - `post` <https://www.evernote.com/shard/s189/sh/fd165646-b630-48b7-844c-86ad2f07fcda/c9ab960af967ef847097f21d94b0bff7>
  - `code` <https://github.com/DeNeutoy/act-tensorflow>

----
#### ["Dynamic Routing Between Capsules"](https://arxiv.org/abs/1710.09829) Sabour, Frosst, Hinton
  `information routing` `CapsNet`
>	"A capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or object part. We use the length of the activity vector to represent the probability that the entity exists and its orientation to represent the instantiation paramters. Active capsules at one level make predictions, via transformation matrices, for the instantiation parameters of higher-level capsules. When multiple predictions agree, a higher level capsule becomes active. We show that a discrimininatively trained, multi-layer capsule system achieves state-of-the-art performance on MNIST and is considerably better than a convolutional net at recognizing highly overlapping digits. To achieve these results we use an iterative routing-by-agreement mechanism: A lower-level capsule prefers to send its output to higher level capsules whose activity vectors have a big scalar product with the prediction coming from the lower-level capsule."  
>	"Dynamic routing can be viewed as a parallel attention mechanism that allows each capsule at one level to attend to some active capsules at the level below and to ignore others. This should allow the model to recognize multiple objects in the image even if objects overlap. The routing-by-agreement should make it possible to use a prior about shape of objects to help segmentation and it should obviate the need to make higher-level segmentation decisions in the domain of pixels."  
>	"For thirty years, the state-of-the-art in speech recognition used hidden Markov models with Gaussian mixtures as output distributions. These models were easy to learn on small computers, but they had a representational limitation that was ultimately fatal: The one-of-n representations they use are exponentially inefficient compared with, say, a recurrent neural network that uses distributed representations. To double the amount of information that an HMM can remember about the string it has generated so far, we need to square the number of hidden nodes. For a recurrent net we only need to double the number of hidden neurons.  
>	Now that convolutional neural networks have become the dominant approach to object recognition, it makes sense to ask whether there are any exponential inefficiencies that may lead to their demise. A good candidate is the difficulty that convolutional nets have in generalizing to novel viewpoints. The ability to deal with translation is built in, but for the other dimensions of an affine transformation we have to chose between replicating feature detectors on a grid that grows exponentially with the number of dimensions, or increasing the size of the labelled training set in a similarly exponential way. Capsules avoid these exponential inefficiencies by converting pixel intensities into vectors of instantiation parameters of recognized fragments and then applying transformation matrices to the fragments to predict the instantiation parameters of larger fragments. Transformation matrices that learn to encode the intrinsic spatial relationship between a part and a whole constitute viewpoint invariant knowledge that automatically generalizes to novel viewpoints.  
>	Capsules make a very strong representational assumption: At each location in the image, there is at most one instance of the type of entity that a capsule represents. This assumption eliminates the binding problem and allows a capsule to use a distributed representation (its activity vector) to encode the instantiation parameters of the entity of that type at a given location. This distributed representation is exponentially more efficient than encoding the instantiation parameters by activating a point on a high-dimensional grid and with the right distributed representation, capsules can then take full advantage of the fact that spatial relationships can be modelled by matrix multiplies.  
>	Capsules use neural activities that vary as viewpoint varies rather than trying to eliminate viewpoint variation from the activities. This gives them an advantage over "normalization" methods like spatial transformer networks: They can deal with multiple different affine transformations of different objects or object parts at the same time.  
>	Research on capsules is now at a similar stage to research on recurrent neural networks for speech recognition at the beginning of this century. There are fundamental representational reasons for believing that it is a better approach but it probably requires a lot more small insights before it can out-perform a highly developed technology. The fact that a simple capsules system already gives unparalleled performance at segmenting overlapping digits is an early indication that capsules are a direction worth exploring."  
>
>	"The core idea of capsules is that low level features predict the existence and pose of higher level features; collisions are non-coincidental. E.g. paw HERE predicts tiger THERE, nose HERE predicts tiger THERE, stripe HERE predicts tiger THERE - paw and nose and stripe predict tiger in the SAME PLACE! That's unlikely to be a coincidence, there's probably a tiger.  
>	The core idea of pooling is that high level features are correlated with the existence of low-level features across sub-regions. E.g. I see a paw and a nose and a stripe - I guess we've got some tigers up in this. Even if the paw predicts a Tiger Rampant and the nose predicts a Tiger Face-On and the stripe predicts a Tiger Sideways. Hence CNN's disastrous vulnerability to adversarial stimuli."  
>
>	"A fully connected layer would route the features based on their agreement with a learned weight vector. This defeats the intent of dynamic routing, the whole purpose of which is to route activations to capsules where they agree with other activations. It does the routing based on a fast iterative process in the forward pass, not a slow learning process like gradient descent."  
  - `video` ["What is wrong with convolutional neural nets?"](https://youtube.com/watch?v=Mqt8fs6ZbHk) (Hinton)
  - `video` ["What's wrong with convolutional nets?"](http://techtv.mit.edu/collections/bcs/videos/30698-what-s-wrong-with-convolutional-nets) (Hinton) ([transcription](https://github.com/WalnutiQ/walnut/issues/157))
  - `video` ["Does the Brain do Inverse Graphics?"](https://youtube.com/watch?v=TFIMqt0yT2I) (Hinton)
  - `video` <https://youtube.com/watch?v=pPN8d0E3900> (Geron)
  - `video` <https://youtube.com/watch?v=EATWLTyLfmc> (Canziani)
  - `video` <https://youtube.com/watch?v=hYt3FcJUf6w> (Uziela)
  - `video` <https://youtube.com/watch?v=VKoLGnq15RM> (Raval)
  - `video` <https://youtube.com/watch?v=UZ9BgrofhKk> (Kozlov) `in russian`
  - `post` <https://medium.com/@pechyonkin/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b>
  - `post` <https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc>
  - `post` <https://kndrck.co/posts/capsule_networks_explained/>
  - `post` <https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/>
  - `post` <https://medium.com/@mike_ross/a-visual-representation-of-capsule-network-computations-83767d79e737>
  - `notes` <https://blog.acolyer.org/2017/11/13/dynamic-routing-between-capsules/>
  - `code` <https://github.com/llSourcell/capsule_networks>
  - `code` <https://github.com/InnerPeace-Wu/CapsNet-tensorflow>
  - `code` <https://github.com/naturomics/CapsNet-Tensorflow>
  - `code` <https://github.com/gram-ai/capsule-networks>
  - `code` <https://github.com/adambielski/CapsNet-pytorch>
  - `code` <https://github.com/nishnik/CapsNet-PyTorch>
  - `code` <https://github.com/XifengGuo/CapsNet-Keras>
  - `paper` ["Transforming Auto-encoders"](http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf) by Hinton, Krizhevsky, Wang
  - `paper` ["Optimizing Neural Networks that Generate Images"](http://www.cs.toronto.edu/~tijmen/tijmen_thesis.pdf) by Tieleman ([code](https://github.com/mrkulk/Unsupervised-Capsule-Network))

#### ["Matrix Capsules with EM Routing"](https://openreview.net/forum?id=HJWLfGWRb)
  `information routing` `CapsNet`
  - `video` <https://youtube.com/watch?v=hYt3FcJUf6w> (Uziela)
  - `notes` <https://blog.acolyer.org/2017/11/14/matrix-capsules-with-em-routing/>
  - `post` <https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-Capsule-Network/>

#### ["Decoupled Neural Interfaces using Synthetic Gradients"](http://arxiv.org/abs/1608.05343) Jaderberg, Czarnecki, Osindero, Vinyals, Graves, Silver, Kavukcuoglu
  `information routing`
>	"We incorporate a learnt model of error gradients, which means we can update networks without full backpropagation. We show how this can be applied to feed-forward networks which allows every layer to be trained asynchronously, to RNNs which extends the time over which models can remember, and to multi-network systems to allow communication."  
  - `post` <https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/>
  - `video` <https://youtu.be/tA8nRlBEVr0?t=14m40s> + <https://youtube.com/watch?v=-u32TOPGIbQ> (Graves)
  - `video` <https://youtube.com/watch?v=toZprSCCmNI> (Gupta)
  - `video` <https://youtube.com/watch?v=qirjknNY1zo> (Raval)
  - `post` <https://iamtrask.github.io/2017/03/21/synthetic-gradients/>
  - `notes` <http://cnichkawde.github.io/SyntheticGradients.html>
  - `code` <https://github.com/koz4k/dni-pytorch>

#### ["Understanding Synthetic Gradients and Decoupled Neural Interfaces"](http://arxiv.org/abs/1703.00522) Czarnecki, Swirszcz, Jaderberg, Osindero, Vinyals, Kavukcuoglu
  `information routing`
  - `video` <https://vimeo.com/238275152> (Swirszcz)

#### ["Learning to Communicate with Deep Multi-Agent Reinforcement Learning"](http://arxiv.org/abs/1605.06676) Foerster, Assael, de Freitas, Whiteson
  `information routing`
  - `video` <https://youtu.be/SAcHyzMdbXc?t=19m> (de Freitas)
  - `video` <https://youtube.com/watch?v=xL-GKD49FXs> (Foerster)
  - `video` <http://videolectures.net/deeplearning2016_foerster_learning_communicate/> (Foerster)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.07133>
  - `code` <https://github.com/iassael/learning-to-communicate>

#### ["Learning Multiagent Communication with Backpropagation"](http://arxiv.org/abs/1605.07736) Sukhbaatar, Szlam, Fergus
  `information routing`
  - `video` <https://youtu.be/SAcHyzMdbXc?t=19m> (de Freitas)
  - `video` <https://youtu.be/_iVVXWkoEAs?t=30m6s> (Fergus)
  - `code` <https://github.com/facebookresearch/CommNet>
  - `code` <https://github.com/rickyhan/CommNet>

----
#### ["SMASH: One-Shot Model Architecture Search through HyperNetworks"](https://arxiv.org/abs/1708.05344) Brock, Lim, Ritchie, Weston
  `architecture search`
>	"The architecture: at each training step, generate the schematics of a random NN architecture; feed the skeleton into the hypernetwork, which will directly spit out numbers for each neuron (as a convolutional hypernetwork it can handle big and small NNs the same way); with the fleshed out NN, train 1 minibatch on the image classification task as usual, and update its parameters; use that update as the error for the hypernetwork to train it to spit out weights for that skeleton which are slightly closer to what it was after 1 minibatch. After training the hypernetwork many times on many random NN architectures, its generated weights will be close to what training each random NN architecture from scratch would have been. Now you can simply generate lots of random NN architectures, fill them in, run them on a small validation set, and see their final performance without ever actually training them fully (which would be like 10,000x more expensive). So this runs on 1 GPU in a day or two versus papers like Zoph which used 800 GPUs for a few weeks. It’s amazing this works, and like synthetic gradients it troubles me a little because it implies that even complex highly sophisticated NNs are in some sense simple & predictable as their weights/error-gradients can be predicted by other NNs which are as small as linear layers or don’t even see the data, and thus are incredibly wasteful in both training & parameter size, implying a large hardware overhang."  
  - `video` <https://youtube.com/watch?v=79tmPL9AL48>
  - `code` <https://github.com/ajbrock/SMASH>

#### ["Learning Transferable Architectures for Scalable Image Recognition"](https://arxiv.org/abs/1707.07012) Zoph, Vasudevan, Shlens, Le
  `architecture search`
  - `post` <https://research.googleblog.com/2017/11/automl-for-large-scale-image.html>

#### ["Neural Architecture Search with Reinforcement Learning"](http://arxiv.org/abs/1611.01578) Zoph, Le
  `architecture search`
  - `post` <https://research.googleblog.com/2017/05/using-machine-learning-to-explore.html>
  - `video` <https://youtube.com/watch?v=XDtFXBYpl1w> (Le)
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255/> (1:08:31) (Zoph)
  - `notes` <https://blog.acolyer.org/2017/05/10/neural-architecture-search-with-reinforcement-learning/>



---
### meta-learning

----
#### ["Some Considerations on Learning to Explore via Meta-Reinforcement Learning"](https://openreview.net/forum?id=Skk3Jm96W)
>	"We introduce two new algorithms: E-MAML and E-RL2, which are derived by reformulating the underlying meta-learning objective to account for the impact of initial sampling on future (post-meta-updated) returns."  
>	"Meta RL agent must not learn how to master the environments it is given, but rather it must learn how to learn so that it can quickly train at test time."  

----
#### ["Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments"](https://arxiv.org/abs/1710.03641) Al-Shedivat, Bansal, Burda, Sutskever, Mordatch, Abbeel
  `MAML`
>	"extending Model-Agnostic Meta-Learning to the case of dynamically changing tasks"
  - <https://sites.google.com/view/adaptation-via-metalearning> (demo)

#### ["Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm"](https://arxiv.org/abs/1710.11622) Finn, Levine
  `MAML`

#### ["Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"](https://arxiv.org/abs/1703.03400) Finn, Abbeel, Levine
  `MAML`
>	"Tasks are sampled and a policy gradient update is computed for each task with respect to a fixed initial set of parameters. Subsequently, a meta update is performed where a gradient step is taken that moves the initial parameter in a direction that would have maximally benefited the average return over all of the sub-updates."  
>	"Unlike prior methods, the MAML learner’s weights are updated using the gradient, rather than a learned update rule. Our method does not introduce any additional parameters into the learning process and does not require a particular learner model architecture."  
>	"MAML finds a shared parameter θ such that for a given task, one gradient step on θ using the training set will yield a model with good predictions on the test set. Then, a meta-gradient update is performed from the test error through the one gradient step in the training set, to update θ."  
  - <https://sites.google.com/view/maml> (demo)
  - `video` <https://youtu.be/Ko8IBbYjdq8?t=18m51s> (Finn)
  - `video` <https://youtu.be/lYU5nq0dAQQ?t=44m57s> (Levine)
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=1h1m22s> + <https://youtube.com/watch?v=ID150Tl-MMw&t=1h9m10s> (Abbeel)
  - `post` <http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/> (Finn)
  - `code` <https://github.com/cbfinn/maml>

#### ["Optimization as a Model for Few-Shot Learning"](https://openreview.net/forum?id=rJY0-Kcll) Ravi, Larochelle
>	"Using LSTM meta-learner in a few-shot classification setting, where the traditional learner was a convolutional-network-based classifier. In this setting, the whole meta-learning algorithm is decomposed into two parts: the traditional learner’s initial parameters are trained to be suitable for fast gradient-based adaptation; the LSTM meta-learner is trained to be an optimization algorithm adapted for meta-learning tasks."  
>	"Encoding network reads the training set and generate the parameters of a model, which is trained to perform well on the testing set."  
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255/> (1:26:48) (Ravi)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/> (1:08:08) (de Freitas)
  - `code` <https://github.com/twitter/meta-learning-lstm>

----
#### ["A Simple Neural Attentive Meta-Learner"](https://openreview.net/forum?id=B1DmUzWAW)

#### ["Meta-Learning with Temporal Convolutions"](https://arxiv.org/abs/1707.03141) Mishra, Rohaninejad, Chen, Abbeel
>	"Like RL^2 but with LSTM network replaced by dilated temporal convolutional network and attention."  
>	"Most recent approaches to meta-learning are extensively hand-designed, either using architectures that are specialized to a particular application, or hard-coding algorithmic components that tell the meta-learner how to solve the task. We propose a class of simple and generic meta-learner architectures, based on temporal convolutions, that is domain-agnostic and has no particular strategy or algorithm encoded into it."  
>	"TCML architectures are nothing more than a deep stack of convolutional layers, making them simple, generic, and versatile, and the causal structure allows them to process sequential data in a sophisticated manner. RNNs also have these properties, but traditional architectures can only propagate information through time via their hidden state, and so there are fewer paths for information to flow from past to present. TCMLs do a better job of preserving the temporal structure of the input sequence; the convolutional structure offers more direct, high-bandwidth access to past information, allowing them to perform more sophisticated computation on a fixed temporal segment."  
>	"TCML is closest in spirit to [Santoro et al.](http://arxiv.org/abs/1605.06065); however, our experiments indicate that TCML outperforms such traditional RNN architectures. We can view the TCML architecture as a flavor of RNN that can remember information through the activations of the network rather than through an explicit memory module. Because of its convolutional structure, the TCML better preserves the temporal structure of the inputs it receives, at the expense of only being able to remember information for a fixed amount of time. However, by exponentially increasing the dilation factors of the higher convolutional layers, TCML architectures can tractably store information for long periods of time."  
  - `video` <https://youtu.be/TERCdog1ddE?t=49m39s> (Abbeel)

#### ["RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning"](http://arxiv.org/abs/1611.02779) Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel
>	"MDPs encountered in real world = tiny subset of all MDPs that could be defined"  
>	"How to acquire a good prior for real-world MDPs?"  
>	"How to design algorithms that make use of such prior information?"  
>	"Key idea: learn a fast RL algorithm that make use of such prior information"  
>
>	"RNN is made to ingest multiple rollouts from many different MDPs and then perform a policy gradient update through the entire temporal span of the RNN. The hope is that the RNN will learn a faster RL algorithm in its memory weights."  
>	"Suppose L represents an RNN. Let Envk(a) be a function that takes an action, uses it to interact with the MDP representing task k, and returns the next observation o, reward r, and a termination flag d. Then we have:  
>	xt = [ot−1, at−1, rt−1, dt−1]  
>	L(ht, xt) = [at, ht+1]  
>	Envk(at) = [ot, rt, dt]  
>	To train this RNN, we sample N MDPs from M and obtain k rollouts for each MDP by running the MDP through the RNN as above. We then compute a policy gradient update to move the RNN parameters in a direction which maximizes the returns over the k trials performed for each MDP."  
  - `video` <http://www.fields.utoronto.ca/video-archive/2017/02/2267-16530> (19:00) (Abbeel)
  - `video` <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)
  - `video` <https://youtu.be/BskhUBPRrqE?t=6m28s> (Sutskever)
  - `video` <https://youtu.be/19eNQ1CLt5A?t=7m52s> (Sutskever)
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/RL2-Fast_Reinforcement_Learning_via_Slow_Reinforcement_Learning.md>

#### ["Learning to Reinforcement Learn"](http://arxiv.org/abs/1611.05763) Wang et al.
>	"outer episodes (sample a new bandit problem / MDP) and inner episodes (of sampled MDP)"  
>	"use RNN policy with no state reset between inner episodes for outer POMDP"  
  - `video` <https://youtu.be/Y85Zn50Eczs?t=20m18s> (Botvinick)
  - `video` <https://youtube.com/watch?v=SfCa1HQMkuw&t=1h16m56s> (Schulman)
  - `post` <https://hackernoon.com/learning-policies-for-learning-policies-meta-reinforcement-learning-rl²-in-tensorflow-b15b592a2ddf> (Juliani)
  - `code` <https://github.com/awjuliani/Meta-RL>

----
#### ["Learned Optimizers that Scale and Generalize"](http://arxiv.org/abs/1703.04813) Wichrowska, Maheswaranathan, Hoffman, Colmenarejo, Denil, de Freitas, Sohl-Dickstein
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/> (42:40) (de Freitas)
  - `code` <https://github.com/tensorflow/models/tree/master/research/learned_optimizer>

#### ["Learning to Learn without Gradient Descent by Gradient Descent"](https://arxiv.org/abs/1611.03824) Chen, Hoffman, Colmenarejo, Denil, Lillicrap, Botvinick, de Freitas
>	"Differentiable neural computers as alternatives to parallel Bayesian optimization for hyperparameter tuning of other networks."  
>	"Proposes RNN optimizers that match performance of Bayesian optimization methods (e.g. Spearmint, SMAC, TPE) but are massively faster."  
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/> (52:35) (de Freitas)

#### ["Learning to Learn by Gradient Descent by Gradient Descent"](http://arxiv.org/abs/1606.04474) Andrychowicz, Denil, Gomez, Hoffman, Pfau, Schaul, Shillingford, de Freitas
>	"Take some computation where you usually wouldn’t keep around intermediate states, such as a planning computation (say value iteration, where you only keep your most recent estimate of the value function) or stochastic gradient descent (where you only keep around your current best estimate of the parameters). Now keep around those intermediate states as well, perhaps reifying the unrolled computation in a neural net, and take gradients to optimize the entire computation with respect to some loss function. Instances: Value Iteration Networks, Learning to learn by gradient descent by gradient descent."  
  - `video` <https://youtu.be/SAcHyzMdbXc?t=10m24s> (de Freitas)
  - `video` <https://youtu.be/x1kf4Zojtb0?t=1h4m53s> (de Freitas)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/> (27:49) (de Freitas)
  - `notes` <https://theneuralperspective.com/2017/01/04/learning-to-learn-by-gradient-descent-by-gradient-descent/>
  - `notes` <https://blog.acolyer.org/2017/01/04/learning-to-learn-by-gradient-descent-by-gradient-descent/>
  - `post` <https://hackernoon.com/learning-to-learn-by-gradient-descent-by-gradient-descent-4da2273d64f2>
  - `code` <https://github.com/deepmind/learning-to-learn>
  - `code` <https://github.com/ikostrikov/pytorch-meta-optimizer>



---
### few-shot learning

----
#### ["Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"](https://arxiv.org/abs/1703.03400) Finn, Abbeel, Levine
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#model-agnostic-meta-learning-for-fast-adaptation-of-deep-networks-finn-abbeel-levine>

#### ["Optimization as a Model for Few-Shot Learning"](https://openreview.net/forum?id=rJY0-Kcll) Ravi, Larochelle
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#optimization-as-a-model-for-few-shot-learning-ravi-larochelle>

----
#### ["Few-shot Autoregressive Density Estimation: Towards Learning to Learn Distributions"](https://arxiv.org/abs/1710.10304) Reed, Chen, Paine, Oord, Eslami, Rezende, Vinyals, de Freitas

----
#### ["Learning to Remember Rare Events"](http://arxiv.org/abs/1703.03129) Kaiser, Nachum, Roy, Bengio
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#learning-to-remember-rare-events-kaiser-nachum-roy-bengio>

#### ["Prototypical Networks for Few-shot Learning"](https://arxiv.org/abs/1703.05175) Snell, Swersky, Zemel
>	"Extension to Matching Networks which uses euclidean distance instead of cosine and builds a prototype representation of each class for the few-shot learning scenario."  

#### ["Matching Networks for One Shot Learning"](http://arxiv.org/abs/1606.04080) Vinyals, Blundell, Lillicrap, Kavukcuoglu, Wierstra
>	"Given just a few, or even a single, examples of an unseen class, it is possible to attain high classification accuracy on ImageNet using Matching Networks. Matching Networks are trained in the same way as they are tested: by presenting a series of instantaneous one shot learning training tasks, where each instance of the training set is fed into the network in parallel. Matching Networks are then trained to classify correctly over many different input training sets. The effect is to train a network that can classify on a novel data set without the need for a single step of gradient descent."  
>	"End-to-end trainable K-nearest neighbors which accepts support sets of images as input and maps them to desired labels. Attention LSTM takes into account all samples of subset when computing the pair-wise cosine distance between samples."  
  - `poster` <https://pbs.twimg.com/media/Cy7Eyh5WgAAZIw2.jpg:large>
  - `notes` <https://theneuralperspective.com/2017/01/03/matching-networks-for-one-shot-learning/>
  - `notes` <https://blog.acolyer.org/2017/01/03/matching-networks-for-one-shot-learning/>

----
#### ["Meta-Learning with Temporal Convolutions"](https://arxiv.org/abs/1707.03141) Mishra, Rohaninejad, Chen, Abbeel
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#meta-learning-with-temporal-convolutions-mishra-rohaninejad-chen-abbeel>

#### ["One-shot Learning with Memory-Augmented Neural Networks"](http://arxiv.org/abs/1605.06065) Santoro, Bartunov, Botvinick, Wierstra, Lillicrap
  - `video` <http://techtalks.tv/talks/meta-learning-with-memory-augmented-neural-networks/62523/> + <https://vk.com/wall-44016343_8782> (Santoro)
  - `video` <https://youtube.com/watch?v=qos2CcviAuY> (Bartunov) `in russian`
  - `notes` <http://rylanschaeffer.github.io/content/research/one_shot_learning_with_memory_augmented_nn/main.html>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.06065>
  - `code` <https://github.com/tristandeleu/ntm-one-shot>

----
#### ["Variational Memory Addressing in Generative Models"](https://arxiv.org/abs/1709.07116) Bornschein, Mnih, Zoran, Rezende
>	"Attention based memory can be used to augment neural networks to support few-shot learning, rapid adaptability and more generally to support non-parametric extensions. Instead of using the popular differentiable soft-attention mechanism, we propose the use of stochastic hard-attention to retrieve memory content in generative models. This allows us to apply variational inference to memory addressing, which enables us to get significantly more precise memory lookups using target information, especially in models with large memory buffers and with many confounding entries in the memory."  
>	"Aiming to augment generative models with external memory, we interpret the output of a memory module with stochastic addressing as a conditional mixture distribution, where a read operation corresponds to sampling a discrete memory address and retrieving the corresponding content from memory. This perspective allows us to apply variational inference to memory addressing, which enables effective training of the memory module by using the target information to guide memory lookups. Stochastic addressing is particularly well-suited for generative models as it naturally encourages multimodality which is a prominent aspect of most high-dimensional datasets. Treating the chosen address as a latent variable also allows us to quantify the amount of information gained with a memory lookup and measure the contribution of the memory module to the generative process."  
>	"To illustrate the advantages of this approach we incorporate it into a variational autoencoder and apply the resulting model to the task of generative few-shot learning. The intuition behind this architecture is that the memory module can pick a relevant template from memory and the continuous part of the model can concentrate on modeling remaining variations. We demonstrate empirically that our model is able to identify and access the relevant memory contents even with hundreds of unseen Omniglot characters in memory."  

#### ["Fast Adaptation in Generative Models with Generative Matching Networks"](http://arxiv.org/abs/1612.02192) Bartunov, Vetrov
  - `video` <https://youtu.be/XpIDCzwNe78> (Bartunov) ([slides](https://bayesgroup.github.io/bmml_sem/2016/bartunov-oneshot.pdf))
  - `code` <http://github.com/sbos/gmn>

#### ["Towards a Neural Statistician"](http://arxiv.org/abs/1606.02185) Edwards, Storkey
  - `video` <http://techtalks.tv/talks/neural-statistician/63048/> (Edwards)
  - `video` <https://youtu.be/XpIDCzwNe78?t=51m53s> (Bartunov)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1606.02185>

#### ["One-Shot Generalization in Deep Generative Models"](http://arxiv.org/abs/1603.05106) Rezende, Mohamed, Danihelka, Gregor, Wierstra
  - `video` <http://youtube.com/watch?v=TpmoQ_j3Jv4> (demo)
  - `video` <http://techtalks.tv/talks/one-shot-generalization-in-deep-generative-models/62365/> (Rezende)
  - `video` <https://youtu.be/XpIDCzwNe78?t=43m> (Bartunov)
  - `notes` <https://casmls.github.io/general/2017/02/08/oneshot.html>

----
#### ["Active One-shot Learning"](https://cs.stanford.edu/~woodward/papers/active_one_shot_learning_2016.pdf) Woodward, Finn
  - `video` <https://youtube.com/watch?v=CzQSQ_0Z-QU> (Woodward)



---
### unsupervised learning

----
#### ["Independently Controllable Features"](https://arxiv.org/abs/1708.01289) Thomas, Pondard, Bengio, Sarfati, Beaudoin, Meurs, Pineau, Precup, Bengio
  `causal learning`
>	"It has been postulated that a good representation is one that disentangles the underlying explanatory factors of variation. However, it remains an open question what kind of training framework could potentially achieve that. Whereas most previous work focuses on the static setting (e.g. with images), we postulate that some of the causal factors could be discovered if the learner is allowed to interact with its environment. The agent can experiment with different actions and observe their effects. We hypothesize that some of these factors correspond to aspects of the environment which are independently controllable, i.e., that there exists a policy and a learnable feature for each such aspect of the environment, such that this policy can yield changes in that feature with minimal changes to other features that explain the statistical variations in the observed data."  
>	"In interactive environments, the temporal dependency between successive observations creates a new opportunity to notice causal structure in data which may not be apparent using only observational studies. In reinforcement learning, several approaches explore mechanisms that push the internal representations of learned models to be “good” in the sense that they provide better control, and control is a particularly important causal relationship between an agent and elements of its environment."  
>	"We propose and explore a more direct mechanism for representation learning, which explicitly links an agent’s control over its environment with its internal feature representations. Specifically, we hypothesize that some of the factors explaining variations in the data correspond to aspects of the world that can be controlled by the agent. For example, an object that could be pushed around or picked up independently of others is an independently controllable aspect of the environment. Our approach therefore aims to jointly discover a set of features (functions of the environment state) and policies (which change the state) such that each policy controls the associated feature while leaving the other features unchanged as much as possible."  
>	"Assume that there are factors of variation underlying the observations coming from an interactive environment that are independently controllable. That is, a controllable factor of variation is one for which there exists a policy which will modify that factor only, and not the others. For example, the object associated with a set of pixels could be acted on independently from other objects, which would explain variations in its pose and scale when we move it around while leaving the others generally unchanged. The object position in this case is a factor of variation. What poses a challenge for discovering and mapping such factors into computed features is the fact that the factors are not explicitly observed. Our goal is for the agent to autonomously discover such factors – which we call independently controllable features – along with policies that control them. While these may seem like strong assumptions about the nature of the environment, we argue that these assumptions are similar to regularizers, and are meant to make a difficult learning problem (that of learning good representations which disentangle underlying factors) better constrained."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/BengioTPPB17>

#### ["Discovering Causal Signals in Images"](https://arxiv.org/abs/1605.08179) Lopez-Paz, Nishihara, Chintala, Scholkopf, Bottou
  `causal learning`
>	"First, we take a learning approach to observational causal inference, and build a classifier that achieves state-of-the-art performance on finding the causal direction between pairs of random variables, when given samples from their joint distribution. Second, we use our causal direction finder to effectively distinguish between features of objects and features of their contexts in collections of static images. Our experiments demonstrate the existence of (1) a relation between the direction of causality and the difference between objects and their contexts, and (2) observable causal signals in collections of static images."  
>	"Causal features are those that cause the presence of the object of interest in the image (that is, those features that cause the object’s class label), while anticausal features are those caused by the presence of the object in the image (that is, those features caused by the class label)."  
>	"Paper aims to verify experimentally that the higher-order statistics of image datasets can inform about causal relations. Authors conjecture that object features and anticausal features are closely related and vice-versa context features and causal features are not necessarily related. Context features give the background while object features are what it would be usually inside bounding boxes in an image dataset."  
>	"Better algorithms for causal direction should, in principle, help learning features that generalize better when the data distribution changes. Causality should help with building more robust features by awareness of the generating process of the data."  
  - `video` <https://youtube.com/watch?v=DfJeaa--xO0> (Bottou)
  - `post` <http://giorgiopatrini.org/posts/2017/09/06/in-search-of-the-missing-signals/>

----
#### ["Neural Expectation Maximization"](https://arxiv.org/abs/1708.03498) Greff, Steenkiste, Schmidhuber
  `concept learning`
>	"differentiable clustering method that simultaneously learns how to group and represent individual entities"  
  - `code` <https://github.com/sjoerdvansteenkiste/Neural-EM>

#### ["Generative Models of Visually Grounded Imagination"](https://arxiv.org/abs/1705.10762) Vedantam, Fischer, Huang, Murphy
  `concept learning`
>	"Consider how easy it is for people to imagine what a "purple hippo" would look like, even though they do not exist. If we instead said "purple hippo with wings", they could just as easily create a different internal mental representation, to represent this more specific concept. To assess whether the person has correctly understood the concept, we can ask them to draw a few sketches, to illustrate their thoughts. We call the ability to map text descriptions of concepts to latent representations and then to images (or vice versa) visually grounded semantic imagination. We propose a latent variable model for images and attributes, based on variational auto-encoders, which can perform this task. Our method uses a novel training objective, and a novel product-of-experts inference network, which can handle partially specified (abstract) concepts in a principled and efficient way."  
  - `video` <https://youtu.be/IyP1pxgM_eE?t=1h5m14s> (Murphy)

#### ["SCAN: Learning Abstract Hierarchical Compositional Visual Concepts"](https://arxiv.org/abs/1707.03389) Higgins, Sonnerat, Matthey, Pal, Burgess, Botvinick, Hassabis, Lerchner
  `concept learning`
>	"We first use the previously published beta-VAE (Higgins et al., 2017a) architecture to learn a disentangled representation of the latent structure of the visual world, before training SCAN to extract abstract concepts grounded in such disentangled visual primitives through fast symbol association."  
  - `post` <https://deepmind.com/blog/imagine-creating-new-visual-concepts-recombining-familiar-ones/>

#### ["beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"](http://openreview.net/forum?id=Sy2fzU9gl) Higgins, Matthey, Pal, Burgess, Glorot, Botvinick, Mohamed, Lerchner
  `concept learning`
>	"This paper proposes a modification of the variational ELBO in encourage 'disentangled' representations, and proposes a measure of disentanglement."  
>	"Beta-VAE is a VAE with beta coefficient in KL divergence term where beta=1 is exactly same formulation of vanilla VAE. By increasing beta, the weighted factor forces model to learn more disentangled representation than VAE. The authors also proposed disentanglement metric by training a simple classifier with low capacity and use it’s prediction accuracy. But the metric can be only calculated in simulator (ground truth generator) setting where we can control independent factors to generate different samples with controlled property."  
  - <http://tinyurl.com/jgbyzke> (demo)

#### ["Early Visual Concept Learning with Unsupervised Deep Learning"](http://arxiv.org/abs/1606.05579) Higgins, Matthey, Glorot, Pal, Uria, Blundell, Mohamed, Lerchner
  `concept learning`
  - `code` <https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results>

#### ["Towards Conceptual Compression"](http://arxiv.org/abs/1604.08772) Gregor, Besse, Rezende, Danihelka, Wierstra
  `concept learning`
  - `poster` <https://pbs.twimg.com/media/Cy3pYfWWIAA_C9h.jpg:large>

#### ["Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"](http://arxiv.org/abs/1603.08575) Eslami, Heess, Weber, Tassa, Szepesvari, Kavukcuoglu, Hinton
  `concept learning`
>	"Learning to perform inference in partially specified 2D models (variable-sized variational auto-encoders) and fully specified 3D models (probabilistic renderers)."  
>	"Models learn to identify multiple objects - counting, locating and classifying the elements of a scene - without any supervision, e.g., decomposing 3D images with various numbers of objects in a single forward pass of a neural network."  
  - `video` <https://youtube.com/watch?v=4tc84kKdpY4> (demo)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/EslamiHWTKH16>

----
#### ["Generative Temporal Models with Memory"](http://arxiv.org/abs/1702.04649) Gemici, Hung, Santoro, Wayne, Mohamed, Rezende, Amos, Lillicrap
  `learning disentangled representation`
>	"A sufficiently powerful temporal model should separate predictable elements of the sequence from unpredictable elements, express uncertainty about those unpredictable elements, and rapidly identify novel elements that may help to predict the future. To create such models, we introduce Generative Temporal Models augmented with external memory systems."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1702.04649>

#### ["Learning Disentangled Representations with Semi-Supervised Deep Generative Models"](http://arxiv.org/abs/1706.00400) Siddharth, Paige, Meent, Desmaison, Goodman, Kohli, Wood, Torr
  `learning disentangled representation`
>	"Variational autoencoders learn representations of data by jointly training a probabilistic encoder and decoder network. Typically these models encode all features of the data into a single variable. Here we are interested in learning disentangled representations that encode distinct aspects of the data into separate variables. We propose to learn such representations using model architectures that generalize from standard VAEs, employing a general graphical model structure in the encoder and decoder. This allows us to train partially-specified models that make relatively strong assumptions about a subset of interpretable variables and rely on the flexibility of neural networks to learn representations for the remaining variables."  

#### ["Disentangling Factors of Variation in Deep Representations using Adversarial Training"](http://arxiv.org/abs/1611.03383) Mathieu, Zhao, Sprechmann, Ramesh, LeCun
  `learning disentangled representation`
  - `notes` <http://www.shortscience.org/paper?bibtexKey=conf%2Fnips%2FMathieuZZRSL16>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1611.03383>

#### ["Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders"](http://arxiv.org/abs/1611.02648) Dilokthanakul, Mediano, Garnelo, Lee, Salimbeni, Arulkumaran, Shanahan
  `learning disentangled representation`
  - `post` <http://ruishu.io/2016/12/25/gmvae/>

#### ["Inducing Interpretable Representations with Variational Autoencoders"](http://arxiv.org/abs/1611.07492) Siddharth, Paige, Desmaison, Meent, Wood, Goodman, Kohli, Torr
  `learning disentangled representation`

#### ["The Variational Fair Autoencoder"](http://arxiv.org/abs/1511.00830) Louizos, Swersky, Li, Welling, Zemel
  `learning disentangled representation`
>	"We investigate the problem of learning representations that are invariant to certain nuisance or sensitive factors of variation in the data while retaining as much of the remaining information as possible."  
  - `video` <http://videolectures.net/iclr2016_louizos_fair_autoencoder/> (Louizos)

----
#### ["Optimizing the Latent Space of Generative Networks"](https://arxiv.org/abs/1707.05776) Bojanowski, Joulin, Lopez-Paz, Szlam
  `learning embedding`
  - `post` <https://facebook.com/yann.lecun/posts/10154646915277143>

#### ["Unsupervised Learning by Predicting Noise"](https://arxiv.org/abs/1704.05310) Bojanowski, Joulin
  `learning embedding`
>	"The authors give a nice analogy: it's a SOM, but instead of mapping a latent vector to each input vector, the convolutional filters are learned in order to map each input vector to a fixed latent vector. In more words: each image is assigned a unique random latent vector as the label, and the mapping from image to label is taught in a supervised manner. Every few epochs, the label assignments are adjusted (but only within batches due to computational cost), so that an image might be assigned a different latent vector label which it is already close to in 'feature space'."
  - `post` <http://inference.vc/unsupervised-learning-by-predicting-noise-an-information-maximization-view-2/>

#### ["Poincare Embeddings for Learning Hierarchical Representations"](https://arxiv.org/abs/1705.08039) Nickel, Kiela
  `learning embedding`
  - `notes` <https://medium.com/towards-data-science/facebook-research-just-published-an-awesome-paper-on-learning-hierarchical-representations-34e3d829ede7>
  - `notes` <https://medium.com/@hol_io/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795>
  - `code` <https://github.com/TatsuyaShirakawa/poincare-embedding>



---
### generative models

  - [invertible density estimation](#generative-models---invertible-density-estimation)
  - [generative adversarial networks](#generative-models---generative-adversarial-networks)
  - [variational autoencoders](#generative-models---variational-autoencoders)
  - [autoregressive models](#generative-models---autoregressive-models)

----
#### ["Comparison of Maximum Likelihood and GAN-based training of Real NVPs"](https://arxiv.org/abs/1705.05263) Danihelka, Lakshminarayanan, Uria, Wierstra, Dayan
  `evaluation`
>	"We use a tractable generator architecture for which the log-probability densities can be computed exactly. We train the generator architecture by maximum likelihood and we also train the same generator architecture by GAN. We then compare the properties of the learned generators."  
>	"Generators trained by WGAN produce more globally coherent samples even from a relatively shallow generator."  
>	"Minimization of the approximate Wasserstein distance does not correspond to minimization of the negative log-probability density. The negative log-probability densities became worse than densities from a uniform distribution."  
>	"An approximation of the Wasserstein distance ranked correctly generators trained by maximum likelihood."  
>	"The approximate Wasserstein distance between the training data and the generator distribution became smaller than the distance between the test data and the generator distribution. This overfitting was observed for generators trained by maximum likelihood and also for generators trained by WGAN."  
>	"We inspected the negative log-probability density of samples from generator trained by WGAN. The negative log-probability density can be negative, if the probability density is bigger than 1. In contrast, the NVP generator trained by maximum likelihood assigned on average positive values to its own generated samples. A deep generator trained by WGAN learns a distribution lying on a low dimensional manifold. The generator is then putting the probability mass only to a space with a near-zero volume. We may need a more powerful critic to recognize the excessively correlated pixels. Approximating the likelihood by annealed importance sampling (Wu et al., 2016) would not discover this problem, as their analysis assumes a Gaussian observation model with a fixed variance. The problem is not unique to WGAN. We also obtained near-infinite negative log-probability densities when training GAN to minimize the Jensen-Shannon divergence."  
>	"One of the advantages of real NVPs is that we can infer the original latent z0 for a given generated sample. We know that the distribution of the latent variables is the prior N(0,1), if the given images are from the generator. We are curious to see the distribution of the latent variables, if the given images are from the validation set. We display a 2D histogram of the first 2 latent variables z0[1], z0[2]. The histogram was obtained by inferring the latent variables for all examples from the validation set. When the generator was trained by maximum likelihood, the inferred latent variables had the following means and standard deviations: µ1 = 0.05, µ2 = 0.05, σ1 = 1.06, σ2 = 1.03. In contrast, the generator trained by WGAN had inferred latent variables with significantly larger standard deviations: µ1 = 0.02, µ2 = 1.62, σ1 = 3.95, σ2 = 8.96. When generating the latent variables from the N(0,1) prior, the samples from the generator trained by WGAN would have a different distribution than the validation set."  
>	"Real NVPs are invertible transformations and have perfect reconstructions. We can still visualize reconstructions from a partially resampled latent vector. Gregor et al. (2016) and Dinh et al. (2016) visualized ‘conceptual compression’ by inferring the latent variables and then resampling a part of the latent variables from the normal N(0,1) prior. The subsequent reconstruction should still form a valid image. If the original image was generated by the generator, the partially resampled latent vector would still have the normal N(0,1) distribution. We show the reconstructions if resampling the first half or the second half of the latent vector. The generator trained by maximum likelihood has partial reconstructions similar to generated samples. In comparison, the partial reconstructions from the generator trained by WGAN do not resemble samples from WGAN. This again indicates that the validation examples have a different distribution than WGAN samples."  
>	"We looked at the approximate Wasserstein distance between the validation data and the generator distribution. We will train another critic to assign high values to validation samples and low values to generated samples. This independent critic will be used only for evaluation. The generator will not see the gradients from the independent critic. We display the approximate Wasserstein distance between the validation set and the generator distribution. The first thing to notice is the correct ordering of generators trained by maximum likelihood. The deepest generator has the smallest approximate distance from the validation set, as indicated by the thick solid lines. We also display an approximate distance between the training set and generator distribution, and the approximate distance between the test set and the generator distribution. The approximate distance between the test set and the generator distribution is a little bit smaller than the approximate distance between the validation set and the generator distribution. The approximate distance between the training set and the generator distribution is much smaller. The generators are overfitting the training set."  
>	"Real NVP can be used as an encoder in Adversarial Variational Bayes. We were able to measure the gap between the unbiased KL estimate log q(z|x) - log p(z) and its approximation from GAN. We show that Adversarial Variational Bayes underestimates the KL divergence."  

#### ["On the Quantitative Analysis of Decoder-based Generative Models"](http://arxiv.org/abs/1611.04273) Wu, Burda, Salakhutdinov, Grosse
  `evaluation`
>	"We propose to use Annealed Importance Sampling for evaluating log-likelihoods for decoder-based models and validate its accuracy using bidirectional Monte Carlo. Using this technique, we analyze the performance of decoder-based models, the effectiveness of existing log-likelihood estimators, the degree of overfitting, and the degree to which these models miss important modes of the data distribution."  
>	"This paper introduces Annealed Importance Sampling to compute tighter lower bounds and upper bounds for any generative model (with a decoder)."  
  - `video` <https://youtu.be/RZOKRFBtSh4?t=54m8s> (Wu)
  - `code` <https://github.com/tonywu95/eval_gen>

#### ["Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy"](https://arxiv.org/abs/1611.04488) Sutherland, Tung, Strathmann, De, Ramdas, Smola, Gretton
  `evaluation`
>	"We propose a method to optimize the representation and distinguishability of samples from two probability distributions, by maximizing the estimated power of a statistical test based on the maximum mean discrepancy. In context of GANs, the MMD may be used in two roles: first, as a discriminator, either directly on the samples, or on features of the samples. Second, the MMD can be used to evaluate the performance of a generative model, by testing the model’s samples against a reference data set."  
>
>	"If we compare images in pixel space, none of the generated distributions by GANs pass those tests. Basically this method notice that there are artefacts in which the two-sample tests can lock onto to distinguish between real and fake images."  
>	"Authors replicated the experiment with MNIST in Salimans et al. (2016) in which human could not distinguish between fake and real digits and showed that for the MMD test the difference was easy, because of the artifacts and because the GANs were only showing standard digits, but the GAN was not able to generate samples from the tail of the distribution. This experiment shows the risk of using human evaluation of densities estimates, as we focus more on normalcy and under-represent the unexpected. GANs might be only focusing on the modes in the data and ignoring those tails, which might be critical in applications in which the tails carry the information of the extreme events that we might be interested in (e.g. think of weird genetic diseases or extreme weather patterns)."  
  - `video` <https://youtube.com/watch?v=Xpd6DL02C7Q> (Sutherland)
  - `code` <https://github.com/dougalsutherland/opt-mmd>

#### ["Revisiting Classifier Two-Sample Tests for GAN Evaluation and Causal Discovery"](http://arxiv.org/abs/1610.06545) Lopez-Paz, Oquab
  `evaluation`
>	"We propose C2ST use to evaluate the sample quality of generative models with intractable likelihoods such as GANs."  
>	"We showcase the novel application of GANs together with C2ST for causal discovery."  
>
>	"If we compare images in pixel space, none of the generated distributions by GANs pass those tests. Basically this method notice that there are artefacts in which the two-sample tests can lock onto to distinguish between real and fake images."  
>	"Authors also showed that if instead comparing in the pixel space, the comparison is made in some transformed space (in their case the final layer in a Resnet structure), the fake samples and the true samples were indistinguishable. This result is quite appealing, because there are some statistics in which the GANs samples are distributed like the real samples and those statistics are sufficient for our problem in hand then we might be able to rely on GANs as a simulator. The main questions is for which statistics this happens and how broad they are."  

#### ["A Note on the Evaluation of Generative Models"](http://arxiv.org/abs/1511.01844) Theis, Oord, Bethge
  `evaluation`
  - `video` <http://videolectures.net/iclr2016_theis_generative_models/> (Theis)

----
#### ["Variational Approaches for Auto-Encoding Generative Adversarial Networks"](https://arxiv.org/abs/1706.04987) Rosca, Lakshminarayanan, Warde-Farley, Mohamed
  `unifying GANs and VAEs`
  - `video` <https://youtu.be/jAI3rBI6poU?t=1h1m33s> (Ulyanov) `in russian`
  - `code` <https://github.com/victor-shepardson/alpha-GAN>

#### ["On Unifying Deep Generative Models"](https://arxiv.org/abs/1706.00550) Hu, Yang, Salakhutdinov, Xing
  `unifying GANs and VAEs`
>	"We show that GANs and VAEs are essentially minimizing KL divergences of respective posterior and inference distributions with opposite directions, extending the two learning phases of classic wake-sleep algorithm, respectively. The unified view provides a powerful tool to analyze a diverse set of existing model variants, and enables to exchange ideas across research lines in a principled way. For example, we transfer the importance weighting method in VAE literatures for improved GAN learning, and enhance VAEs with an adversarial mechanism for leveraging generated samples."  

#### ["Flow-GAN: Bridging Implicit and Prescribed Learning in Generative Models"](https://arxiv.org/abs/1705.08868) Grover, Dhar, Ermon
  `unifying GANs and VAEs`
>	"generative adversarial network which allows for tractable likelihood evaluation"  
>	"Since it can be trained both adversarially (like a GAN) and in terms of MLE (like a flow model), we can quantitatively evaluate the trade-offs involved. In particular, we also consider a hybrid objective function which involves both types of losses."  
>	"The availability of quantitative metrics allow us to compare to simple baselines which essentially “remember” the training data. Our final results show that naive Gaussian Mixture Models outperforms plain WGAN on both samples quality and log-likelihood for both MNIST and CIFAR-10 which we hope will lead to new directions for both implicit and prescribed learning in generative models."  

#### ["Hierarchical Implicit Models and Likelihood-Free Variational Inference"](http://arxiv.org/abs/1702.08896) Tran, Ranganath, Blei
  `unifying GANs and VAEs`
>	"We introduce hierarchical implicit models. HIMs combine the idea of implicit densities with hierarchical Bayesian modeling, thereby defining models via simulators of data with rich hidden structure."  
>	"We develop likelihood-free variational inference, a scalable variational inference algorithm for HIMs. Key to LFVI is specifying a variational family that is also implicit. This matches the model's flexibility and allows for accurate approximation of the posterior."  
  - `poster` <http://dustintran.com/papers/TranRanganathBlei2017_poster.pdf>
  - `post` <http://dustintran.com/blog/deep-and-hierarchical-implicit-models>
  - `post` <https://bayesgroup.github.io/sufficient-statistics/posts/hierarchical-implicit-models-and-likelihood-free-variational-inference/> `in russian`

#### ["Variational Inference using Implicit Distributions"](http://arxiv.org/abs/1702.08235) Huszar
  `unifying GANs and VAEs`
>	"This paper provides a unifying review of existing algorithms establishing connections between variational autoencoders, adversarially learned inference, operator VI, GAN-based image reconstruction, and more."  
  - `post` <http://inference.vc/variational-inference-with-implicit-probabilistic-models-part-1-2/>
  - `post` <http://inference.vc/variational-inference-with-implicit-models-part-ii-amortised-inference-2/>
  - `post` <http://inference.vc/variational-inference-using-implicit-models-part-iii-joint-contrastive-inference-ali-and-bigan/>
  - `post` <http://inference.vc/variational-inference-using-implicit-models-part-iv-denoisers-instead-of-discriminators/>

#### ["Learning in Implicit Generative Models"](http://arxiv.org/abs/1610.03483) Mohamed, Lakshminarayanan
  `unifying GANs and VAEs`
  - `video` <https://youtu.be/RZOKRFBtSh4?t=5m37s> (Mohamed)
  - `video` <https://youtu.be/jAI3rBI6poU?t=37m56s> (Ulyanov) `in russian`
  - `post` <https://casmls.github.io/general/2017/05/24/ligm.html>

#### ["Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks"](http://arxiv.org/abs/1701.04722) Mescheder, Nowozin, Geiger
  `unifying GANs and VAEs`
>	"Real NVP can be used as an encoder in order to measure the gap between the unbiased KL estimate log q(z|x) - log p(z) and its approximation from GAN. We show that Adversarial Variational Bayes underestimates the KL divergence."  
  - `video` <https://youtu.be/y7pUN2t5LrA?t=14m19s> (Nowozin)
  - `video` <https://youtu.be/xFCuXE1Nb8w?t=26m55s> (Nowozin)
  - `video` <https://youtu.be/m80Vp-jz-Io?t=1h28m34s> (Tolstikhin)
  - `post` <http://inference.vc/variational-inference-with-implicit-models-part-ii-amortised-inference-2/>
  - `notes` <https://casmls.github.io/general/2017/02/23/modified-gans.html>
  - `code` <https://github.com/wiseodd/generative-models/tree/master/VAE/adversarial_vb>
  - `code` <https://gist.github.com/poolio/b71eb943d6537d01f46e7b20e9225149>



---
### generative models - invertible density estimation

[interesting recent papers - generative models](#generative-models)

----
#### ["Masked Autoregressive Flow for Density Estimation"](https://arxiv.org/abs/1705.07057) Papamakarios, Pavlakou, Murray
>	"We describe an approach for increasing the flexibility of an autoregressive model, based on modelling the random numbers that the model uses internally when generating data. By constructing a stack of autoregressive models, each modelling the random numbers of the next model in the stack, we obtain a type of normalizing flow suitable for density estimation. This type of flow is closely related to Inverse Autoregressive Flow and is a generalization of Real NVP."  
>	"MAF:  
>	- fast to calculate p(x)  
>	- slow to sample from  
>	Real NVP:  
>	- fast to calculate p(x)  
>	- fast to sample from  
>	- limited capacity vs MAF"  
  - `video` <https://facebook.com/NIPSlive/videos/1492589610796880/> (2:15) (Papamakarios)
  - `code` <https://github.com/gpapamak/maf>

#### ["Density Estimation using Real NVP"](http://arxiv.org/abs/1605.08803) Dinh, Sohl-Dickstein, Bengio
>	"Real-valued Non Volume Preserving transform:  
>	- one-pass and exact inference and sampling  
>	- explicit learning of a latent representation  
>	- tractable log-likelihood  
>	- coherent and sharp samples"  
  - <https://laurent-dinh.github.io/2016/07/12/real-nvp-visualization.html> (demo)
  - `video` <https://channel9.msdn.com/events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-1> (08:19) (Dinh)
  - `video` <https://periscope.tv/hugo_larochelle/1ypKdAVmbEpGW> (Dinh)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.08803>
  - `code` <https://github.com/tensorflow/models/tree/master/research/real_nvp>
  - `code` <https://github.com/taesung89/real-nvp>



---
### generative models - generative adversarial networks

[interesting older papers](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---generative-adversarial-networks)

[interesting recent papers - generative models](#generative-models)

----
#### ["NIPS 2016 Tutorial: Generative Adversarial Networks"](http://arxiv.org/abs/1701.00160) Goodfellow
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks> (Goodfellow) ([slides](http://iangoodfellow.com/slides/2016-12-04-NIPS.pdf))

#### ["What Are GANs Useful For?"](https://openreview.net/forum?id=HkwrqtlR-)
>	"Generative and discriminative learning are quite different. Discriminative learning has a clear end, while generative modeling is an intermediate step to understand the data or generate hypothesis. The quality of implicit density estimation is hard to evaluate, because we cannot tell how well a data is represented by the model. How can we certainly say that a generative process is generating natural images with the same distribution as we do? In this paper, we noticed that even though GANs might not be able to generate samples from the underlying distribution (or we cannot tell at least), they are capturing some structure of the data in that high dimensional space. It is therefore needed to address how we can leverage those estimates produced by GANs in the same way we are able to use other generative modeling algorithms."  

#### ["Are GANs Created Equal? A Large-Scale Study"](https://arxiv.org/abs/1711.10337) Lucic, Kurach, Michalski, Gelly, Bousquet
>	"We find that most models can reach similar scores with enough hyperparameter optimization and random restarts. This suggests that improvements can arise from a higher computational budget and tuning more than fundamental algorithmic changes. We did not find evidence that any of the tested algorithms consistently outperforms the original one."  
>	"We have started a discussion on how to neutrally and fairly compare GANs. We focus on two sets of evaluation metrics: (i) The Frechet Inception Distance, and (ii) precision, recall and F1. We provide empirical evidence that FID is a reasonable metric due to its robustness with respect to mode dropping and encoding network choices."  
>	"Two evaluation metrics were proposed to quantitatively assess the performance of GANs. Both assume access to a pre-trained classifier. Inception Score is based on the fact that a good model should generate samples for which, when evaluated by the classifier, the class distribution has low entropy. At the same time, it should produce diverse samples covering all classes. In contrast, Frechet Inception Distance is computed by considering the difference in embedding of true and fake data. Assuming that the coding layer follows a multivariate Gaussian distribution, the distance between the distributions is reduced to the Frechet distance between the corresponding Gaussians."  
>	"FID cannot detect overfitting to the training data set, and an algorithm that just remembers all the training examples would perform very well. Finally, FID can probably be “fooled” by artifacts that are not detected by the embedding network."  
>	"We introduce a series of tasks of increasing difficulty for which undisputed measures, such as precision and recall, can be approximately computed."  

----
#### ["Theoretical Limitations of Encoder-Decoder GAN Architectures"](https://arxiv.org/abs/1711.02651) Arora, Risteski, Zhang
  `GAN theory`
>	"Encoder-decoder GANs architectures (e.g., BiGAN and ALI) seek to add an “inference” mechanism to the GANs setup, consisting of a small encoder deep net that maps data-points to their succinct encodings. The intuition is that being forced to train an encoder alongside the usual generator forces the system to learn meaningful mappings from the code to the data-point and vice-versa, which should improve the learning of the target distribution and ameliorate mode-collapse. It should also yield meaningful codes that are useful as features for downstream tasks. The current paper shows rigorously that even on real-life distributions of images, the encode-decoder GAN training objectives (a) cannot prevent mode collapse; i.e. the objective can be near-optimal even when the generated distribution has low and finite support (b) cannot prevent learning meaningless codes for data – essentially white noise. Thus if encoder-decoder GANs do indeed work then it must be due to reasons as yet not understood, since the training objective can be low even for meaningless solutions."  

#### ["Approximation and Convergence Properties of Generative Adversarial Learning"](https://arxiv.org/abs/1705.08991) Liu, Bousquet, Chaudhuri
  `GAN theory`
>	"Two very basic questions on how well GANs can approximate the target distribution µ, even in the presence of a very large number of samples and perfect optimization, remain largely unanswered.  
>	The first relates to the role of the discriminator in the quality of the approximation. In practice, the discriminator is usually restricted to belong to some family, and it is not understood in what sense this restriction affects the distribution output by the generator.  
>	The second question relates to convergence; different variants of GANs have been proposed that involve different objective functions (to be optimized by the generator and the discriminator). However, it is not understood under what conditions minimizing the objective function leads to a good approximation of the target distribution. More precisely, does a sequence of distributions output by the generator that converges to the global minimum under the objective function always converge to the target distribution µ under some standard notion of distributional convergence?"  
>	"We first characterize a very general class of objective functions that we call adversarial divergences, and we show that they capture the objective functions used by a variety of existing procedures that include the original GAN, f-GAN, MMD-GAN, WGAN, improved WGAN, as well as a class of entropic regularized optimal transport problems. We then define the class of strict adversarial divergences – a subclass of adversarial divergences where the minimizer of the objective function is uniquely the target distribution."  
>	"We show that if the objective function is an adversarial divergence that obeys certain conditions, then using a restricted class of discriminators has the effect of matching generalized moments. A concrete consequence of this result is that in linear f-GANs, where the discriminator family is the set of all affine functions over a vector ψ of features maps, and the objective function is an f-GAN, the optimal distribution ν output by the GAN will satisfy Ex~µ[ψ(x)] = Ex~ν[ψ(x)] regardless of the specific f-divergence chosen in the objective function. Furthermore, we show that a neural network GAN is just a supremum of linear GANs, therefore has the same moment-matching effect."  
>	"We show that convergence in an adversarial divergence implies some standard notion of topological convergence. Particularly, we show that provided an objective function is a strict adversarial divergence, convergence to µ in the objective function implies weak convergence of the output distribution to µ. An additional consequence of this result is the observation that as the Wasserstein distance metrizes weak convergence of probability distributions, Wasserstein-GANs have the weakest objective functions in the class of strict adversarial divergences."  
>
>	"Authors worry about two important aspects of GAN convergence: what how good the generative distribution approximates the real distribution; and, when does this convergence takes place. For the first question the answer is the discriminator forces some kind of moment matching between the real and fake distributions. In order to get full representation of the density we will need that the discriminator grows with the data. For the second question, they show a week convergence result. This result is somewhat complementary to Arora et al. (2017), because it indicates that the discriminator complexity needs to grow indefinitely to achieve convergence. The question that remains to be answer is the rate of convergence, as the moments need to be matched for complicated distributions might require large data and complex discriminators. So in practice, we cannot tell if the generated distribution is close enough to the distribution we are interested in."  
>	"Today’s GAN’s results can be explained in the light of these theoretical results. First, we are clearly matching some moments that relate to visual quality of natural images with finite size deep neural networks, but they might not be deep enough to capture all relevant moments and hence they will not be able to match the distribution in these images."  

#### ["Do GANs Actually Learn the Distribution? An Empirical Study"](https://arxiv.org/abs/1706.08224) Arora, Zhang
  `GAN theory`
>	"A recent analysis raised doubts whether GANs actually learn the target distribution when discriminator has finite size. It showed that the training objective can approach its optimum value even if the generated distribution has very low support ---in other words, the training objective is unable to prevent mode collapse. The current note reports experiments suggesting that such problems are not merely theoretical. It presents empirical evidence that well-known GANs approaches do learn distributions of fairly low support, and thus presumably are not learning the target distribution. The main technical contribution is a new proposed test, based upon the famous birthday paradox, for estimating the support size of the generated distribution."  
  - `post` <http://www.offconvex.org/2017/07/07/GANs3/> (Arora)

#### ["Generalization and Equilibrium in Generative Adversarial Nets"](https://arxiv.org/abs/1703.00573) Arora, Ge, Liang, Ma, Zhang
  `GAN theory`
>	"GAN training may not have good generalization properties; e.g., training may appear successful but the trained distribution may be far from target distribution in standard metrics. However, generalization does occur for a weaker metric called neural net distance. It is also shown that an approximate pure equilibrium exists in the discriminator/generator game for a special class of generators with natural training objectives when generator capacity and training set sizes are moderate."  
>	"Authors prove that for the standard metrics (e.g. Shannon-Jensen divergence and Wasserstein-like integral probability metrics), the discriminator might stop discriminating before the estimated distribution converges to the density of the data. They also show a weaker result, in which convergence might happen, but the estimated density might be off, if the discriminator based on a deep neural network is not large enough (sufficient VC dimension)."  
  - `video` <https://youtube.com/watch?v=V7TliSCqOwI> (Arora)
  - `post` <http://www.offconvex.org/2017/03/30/GANs2/> (Arora)

#### ["Towards Principled Methods for Training Generative Adversarial Networks"](https://arxiv.org/abs/1701.04862) Arjovsky, Bottou
  `GAN theory`

----
#### ["Optimizing the Latent Space of Generative Networks"](https://arxiv.org/abs/1707.05776) Bojanowski, Joulin, Lopez-Paz, Szlam
  `GAN objective`
>	"GAN without discriminator"  

#### ["Bayesian GAN"](https://arxiv.org/abs/1705.09558) Saatchi, Wilson
  `GAN objective`
>	"In this paper, we present a simple Bayesian formulation for end-to-end unsupervised and semi-supervised learning with generative adversarial networks. Within this framework, we marginalize the posteriors over the weights of the generator and discriminator using stochastic gradient Hamiltonian Monte Carlo. We interpret data samples from the generator, showing exploration across several distinct modes in the generator weights. We also show data and iteration efficient learning of the true distribution. We also demonstrate state of the art semi-supervised learning performance on several benchmarks, including SVHN, MNIST, CIFAR-10, and CelebA. The simplicity of the proposed approach is one of its greatest strengths: inference is straightforward, interpretable, and stable. Indeed all of the experimental results were obtained without feature matching, normalization, or any ad-hoc techniques."

#### ["AdaGAN: Boosting Generative Models"](https://arxiv.org/abs/1701.02386) Tolstikhin, Gelly, Bousquet, Simon-Gabriel, Scholkopf
  `GAN objective`
>	"We call it Adaptive GAN, but we could actually use any other generator: a Gaussian mixture model, a VAE, a WGAN, or even an unrolled or mode-regularized GAN, which were both already specifically developed to tackle the missing mode problem. Thus, we do not aim at improving the original GAN or any other generative algorithm. We rather propose and analyse a meta-algorithm that can be used on top of any of them. This meta-algorithm is similar in spirit to AdaBoost in the sense that each iteration corresponds to learning a “weak” generative model (e.g., GAN) with respect to a re-weighted data distribution. The weights change over time to focus on the “hard” examples, i.e. those that the mixture has not been able to properly generate so far."  
>
>	"New components are added to the mixture until the original distribution is recovered and authors show exponential convergence to the underlying density. Our concern with the proof is its practical applicability, as it requires that, at each step, the GAN estimated density, call it dQ, and the true underlying density of the data, call it dPd, satisfy that βdQ ≤ dPd. However, it is indeed unknown how to design a generative network that induces a density dQ that would guarantee βdQ ≤ dPd with a non-zero β when dPd is a high-dimensional structure generative process."  
  - `video` <https://youtube.com/watch?v=5EEaY_cVYkk>
  - `video` <https://youtube.com/watch?v=myvoMklo5Uc> (Tolstikhin)
  - `video` <https://youtube.com/watch?v=wPKGIIy4rtU> (Bousquet)

#### ["Improved Training of Wasserstein GANs"](https://arxiv.org/abs/1704.00028) Gulrajani, Ahmed, Arjovsky, Dumoulin, Courville
  `GAN objective`
  - `post` <https://casmls.github.io/general/2017/04/13/gan.html>
  - `notes` <https://bayesgroup.github.io/sufficient-statistics/posts/wasserstein-generative-adversarial-networks/> `in russian`
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/improved_wasserstein_gan>
  - `code` <https://github.com/igul222/improved_wgan_training>

#### ["Wasserstein GAN"](https://arxiv.org/abs/1701.07875) Arjovsky, Chintala, Bottou
  `GAN objective`
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
  - `notes` <https://casmls.github.io/general/2017/02/23/modified-gans.html>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1701.07875>
  - `notes` <https://bayesgroup.github.io/sufficient-statistics/posts/wasserstein-generative-adversarial-networks/> `in russian`
  - `code` <https://github.com/martinarjovsky/WassersteinGAN>
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/wasserstein_gan>
  - `code` <https://github.com/shekkizh/WassersteinGAN.tensorflow>
  - `code` <https://github.com/kuleshov/tf-wgan>
  - `code` <https://github.com/blei-lab/edward/blob/master/examples/gan_wasserstein.py>
  - `code` <https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/WassersteinGAN>

#### ["f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization"](http://arxiv.org/abs/1606.00709) Nowozin, Cseke, Tomioka
  `GAN objective`
>	"We show that the generative-adversarial approach is a special case of an existing more general variational divergence estimation approach. We show that any f-divergence can be used for training generative neural samplers. We discuss the benefits of various choices of divergence functions on training complexity and the quality of the obtained generative models."  
>	"For a given f-divergence, the model learns the composition of the density ratio (of data to model density) with the derivative of f, by comparing generator and data samples. This provides a lower bound on the “true” divergence that would be obtained if the density ratio were perfectly known. In the event that the model is in a smaller class than the true data distribution, this broader family of divergences implements a variety of different approximations: some focus on individual modes of the true sample density, others try to cover the support."  
  - `video` <https://youtube.com/watch?v=I1M_jGWp5n0>
  - `video` <https://youtube.com/watch?v=kQ1eEXgGsCU> (Nowozin)
  - `video` <https://youtube.com/watch?v=y7pUN2t5LrA> (Nowozin)
  - `video` <https://youtu.be/jAI3rBI6poU?t=14m31s> (Ulyanov) `in russian`
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/f_gan>

----
#### ["GANs Trained by a Two Time-Scale Update Rule Converge to a Nash Equilibrium"](https://arxiv.org/abs/1706.08500) Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter
  `GAN training protocol`
>	"We propose a two time-scale update rule (TTUR) for training GANs with stochastic gradient descent that has an individual learning rate for both the discriminator and the generator."  
>	"For the evaluation of the performance of GANs at image generation, we introduce the "Frechet Inception Distance" (FID) which captures the similarity of generated images to real ones better than the Inception Score."  
>	"In experiments, TTUR improves learning for DCGANs, improved Wasserstein GANs, and BEGANs."  
>	"to the best of our knowledge this is the first convergence proof for GANs"  
  - `video` <https://youtu.be/h6eQrkkU9SA?t=21m6s> (Hochreiter)
  - `code` <https://github.com/bioinf-jku/TTUR>

#### ["How to Train Your DRAGAN"](https://arxiv.org/abs/1705.07215) Kodali, Abernethy, Hays, Kira
  `GAN training protocol`
  - `code` <https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/DRAGAN.py>

#### ["BEGAN: Boundary Equilibrium Generative Adversarial Networks"](https://arxiv.org/abs/1703.10717) Berthelot, Schumm, Metz
  `GAN training protocol`
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

#### ["Unrolled Generative Adversarial Networks"](http://arxiv.org/abs/1611.02163) Metz, Poole, Pfau, Sohl-Dickstein
  `GAN training protocol`
>	"We introduce a method to stabilize GANs by defining the generator objective with respect to an unrolled optimization of the discriminator. This allows training to be adjusted between using the optimal discriminator in the generator's objective, which is ideal but infeasible in practice, and using the current value of the discriminator, which is often unstable and leads to poor solutions. We show how this technique solves the common problem of mode collapse, stabilizes training of GANs with complex recurrent generators, and increases diversity and coverage of the data distribution by the generator."  
  - `video` <https://youtu.be/RZOKRFBtSh4?t=26m16s> (Metz)
  - `code` <https://github.com/poolio/unrolled_gan>

#### ["A Connection between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models"](https://arxiv.org/abs/1611.03852) Finn, Christiano, Abbeel, Levine
  `GAN training protocol`
>	"sampling-based MaxEnt IRL is a GAN with a special form of discriminator and uses RL to optimize the generator"  
  - `video` <https://youtu.be/d9DlQSJQAoI?t=23m6s> (Finn)
  - `video` <https://youtu.be/RZOKRFBtSh4?t=10m48s> (Finn)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-3> (33:17) (Levine)
  - `notes` <http://tsong.me/blog/gan-irl-ebm/>
  - `notes` <http://pemami4911.github.io/paper-summaries/generative-adversarial-networks/2017/02/12/gans-irl-ebm.html>

#### ["Connecting Generative Adversarial Networks and Actor-Critic Methods"](https://arxiv.org/abs/1610.01945) Pfau, Vinyals
  `GAN training protocol`
  - `video` <https://youtube.com/watch?v=RZOKRFBtSh4&t=1m5s> (Pfau)
  - `video` <https://youtube.com/watch?v=1zQDCkqj3Tc> (Pfau)
  - `video` <https://youtu.be/xfyK03MEZ9Q?t=5h21m15s> (Bagnell)

----
#### ["Adversarially Regularized Autoencoders"](https://arxiv.org/abs/1706.04223) Zhao, Kim, Zhang, Rush, LeCun
  `GAN applications` `discrete output`

#### ["Maximum-Likelihood Augmented Discrete Generative Adversarial Networks"](http://arxiv.org/abs/1702.07983) Che, Li, Zhang, Hjelm, Li, Song, Bengio
  `GAN applications` `discrete output`

#### ["Boundary-Seeking Generative Adversarial Networks"](http://arxiv.org/abs/1702.08431) Hjelm, Jacob, Che, Cho, Bengio
  `GAN applications` `discrete output`
>	"We observe that the recently proposed Gumbel-Softmax technique for re-parametrizing the discrete variables does not work for training a GAN with discrete data."  
  - `post` <http://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/>
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/boundary_seeking_gan>

#### ["GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution"](http://arxiv.org/abs/1611.04051) Kusner, Hernandez-Lobato
  `GAN applications` `discrete output`
  - `post` <https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html>

----
#### ["High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"](https://arxiv.org/abs/1711.11585) Wang, Liu, Zhu, Tao, Kautz, Catanzaro
  `GAN applications` `image synthesis`
  - <https://tcwang0509.github.io/pix2pixHD/> (demo)
  - `code` <https://github.com/NVIDIA/pix2pixHD>

#### ["Progressive Growing of GANs for Improved Quality, Stability, and Variation"](https://arxiv.org/abs/1710.10196) Karras, Aila, Laine, Lehtinen
  `GAN applications` `image synthesis`
>	"Instead of Wassersteinizing, just keep the KL loss, but get rid of the disjoint support problem by doing multiresolution approximation of the data distribution."  
  - `video` <https://youtube.com/watch?v=XOxxPcy5Gr4> (demo)
  - `code` <https://github.com/tkarras/progressive_growing_of_gans>
  - `code` <https://github.com/ptrblck/prog_gans_pytorch_inference>

----
#### ["Learning from Simulated and Unsupervised Images through Adversarial Training"](http://arxiv.org/abs/1612.07828) Shrivastava, Pfister, Tuzel, Susskind, Wang, Webb
  `GAN applications` `domain adaptation`
  - `video` <https://youtube.com/watch?v=P3ayMdNdokg> (Shrivastava)
  - `code` <https://github.com/carpedm20/simulated-unsupervised-tensorflow>

#### ["Unsupervised Pixel-Level Domain Adaptation with Generative Asversarial Networks"](http://arxiv.org/abs/1612.05424) Bousmalis, Silberman, Dohan, Erhan, Krishnan
  `GAN applications` `domain adaptation`
  - `video` <https://youtube.com/watch?v=VhsTrWPvjcA> (Bousmalis)

----
#### ["Learning to Discover Cross-Domain Relations with Generative Adversarial Networks"](https://arxiv.org/abs/1703.05192) Kim et al.
  `GAN applications` `domain translation`
  - `code` <https://github.com/SKTBrain/DiscoGAN>

#### ["DualGAN: Unsupervised Dual Learning for Image-to-Image Translation"](https://arxiv.org/abs/1704.02510) Yi et al.
  `GAN applications` `domain translation`
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/dual_gan>

#### ["Unsupervised Image-to-Image Translation Networks"](http://arxiv.org/abs/1703.00848) Liu, Breuel, Kautz
  `GAN applications` `domain translation`



---
### generative models - variational autoencoders

[interesting older papers](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---variational-autoencoder)

[interesting recent papers - generative models](#generative-models)  
[interesting recent papers - bayesian deep learning](#bayesian-deep-learning)  
[interesting recent papers - unsupervised learning](#unsupervised-learning)  

----
#### ["An Information-Theoretic Analysis of Deep Latent-Variable Models"](https://arxiv.org/abs/1711.00464) Alemi, Poole, Fischer, Dillon, Saurous, Murphy
>	"We present an information-theoretic framework for understanding trade-offs in unsupervised learning of deep latent-variables models using variational inference. This framework emphasizes the need to consider latent-variable models along two dimensions: the ability to reconstruct inputs (distortion) and the communication cost (rate). We derive the optimal frontier of generative models in the two-dimensional rate-distortion plane, and show how the standard evidence lower bound objective is insufficient to select between points along this frontier. However, by performing targeted optimization to learn generative models with different rates, we are able to learn many models that can achieve similar generative performance but make vastly different trade-offs in terms of the usage of the latent variable."  

----
#### ["Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) Oord, Vinyals, Kavukcuoglu
  `VQ-VAE`
>	"Language is inherently discrete, similarly speech is typically represented as a sequence of symbols. Images can often be described concisely by language. Furthermore, discrete representations are a natural fit for complex reasoning, planning and predictive learning."  
>	"We introduce a new family of generative models succesfully combining the VAE framework with discrete latent representations through a novel parameterisation of the posterior distribution of (discrete) latents given an observation. Our model, which relies on vector quantization, is simple to train, does not suffer from large variance, and avoids the “posterior collapse” issue which has been problematic with many VAE models that have a powerful decoder, often caused by latents being ignored. Additionally, it is the first discrete latent VAE model that get similar performance as its continuous counterparts, while offering the flexibility of discrete distributions."  
>	"Since VQ-VAE can make effective use of the latent space, it can successfully model important features that usually span many dimensions in data space (for example objects span many pixels in images, phonemes in speech, the message in a text fragment, etc.) as opposed to focusing or spending capacity on noise and imperceptible details which are often local."  
>	"When paired with a powerful prior, our samples are coherent and high quality on a wide variety of applications such as speech and video generation. We use a PixelCNN over the discrete latents for images, and a WaveNet for raw audio. Training the prior and the VQ-VAE jointly, which could strengthen our results, is left as future research."  
>	"The discrete latent space captures the important aspects of the audio, such as the content of the speech, in a very compressed symbolic representation. Because of this we can now train another WaveNet on top of these latents which can focus on modeling the long-range temporal dependencies without having to spend too much capacity on imperceptible details. With enough data one could even learn a language model directly from raw audio."  
>	"When we condition the decoder in the VQ-VAE on the speaker-id, we can extract latent codes from a speech fragment and reconstruct with a different speaker-id. The VQ-VAE never saw any aligned data during training and was always optimizing the reconstruction of the orginal waveform. These experiments suggest that the encoder has factored out speaker-specific information in the encoded representations, as they have same meaning across different voice characteristics. This behaviour arises naturally because the decoder gets the speaker-id for free so the limited bandwith of latent codes gets used for other speaker-independent, phonetic information. In the paper we show that the latent codes discovered by the VQ-VAE are actually very closely related to the human-designed alphabet of phonemes."  
>	"We show promising results on learning long term structure of environments for reinforcement learning."  
  - `post` <https://avdnoord.github.io/homepage/vqvae/> (demo)
  - `slides` <https://avdnoord.github.io/homepage/slides/SANE2017.pdf>

#### ["Variational Lossy Autoencoder"](http://arxiv.org/abs/1611.02731) Chen, Kingma, Salimans, Duan, Dhariwal, Schulman, Sutskever, Abbeel
>	"Information that can be modeled locally by decoding distribution p(x|z) without access to z will be encoded locally and only the remainder will be encoded in z.  
>	There are two ways to utilize this information:  
>	- Use explicit information placement to restrict the reception of the autoregressive model, thereby forcing the model to use the latent code z which is globally provided.  
>	- Parametrize the prior distribution with a autoregressive model showing that a type of autoregressive latent code can reduce inefficiency in Bits-Back coding."  
  - `post` <http://tsong.me/blog/lossy-vae/>

----
#### ["Grammar Variational Autoencoder"](http://arxiv.org/abs/1703.01925) Kusner, Paige, Hernandez-Lobato
>	"Generative modeling of discrete data such as arithmetic expressions and molecular structures poses significant challenges. State-of-the-art methods often produce outputs that are not valid. We make the key observation that frequently, discrete data can be represented as a parse tree from a context-free grammar. We propose a variational autoencoder which encodes and decodes directly to and from these parse trees, ensuring the generated outputs are always valid. Surprisingly, we show that not only does our model more often generate valid outputs, it also learns a more coherent latent space in which nearby points decode to similar discrete outputs. We demonstrate the effectiveness of our learned models by showing their improved performance in Bayesian optimization for symbolic regression and molecular synthesis."  
  - `video` <https://youtube.com/watch?v=XkY1z6kCY_s> (Hernandez-Lobato)
  - `video` <https://vimeo.com/238222537> (Kusner)
  - `video` <https://youtube.com/watch?v=ar4Fm1V65Fw> (Paige)
  - `notes` <https://bayesgroup.github.io/sufficient-statistics/posts/grammar-vae/> `in russian`

----
#### ["Toward Controlled Generation of Text"](http://arxiv.org/abs/1703.00955) Hu, Yang, Liang, Salakhutdinov, Xing
  - `video` <https://vimeo.com/238222247> (Hu)

#### ["Improved Variational Autoencoders for Text Modeling using Dilated Convolutions"](http://arxiv.org/abs/1702.08139) Yang, Hu, Salakhutdinov, Berg-Kirkpatrick
  - `video` <https://vimeo.com/238222483> (Hu)
  - `notes` <https://bayesgroup.github.io/sufficient-statistics/posts/improved-variational-autoencoders-for-text-modeling-using-dilated-convolutions/> `in russian`

#### ["A Hybrid Convolutional Variational Autoencoder for Text Generation"](http://arxiv.org/abs/1702.02390) Semeniuta, Severyn, Barth
  - `code` <https://github.com/stas-semeniuta/textvae>

#### ["Generating Sentences from a Continuous Space"](http://arxiv.org/abs/1511.06349) Bowman, Vilnis, Vinyals, Dai, Jozefowicz, Bengio
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1511.06349>
  - `code` <https://github.com/analvikingur/pytorch_RVAE>
  - `code` <https://github.com/cheng6076/Variational-LSTM-Autoencoder>



---
### generative models - autoregressive models

[interesting older papers](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---autoregressive-models)

[interesting recent papers - generative models](#generative-models)

----
#### ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
  `Transformer`
>	"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."  
  - `post` <https://research.googleblog.com/2017/08/transformer-novel-neural-network.html>
  - `video` <https://youtube.com/watch?v=rBCqOTEfxvg> (Kaiser)
  - `video` <https://youtube.com/watch?v=iDulhoQ2pro> (Kilcher)
  - `video` <https://youtu.be/_XRBlhzb31U?t=48m35s> (Figurnov) `in russian`
  - `audio` <https://soundcloud.com/nlp-highlights/36-attention-is-all-you-need-with-ashish-vaswani-and-jakob-uszkoreit> (Vaswani, Uszkoreit)
  - `post` <https://machinethoughts.wordpress.com/2017/09/01/deep-meaning-beyond-thought-vectors/>
  - `notes` <https://blog.tomrochette.com/machine-learning/papers/ashish-vaswani-attention-is-all-you-need>
  - `notes` <https://medium.com/@sharaf/a-paper-a-day-24-attention-is-all-you-need-26eb2da90a91>
  - `code` <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py>
  - `code` <https://colab.research.google.com/notebook#fileId=/v2/external/notebooks/t2t/hello_t2t.ipynb>
  - `code` <https://github.com/jadore801120/attention-is-all-you-need-pytorch>

----
#### ["Parallel WaveNet: Fast High-Fidelity Speech Synthesis"](https://arxiv.org/abs/1711.10433) Oord et al.
  `WaveNet`
>	"Inverse autoregressive flows represent a kind of dual formulation of deep autoregressive modelling, in which sampling can be performed in parallel, while the inference procedure required for likelihood estimation is sequential and slow. The goal of this paper is to marry the best features of both models: the efficient training of WaveNet and the efficient sampling of IAF networks. The bridge between them is a new form of neural network distillation, which we refer to as Probability Density Distillation, where a trained WaveNet model is used as a teacher for training feedforward IAF model with no significant difference in quality."  
  - <https://deepmind.com/blog/wavenet-launches-google-assistant/> (demo)
  - `post` <https://deepmind.com/blog/high-fidelity-speech-synthesis-wavenet/>

#### ["WaveNet: A Generative Model for Raw Audio"](http://arxiv.org/abs/1609.03499) Oord et al.
  `WaveNet`
  - `post` <https://deepmind.com/blog/wavenet-generative-model-raw-audio/>
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-1> (42:36) (van den Oord)
  - `video` <https://youtube.com/watch?v=leu286ciQcE> (Kalchbrenner)
  - `code` <https://github.com/tomlepaine/fast-wavenet>
  - `code` <https://github.com/ibab/tensorflow-wavenet>
  - `code` <https://github.com/basveeling/wavenet>

#### ["Neural Machine Translation in Linear Time"](http://arxiv.org/abs/1610.10099) Kalchbrenner, Espeholt, Simonyan, Oord, Graves, Kavukcuoglu
  `ByteNet`
>	"Generalizes LSTM seq2seq by preserving the resolution. Dynamic unfolding instead of attention. Linear time computation."  
>
>	"The authors apply a WaveNet-like architecture to the task of Machine Translation. Encoder (Source Network) and Decoder (Target Network) are CNNs that use Dilated Convolutions and they are stacked on top of each other. The Target Network uses Masked Convolutions to ensure that it only relies on information from the past. Crucially, the time complexity of the network is c(|S| + |T|), which is cheaper than that of the common seq2seq attention architecture (|S|*|T|). Through dilated convolutions the network has constant path lengths between [source input -> target output] and [target inputs -> target output] nodes. This allows for efficient propagation of gradients."  
  - `video` <https://youtube.com/watch?v=leu286ciQcE> (Kalchbrenner)
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/nmt-linear-time.md>
  - `code` <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/bytenet.py>

#### ["Parallel Multiscale Autoregressive Density Estimation"](http://arxiv.org/abs/1703.03664) Reed, Oord, Kalchbrenner, Colmenarejo, Wang, Belov, de Freitas
  `PixelCNN`
>	"O(log N) sampling instead of O(N)"  

#### ["PixelCNN++: A PixelCNN Implementation with Discretized Logistic Mixture Likelihood and Other Modifications"](https://arxiv.org/abs/1701.05517) Salimans, Karpathy, Chen, Kingma
  `PixelCNN`
  - `code` <https://github.com/openai/pixel-cnn>

#### ["Conditional Image Generation with PixelCNN Decoders"](http://arxiv.org/abs/1606.05328) Oord, Kalchbrenner, Vinyals, Espeholt, Graves, Kavukcuoglu
  `PixelCNN`
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-1> (27:26) (Oord)
  - `post` <http://sergeiturukin.com/2017/02/22/pixelcnn.html> + <http://sergeiturukin.com/2017/02/24/gated-pixelcnn.html>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1606.05328#shagunsodhani>
  - `code` <https://github.com/openai/pixel-cnn>
  - `code` <https://github.com/anantzoid/Conditional-PixelCNN-decoder>

#### ["Pixel Recurrent Neural Networks"](http://arxiv.org/abs/1601.06759) Oord, Kalchbrenner, Kavukcuoglu
  `PixelRNN`
  - `video` <http://techtalks.tv/talks/pixel-recurrent-neural-networks/62375/> (Oord)
  - `post` <https://github.com/tensorflow/magenta/blob/master/magenta/reviews/pixelrnn.md> (Kastner)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/OordKK16>
  - `code` <https://github.com/carpedm20/pixel-rnn-tensorflow>

----
#### ["Actively Learning What Makes a Discrete Sequence Valid"](https://arxiv.org/abs/1708.04465) Janz, Westhuizen, Hernandez-Lobato

#### ["Learning to Decode for Future Success"](http://arxiv.org/abs/1701.06549) Li, Monroe, Jurafsky

#### ["Self-critical Sequence Training for Image Captioning"](http://arxiv.org/abs/1612.00563) Rennie, Marcheret, Mroueh, Ross, Goel
>	"REINFORCE with reward normalization but without baseline estimation"  
  - `video` <https://youtube.com/watch?v=UnT5wTe13yc> (Rennie)
  - `video` <https://yadi.sk/i/-U5w4NpJ3H5TWD> (Ratnikov) `in russian`
  - `code` <https://github.com/ruotianluo/self-critical.pytorch>

#### ["Sequence Tutor: Conservative Fine-tuning of Sequence Generation Models with KL-control"](https://arxiv.org/abs/1611.02796) Jaques, Gu, Bahdanau, Hernandez-Lobato, Turner, Eck
>	"In contrast to relying solely on possibly biased data, our approach allows for encoding high-level domain knowledge into the RNN, providing a general, alternative tool for training sequence models."  
  - `post` <https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning/>
  - `video` <https://vimeo.com/240608475> (Jaques)
  - `video` <https://www.technologyreview.com/s/604010/google-brain-wants-creative-ai-to-help-humans-make-a-new-kind-of-art/> (10:45) (Eck)
  - `code` <https://github.com/tensorflow/magenta/tree/master/magenta/models/rl_tuner>

#### ["Professor Forcing: A New Algorithm for Training Recurrent Networks"](http://arxiv.org/abs/1610.09038) Lamb, Goyal, Zhang, Zhang, Courville, Bengio
>	"In professor forcing, G is simply an RNN that is trained to predict the next element in a sequence and D a discriminative bi-directional RNN. G is trained to fool D into thinking that the hidden states of G occupy the same state space at training (feeding ground truth inputs to the RNN) and inference time (feeding generated outputs as the next inputs). D, in turn, is trained to tell apart the hidden states of G at training and inference time. At the Nash equilibrium, D cannot tell apart the state spaces any better and G cannot make them any more similar. This is motivated by the problem that RNNs typically diverge to regions of the state space that were never observed during training and which are hence difficult to generalize to."  
  - `video` <https://youtube.com/watch?v=I7UFPBDLDIk>
  - `video` <http://videolectures.net/deeplearning2016_goyal_new_algorithm/> (Goyal)
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/professor-forcing.md>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1610.09038>

#### ["An Actor-Critic Algorithm for Sequence Prediction"](http://arxiv.org/abs/1607.07086) Bahdanau, Brakel, Xu, Goyal, Lowe, Pineau, Courville, Bengio

#### ["Length Bias in Encoder Decoder Models and a Case for Global Conditioning"](http://arxiv.org/abs/1606.03402) Sountsov, Sarawagi
  `eliminating beam search`

#### ["Sequence-to-Sequence Learning as Beam-Search Optimization"](http://arxiv.org/abs/1606.02960) Wiseman, Rush
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-2> (44:02) (Wiseman)
  - `video` <https://periscope.tv/hugo_larochelle/1eaKbLQXbWdJX> (31:19) (Rush)
  - `video` <https://vimeo.com/240428387#t=1h3m16s> (Jaitly)
  - `notes` <http://shortscience.org/paper?bibtexKey=journals/corr/1606.02960>
  - `notes` <https://medium.com/@sharaf/a-paper-a-day-2-sequence-to-sequence-learning-as-beam-search-optimization-92424b490350>



---
### reinforcement learning - model-free methods

[interesting older papers - value-based methods](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---value-based-methods)  
[interesting older papers - policy-based methods](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---policy-based-methods)  

----
#### ["Is the Bellman Residual a Bad Proxy?"](https://hal.archives-ouvertes.fr/hal-01629739/document) Geist, Piot, Pietquin
  `Q-learning`
>	"This paper aims at theoretically and empirically comparing two standard optimization criteria for Reinforcement Learning: 1) maximization of the mean value and 2) minimization of the Bellman residual. For that purpose, we place ourselves in the framework of policy search algorithms, that are usually designed to maximize the mean value, and derive a method that minimizes the residual ||T∗ vπ − vπ||1,ν over policies. A theoretical analysis shows how good this proxy is to policy optimization, and notably that it is better than its value-based counterpart. We also propose experiments on randomly generated generic Markov decision processes, specifically designed for studying the influence of the involved concentrability coefficient. They show that the Bellman residual is generally a bad proxy to policy optimization and that directly maximizing the mean value is more likely to result in efficient and robust reinforcement learning algorithms, despite the current lack of deep theoretical analysis. This might seem obvious, as directly addressing the problem of interest is usually better, but given the prevalence of (projected) Bellman residual minimization in value-based reinforcement learning, we believe that this question is worth to be considered."  
>	"Bellman residuals are prevalent in Approximate Dynamic Programming. Notably, value iteration minimizes such a residual using a fixed-point approach and policy iteration minimizes it with a Newton descent. On another hand, maximizing the mean value is prevalent in policy search approaches."  
>	"As Bellman residual minimization methods are naturally value-based and mean value maximization approaches policy-based, we introduced a policy-based residual minimization algorithm in order to study both optimization problems together. For the introduced residual method, we proved a proxy bound, better than value-based residual minimization. The different nature of the bounds made the comparison difficult, but both involve the same concentrability coefficient, a term often underestimated in RL bounds. Therefore, we compared both approaches empirically on a set of randomly generated Garnets, the study being designed to quantify the influence of this concentrability coefficient. From these experiments, it appears that the Bellman residual is a good proxy for the error (the distance to the optimal value function) only if, luckily, the concentrability coefficient is small for the considered MDP and the distribution of interest, or one can afford a change of measure for the optimization problem, such that the sampling distribution is close to the ideal one."  

#### ["Bayesian Policy Gradients via Alpha Divergence Dropout Inference"](https://arxiv.org/abs/1712.02037) Henderson, Doan, Islam, Meger
  `α-BNN` `Q-learning` `continuous control`
>	"Policy gradient methods have had great success in solving continuous control tasks, yet the stochastic nature of such problems makes deterministic value estimation difficult. We propose an approach which instead estimates a distribution by fitting the value function with a Bayesian Neural Network. We optimize an α-divergence objective with Bayesian dropout approximation to learn and estimate this distribution. We show that using the Monte Carlo posterior mean of the Bayesian value function distribution, rather than a deterministic network, improves stability and performance of policy gradient methods (TRPO, PPO and DDPG) in continuous control MuJoCo simulations."  
>	"We build off of work in Bayesian Neural Networks and distributional perspectives in uncertain value functions to demonstrate an extension which can successfully be leveraged in continuous control domains with dropout uncertainty estimates."  
>	"We find large improvements in Proximal Policy Optimization from using the posterior mean. We suspect that these improvements are partially due to the added exploration, aiding the on-policy and adaptive nature of PPO, during early stages of learning where the uncertainty distribution is large. This is backed by ablation analysis where we find that PPO in the Half-Cheetah environment sees larger improvements from a higher dropout rate (presumably yielding slightly more exploratory value estimates)."  
>	"We found that the Q-value estimates in DDPG were much lower than in the baseline version. We believe this is due to a variance reduction property as in Double-DQN. While in Double-DQN, two function approximators are used for the Q-value function, using dropout effectively creates a parallel to Double-DQN where many different approximators are used from the same set of network weights."  

#### ["Distributional Reinforcement Learning with Quantile Regression"](https://arxiv.org/abs/1710.10044) Dabney, Rowland, Bellemare, Munos
  `QR-DQN` `Q-learning`
>	"One of the theoretical contributions of the C51 work was a proof that the distributional Bellman operator is a contraction in a maximal form of the Wasserstein metric between probability distributions. In this context, the Wasserstein metric is particularly interesting because it does not suffer from disjoint-support issues which arise when performing Bellman updates. Unfortunately, this result does not directly lead to a practical algorithm: the Wasserstein metric, viewed as a loss, cannot generally be minimized using stochastic gradient methods. This negative result left open the question as to whether it is possible to devise an online distributional reinforcement learning algorithm which takes advantage of the contraction result. Instead, the C51 algorithm first performs a heuristic projection step, followed by the minimization of a KL divergence between projected Bellman update and prediction. The work therefore leaves a theory-practice gap in our understanding of distributional reinforcement learning, which makes it difficult to explain the good performance of C51. Thus, the existence of a distributional algorithm that operates end-to-end on the Wasserstein metric remains an open question."  
>	"In this paper, we answer this question affirmatively. By appealing to the theory of quantile regression, we show that there exists an algorithm, applicable in a stochastic approximation setting, which can perform distributional reinforcement learning over the Wasserstein metric. Our method relies on the following techniques:  
>	- We “transpose” the parametrization from C51: whereas the former uses N fixed locations for its approximation distribution and adjusts their probabilities, we assign fixed, uniform probabilities to N adjustable locations  
>	- We show that quantile regression may be used to stochastically adjust the distributions’ locations so as to minimize the Wasserstein distance to a target distribution  
>	- We formally prove contraction mapping results for our overall algorithm, and use these results to conclude that our method performs distributional RL end-to-end under the Wasserstein metric, as desired"  
  - `post` <https://mtomassoli.github.io/2017/12/08/distributional_rl/>
  - `code` <https://github.com/NervanaSystems/coach/blob/master/agents/qr_dqn_agent.py>

#### ["A Distributional Perspective on Reinforcement Learning"](https://arxiv.org/abs/1707.06887) Bellemare, Dabney, Munos
  `C51` `Categorical DQN` `Q-learning`
>	"The value function gives the expected future discounted return. This ignores variance and multi-modality. Authors argue for modelling the full distribution of the return."  
>	"Distributional Bellman equation"  
>	"It is unclear if method works because of modelling uncertainty over rewards, training network with richer signal (categorical loss) or using distributional Bellman update."  
  - `post` <https://deepmind.com/blog/going-beyond-average-reinforcement-learning/>
  - `video` <https://youtube.com/watch?v=yFBwyPuO2Vg> (demo)
  - `video` <https://vimeo.com/235922311> (Bellemare)
  - `video` <https://vimeo.com/237274251> (Bellemare)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=4m45s> (Mnih)
  - `post` <https://mtomassoli.github.io/2017/12/08/distributional_rl/>
  - `post` <https://flyyufelix.github.io/2017/10/24/distributional-bellman.html>
  - `code` <https://github.com/flyyufelix/C51-DDQN-Keras>
  - `code` <https://github.com/reinforceio/tensorforce/blob/master/tensorforce/models/categorical_dqn_model.py>
  - `code` <https://github.com/floringogianu/categorical-dqn>

#### ["Discrete Sequential Prediction of Continuous Actions for Deep RL"](https://arxiv.org/abs/1705.05035) Metz, Ibarz, Jaitly, Davidson
  `SDQN` `Q-learning`
>	"We draw inspiration from the recent success of sequence-to-sequence models for structured prediction problems to develop policies over discretized spaces. Central to this method is the realization that complex functions over high dimensional spaces can be modeled by neural networks that use next step prediction. Specifically, we show how Q-values and policies over continuous spaces can be modeled using a next step prediction model over discretized dimensions. With this parameterization, it is possible to both leverage the compositional structure of action spaces during learning, as well as compute maxima over action spaces (approximately). On a simple example task we demonstrate empirically that our method can perform global search, which effectively gets around the local optimization issues that plague DDPG and NAF. We apply the technique to off-policy (Q-learning) methods and show that our method can achieve the state-of-the-art for off-policy methods on several continuous control tasks."  

#### ["Learning to Play in a Day: Faster Deep Reinforcement Learning by Optimality Tightening"](https://arxiv.org/abs/1611.01606) He, Liu, Schwing, Peng
  `Q-learning`
>	"We propose a novel training algorithm for reinforcement learning which combines the strength of deep Q-learning with a constrained optimization approach to tighten optimality and encourage faster reward propagation."  
  - `video` <https://yadi.sk/i/yBO0q4mI3GAxYd> (1:10:20) (Fritsler) `in russian`
  - `video` <https://youtu.be/mrj_hyH974o?t=16m13s> (Podoprikhin) `in russian`

----
#### ["Natural Value Approximators: Learning when to Trust Past Estimates"](http://papers.nips.cc/paper/6807-natural-value-approximators-learning-when-to-trust-past-estimates) Xu, Modayil, Hasselt, Barreto, Silver, Schaul
  `NVA` `value-based`
>	"Neural networks are most effective for value function approximation when the desired target function is smooth. However, value functions are, by their very nature, discontinuous functions with sharp variations over time. We introduce a representation of value that matches the natural temporal structure of value functions."  
>	"A value function represents the expected sum of future discounted rewards. If non-zero rewards occur infrequently but reliably, then an accurate prediction of the cumulative discounted reward rises as such rewarding moments approach and drops immediately after. This is a pervasive scenario because many domains associate positive or negative reinforcements to salient events (like picking up an object, hitting a wall, or reaching a goal position). The problem is that the agent’s observations tend to be smooth in time, so learning an accurate value estimate near those sharp drops puts strain on the function approximator - especially when employing differentiable function approximators such as neural networks that naturally make smooth maps from observations to outputs."  
>	"We incorporate the temporal structure of cumulative discounted rewards into the value function itself. The main idea is that, by default, the value function can respect the reward sequence. If no reward is observed, then the next value smoothly matches the previous value, but becomes a little larger due to the discount. If a reward is observed, it should be subtracted out from the previous value: in other words a reward that was expected has now been consumed. The natural value approximator combines the previous value with the observed rewards and discounts, which makes this sequence of values easy to represent by a smooth function approximator such as a neural network."  

#### ["Regret Minimization for Partially Observable Deep Reinforcement Learning"](https://arxiv.org/abs/1710.11424) Jin, Levine, Keutzer
  `ARM` `value-based`
>	"Algorithm based on counterfactual regret minimization that iteratively updates an approximation to a cumulative clipped advantage function."  
>	"In contrast to prior methods, advantage-based regret minimization is well suited to partially observed or non-Markovian environments."  

#### ["Multi-step Reinforcement Learning: A Unifying Algorithm"](https://arxiv.org/abs/1703.01327) De Asis, Hernandez-Garcia, Holland, Sutton
  `Q(σ)` `value-based`
>	"Currently, there are a multitude of algorithms that can be used to perform TD control, including Sarsa, Q-learning, and Expected Sarsa. These methods are often studied in the one-step case, but they can be extended across multiple time steps to achieve better performance. Each of these algorithms is seemingly distinct, and no one dominates the others for all problems. In this paper, we study a new multi-step action-value algorithm called Q(σ) which unifies and generalizes these existing algorithms, while subsuming them as special cases. A new parameter, σ, is introduced to allow the degree of sampling performed by the algorithm at each step during its backup to be continuously varied, with Sarsa existing at one extreme (full sampling), and Expected Sarsa existing at the other (pure expectation)."  
  - `video` <https://youtube.com/watch?v=MidZJ-oCpRk> (De Asis)

#### ["Convergent Tree-Backup and Retrace with Function Approximation"](https://arxiv.org/abs/1705.09322) Touati, Bacon, Precup, Vincent
  `Retrace` `value-based` `off-policy evaluation`
>	"We show that Tree Backup and Retrace algorithms are unstable with linear function approximation, both in theory and with specific examples. We addressed these issues by formulating gradient-based versions of these algorithms which minimize the mean-square projected Bellman error. Using a saddle-point formulation, we were also able to provide convergence guarantees and characterize the convergence rate of our algorithms."  
>	"The design and analysis of off-policy algorithms using all the features of reinforcement learning, e.g. bootstrapping, multi-step updates (eligibility traces), and function approximation has been explored extensively over three decades. While off-policy learning and function approximation have been understood in isolation, their combination with multi-steps bootstrapping produces a so-called deadly triad, i.e., many algorithms in this category are unstable. A convergent approach to this triad is provided by importance sampling, which bends the behavior policy distribution onto the target one. However, as the length of the trajectories increases, the variance of importance sampling corrections tends to become very large. An alternative approach which was developed for tabular representations of the value function is the tree backup algorithm which, remarkably, does not rely on importance sampling directly. Tree Backup has recently been revisited by authors of Retrace(λ) algorithm. Both Tree Backup and Retrace(λ) were only shown to converge with a tabular value function representation, and whether they would also converge with function approximation was an open question, which we tackle in this paper."  

#### ["Safe and Efficient Off-Policy Reinforcement Learning"](http://arxiv.org/abs/1606.02647) Munos, Stepleton, Harutyunyan, Bellemare
  `Retrace` `value-based` `off-policy evaluation`
>	"Retrace(λ) is a new strategy to weight a sample for off-policy learning, it provides low-variance, safe and efficient updates."  
>	"Our goal is to design a RL algorithm with two desired properties. Firstly, to use off-policy data, which is important for exploration, when we use memory replay, or observe log-data. Secondly, to use multi-steps returns in order to propagate rewards faster and avoid accumulation of approximation/estimation errors. Both properties are crucial in deep RL. We introduce the “Retrace” algorithm, which uses multi-steps returns and can safely and efficiently utilize any off-policy data."  
>	"open issue: off policy unbiased, low variance estimators for long horizon delayed reward problems"  
>	"As a corollary, we prove the convergence of Watkins’ Q(λ), which was an open problem since 1989."  
  - `video` <https://youtu.be/WuFMrk3ZbkE?t=35m30s> (Bellemare)
  - `video` <https://youtube.com/watch?v=8hK0NnG_DhY&t=25m27s> (Brunskill)

#### ["Q(λ) with Off-Policy Corrections"](http://arxiv.org/abs/1602.04951) Harutyunyan, Bellemare, Stepleton, Munos
  `Q(λ)` `value-based` `off-policy evaluation`
>	"We propose and analyze an alternate approach to off-policy multi-step temporal difference learning, in which off-policy returns are corrected with the current Q-function in terms of rewards, rather than with the target policy in terms of transition probabilities. We prove that such approximate corrections are sufficient for off-policy convergence both in policy evaluation and control, provided certain conditions. These conditions relate the distance between the target and behavior policies, the eligibility trace parameter and the discount factor, and formalize an underlying tradeoff in off-policy TD(λ)."  
>	"Unlike traditional off-policy learning algorithms Q(λ) methods do not involve weighting returns by their policy probabilities, yet under the right conditions converge to the correct TD fixed points."  
>	"The value function assesses actions in terms of the following expected cumulative reward, and thus provides a way to directly correct immediate rewards, rather than transitions. We show in this paper that such approximate corrections can be sufficient for off-policy convergence, subject to a tradeoff condition between the eligibility trace parameter and the distance between the target and behavior policies. The two extremes of this tradeoff are one-step Q-learning, and on-policy learning. Formalizing the continuum of the tradeoff is one of the main insights of this paper."  
>	"In control, Q*(λ) is in fact identical to Watkins’s Q(λ), except it does not cut the eligiblity trace at off-policy actions."  
  - `video` <https://youtube.com/watch?v=8hK0NnG_DhY&t=25m27s> (Brunskill)

#### ["Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning"](http://arxiv.org/abs/1604.00923) Thomas, Brunskill
  `value-based` `off-policy evaluation`
  - `video` <https://youtube.com/watch?v=8hK0NnG_DhY&t=15m44s> (Brunskill)

#### ["Taming the Noise in Reinforcement Learning via Soft Updates"](https://arxiv.org/abs/1512.08562) Fox, Pakman, Tishby
  `G-learning` `value-based`
>	"Model-free reinforcement learning algorithms, such as Q-learning, perform poorly in the early stages of learning in noisy environments, because much effort is spent unlearning biased estimates of the state-action value function. The bias results from selecting, among several noisy estimates, the apparent optimum, which may actually be suboptimal. We propose G-learning, a new off-policy learning algorithm that regularizes the value estimates by penalizing deterministic policies in the beginning of the learning process. We show that this method reduces the bias of the value-function estimation, leading to faster convergence to the optimal value and the optimal policy. The stochastic nature of G-learning also makes it avoid some exploration costs, a property usually attributed only to on-policy algorithms."  
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Taming_the_Noise_in_Reinforcement_Learning_via_Soft_Updates.md>
  - `code` <https://github.com/noahgolmant/simpledgn>

----
#### ["Equivalence Between Policy Gradients and Soft Q-Learning"](https://arxiv.org/abs/1704.06440) Schulman, Chen, Abbeel
  `soft Q-learning` `policy gradient`
>	"Q-learning methods can be effective and sample-efficient when they work, however, it is not well-understood why they work, since empirically, the Q-values they estimate are very inaccurate. A partial explanation may be that Q-learning methods are secretly implementing policy gradient updates. We show that there is a precise equivalence between Q-learning and policy gradient methods in the setting of entropy-regularized reinforcement learning, that "soft" (entropy-regularized) Q-learning is exactly equivalent to a policy gradient method."  
  - `video` <https://vimeo.com/240428644#t=1h16m18s> (Levine)
  - `post` <http://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/>

#### ["Reinforcement Learning with Deep Energy-Based Policies"](https://arxiv.org/abs/1702.08165) Haarnoja, Tang, Abbeel, Levine
  `soft Q-learning` `policy gradient`
>	"parameterise and estimate a so called soft Q-function directly, implicitly inducing a maximum entropy policy"  
  - <https://sites.google.com/view/softqlearning/home> (demo)
  - `video` <https://vimeo.com/240428644#t=1h16m18s> (Levine)
  - `post` <http://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/>
  - `code` <https://github.com/haarnoja/softqlearning>

#### ["Bridging the Gap Between Value and Policy Reinforcement Learning"](http://arxiv.org/abs/1702.08892) Nachum, Norouzi, Xu, Schuurmans
  `PCL` `soft Q-learning` `policy gradient`
>	"Softmax temporally consistent action values satisfy a strong consistency property with optimal entropy regularized policy probabilities along any action sequence, regardless of provenance."  
>	"Path Consistency Learning minimizes inconsistency measured along multi-step action sequences extracted from both on- and off-policy traces."  
>	"We show how a single model can be used to represent both a policy and its softmax action values. Beyond eliminating the need for a separate critic, the unification demonstrates how policy gradients can be stabilized via self-bootstrapping from both on- and off-policy data. An experimental evaluation demonstrates that both algorithms can significantly outperform strong actor-critic and Q-learning baselines across several benchmark tasks."  
  - `video` <https://youtu.be/fZNyHoXgV7M?t=1h16m17s> (Norouzi)
  - `notes` <https://github.com/ethancaballero/paper-notes/blob/master/Bridging%20the%20Gap%20Between%20Value%20and%20Policy%20Based%20Reinforcement%20Learning.md>
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Bridging_the_Gap_Between_Value_and_Policy_Based_Reinforcement_Learning.md>
  - `code` <https://github.com/tensorflow/models/tree/master/research/pcl_rl>
  - `code` <https://github.com/rarilurelo/pcl_keras>
  - `code` <https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/pcl.py>

#### ["Combining policy gradient and Q-learning"](http://arxiv.org/abs/1611.01626) O'Donoghue, Munos, Kavukcuoglu, Mnih
  `PGQ` `Q-learning` `policy gradient`
>	"We establish an equivalency between action-value fitting techniques and actor-critic algorithms, showing that regularized policy gradient techniques can be interpreted as advantage function learning algorithms."  
>	"This connection allows us to estimate the Q-values from the action preferences of the policy, to which we apply Q-learning updates."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/ODonoghueMKM16>
  - `code` <https://github.com/Fritz449/Asynchronous-RL-agent>
  - `code` <https://github.com/abhishm/PGQ>

----
#### ["Expected Policy Gradients"](https://arxiv.org/abs/1706.05374) Ciosek, Whiteson
  `EPG` `policy gradient` `on-policy`
>	"EPG unify stochastic policy gradients (SPG) and deterministic policy gradients (DPG) for reinforcement learning. Inspired by expected SARSA, EPG integrates across the action when estimating the gradient, instead of relying only on the action selected during the sampled trajectory. We establish a new general policy gradient theorem, of which the stochastic and deterministic policy gradient theorems are special cases."  
>	"We also prove that EPG reduces the variance of the gradient estimates without requiring deterministic policies and, for the Gaussian case, with no computational overhead. When the policy is Gaussian, we can now reinterpret deterministic policy gradients as an on-policy method: the deterministic policy of the original formulation is just the result of analytically integrating across actions in the Gaussian policy."  
>	"Both SPG and DPG approaches have significant shortcomings. For SPG, variance in the gradient estimates means that many trajectories are usually needed for learning. Since gathering trajectories is typically expensive, there is a great need for more sample efficient methods. DPG’s use of deterministic policies mitigates the problem of variance in the gradient but raises other difficulties. The theoretical support for DPG is limited since it assumes a critic that approximates ∇aQ when in practice it approximates Q instead. In addition, DPG learns off-policy, which is undesirable when we want learning to take the cost of exploration into account. More importantly, learning off-policy necessitates designing a suitable exploration policy, which is difficult in practice. In fact, efficient exploration in DPG is an open problem and most applications simply use independent Gaussian noise or the Ornstein-Uhlenbeck heuristic."  
>	"EPG also enables a practical contribution. Under certain conditions, we get an analytical expression for the covariance of the Gaussian that leads to a principled directed exploration strategy for continuous problems. We show that it is optimal in a certain sense to explore with a Gaussian policy such that the covariance is proportional to exp(H), where H is the scaled Hessian of the critic with respect to the actions. We present empirical results confirming that this new approach to exploration substantially outperforms DPG with Ornstein-Uhlenbeck exploration in four challenging MuJoCo domains."  

#### ["Scalable Trust-region Method for Deep Reinforcement Learning using Kronecker-factored Approximation"](https://arxiv.org/abs/1708.05144) Wu, Mansimov, Liao, Grosse, Ba
  `ACKTR` `policy gradient` `on-policy`
>	"KFAC for simple networks is equivalent to gradient descent where activation and backprop values are whitened."  
  - `video` <https://youtube.com/watch?v=0rrffaYuUi4> (Wu)
  - `video` ["Optimizing Neural Networks using Structured Probabilistic Models of the Gradient Computation"](https://fields.utoronto.ca/video-archive/2017/02/2267-16498) (Grosse)
  - `video` ["Optimizing NN using Kronecker-factored Approximate Curvature"](https://youtube.com/watch?v=FLV-MLPt3sU) (Kropotov)
  - `slides` <https://csc2541-f17.github.io/slides/lec10.pdf#page=55> (Grosse)
  - `post` <https://medium.com/@yaroslavvb/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0>
  - `code` <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr>

#### ["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347) Schulman, Wolski, Dhariwal, Radford, Klimov
  `PPO` `policy gradient` `on-policy`
>	"PPO alternates between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates."
  - `post` <https://blog.openai.com/openai-baselines-ppo/> (demo)
  - `post` <https://learningai.io/projects/2017/07/28/ai-gym-workout.html>
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Proximal_Policy_Optimization_Algorithms.md>
  - `code` <https://github.com/openai/baselines/tree/master/baselines/pposgd>
  - `code` <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr>
  - `code` <https://github.com/ShangtongZhang/DeepRL>

#### ["Emergence of Locomotion Behaviours in Rich Environments"](https://arxiv.org/abs/1707.02286) Heess et al.
  `DPPO` `policy gradient` `on-policy`
>	"parallelized Proximal Policy Optimization"  
  - `video` <https://youtube.com/watch?v=hx_bgoTF7bs> (demo)
  - `video` <https://vimeo.com/238221551#t=42m48s> (Hadsell)
  - `code` <https://github.com/ShangtongZhang/DeepRL>
  - `code` <https://github.com/alexis-jacq/Pytorch-DPPO>

#### ["Evolution Strategies as a Scalable Alternative to Reinforcement Learning"](http://arxiv.org/abs/1703.03864) Salimans, Ho, Chen, Sidor, Sutskever
  `ES` `policy gradient` `on-policy`
>	(Karpathy) "ES is much simpler than RL, and there's no need for backprop, it's highly parallelizable, has fewer hyperparams, needs no value functions."  
>	"In our preliminary experiments we found that using ES to estimate the gradient on the MNIST digit recognition task can be as much as 1,000 times slower than using backpropagation. It is only in RL settings, where one has to estimate the gradient of the expected reward by sampling, where ES becomes competitive."  
  - `post` <https://blog.openai.com/evolution-strategies/>
  - `video` <https://youtube.com/watch?v=SQtOI9jsrJ0> (Chen) `video`
  - `video` <https://youtube.com/watch?v=Rd0UdJFYkqI> (Temirchev) `in russian`
  - `post` <http://inference.vc/evolutionary-strategies-embarrassingly-parallelizable-optimization/> (Huszar)
  - `post` <http://inference.vc/evolution-strategies-variational-optimisation-and-natural-es-2/> (Huszar)
  - `post` <http://davidbarber.github.io/blog/2017/04/03/variational-optimisation/> (Barber)
  - `post` <http://argmin.net/2017/04/03/evolution/> (Recht)
  - `paper` ["Random Gradient-Free Minimization of Convex Functions"](https://mipt.ru/dcam/students/elective/a_5gc1te/RandomGradFree.PDF) by Nesterov
  - `paper` ["Natural Evolution Strategies"](http://jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) by Wierstra et al.
  - `code` <https://github.com/openai/evolution-strategies-starter>
  - `codd` <https://github.com/atgambardella/pytorch-es>

----
#### ["Maximum a Posteriori Policy Optimisation"](https://openreview.net/forum?id=S1ANxQW0b)
  `MPO` `policy gradient` `on-policy + off-policy`
>	"To derive our algorithm, we take advantage of the duality between control and estimation by using Expectation Maximization, a powerful tool from the probabilistic estimation toolbox, in order to solve control problems. This duality can be understood as replacing the question “what are the actions which maximise future rewards?” with the question “assuming future success in maximising rewards, what are the actions most likely to have been taken?”. By using this estimation objective we have more control over the policy change in both E and M steps yielding robust learning. We show that several algorithms, including TRPO, can be directly related to this perspective."  
>	"We leverage the fast convergence properties of EM-style coordinate ascent by alternating a non-parametric data-based E-step which re-weights state-action samples, with a supervised, parametric M-step using deep neural networks. This process is stable enough to allow us to use full covariance matrices, rather than just diagonal, in our policies."  
>	"The derivation of our algorithm starts from the classical connection between RL and probabilistic inference. Rather than estimating a single optimal trajectory methods in this space attempt to identify a distribution of plausible solutions by trading off expected return against the (relative) entropy of this distribution. Building on classical work such as Dayan & Hinton (1997) we cast policy search as a particular instance of the class of Expectation Maximization algorithms. Our algorithm then combines properties of existing approaches in this family with properties of recent off-policy algorithms for neural networks."  
>	"Specifically we propose an alternating optimization scheme that leads us to a novel, off-policy algorithm that is (a) data efficient; (b) robust and effectively hyper-parameter free; (c) applicable to complex control problems that have so far been considered outside the realm of off-policy algorithms; (d) allows for effective parallelisation."  
>	"Our algorithm separates policy learning into two alternating phases which we refer to as E and M step in reference to the EM algorithm:  
>	- E-step: In the E-step we obtain an estimate of the distribution of return-weighted trajectories. We perform this step by re-weighting state action samples using a learned value function. This is akin to posterior inference step when estimating the parameters of a probabilistic models with latent variables.  
>	- M-step: In the M-step we update the parametric policy in a supervised learning step using the reweighted state-action samples from the E-step as targets. This corresponds to the update of the model parameters given the complete data log-likelihood when performing EM for a probabilistic model."  
>	"These choices lead to the following desirable properties: (a) low-variance estimates of the expected return via function approximation; (b) low-sample complexity of value function estimate via robust off-policy learning; (c) minimal parametric assumption about the form of the trajectory distribution in the E-step; (d) policy updates via supervised learning in the M step; (e) robust updates via hard trust-region constraints in both the E and the M step."  
  - `video` <http://dropbox.com/s/pgcmjst7t0zwm4y/MPO.mp4> (demo)

#### ["The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously"](https://arxiv.org/abs/1707.03300) Cabi, Colmenarejo, Hoffman, Denil, Wang, de Freitas
  `IU` `policy gradient` `on-policy + off-policy`
>	"We hypothesize that a single stream of experience offers agents the opportunity to learn and perfect many policies both on purpose and incidentally, thus accelerating the acquisition of grounded knowledge. To investigate this hypothesis, we propose a deep actor-critic architecture, trained with DDPG, for learning several policies concurrently. The architecture enables the agent to attend to one task on-policy, while unintentionally learning to solve many other tasks off-policy. Importantly, the policies learned unintentionally can be harnessed for intentional use even if those policies were never followed before."  
>	"More precisely, this intentional-unintentional architecture consists of two neural networks. The actor neural network has multiple-heads representing different policies with shared lower-level representations. The critic network represents several state-action value functions, sharing a common representation for the observations."  
>	"Our experiments demonstrate that when acting according to the policy associated with one of the hardest tasks, we are able to learn all other tasks off-policy. The results for the playroom domain also showed that by increasing the number of tasks, all actors and critics learn faster. In fact, in some settings, learning with many goals was essential to solve hard many-body control tasks."  
  - `video` <https://youtu.be/xfyK03MEZ9Q?t=7h20m39s> (Cabi)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/> (20:40) (de Freitas)

#### ["Trust-PCL: An Off-Policy Trust Region Method for Continuous Control"](https://arxiv.org/abs/1707.01891) Nachum, Norouzi, Xu, Schuurmans
  `Trust-PCL` `policy gradient` `on-policy + off-policy`
>	"off-policy TRPO"  
>	"Optimal relative-entropy regularized policy satisfies path consistencies relating state values at ends of path to log-probabilities of actions along path. Trust-PCL implicitly optimizes trust region constraint using off-policy data by minimizing inconsistencies with maintained lagged policy along paths sampled from replay buffer."  
  - `video` <https://youtu.be/xfyK03MEZ9Q?t=7h38m11s> (Nachum)
  - `code` <https://github.com/tensorflow/models/tree/master/research/pcl_rl>

#### ["Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning"](https://arxiv.org/abs/1706.00387) Gu, Lillicrap, Ghahramani, Turner, Scholkopf, Levine
  `IPG` `policy gradient` `on-policy + off-policy`
>	"REINFORCE, TRPO, Q-Prop, DDPG, SVG(0), PGQ, ACER are special limiting cases of IPG."  

#### ["Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic"](http://arxiv.org/abs/1611.02247) Gu, Lillicrap, Ghahramani, Turner, Levine
  `Q-Prop` `policy gradient` `on-policy + off-policy`
>	"Batch policy gradient methods offer stable learning, but at the cost of high variance, which often requires large batches. TD-style methods, such as off-policy actor-critic and Q-learning, are more sample-efficient but biased, and often require costly hyperparameter sweeps to stabilize. In this work, we aim to develop methods that combine the stability of policy gradients with the efficiency of off-policy RL."  

>	"A policy gradient method that uses a Taylor expansion of the off-policy critic as a control variate. Q-Prop is both sample efficient and stable, and effectively combines the benefits of on-policy and off-policy methods."  
>	"- unbiased gradient  
>	 - combine PG and AC gradients  
>	 - learns critic from off-policy data  
>	 - learns policy from on-policy data"  
>	"Q-Prop works with smaller batch size than TRPO-GAE  
>	Q-Prop is significantly more sample-efficient than TRPO-GAE"  
>	"policy gradient algorithm that is as fast as value estimation"  
>	"take off-policy algorithm and correct it with on-policy algorithm on residuals"  
>	"can be understood as REINFORCE with state-action-dependent baseline with bias correction term instead of unbiased state-dependent baseline"  
  - `video` <https://facebook.com/iclr.cc/videos/1712224178806641/> (1:36:47) (Gu)
  - `video` <https://youtu.be/M6nfipCxQBc?t=16m11s> (Lillicrap)
  - `notes` <http://www.alexirpan.com/rl-derivations/#q-prop>
  - `code` <https://github.com/shaneshixiang/rllabplusplus>

#### ["The Reactor: A Sample-Efficient Actor-Critic Architecture"](https://arxiv.org/abs/1704.04651) Gruslys, Azar, Bellemare, Munos
  `Reactor` `policy gradient` `on-policy + off-policy`
>	"Retrace-actor"  
>	"The deep recurrent neural network outputs a target policy π (the actor), an action-value Q-function (the critic) evaluating the current policy π, and an estimated behavioural policy µˆ which we use for off-policy correction. The agent maintains a memory buffer filled with past experiences. The critic is trained by the multi-step off-policy Retrace algorithm and the actor is trained by a novel β-leave-one-out policy gradient estimate (which uses both the off-policy corrected return and the estimated Q-function)."  

#### ["Sample Efficient Actor-Critic with Experience Replay"](http://arxiv.org/abs/1611.01224) Wang, Bapst, Heess, Mnih, Munos, Kavukcuoglu, de Freitas
  `ACER` `policy gradient` `on-policy + off-policy`
>	"Experience replay is a valuable tool for improving sample efficiency and state-of-the-art deep Q-learning methods have been up to this point the most sample efficient techniques on Atari by a significant margin. However, we need to do better than deep Q-learning, because it has two important limitations. First, the deterministic nature of the optimal policy limits its use in adversarial domains. Second, finding the greedy action with respect to the Q function is costly for large action spaces."  
>	"Policy gradient methods are restricted to continuous domains or to very specific tasks such as playing Go. The existing variants applicable to both continuous and discrete domains, such as the on-policy A3C, are sample inefficient."  
>	"ACER capitalizes on recent advances in deep neural networks, variance reduction techniques, the off-policy Retrace algorithm and parallel training of RL agents. Yet, crucially, its success hinges on innovations advanced in this paper: truncated importance sampling with bias correction, stochastic dueling network architectures, and efficient trust region policy optimization."  
>	"On the theoretical front, the paper proves that the Retrace operator can be rewritten from our proposed truncated importance sampling with bias correction technique."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2FWangBHMMKF16>
  - `code` <https://github.com/openai/baselines/tree/master/baselines/acer>
  - `code` <https://github.com/hercky/ACER_tf>
  - `code` <https://github.com/Kaixhin/ACER>

----
#### ["RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning"](http://arxiv.org/abs/1611.02779) Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel
  `RL^2` `meta-learning`
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#rl2-fast-reinforcement-learning-via-slow-reinforcement-learning--duan-schulman-chen-bartlett-sutskever-abbeel>

#### ["Learning to Reinforcement Learn"](http://arxiv.org/abs/1611.05763) Wang et al.
  `meta-learning`
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#learning-to-reinforcement-learn-wang-et-al>



---
### reinforcement learning - model-based methods

[interesting older papers](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---model-based-methods)

----
#### ["Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"](https://arxiv.org/abs/1708.02596) Nagabandi, Kahn, Fearing, Levine
  `planning` `using available environment model`
>	"Calculating optimal plan is difficult due to the dynamics and reward functions being nonlinear, but many techniques exist for obtaining approximate solutions to finite-horizon control problems that are sufficient for succeeding at the desired task. We use a simple random-sampling shooting method in which K candidate action sequences are randomly generated, the corresponding state sequences are predicted using the learned dynamics model, the rewards for all sequences are calculated, and the candidate action sequence with the highest expected cumulative reward is chosen. Rather than have the policy execute this action sequence in open-loop, we use model predictive control: the policy executes only the first action, receives updated state information, and recalculates the optimal action sequence at the next time step. This combination of predictive dynamics model plus controller is beneficial in that the model is trained only once, but by simply changing the reward function, we can accomplish a variety of goals at run-time, without a need for live task-specific retraining."  
  - <https://sites.google.com/view/mbmf> (demo)
  - `post` <http://bair.berkeley.edu/blog/2017/11/30/model-based-rl/>
  - `code` <https://github.com/nagaban2/nn_dynamics>

#### ["Model-Based Planning in Discrete Action Spaces"](https://arxiv.org/abs/1705.07177) Henaff, Whitney, LeCun
  `planning` `using available environment model`
>	"We show that by using a simple paramaterization of actions on the simplex combined with input noise during planning, we are able to effectively perform gradient-based planning in discrete action spaces."  
#### ["Counterfactual Control for Free from Generative Models"](https://arxiv.org/abs/1702.06676) Guttenberg, Yu, Kanai
  `planning` `using available environment model`
>	"generative model learning the joint distribution between actions and future states can be used to automatically infer a control scheme for any desired reward function, which may be altered on the fly without retraining the model"  
>	"problem of action selection is reduced to one of gradient descent on the latent space of the generative model, with the model itself providing the means of evaluating outcomes and finding the gradient, much like how the reward network in Deep Q-Networks provides gradient information for the action generator"  

#### ["Blazing the Trails before Beating the Path: Sample-efficient Monte-Carlo Planning"](https://papers.nips.cc/paper/6253-blazing-the-trails-before-beating-the-path-sample-efficient-monte-carlo-planning.pdf) Grill, Valko, Munos
  `planning` `using available environment model`
>	"We study the sampling-based planning problem in Markov decision processes (MDPs) that we can access only through a generative model, usually referred to as Monte-Carlo planning."  
>	"Our objective is to return a good estimate of the optimal value function at any state while minimizing the number of calls to the generative model, i.e. the sample complexity."  
>	"TrailBlazer is an adaptive algorithm that exploits possible structures of the MDP by exploring only a subset of states reachable by following near-optimal policies."  
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Blazing-the-trails-before-beating-the-path-Sample-efficient-Monte-Carlo-planning> (Grill)

----
#### ["Learning Generalized Reactive Policies using Deep Neural Networks"](https://arxiv.org/abs/1708.07280) Groshev, Tamar, Srivastava, Abbeel
  `learning to guide planning` `using available environment model` `search-based policy iteration`
>	"learning a reactive policy that imitates execution traces produced by a planner"  
>	"We consider the problem of learning for planning, where knowledge acquired while planning is reused to plan faster in new problem instances. For robotic tasks, among others, plan execution can be captured as a sequence of visual images."  
>	"We investigate architectural properties of deep networks that are suitable for learning long-horizon planning behavior, and explore how to learn, in addition to the policy, a heuristic function that can be used with classical planners or search algorithms such as A*."  
>	"Results on the challenging Sokoban domain suggest that DNNs have the capability to extract powerful features from observations, and the potential to learn the type of ‘visual thinking’ that makes some planning problems easy to humans but very hard for automatic planners."  
  - <https://sites.google.com/site/learn2plannips/> (demo)

#### ["Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"](https://arxiv.org/abs/1712.01815) Silver et al.
  `learning to guide planning` `using available environment model` `search-based policy iteration`
>	"The game of chess is the most widely-studied domain in the history of artificial intelligence. The strongest programs are based on a combination of sophisticated search techniques, domain-specific adaptations, and handcrafted evaluation functions that have been refined by human experts over several decades. In contrast, the AlphaGo Zero program recently achieved superhuman performance in the game of Go, by tabula rasa reinforcement learning from games of self-play. In this paper, we generalise this approach into a single AlphaZero algorithm that can achieve, tabula rasa, superhuman performance in many challenging domains. Starting from random play, and given no domain knowledge except the game rules, AlphaZero achieved within 24 hours a superhuman level of play in the games of chess and shogi (Japanese chess) as well as Go, and convincingly defeated a world-champion program in each case."  
>	"no opening book, no endgame database, no heuristics, no anything"  
>	"One reason why MCTS is so effective compared to Alpha-Beta when you start to use function approximators is that neural network will inevitably have approximation errors. Alpha-Beta search is kind of minimax search and is like glorified big max operator alternating with mins, which will pick out biggest errors in function approximation and propagate it to the root of search tree. Whilst MCTS is averaging over evaluations which tends to cancel out errors in search and can be more effective because of that."  
>	"AlphaGo Zero tuned the hyper-parameter of its search by Bayesian optimization. In AlphaZero they reuse the same hyper-parameters for all games without game-specific tuning. The sole exception is the noise that is added to the prior policy to ensure exploration; this is scaled in proportion to the typical number of legal moves for that game type. Like AlphaGo Zero, the board state is encoded by spatial planes based only on the basic rules for each game. The actions are encoded by either spatial planes or a flat vector, again based only on the basic rules for each game. They applied the AlphaZero algorithm to chess, shogi, and also Go. Unless otherwise specified, the same algorithm settings, network architecture, and hyper-parameters were used for all three games."  
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#games> (demo)
  - `video` <https://youtu.be/A3ekFcZ3KNw?t=23m28s> (Silver)

#### ["Mastering the Game of Go without Human Knowledge"](https://deepmind.com/documents/119/agz_unformatted_nature.pdf) Silver et al.
  `learning to guide planning` `using available environment model` `search-based policy iteration`
>	"AlphaGo Zero learns two functions (which take as input the current board):  
>	- A prior over moves p is trained to predict what AlphaGo will eventually decide to do  
>	- A value function v is trained to predict which player will win (if AlphaGo plays both sides)  
>	Both are trained with supervised learning. Once we have these two functions, AlphaGo actually picks its moves by using 1600 steps of Monte Carlo tree search (MCTS), using p and v to guide the search. It trains p to bypass this expensive search process and directly pick good moves. As p improves, the expensive search becomes more powerful, and p chases this moving target."  
>
>	"AlphaGo Zero uses a quite different approach to deep RL than typical (model-free) algorithms such as policy gradient or Q-learning. By using AlphaGo search we massively improve the policy and self-play outcomes - and then we apply simple, gradient based updates to train the next policy + value network. This appears to be much more stable than incremental, gradient-based policy improvements that can potentially forget previous improvements."  
>	"We chose to focus more on reinforcement learning, as we believed it would ultimately take us beyond human knowledge. Our recent results actually show that a supervised-only approach can achieve a surprisingly high performance - but that reinforcement learning was absolutely key to progressing far beyond human levels."  
>
>	"AlphaGo improves the policy through REINFORCE, which is highly sample-inefficient. Then, it learns the value function for that policy. In REINFORCE one generates trajectories and then changes their probability based on the outcome of the match.  
>	AlphaGo Zero, on the other hand, changes the trajectories themselves. During self-play, an expert (MCTS) tells the policy-value network how to improve its policy-part right away. Moreover, the improved move is the one that's played, so, in the end, the outcome will be based on the improved policy. Therefore we're basically doing Generalized Policy Iteration because we're greedily improving the policy as we go and learning the value of this improved policy."  
>
>	"Differences from AlphaGo:  
>	- No human data. Learns solely by self-play reinforcement learning, starting from random.  
>	- No human features. Only takes raw board as input.  
>	- Single neural network. Policy and value networks are combined into one neural network (resnet).  
>	- Simpler search. No randomised Monte-Carlo rollouts, only uses neural network to evaluate."  
  - `post` <https://deepmind.com/blog/alphago-zero-learning-scratch/>
  - `post` <http://inference.vc/alphago-zero-policy-improvement-and-vector-fields/>
  - `post` <http://tim.hibal.org/blog/alpha-zero-how-and-why-it-works/>
  - `post` <https://reddit.com/r/MachineLearning/comments/76xjb5/ama_we_are_david_silver_and_julian_schrittwieser/dolnq31/> (Anthony)
  - `video` <https://youtu.be/A3ekFcZ3KNw?t=10m50s> (Silver)
  - `video` <https://youtube.com/watch?v=6fKG4wJ7uBk> (Baudis)
  - `video` <https://youtube.com/watch?v=XuzIqE2IshY> (Kington)
  - `video` <https://youtube.com/watch?v=vC66XFoN4DE> (Raval)
  - `notes` <https://blog.acolyer.org/2017/11/17/mastering-the-game-of-go-without-human-knowledge/>
  - `notes` <https://dropbox.com/s/fuwhivftv998f6q/AlphaGoZeroPseudoCode.pdf>
  - `code` <https://github.com/gcp/leela-zero/>

#### ["Thinking Fast and Slow with Deep Learning and Tree Search"](https://arxiv.org/abs/1705.08439) Anthony, Tian, Barber
  `learning to guide planning` `using available environment model` `search-based policy iteration` `Expert Iteration`
>	"Planning new policies is performed by tree search, while a deep neural network generalises those plans"  
>	"Expert Iteration (ExIt) can be viewed as an extension of Imitation Learning methods to domains where the best known experts are unable to achieve satisfactory performance. In standard IL an apprentice is trained to imitate the behaviour of an expert. In ExIt, we extend this to an iterative learning process. Between each iteration, we perform an Expert Improvement step, where we bootstrap the (fast) apprentice policy to increase the performance of the (comparatively slow) expert."  
>	"Imitation Learning is generally appreciated to be easier than Reinforcement Learning, and this partly explains why ExIt is more successful than model-free methods like REINFORCE. Furthermore, for MCTS to recommend a move, it must be unable to find any weakness with its search. Effectively, therefore, a move played by MCTS is good against a large selection of possible opponents. In contrast, in regular self play (in which the opponent move is made by the network playing as the opposite colour), moves are recommended if they beat only this single opponent under consideration. This is, we believe, a key insight into why ExIt works well (when using MCTS as the expert) - the apprentice effectively learns to play well against many opponents."  
  - `post` <https://davidbarber.github.io/blog/2017/11/07/Learning-From-Scratch-by-Thinking-Fast-and-Slow-with-Deep-Learning-and-Tree-Search/> (Barber)
  - `post` <https://reddit.com/r/MachineLearning/comments/76xjb5/ama_we_are_david_silver_and_julian_schrittwieser/dolnq31/> (Anthony)

#### ["DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker"](http://arxiv.org/abs/1701.01724) Moravcik et al.
  `learning to guide planning` `using available environment model`
  - `paper` <http://science.sciencemag.org/content/early/2017/03/01/science.aam6960>
  - `video` <https://youtube.com/playlist?list=PLX7NnbJAq7PlA2XpynViLOigzWtmr6QVZ> (demo matches)
  - <http://deepstack.ai>
  - <http://twitter.com/DeepStackAI>
  - `video` <https://youtu.be/02xIkHowQOk?t=11m45s> (Bowling)
  - `video` <https://youtube.com/watch?v=qndXrHcV1sM> (Bowling)
  - `video` <http://videolectures.net/aaai2017_bowling_sandholm_poker> (Bowling, Sandholm)
  - `code` <https://github.com/lifrordi/DeepStack-Leduc>

----
#### ["Learning to Search with MCTSnets"](https://openreview.net/forum?id=r1TA9ZbA-)
  `learning to plan` `using available environment model`
>	"Planning problems are most typically solved by tree search algorithms that simulate ahead into the future, evaluate future states, and back-up those evaluations to the root of a search tree. Among these algorithms, Monte-Carlo tree search is one of the most general, powerful and widely used. A typical implementation of MCTS uses cleverly designed rules, optimised to the particular characteristics of the domain. These rules control where the simulation traverses, what to evaluate in the states that are reached, and how to back-up those evaluations. In this paper we instead learn where, what and how to search. Our architecture, which we call an MCTSnet, incorporates simulation-based search inside a neural network, by expanding, evaluating and backing-up a vector embedding. The parameters of the network are trained end-to-end using gradient-based optimisation. When applied to small searches in the well-known planning problem Sokoban, the learned search algorithm significantly outperformed MCTS baselines."  
>	"Although we have focused on a supervised learning setup, our approach could easily be extended to a reinforcement learning setup by leveraging policy iteration with MCTS. We have focused on small searches, more similar in scale to the plans that are processed by the human brain, than to the massive-scale searches in high-performance games or planning applications. In fact, our learned search performed better than a standard MCTS with more than an order-of-magnitude more computation, suggesting that neural approaches to search may ultimately replace their handcrafted counterparts."  
>	"We present a neural network architecture that includes the same processing stages as a typical MCTS, but inside the neural network itself, as a dynamic computational graph. The key idea is to represent the internal state of the search, at each node, by a memory vector. The computation of the network proceeds forwards from the root state, just like a simulation of MCTS, using a simulation policy based on the memory vector to select the trajectory to traverse. The leaf state is then processed by an embedding network to initialize the memory vector at the leaf. The network proceeds backwards up the trajectory, updating the memory at each visited state according to a backup network that propagates from child to parent. Finally, the root memory vector is used to compute an overall prediction of value or action."  
>	"The major benefit of our planning architecture, compared to more traditional planning algorithms, is that it can be exposed to gradient-based optimisation. This allows us to replace every component of MCTS with a richer, learnable equivalent - while maintaining the desirable structural properties of MCTS such as the use of a model, iterative local computations, and structured memory. We jointly train the parameters of the evaluation network, backup network and simulation policy so as to optimise the overall predictions of the MCTS network. The majority of the network is fully differentiable, allowing for efficient training by gradient descent. Still, internal action sequences directing the control flow of the network cannot be differentiated, and learning this internal policy presents a challenging credit assignment problem. To address this, we propose a novel, generally applicable approximate scheme for credit assignment that leverages the anytime property of our computational graph, allowing us to also effectively learn this part of the search network from data."  

#### ["Learning Dynamic State Abstractions for Model-based Reinforcement Learning"](https://openreview.net/forum?id=HJw8fAgA-)
  `learning to plan` `using available environment model` `I2A`
>	"We have shown that state-space models directly learned from raw pixel observations are good candidates for model-based RL: 1) they are powerful enough to capture complex environment dynamics, exhibiting similar accuracy to frame-auto-regressive models; 2) they allow for computationally efficient Monte-Carlo rollouts; 3) their learned dynamic state-representations are excellent features for evaluating and anticipating future outcomes compared to raw pixels. This enabled Imagination Augemented Agents to outperform strong model-free baselines on MS PACMAN."  
>	"On a conceptual level, we present (to the best of our knowledge) the first results on what we termed learning-to-query. We show learning a rollout policy by backpropagating policy gradients leads to consistent (if modest) improvements."  
>	"We address, the Imagination-Augmented Agent framework, the main challenge posed by model-based RL: training accurate, computationally efficient models on more complex domains and using them with agents. First, we consider computationally efficient state-space environment models that make predictions at a higher level of abstraction, both spatially and temporally, than at the level of raw pixel observations. Such models substantially reduce the amount of computation required to perform rollouts, as future states can be represented much more compactly. Second, in order to increase model accuracy, we examine the benefits of explicitly modeling uncertainty in the transitions between these abstract states. Finally, we explore different strategies of learning rollout policies that define the interface between agent and environment model: We consider the possibility of learning to query the internal model, for guiding the Monte-Carlo rollouts of the model towards informative outcomes."  
>	"Here, we adopted the I2A assumption of having access to a pre-trained envronment model. In future work, we plan to drop this assumption and jointly learn the model and the agent."  

#### ["Imagination-Augmented Agents for Deep Reinforcement Learning"](https://arxiv.org/abs/1707.06203) Weber et al.
  `learning to plan` `using available environment model` `I2A`
>	"In contrast to most existing model-based reinforcement learning and planning methods, which prescribe how a model should be used to arrive at a policy, I2As learn to interpret predictions from an imperfect learned environment model to construct implicit plans in arbitrary ways, by using predictions as additional context in deep policy networks."  
>	"I2A's policy and value functions are informed by the outputs of two separate pathways: 1) a model-free path, that tries to estimate the value and which action to take directly from the latest observation ot using a CNN; and 2) a model-based path, designed in the following way. The I2A is endowed with a pretrained, fixed environment model. At every time, conditioned on past observations and actions, it uses the model to simulate possible futures (Monte-Carlo rollouts) represented by imagnations over some horizon, under a rollout policy. It then extracts informative features from the rollout imaginations, and uses these, together with the results from the model-free path, to compute policy and value functions. It has been shown that I2As are robust to model imperfections: they learn to interpret imaginations produced from the internal models in order to inform decision making as part of standard return maximization."  
  - `video` <https://drive.google.com/drive/folders/0B4tKsKnCCZtQY2tTOThucHVxUTQ> (demo)
  - `post` <https://deepmind.com/blog/agents-imagine-and-plan/>
  - `video` <https://youtube.com/watch?v=agXIYMCICcc> (Kilcher)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=39m27s> (Mnih)

#### ["Metacontrol for Adaptive Imagination-Based Optimization"](https://arxiv.org/abs/1705.02670) Hamrick et al.
  `learning to plan` `using available environment model`
>	"Rather than learning a single, fixed policy for solving all instances of a task, we introduce a metacontroller which learns to optimize a sequence of "imagined" internal simulations over predictive models of the world in order to construct a more informed, and more economical, solution. The metacontroller component is a model-free reinforcement learning agent, which decides both how many iterations of the optimization procedure to run, as well as which model to consult on each iteration. The models (which we call "experts") can be state transition models, action-value functions, or any other mechanism that provides information useful for solving the task, and can be learned on-policy or off-policy in parallel with the metacontroller."  
>	"learns an adaptive optimization policy for one-shot decision-making in contextual bandit problems"  

#### ["Self-Correcting Models for Model-Based Reinforcement Learning"](https://arxiv.org/abs/1612.06018) Talvitie
  `learning to plan` `using available environment model` `Hallucinated DAgger-MC`
>	"When an agent cannot represent a perfectly accurate model of its environment’s dynamics, model-based reinforcement learning can fail catastrophically. Planning involves composing the predictions of the model; when flawed predictions are composed, even minor errors can compound and render the model useless for planning. Hallucinated Replay trains the model to “correct” itself when it produces errors, substantially improving MBRL with flawed models. This paper theoretically analyzes this approach, illuminates settings in which it is likely to be effective or ineffective, and presents a novel error bound, showing that a model’s ability to self-correct is more tightly related to MBRL performance than one-step prediction error."  
>	"Model-free methods are generally robust to representational limitations that prevent convergence to optimal behavior. In contrast, when the model representation is insufficient to perfectly capture the environment’s dynamics (even in seemingly innocuous ways), or when the planner produces suboptimal plans, MBRL methods can fail catastrophically."  
>	"This paper presents novel error bounds that reveal the theoretical principles that underlie the empirical success of Hallucinated Replay. It presents negative results that identify settings where hallucinated training would be ineffective and identifies a case where it yields a tighter performance bound than standard training. This result allows the derivation of a novel MBRL algorithm with theoretical performance guarantees that are robust to model class limitations."  
  - `code` <http://github.com/etalvitie/hdaggermc>

----
#### ["Learning Model-based Planning from Scratch"](https://arxiv.org/abs/1707.06170) Pascanu, Li, Vinyals, Heess, Buesing, Racaniere, Reichert, Weber, Wierstra, Battaglia
  `learning to plan` `learning abstract environment model` `IBP`
>	"- A fully learnable model-based planning agent for continuous control.  
>	- An agent that learns to construct a plan via model-based imagination.  
>	- An agent which uses its model of the environment in two ways: for imagination-based planning and gradient-based policy optimization.  
>	- A novel approach for learning to build, navigate, and exploit 'imagination trees'."  
>	"Before any action, agent can perform a variable number of imagination steps, which involve proposing an imagined action and evaluating it with its model-based imagination. All imagined actions and outcomes are aggregated, iteratively, into a "plan context" which conditions future real and imagined actions. The agent can even decide how to imagine: testing out alternative imagined actions, chaining sequences of actions together, or building a more complex "imagination tree" by navigating flexibly among the previously imagined states using a learned policy. And our agent can learn to plan economically, jointly optimizing for external rewards and computational costs associated with using its imagination."  
>	"The imagination-based model of the environment was an interaction network, a powerful neural architecture to model graph-like data. The model was trained to make next-step predictions of the state in a supervised fashion, with error gradients computed by backpropagation. The data was collected from the observations the agent makes when acting in the real world."  
  - `video` <https://drive.google.com/drive/folders/0B3u8dCFTG5iVaUxzbzRmNldGcU0> (demo)
  - `post` <https://deepmind.com/blog/agents-imagine-and-plan/>
  - `video` <https://youtube.com/watch?v=56GW1IlWgMg> (Kilcher)
  - `paper` ["Interaction Networks for Learning about Objects, Relations and Physics"](http://arxiv.org/abs/1612.00222) by Battaglia et al.

#### ["Self-supervised Deep Reinforcement Learning with Generalized Computation Graphs for Robot Navigation"](https://arxiv.org/abs/1709.10489) Kahn, Villaflor, Ding, Abbeel, Levine
  `learning to plan` `learning abstract environment model`
>	"generalized computation graph that subsumes value-based model-free methods and model-based methods, with specific instantiations interpolating between model-free and model-based"  
  - `video` <https://youtube.com/watch?v=vgiW0HlQWVE> (demo)
  - `code` <http://github.com/gkahn13/gcg>

#### ["Value Prediction Network"](https://arxiv.org/abs/1707.03497) Oh, Singh, Lee
  `learning to plan` `learning abstract environment model`
>	"VPN combines model-based RL (i.e., learning the dynamics of an abstract state space sufficient for computing future rewards and values) and model-free RL (i.e., mapping the learned abstract states to rewards and values) in a unified framework. In order to train a VPN, we propose a combination of temporal-difference search (TD search) and n-step Q-learning. In brief, VPNs learn to predict values via Q-learning and rewards via supervised learning. At the same time, VPNs perform lookahead planning to choose actions and compute bootstrapped target Q-values."  
>	"Extends the Predictron model from policy evaluation to optimal control."  
>	"Uses the model to construct a look-ahead tree only when constructing bootstrap targets and selecting actions, similarly to TD-search. Crucially, the model is not embedded in a planning algorithm during optimisation."  
  - `video` <http://videolectures.net/deeplearning2017_singh_reinforcement_learning/> (1:12:46) (Singh)
  - `video` <https://youtu.be/PRQ8-FwDPRE?t=16m> (Holland)

#### ["The Predictron: End-to-End Learning and Planning"](https://arxiv.org/abs/1612.08810) Silver et al.
  `learning to plan` `learning abstract environment model`
>	"The Predictron consists of a fully abstract model, represented by a Markov reward process, that can be rolled forward multiple “imagined" planning steps. Each forward pass of the predictron accumulates internal rewards and values over multiple planning depths. The predictron is trained end-to-end so as to make these accumulated values accurately approximate the true value function."  
>	"trains deep network to implicitly plan via iterative rollouts"  
>	"uses implicit environment model which does not capture dynamics"  
>	"only applied to learning Markov reward processes rather than solving Markov decision processes"  
  - `video` <https://youtube.com/watch?v=BeaLdaN2C3Q> (demo)
  - `video` <https://vimeo.com/238243832>
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=55m9s> (Abbeel)
  - `video` <http://videolectures.net/deeplearning2017_singh_reinforcement_learning/> (1:12:46) (Singh)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=38m6s> (Mnih)
  - `code` <https://github.com/zhongwen/predictron>
  - `code` <https://github.com/muupan/predictron>

#### ["Cognitive Mapping and Planning for Visual Navigation"](https://arxiv.org/abs/1702.03920) Gupta, Davidson, Levine, Sukthankar, Malik
  `learning to plan` `learning abstract environment model`
>	"1st person mapping + navigation with VIN"  
  - `video` <https://youtu.be/ID150Tl-MMw?t=54m24s> (demo)
  - `code` <https://github.com/tensorflow/models/tree/master/research/cognitive_mapping_and_planning>

#### ["Value Iteration Networks"](http://arxiv.org/abs/1602.02867) Tamar, Wu, Thomas, Levine, Abbeel
  `learning to plan` `learning abstract environment model`
>	"trains deep network to implicitly plan via iterative rollouts"  
>	"uses implicit environment model which does not capture dynamics"  
  - `video` <https://youtu.be/ID150Tl-MMw?t=54m24s> (demo)
  - `video` <https://youtube.com/watch?v=tXBHfbHHlKc> (Tamar) ([slides](http://technion.ac.il/~danielm/icml_slides/Talk7.pdf))
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Value-Iteration-Networks> (Tamar)
  - `video` <http://www.fields.utoronto.ca/video-archive/2017/02/2267-16530> (31:50) (Abbeel)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=38m6s> (Mnih)
  - `notes` <https://github.com/karpathy/paper-notes/blob/master/vin.md>
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Value_Iteration_Networks.md>
  - `notes` <https://blog.acolyer.org/2017/02/09/value-iteration-networks/>
  - `code` <https://github.com/avivt/VIN>
  - `code` <https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks>
  - `code` <https://github.com/zuoxingdong/VIN_PyTorch_Visdom>
  - `code` <https://github.com/zuoxingdong/VIN_TensorFlow>
  - `code` <https://github.com/kentsommer/pytorch-value-iteration-networks>
  - `code` <https://github.com/onlytailei/Value-Iteration-Networks-PyTorch>

----
#### ["Visual Interaction Networks"](https://arxiv.org/abs/1706.01433) Watters, Tacchetti, Weber, Pascanu, Battaglia, Zoran
  `learning to simulate` `learning abstract environment model`
>	"general-purpose model for learning dynamics of physical system from raw visual observations without prior knowledge"  
>	"Model consists of a perceptual front-end based on convolutional neural networks and a dynamics predictor based on interaction networks. Through joint training, the perceptual front-end learns to parse a dynamic visual scene into a set of factored latent object representations. The dynamics predictor learns to roll these states forward in time by computing their interactions and dynamics, producing a predicted physical trajectory of arbitrary length."  
>	"We found that from just six input video frames the Visual Interaction Network can generate accurate future trajectories of hundreds of time steps on a wide range of physical systems. Our model can also be applied to scenes with invisible objects, inferring their future states from their effects on the visible objects, and can implicitly infer the unknown mass of objects."  
>	"The VIN is learnable and can be trained from supervised data sequences which consist of input image frames and target object state values. It can learn to approximate a range of different physical systems which involve interacting entities by implicitly internalizing the rules necessary for simulating their dynamics and interactions."  
  - `video` <https://goo.gl/FD1XX5> + <https://goo.gl/4SSGP0> (demo)
  - `post` <https://deepmind.com/blog/neural-approach-relational-reasoning/>

#### ["Interaction Networks for Learning about Objects, Relations and Physics"](http://arxiv.org/abs/1612.00222) Battaglia, Pascanu, Lai, Rezende, Kavukcuoglu
  `learning to simulate` `learning abstract environment model`
>	"Model which can reason about how objects in complex systems interact, supporting dynamical predictions, as well as inferences about the abstract properties of the system."  
>	"Model takes graphs as input, performs object- and relation-centric reasoning in a way that is analogous to a simulation, and is implemented using deep neural networks."  
>	"Our results show it can be trained to accurately simulate the physical trajectories of dozens of objects over thousands of time steps, estimate abstract quantities such as energy, and generalize automatically to systems with different numbers and configurations of objects and relations."  
>	"Our interaction network implementation is the first general-purpose, learnable physics engine, and a powerful general framework for reasoning about object and relations in a wide variety of complex real-world domains."  
>	"The interaction network may also serve as a powerful model for model-predictive control inputting active control signals as external effects – because it is differentiable, it naturally supports gradient-based planning."  
  - `notes` <https://blog.acolyer.org/2017/01/02/interaction-networks-for-learning-about-objects-relations-and-physics/>
  - `code` <https://github.com/jaesik817/Interaction-networks_tensorflow>

----
#### ["Stochastic Variational Video Prediction"](https://arxiv.org/abs/1710.11252) Babaeizadeh, Finn, Erhan, Campbell, Levine
  `learning to simulate` `learning video prediction model`
>	"Real-world events can be stochastic and unpredictable, and the high dimensionality and complexity of natural images requires the predictive model to build an intricate understanding of the natural world. Many existing methods tackle this problem by making simplifying assumptions about the environment. One common assumption is that the outcome is deterministic and there is only one plausible future. This can lead to low-quality predictions in real-world settings with stochastic dynamics."  
>	"SV2P predicts a different possible future for each sample of its latent variables. Our model is the first to provide effective stochastic multi-frame prediction for real-world video. We demonstrate the capability of the proposed method in predicting detailed future frames of videos on multiple real-world datasets, both action-free and action-conditioned."  
  - <https://sites.google.com/site/stochasticvideoprediction/> (demo)

#### ["Self-Supervised Visual Planning with Temporal Skip Connections"](https://arxiv.org/abs/1710.05268) Ebert, Finn, Lee, Levine
  `learning to simulate` `learning video prediction model`
  - <https://sites.google.com/view/sna-visual-mpc> (demo)
  - `video` <https://youtube.com/watch?v=6k7GHG4IUCY>
  - `video` <https://youtu.be/UDLI9K6b9G8?t=1h14m56s> (Ebert)
  - `code` <https://github.com/febert/visual_mpc>

----
#### ["Learning Multimodal Transition Dynamics for Model-Based Reinforcement Learning"](https://arxiv.org/abs/1705.00470) Moerland, Broekens, Jonker
  `learning to simulate` `learning full environment model`
>	"So why is model-based RL not the standard approach? Model-based RL consists of two steps: 1) transition function estimation through supervised learning, and 2) (sample-based) planning over the learned model. Each step has a particular challenging aspect. For this work we focus on a key challenge of the first step: stochasticity in the transition dynamics. Stochasticity is an inherent property of many environments, and increases in real-world settings due to sensor noise. Transition dynamics usually combine both deterministic aspects (such as the falling trajectory of an object due to gravity) and stochastic elements (such as the behaviour of another car on the road). Our goal is to learn to jointly predict these. Note that stochasticity has many forms, both homoscedastic versus heteroscedastic, and unimodal versus multimodal. In this work we specifically focus on multimodal stochasticity, as this should theoretically pose the largest challenge."  
>	"We focus on deep generative models as they can approximate complex distributions and scale to high-dimensional domains. For model-based RL we have additional requirements, as we are ultimately interested in using the model for sample-based planning. This usually requires sampling a lot of traces, so we require models that are 1) easy to sample from, 2) ideally allow planning at an abstract level. Implicit density models, like Generative Adverserial Networks lack a clear probabilistic objective function, which was the focus of this work. Among the explicit density models, there are two categories. Change of variable formula models, like Real NVP, have the drawback that the latent space dimension must equal the observation space. Fully visible belief nets like pixelCNN, which factorize the likelihood in an auto-regressive fashion, hold state-of-the-art likelihood results. However, they have the drawback that sampling is a sequential operation (e.g. pixel-by-pixel, which is computationally expensive), and they do not allow for latent level planning either. Therefore, most suitable for model-based RL seem approximate density models, most noteworthy the Variational Auto-Encoder framework. These models can estimate stochasticity at a latent level, allow for latent planning, are easy to sample from, and have a clear probabilistic interpretation."  
>	"An important challenge is planning under uncertainty. RL initially provides correlated data from a limited part of state-space. When planning over this model, we should not extrapolate too much, nor trust our model to early with limited data. Note that ‘uncertainty’ (due to limited data) is fundamentally different from the ‘stochasticity’ (true probabilistic nature of the domain) discussed in this paper."  
  - `code` <http://github.com/tmoer/multimodal_varinf>

#### ["Prediction and Control with Temporal Segment Models"](https://arxiv.org/abs/1703.04070) Mishra, Abbeel, Mordatch
  `learning to simulate` `learning full environment model`
>	"We learn the distribution over future state trajectories conditioned on past state, past action, and planned future action trajectories, as well as a latent prior over action trajectories. Our approach is based on convolutional autoregressive models and variational autoencoders. It makes stable and accurate predictions over long horizons for complex, stochastic systems, effectively expressing uncertainty and modeling the effects of collisions, sensory noise, and action delays."  
  - `video` <https://vimeo.com/237267784> (Mishra)

#### ["Learning and Policy Search in Stochastic Dynamic Systems with Bayesian Neural Networks"](https://arxiv.org/abs/1605.07127) Depeweg, Hernandez-Lobato, Doshi-Velez, Udluft
  `learning to simulate` `learning full environment model`
>	"Monte-Carlo model-based policy gradient technique in continuous stochastic systems"  
>	"Proposed approach enables automatic identification of arbitrary stochastic patterns such as multimodality and heteroskedasticity, without having to manually incorporate these into the model."  
>	"We have extended Bayesian neural network with addition of a random input noise source z. This enables principled Bayesian inference over complex stochastic functions. We have also presented an algorithm that uses random roll-outs and stochastic optimization for learning a parameterized policy in a batch scenario. Our BNNs with random inputs have allowed us to solve a challenging benchmark problem where model-based approaches usually fail."  
>	"For safety, we believe having uncertainty over the underlying stochastic functions will allow us to optimize policies by focusing on worst case results instead of on average performance. For exploration, having uncertainty on the stochastic functions will be useful for efficient data collection."  
>	"The optimal policy can be significantly affected by the noise present in the state transitions. This is illustrated by the drunken spider story, in which a spider has two possible paths to go home: either by crossing the bridge or by walking around the lake. In the absence of noise, the bridge option is prefered since it is shorter. However, after heavily drinking alcohol, the spider’s movements may randomly deviate left or right. Since the bridge is narrow, and spiders do not like swimming, the prefered trajectory is now to walk around the lake. The previous example shows how noise can significantly affect optimal control. For example, the optimal policy may change depending on whether the level of noise is high or low. Therefore, we expect to obtain significant improvements in model-based reinforcement learning by capturing with high accuracy any noise patterns present in the state transition data."  
  - `post` <https://medium.com/towards-data-science/bayesian-neural-networks-with-random-inputs-for-model-based-reinforcement-learning-36606a9399b4> (Hernandez-Lobato)
  - `video` <https://youtube.com/watch?v=0H3EkUPENSY> (Hernandez-Lobato)
  - `code` <https://github.com/siemens/policy_search_bb-alpha>

#### ["Recurrent Environment Simulators"](https://arxiv.org/abs/1704.02254) Chiappa, Racaniere, Wierstra, Mohamed
  `learning to simulate` `learning full environment model`
>	"We improve on previous environment simulators from high-dimensional pixel observations by introducing recurrent neural networks that are able to make temporally and spatially coherent predictions for hundreds of time-steps into the future."  
>	"We address the issue of computationally inefficiency with a model that does not need to generate a high-dimensional image at each time-step."  
>	"It is a deterministic model designed for deterministic environments. Clearly most real world environments involve noisy state transitions."  
  - `video` <https://drive.google.com/file/d/0B_L2b7VHvBW2NEQ1djNjU25tWUE/view> (TORCS demo)
  - `video` <https://drive.google.com/file/d/0B_L2b7VHvBW2UjMwWVRoM3lTeFU/view> (TORCS demo)
  - `video` <https://drive.google.com/file/d/0B_L2b7VHvBW2UWl5YUtSMXdUbnc/view> (3D Maze demo)



---
### reinforcement learning - exploration and intrinsic motivation

[interesting older papers](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---exploration-and-intrinsic-motivation)  
[interesting older papers - artificial curiosity and creativity](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#interesting-papers---artificial-curiosity-and-creativity)  

----
#### ["Contextual Decision Processes with Low Bellman Rank are PAC-Learnable"](https://arxiv.org/abs/1610.09512) Jiang, Krishnamurthy, Agarwal, Langford, Schapire
  `provably correct and sample efficient exploration`
>	"This paper studies systematic exploration for reinforcement learning with rich observations and function approximation. We introduce a new model called contextual decision processes, that unifies and generalizes most prior settings. Our first contribution is a complexity measure, the Bellman rank, that we show enables tractable learning of near-optimal behavior in these processes and is naturally small for many well-studied reinforcement learning settings. Our second contribution is a new reinforcement learning algorithm that engages in systematic exploration to learn contextual decision processes with low Bellman rank. Our algorithm provably learns near-optimal behavior with a number of samples that is polynomial in all relevant parameters but independent of the number of unique observations. The approach uses Bellman error minimization with optimistic exploration and provides new insights into efficient exploration for reinforcement learning with function approximation."  
>	"Approximation of value function with function from some class is a powerful practical approach with implicit assumption that true value function is approximately in class.  
>	Even with this assumption:  
>	- no guarantee methods will work  
>	- no bound on how much data needed  
>	- no theory on how to explore in large spaces"  
  - `video` <https://vimeo.com/235929810> (Schapire)
  - `video` <https://vimeo.com/238228755> (Jiang)

#### ["The Uncertainty Bellman Equation and Exploration"](https://arxiv.org/abs/1709.05380) O'Donoghue, Osband, Munos, Mnih
  `exploration in state-action space guided by uncertainty in value function`
>	"We consider uncertainty Bellman equation which connects the uncertainty at any time-step to the expected uncertainties at subsequent time-steps, thereby extending the potential exploratory benefit of a policy beyond individual time-steps. We prove that the unique fixed point of the UBE yields an upper bound on the variance of the estimated value of any fixed policy. This bound can be much tighter than traditional count-based bonuses that compound standard deviation rather than variance. Importantly, and unlike several existing approaches to optimism, this method scales naturally to large systems with complex generalization."  

----
#### ["Count-Based Exploration with Neural Density Models"](http://arxiv.org/abs/1703.01310) Ostrovski, Bellemare, Oord, Munos
  `exploration in state space guided by probability of observation`
>	"PixelCNN for exploration, neural alternative to Context Tree Switching"  
  - `video` <http://youtube.com/watch?v=qSfd27AgcEk> (Bellemare)
  - `video` <https://vimeo.com/238243932> (Bellemare)

#### ["EX2: Exploration with Exemplar Models for Deep Reinforcement Learning"](https://arxiv.org/abs/1703.01260) Fu, Co-Reyes, Levine
  `exploration in state space guided by probability of observation`
>	"Many of the most effective exploration techniques rely on tabular representations, or on the ability to construct a generative model over states and actions. This paper introduces a novel approach, EX2, which approximates state visitation densities by training an ensemble of discriminators, and assigns reward bonuses to rarely visited states."  

#### ["#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning"](http://arxiv.org/abs/1611.04717) Tang, Houthooft, Foote, Stooke, Chen, Duan, Schulman, Turck, Abbeel
  `exploration in state space guided by probability of observation`
>	"The authors encourage exploration by adding a pseudo-reward of the form beta/sqrt(count(state)) for infrequently visited states. State visits are counted using Locality Sensitive Hashing (LSH) based on an environment-specific feature representation like raw pixels or autoencoder representations. The authors show that this simple technique achieves gains in various classic RL control tasks and several games in the ATARI domain. While the algorithm itself is simple there are now several more hyperaprameters to tune: The bonus coefficient beta, the LSH hashing granularity (how many bits to use for hashing) as well as the type of feature representation based on which the hash is computed, which itself may have more parameters. The experiments don't paint a consistent picture and different environments seem to need vastly different hyperparameter settings, which in my opinion will make this technique difficult to use in practice."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1611.04717>
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/%23Exploration:_A_Study_of_Count-Based_Exploration_for_Deep_Reinforcement_Learning.md>

----
#### ["Unsupervised Real-Time Control through Variational Empowerment"](https://arxiv.org/abs/1710.05101) Karl, Soelch, Becker-Ehmck, Benbouzid, Smagt, Bayer
  `exploration in state space guided by empowerment - channel capacity between actions and states`

#### ["Variational Intrinsic Control"](http://arxiv.org/abs/1611.07507) Gregor, Rezende, Wierstra
  `exploration in state space guided by empowerment - channel capacity between actions and states`
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
#### ["Curiosity-driven Exploration by Self-supervised Prediction"](https://arxiv.org/abs/1705.05363) Pathak, Agrawal, Efros, Darrell
  `exploration in state space guided by prediction error`
>	"Our main contribution is in designing an intrinsic reward signal based on prediction error of the agent’s knowledge about its environment that scales to high-dimensional continuous state spaces like images, bypasses the hard problem of predicting pixels and is unaffected by the unpredictable aspects of the environment that do not affect the agent."  
  - `video` <https://vimeo.com/237270588> (Pathak)
  - `post` <https://pathak22.github.io/noreward-rl/index.html> (demo)
  - `code` <https://github.com/pathak22/noreward-rl>

#### ["Surprise-Based Intrinsic Motivation for Deep Reinforcement Learning"](http://arxiv.org/abs/1703.01732) Achiam, Sastry
  `exploration in state-action space guided by prediction error`
>	"Authors present two tractable approximations to their framework - one which ignores the stochasticity of the true environmental dynamics, and one which approximates the rate of information gain (somewhat similar to Schmidhuber's formal theory of creativity, fun and intrinsic motivation)."  
>	"Stadie et al. learn deterministic dynamics models by minimizing Euclidean loss - whereas in our work, we learn stochastic dynamics with cross entropy loss - and use L2 prediction errors for intrinsic motivation."  
>	"Our results suggest that surprisal is a viable alternative to VIME in terms of performance, and is highly favorable in terms of computational cost. In VIME, a backwards pass through the dynamics model must be computed for every transition tuple separately to compute the intrinsic rewards, whereas our surprisal bonus only requires forward passes through the dynamics model for intrinsic reward computation. Furthermore, our dynamics model is substantially simpler than the Bayesian neural network dynamics model of VIME. In our speed test, our bonus had a per-iteration speedup of a factor of 3 over VIME."  

----
#### ["Some Considerations on Learning to Explore via Meta-Reinforcement Learning"](https://openreview.net/forum?id=Skk3Jm96W)
  `learning to explore in state-action space`
>	"We introduce two new algorithms: E-MAML and E-RL2, which are derived by reformulating the underlying meta-learning objective to account for the impact of initial sampling on future (post-meta-updated) returns."  
>	"Meta RL agent must not learn how to master the environments it is given, but rather it must learn how to learn so that it can quickly train at test time."  
>	"It is likely that future work in this area will focus on meta-learning a curiosity signal which is robust and transfers across tasks. Perhaps this will enable meta agents which learn to explore rather than being forced to explore by mathematical trickery in their objectives."  

#### ["RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning"](http://arxiv.org/abs/1611.02779) Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel
  `learning to explore in state-action space` `meta-learning`
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#rl2-fast-reinforcement-learning-via-slow-reinforcement-learning--duan-schulman-chen-bartlett-sutskever-abbeel>

#### ["Learning to Reinforcement Learn"](http://arxiv.org/abs/1611.05763) Wang et al.
  `learning to explore in state-action space` `meta-learning`
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#learning-to-reinforcement-learn-wang-et-al>

#### ["Exploration Potential"](http://arxiv.org/abs/1609.04994) Leike
  `learning to explore in state-action space`
>	"We introduce exploration potential, a quantity that measures how much a reinforcement learning agent has explored its environment class. In contrast to information gain, exploration potential takes the problem's reward structure into account."  

----
#### ["Noisy Networks for Exploration"](https://arxiv.org/abs/1706.10295) Fortunato, Azar, Piot, Menick, Osband, Graves, Mnih, Munos, Hassabis, Pietquin, Blundell, Legg
  `state dependent exploration` 
>	"scale of perturbation to parameters is learned along with original objective function"  
  - `video` <https://youtu.be/fevMOp5TDQs?t=1h27s> (Mnih)
  - `code` <https://github.com/Kaixhin/NoisyNet-A3C>
  - `code` <https://github.com/andrewliao11/NoisyNet-DQN>

#### ["Parameter Space Noise for Exploration"](https://arxiv.org/abs/1706.01905) Plappert, Houthooft, Dhariwal, Sidor, Chen, Chen, Asfour, Abbeel, Andrychowicz
  `state dependent exploration` 
  - `post` <https://blog.openai.com/better-exploration-with-parameter-noise/>

----
#### ["UCB and InfoGain Exploration via Q-Ensembles"](https://arxiv.org/abs/1706.01502) Chen, Sidor, Abbeel, Schulman
  `approximate bayesian exploration`

#### ["Deep Exploration via Randomized Value Functions"](https://arxiv.org/abs/1703.07608) Osband, Russo, Wen, Roy
  `approximate bayesian exploration`
>	"A very recent thread of work builds on count-based (or upper-confidence-bound-based) exploration schemes that operate with value function learning. These methods maintain a density over the state-action space of pseudo-counts, which represent the quantity of data gathered that is relevant to each state-action pair. Such algorithms may offer a viable approach to deep exploration with generalization. There are, however, some potential drawbacks. One is that a separate representation is required to generalize counts, and it's not clear how to design an effective approach to this. As opposed to the optimal value function, which is fixed by the environment, counts are generated by the agent’s choices, so there is no single target function to learn. Second, the count model generates reward bonuses that distort data used to fit the value function, so the value function representation needs to be designed to not only capture properties of the true optimal value function but also such distorted versions. Finally, these approaches treat uncertainties as uncoupled across state-action pairs, and this can incur a substantial negative impact on statistical efficiency."  
  - `video` <http://youtube.com/watch?v=ck4GixLs4ZQ> (Osband) + [slides](https://docs.google.com/presentation/d/1lis0yBGT-uIXnAsi0vlP3SuWD2svMErJWy_LYtfzMOA/)

----
#### ["The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously"](https://arxiv.org/abs/1707.03300) Cabi, Colmenarejo, Hoffman, Denil, Wang, de Freitas
  `structured exploration in policy space guided by learning additional tasks`

#### ["Reinforcement Learning with Unsupervised Auxiliary Tasks"](http://arxiv.org/abs/1611.05397) Jaderberg, Mnih, Czarnecki, Schaul, Leibo, Silver, Kavukcuoglu
  `UNREAL` `structured exploration in policy space guided by learning additional tasks`
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
  - `video` <https://yadi.sk/i/_2_0yqeW3HDbcn> (18:25) (Panin) `in russian`
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/b097e313dc59c956575fb1bf23b64fa8d1d84057/notes/rl-auxiliary-tasks.md>
  - `code` <https://github.com/miyosuda/unreal>

#### ["Learning to Navigate in Complex Environments"](http://arxiv.org/abs/1611.03673) Mirowski, Pascanu, Viola, Soyer, Ballard, Banino, Denil, Goroshin, Sifre, Kavukcuoglu, Kumaran, Hadsell
  `structured exploration in policy space guided by learning additional tasks`
>	"Auxiliary tasks:  
>	- depth predictor  
>	- loop closure predictor  
>	Additional inputs: reward, action, velocity"  
  - `video` <https://youtu.be/PS4iJ7Hk_BU> + <https://youtu.be/-HsjQoIou_c> + <https://youtu.be/kH1AvRAYkbI> + <https://youtu.be/5IBT2UADJY0> + <https://youtu.be/e10mXgBG9yo> (demo)
  - `video` <http://youtube.com/watch?v=5Rflbx8y7HY> (Mirowski)
  - `video` <http://youtu.be/0e_uGa7ic74?t=8m53s> + <https://vimeo.com/238221551#t=22m37s> (Hadsell)
  - `notes` <http://pemami4911.github.io/paper-summaries/2016/12/20/learning-to-navigate-in-complex-envs.html>
  - `code` <https://github.com/tgangwani/GA3C-DeepNavigation>

#### ["Loss is Its Own Reward: Self-Supervision for Reinforcement Learning"](http://arxiv.org/abs/1612.07307) Shelhamer, Mahmoudieh, Argus, Darrell
  `structured exploration in policy space guided by learning additional tasks`

#### ["Feature Control as Intrinsic Motivation for Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1705.06769) Dilokthanakul, Kaplanis, Pawlowski, Shanahan
  `structured exploration in policy space guided by learning additional tasks`
>	"Extracting reward from features makes sparse-reward environment into dense one. Authors incorporate this strategy to solve hierarchical RL."  
>	"Authors solve Montezuma with Options framework (but without dynamic terminal condition). The agent is encouraged to change pixel or visual-feature given the option from meta-control and this dense feedback makes agent to learn basic skill under high sparsity."  

----
#### ["Overcoming Exploration in Reinforcement Learning with Demonstrations"](https://arxiv.org/abs/1709.10089) Nair, McGrew, Andrychowicz, Zaremba, Abbeel
  `structured exploration in policy space guided by learning progress and demonstrations`
>	"We use demonstrations to overcome the exploration problem and successfully learn to perform long-horizon, multi-step robotics tasks with continuous control such as stacking blocks with a robot arm."  
>	"Our method, which builds on top of Deep Deterministic Policy Gradients and Hindsight Experience Replay, provides an order of magnitude of speedup over RL on simulated robotics tasks."  
>	"Our method is able to solve tasks not solvable by either RL or behavior cloning alone, and often ends up outperforming the demonstrator policy."  
  - <http://ashvin.me/demoddpg-website/> (demo)

#### ["Hindsight Experience Replay"](https://arxiv.org/abs/1707.01495) Andrychowicz, Wolski, Ray, Schneider, Fong, Welinder, McGrew, Tobin, Abbeel, Zaremba
  `structured exploration in policy space guided by learning progress`
>	"HER may be seen as a form of implicit curriculum as the goals used for replay naturally shift from ones which are simple to achieve even by a random agent to more difficult ones. However, in contrast to explicit curriculum, HER does not require having any control over the distribution of initial environment states."  
>	"Not only does HER learn with extremely sparse rewards, in our experiments it also performs better with sparse rewards than with shaped ones. These results are indicative of the practical challenges with reward shaping, and that shaped rewards would often constitute a compromise on the metric we truly care about (such as binary success/failure)."  
  - <https://sites.google.com/site/hindsightexperiencereplay/> (demo)
  - `video` <https://youtu.be/BCzFs9Xb9_o?t=21m2s> (Sutskever)
  - `video` <https://youtu.be/TERCdog1ddE?t=50m45s> (Abbeel)
  - `paper` ["Universal Value Function Approximators"](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#schaul-horgan-gregor-silver---universal-value-function-approximators) by Schaul et al.

#### ["Reverse Curriculum Generation for Reinforcement Learning"](https://arxiv.org/abs/1707.05300) Florensa, Held, Wulfmeier, Zhang, Abbeel
  `structured exploration in policy space guided by learning progress`
>	"Many tasks require to reach a desired configuration (goal) from everywhere."  
>	"Challenging for current RL: inherently sparse rewards, most start positions get 0 reward."  
>
>	"Solve the task in reverse, first training from positions closer to the goal and then bootstrap this knowledge to solve from further."  
>	"Sample more start states from where you succeed sometimes but not always (for best efficiency)."  
  - `video` <https://youtu.be/xfyK03MEZ9Q?t=7h32m35s> (Florensa)

#### ["Teacher-Student Curriculum Learning"](https://arxiv.org/abs/1707.00183) Matiisen, Oliver, Cohen, Schulman
  `structured exploration in policy space guided by learning progress`

#### ["Automated Curriculum Learning for Neural Networks"](https://arxiv.org/abs/1704.03003) Graves, Bellemare, Menick, Munos, Kavukcuoglu
  `structured exploration in policy space guided by learning progress`
>	"We focus on variants of prediction gain, and also introduce a novel class of progress signals which we refer to as complexity gain. Derived from minimum description length principles, complexity gain equates acquisition of knowledge with an increase in effective information encoded in the network weights."  
>	"VIME uses a reward signal that is closely related to variational complexity gain. The difference is that while VIME measures the KL between the posterior before and after a step in parameter space, we consider the change in KL between the posterior and prior induced by the step. Therefore, while VIME looks for any change to the posterior, we focus only on changes that alter the divergence from the prior. Further research will be needed to assess the relative merits of the two signals."  
>	"For maximum likelihood training, we found prediction gain to be the most consistent signal, while for variational inference training, gradient variational complexity gain performed best. Importantly, both are instantaneous, in the sense that they can be evaluated using only the samples used for training."  
  - `video` <https://youtu.be/-u32TOPGIbQ?t=2m43s> (Graves)
  - `video` <https://vimeo.com/237275086> (Bellemare)

#### ["Automatic Goal Generation for Reinforcement Learning Agents"](https://arxiv.org/abs/1705.06366) Graves, Bellemare, Menick, Munos, Kavukcuoglu
  `structured exploration in policy space guided by learning progress`
  - `notes` <https://blog.tomrochette.com/machine-learning/papers/alex-graves-automated-curriculum-learning-for-neural-networks>

#### ["Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play"](http://arxiv.org/abs/1703.05407) Sukhbaatar, Lin, Kostrikov, Synnaeve, Szlam, Fergus
  `structured exploration in policy space guided by learning progress`
>	"self-play between the policy and a task-setter in order to automatically generate goal states which are on the border of what the current policy can achieve"  
  - `video` <https://youtube.com/watch?v=EHHiFwStqaA> (demo)
  - `video` <https://youtube.com/watch?v=X1O21ziUqUY> (Fergus)
  - `video` <https://youtube.com/watch?v=5dNAnCYBFN4> (Szlam)
  - `post` <http://giorgiopatrini.org/posts/2017/09/06/in-search-of-the-missing-signals/>

#### ["Towards Information-Seeking Agents"](http://arxiv.org/abs/1612.02605) Bachman, Sordoni, Trischler
  `structured exploration in policy space guided by learning progress`
  - `video` <https://youtube.com/watch?v=3bSquT1zqj8> (demo)

#### ["Learning to Perform Physics Experiments via Deep Reinforcement Learning"](http://arxiv.org/abs/1611.01843) Denil, Agrawal, Kulkarni, Erez, Battaglia, de Freitas
  `structured exploration in policy space guided by learning progress`
>	"By letting our agents conduct physical experiments in an interactive simulated environment, they learn to manipulate objects and observe the consequences to infer hidden object properties."  
>	"By systematically manipulating the problem difficulty and the cost incurred by the agent for performing experiments, we found that agents learn different strategies that balance the cost of gathering information against the cost of making mistakes in different situations."  
  - `video` <https://youtu.be/SAcHyzMdbXc?t=16m6s> (de Freitas)



---
### reinforcement learning - hierarchical

[interesting older papers](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---hierarchical-reinforcement-learning)

----
#### ["Learning with Options that Terminate Off-Policy"](https://arxiv.org/abs/1711.03817) Harutyunyan, Vrancx, Bacon, Precup, Nowe

#### ["Meta Learning Shared Hierarchies"](https://arxiv.org/abs/1710.09767) Frans, Ho, Chen, Abbeel, Schulman
  - `post` <https://blog.openai.com/learning-a-hierarchy/> (demo)
  - `video` <https://youtu.be/BCzFs9Xb9_o?t=32m35s> (Sutskever)
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Meta_Learning_Shared_Hierarchies.md>
  - `code` <https://github.com/openai/mlsh>

#### ["Eigenoption Discovery Through The Deep Successor Rerpesentation"](https://arxiv.org/abs/1710.11089) Machado, Rosenbaum, Guo, Liu, Tesauro, Campbell

#### ["Stochastic Neural Networks for Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1704.03012) Florensa, Duan, Abbeel
>	"Our SNN hierarchical approach outperforms state-of-the-art intrinsic motivation results like VIME (Houthooft et al., 2016)."  
  - `video` <https://youtube.com/playlist?list=PLEbdzN4PXRGVB8NsPffxsBSOCcWFBMQx3> (demo)
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Stochastic_Neural_Networks_for_Hierarchical_Reinforcement_Learning.md>
  - `code` <https://github.com/florensacc/snn4hrl>

#### ["FeUdal Networks for Hierarchical Reinforcement Learning"](http://arxiv.org/abs/1703.01161) Vezhnevets, Osindero, Schaul, Heess, Jaderberg, Silver, Kavukcuoglu
>	"A novel architecture that formulates sub-goals as directions in latent state space, which, if followed, translates into a meaningful behavioural primitives. FuN clearly separates the module that discovers and sets sub-goals from the module that generates behaviour through primitive actions. This creates a natural hierarchy that is stable and allows both modules to learn in complementary ways."  
>	"Agent with two level hierarchy: manager and worker."  
>	"Manager does not act in environment directly, sets goals for worker and gets rewarded for setting good goals with true reward."  
>	"Worker acts in environment and gets rewarded for achieving goals by manager - this is potentially much richer learning signal."  
>	"Key problems: how to represent goals and determine when they've been achieved."  
>
>	"Encourage manager to predict advantageous direction in latent space and give intrinsic reward to worker to follow the direction."  
>	"Options framework:  
>	- bottom level: option, a sub-policy with terminal condition  
>	- top level: policy over options  
>	FeUdal framework:  
>	- bottom level: action  
>	- top level: provide meaningful and explicit goal for bottom level  
>	- sub-goal: a direction in latent space"  
  - `video` <https://youtube.com/watch?v=0e_uGa7ic74&t=29m20s> (demo)
  - `video` <https://vimeo.com/238243758> (Vezhnevets)
  - `video` <https://youtube.com/watch?v=0e_uGa7ic74&t=20m10s> + <https://vimeo.com/238221551#t=6m21s> (Hadsell)
  - `video` <https://youtube.com/watch?v=bsuvM1jO-4w&t=46m31s> (Mnih)
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=57m51s> (Abbeel)
  - `code` <https://github.com/dmakian/feudal_networks>

#### ["A Laplacian Framework for Option Discovery in Reinforcement Learning"](https://arxiv.org/abs/1703.00956) Machado, Bellemare, Bowling
>	"Proto-value functions are a well-known approach for representation learning in MDPs. We address the option discovery problem by showing how PVFs implicitly define options. We do it by introducing eigenpurposes, intrinsic reward functions derived from the learned representations. The options discovered from eigenpurposes traverse the principal directions of the state space."  
>	"Our algorithm can be seen as a bottom-up approach, in which we construct options before the agent observes any informative reward. These options are composed to generate the desired policy. Options discovered this way tend to be independent of an agent’s intention."  
  - `video` <https://youtube.com/watch?v=2BVicx4CDWA> (demo)
  - `video` <https://vimeo.com/220484541> (Machado)
  - `video` <https://vimeo.com/237274347> (Machado)

#### ["Variational Intrinsic Control"](http://arxiv.org/abs/1611.07507) Gregor, Rezende, Wierstra
  - `code` <https://github.com/sygi/vic-tensorflow>

#### ["Modular Multitask Reinforcement Learning with Policy Sketches"](http://arxiv.org/abs/1611.01796) Andreas, Klein, Levine
  - `video` <https://vimeo.com/237274402> (Andreas)
  - `video` <https://youtube.com/watch?v=NRIcDEB64x8> (Andreas)
  - `code` <https://github.com/jacobandreas/psketch>

#### ["Principled Option Learning in Markov Decision Processes"](https://arxiv.org/abs/1609.05524) Fox, Moshkovitz, Tishby
>	"We suggest a mathematical characterization of good sets of options using tools from information theory. This characterization enables us to find conditions for a set of options to be optimal and an algorithm that outputs a useful set of options and illustrate the proposed algorithm in simulation."  
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Principled_Option_Learning_in_Markov_Decision_Processes.md>

#### ["The Option-Critic Architecture"](http://arxiv.org/abs/1609.05140) Bacon, Harb, Precup
  - `video` <https://youtube.com/watch?v=8r_EoYnPjGk> (Bacon)
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=56m55s> (Abbeel)
  - `slides` <http://pierrelucbacon.com/optioncritic-aaai2017-slides.pdf>
  - `poster` <http://pierrelucbacon.com/optioncriticposter.pdf>
  - `notes` <http://tsong.me/blog/option-critic/>
  - `code` <https://github.com/jeanharb/option_critic>

#### ["Probabilistic Inference for Determining Options in Reinforcement Learning"](https://link.springer.com/article/10.1007/s10994-016-5580-x) Daniel, Hoof, Peters, Neumann
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Probabilistic_Inference_for_Determining_Options_in_Reinforcement_Learning.md>

#### ["Strategic Attentive Writer for Learning Macro-Actions"](http://arxiv.org/abs/1606.04695) Vezhnevets, Mnih, Agapiou, Osindero, Graves, Vinyals, Kavukcuoglu
  - `video` <https://youtube.com/watch?v=niMOdSu3yio> (demo)
  - `notes` <https://theberkeleyview.wordpress.com/2017/01/03/strategic-attentive-writer-for-learning-macro-actions/>
  - `notes` <https://blog.acolyer.org/2017/01/06/strategic-attentive-writer-for-learning-macro-actions/>



---
### reinforcement learning - transfer

----
#### ["DARLA: Improving Zero-Shot Transfer in Reinforcement Learning"](https://arxiv.org/abs/1707.08475) Higgins et al.
  `semantic representation`
>	"DisentAngled Representation Learning Agent first learns a visual system that encodes the observations it receives from the environment as disentangled representations, in a completely unsupervised manner. Once DARLA can see, it is able to acquire source policies that are robust to many domain shifts - even with no access to the target domain."  
>	"The final goal of adaptation is to learn two different tasks where state is different but action space is same and have similar transition and reward function. The training of DARLA is composed of two stages:  
>	- learn beta-VAE (learn to see)  
>	- learn policy network (learn to act)"  
  - `video` <https://youtube.com/watch?v=sZqrWFl0wQ4> (demo)
  - `video` <https://vimeo.com/237274156> (Higgins)

#### ["Towards Deep Symbolic Reinforcement Learning"](http://arxiv.org/abs/1609.05518) Garnelo, Arulkumaran, Shanahan
  `semantic representation`
>	"Contemporary DRL systems require very large datasets to work effectively, entailing that they are slow to learn even when such datasets are available. Moreover, they lack the ability to reason on an abstract level, which makes it difficult to implement high-level cognitive functions such as transfer learning, analogical reasoning, and hypothesis-based reasoning. Finally, their operation is largely opaque to humans, rendering them unsuitable for domains in which verifiability is important. We propose an end-to-end RL architecture comprising a neural back end and a symbolic front end with the potential to overcome each of these shortcomings."  
>	"Resulting system learns effectively and, by acquiring a set of symbolic rules that are easily comprehensible to humans, dramatically outperforms a conventional, fully neural DRL system on a stochastic variant of the game."  
>	"We tested the transfer learning capabilities of our algorithm by training an agent only on games of the grid variant then testing it on games of the random variant. After training, the unsupervised neural back end of the system is able to form a symbolic representation of any given frame within the micro-world of the game. In effect it has acquired the ontology of that micro-world, and this capability can be applied to any game within that micro-world irrespective of its specific rules. In the present case, no re-training of the back end was required when the system was applied to new variants of the game."  
>	"The key is for the system to understand when a new situation is analogous to one previously encountered or, more potently, to hypothesise that a new situation contains elements of several previously encountered situations combined in a novel way. In the present system, this capability is barely exploited."  
  - `video` <https://youtube.com/watch?v=HOAVhPy6nrc> (Shanahan)

----
#### ["Mutual Alignment Transfer Learning"](https://arxiv.org/abs/1707.07907) Wulfmeier, Posner, Abbeel
  `simulation to real world`
>	"While sample complexity can be reduced by training policies in simulation, such policies can perform sub-optimally on the real platform given imperfect calibration of model dynamics. We present an approach -- supplemental to fine tuning on the real robot -- to further benefit from parallel access to a simulator during training and reduce sample requirements on the real robot. The developed approach harnesses auxiliary rewards to guide the exploration for the real world agent based on the proficiency of the agent in simulation and vice versa. In this context, we demonstrate empirically that the reciprocal alignment for both agents provides further benefit as the agent in simulation can adjust to optimize its behaviour for states commonly visited by the real-world agent."  
>	"We propose MATL, which instead of directly adapting the simulation policy, guides the exploration for both systems towards mutually aligned state distributions via auxiliary rewards. The method employs an adversarial approach to train policies with additional rewards based on confusing a discriminator with respect to the originating system for state sequences visited by the agents. By guiding the target agent on the robot towards states that the potentially more proficient source agent visits in simulation, we can accelerate training. In addition to aligning the robot policy to adapt to progress in simulation, we extend the approach to mutually align both systems which can be beneficial as the agent in simulation will be driven to explore better trajectories from states visited by the real-world policy."  
>	"We demonstrate that auxiliary rewards, which guide the exploration on the target platform, improve performance in environments with sparse rewards and can even guide the agent if only uninformative or no environment rewards at all are given for the target agent."  
>	"In addition to aligning the robot policy to adapt to progress in simulation, the reciprocal alignment of the simulation policy can be beneficial as the agent in simulation will be driven to explore better behaviours from states visited by the robot agent."  
  - <https://sites.google.com/view/matl> (demo)
  - `video` <https://youtu.be/xfyK03MEZ9Q?t=6h24m47s> (Wulfmeier)

----
#### ["Distral: Robust Multitask Reinforcement Learning"](https://arxiv.org/abs/1707.04175) Teh, Bapst, Czarnecki, Quan, Kirkpatrick, Hadsell, Heess, Pascanu
  `policy distillation`
>	"The assumption is that the tasks are related to each other (e.g. being in the same environment or having the same physics) and so good action sequences tend to recur across tasks. Our method achieves this by simultaneously distilling task-specific policies into a common default policy, and transferring this common knowledge across tasks by regularising all task-specific policies towards the default policy."  
>	"We observe that distillation arises naturally as one half of an optimization procedure when using KL divergences to regularize the output of task models towards a distilled model. The other half corresponds to using the distilled model as a regularizer for training the task models."  
>	"Another observation is that parameters in deep networks do not typically by themselves have any semantic meaning, so instead of regularizing networks in parameter space, it is worthwhile considering regularizing networks in a more semantically meaningful space, e.g. of policies."  
  - `video` <http://www.fields.utoronto.ca/video-archive/2017/11/2509-17850> (Teh)
  - `video` <https://vimeo.com/238221551#t=20m7s> (Hadsell)
  - `notes` <http://shaofanlai.com/post/37>

#### ["Generalizing Skills with Semi-Supervised Reinforcement Learning"](http://arxiv.org/abs/1612.00429) Finn, Yu, Fu, Abbeel, Levine
  `policy distillation`
>	"It is often quite practical to provide the agent with reward functions in a limited set of situations, such as when a human supervisor is present or in a controlled setting. Can we make use of this limited supervision, and still benefit from the breadth of experience an agent might collect on its own? We formalize this problem as semisupervised reinforcement learning, where the reward function can only be evaluated in a set of "labeled" MDPs, and the agent must generalize its behavior to the wide range of states it might encounter in a set of "unlabeled" MDPs, by using experience from both settings. Our proposed method infers the task objective in the unlabeled MDPs through an algorithm that resembles inverse RL, using the agent's own prior experience in the labeled MDPs as a kind of demonstration of optimal behavior. We evaluate our method on challenging tasks that require control directly from images, and show that our approach can improve the generalization of a learned deep neural network policy by using experience for which no reward function is available. We also show that our method outperforms direct supervised learning of the reward."  
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-3> (39:26) (Levine)

#### ["Policy Distillation"](http://arxiv.org/abs/1511.06295) Rusu, Colmenarejo, Gulcehre, Desjardins, Kirkpatrick, Pascanu, Mnih, Kavukcuoglu, Hadsell
  `policy distillation`
>	"Our new paper uses distillation to consolidate lots of policies into a single deep network. This works remarkably well, and can be applied online, during Q-learning, so that policies are compressed, distilled, and refined whilst being learned. Atari policies are actually improved through distillation and generalize better (with higher scores and lower variance) during novel starting state evaluation."  

#### ["Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning"](http://arxiv.org/abs/1511.06342) Parisotto, Ba, Salakhutdinov
  `policy distillation`
>	"single policy network learning to act in a set of distinct tasks through the guidance of an expert teacher for each task"  
  - `code` <https://github.com/eparisotto/ActorMimic>

----
#### ["Learning to Act by Predicting the Future"](https://arxiv.org/abs/1611.01779) Dosovitskiy, Koltun
  `successor features`
>	"application of deep successor reinforcement learning"  
  - `video` <https://youtube.com/watch?v=947bSUtuSQ0> + <https://youtube.com/watch?v=947bSUtuSQ0> (demo)
  - `video` <https://youtube.com/watch?v=buUF5F8UCH8> (Lamb, Ozair)
  - `video` <https://youtube.com/watch?v=Q0ldKJbAwR8> (Dosovitskiy) `in russian`
  - `video` <https://yadi.sk/i/pMdw-_uI3Gke7Z> (1:02:03) (Shvechikov) `in russian`
  - `post` <https://oreilly.com/ideas/reinforcement-learning-for-complex-goals-using-tensorflow>
  - `post` <https://flyyufelix.github.io/2017/11/17/direct-future-prediction.html>
  - `notes` <https://danieltakeshi.github.io/2017/10/10/learning-to-act-by-predicting-the-future/>
  - `notes` <https://blog.acolyer.org/2017/05/12/learning-to-act-by-predicting-the-future/>
  - `code` <https://github.com/IntelVCL/DirectFuturePrediction>
  - `code` <https://github.com/NervanaSystems/coach/blob/master/agents/dfp_agent.py>
  - `code` <https://github.com/flyyufelix/Direct-Future-Prediction-Keras>

#### ["Successor Features for Transfer in Reinforcement Learning"](http://arxiv.org/abs/1606.05312) Barreto, Munos, Schaul, Silver
  `successor features`
>	"Our approach rests on two key ideas: "successor features", a value function representation that decouples the dynamics of the environment from the rewards, and "generalised policy improvement", a generalisation of dynamic programming’s policy improvement step that considers a set of policies rather than a single one. Put together, the two ideas lead to an approach that integrates seamlessly within the reinforcement learning framework and allows transfer to take place between tasks without any restriction."  

#### ["Deep Successor Reinforcement Learning"](https://arxiv.org/abs/1606.02396) Kulkarni, Saeedi, Gautam, Gershman
  `successor features`
  - `video` <https://youtube.com/watch?v=OCHwXxSW70o> (Kulkarni)
  - `video` <https://youtube.com/watch?v=kNqXCn7K-BM> (Garipov)
  - `code` <https://github.com/Ardavans/DSR>

----
#### ["Learning and Transfer of Modulated Locomotor Controllers"](http://arxiv.org/abs/1610.05182) Heess, Wayne, Tassa, Lillicrap, Riedmiller, Silver
  `modular networks`
  - `video` <https://youtube.com/watch?v=sboPYvhpraQ> (demo)
  - `video` <https://youtube.com/watch?v=0e_uGa7ic74&t=31m4s> (Hadsell)
  - `video` <https://vimeo.com/238221551#t=42m48s> (Hadsell)

#### ["Learning Modular Neural Network Policies for Multi-Task and Multi-Robot Transfer"](http://arxiv.org/abs/1609.07088) Devin, Gupta, Darrell, Abbeel, Levine
  `modular networks`
  - `video` <https://youtube.com/watch?v=n4EgRwzJE1o>
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=56m20s> (Abbeel)

#### ["Learning Invariant Feature Spaces to Transfer Skills with Reinforcement Learning"](https://arxiv.org/abs/1703.02949) Gupta, Devin, Liu, Abbeel, Levine
  `modular networks`

#### ["Progressive Neural Networks"](http://arxiv.org/abs/1606.04671) Rusu, Rabinowitz, Desjardins, Soyer, Kirkpatrick, Kavukcuoglu, Pascanu, Hadsell
  `modular networks`
  - `video` <https://youtube.com/watch?v=aWAP_CWEtSI> (Hadsell)
  - `video` <http://techtalks.tv/talks/progressive-nets-for-sim-to-real-transfer-learning/63043/> (Hadsell)
  - `video` <https://youtu.be/xfyK03MEZ9Q?t=5h47m16s> (Hadsell)
  - `notes` <https://blog.acolyer.org/2016/10/11/progressive-neural-networks/>
  - `code` <https://github.com/synpon/prog_nn>



---
### reinforcement learning - imitation

[interesting older papers - behavioral cloning](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---behavioral-cloning)  
[interesting older papers - inverse reinforcement learning](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---inverse-reinforcement-learning)  

----
#### ["Deep Reinforcement Learning from Human Preferences"](https://arxiv.org/abs/1706.03741) Christiano, Leike, Brown, Martic, Legg, Amodei
  `reinforcement learning from preferences`
  - `video` <https://drive.google.com/drive/folders/0BwcFziBYuA8RM2NTdllSNVNTWTg> (demo)
  - `post` <https://deepmind.com/blog/learning-through-human-feedback/>
  - `post` <https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/>
  - `code` <https://github.com/nottombrown/rl-teacher>

----
#### ["Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards"](https://arxiv.org/abs/1707.08817) Vecerik et al.
  `reinforcement learning from demonstrations`
>	"Our work combines imitation learning with learning from task rewards, so that the agent is able to improve upon the demonstrations it has seen."  
>	"Most work on RL in high-dimensional continuous control problems relies on well-tuned shaping rewards both for communicating the goal to the agent as well as easing the exploration problem. While many of these tasks can be defined by a terminal goal state fairly easily, tuning a proper shaping reward that does not lead to degenerate solutions is very difficult. We replaced these difficult to tune shaping reward functions with demonstrations of the task from a human demonstrator. This eases the exploration problem without requiring careful tuning of shaping rewards."  
  - `video` <https://youtube.com/watch?v=WGJwLfeVN9w> + <https://youtube.com/watch?v=Vno6FGqhvDc> (demo)

#### ["Time-Contrastive Networks: Self-Supervised Learning from Video"](https://arxiv.org/abs/1704.06888) Sermanet, Lynch, Chebotar, Hsu, Jang, Schaal, Levine
  `reinforcement learning from demonstrations` `TCN`
  - <https://sermanet.github.io/imitate/>
  - `video` <https://youtube.com/watch?v=b1UTUQpxPSY>
  - `code` <https://github.com/tensorflow/models/tree/master/research/tcn>

#### ["Learning from Demonstrations for Real World Reinforcement Learning"](https://arxiv.org/abs/1704.03732) Hester et al.
  `reinforcement learning from demonstrations` `DQfD`
>	"DQfD leverages small sets of demonstration data to massively accelerate the learning process even from relatively small amounts of demonstration data and is able to automatically assess the necessary ratio of demonstration data while learning thanks to a prioritized replay mechanism. DQfD works by combining temporal difference updates with supervised classification of the demonstrator's actions."  
  - `video` <https://youtube.com/playlist?list=PLdjpGm3xcO-0aqVf--sBZHxCKg-RZfa5T> (demo)
  - `code` <https://github.com/reinforceio/tensorforce/blob/master/tensorforce/models/dqfd_model.py>
  - `code` <https://github.com/go2sea/DQfD>

#### ["Deeply AggreVaTeD: Differentiable Imitation Learning for Sequential Prediction"](https://arxiv.org/abs/1703.01030) Sun, Venkatraman, Gordon, Boots, Bagnell
  `reinforcement learning from demonstrations` `AggreVaTeD`
>	"a policy gradient extension of DAgger"  
  - `video` <https://vimeo.com/238243230> (Sun)

#### ["Query-Efficient Imitation Learning for End-to-End Autonomous Driving"](https://arxiv.org/abs/1605.06450) Zhang, Cho
  `reinforcement learning from demonstrations` `SafeDAgger`
  - `video` <https://youtu.be/soZXAH3leeQ?t=15m51s> (Cho)

----
#### ["One-Shot Visual Imitation Learning via Meta-Learning"](https://arxiv.org/abs/1709.04905) Finn, Yu, Zhang, Abbeel, Levine
  `imitation learning from visual observations` `meta-learning`
  - `video` <https://youtu.be/lYU5nq0dAQQ?t=51m> (Levine)
  - `code` <https://github.com/tianheyu927/mil>

#### ["One-Shot Imitation Learning"](http://arxiv.org/abs/1703.07326) Duan, Andrychowicz, Stadie, Ho, Schneider, Sutskever, Abbeel, Zaremba
  `imitation learning` `meta-learning`
  - `video` <http://bit.ly/one-shot-imitation> (demo)
  - `post` <https://blog.openai.com/robots-that-learn/>
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/> (1:03:10) (de Freitas)
  - `notes` <https://medium.com/incogito/openais-new-approach-for-one-shot-imitation-learning-a-sneak-peak-into-the-future-of-ai-efcdddca8e2e>
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/One-Shot_Imitation_Learning.md>

----
#### ["Imitation from Observation: Learning to Imitate Behaviors from Raw Video via Context Translation"](https://arxiv.org/abs/1707.03374) Liu, Gupta, Abbeel, Levine
  `imitation learning from visual observations`
>	"Standard imitation learning methods assume that the agent receives examples of observation-action tuples that could be provided, for instance, to a supervised learning algorithm. This stands in contrast to how humans and animals imitate: we observe another person performing some behavior and then figure out which actions will realize that behavior, compensating for changes in viewpoint, surroundings, and embodiment. We term this kind of imitation learning as imitation-from-observation and propose an imitation learning method based on video prediction with context translation and deep reinforcement learning. This lifts the assumption in imitation learning that the demonstration should consist of observations and actions in the same environment, and enables a variety of interesting applications, including learning robotic skills that involve tool use simply by observing videos of human tool use."  
  - <https://sites.google.com/site/imitationfromobservation/>
  - `video` <https://youtube.com/watch?v=kJBRDhInbmU> (demo)

#### ["Third Person Imitation Learning"](https://arxiv.org/abs/1703.01703) Stadie, Abbeel, Sutskever
  `imitation learning from visual observations` `adversarial imitation learning`
>	"Learning a policy from third-person experience is different from standard imitation learning which assumes the same "viewpoint" for teacher and student. The authors build upon Generative Adversarial Imitation Learning, which uses a GAN framework to discriminate between teacher and student experience and force the student to behave close to the teacher. However, when using third-person experience from a different viewpoint the discriminator would simply learn to discriminate between viewpoints instead of behavior and the framework isn't easily applicable. The authors' solution is to add a second discriminator to maximize a domain confusion loss based on the same feature representation. The objective is to learn the same (viewpoint-independent) feature representation for both teacher and student experience while also learning to discriminate between teacher and student observations. In other words, the objective is to maximize domain confusion while minimizing class loss."  
  - `video` <http://www.fields.utoronto.ca/video-archive/2017/02/2267-16530> (48:45) (Abbeel)
  - `notes` <http://ruotianluo.github.io/2016/11/20/third-person/>
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Third-Person_Imitation_Learning.md>

#### ["Unsupervised Perceptual Rewards for Imitation Learning"](http://arxiv.org/abs/1612.06699) Sermanet, Xu, Levine
  `imitation learning from visual observations`
>	"To our knowledge, these are the first results showing that complex robotic manipulation skills can be learned directly and without supervised labels from a video of a human performing the task."  

----
#### ["Learning Robust Rewards with Adversarial Inverse Reinforcement Learning"](https://arxiv.org/abs/1710.11248) Fu, Luo, Levine
  `adversarial imitation learning`

#### ["Multi-Modal Imitation Learning from Unstructured Demonstrations using Generative Adversarial Nets"](https://arxiv.org/abs/1705.10479) Hausman, Chebotar, Schaal, Sukhatme, Lim
  `adversarial imitation learning`
>	"Imitation learning has traditionally been applied to learn a single task from demonstrations thereof. The requirement of structured and isolated demonstrations limits the scalability of imitation learning approaches as they are difficult to apply to real-world scenarios, where robots have to be able to execute a multitude of tasks. In this paper, we propose a multi-modal imitation learning framework that is able to segment and imitate skills from unlabelled and unstructured demonstrations by learning skill segmentation and imitation learning jointly."  
>	"The presented approach learns the notion of intention and is able to perform different tasks based on the policy intention input."  
>	"We consider a possibility to discover different skills that can all start from the same initial state, as opposed to hierarchical reinforcement learning where the goal is to segment a task into a set of consecutive subtasks. We demonstrate that our method may be used to discover the hierarchical structure of a task similarly to the hierarchical reinforcement learning approaches."  
  - <http://sites.google.com/view/nips17intentiongan> (demo)
  - `video` <https://youtu.be/xfyK03MEZ9Q?t=7h43m18s> (Hausman)

#### ["Inferring the Latent Structure of Human Decision-Making from Raw Visual Inputs"](https://arxiv.org/abs/1703.08840) Li, Song, Ermon
  `adversarial imitation learning`
>	"Expert demonstrations provided by humans often show significant variability due to latent factors that are typically not explicitly modeled. We propose a new algorithm that can infer the latent structure of expert demonstrations in an unsupervised way. Our method, built on top of GAIL, can not only imitate complex behaviors, but also learn interpretable and meaningful representations of complex behavioral data, including visual demonstrations, and recover semantically meaningful factors of variation in the data."  
  - `video` <https://youtube.com/watch?v=YtNPBAW6h5k> (demo)
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Inferring_The_Latent_Structure_of_Human_Decision-Making_from_Raw_Visual_Inputs.md>
  - `code` <https://github.com/YunzhuLi/InfoGAIL>

#### ["Robust Imitation of Diverse Behaviors"](https://arxiv.org/abs/1707.02747) Wang, Merel, Reed, Wayne, Freitas, Heess
  `adversarial imitation learning`
>	"We develop a new version of GAIL that (1) is much more robust than the purely-supervised controller, especially with few demonstrations, and (2) avoids mode collapse, capturing many diverse behaviors when GAIL on its own does not."  
>	"The base of our model is a new type of variational autoencoder on demonstration trajectories that learns semantic policy embeddings, which can be smoothly interpolated with a resulting smooth interpolation of reaching behavior."  
  - `post` <https://deepmind.com/blog/producing-flexible-behaviours-simulated-environments/>
  - `video` <https://youtube.com/watch?v=necs0XfnFno> (demo)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/> (1:16:00) (de Freitas)
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Robust_Imitation_of_Diverse_Behaviors.md>

#### ["End-to-End Differentiable Adversarial Imitation Learning"](http://proceedings.mlr.press/v70/baram17a.html) Baram, Anschel, Caspi, Mannor
  `adversarial imitation learning`
>	"Model-free approach does not allow the system to be differentiable, which requires the use of high-variance gradient estimations."  
>	"We show how to use a forward model to make the system fully differentiable, which enables us to train policies using the stochastic gradient of discriminator."  
  - `video` <https://vimeo.com/237272985> (Baram)
  - `slides` <http://icri-ci.technion.ac.il/files/2017/05/14-Shie-Mannor-170509.pdf>
  - `code` <https://github.com/itaicaspi/mgail>

#### ["Generative Adversarial Imitation Learning"](http://arxiv.org/abs/1606.03476) Ho, Ermon
  `adversarial imitation learning` `GAIL`
>	"Uses a GAN framework to discriminate between teacher and student experience and force the student to behave close to the teacher."  
  - `video` <https://youtube.com/watch?v=bcnCo9RxhB8> (Ermon)
  - `video` <https://youtu.be/d9DlQSJQAoI?t=22m12s> (Finn)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/> (1:09:43) (de Freitas)
  - `notes` <http://tsong.me/blog/gail/>
  - `code` <https://github.com/openai/imitation>
  - `code` <https://github.com/DanielTakeshi/rl_algorithms/tree/master/il>

#### ["A Connection between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models"](https://arxiv.org/abs/1611.03852) Finn, Christiano, Abbeel, Levine
  `adversarial imitation learning` `maximum entropy inverse reinforcement learning`
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#a-connection-between-generative-adversarial-networks-inverse-reinforcement-learning-and-energy-based-models-finn-christiano-abbeel-levine>

#### ["Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization"](https://arxiv.org/abs/1603.00448) Finn, Levine, Abbeel
  `maximum entropy inverse reinforcement learning`
>	"technique that lets one apply Maximum Entropy Inverse Optimal Control without the double-loop procedure and using policy gradient techniques"  
  - `video` <https://youtube.com/watch?v=hXxaepw0zAw> (demo)
  - `video` <http://techtalks.tv/talks/guided-cost-learning-deep-inverse-optimal-control-via-policy-optimization/62472/> (Finn)
  - `video` <https://youtu.be/d9DlQSJQAoI?t=18m17s> (Finn)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-3> (22:48) (Levine)



---
### reinforcement learning - multi-agent

----
#### ["A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning"](https://arxiv.org/abs/1711.00832) Lanctot, Zambaldi, Gruslys, Lazaridou, Tuyls, Perolat, Silver, Graepel
>	"We first observe that independent reinforcement learners produce policies that can be jointly correlated, failing to generalize well during execution with other agents. We quantify this effect by proposing a new metric called joint policy correlation. We then propose an algorithm motivated by game-theoretic foundations, which generalises several previous approaches such as fictitious play, iterated best response, independent RL, and double oracle. We show that our algorithm can reduce joint policy correlation significantly in first-person coordination games, and finds robust counter-strategies in a common poker benchmark game."  

#### ["Learning Nash Equilibrium for General-Sum Markov Games from Batch Data"](https://arxiv.org/abs/1606.08718) Perolat, Strub, Piot, Pietquin
>	"We address the problem of learning a Nash equilibrium in γ-discounted multiplayer general-sum Markov Games in a batch setting. As the number of players increases in MG, the agents may either collaborate or team apart to increase their final rewards. One solution to address this problem is to look for a Nash equilibrium. Although, several techniques were found for the subcase of two-player zero-sum MGs, those techniques fail to find a Nash equilibrium in general-sum Markov Games."  
>	"We introduce a new (weaker) definition of ε-Nash equilibrium in MGs which grasps the strategy’s quality for multiplayer games. We prove that minimizing the norm of two Bellman-like residuals implies to learn such an ε-Nash equilibrium. Then, we show that minimizing an empirical estimate of the Lp norm of these Bellman-like residuals allows learning for general-sum games within the batch setting. Finally, we introduce a neural network architecture that successfully learns a Nash equilibrium in generic multiplayer general-sum turn-based MGs."  

----
#### ["Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"](https://arxiv.org/abs/1706.02275) Lowe, Wu, Tamar, Harb, Abbeel, Mordatch
  - `video` <https://youtube.com/watch?v=QCmBo91Wy64> (demo)
  - `code` <https://github.com/openai/multiagent-particle-envs>

#### ["Counterfactual Multi-Agent Policy Gradients"](https://arxiv.org/abs/1705.08926) Foerster, Farquhar, Afouras, Nardelli, Whiteson
>	"We evaluate COMA in the testbed of StarCraft unit micromanagement, using a decentralised variant with significant partial observability. COMA significantly improves average performance over other multi-agent actor-critic methods in this setting, and the best performing agents are competitive with state-of-the-art centralised controllers that get access to the full state."  
  - `video` <https://youtube.com/watch?v=3OVvjE5B9LU> (Whiteson)

#### ["Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning"](https://arxiv.org/abs/1702.08887) Foerster, Nardelli, Farquhar, Afouras, Torr, Kohli, Whiteson
  - `video` <https://vimeo.com/238243859> (Foerster)

----
#### ["Emergent Complexity via Multi-Agent Competition"](https://arxiv.org/abs/1710.03748) Bansal, Pachocki, Sidor, Sutskever, Mordatch
  - `blog` <https://blog.openai.com/competitive-self-play/> (demo)
  - `code` <https://github.com/openai/multiagent-competition>

#### ["Learning with Opponent-Learning Awareness"](https://arxiv.org/abs/1709.04326) Foerster, Chen, Al-Shedivat, Whiteson, Abbeel, Mordatch
  - `blog` <https://blog.openai.com/learning-to-model-other-minds/> (demo)



---
### program induction

----
#### ["Learning Explanatory Rules from Noisy Data"](https://arxiv.org/abs/1711.04574) Evans, Grefenstette

#### ["Learning Neural Programs To Parse Programs"](https://arxiv.org/abs/1706.01284) Chen, Liu, Song
>	"We explore a new direction to learn domain-specific programs significantly more complex than previously considered in the literature of learning programs from input-output examples only. In particular, we consider an exemplary problem to learn a program to parse an input satisfying a context-free grammar into its abstract syntax tree. This problem is challenging when the underlying grammar is unknown, and only input-output examples are provided. The program to be synthesized in this problem, i.e., a parser, is more complex than programs consisting of string operations as in many previous work, and thus serves as a good next step challenge to tackle in the domain of learning programs from input-output examples."  
>	"Recent works propose to use sequence-to-sequence models to directly generate parse trees from inputs. However, they often do not generalize well, and our experiments show that their test accuracy is almost 0% on inputs longer than those seen in training."  
>	"This work is the first successful demonstration that reinforcement learning can be applied to train a neural program operating a non-differentiable machine with input-output pairs only, while the learned neural program can fully generalize to longer inputs on a non-trivial task."  
>	"To show that our approach is general and can learn to parse different types of context-free languages using the same architecture and approach, we evaluate it on learning the parsing programs for an imperative language and a functional one, and demonstrate that our approach can successfully learn both of them, and the learned programs can achieve 100% on test set whose inputs are 100x longer than training samples."  
>	"We propose a new approach to learn a hybrid program, a differentiable neural program operating a domain-specific non-differentiable machine, from input-output examples only. Learning such a hybrid program combines the advantage of both differentiable and non-differentiable machines to enable learning more complex programs."  
>	"First, we propose LL machines as an example domain-specific non-differentiable machine to be operated by neural programs, for learning parsers. Intuitively, an LL machine provides a high-level abstraction to regularize the learned programs to be within the space of LL(1) parsers. The instructions provided by an LL machine provide richer semantic information than the primitives considered in previous works, so that the learning algorithm can take advantage of such information to learn more complex programs."  
>	"Second, we propose novel reinforcement learning-based techniques to train a neural program. Specifically, we solve the training problem in two phases: (1) we search for a valid execution trace set for each input-output example; then (2) we search for a set of input-output-trace combinations, so that a neural program can be trained to fit all training examples."  
  - `notes` <https://github.com/carpedm20/paper-notes/blob/master/notes/neural-ll-parser.md>
  - `code` <http://github.com/liuchangacm/neuralparser>

#### ["RobustFill: Neural Program Learning under Noisy I/O"](https://arxiv.org/abs/1703.07469) Devlin, Uesato, Bhupatiraju, Singh, Mohamed, Kohli
  - `video` <https://vimeo.com/238227939> (Uesato, Bhupatiraju)
  - `video` <https://facebook.com/NIPSlive/videos/1489196207802887/> (31:06) (Reed)

#### ["Differentiable Programs with Neural Libraries"](https://arxiv.org/abs/1611.02109) Gaunt, Brockschmidt, Kushman, Tarlow
  - `video` <https://vimeo.com/238227833> (Gaunt)

#### ["Neuro-Symbolic Program Synthesis"](https://arxiv.org/abs/1611.01855) Parisotto, Mohamed, Singh, Li, Zhou, Kohli

#### ["TerpreT: A Probabilistic Programming Language for Program Induction"](http://arxiv.org/abs/1608.04428) Gaunt, Brockschmidt, Singh, Kushman, Kohli, Taylor, Tarlow
>	"These works raise questions of (a) whether new models can be designed specifically to synthesize interpretable source code that may contain looping and branching structures, and (b) whether searching over program space using techniques developed for training deep neural networks is a useful alternative to the combinatorial search methods used in traditional IPS. In this work, we make several contributions in both of these directions."  
>	"Shows that differentiable interpreter-based program induction is inferior to discrete search-based techniques used by the programming languages community. We are then left with the question of how to make progress on program induction using machine learning techniques."  
  - `video` <https://youtu.be/vzDuVhFMB9Q?t=2m40s> (Gaunt)

#### ["Programming with a Differentiable Forth Interpreter"](http://arxiv.org/abs/1605.06640) Bošnjak, Rocktaschel, Naradowsky, Riedel
  `learning details of probabilistic program`
>	"The paper talks about a certain class of neural networks that incorporate procedural knowledge. The way they are constructed is by compiling Forth code (procedural) to TensorFlow expressions (linear algebra) to be able to train slots (missing pieces in the code) end-to-end from input-output pairs using backpropagation."  
  - `video` <https://vimeo.com/238227890> (Bosnjak)
  - `video` <https://facebook.com/NIPSlive/videos/1489196207802887/> (26:57) (Reed)

#### ["Making Neural Programming Architectures Generalize via Recursion"](https://arxiv.org/abs/1704.06611) Cai, Shin, Song
  `Neural Programmer-Interpreter with recursion`
>	"We implement recursion in the Neural Programmer-Interpreter framework on four tasks: grade-school addition, bubble sort, topological sort, and quicksort."  
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255/> (49:59) (Cai)
  - `notes` <https://theneuralperspective.com/2017/03/14/making-neural-programming-architecture-generalize-via-recursion/>

#### ["Adaptive Neural Compilation"](http://arxiv.org/abs/1605.07969) Bunel, Desmaison, Kohli, Torr, Kumar



---
### reasoning

[interesting older papers - reasoning](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---reasoning)  
[interesting older papers - question answering over knowledge bases](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-knowledge-bases)  
[interesting older papers - question answering over texts](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-texts)  

----
#### ["End-to-end Differentiable Proving"](https://arxiv.org/abs/1705.11040) Rocktaschel, Riedel
  `learning logic`
>	"We introduce neural networks for end-to-end differentiable theorem proving that operate on dense vector representations of symbols. These neural networks are constructed recursively by taking inspiration from the backward chaining algorithm as used in Prolog. Specifically, we replace symbolic unification with a differentiable computation on vector representations of symbols using a radial basis function kernel, thereby combining symbolic reasoning with learning subsymbolic vector representations. By using gradient descent, the resulting neural network can be trained to infer facts from a given incomplete knowledge base. It learns to (i) place representations of similar symbols in close proximity in a vector space, (ii) make use of such similarities to prove facts, (iii) induce logical rules, and (iv) use provided and induced logical rules for complex multi-hop reasoning. We demonstrate that this architecture outperforms ComplEx, a state-of-the-art neural link prediction model, on four benchmark knowledge bases while at the same time inducing interpretable function-free first-order logic rules."  
>	"Aim:  
>	- Neural network for proving queries to a knowledge base  
>	- Proof success differentiable w.r.t. vector representations of symbols  
>	- Learn vector representations of symbols end-to-end from proof success  
>	- Make use of provided rules in soft proofs  
>	- Induce interpretable rules end-to-end from proof success"  
>	"Limitations:  
>	- knowledge bases with <10k facts  
>	- small proof depth"  
  - `video` <https://facebook.com/nipsfoundation/videos/1554402331317667> (28:51)
  - `poster` <https://rockt.github.io/pdf/rocktaschel2017end-poster.pdf>
  - `slides` <https://rockt.github.io/pdf/rocktaschel2017end-slides.pdf> (Rocktaschel)
  - `slides` <http://aitp-conference.org/2017/slides/Tim_aitp.pdf> (Rocktaschel)
  - `slides` <http://on-demand.gputechconf.com/gtc-eu/2017/presentation/23372-tim-rocktäschel-gpu-accelerated-deep-neural-networks-for-end-to-end-differentiable-planning-and-reasoning.pdf> (Rocktaschel)
  - `audio` <https://soundcloud.com/nlp-highlights/19a> (Rocktaschel)
  - `paper` ["Learning Knowledge Base Inference with Neural Theorem Provers"](http://akbc.ws/2016/papers/14_Paper.pdf) by Rocktaschel and Riedel

#### ["Differentiable Learning of Logical Rules for Knowledge Base Completion"](https://arxiv.org/abs/1702.08367) Yang, Yang, Cohen
  `learning logic` `TensorLog`
>	"We study the problem of learning probabilistic first-order logical rules for knowledge base reasoning. This learning problem is difficult because it requires learning the parameters in a continuous space as well as the structure in a discrete space. We propose a framework, Neural Logic Programming, that combines the parameter and structure learning of first-order logical rules in an end-to-end differentiable model. This approach is inspired by a recently-developed differentiable logic called TensorLog, where inference tasks can be compiled into sequences of differentiable operations. We design a neural controller system that learns to compose these operations."  
  - `code` <https://github.com/fanyangxyz/Neural-LP>

#### ["TensorLog: A Differentiable Deductive Database"](http://arxiv.org/abs/1605.06523) Cohen
  `learning logic` `TensorLog`
  - `slides` <http://starai.org/2016/slides/william-cohen.pptx>
  - `code` <https://github.com/TeamCohen/TensorLog>

#### ["Learning Continuous Semantic Representations of Symbolic Expressions"](https://arxiv.org/abs/1611.01423) Allamanis, Chanthirasegaran, Kohli, Sutton
  `learning logic`
>	"We propose a new architecture, called neural equivalence networks, for the problem of learning continuous semantic representations of algebraic and logical expressions. These networks are trained to represent semantic equivalence, even of expressions that are syntactically very different. The challenge is that semantic representations must be computed in a syntax-directed manner, because semantics is compositional, but at the same time, small changes in syntax can lead to very large changes in semantics, which can be difficult for continuous neural architectures."  
  - <http://groups.inf.ed.ac.uk/cup/semvec/> (demo)
  - `video` <https://vimeo.com/238222290> (Sutton)
  - `code` <https://github.com/mast-group/eqnet>

----
#### ["Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning"](https://arxiv.org/abs/1711.05851) Das, Dhuliawala, Zaheer, Vilnis, Durugkar, Krishnamurthy, Smola, McCallum
  `question answering over knowledge bases`

#### ["Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks"](https://arxiv.org/abs/1704.08384) Das, Zaheer, Reddy, McCallum
  `question answering over knowledge bases`
  - `video` <https://youtu.be/lc68_d_DnYs?t=7m28s> (Neelakantan)
  - `code` <https://github.com/rajarshd/TextKBQA>

#### ["Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision"](http://arxiv.org/abs/1611.00020) Liang, Berant, Le, Forbus, Lao
  `question answering over knowledge bases`
>	"We propose the Manager-Programmer-Computer framework, which integrates neural networks with non-differentiable memory to support abstract, scalable and precise operations through a friendly neural computer interface. Specifically, we introduce a Neural Symbolic Machine, which contains a sequence-to-sequence neural "programmer", and a non-differentiable "computer" that is a Lisp interpreter with code assist."  
  - `notes` <https://northanapon.github.io/papers/2017/01/16/neural-symbolic-machine.html>
  - `notes` <https://github.com/carpedm20/paper-notes/blob/master/notes/neural-symbolic-machine.md>

#### ["Learning a Natural Language Interface with Neural Programmer"](http://arxiv.org/abs/1611.08945) Neelakantan, Le, Abadi, McCallum, Amodei
  `question answering over knowledge bases`
  - `video` <http://youtu.be/lc68_d_DnYs?t=24m44s> (Neelakantan)
  - `code` <https://github.com/tensorflow/models/tree/master/research/neural_programmer>

----
#### ["Multi-Mention Learning for Reading Comprehension with Neural Cascades"](https://arxiv.org/abs/1711.00894) Swayamdipta, Parikh, Kwiatkowski
  `question answering over texts` `documents collection`

#### ["Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering"](https://arxiv.org/abs/1711.05116) Wang et al.
  `question answering over texts` `documents collection`

#### ["R3: Reinforced Reader-Ranker for Open-Domain Question Answering"](https://arxiv.org/abs/1709.00023) Wang et al.
  `question answering over texts` `documents collection`
>	"First, we propose a new pipeline for open-domain QA with a Ranker component, which learns to rank retrieved passages in terms of likelihood of extracting the ground-truth answer to a given question. Second, we propose a novel method that jointly trains the Ranker along with an answer-extraction Reader model, based on reinforcement learning."  

#### ["Reading Wikipedia to Answer Open-Domain Questions"](https://arxiv.org/abs/1704.00051) Chen, Fisch, Weston, Bordes
  `question answering over texts` `documents collection` `DrQA`
  - `code` <https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa>
  - `code` <https://github.com/hitvoice/DrQA>

#### ["Simple and Effective Multi-Paragraph Reading Comprehension"](https://arxiv.org/abs/1710.10723) Clark, Gardner
  `question answering over texts` `multi-paragraph document`

#### ["Coarse-to-Fine Question Answering for Long Documents"](http://arxiv.org/abs/1611.01839) Choi, Hewlett, Lacoste, Polosukhin, Uszkoreit, Berant
  `question answering over texts` `multi-paragraph document`

#### ["Key-Value Memory Networks for Directly Reading Documents"](http://arxiv.org/abs/1606.03126) Miller, Fisch, Dodge, Amir-Hossein Karimi, Bordes, Weston
  `question answering over texts` `multi-paragraph document`
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1606.03126>
  - `code` <https://github.com/facebook/MemNN/blob/master/KVmemnn>

#### ["R-NET: Machine Reading Comprehension with Self-matching Networks"](https://microsoft.com/en-us/research/publication/mrc/) Wang et al.
  `question answering over texts` `single paragraph`
  - `post` <http://yerevann.github.io/2017/08/25/challenges-of-reproducing-r-net-neural-network-using-keras/>
  - `code` <https://github.com/YerevaNN/R-NET-in-Keras>
  - `code` <https://github.com/minsangkim142/R-net>
  - `paper` ["Gated Self-Matching Networks for Reading Comprehension and Question Answering"](http://aclweb.org/anthology/P17-1018) by Wang et al.

#### ["Tracking the World State with Recurrent Entity Networks"](https://arxiv.org/abs/1612.03969) Henaff, Weston, Szlam, Bordes, LeCun
  `question answering over texts` `single paragraph`
>	"There's a bunch of memory slots that each can be used to represent a single entity. The first time an entity appears, it's written to a slot. Every time that something happens in the story that corresponds to a change in the state of an entity, the change in the state of that entity is combined with the entity's previous state via a modified GRU update equation and rewritten to the same slot."  
  - `code` <https://github.com/jimfleming/recurrent-entity-networks>
  - `code` <https://github.com/siddk/entity-network>

#### ["Gated-Attention Readers for Text Comprehension"](http://arxiv.org/abs/1606.01549) Dhingra, Liu, Yang, Cohen, Salakhutdinov
  `question answering over texts` `single paragraph`
  - `video` <https://youtube.com/watch?v=ZSDrM-tuOiA> (Salakhutdinov)
  - `notes` <https://theneuralperspective.com/2017/01/19/gated-attention-readers-for-text-comprehension/>
  - `code` <https://github.com/bdhingra/ga-reader>

#### ["Machine Comprehension Using Match-LSTM And Answer Pointer"](http://arxiv.org/abs/1608.07905) Wang, Jiang
  `question answering over texts` `single paragraph`

----
#### ["A Generative Vision Model that Trains with High Data Efficiency and Breaks Text-based CAPTCHAs"](http://science.sciencemag.org/content/early/2017/10/26/science.aag2612.full) George et al.
  `question answering over images`
>	"Learning from few examples and generalizing to dramatically different situations are capabilities of human visual intelligence that are yet to be matched by leading machine learning models. By drawing inspiration from systems neuroscience, we introduce a probabilistic generative model for vision in which message-passing based inference handles recognition, segmentation and reasoning in a unified way. The model demonstrates excellent generalization and occlusion-reasoning capabilities, and outperforms deep neural networks on a challenging scene text recognition benchmark while being 300-fold more data efficient. In addition, the model fundamentally breaks the defense of modern text-based CAPTCHAs by generatively segmenting characters without CAPTCHA-specific heuristics. Our model emphasizes aspects like data efficiency and compositionality that may be important in the path toward general artificial intelligence."  
  - `post` <https://vicarious.com/2017/10/26/common-sense-cortex-and-captcha/>
  - `code` <https://github.com/vicariousinc/science_rcn>

#### ["Recurrent Relational Networks for Complex Relational Reasoning"](https://arxiv.org/abs/1711.08028) Palm, Paquet, Winther
  `question answering over images`
>	"We introduce the recurrent relational network which can solve tasks requiring an order of magnitude more steps of reasoning than the relational network. We apply it to solving Sudoku puzzles and achieve state-of-the-art results solving 96.6% of the hardest Sudoku puzzles. For comparison the relational network fails to solve any puzzles. We also apply our model to the BaBi textual QA dataset solving 19/20 tasks which is competitive with state- of-the-art sparse differentiable neural computers. The recurrent relational network is a general purpose module that can be added to any neural network model to add a powerful relational reasoning capacity."  
>	"Both relational networks, interaction networks and our proposed model can be seen as an instance of Graph Neural Networks. Our main contribution is showing how these can be used for complex relational reasoning."  
>	"Our model can be seen as a completely learned message passing algorithm. Belief propagation is a hand-crafted message passing algorithm for performing exact inference in directed acyclic graphical models. If the graph has cycles, one can use a variant, loopy belief propagation, but it is not guaranteed to be exact, unbiased or even converge. Empirically it works well though and it is widely used."  
>	"There are plenty of algorithms out there for solving Sudokus. The RRN differs from these traditional algorithms in two important ways:  
>	- It is a neural network module that learns an algorithm from data rather than being hand-crafted.  
>	- It can be added to any other neural network and trained end-to-end to add a complex relational reasoning capacity."  
>	"We trained a RRN to solve Sudokus by considering each cell an object, which affects each other cell in the same row, column and box. We didn’t tell it about any strategy or gave it any other hints. The network learned a powerful strategy which solves 96.6% of even the hardest Sudoku’s with only 17 givens. For comparison the non-recurrent RN failed to solve any of these puzzles, despite having more parameters and being trained for longer."  
  - `post` <https://rasmusbergpalm.github.io/recurrent-relational-networks/>
  - `code` <https://github.com/rasmusbergpalm/recurrent-relational-networks>

#### ["FiLM: Visual Reasoning with a General Conditioning Layer"](https://arxiv.org/abs/1709.07871) Perez, Strub, Vries, Dumoulin, Courville
  `question answering over images`
>	"FiLM layer carries out a simple, feature-wise affine transformation on a neural network’s intermediate features, conditioned on an arbitrary input. In the case of visual reasoning, FiLM layers enable a RNN over an input question to influence CNN computation over an image. This process adaptively and radically alters the CNN’s behavior as a function of the input question, allowing the overall model to carry out a variety of reasoning tasks, ranging from counting to comparing. It also enables the CNN to properly localize question-referenced objects."  
>	"Ability to answer image-related questions requires learning a question-dependent, structured reasoning process over images from language. Standard deep learning approaches tend to exploit biases in the data rather than learn this underlying structure, while leading methods learn to visually reason successfully but are hand-crafted for reasoning."  
>	"The crazy thing is that the model does not include anything for reasoning and does not indicate anything about reasoning."  
  - `video` <https://youtu.be/02xIkHowQOk?t=2h44m55s> (Perez)
  - `video` <https://youtube.com/watch?v=BZKzHAOilNo> (Courville)

#### ["A Simple Neural Network Module for Relational Reasoning"](https://arxiv.org/abs/1706.01427) Santoro, Raposo, Barrett, Malinowski, Pascanu, Battaglia, Lillicrap
  `question answering over images`
  - `post` <https://deepmind.com/blog/neural-approach-relational-reasoning/>
  - `video` <https://youtube.com/channel/UCIAnkrNn45D0MeYwtVpmbUQ> (demo)
  - `video` <https://youtu.be/02xIkHowQOk?t=2h38m> (Kahou)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1706.01427>
  - `code` <https://github.com/kimhc6028/relational-networks>
  - `code` <https://github.com/gitlimlab/Relation-Network-Tensorflow>
  - `code` <https://github.com/Alan-Lee123/relation-network>

#### ["Inferring and Executing Programs for Visual Reasoning"](https://arxiv.org/abs/1705.03633) Johnson, Hariharan, Maaten, Hoffman, Fei-Fei, Zitnick, Girshick
  `question answering over images`
  - `video` <https://youtube.com/watch?v=3pCLma2FqSk> (Johnson)
  - `video` <https://youtu.be/02xIkHowQOk?t=2h49m1s> (Perez)
  - `code` <https://github.com/facebookresearch/clevr-iep>

#### ["Learning to Reason: End-to-End Module Networks for Visual Question Answering"](https://arxiv.org/abs/1704.05526) Hu, Andreas, Rohrbach, Darrell, Saenko
  `question answering over images`
  - <http://ronghanghu.com/n2nmn/>
  - `post` <http://bair.berkeley.edu/blog/2017/06/20/learning-to-reason-with-neural-module-networks>
  - `video` <https://youtu.be/ejQNdTdyTBM?t=28m8s> (Kretov) `in russian`
  - `code` <https://github.com/ronghanghu/n2nmn>
  - `code` <https://github.com/tensorflow/models/tree/master/research/qa_kg>

----
#### ["Learning to Perform Physics Experiments via Deep Reinforcement Learning"](http://arxiv.org/abs/1611.01843) Denil, Agrawal, Kulkarni, Erez, Battaglia, Freitas
  `question answering over 3D world`
>	"We introduce a basic set of tasks that require agents to estimate properties such as mass and cohesion of objects in an interactive simulated environment where they can manipulate the objects and observe the consequences. We found that state of art deep reinforcement learning methods can learn to perform the experiments necessary to discover such hidden properties. By systematically manipulating the problem difficulty and the cost incurred by the agent for performing experiments, we found that agents learn different strategies that balance the cost of gathering information against the cost of making mistakes in different situations."  
  - `video` <https://youtu.be/SAcHyzMdbXc?t=16m6s> (de Freitas)



---
### language grounding

----
#### ["A Paradigm for Situated and Goal-Driven Language Learning"](https://arxiv.org/abs/1610.03585) Gauthier, Mordatch
  `goal-driven language learning`
  - `post` ["On 'Solving Language'"](http://foldl.me/2016/solving-language/) (Gauthier)
  - `post` ["Situated Language Learning"](http://foldl.me/2016/situated-language-learning/) (Gauthier)

#### ["Learning with Latent Language"](https://arxiv.org/abs/1711.00482) Andreas, Klein, Levine
  `goal-driven language learning`
>	"optimizing models in a space parameterized by natural language"  
>	"Using standard neural encoder–decoder components to build models for representation and search in this space, we demonstrated that our approach outperforms strong baselines on classification, structured prediction and reinforcement learning tasks."  
>	"The approach outperforms both multi-task and meta-learning approaches that map directly from training examples to outputs by way of a real-valued parameterization, as well as approaches that make use of natural language annotations as an additional supervisory signal rather than an explicit latent parameter. The natural language concept descriptions inferred by our approach often agree with human annotations when they are correct, and provide an interpretable debugging signal when incorrect. In short, by equipping models with the ability to “think out loud” when learning, they become both more comprehensible and more accurate."  
>	"- Language encourages compositional generalization: standard deep learning architectures are good at recognizing new instances of previously encountered concepts, but not always at generalizing to new ones. By forcing decisions to pass through a linguistic bottleneck in which the underlying compositional structure of concepts is explicitly expressed, stronger generalization becomes possible.  
>	- Language simplifies structured exploration: relatedly, linguistic scaffolding can provide dramatic advantages in problems like reinforcement learning that require exploration: models with latent linguistic parameterizations can sample in this space, and thus limit exploration to a class of behaviors that are likely a priori to be goal-directed and interpretable.  
>	- Language can help learning: in multitask settings, it can even improve learning on tasks for which no language data is available at training or test time. While some of these advantages are also provided by techniques like program synthesis that are built on top of formal languages, natural language is at once more expressive and easier to obtain than formal supervision."  
  - `post` <http://blog.jacobandreas.net/fake-language.html>
  - `code` <http://github.com/jacobandreas/l3>

#### ["Grounded Language Learning in a Simulated 3D World"](https://arxiv.org/abs/1706.06551) Hermann et al.
  `goal-driven language learning`
>	"The agent learns simple language by making predictions about the world in which that language occurs, and by discovering which combinations of words, perceptual cues and action decisions result in positive outcomes. Its knowledge is distributed across language, vision and policy networks, and pertains to modifiers, relational concepts and actions, as well as concrete objects. Its semantic representations enable the agent to productively interpret novel word combinations, to apply known relations and modifiers to unfamiliar objects and to re-use knowledge pertinent to the concepts it already has in the process of acquiring new concepts."  
>	"While our simulations focus on language, the outcomes are relevant to machine learning in a more general sense. In particular, the agent exhibits active, multi-modal concept induction, the ability to transfer its learning and apply its knowledge representations in unfamiliar settings, a facility for learning multiple, distinct tasks, and the effective synthesis of unsupervised and reinforcement learning. At the same time, learning in the agent reflects various effects that are characteristic of human development, such as rapidly accelerating rates of vocabulary growth, the ability to learn from both rewarded interactions and predictions about the world, a natural tendency to generalise and re-use semantic knowledge, and improved outcomes when learning is moderated by curricula."  
  - `video` <https://youtube.com/watch?v=wJjdu1bPJ04> (demo)
  - `video` <http://videolectures.net/deeplearning2017_blunsom_language_processing/> (48:54) (Blunsom)

#### ["Programmable Agents"](https://arxiv.org/abs/1706.06383) Denil, Colmenarejo, Cabi, Saxton, Freitas
  `goal-driven language learning`
  - `video` <https://youtube.com/playlist?list=PLs1LSEoK_daRDnPUB2u7VAXSonlNU7IcV> (demo)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/> (32:57) (de Freitas)
  - `code` <https://github.com/jaesik817/programmable-agents_tensorflow>

#### ["Gated-Attention Architectures for Task-Oriented Language Grounding"](https://arxiv.org/abs/1706.07230) Chaplot, Sathyendra, Pasumarthi, Rajagopal, Salakhutdinov
  `goal-driven language learning`

#### ["A Deep Compositional Framework for Human-like Language Acquisition in Virtual Environment"](https://arxiv.org/abs/1703.09831) Yu, Zhang, Xu
  `goal-driven language learning`

----
#### ["Natural Language Does Not Emerge ‘Naturally’ in Multi-Agent Dialog"](https://arxiv.org/abs/1706.08502) Kottur, Moura, Lee, Batra
  `language emergence in dialog`

#### ["Emergent Language in a Multi-Modal, Multi-Step Referential Game"](https://arxiv.org/abs/1705.10369) Evtimova, Drozdov, Kiela, Cho
  `language emergence in dialog`

#### ["Translating Neuralese"](https://arxiv.org/abs/1704.06960) Andreas, Dragan, Klein
  `language emergence in dialog`
>	"Authors take the vector messages (“neuralese”) passed between two machines trained to perform a collaborative task, and translate them into natural language utterances. To overcome the absence of neuralese-to-English parallel data, authors consider a pair of messages equivalent if they are used in similar scenarios by human and machine agents."  
  - `audio` <https://soundcloud.com/nlp-highlights/34-translating-neuralese-with-jacob-andreas> (Andreas)
  - `code` <http://github.com/jacobandreas/neuralese>

#### ["Deal or No Deal? End-to-End Learning for Negotiation Dialogues"](https://arxiv.org/abs/1706.05125) Lewis, Yarats, Dauphin, Parikh, Batra
  `language emergence in dialog`
  - `post` <https://code.facebook.com/posts/1686672014972296>
  - `video` <https://ku.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=e76f464c-6f81-4e31-b942-839312cf0f8c> (Lewis)
  - `code` <https://github.com/facebookresearch/end-to-end-negotiator>

#### ["Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning"](https://arxiv.org/abs/1703.06585) Das, Kottur, Moura, Lee, Batra
  `language emergence in dialog`
  - `video` <https://youtube.com/watch?v=SztC8VOWwRQ> (demo)
  - `video` <https://youtube.com/watch?v=I9OlorMh7wU> (Das)
  - `video` <http://videolectures.net/deeplearning2017_parikh_batra_deep_rl/> (part 2, 25:47) (Batra)

#### ["Emergence of Grounded Compositional Language in Multi-Agent Populations"](http://arxiv.org/abs/1703.04908) Mordatch, Abbeel
  `language emergence in dialog`
>	"Though the agents come up with words that we found to correspond to objects and other agents, as well as actions like 'Look at' or 'Go to', to the agents these words are abstract symbols represented by one-hot vector - we label these one-hot vectors with English words that capture their meaning for the sake of interpretability."  
>
>	"One possible scenario is from goal oriented-dialog systems. Where one agent tries to transmit to another certain API call that it should perform (book restaurant, hotel, whatever). I think these models can make it more data efficient. At the first stage two agents have to communicate and discover their own language, then you can add regularization to make the language look more like natural language and on the final stage, you are adding a small amount of real data (dialog examples specific for your task). I bet that using additional communication loss will make the model more data efficient."  
>
>	"The big outcome to hunt for in this space is a post-gradient descent learning algorithm. Of course you can make agents that play the symbol grounding game, but it's not a very big step from there to compression of data, and from there to compression of 'what you need to know to solve the problem you're about to encounter' - at which point you have a system which can learn by training or learn by receiving messages. It was pretty easy to get stuff like one agent learning a classifier, encoding it in a message, and transmitting it to a second agent who has to use it for zero-shot classification. But it's still single-task specific communication, so there's no benefit to the agent for receiving, say, the messages associated with the previous 100 tasks. The tricky thing is going from there to something more abstract and cumulative, so that you can actually use message generation as an iterative learning mechanism. I think a large part of that difficulty is actually designing the task ensemble, not just the network architecture."  
  - `video` <https://youtube.com/watch?v=liVFy7ZO4OA> (demo)
  - `post` <https://blog.openai.com/learning-to-communicate/>
  - `video` <https://youtu.be/02xIkHowQOk?t=1h17m45s> (Lowe)
  - `video` <https://youtube.com/watch?v=f4gKhK8Q6mY&t=22m20s> (Abbeel)
  - `paper` ["A Paradigm for Situated and Goal-Driven Language Learning"](https://arxiv.org/abs/1610.03585)  

#### ["Multi-Agent Cooperation and the Emergence of (Natural) Language"](https://arxiv.org/abs/1612.07182) Lazaridou, Peysakhovich, Baroni
  `language emergence in dialog`
  - `video` <https://facebook.com/iclr.cc/videos/1712966538732405/> (Peysakhovich)

#### ["Learning Language Games through Interaction"](http://arxiv.org/abs/1606.02447) Wang, Liang, Manning
  `language emergence in dialog`
  - `post` <http://nlp.stanford.edu/blog/interactive-language-learning/>
  - `video` <http://youtube.com/watch?v=PfW4_3tCiw0> (demo, calendar)
  - <http://shrdlurn.sidaw.xyz> (demo, blocks world)
  - `video` <https://youtube.com/watch?v=iuazFltYgCE> (Wang)
  - `video` <https://youtu.be/mhHfnhh-pB4?t=1h5m45s> (Liang)
  - `video` <https://youtu.be/6O5sttckalE?t=40m45s> (Liang)



---
### natural language processing

[interesting older papers](https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#interesting-papers)

----
#### ["Non-Autoregressive Neural Machine Translation"](https://arxiv.org/abs/1711.02281) Gu, Bradbury, Xiong, Li, Socher
  `translation`
  - `post` <https://einstein.ai/research/non-autoregressive-neural-machine-translation>

#### ["Unsupervised Machine Translation Using Monolingual Corpora Only"](https://arxiv.org/abs/1711.00043) Lample, Denoyer, Ranzato
  `translation`
>	"learn to map sentences of the two languages into the same feature space by denoising both via auto-encoding and via cross-domain encoding"  
  - `notes` <http://ankitg.me/blog/2017/11/05/unsupervised-machine-translation.html>

#### ["Unsupervised Neural Machine Translation"](https://arxiv.org/abs/1710.11041) Artetxe, Labaka, Agirre, Cho
  `translation`
  - `notes` <http://ankitg.me/blog/2017/11/05/unsupervised-machine-translation.html>

#### ["Word Translation Without Parallel Data"](https://arxiv.org/abs/1710.04087) Conneau, Lample, Ranzato, Denoyer, Jegou
  `translation`
>	"Our method leverages adversarial training to learn a linear mapping from a source to a target space and operates in two steps. First, in a two-player game, a discriminator is trained to distinguish between the mapped source embeddings and the target embeddings, while the mapping (which can be seen as a generator) is jointly trained to fool the discriminator. Second, we extract a synthetic dictionary from the resulting shared embedding space and fine-tune the mapping with the closed-form Procrustes solution."  
>	"(A) There are two distributions of word embeddings, English words in red denoted by X and Italian words in blue denoted by Y, which we want to align/translate. Each dot represents a word in that space. The size of the dot is proportional to the frequency of the words in the training corpus of that language.  
>	(B) Using adversarial learning, we learn a rotation matrix W which roughly aligns the two distributions. The green stars are randomly selected words that are fed to the discriminator to determine whether the two word embeddings come from the same distribution.  
>	(C) The mapping W is further refined via Procrustes. This method uses frequent words aligned by the previous step as anchor points, and minimizes an energy function that corresponds to a spring system between anchor points. The refined mapping is then used to map all words in the dictionary.  
>	(D) Finally, we translate by using the mapping W and a distance metric, dubbed CSLS, that expands the space where there is high density of points (like the area around the word “cat”), so that “hubs” (like the word “cat”) become less close to other word vectors than they would otherwise (compare to the same region in panel (A))."  

#### ["Style Transfer from Non-Parallel Text by Cross-Alignment"](https://arxiv.org/abs/1705.09655) Shen, Lei, Barzilay, Jaakkola
  `translation`
  - `video` <https://facebook.com/NIPSlive/videos/1491773310878510/> (1:20:43) (Shen)

----
#### ["Bag of Tricks for Efficient Text Classification"](http://arxiv.org/abs/1607.01759) Joulin, Grave, Bojanowski, Mikolov
  `classification` `fastText`
>	"At par with deep learning models in terms of accuracy though an order of magnitude faster in performance."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1607.01759>
  - `notes` <https://medium.com/paper-club/bag-of-tricks-for-efficient-text-classification-818bc47e90f>
  - `code` <https://fasttext.cc>
  - `code` <https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py>

----
#### ["Discovering Discrete Latent Topics with Neural Variational Inference"](https://arxiv.org/abs/1706.00359) Miao, Grefenstette, Blunsom
  `topic modeling`
>	"Traditional inference methods have sought closed-form derivations for updating the models, however as the expressiveness of these models grows, so does the difficulty of performing fast and accurate inference over their parameters. This paper presents alternative neural approaches to topic modelling by providing parameterisable distributions over topics which permit training by backpropagation in the framework of neural variational inference. In addition, with the help of a stick-breaking construction, we propose a recurrent network that is able to discover a notionally unbounded number of topics, analogous to Bayesian non-parametric topic models."  
  - `video` <https://vimeo.com/238222598> (Miao)

----
#### ["Learning a Neural Semantic Parser from User Feedback"](https://arxiv.org/abs/1704.08760) Iyer, Konstas, Cheung, Krishnamurthy, Zettlemoyer
  `semantic parsing`
>	"We learn a semantic parser for an academic domain from scratch by deploying an online system using our interactive learning algorithm. After three train-deploy cycles, the system correctly answered 63.51% of user’s questions. To our knowledge, this is the first effort to learn a semantic parser using a live system, and is enabled by our models that can directly parse language to SQL without manual intervention."  
#### ["Semantic Parsing with Semi-Supervised Sequential Autoencoders"](http://arxiv.org/abs/1609.09315) Kocisky, Melis, Grefenstette, Dyer, Ling, Blunsom, Hermann
  `semantic parsing`

#### ["Open-Vocabulary Semantic Parsing with both Distributional Statistics and Formal Knowledge"](http://arxiv.org/abs/1607.03542) Gardner, Krishnamurthy
  `semantic parsing`

#### ["Language to Logical Form with Neural Attention"](http://arxiv.org/abs/1601.01280) Dong, Lapata
  `semantic parsing`

----
#### ["Globally Normalized Transition-Based Neural Networks"](http://arxiv.org/abs/1603.06042) Andor, Alberti, Weiss, Severyn, Presta, Ganchev, Petrov, Collins
  `dependency parsing` `SyntaxNet` `Parsey McParseface`
>	"The parser uses a feed forward NN, which is much faster than the RNN usually used for parsing. Also the paper is using a global method to solve the label bias problem. This method can be used for many tasks and indeed in the paper it is used also to shorten sentences by throwing unnecessary words. The label bias problem arises when predicting each label in a sequence using a softmax over all possible label values in each step. This is a local approach but what we are really interested in is a global approach in which the sequence of all labels that appeared in a training example are normalized by all possible sequences. This is intractable so instead a beam search is performed to generate alternative sequences to the training sequence. The search is stopped when the training sequence drops from the beam or ends. The different beams with the training sequence are then used to compute the global loss."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1603.06042>
  - `code` <https://github.com/tensorflow/models/tree/master/research/syntaxnet>

----
#### ["A Simple but Tough-to-Beat Baseline for Sentence Embeddings"](https://openreview.net/pdf?id=SyK00v5xx) Arora, Liang, Ma
  `text embedding`
>	"The success of neural network methods for computing word embeddings has motivated methods for generating semantic embeddings of longer pieces of text, such as sentences and paragraphs. Surprisingly, Wieting et al (ICLR’16) showed that such complicated methods are outperformed, especially in out-of-domain (transfer learning) settings, by simpler methods involving mild retraining of word embeddings and basic linear regression. The method of Wieting et al. requires retraining with a substantial labeled dataset such as Paraphrase Database (Ganitkevitch et al., 2013). The current paper goes further, showing that the following completely unsupervised sentence embedding is a formidable baseline: Use word embeddings computed using one of the popular methods on unlabeled corpus like Wikipedia, represent the sentence by a weighted average of the word vectors, and then modify them a bit using PCA/SVD. This weighting improves performance by about 10% to 30% in textual similarity tasks, and beats sophisticated supervised methods including RNN’s and LSTM’s."  
>	"The paper also gives a theoretical explanation of the success of the above unsupervised method using a latent variable generative model for sentences, which is a simple extension of the model in Arora et al. (TACL’16) with new “smoothing” terms that allow for words occurring out of context, as well as high probabilities for words like and, not in all contexts."  
  - `video` <https://youtube.com/watch?v=BCsOrewkmH4> (Liang)
  - `video` <https://youtu.be/KR46z_V0BVw?t=49m10s> (Arora)
  - <https://akshayka.github.io/papers/html/arora2017sentence-embeddings.html>
  - <https://github.com/PrincetonML/SIF>
  - <https://github.com/YingyuLiang/SIF>

#### ["On the Use of Word Embeddings Alone to Represent Natural Language Sequences"](https://openreview.net/forum?id=Sy5OAyZC-)
>	"To construct representations for natural language sequences, information from two main sources needs to be captured: (i) semantic meaning of individual words, and (ii) their compositionality. These two types of information are usually represented in the form of word embeddings and compositional functions, respectively. For the latter, Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs) have been considered. There has not been a rigorous evaluation regarding the relative importance of each component to different text-representation-based tasks; i.e., how important is the modeling capacity of word embeddings alone, relative to the added value of a compositional function? We conduct an extensive comparative study between Simple Word Embeddings-based Models (SWEMs), with no compositional parameters, relative to employing word embeddings within RNN/CNN-based models. Surprisingly, SWEMs exhibit comparable or even superior performance in the majority of cases considered."  

#### ["Learning to Compute Word Embeddings On the Fly"](https://arxiv.org/abs/1706.00286) Bahdanau, Bosc, Jastrzebski, Grefenstette, Vincent, Bengio
  `word embedding`
  - `notes` <https://theneuralperspective.com/2017/06/05/more-on-embeddings-spring-2017/>

----
#### ["Unbounded Cache Model for Online Language Modeling with Open Vocabulary"](https://arxiv.org/abs/1711.02604) Grave, Cisse, Joulin
  `language modeling`
>	"We propose an extension of continuous cache models, which can scale to larger contexts. We use a large scale non-parametric memory component that stores all the hidden activations seen in the past. We leverage recent advances in approximate nearest neighbor search and quantization algorithms to store millions of representations while searching them efficiently."  

#### ["Improving Neural Language Models with a Continuous Cache"](http://arxiv.org/abs/1612.04426) Grave, Joulin, Usunier
  `language modeling` `adaptive softmax`

#### ["Pointer Sentinel Mixture Models"](http://arxiv.org/abs/1609.07843) Merity, Xiong, Bradbury, Socher
  `language modeling`
>	"The authors combine a standard LSTM softmax with Pointer Networks in a mixture model called Pointer-Sentinel LSTM. The pointer networks helps with rare words and long-term dependencies but is unable to refer to words that are not in the input. The opposite is the case for the standard softmax."  
  - `video` <https://youtube.com/watch?v=Ibt8ZpbX3D8> (Merity)
  - `video` <https://youtu.be/Q7ifcUuMZvk?t=30m11s> (Socher)
  - `notes` <https://theneuralperspective.com/2016/10/04/pointer-sentinel-mixture-models/>
