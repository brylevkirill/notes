interesting recent papers:

  * [**deep learning theory**](#deep-learning-theory)
  * [**bayesian deep learning**](#bayesian-deep-learning)
  * [**compute and memory architectures**](#compute-and-memory-architectures)
  * [**meta-learning**](#meta-learning)
  * [**few-shot learning**](#few-shot-learning)
  * [**unsupervised learning**](#unsupervised-learning)
  * [**generative models**](#generative-models)
    - [**flow models**](#generative-models---flow-models)
    - [**generative adversarial networks**](#generative-models---generative-adversarial-networks)
    - [**variational autoencoders**](#generative-models---variational-autoencoders)
    - [**autoregressive models**](#generative-models---autoregressive-models)
  * [**reinforcement learning**](#reinforcement-learning---model-free-methods)
    - [**model-free methods**](#reinforcement-learning---model-free-methods)
    - [**model-based methods**](#reinforcement-learning---model-based-methods)
    - [**exploration and intrinsic motivation**](#reinforcement-learning---exploration-and-intrinsic-motivation)
    - [**hierarchical**](#reinforcement-learning---hierarchical)
    - [**transfer**](#reinforcement-learning---transfer)
    - [**imitation**](#reinforcement-learning---imitation)
    - [**multi-agent**](#reinforcement-learning---multi-agent)
  * [**program synthesis**](#program-synthesis)
  * [**reasoning**](#reasoning)
  * [**language grounding**](#language-grounding)
  * [**natural language processing**](#natural-language-processing)

----
interesting older papers:

  - [**artificial intelligence**](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#interesting-papers)
  - [**knowledge representation and reasoning**](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers)
  - [**machine learning**](https://github.com/brylevkirill/notes/blob/master/Machine%20Learning.md#interesting-papers)
  - [**deep learning**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers)
  - [**reinforcement learning**](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers)
  - [**bayesian inference and learning**](https://github.com/brylevkirill/notes/blob/master/Bayesian%20Inference%20and%20Learning.md#interesting-papers)
  - [**probabilistic programming**](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md#interesting-papers)
  - [**natural language processing**](https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#interesting-papers)
  - [**information retrieval**](https://github.com/brylevkirill/notes/blob/master/Information%20Retrieval.md#interesting-papers)



---
### deep learning theory

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---theory)

----
#### ["Approximating CNNs with Bag-of-local-Features Models Works Surprisingly Well on ImageNet"](https://arxiv.org/abs/1904.00760) Brendel, Bethge
  `generalization` `BagNet` `ICLR 2019`
>	"We introduce a high-performance DNN architecture on ImageNet whose decisions are considerably easier to explain. Our model, a simple variant of the ResNet-50 architecture called BagNet, classifies an image based on the occurrences of small local image features without taking into account their spatial ordering. This strategy is closely related to the bag-of-feature (BoF) models popular before the onset of deep learning and reaches a surprisingly high accuracy on ImageNet (87.6% top-5 for 32 x 32 px features and Alexnet performance for 16 x16 px features). The constraint on local features makes it straight-forward to analyse how exactly each part of the image influences the classification. Furthermore, the BagNets behave similar to state-of-the art deep neural networks such as VGG-16, ResNet-152 or DenseNet-169 in terms of feature sensitivity, error distribution and interactions between image parts. This suggests that the improvements of DNNs over previous bag-of-feature classifiers in the last few years is mostly achieved by better fine-tuning rather than by qualitatively different decision strategies."  
  - `post` <https://medium.com/bethgelab/neural-networks-seem-to-follow-a-puzzlingly-simple-strategy-to-classify-images-f4229317261f>
  - `post` <https://blog.evjang.com/2019/02/bagnet.html>

#### ["Excessive Invariance Causes Adversarial Vulnerability"](https://arxiv.org/abs/1811.00401) Jacobsen, Behrmann, Zemel, Bethge
  `generalization` `ICLR 2019`
>	"Deep neural networks exhibit striking failures on out-of-distribution inputs. One core idea of adversarial example research is to reveal neural network errors under such distribution shifts. We decompose these errors into two complementary sources: sensitivity and invariance. We show deep networks are not only too sensitive to task-irrelevant changes of their input, as is well-known from epsilon-adversarial examples, but are also too invariant to a wide range of task-relevant changes, thus making vast regions in input space vulnerable to adversarial attacks. We show such excessive invariance occurs across various tasks and architecture types. On MNIST and ImageNet one can manipulate the class-specific content of almost any image without changing the hidden activations. We identify an insufficiency of the standard cross-entropy loss as a reason for these failures. Further, we extend this objective based on an information-theoretic analysis so it encourages the model to consider all task-dependent features in its decision. This provides the first approach tailored explicitly to overcome excessive invariance and resulting vulnerabilities."  
>	"Failures of deep networks under distribution shift and their difficulty in out-of-distribution generalization are prime examples of the limitations in current machine learning models. The field of adversarial example research aims to close this gap from a robustness point of view. While a lot of work has studied epsilon-adversarial examples, recent trends extend the efforts towards the unrestricted case. However, adversarial examples with no restriction are hard to formalize beyond testing error. We introduce a reverse view on the problem to: (1) show that a major cause for adversarial vulnerability is excessive invariance to semantically meaningful variations, (2) demonstrate that this issue persists across tasks and architectures; and (3) make the control of invariance tractable via fully-invertible networks."  
>	"We propose an invertible network architecture that gives explicit access to its decision space, enabling class-specific manipulations to images while leaving all dimensions of the representation seen by the final classifier invariant."  
>	"We demonstrated how a bijective network architecture enables us to identify large adversarial subspaces on multiple datasets like the adversarial spheres, MNIST and ImageNet. Afterwards, we formalized the distribution shifts causing such undesirable behavior via information theory. Using this framework, we find one of the major reasons is the insufficiency of the vanilla cross-entropy loss to learn semantic representations that capture all task-dependent variations in the input. We extend the loss function by components that explicitly encourage a split between semantically meaningful and nuisance features. Finally, we empirically show that this split can remove unwanted invariances by performing a set of targeted invariance-based distribution shift experiments."  
>	"All images shown cause a competitive ImageNet-trained network to output the exact same probabilities over all 1000 classes (logits shown above each image). The leftmost image is from the ImageNet validation set; all other images are constructed such that they match the non-class related information of images taken from other classes. The excessive invariance revealed by this set of adversarial examples demonstrates that the logits contain only a small fraction of the information perceptually relevant to humans for discrimination between the classes."  
>	"The invariance perspective suggests that adversarial vulnerability is a consequence of narrow learning, yielding classifiers that rely only on few highly predictive features in their decisions. This has
also been supported by the observation that deep networks strongly rely on spectral statistical regularities, or stationary statistics to make their decisions, rather than more abstract features like shape and appearance. We hypothesize that a major reason for this excessive invariance can be understood from an information-theoretic viewpoint of cross-entropy, which maximizes a bound on the mutual information between labels and representation, giving no incentive to explain all class-dependent aspects of the input. This may be desirable in some cases, but to achieve truly general understanding of a scene or an object, machine learning models have to learn to successfully separate essence from nuisance and subsequently generalize even under shifted input distributions."  
  - `post` <https://medium.com/@j.jacobsen/deep-classifiers-ignore-almost-everything-they-see-and-how-we-may-be-able-to-fix-it-a6888012516f>

#### ["Measuring the Tendency of CNNs to Learn Surface Statistical Regularities"](https://arxiv.org/abs/1711.11561) Jo, Bengio
  `generalization`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#measuring-the-tendency-of-cnns-to-learn-surface-statistical-regularities-jo-bengio>

#### ["The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"](https://arxiv.org/abs/1803.03635) Frankle, Carbin
  `generalization` `ICLR 2019`
>	"Neural network pruning techniques can reduce the parameter counts of trained networks by over 90%, decreasing storage requirements and improving computational performance of inference without compromising accuracy. However, contemporary experience is that the sparse architectures produced by pruning are difficult to train from the start, which would similarly improve training performance.
We find that a standard pruning technique naturally uncovers subnetworks whose initializations made them capable of training effectively. Based on these results, we articulate the "lottery ticket hypothesis:" dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations. The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective."  
>	"We present an algorithm to identify winning tickets and a series of experiments that support the lottery ticket hypothesis and the importance of these fortuitous initializations. We consistently find winning tickets that are less than 10-20% of the size of several fully-connected and convolutional feed-forward architectures for MNIST and CIFAR10. Above this size, the winning tickets that we find learn faster than the original network and reach higher test accuracy."  
  - `video` <https://youtube.com/watch?v=s7DqRZVvRiQ> (Frankle)

#### ["Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask"](https://arxiv.org/abs/1905.01067) Zhou, Lan, Liu, Yosinski
  `generalization`
>	"The recent “Lottery Ticket Hypothesis” paper by Frankle & Carbin showed that a simple approach to creating sparse networks (keep the large weights) results in models that are trainable from scratch, but only when starting from the same initial weights.  The performance of these networks often exceeds the performance of the non-sparse base model, but for reasons that were not well understood. In this paper we study the three critical components of the Lottery Ticket algorithm, showing that each may be varied significantly without impacting the overall results. Ablating these factors leads to new insights for why LT networks perform as well as they do. We show why setting weights to zero is important, how signs are all you need to make the re-initialized network train, and why masking behaves like training. Finally, we discover the existence of Supermasks, or masks that can be applied to an untrained, randomly initialized network to produce a model with performance far better than chance (86% on MNIST, 41% on CIFAR-10)."
  - `post` <https://www.shortscience.org/paper?bibtexKey=zhou2019deconstructing> (notes)

#### ["Deep Image Prior"](https://arxiv.org/abs/1711.10925) Ulyanov, Vedaldi, Lempitsky
  `generalization`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#deep-image-prior-ulyanov-vedaldi-lempitsky>

#### ["Understanding Deep Learning Requires Rethinking Generalization"](http://arxiv.org/abs/1611.03530) Zhang, Bengio, Hardt, Recht, Vinyals
  `generalization`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#understanding-deep-learning-requires-rethinking-generalization-zhang-bengio-hardt-recht-vinyals>

#### ["Sharp Minima Can Generalize For Deep Nets"](https://arxiv.org/abs/1703.04933) Dinh, Pascanu, Bengio, Bengio
  `generalization`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#sharp-minima-can-generalize-for-deep-nets-dinh-pascanu-bengio-bengio>

#### ["A Closer Look at Memorization in Deep Networks"](https://arxiv.org/abs/1706.05394) Arpit et al.
  `generalization`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#a-closer-look-at-memorization-in-deep-networks-arpit-et-al>

#### ["mixup: Beyond Empirical Risk Minimization"](https://arxiv.org/abs/1710.09412) Zhang, Cisse, Dauphin, Lopez-Paz
  `generalization`
>	"mixup trains a neural network on convex combinations of pairs of examples and their labels. By doing so, mixup regularizes the neural network to favor simple linear behavior in-between training examples."  
>	"Empirical Risk Minimization allows large neural networks to memorize (instead of generalize from) the training data even in the presence of strong regularization, or in classification problems where the labels are assigned at random. On the other hand, neural networks trained with ERM change their predictions drastically when evaluated on examples just outside the training distribution, also known as adversarial examples. This evidence suggests that ERM is unable to explain or provide generalization on testing distributions that differ only slightly from the training data."  
>	"In Vicinal Risk Minimization, human knowledge is required to describe a vicinity or neighborhood around each example in the training data. Then, additional virtual examples can be drawn from the vicinity distribution of the training examples to enlarge the support of the training distribution. For instance, when performing image classification, it is common to define the vicinity of one image as the set of its horizontal reflections, slight rotations, and mild scalings. mixup extends the training distribution by incorporating the prior knowledge that linear interpolations of feature vectors should lead to linear interpolations of the associated targets."  
  - `post` <http://inference.vc/mixup-data-dependent-data-augmentation/>
  - `code` <https://github.com/leehomyc/mixup_pytorch>
  - `paper` ["MixMatch: A Holistic Approach to Semi-Supervised Learning"](https://arxiv.org/abs/1905.02249) by Berthelot et al.

#### ["Opening the Black Box of Deep Neural Networks via Information"](http://arxiv.org/abs/1703.00810) Shwartz-Ziv, Tishby
  `generalization` `information bottleneck`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#opening-the-black-box-of-deep-neural-networks-via-information-shwartz-ziv-tishby>

#### ["Deep Variational Information Bottleneck"](https://arxiv.org/abs/1612.00410) Alemi, Fischer, Dillon, Murphy
  `generalization` `information bottleneck`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#deep-variational-information-bottleneck-alemi-fischer-dillon-murphy>

#### ["On the Emergence of Invariance and Disentangling in Deep Representations"](https://arxiv.org/abs/1706.01350) Achille, Soatto
  `generalization` `information bottleneck`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#on-the-emergence-of-invariance-and-disentangling-in-deep-representations-achille-soatto>

----
#### ["A Bayesian Perspective on Generalization and Stochastic Gradient Descent"](https://arxiv.org/abs/1710.06451) Smith, Le
  `optimization` `generalization`
>	"How can we predict if a minimum will generalize to the test set, and why does stochastic gradient descent find minima that generalize well? Our work is inspired by Zhang et al. (2017), who showed deep networks can easily memorize randomly labeled training data, despite generalizing well when shown real labels of the same inputs. We show here that the same phenomenon occurs in small linear models. These observations are explained by evaluating the Bayesian evidence, which penalizes sharp minima but is invariant to model parameterization. We also explore the "generalization gap" between small and large batch training, identifying an optimum batch size which maximizes the test set accuracy."  
>	"The optimum batch size is proportional to the learning rate and the training set size. We verify these predictions empirically."  
  - `paper` ["Don’t Decay the Learning Rate, Increate the Batch Size"](https://arxiv.org/abs/1711.00489) by Smith, Kindermans, Le

#### ["On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"](https://arxiv.org/abs/1609.04836) Keskar, Mudigere, Nocedal, Smelyanskiy, Tang
  `optimization` `generalization`
>	"Deep networks generalise better with smaller batch-size when no other form of regularisation is used. And it may be because SGD biases learning towards flat local minima, rather than sharp local minima."  
>	"Using large batch sizes tends to find sharped minima and generalize worse. This means that we can’t talk about generalization without taking training algorithm into account."  
  - `video` <https://youtu.be/cHjI37DsQCQ?t=29m39s> (Selvaraj)
  - `video` <http://videolectures.net/deeplearning2017_larochelle_neural_networks/> (part 2, 1:25:55) (Larochelle)
  - `slides` <https://iclr.cc/archive/www/lib/exe/fetch.php%3Fmedia=iclr2017:nocedal_iclr2017.pdf>
  - `post` <http://inference.vc/everything-that-works-works-because-its-bayesian-2/>
  - `code` <https://github.com/keskarnitish/large-batch-training>
  - `paper` ["Flat Minima"](http://www.bioinf.jku.at/publications/older/3304.pdf) by Hochreiter, Schmidhuber ([overview](https://youtu.be/NZEAqdepq0w?t=28m40s) by Sepp Hochreiter `video`)

----
#### ["How Does Batch Normalization Help Optimization? (No, It Is Not About Internal Covariate Shift)"](https://arxiv.org/abs/1805.11604) Santurkar, Tsipras, Ilyas, Madry
  `optimization`
>	"Batch Normalization (BatchNorm) is a widely adopted technique that enables faster and more stable training of deep neural networks (DNNs). Despite its pervasiveness, the exact reasons for BatchNorm's effectiveness are still poorly understood. The popular belief is that this effectiveness stems from controlling the change of the layers' input distributions during training to reduce the so-called "internal covariate shift". In this work, we demonstrate that such distributional stability of layer inputs has little to do with the success of BatchNorm. Instead, we uncover a more fundamental impact of BatchNorm on the training process: it makes the optimization landscape significantly smoother. This smoothness induces a more predictive and stable behavior of the gradients, allowing for faster training."  
  - `notes` <https://twitter.com/arimorcos/status/1001856542268952576>

#### ["Self-Normalizing Neural Networks"](https://arxiv.org/abs/1706.02515) Klambauer, Unterthiner, Mayr, Hochreiter
  `optimization` `SELU`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#self-normalizing-neural-networks-klambauer-unterthiner-mayr-hochreiter>

#### ["The Shattered Gradients Problem: If resnets are the answer, then what is the question?"](https://arxiv.org/abs/1702.08591) Balduzzi, Frean, Leary, Lewis, Ma, McWilliams
  `optimization`
>	"We show that the correlation between gradients in standard feedforward networks decays exponentially with depth resulting in gradients that resemble white noise."  
>	"We present a new “looks linear” (LL) initialization that prevents shattering. Preliminary experiments show the new initialization allows to train very deep networks without the addition of skip-connections."  
>	"In a randomly initialized network, the gradients of deeper layers are increasingly uncorrelated. Shattered gradients play havoc with the optimization methods currently in use and may explain the difficulty in training deep feedforward networks even when effective initialization and batch normalization are employed. Averaging gradients over minibatches becomes analogous to integrating over white noise – there is no clear trend that can be summarized in a single average direction. Shattered gradients can also introduce numerical instabilities, since small differences in the input can lead to large differences in gradients."  
>	"Skip-connections in combination with suitable rescaling reduce shattering. Specifically, we show that the rate at which correlations between gradients decays changes from exponential for feedforward architectures to sublinear for resnets. The analysis uncovers a surprising and unexpected side-effect of batch normalization."  
  - `video` <https://vimeo.com/237275640> (McWilliams)

#### ["First-order Methods Almost Always Avoid Saddle Points"](https://arxiv.org/abs/1710.07406) Lee, Panageas, Piliouras, Simchowitz, Jordan, Recht
  `optimization`
>	"We establish that first-order methods avoid saddle points for almost all initializations. Our results apply to a wide variety of first-order methods, including gradient descent, block coordinate descent, mirror descent and variants thereof. The connecting thread is that such algorithms can be studied from a dynamical systems perspective in which appropriate instantiations of the Stable Manifold Theorem allow for a global stability analysis. Thus, neither access to second-order derivative information nor randomness beyond initialization is necessary to provably avoid saddle points."  

#### ["Gradient Descent Converges to Minimizers"](https://arxiv.org/abs/1602.04915) Lee, Simchowitz, Jordan, Recht
  `optimization`
>	"We show that gradient descent converges to a local minimizer, almost surely with random initialization. This is proved by applying the Stable Manifold Theorem from dynamical systems theory."  

#### ["The Marginal Value of Adaptive Gradient Methods in Machine Learning"](https://arxiv.org/abs/1705.08292) Wilson, Roelofs, Stern, Srebro, Recht
  `optimization`
>	"Authors argued that adaptive optimization methods tend to generalize less well than SGD and SGD with momentum (although they did not include K-FAC in their study)."  
>	"Despite the fact that our experimental evidence demonstrates that adaptive methods are not advantageous for machine learning, the Adam algorithm remains incredibly popular. Adaptive gradient methods are particularly popular for training GANs and Q-learning with function approximation. Both of these applications stand out because they are not solving optimization problems. It is possible that the dynamics of Adam are accidentally well matched to these sorts of optimization-free iterative search procedures. It is also possible that carefully tuned stochastic gradient methods may work as well or better in both of these applications."  
  - `video` <https://facebook.com/nipsfoundation/videos/1554657104625523?t=133> (Roelofs)

----
#### ["Measuring the Intrinsic Dimension of Objective Landscapes"](https://arxiv.org/abs/1804.08838) Li, Farkhoor, Liu, Yosinski
  `interpretability`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#measuring-the-intrinsic-dimension-of-objective-landscapes-li-farkhoor-liu-yosinski>

#### ["Interpretation of Neural Network is Fragile"](https://arxiv.org/abs/1710.10547) Ghorbani, Abid, Zou
  `interpretability`
>	"In this paper, we show that interpretation of deep learning predictions is extremely fragile in the following sense: two perceptively indistinguishable inputs with the same predicted label can be assigned very different interpretations. We systematically characterize the fragility of several widely-used feature-importance interpretation methods (saliency maps, relevance propagation, and DeepLIFT) on ImageNet and CIFAR-10. Our experiments show that even small random perturbation can change the feature importance and new systematic perturbations can lead to dramatically different interpretations without changing the label. We extend these results to show that interpretations based on exemplars (e.g. influence functions) are similarly fragile. Our analysis of the geometry of the Hessian matrix gives insight on why fragility could be a fundamental challenge to the current interpretation approaches."  

#### ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599) Guo, Pleiss, Sun, Weinberger
  `interpretability`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#on-calibration-of-modern-neural-networks-guo-pleiss-sun-weinberger>

#### ["Understanding Black-box Predictions via Influence Functions"](https://arxiv.org/abs/1703.04730) Koh, Liang
  `interpretability`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#understanding-black-box-predictions-via-influence-functions-koh-liang>



---
### bayesian deep learning

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---bayesian-deep-learning)  
[**interesting older papers - bayesian inference and learning**](https://github.com/brylevkirill/notes/blob/master/Bayesian%20Inference%20and%20Learning.md#interesting-papers)  

[**interesting recent papers - variational autoencoders**](#generative-models---variational-autoencoders)  
[**interesting recent papers - unsupervised learning**](#unsupervised-learning)  
[**interesting recent papers - model-based reinforcement learning**](#reinforcement-learning---model-based-methods)  
[**interesting recent papers - exploration and intrinsic motivation**](#reinforcement-learning---exploration-and-intrinsic-motivation)  

----
#### ["A Bayesian Perspective on Generalization and Stochastic Gradient Descent"](https://arxiv.org/abs/1710.06451) Smith, Le
  `stochastic gradient descent` `generalization`
>	"How can we predict if a minimum will generalize to the test set, and why does stochastic gradient descent find minima that generalize well? Our work is inspired by Zhang et al. (2017), who showed deep networks can easily memorize randomly labeled training data, despite generalizing well when shown real labels of the same inputs. We show here that the same phenomenon occurs in small linear models. These observations are explained by evaluating the Bayesian evidence, which penalizes sharp minima but is invariant to model parameterization. We also explore the "generalization gap" between small and large batch training, identifying an optimum batch size which maximizes the test set accuracy."  
>	"The optimum batch size is proportional to the learning rate and the training set size. We verify these predictions empirically."  

----
#### ["Noisy Natural Gradient as Variational Inference"](https://arxiv.org/abs/1712.02390) Zhang, Sun, Duvenaud, Grosse
  `natural gradient descent` `approximate inference`
>	"Stochastic variational inference typically imposes restrictive factorization assumptions on the approximate posterior, such as fully independent weights. There have been attempts to fit more expressive approximating distributions which capture correlations such as matrix-variate Gaussians or multiplicative normalizing flows, but fitting such models can be expensive without further approximations."  
>	"We show that natural gradient ascent with adaptive weight noise can be interpreted as fitting a variational posterior to maximize the evidence lower bound. This insight allows us to train full covariance, fully factorized, and matrix variate Gaussian variational posteriors using noisy versions of natural gradient, Adam, and K-FAC, respectively."  
>	"On standard regression benchmarks, our noisy K-FAC algorithm makes better predictions and matches HMC’s predictive variances better than existing methods. Its improved uncertainty estimates lead to more efficient exploration in the settings of active learning and intrinsic motivation for reinforcement learning."  
>	"We introduce and exploit a surprising connection between natural gradient descent and variational inference. In particular, several approximate natural gradient optimizers have been proposed which fit tractable approximations to the Fisher matrix to gradients sampled during training such as Adam and K-FAC. While these procedures were described as natural gradient descent on the weights using an approximate Fisher matrix, we reinterpret these algorithms as natural gradient on a variational posterior using the exact Fisher matrix. Both the weight updates and the Fisher matrix estimation can be seen as natural gradient ascent on a unified evidence lower bound, analogously to how Neal and Hinton interpreted the E and M steps of Expectation-Maximization as coordinate ascent on a single objective. Using this insight, we give an alternative training method for variational Bayesian neural networks. For a factorial Gaussian posterior, it corresponds to a diagonal natural gradient method with weight noise, and matches the performance of Bayes By Backprop, but converges faster. We also present noisy K-FAC, an efficient and GPU-friendly method for fitting a full matrix-variate Gaussian posterior, using a variant of Kronecker-Factored Approximate Curvature with correlated weight noise."  
  - `video` <https://youtube.com/watch?v=bWItvHYqKl8> (Grosse)
  - `video` <https://vimeo.com/287804838> (Zhang)
  - `code` <https://github.com/wlwkgus/NoisyNaturalGradient>

#### ["Stochastic Gradient Descent Performs Variational Inference, Converges to Limit Cycles for Deep Networks"](https://arxiv.org/abs/1710.11029) Chaudhari, Soatto
  `stochastic gradient descent` `approximate inference`
>	"We prove that SGD minimizes an average potential over the posterior distribution of weights along with an entropic regularization term. This potential is however not the original loss function in general. So SGD does perform variational inference, but for a different loss than the one used to compute the gradients. Even more surprisingly, SGD does not even converge in the classical sense: we show that the most likely trajectories of SGD for deep networks do not behave like Brownian motion around critical points. Instead, they resemble closed loops with deterministic components. We prove that such “out-of-equilibrium” behavior is a consequence of the fact that the gradient noise in SGD is highly non-isotropic; the covariance matrix of mini-batch gradients has a rank as small as 1% of its dimension."  
>	"It is widely believed that SGD is an implicit regularizer. This belief stems from its remarkable empirical performance. Our results show that such intuition is very well-placed. Thanks to the special architecture of deep networks where gradient noise is highly non-isotropic, SGD helps itself to a potential Φ with properties that lead to both generalization and acceleration."  
  - `video` <https://youtube.com/watch?v=NFeZ6MggJjw> (Chaudhari)

#### ["Stochastic Gradient Descent as Approximate Bayesian Inference"](https://arxiv.org/abs/1704.04289) Mandt, Hoffman, Blei
  `stochastic gradient descent` `approximate inference`
>	"Authors interpreted SGD as a stochastic differential equation, in order to discuss how SGD could be modified to perform approximate Bayesian posterior sampling. However they state that their analysis holds only in the neighborhood of a minimum, while Keskar et al. (2016) showed that the beneficial effects of noise are most pronounced at the start of training."  
  - `video` <https://vimeo.com/249562203#t=2m21s> (Hoffman)
  - `notes` <https://reddit.com/r/MachineLearning/comments/6d7nb1/d_machine_learning_wayr_what_are_you_reading_week/dihh54a/>

#### ["A Variational Analysis of Stochastic Gradient Algorithms"](https://arxiv.org/abs/1602.02666) Mandt, Hoffman, Blei
  `stochastic gradient descent` `approximate inference`
>	"With constant learning rates, SGD is a stochastic process that, after an initial phase of convergence, generates samples from a stationary distribution. We show that SGD with constant rates can be effectively used as an approximate posterior inference algorithm for probabilistic modeling. Specifically, we show how to adjust the tuning parameters of SGD such as to match the resulting stationary distribution to the posterior."  
>	"This analysis rests on interpreting SGD as a continuous-time stochastic process and then minimizing the Kullback-Leibler divergence between its stationary distribution and the target posterior. We model SGD as a multivariate Ornstein-Uhlenbeck process and then use properties of this process to derive the optimal parameters."  
  - `video` <http://techtalks.tv/talks/a-variational-analysis-of-stochastic-gradient-algorithms/62505/> (Mandt)

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
  - `video` <https://facebook.com/nipsfoundation/videos/1553634558061111?t=4372> (Gal)
  - `post` <https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/> (Kendall)

#### ["Decomposition of Uncertainty for Active Learning and Reliable Reinforcement Learning in Stochastic Systems"](https://arxiv.org/abs/1710.07283) Depeweg, Hernandez-Lobato, Doshi-Velez, Udluft
  `uncertainty estimation`
>	"We have studied a decomposition of predictive uncertainty into its epistemic and aleatoric components when working with Bayesian neural networks with latent variables. This decomposition naturally arises when applying information-theoretic active learning setting, and it also enabled us to derive a novel risk objective for reliable reinforcement learning by decomposing risk into a model bias and noise. In both of these settings, our approch allows us to efficiently learn in the face of sophisticated stochastic functions."  
>	"We investigate safe RL using a risk-sensitive criterion which focuses only on risk related to model bias, that is, the risk of the policy performing at test time significantly different from at training time. The proposed criterion quantifies the amount of epistemic uncertainty (model bias risk) in the model’s predictive distribution and ignores any risk stemming from the aleatoric uncertainty."  
>	"We can identify two distinct forms of uncertainties in the class of models given by BNNs with latent variables. Aleatoric uncertainty captures noise inherent in the observations. On the other hand, epistemic uncertainty accounts for uncertainty in the model. In particular, epistemic uncertainty arises from our lack of knowledge of the values of the synaptic weights in the network, whereas aleatoric uncertainty originates from our lack of knowledge of the value of the latent variables. In the domain of model-based RL the epistemic uncertainty is the source of model bias. When there is high discrepancy between model and real-world dynamics, policy behavior may deteriorate. In analogy to the principle that ”a chain is only as strong as its weakest link” a drastic error in estimating the ground truth MDP at a single transition stepcan render the complete policy useless."  
  - `video` <https://vimeo.com/287804426> (Hernandez-Lobato)

#### ["Deep and Confident Prediction for Time Series at Uber"](https://arxiv.org/abs/1709.01907) Zhu, Laptev
  `uncertainty estimation`
  - `post` <https://eng.uber.com/neural-networks-uncertainty-estimation/>

#### ["Dropout Inference in Bayesian Neural Networks with Alpha-divergences"](https://arxiv.org/abs/1703.02914) Li, Gal
  `uncertainty estimation`
>	"We demonstrate improved uncertainty estimates and accuracy compared to VI in dropout networks. We study our model’s epistemic uncertainty far away from the data using adversarial images, showing that these can be distinguished from non-adversarial images by examining our model’s uncertainty."  
  - `video` <https://vimeo.com/238221241> (Li)
  - `code` <https://github.com/YingzhenLi/Dropout_BBalpha>

#### ["Deep Bayesian Active Learning with Image Data"](https://arxiv.org/abs/1703.02910) Gal, Islam, Ghahramani
  `uncertainty estimation`
  - `video` <https://vimeo.com/240606680> (Hernandez-Lobato)
  - `code` <https://github.com/Riashat/Deep-Bayesian-Active-Learning>

#### ["Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"](http://arxiv.org/abs/1612.01474) Lakshminarayanan, Pritzel, Blundell
  `uncertainty estimation`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles-lakshminarayanan-pritzel-blundell>

----
#### ["Model Selection in Bayesian Neural Networks via Horseshoe Priors"](https://arxiv.org/abs/1705.10388) Ghosh, Doshi-Velez
  `model selection`
>	"Model selection - even choosing the number of nodes - remains an open question. In this work, we apply a horseshoe prior over node pre-activations of a Bayesian neural network, which effectively turns off nodes that do not help explain the data. We demonstrate that our prior prevents the BNN from underfitting even when the number of nodes required is grossly over-estimated. Moreover, this model selection over the number of nodes doesn’t come at the expense of predictive or computational performance; in fact, we learn smaller networks with comparable predictive performance to current approaches."  
  - `post` <https://bayesgroup.github.io/sufficient-statistics/posts/the-horseshoe-prior/> `in russian`

#### ["Bayesian Compression for Deep Learning"](https://arxiv.org/abs/1705.08665) Louizos, Ullrich, Welling
  `model selection`
  - `video` <http://videolectures.net/deeplearning2017_ullrich_bayesian_compression/> (Ullrich)
  - `post` <https://bayesgroup.github.io/sufficient-statistics/posts/the-horseshoe-prior/> `in russian`
  - `paper` ["Variational Dropout and the Local Reparameterization Trick"](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#variational-dropout-and-the-local-reparameterization-trick-kingma-salimans-welling) by Kingma, Salimans, Welling `summary`
  - `paper` ["Variational Gaussian Dropout is not Bayesian"](https://arxiv.org/abs/1711.02989) by Hron, Matthews, Ghahramani ([talk](https://youtu.be/k5hb4V73RY0?t=14m34s) by Alexander Matthews `video`)

#### ["Structured Bayesian Pruning via Log-Normal Multiplicative Noise"](https://arxiv.org/abs/1705.07283) Neklyudov, Molchanov, Ashukha, Vetrov
  `model selection`
  - `video` <https://youtube.com/watch?v=3zEYjw-cB4Y>
  - `video` <https://youtube.com/watch?v=SjYKP8BFhgw> (Nekludov)
  - `video` <https://youtu.be/jJDVYAxyE3U?t=32m45s> (Molchanov) `in russian`
  - `code` <https://github.com/necludov/group-sparsity-sbp>
  - `paper` ["Variational Dropout and the Local Reparameterization Trick"](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#variational-dropout-and-the-local-reparameterization-trick-kingma-salimans-welling) by Kingma, Salimans, Welling `summary`
  - `paper` ["Variational Gaussian Dropout is not Bayesian"](https://arxiv.org/abs/1711.02989) by Hron, Matthews, Ghahramani ([talk](https://vimeo.com/287804267) by Jiri Hron `video`, [talk](https://youtu.be/k5hb4V73RY0?t=14m34s) by Alexander Matthews `video`)

#### ["Variational Dropout Sparsifies Deep Neural Networks"](https://arxiv.org/abs/1701.05369) Molchanov, Ashukha, Vetrov
  `model selection`
>	"Interpretation of Gaussian dropout as performing variational inference in a network with log uniform priors over weights leads to sparsity in weights. This is an interesting approach, wherein sparsity stemsfrom variational optimization instead of the prior."  
  - `video` <https://vimeo.com/238221185> (Molchanov)
  - `video` <https://youtube.com/watch?v=jJDVYAxyE3U> (Molchanov) `in russian`
  - `video` <https://youtu.be/3Lxb-DqPtv4?t=4h45m22s> (Ashukha) `in russian`
  - `code` <https://github.com/BayesWatch/tf-variational-dropout>
  - `paper` ["Variational Dropout and the Local Reparameterization Trick"](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#variational-dropout-and-the-local-reparameterization-trick-kingma-salimans-welling) by Kingma, Salimans, Welling `summary`
  - `paper` ["Variational Gaussian Dropout is not Bayesian"](https://arxiv.org/abs/1711.02989) by Hron, Matthews, Ghahramani ([talk](https://youtu.be/k5hb4V73RY0?t=14m34s) by Alexander Matthews `video`)

----
#### ["Generalizing Hamiltonian Monte Carlo with Neural Networks"](https://arxiv.org/abs/1711.09268) Levy, Hoffman, Sohl-Dickstein
  `sampling` `posterior approximation`
>	"We present a general-purpose method to train Markov chain Monte Carlo kernels, parameterized by deep neural networks, that converge and mix quickly to their target distribution. Our method generalizes Hamiltonian Monte Carlo and is trained to maximize expected squared jumped distance, a proxy for mixing speed. We demonstrate large empirical gains on a collection of simple but challenging distributions, for instance achieving a 49x improvement in effective sample size in one case, and mixing when standard HMC makes no measurable progress in a second."  
>	"Hamiltonian Monte Carlo + Real NVP = trainable MCMC sampler that generalizes, and far outperforms, HMC."  
  - `notes` <https://colindcarroll.com/2018/01/20/a-summary-of-generalizing-hamiltonian-monte-carlo-with-neural-networks/>
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
  - `video` <https://youtu.be/quIuMYSLaYM?t=8m42s> (Liu)
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
  - `post` <http://bjlkeng.github.io/posts/variational-autoencoders-with-inverse-autoregressive-flows>
  - `post` <http://akosiorek.github.io/ml/2018/04/03/norm_flows.html>
  - `post` <https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#inverse-autoregressive-flow>
  - `code` <https://github.com/openai/iaf>

#### ["Neural Variational Inference and Learning in Undirected Graphical Models"](https://arxiv.org/abs/1711.02679) Kuleshov, Ermon
  `variational inference` `posterior approximation` `auxiliary variables` `discrete latent variables`
>	"We propose black-box learning and inference algorithms for undirected models that optimize a variational approximation to the log-likelihood of the model. Central to our approach is an upper bound on the log-partition function parametrized by a function q that we express as a flexible neural network."  
>	"Our approach makes generative models with discrete latent variables both more expressive and easier to train."  
>	"Our approach can also be used to train hybrid directed/undirected models using a unified variational framework. Such hybrid models are similar in spirit to deep belief networks. From a statistical point of view, a latent variable prior makes the model more flexible and allows it to better fit the data distribution. Such models may also learn structured feature representations: previous work has shown that undirected modules may learn classes of digits, while lower, directed layers may learn to represent finer variation. Finally, undirected models like the ones we study are loosely inspired by the brain and have been studied from that perspective. In particular, the undirected prior has been previously interpreted as an associative memory module."  
>	"Our work proposes an alternative to sampling-based learning methods; most variational methods for undirected models center on inference. Our approach scales to small and medium-sized datasets, and is most useful within hybrid directed-undirected generative models. It approaches the speed of the Persistent Contrastive Divergence method and offers additional benefits, such as partition function tracking and accelerated sampling. Most importantly, our algorithms are black-box, and do not require knowing the structure of the model to derive gradient or partition function estimators. We anticipate that our methods will be most useful in automated inference systems such as Edward."  
>	"Our approach offers a number of advantages over previous methods. First, it enables training undirected models in a black-box manner, i.e. we do not need to know the structure of the model to compute gradient estimators (e.g., as in Gibbs sampling); rather, our estimators only require evaluating a model’s unnormalized probability. When optimized jointly over q and p, our bound also offers a way to track the partition function during learning. At inference-time, the learned approximating distribution q may be used to speed-up sampling from the undirected model my initializing an MCMC chain (or it may itself provide samples)."  
  - `video` <https://youtu.be/quIuMYSLaYM?t=2m36s> (Kuleshov)

#### ["Auxiliary Deep Generative Models"](http://arxiv.org/abs/1602.05473) Maaløe, Sønderby, Sønderby, Winther
  `variational inference` `posterior approximation` `auxiliary variables`
>	"Several families of q have been proposed to ensure that the approximating distribution is sufficiently flexible to fit p. This work makes use of a class of distributions q(x,a) = q(x|a)q(a) that contain auxiliary variables a; these are latent variables that make the marginal q(x) multimodal, which in turn enables it to approximate more closely a multimodal target distribution p(x)."  
  - `video` <http://techtalks.tv/talks/auxiliary-deep-generative-models/62509/> (Maaløe)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Variational-Inference-Foundations-and-Modern-Methods> (1:23:13) (Mohamed)
  - `code` <https://github.com/larsmaaloee/auxiliary-deep-generative-models>

#### ["Hierarchical Variational Models"](https://arxiv.org/abs/1511.02386) Ranganath, Tran, Blei
  `variational inference` `posterior approximation` `auxiliary variables`
>	"Several families of q have been proposed to ensure that the approximating distribution is sufficiently flexible to fit p. This work makes use of a class of distributions q(x,a) = q(x|a)q(a) that contain auxiliary variables a; these are latent variables that make the marginal q(x) multimodal, which in turn enables it to approximate more closely a multimodal target distribution p(x)."  
  - `video` <http://techtalks.tv/talks/hierarchical-variational-models/62504/> (Ranganath)
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Variational-Inference-Foundations-and-Modern-Methods> (1:23:13) (Mohamed)

#### ["Variational Inference for Monte Carlo Objectives"](http://arxiv.org/abs/1602.06725) Mnih, Rezende
  `variational inference` `posterior approximation` `discrete latent variables`
  - `video` <http://techtalks.tv/talks/variational-inference-for-monte-carlo-objectives/62507/>
  - `video` <https://youtu.be/_XRBlhzb31U?t=27m16s> (Figurnov) `in russian`
  - `notes` <http://artem.sobolev.name/posts/2016-07-14-neural-variational-importance-weighted-autoencoders.html>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/MnihR16>
  - `notes` <http://tuananhle.co.uk/notes/vimco.html>

#### ["Importance Weighted Autoencoders"](http://arxiv.org/abs/1509.00519) Burda, Grosse, Salakhutdinov
  `variational inference` `posterior approximation`
>	"As we show empirically, the VAE objective can lead to overly simplified representations which fail to use the network's entire modeling capacity. We present the importance weighted autoencoder, a generative model with the same architecture as the VAE, but which uses a strictly tighter log-likelihood lower bound derived from importance weighting. In the IWAE, the recognition network uses multiple samples to approximate the posterior, giving it increased flexibility to model complex posteriors which do not fit the VAE modeling assumptions."  
  - `video` <https://facebook.com/nipsfoundation/videos/1555493854541848?t=1771> (Teh)
  - `post` <http://dustintran.com/blog/importance-weighted-autoencoders/>
  - `post` <https://casmls.github.io/general/2017/04/24/iwae-aae.html>
  - `notes` <http://artem.sobolev.name/posts/2016-07-14-neural-variational-importance-weighted-autoencoders.html>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/BurdaGS15>
  - `code` <https://github.com/yburda/iwae>
  - `code` <https://github.com/arahuja/generative-tf>
  - `code` <https://github.com/blei-lab/edward/blob/master/examples/iwvi.py>

----
#### ["Implicit Reparameterization Gradients"](https://arxiv.org/abs/1805.08498) Figurnov, Mohamed, Mnih
  `variables with complex distributions`
>	"Unlike reparameterization trick, implicit reparameterization gradients are applicable to a number of important continuous distributions with numerically tractable CDFs such as truncated, mixture, Gamma, Beta, Dirichlet, Student-t, or von Mises, which can be used as easily as the Normal distribution in stochastic computation graphs and are both faster and more accurate than alternative approaches."  
>	"Implicit reparameterization gradients can outperform existing stochastic variational methods at training the Latent Dirichlet Allocation topic model in a black-box fashion using amortized inference."  
>	"Implicit reparameterization gradients can be used to train VAEs with Gamma, Beta, and von Mises latent variables, leading to latent spaces with interesting alternative topologies."  
>	"Following Graves, we use implicit differentiation to differentiate the CDF rather than its inverse. While the method of Graves is only practical for distributions with analytically tractable CDFs and has been used solely with mixture distributions, we leverage automatic differentiation to handle distributions with numerically tractable CDFs."  
>	"One common example of implicit differentiation is the inverse function theorem. Suppose that you can easily compute y = f(x), but computing f^-1(y) is expensive/hard. Then, you can take points x, compute y = f(x), and by definition have x = f^-1(y). The inverse function theorem tells you that (f^-1(y))' = 1 / f'(x) . So, you can compute the derivative of the inverse even when you cannot easily compute x = f^-1(y) for any y. Implicit differentiation is an extension of this idea that handles the case when the inverse is taken w.r.t. one argument and differentiation is performed w.r.t. another."  
  - `video` <https://youtu.be/BV465lgleHA?t=44m31s> (Figurnov)
  - `video` <https://youtube.com/watch?v=5bGdZhonDrg> (Mnih)
  - <https://i.imgur.com/iEVFES7.png>
  - <https://www.math.ucdavis.edu/~kouba/CalcOneDIRECTORY/implicitdiffdirectory/ImplicitDiff.html>
  - `code` <https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/latent_dirichlet_allocation.py>

#### ["Backpropagation through the Void: Optimizing Control Variates for Black-box Gradient Estimation"](https://arxiv.org/abs/1711.00123) Grathwohl, Choi, Wu, Roeder, Duvenaud
  `variables with discrete distributions` `non-differentiable loss` `RELAX`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#backpropagation-through-the-void-optimizing-control-variates-for-black-box-gradient-estimation-grathwohl-choi-wu-roeder-duvenaud>

#### ["REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models"](http://arxiv.org/abs/1703.07370) Tucker, Mnih, Maddison, Lawson, Sohl-Dickstein
  `variables with discrete distributions` `REBAR`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#rebar-low-variance-unbiased-gradient-estimates-for-discrete-latent-variable-models-tucker-mnih-maddison-lawson-sohl-dickstein>

#### ["The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables"](http://arxiv.org/abs/1611.00712) Maddison, Mnih, Teh
  `variables with discrete distributions`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#the-concrete-distribution-a-continuous-relaxation-of-discrete-random-variables-maddison-mnih-teh>

#### ["Categorical Reparametrization with Gumbel-Softmax"](http://arxiv.org/abs/1611.01144) Jang, Gu, Poole
  `variables with discrete distributions`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#categorical-reparametrization-with-gumbel-softmax-jang-gu-poole>

#### ["Learning Latent Permutations with Gumbel-Sinkhorn Networks"](https://arxiv.org/abs/1802.08665) Mena, Belanger, Linderman, Snoek
  `variables with complex discrete distributions`
>	"A new method for gradient-descent inference of permutations."  

#### ["Reparameterizing the Birkhoff Polytope for Variational Permutation Inference"](https://arxiv.org/abs/1710.09508) Linderman, Mena, Cooper, Paninski, Cunningham
  `variables with complex discrete distributions`
>	"Many matching, tracking, sorting, and ranking problems require probabilistic reasoning about possible permutations, a set that grows factorially with dimension. Combinatorial optimization algorithms may enable efficient point estimation, but fully Bayesian inference poses a severe challenge in this high-dimensional, discrete space. To surmount this challenge, we start with the usual step of relaxing a discrete set (here, of permutation matrices) to its convex hull, which here is the Birkhoff polytope: the set of all doublystochastic matrices. We then introduce two novel transformations: first, an invertible and differentiable stick-breaking procedure that maps unconstrained space to the Birkhoff polytope; second, a map that rounds points toward the vertices of the polytope. Both transformations include a temperature parameter that, in the limit, concentrates the densities on permutation matrices."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1710.09508>

#### ["Reparameterization Gradients through Acceptance-Rejection Sampling Algorithms"](http://arxiv.org/abs/1610.05683) Naesseth, Ruiz, Linderman, Blei
  `variables with complex distributions` `RSVI`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#reparameterization-gradients-through-acceptance-rejection-sampling-algorithms-naesseth-ruiz-linderman-blei>

#### ["The Generalized Reparameterization Gradient"](http://arxiv.org/abs/1610.02287) Ruiz, Titsias, Blei
  `variables with complex distributions`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#the-generalized-reparameterization-gradient-ruiz-titsias-blei>

#### ["Stochastic Backpropagation through Mixture Density Distributions"](http://arxiv.org/abs/1607.05690) Graves
  `variables with mixture distributions`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#stochastic-backpropagation-through-mixture-density-distributions-graves>

#### ["Stick-Breaking Variational Autoencoders"](http://arxiv.org/abs/1605.06197) Nalisnick, Smyth
  `variables with stochastic dimensionality`
  - `post` <http://blog.shakirm.com/2015/12/machine-learning-trick-of-the-day-6-tricks-with-sticks/>
  - `code` <https://github.com/enalisnick/stick-breaking_dgms>

----
#### ["Deep Neural Networks as Gaussian Processes"](https://arxiv.org/abs/1711.00165) Lee, Bahri, Novak, Schoenholz, Pennington, Sohl-Dickstein
  `bayesian model`

#### ["Bayesian GAN"](https://arxiv.org/abs/1705.09558) Saatchi, Wilson
  `bayesian model`
>	"Instead of learning one generative network learn a distribution over networks. To generate an example: draw random network, draw random sample."  
>	"We marginalize the posteriors over the weights of generator and discriminator using stochastic gradient Hamiltonian Monte Carlo."  
>	"The simplicity of the proposed approach is one of its greatest strengths: inference is straightforward, interpretable, and stable. Indeed all of the experimental results were obtained without feature matching, normalization, or any ad-hoc techniques."  
>	"We interpret data samples from the generator, showing exploration across several distinct modes in the generator weights."  
  - `video` <https://youtube.com/watch?v=24A8tWs6aug>
  - `video` <https://facebook.com/nipsfoundation/videos/1554402331317667?t=4778> (Saatchi, Wilson)
  - `video` <https://youtube.com/watch?v=8rOnLuD2l6o> (Wilson)
  - `video` <https://youtu.be/ZHucm52V3Zw?t=52m36s> (Umnov)
  - `code` <https://github.com/andrewgordonwilson/bayesgan/>

#### ["Bayesian Recurrent Neural Networks"](https://arxiv.org/abs/1704.02798) Fortunato, Blundell, Vinyals
  `bayesian model` `uncertainty estimation` `Bayes by Backprop`
  - `video` <https://vimeo.com/249562717> (Fortunato)
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
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#composing-graphical-models-with-neural-networks-for-structured-representations-and-fast-inference-johnson-duvenaud-wiltschko-datta-adams>

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
  - `video` <https://youtube.com/watch?v=jp3noyIYAbA> (Wood)
  - `post` <http://tuananhle.co.uk/notes/amortized-inference.html>



---
### compute and memory architectures

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---architectures)

----
#### ["Hybrid Computing using a Neural Network with Dynamic External Memory"](http://rdcu.be/kXhV) Graves et al.
  `compute and memory` `Differentiable Neural Computer`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#hybrid-computing-using-a-neural-network-with-dynamic-external-memory-graves-et-al>

#### ["Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes"](http://arxiv.org/abs/1610.09027) Rae, Hunt, Harley, Danihelka, Senior, Wayne, Graves, Lillicrap
  `compute and memory` `Differentiable Neural Computer`
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255?t=1727> (Graves)

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
#### ["Continual Lifelong Learning with Neural Networks: A Review"](https://arxiv.org/abs/1802.07569) Parisi, Kemker, Part, Kanan, Wermter
  `continual learning`

#### ["Learning Long-term Dependencies with Deep Memory States"](https://people.eecs.berkeley.edu/~vitchyr/learning_long_term_dependencies_with_deep_memory_states__pong_gu_levine.pdf) Pong, Gu, Levine
  `continual learning`
>	"Training an agent to use past memories to adapt to new tasks and environments is important for lifelong learning algorithms. Training such an agent to use its memory efficiently is difficult as the size of its memory grows with each successive interaction. Previous work has not yet addressed this problem, as they either use backpropagation through time (BPTT), which is computationally expensive, or truncated BPTT, which cannot learn long-term dependencies, to train recurrent policies. We propose a reinforcement learning method that addresses the limitations of truncated BPTT by using a learned critic to estimate truncated gradients and by saving and loading hidden states outputted by recurrent neural networks. We present results showing that our algorithm can learn long-term dependencies while avoiding the computational constraints of BPTT. These results suggest that our method can potentially be used to train an agent that stores and effectively learns from past memories."  
>	"While feed-forward, reactive policies can perform complex skills in isolation, the ability to store past events in an internal memory is crucial for a wide range of behaviors. For example, a robot navigating a building might need to incorporate past observations to optimally estimate its location, or remember a command previously issued by a person. Perhaps more importantly, long-term memory can be utilized for lifelong learning, where an agent uses past experiences to quickly modify its behavior in response to changing environments. Such recurrent meta-learning has been demonstrated on a variety of supervised learning tasks, and more recently applied to a variety of reinforcement learning tasks. However, realistic applications of policies with memory may demand particularly long-term memorization. For example, a robot tasked with setting the silverware on the table would need to remember where it last stored it, potentially hours or days ago. This kind of long-term memorization is very difficult with current reinforcement learning methods. Specialized architectures have been developed that improve the capabilities of recurrent networks to store information, but such methods still require back-propagation through time (BPTT) for training, which typically limits how far back the error is propagated to at most the length of the trajectory. Past this size, the gradient is truncated. Truncating the gradient between when the policy must perform a crucial task (such as finding the silverware) and the observation that needs to be memorized to know which action to perform (the last location of the silverware) can make it impossible to successfully perform the task."  
>	"While some tasks may be solved by loading entire episodes into memory and avoiding truncation, a lifelong learning agent has no notion of episodes. Instead, a lifelong learning agent lives out a single episode that continuously grows. Computational constraints, both in terms of memory and practical training times, impose a fundamental limit on the memory capacity of neural network policies. Rather than loading full episodes or truncating gradients, one can instead augment the original MDP with memory states. In addition to regular MDP actions, a policy outputs a vector called memory states, which it receives as input at the next time step. These memory states are equivalent to hidden states in normal recurrent neural network, but by interpreting memory states as just another part of the MDP state, recurrent policies can be trained using standard reinforcement learning methods, including efficient off-policy algorithms that can handle potentially infinite episode lengths. However, the use of memory states forces the learning algorithm to rely on the much less efficient gradient-free RL optimization to learn memorization strategies, rather than the low-variance gradients obtained from back-propagation through time (BPTT). For this reason, even truncated BPTT is usually preferred over the memory states approach when using model-free RL algorithms."  
>	"We propose a hybrid recurrent reinforcement learning algorithm that combines both memory states and BPTT. To obtain a practical algorithm that enables memorization over potentially unbounded episodes, we must use some form of memory states to manage computational constraints. However, we must also use BPTT as much as possible to make it feasible for the learner to acquire appropriate memorization strategies. Our actor-critic algorithm includes memory states and write actions, but performs analytic BPTT within each batch, loading subsequences for each training iteration. This approach allows us to use batches of reasonable size with enough subsequences to decorrelate each batch, while still benefiting from the efficiency of BPTT. Unfortunately, the use of memory states by itself is insufficient to provide for a  Markovian state description, since an untrained policy may not store the right information in the memory. This makes it difficult to use memory states with a critic, which assumes Markovian state. To address this issue, we also propose a method for backpropagating Bellman error gradients, which encourages the policy to take write actions that reduce future Bellman error."  

#### ["Learning to Remember Rare Events"](http://arxiv.org/abs/1703.03129) Kaiser, Nachum, Roy, Bengio
  `continual learning` `catastrophic forgetting`
>	"We present a large-scale life-long memory module for use in deep learning. The module exploits fast nearest-neighbor algorithms for efficiency and thus scales to large memory sizes. Except for the nearest-neighbor query, the module is fully differentiable and trained end-to-end with no extra supervision. It operates in a life-long manner, i.e., without the need to reset it during training. Our memory module can be easily added to any part of a supervised neural network. The enhanced network gains the ability to remember and do life-long one-shot learning. Our module remembers training examples shown many thousands of steps in the past and it can successfully generalize from them."  
  - `code` <https://github.com/tensorflow/models/tree/master/research/learning_to_remember_rare_events>

#### ["A Growing Long-term Episodic and Semantic Memory"](http://arxiv.org/abs/1610.06402) Pickett, Al-Rfou, Shao, Tar
  `continual learning` `catastrophic forgetting`
>	"We describe a lifelong learning system that leverages a fast, though non-differentiable, content-addressable memory which can be exploited to encode both a long history of sequential episodic knowledge and semantic knowledge over many episodes for an unbounded number of domains."  

#### ["Variational Continual Learning"](https://arxiv.org/abs/1710.10628) Nguyen, Li, Bui, Turner
  `continual learning` `catastrophic forgetting`
>	"The framework can successfully train both deep discriminative models and deep generative models in complex continual learning settings where existing tasks evolve over time and entirely new tasks emerge. Experimental results show that variational continual learning outperforms state-of-the-art continual learning methods on a variety of tasks, avoiding catastrophic forgetting in a fully automatic way."  
  - `video` <https://youtube.com/watch?v=qRXPS_6fAfE> (Turner)

#### ["Gradient Episodic Memory for Continuum Learning"](https://arxiv.org/abs/1706.08840) Lopez-Paz, Ranzato
  `continual learning` `catastrophic forgetting`
  - `notes` <http://rayraycano.github.io/data%20science/tech/2017/07/31/A-Paper-a-Day-GEM.html>

#### ["Improved Multitask Learning Through Synaptic Intelligence"](https://arxiv.org/abs/1703.04200) Zenke, Poole, Ganguli
  `continual learning` `catastrophic forgetting`
>	"The regularization penalty is similar to EWC. However, our approach computes the per-synapse consolidation strength in an online fashion, whereas for EWC synaptic importance is computed offline after training on a designated task."  
  - `video` <https://vimeo.com/238242232> (Zenke)
  - `code` <https://github.com/ganguli-lab/pathint>
  - `code` <https://github.com/spiglerg/TF_ContinualLearningViaSynapticIntelligence>

#### ["PathNet: Evolution Channels Gradient Descent in Super Neural Networks"](http://arxiv.org/abs/1701.08734) Fernando, Banarse, Blundell, Zwols, Ha, Rusu, Pritzel, Wierstra
  `continual learning` `catastrophic forgetting`
  - `video` <https://youtube.com/watch?v=Wkz4bG_JlcU>
  - `video` <https://vimeo.com/250399122> (Fernando)
  - `post` <https://medium.com/intuitionmachine/pathnet-a-modular-deep-learning-architecture-for-agi-5302fcf53273>
  - `code` <https://github.com/jaesik817/pathnet>

#### ["Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"](https://arxiv.org/abs/1701.06538) Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean
  `continual learning` `catastrophic forgetting`
>	"The Mixture of Experts Layer is trained using back-propagation. The Gating Network outputs an (artificially made) sparse vector that acts as a chooser of which experts to consult. More than one expert can be consulted at once (although the paper doesn’t give any precision on the optimal number of experts). The Gating Network also decides on output weights for each expert."  
>	"The MoE with experts shows higher accuracy (or lower perplexity) than the state of the art using only 16% of the training time."  
  - `video` <http://videocrm.ca/Machine18/Machine18-20180423-5-YoshuaBengio.mp4> (14:39) (Bengio)
  - `video` <https://youtube.com/watch?v=nNZceFX2tQU> (Zakirov) `in russian`
  - `notes` <https://github.com/dennybritz/deeplearning-papernotes/blob/3718d181a0fed5ed806582822ed0dbde530122bf/notes/mixture-experts.md>

#### ["Overcoming Catastrophic Forgetting in Neural Networks"](http://arxiv.org/abs/1612.00796) Kirkpatrick et al.
  `continual learning` `catastrophic forgetting` `Elastic Weight Consolidation` `EWC`
>	"EWC adds regularization term to the loss which reflects a Gaussian prior for each parameter of neural network whose means are the old parameters. It uses the approximate Fisher information as a way of estimating the Hessian to assess importance, which implicitly sets the variance of each parameter prior."  
>	"The quadratic penalty/penalties prevent the network from forgetting what it has learnt from previous data - you can think of the quadratic penalty as a summary of the information from the data it has seen so far."  
>	"You can apply EWC at the level of learning tasks sequentially, or you can even apply it to on-line learning in a single task (in case you can't loop over the same minibatches several time like you do in SGD)."  
>	"EWC is an on-line sequential (diagonalized) Laplace approximation of Bayesian learning."  
  - `paper` <http://www.pnas.org/content/early/2017/03/13/1611835114.abstract>
  - `post` <https://deepmind.com/blog/enabling-continual-learning-in-neural-networks/>
  - `video` <https://vimeo.com/238221551#t=13m9s> (Hadsell)
  - `video` <https://facebook.com/nipsfoundation/videos/1555493854541848?t=1078> (Teh)
  - `post` <http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/main.html>
  - `post` <http://inference.vc/comment-on-overcoming-catastrophic-forgetting-in-nns-are-multiple-penalties-needed-2/>
  - `notes` <https://theneuralperspective.com/2017/04/01/overcoming-catastrophic-forgetting-in-neural-networks/>
  - `notes` <http://shortscience.org/paper?bibtexKey=kirkpatrick2016overcoming>
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
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255?t=2368> (Graves)
  - `video` <https://vimeo.com/240428387#t=1h28m28s> (Vinyals)
  - `video` <https://youtube.com/watch?v=xbWzoAbb8dM> (Laver)
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
>
>	"- requires less training data  
>	- position and pose information are preserved (equivarience)  
>	- promising for image segmentation and object detection  
>	- routing by agreement is great for overlapping objects (explaining away)  
>	- capsule activations nicely map the hierarchy of parts  
>	- offers robustness to affine transformations  
>	- activation vectors are easier to interpret (rotation, thickness, skew)"  
  - `video` <https://youtube.com/watch?v=gq-7HgzfDBM>
  - `video` <https://facebook.com/nipsfoundation/videos/1553634558061111?t=5824> (Frosst)
  - `video` ["What is wrong with convolutional neural nets?"](https://youtube.com/watch?v=Mqt8fs6ZbHk) (Hinton)
  - `video` ["What's wrong with convolutional nets?"](http://techtv.mit.edu/collections/bcs/videos/30698-what-s-wrong-with-convolutional-nets) (Hinton) ([transcription](https://github.com/WalnutiQ/walnut/issues/157))
  - `video` ["Does the Brain do Inverse Graphics?"](https://youtube.com/watch?v=TFIMqt0yT2I) (Hinton)
  - `video` <https://youtube.com/watch?v=pPN8d0E3900> (Geron)
  - `video` <https://youtube.com/watch?v=EATWLTyLfmc> (Canziani)
  - `video` <https://youtube.com/watch?v=hYt3FcJUf6w> (Uziela)
  - `video` <https://youtube.com/watch?v=VKoLGnq15RM> (Raval)
  - `video` <https://youtube.com/watch?v=UZ9BgrofhKk> (Kozlov) `in russian`
  - `video` <https://youtube.com/watch?v=8R3gXmh1F0c> (Lykov) `in russian`
  - `post` <https://oreilly.com/ideas/introducing-capsule-networks>
  - `post` <https://medium.com/@pechyonkin/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b>
  - `post` <https://hackernoon.com/uncovering-the-intuition-behind-capsule-networks-and-inverse-graphics-part-i-7412d121798d>
  - `post` <https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/>
  - `post` <https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc>
  - `post` <https://medium.com/@mike_ross/a-visual-representation-of-capsule-network-computations-83767d79e737>
  - `notes` <https://blog.acolyer.org/2017/11/13/dynamic-routing-between-capsules/>
  - `code` <https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb>
  - `code` <https://github.com/Sarasra/models/tree/master/research/capsules>
  - `code` <https://github.com/loretoparisi/CapsNet>
  - `paper` ["Transforming Auto-encoders"](http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf) by Hinton, Krizhevsky, Wang
  - `paper` ["Optimizing Neural Networks that Generate Images"](http://www.cs.toronto.edu/~tijmen/tijmen_thesis.pdf) by Tieleman ([code](https://github.com/mrkulk/Unsupervised-Capsule-Network))

#### ["Matrix Capsules with EM Routing"](https://openreview.net/forum?id=HJWLfGWRb) Hinton, Sabour, Frosst
  `information routing` `CapsNet`
  - `video` <https://youtu.be/hYt3FcJUf6w?t=50m38s> (Uziela)
  - `video` <https://youtube.com/watch?v=akq6PNnkKY8> (Ou)
  - `post` <https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-Capsule-Network/>
  - `post` <https://towardsdatascience.com/demystifying-matrix-capsules-with-em-routing-part-1-overview-2126133a8457>
  - `post` <https://staff.fnwi.uva.nl/s.abnar/?p=108>
  - `notes` <https://blog.acolyer.org/2017/11/14/matrix-capsules-with-em-routing/>
  - `code` <https://github.com/loretoparisi/CapsNet>

#### ["Decoupled Neural Interfaces using Synthetic Gradients"](http://arxiv.org/abs/1608.05343) Jaderberg, Czarnecki, Osindero, Vinyals, Graves, Silver, Kavukcuoglu
  `information routing`
>	"We incorporate a learnt model of error gradients, which means we can update networks without full backpropagation. We show how this can be applied to feed-forward networks which allows every layer to be trained asynchronously, to RNNs which extends the time over which models can remember, and to multi-network systems to allow communication."  
  - `post` <https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/>
  - `video` <https://youtu.be/tA8nRlBEVr0?t=14m40s> + <https://youtube.com/watch?v=-u32TOPGIbQ> (Graves)
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255?t=1858> (Graves)
  - `video` <https://youtube.com/watch?v=1z_Gv98-mkQ> (Geron)
  - `video` <https://youtube.com/watch?v=qirjknNY1zo> (Raval)
  - `post` <https://iamtrask.github.io/2017/03/21/synthetic-gradients/>
  - `notes` <http://cnichkawde.github.io/SyntheticGradients.html>
  - `code` <https://github.com/hannw/sgrnn>
  - `code` <https://github.com/koz4k/dni-pytorch>

#### ["Understanding Synthetic Gradients and Decoupled Neural Interfaces"](http://arxiv.org/abs/1703.00522) Czarnecki, Swirszcz, Jaderberg, Osindero, Vinyals, Kavukcuoglu
  `information routing`
  - `video` <https://vimeo.com/238275152> (Swirszcz)

----
#### ["SMASH: One-Shot Model Architecture Search through HyperNetworks"](https://arxiv.org/abs/1708.05344) Brock, Lim, Ritchie, Weston
  `architecture search` `SMASH`
>	"The architecture: at each training step, generate the schematics of a random NN architecture; feed the skeleton into the hypernetwork, which will directly spit out numbers for each neuron (as a convolutional hypernetwork it can handle big and small NNs the same way); with the fleshed out NN, train 1 minibatch on the image classification task as usual, and update its parameters; use that update as the error for the hypernetwork to train it to spit out weights for that skeleton which are slightly closer to what it was after 1 minibatch. After training the hypernetwork many times on many random NN architectures, its generated weights will be close to what training each random NN architecture from scratch would have been. Now you can simply generate lots of random NN architectures, fill them in, run them on a small validation set, and see their final performance without ever actually training them fully (which would be like 10,000x more expensive). So this runs on 1 GPU in a day or two versus papers like Zoph which used 800 GPUs for a few weeks. It’s amazing this works, and like synthetic gradients it troubles me a little because it implies that even complex highly sophisticated NNs are in some sense simple & predictable as their weights/error-gradients can be predicted by other NNs which are as small as linear layers or don’t even see the data, and thus are incredibly wasteful in both training & parameter size, implying a large hardware overhang."  
  - `video` <https://youtube.com/watch?v=79tmPL9AL48>
  - `code` <https://github.com/ajbrock/SMASH>

#### ["Learning Transferable Architectures for Scalable Image Recognition"](https://arxiv.org/abs/1707.07012) Zoph, Vasudevan, Shlens, Le
  `architecture search` `NASNet`
  - `video` <https://vimeo.com/250399509> (Le)
  - `post` <https://research.googleblog.com/2017/11/automl-for-large-scale-image.html>

#### ["Neural Architecture Search with Reinforcement Learning"](http://arxiv.org/abs/1611.01578) Zoph, Le
  `architecture search`
  - `post` <https://research.googleblog.com/2017/05/using-machine-learning-to-explore.html>
  - `video` <https://youtube.com/watch?v=XDtFXBYpl1w> (Le)
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255?t=4111> (Zoph)
  - `notes` <https://blog.acolyer.org/2017/05/10/neural-architecture-search-with-reinforcement-learning/>



---
### meta-learning

----
#### ["On First-Order Meta-Learning Algorithms"](https://arxiv.org/abs/1803.02999) Nichol, Achiam, Schulman
  `Reptile` `learning initialization algorithm`
>	"We analyze a family of algorithms for learning a parameter initialization that can be fine-tuned quickly on a new task, using only first-order derivatives for the meta-learning updates. This family includes and generalizes first-order MAML, an approximation to MAML obtained by ignoring second-order derivatives. It also includes Reptile which works by repeatedly sampling a task, training on it, and moving the initialization towards the trained weights on that task."  
  - `post` <https://blog.openai.com/reptile>
  - `notes` <https://yobibyte.github.io/files/paper_notes/Reptile___a_Scalable_Metalearning_Algorithm__Alex_Nichol_and_John_Schulman__2018.pdf>
  - `code` <https://github.com/openai/supervised-reptile>

#### ["Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments"](https://arxiv.org/abs/1710.03641) Al-Shedivat, Bansal, Burda, Sutskever, Mordatch, Abbeel
  `learning initialization algorithm` `continual learning`
>	"extending Model-Agnostic Meta-Learning to the case of dynamically changing tasks"  
  - <https://sites.google.com/view/adaptation-via-metalearning> (demo)
  - `post` <https://blog.openai.com/meta-learning-for-wrestling>
  - `video` <https://facebook.com/iclr.cc/videos/2126769937352061?t=4851> (Al-Shedivat)
  - `video` <https://facebook.com/nipsfoundation/videos/1554594181298482?t=1838> (Abbeel)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1710.03641>
  - `code` <https://github.com/openai/robosumo>

#### ["Some Considerations on Learning to Explore via Meta-Reinforcement Learning"](https://arxiv.org/abs/1803.01118) Stadie, Yang, Houthooft, Chen, Duan, Wu, Abbeel, Sutskever
  `E-MAML` `E-RL^2` `learning initialization algorithm`
>	"We introduce two new algorithms: E-MAML and E-RL2, which are derived by reformulating the underlying meta-learning objective to account for the impact of initial sampling on future (post-meta-updated) returns."  
>	"Meta RL agent must not learn how to master the environments it is given, but rather it must learn how to learn so that it can quickly train at test time."  
>	"Due to RL^2’s policy-gradient-based optimization procedure, it does not directly optimize the final policy performance nor exhibit exploration. In E-RL^2 the rewards for episodes sampled early in the learning process are deliberately set to zero to drive exploratory behavior."  
>	"It is likely that future work in this area will focus on meta-learning a curiosity signal which is robust and transfers across tasks. Perhaps this will enable meta agents which learn to explore rather than being forced to explore by mathematical trickery in their objectives."  
  - `video` <https://youtu.be/16UUb4HF0fo?t=54m56s> (Golikov) `in russian`
  - `code` <https://github.com/bstadie/krazyworld>

#### ["Recasting Gradient-Based Meta-Learning as Hierarchical Bayes"](https://arxiv.org/abs/1801.08930) Grant, Finn, Levine, Darrell, Griffiths
  `MAML` `learning initialization algorithm`

#### ["Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm"](https://arxiv.org/abs/1710.11622) Finn, Levine
  `MAML` `learning initialization algorithm`
>	"It is particularly important to develop meta-learning algorithms that can generalize well from a manageable number of training tasks, in the sense that it can perform well on new tasks sampled from the same distribution. One idea is to restrict the set of algorithms expressible by the parameterized model. At one extreme we have methods such as RL^2, TCML and one-shot imitation learning, which use generic recurrent architectures and are, in theory, capable of approximating any algorithm that can run on a Turing machine. At the other extreme, we have methods that merely tune hyperparameters over a set of training tasks. There are many possibilities between these two extremes. MAML restricts the underlying algorithm to be policy gradient and only meta-learn an initial set of parameters."  
  - `video` <https://facebook.com/nipsfoundation/videos/1554594181298482?t=1204> (Abbeel)

#### ["Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"](https://arxiv.org/abs/1703.03400) Finn, Abbeel, Levine
  `MAML` `learning initialization algorithm`
>	"End-to-end learning of parameter θ that is good init for fine-tuning for many tasks."  
>	"MAML finds a shared parameter θ such that for a given task, one gradient step on θ using the training set will yield a model with good predictions on the test set. Then, a meta-gradient update is performed from the test error through the one gradient step in the training set, to update θ."  
>	"Tasks are sampled and a policy gradient update is computed for each task with respect to a fixed initial set of parameters. Subsequently, a meta update is performed where a gradient step is taken that moves the initial parameter in a direction that would have maximally benefited the average return over all of the sub-updates."  
>	"Unlike prior methods, the MAML learner’s weights are updated using the gradient, rather than a learned update rule. Our method does not introduce any additional parameters into the learning process and does not require a particular learner model architecture."  
  - <https://sites.google.com/view/maml> (demo)
  - `video` <https://youtu.be/Ko8IBbYjdq8?t=18m51s> (Finn)
  - `video` <https://youtu.be/lYU5nq0dAQQ?t=44m57s> (Levine)
  - `video` <https://facebook.com/nipsfoundation/videos/1554594181298482?t=1085> (Abbeel)
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=1h1m22s> + <https://youtube.com/watch?v=ID150Tl-MMw&t=1h9m10s> (Abbeel)
  - `video` <https://youtu.be/16UUb4HF0fo?t=7m34s> (Golikov) `in russian`
  - `post` <http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/> (Finn)
  - `post` <https://blog.evjang.com/2019/02/maml-jax.html>
  - `post` <http://noahgolmant.com/maml.html>
  - `code` <https://github.com/tristandeleu/pytorch-maml-rl>
  - `code` <https://github.com/cbfinn/maml>
  - `code` <https://github.com/cbfinn/maml_rl>

----
#### ["Optimization as a Model for Few-Shot Learning"](https://openreview.net/forum?id=rJY0-Kcll) Ravi, Larochelle
  `learning initialization algorithm` `learning optimization algorithm`
>	"Meta-learning algorithm is decomposed into two parts: the traditional learner’s initial parameters are trained to be suitable for fast gradient-based adaptation; the LSTM meta-learner is trained to be an optimization algorithm adapted for meta-learning tasks."  
>	"Encoding network reads the training set and generate the parameters of a model, which is trained to perform well on the testing set."  
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255?t=5208> (Ravi)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/#t=4088> (de Freitas)
  - `video` <https://youtu.be/QIcpGa-_bvA?t=34m> (Vinyals)
  - `code` <https://github.com/twitter/meta-learning-lstm>
  - `code` <https://github.com/gitabcworld/FewShotLearning>

----
#### ["A Simple Neural Attentive Meta-Learner"](https://arxiv.org/abs/1707.03141) Mishra, Rohaninejad, Chen, Abbeel
  `SNAIL` `learning learning algorithm`
  - <https://sites.google.com/view/snail-iclr-2018/> (demo)
  - `video` <https://facebook.com/nipsfoundation/videos/1554594181298482?t=970> (Abbeel)

#### ["Meta-Learning with Temporal Convolutions"](https://arxiv.org/abs/1707.03141) Mishra, Rohaninejad, Chen, Abbeel
  `TCML` `learning learning algorithm`
>	"Like RL^2 but with LSTM network replaced by dilated temporal convolutional network and attention."  
>	"Most recent approaches to meta-learning are extensively hand-designed, either using architectures that are specialized to a particular application, or hard-coding algorithmic components that tell the meta-learner how to solve the task. We propose a class of simple and generic meta-learner architectures, based on temporal convolutions, that is domain-agnostic and has no particular strategy or algorithm encoded into it."  
>	"TCML architectures are nothing more than a deep stack of convolutional layers, making them simple, generic, and versatile, and the causal structure allows them to process sequential data in a sophisticated manner. RNNs also have these properties, but traditional architectures can only propagate information through time via their hidden state, and so there are fewer paths for information to flow from past to present. TCMLs do a better job of preserving the temporal structure of the input sequence; the convolutional structure offers more direct, high-bandwidth access to past information, allowing them to perform more sophisticated computation on a fixed temporal segment."  
>	"TCML is closest in spirit to [Santoro et al.](http://arxiv.org/abs/1605.06065); however, our experiments indicate that TCML outperforms such traditional RNN architectures. We can view the TCML architecture as a flavor of RNN that can remember information through the activations of the network rather than through an explicit memory module. Because of its convolutional structure, the TCML better preserves the temporal structure of the inputs it receives, at the expense of only being able to remember information for a fixed amount of time. However, by exponentially increasing the dilation factors of the higher convolutional layers, TCML architectures can tractably store information for long periods of time."  
  - `video` <https://youtu.be/TERCdog1ddE?t=49m39s> (Abbeel)

#### ["RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning"](http://arxiv.org/abs/1611.02779) Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel
  `RL^2` `learning learning algorithm`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#rl2-fast-reinforcement-learning-via-slow-reinforcement-learning-duan-schulman-chen-bartlett-sutskever-abbeel>

#### ["Learning to Reinforcement Learn"](http://arxiv.org/abs/1611.05763) Wang et al.
  `learning learning algorithm`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#learning-to-reinforcement-learn-wang-et-al>

----
#### ["Learned Optimizers that Scale and Generalize"](http://arxiv.org/abs/1703.04813) Wichrowska, Maheswaranathan, Hoffman, Colmenarejo, Denil, de Freitas, Sohl-Dickstein
  `learning optimization algorithm`
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/#t=2560> (de Freitas)
  - `code` <https://github.com/tensorflow/models/tree/master/research/learned_optimizer>

#### ["Learning to Learn without Gradient Descent by Gradient Descent"](https://arxiv.org/abs/1611.03824) Chen, Hoffman, Colmenarejo, Denil, Lillicrap, Botvinick, de Freitas
  `learning optimization algorithm`
>	"Differentiable neural computers as alternatives to parallel Bayesian optimization for hyperparameter tuning of other networks."  
>	"Proposes RNN optimizers that match performance of Bayesian optimization methods (e.g. Spearmint, SMAC, TPE) but are massively faster."  
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/#t=3155> (de Freitas)

#### ["Learning to Learn by Gradient Descent by Gradient Descent"](http://arxiv.org/abs/1606.04474) Andrychowicz, Denil, Gomez, Hoffman, Pfau, Schaul, Shillingford, de Freitas
  `learning optimization algorithm`
>	"Take some computation where you usually wouldn’t keep around intermediate states, such as a planning computation (say value iteration, where you only keep your most recent estimate of the value function) or stochastic gradient descent (where you only keep around your current best estimate of the parameters). Now keep around those intermediate states as well, perhaps reifying the unrolled computation in a neural net, and take gradients to optimize the entire computation with respect to some loss function."  
  - `video` <https://youtu.be/SAcHyzMdbXc?t=10m24s> (de Freitas)
  - `video` <https://youtu.be/x1kf4Zojtb0?t=1h4m53s> (de Freitas)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/#t=1669> (de Freitas)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1606.04474>
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
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1703.05175>
  - `code` <https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch>
  - `code` <https://github.com/abdulfatir/prototypical-networks-tensorflow>

#### ["Matching Networks for One Shot Learning"](http://arxiv.org/abs/1606.04080) Vinyals, Blundell, Lillicrap, Kavukcuoglu, Wierstra
>	"Given just a few, or even a single, examples of an unseen class, it is possible to attain high classification accuracy on ImageNet using Matching Networks. Matching Networks are trained in the same way as they are tested: by presenting a series of instantaneous one shot learning training tasks, where each instance of the training set is fed into the network in parallel. Matching Networks are then trained to classify correctly over many different input training sets. The effect is to train a network that can classify on a novel data set without the need for a single step of gradient descent."  
>	"End-to-end trainable K-nearest neighbors which accepts support sets of images as input and maps them to desired labels. Attention LSTM takes into account all samples of subset when computing the pair-wise cosine distance between samples."  
  - `video` <https://youtu.be/QIcpGa-_bvA?t=31m41s> (Vinyals)
  - `video` <https://youtube.com/watch?v=Q8AtnbHOQ-4> (Ghosh)
  - `notes` <https://pbs.twimg.com/media/Cy7Eyh5WgAAZIw2.jpg:large>
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
  - `video` <https://youtube.com/watch?v=2CHdHmhPq5E> (Bartunov)
  - `video` <https://youtube.com/watch?v=XpIDCzwNe78> (Bartunov) ([slides](https://bayesgroup.github.io/bmml_sem/2016/bartunov-oneshot.pdf))
  - `code` <http://github.com/sbos/gmn>

#### ["Towards a Neural Statistician"](http://arxiv.org/abs/1606.02185) Edwards, Storkey
  - `video` <http://techtalks.tv/talks/neural-statistician/63048/> (Edwards)
  - `video` <https://youtube.com/watch?v=29t1qc7IWro> (Edwards)
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
#### ["Causal Reasoning from Meta-reinforcement Learning"](https://arxiv.org/abs/1901.08162) Dasgupta et al.
  `causal learning`
  - <https://github.com/brylevkirill/notes/blob/master/Causal%20Inference.md#causal-reasoning-from-meta-reinforcement-learning-dasgupta-et-al>

#### ["Independently Controllable Features"](https://arxiv.org/abs/1708.01289) Thomas, Pondard, Bengio, Sarfati, Beaudoin, Meurs, Pineau, Precup, Bengio
  `causal learning`
>	"It has been postulated that a good representation is one that disentangles the underlying explanatory factors of variation. However, it remains an open question what kind of training framework could potentially achieve that. Whereas most previous work focuses on the static setting (e.g. with images), we postulate that some of the causal factors could be discovered if the learner is allowed to interact with its environment. The agent can experiment with different actions and observe their effects. We hypothesize that some of these factors correspond to aspects of the environment which are independently controllable, i.e., that there exists a policy and a learnable feature for each such aspect of the environment, such that this policy can yield changes in that feature with minimal changes to other features that explain the statistical variations in the observed data."  
>	"In interactive environments, the temporal dependency between successive observations creates a new opportunity to notice causal structure in data which may not be apparent using only observational studies. In reinforcement learning, several approaches explore mechanisms that push the internal representations of learned models to be “good” in the sense that they provide better control, and control is a particularly important causal relationship between an agent and elements of its environment."  
>	"We propose and explore a more direct mechanism for representation learning, which explicitly links an agent’s control over its environment with its internal feature representations. Specifically, we hypothesize that some of the factors explaining variations in the data correspond to aspects of the world that can be controlled by the agent. For example, an object that could be pushed around or picked up independently of others is an independently controllable aspect of the environment. Our approach therefore aims to jointly discover a set of features (functions of the environment state) and policies (which change the state) such that each policy controls the associated feature while leaving the other features unchanged as much as possible."  
>	"Assume that there are factors of variation underlying the observations coming from an interactive environment that are independently controllable. That is, a controllable factor of variation is one for which there exists a policy which will modify that factor only, and not the others. For example, the object associated with a set of pixels could be acted on independently from other objects, which would explain variations in its pose and scale when we move it around while leaving the others generally unchanged. The object position in this case is a factor of variation. What poses a challenge for discovering and mapping such factors into computed features is the fact that the factors are not explicitly observed. Our goal is for the agent to autonomously discover such factors – which we call independently controllable features – along with policies that control them. While these may seem like strong assumptions about the nature of the environment, we argue that these assumptions are similar to regularizers, and are meant to make a difficult learning problem (that of learning good representations which disentangle underlying factors) better constrained."  
>	"  
>	- Expand to sequences of actions and use them to define options  
>	- Notion of objects and attributes naturally falls out  
>	- Extension to non-static set of objects: types & instances  
>	- Objects are groups of controllable features: by whom? Agents  
>	- Factors controlled by other agents; mirror neurons  
>	- Because the set of objects may be unbounded, we need to learn to represent policies themselves, and the definition of an object is bound to the policies associated with it (for using it and changing its attributes)"  
  - `video` <https://youtu.be/Yr1mOzC93xs?t=23m13s> (Bengio)
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
#### ["Towards a Definition of Disentangled Representations"](https://arxiv.org/abs/1812.02230) Higgins, Amos, Pfau, Racaniere, Matthey, Rezende, Lerchner
  `concept learning`
>	"How can intelligent agents solve a diverse set of tasks in a data-efficient manner? The disentangled representation learning approach posits that such an agent would benefit from separating out (disentangling) the underlying structure of the world into disjoint parts of its representation. However, there is no generally agreed-upon definition of disentangling, not least because it is unclear how to formalise the notion of world structure beyond toy datasets with a known ground truth generative process. Here we propose that a principled solution to characterising disentangled representations can be found by focusing on the transformation properties of the world. In particular, we suggest that those transformations that change only some properties of the underlying world state, while leaving all other properties invariant, are what gives exploitable structure to any kind of data. Similar ideas have already been successfully applied in physics, where the study of symmetry transformations has revolutionised the understanding of the world structure. By connecting symmetry transformations to vector representations using the formalism of group and representation theory we arrive at the first formal definition of disentangled representations. Our new definition is in agreement with many of the current intuitions about disentangling, while also providing principled resolutions to a number of previous points of contention. While this work focuses on formally defining disentangling – as opposed to solving the learning problem – we believe that the shift in perspective to studying data transformations can stimulate the development of better representation learning algorithms."  

#### ["Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"](https://arxiv.org/abs/1811.12359) Locatello, Bauer, Lucic, Gelly, Scholkopf, Bachem
  `concept learning`
>	"The key assumption is that real-world data is generated by a few explanatory factors of variation and that these factors can be recovered by unsupervised learning algorithms. A large number of unsupervised learning approaches based on auto-encoding and quantitative evaluation metrics of disentanglement have been proposed; yet, the efficacy of the proposed approaches and utility of proposed notions of disentanglement has not been challenged in prior work. In this paper, we provide a sober look on recent progress in the field and challenge some common assumptions."  
>	"We first theoretically show that the unsupervised learning of disentangled representations is fundamentally impossible without inductive biases on both the models and the data. Then, we train more than 12 000 models covering the six most prominent methods, and evaluate them across six disentanglement metrics in a reproducible large-scale experimental study on seven different data sets. On the positive side, we observe that different methods successfully enforce properties “encouraged” by the corresponding losses. On the negative side, we observe that in our study (1) “good” hyperparameters seemingly cannot be identified without access to ground-truth labels, (2) good hyperparameters neither transfer across data sets nor across disentanglement metrics, and (3) that increased disentanglement does not seem to lead to a decreased sample complexity of learning for downstream tasks."  
>	"We theoretically prove that (perhaps unsurprisingly) the unsupervised learning of disentangled representations is fundamentally impossible without inductive biases both on the considered learning approaches and the data sets."  
>	"We investigate current approaches and their inductive biases in a reproducible large-scale experimental study with a sound experimental protocol for unsupervised disentanglement learning. We implement from scratch six recent unsupervised disentanglement learning methods as well as six disentanglement measures and train more than 12 000 models on seven data sets."  
>	"We evaluate our experimental results and challenge many common assumptions in unsupervised disentanglement learning: (i) While all considered methods prove effective at ensuring that the individual dimensions of the aggregated posterior (which is sampled) are not correlated, only one method also consistently ensures that the individual dimensions of the representation (which is taken to be the mean) are not correlated. (ii) We do not find any evidence that they can be used to reliably learn disentangled representations in an unsupervised manner as hyper parameters seem to matter more than the model and “good” hyperparameters seemingly cannot be identified without access to ground-truth labels. Similarly, we observe that good hyperparameters neither transfer across data sets nor across disentanglement metrics. (iii) For the considered models and data sets, we cannot validate the assumption that disentanglement is useful for downstream tasks, for example through a decreased sample complexity of learning."  
>	"Based on these empirical evidence, we suggest three critical areas of further research: (i) The role of inductive biases and implicit and explicit supervision should be made explicit: unsupervised model selection persists as a key question. (ii) The concrete practical benefits of enforcing a specific notion of disentanglement of the learned representations should be demonstrated. (iii) Experiments should be conducted in a reproducible experimental setup on data sets of varying degrees of difficulty."  
  - `post` <https://ai.googleblog.com/2019/04/evaluating-unsupervised-learning-of.html>
  - `video` <https://slideslive.com/38915874/challenging-common-assumptions-in-the-unsupervised-learning-of-disentangled-representations> (Locatello)
  - `video` <https://youtube.com/watch?v=WYrvh50yu6s> (Kilcher)
  - `code` <https://github.com/google-research/disentanglement_lib>

#### ["Neural Scene Representation and Rendering"](https://deepmind.com/documents/211/Neural_Scene_Representation_and_Rendering_preprint.pdf) Eslami et al.
  `concept learning` `GQN`
>	"Generative Query Network, a framework within which machines learn to represent scenes using only their own sensors. The GQN takes as input images of a scene taken from different viewpoints, constructs an internal representation, and uses this representation to predict the appearance of that scene from previously unobserved viewpoints. The GQN demonstrates representation learning without human labels or domain knowledge, paving the way toward machines that autonomously learn to understand the world around them."  
>	"Classical neural approaches to this learning problem - e.g., autoencoding and density models - are required to capture only the distribution of observed images, and there is no explicit mechanism to encourage learning of how different views of the same 3D scene relate to one another. The expectation is that statistical compression principles will be sufficient to enable networks to discover the 3D structure of the environment; however, in practice, they fall short of achieving this kind of meaningful representation and instead focus on regularities of colors and patches in the image space."  
  - `video` <https://youtube.com/watch?v=G-kWNQJ4idw> (demo)
  - `video` <https://youtube.com/watch?v=IVSZnTknyqw> (demo)
  - `video` <https://youtube.com/watch?v=XJnuEO59XfQ> (Chen)
  - `post` <https://deepmind.com/blog/neural-scene-representation-and-rendering>
  - `code` <https://github.com/wohlert/generative-query-network-pytorch>
  - `code` <https://github.com/ogroth/tf-gqn>
  - `paper` <http://science.sciencemag.org/content/sci/360/6394/1204.full.pdf>

#### ["Relational Neural Expectation Maximization: Unsupervised Discovery of Objects and their Interactions"](https://arxiv.org/abs/1802.10353) Steenkiste, Chang, Greff, Schmidhuber
  `concept learning`
>	"A novel method that learns to discover objects and model their physical interactions from raw visual images in a purely unsupervised fashion. It incorporates prior knowledge about the compositional nature of human perception to factor interactions between object-pairs and learn efficiently."  
  - <https://sites.google.com/view/r-nem-gifs>
  - `video` <https://youtu.be/IjkNnu8CCnY?t=30m55s> (Chang)

#### ["Neural Expectation Maximization"](https://arxiv.org/abs/1708.03498) Greff, Steenkiste, Schmidhuber
  `concept learning`
>	"differentiable clustering method that simultaneously learns how to group and represent individual entities"  
  - `code` <https://github.com/sjoerdvansteenkiste/Neural-EM>

#### ["Generative Models of Visually Grounded Imagination"](https://arxiv.org/abs/1705.10762) Vedantam, Fischer, Huang, Murphy
  `concept learning`
>	"Consider how easy it is for people to imagine what a "purple hippo" would look like, even though they do not exist. If we instead said "purple hippo with wings", they could just as easily create a different internal mental representation, to represent this more specific concept. To assess whether the person has correctly understood the concept, we can ask them to draw a few sketches, to illustrate their thoughts. We call the ability to map text descriptions of concepts to latent representations and then to images (or vice versa) visually grounded semantic imagination. We propose a latent variable model for images and attributes, based on variational auto-encoders, which can perform this task. Our method uses a novel training objective, and a novel product-of-experts inference network, which can handle partially specified (abstract) concepts in a principled and efficient way."  
  - `video` <https://youtu.be/CoXE5DhTX-A?t=35m46s> (Murphy)
  - `video` <https://youtu.be/IyP1pxgM_eE?t=1h5m14s> (Murphy)
  - `code` <https://github.com/google/joint_vae>

#### ["Understanding Disentangling in β-VAE"](https://arxiv.org/abs/1804.03599) Burgess et al.
  `concept learning` `β-VAE`
>	"We present new intuitions and theoretical assessments of the emergence of disentangled representation in variational autoencoders. Taking a rate-distortion theory perspective, we show the circumstances under which representations aligned with the underlying generative factors of variation of data emerge when optimising the modified ELBO bound in β-VAE, as training progresses. From these insights, we propose a modification to the training regime of β-VAE, that progressively increases the information capacity of the latent code during training. This modification facilitates the robust learning of disentangled representations in β-VAE, without the previous trade-off in reconstruction accuracy."  
  - `post` <https://towardsdatascience.com/what-a-disentangled-net-we-weave-representation-learning-in-vaes-pt-1-9e5dbc205bd1>

#### ["SCAN: Learning Abstract Hierarchical Compositional Visual Concepts"](https://arxiv.org/abs/1707.03389) Higgins, Sonnerat, Matthey, Pal, Burgess, Botvinick, Hassabis, Lerchner
  `concept learning` `β-VAE`
>	"We first use the previously published beta-VAE (Higgins et al., 2017a) architecture to learn a disentangled representation of the latent structure of the visual world, before training SCAN to extract abstract concepts grounded in such disentangled visual primitives through fast symbol association."  
  - `post` <https://deepmind.com/blog/imagine-creating-new-visual-concepts-recombining-familiar-ones/>
  - `video` <https://youtu.be/XNGo9xqpgMo?t=18m43s> (Higgins)

#### ["beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"](http://openreview.net/forum?id=Sy2fzU9gl) Higgins, Matthey, Pal, Burgess, Glorot, Botvinick, Mohamed, Lerchner
  `concept learning` `β-VAE`
>	"This paper proposes a modification of the variational ELBO in encourage 'disentangled' representations, and proposes a measure of disentanglement."  
>	"Beta-VAE is a VAE with beta coefficient in KL divergence term where beta=1 is exactly same formulation of vanilla VAE. By increasing beta, the weighted factor forces model to learn more disentangled representation than VAE. The authors also proposed disentanglement metric by training a simple classifier with low capacity and use it’s prediction accuracy. But the metric can be only calculated in simulator (ground truth generator) setting where we can control independent factors to generate different samples with controlled property."  
  - <http://tinyurl.com/jgbyzke> (demo)
  - `video` <https://youtu.be/XNGo9xqpgMo?t=10m8s> (Higgins)
  - `video` <https://youtu.be/Wgvcxd98tvU?t=27m17s> (Achille)
  - `post` <https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html#beta-vae>

#### ["Early Visual Concept Learning with Unsupervised Deep Learning"](http://arxiv.org/abs/1606.05579) Higgins, Matthey, Glorot, Pal, Uria, Blundell, Mohamed, Lerchner
  `concept learning` `β-VAE`
  - `video` <https://cds.cern.ch/record/2302480> (52:29) (Rezende)
  - `code` <https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results>

#### ["Towards Conceptual Compression"](http://arxiv.org/abs/1604.08772) Gregor, Besse, Rezende, Danihelka, Wierstra
  `concept learning`
  - `video` <https://cds.cern.ch/record/2302480> (49:02) (55:12) (Rezende)
  - `poster` <https://pbs.twimg.com/media/Cy3pYfWWIAA_C9h.jpg:large>

#### ["Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"](http://arxiv.org/abs/1603.08575) Eslami, Heess, Weber, Tassa, Szepesvari, Kavukcuoglu, Hinton
  `concept learning`
>	"Learning to perform inference in partially specified 2D models (variable-sized variational auto-encoders) and fully specified 3D models (probabilistic renderers)."  
>	"Models learn to identify multiple objects - counting, locating and classifying the elements of a scene - without any supervision, e.g., decomposing 3D images with various numbers of objects in a single forward pass of a neural network."  
  - `video` <https://youtube.com/watch?v=4tc84kKdpY4> (demo)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/EslamiHWTKH16>
  - `post` <http://akosiorek.github.io/ml/2017/09/03/implementing-air.html>
  - `post` <http://pyro.ai/examples/air.html>
  - `notes` <http://tuananhle.co.uk/notes/air.html>
  - `code` <https://github.com/akosiorek/attend_infer_repeat>
  - `code` <https://github.com/uber/pyro/tree/dev/examples/air>

----
#### ["Synthesizing Programs for Images using Reinforced Adversarial Learning"](https://arxiv.org/abs/1804.01118) Ganin, Kulkarni, Babuschkin, Eslami, Vinyals
  `learning disentangled representation` `SPIRAL`
>	"Adversarially trained agent that generates a program which is executed by a graphics engine to interpret and sample images. The goal of this agent is to fool a discriminator network that distinguishes between real and rendered data, trained with a distributed reinforcement learning setup without any supervision. To the best of our knowledge, this is the first demonstration of an end-to-end, unsupervised and adversarial inverse graphics agent on challenging real world and synthetic 3D datasets."  
>	"Trust discriminator to guide learning by using its score as reward for IMPALA agent instead of propagating gradients as in typical GAN."  
>	"Goal is to achieve better generalisation through use of tools in grounded environment."  
>	"Unsupervised learning is not only about predicting inputs - SPIRAL learns the rewards through which learning happens and learns the policy to generate a program that generates inputs."  
  - `post` <https://deepmind.com/blog/learning-to-generate-images>
  - `video` <https://youtu.be/iSyvwAwa7vk> (demo)
  - `video` <https://youtube.com/watch?v=kkihoMMpBb0> (Vinyals)
  - `video` <https://facebook.com/iclr.cc/videos/2125495797479475?t=2069> (Kavukcuoglu)

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
#### ["Unsupervised Learning by Predicting Noise"](https://arxiv.org/abs/1704.05310) Bojanowski, Joulin
  `learning embedding`
>	"The authors give a nice analogy: it's a SOM, but instead of mapping a latent vector to each input vector, the convolutional filters are learned in order to map each input vector to a fixed latent vector. In more words: each image is assigned a unique random latent vector as the label, and the mapping from image to label is taught in a supervised manner. Every few epochs, the label assignments are adjusted (but only within batches due to computational cost), so that an image might be assigned a different latent vector label which it is already close to in 'feature space'."
  - `post` <http://inference.vc/unsupervised-learning-by-predicting-noise-an-information-maximization-view-2/>

#### ["Poincare Embeddings for Learning Hierarchical Representations"](https://arxiv.org/abs/1705.08039) Nickel, Kiela
  `learning embedding`
  - `video` <https://cds.cern.ch/record/2306315> (Nickel)
  - `video` <https://facebook.com/nipsfoundation/videos/1553634558061111?t=3781> (Nickel)
  - `notes` <https://medium.com/towards-data-science/facebook-research-just-published-an-awesome-paper-on-learning-hierarchical-representations-34e3d829ede7>
  - `notes` <https://medium.com/@hol_io/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795>
  - `post` <https://rare-technologies.com/implementing-poincare-embeddings/>
  - `code` <https://github.com/facebookresearch/poincare-embeddings>
  - `code` <https://github.com/TatsuyaShirakawa/poincare-embedding>
  - `code` <https://github.com/lateral/poincare-embeddings>



---
### generative models

  - [flow models](#generative-models---flow-models)
  - [generative adversarial networks](#generative-models---generative-adversarial-networks)
  - [variational autoencoders](#generative-models---variational-autoencoders)
  - [autoregressive models](#generative-models---autoregressive-models)

----
#### ["Do Deep Generative Models Know What They Don't Know?"](https://arxiv.org/abs/1810.09136) Nalisnick, Matsukawa, Teh, Gorur, Lakshminarayanan
  `evaluation` `out-of-distribution`
>	"We show that deep generative models can assign higher likelihood to out-of-distribution inputs than the training data."  
>	"A neural network deployed in the wild may be asked to make predictions for inputs that were drawn from a different distribution than that of the training data. A plethora of work has demonstrated that it is easy to find or synthesize inputs for which a neural network is highly confident yet wrong. Generative models are widely viewed to be robust to such mistaken confidence as modeling the density of the input features can be used to detect novel, out-of-distribution inputs. In this paper we challenge this assumption. We find that the model density from flow-based models, VAEs and PixelCNN cannot distinguish images of common objects such as dogs, trucks, and horses (i.e. CIFAR-10) from those of house numbers (i.e. SVHN), assigning a higher likelihood to the latter when the model is trained on the former. We focus our analysis on flow-based generative models in particular since they are trained and evaluated via the exact marginal likelihood. We find such behavior persists even when we restrict the flow models to constant-volume transformations. These transformations admit some theoretical analysis, and we show that the difference in likelihoods can be explained by the location and variances of the data and the model curvature, which shows that such behavior is more general and not just restricted to the pairs of datasets used in our experiments. Our results caution against using the density estimates from deep generative models to identify inputs similar to the training distribution, until their behavior on out-of-distribution inputs is better understood."  
>	"We have shown that comparing likelihoods alone cannot identify the training set or inputs like it. Moreover, our analysis shows that the SVHN vs CIFAR-10 problem we report would persist for any constant-volume flow no matter the parameter settings nor the choice of latent density (as long as it is log-concave). The models seem to capture low-level statistics rather than high-level semantics. While we cannot conclude that this is necessarily a pathology in deep generative models, it does suggest they need to be further improved. It could be a problem that plagues any generative model, no matter how high its capacity."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1810.09136>

#### ["Generative Ensembles for Robust Anomaly Detection"](https://arxiv.org/abs/1810.01392) Choi, Jang
  `evaluation` `out-of-distribution`
>	"Deep generative models are capable of learning probability distributions over large, high-dimensional datasets such as images, video and natural language. Generative models trained on samples from p(x) ought to assign low likelihoods to out-of-distribution samples from q(x), making them suitable for anomaly detection applications. We show that in practice, likelihood models are themselves susceptible to OoD errors, and even assign large likelihoods to images from other natural datasets. To mitigate these issues, we propose Generative Ensembles, a model-independent technique for OoD detection that combines density-based anomaly detection with uncertainty estimation. Our method outperforms ODIN and VIB baselines on image datasets, and achieves comparable performance to a classification model on the Kaggle Credit Fraud dataset."  

#### ["Pros and Cons of GAN Evaluation Measures"](https://arxiv.org/abs/1802.03446) Borji
  `evaluation`

#### ["Flow-GAN: Bridging Implicit and Prescribed Learning in Generative Models"](https://arxiv.org/abs/1705.08868) Grover, Dhar, Ermon
  `evaluation`
>	"generative adversarial network which allows for tractable likelihood evaluation"  
>	"Since it can be trained both adversarially (like a GAN) and in terms of MLE (like a flow model), we can quantitatively evaluate the trade-offs involved. In particular, we also consider a hybrid objective function which involves both types of losses."  
>	"The availability of quantitative metrics allow us to compare to simple baselines which essentially “remember” the training data. Our final results show that naive Gaussian Mixture Models outperforms plain WGAN on both samples quality and log-likelihood for both MNIST and CIFAR-10 which we hope will lead to new directions for both implicit and prescribed learning in generative models."  
  - `post` <https://distill.pub/2019/gan-open-problems/#tradeoffs>

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
  - `post` <https://colinraffel.com/blog/gans-and-divergence-minimization.html>
  - `post` <https://distill.pub/2019/gan-open-problems/#tradeoffs>
  - `slides` <http://www.gatsby.ucl.ac.uk/~balaji/Understanding-GANs.pdf>

#### ["On the Quantitative Analysis of Decoder-based Generative Models"](http://arxiv.org/abs/1611.04273) Wu, Burda, Salakhutdinov, Grosse
  `evaluation`
>	"We propose to use Annealed Importance Sampling for evaluating log-likelihoods for decoder-based models and validate its accuracy using bidirectional Monte Carlo. Using this technique, we analyze the performance of decoder-based models, the effectiveness of existing log-likelihood estimators, the degree of overfitting, and the degree to which these models miss important modes of the data distribution."  
>	"This paper introduces Annealed Importance Sampling to compute tighter lower bounds and upper bounds for any generative model (with a decoder)."  
>	"GAN training obtains much worse likelihood than the MLE."  
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
>	"Authors emphasize that an improvement of log-likelihood does not necessarily translate to higher perceptual quality, and that the KL loss is more likely to produce atypical samples than some other training criteria."  
  - `video` <http://videolectures.net/iclr2016_theis_generative_models/> (Theis)

----
#### ["Autoregressive Quantile Networks for Generative Modeling"](http://arxiv.org/abs/1806.05575) Ostrovski, Dabney, Munos
  `alternative to KL divergence` `AIQN` `PixelIQN` `AIQN-VAE`
>	"Most existing generative models for images belong to one of two classes. The first are likelihood-based models, trained with an element-wise KL reconstruction loss, which, while perceptually meaningless, provides robust optimization properties and high sample diversity. The second are GANs, trained based on a discriminator loss, typically better aligned with a perceptual metric and enabling the generator to produce realistic, globally consistent samples. Their advantages come at the cost of a harder optimization problem, high parameter sensitivity, and most importantly, a tendency to collapse modes of the data distribution."  
>	"AIQNs are a new, fundamentally different, technique for generative modeling. By using a quantile regression loss instead of KL divergence, they combine some of the best properties of the two model classes. By their nature, they preserve modes of the learned distribution, while producing perceptually appealing high-quality samples. The inevitable approximation trade-offs a generative model makes when constrained by capacity or insufficient training can vary significantly depending on the loss used. We argue that the proposed quantile regression loss aligns more effectively with a given metric and therefore makes subjectively more advantageous trade-offs."  
>	"IQN, computationally cheap and technically simple, can be readily applied to existing architectures, PixelCNN and VAE, improving robustness and sampling quality of the underlying model."  
>	"PixelIQN model achieves a performance level comparable to that of the fully trained PixelCNN with only about one third the number of training updates (and about one third of the wall-clock time)."  
>	"PixelIQN, due to the continuous nature of the quantile function, can be used to learn distributions over lower-dimensional, latent spaces, such as those produced by an autoencoder, variational or otherwise. Specifically, we use a standard VAE, but simultaneously train a small AIQN to model the training distribution over latent codes. For sampling, we then generate samples of the latent distribution using AIQN instead of the VAE prior. This approach works well for two reasons. First, even a thoroughly trained VAE does not produce an encoder that fully matches the Gaussian prior. Generaly, the data distribution exists on a non-Gaussian manifold in the latent space, despite the use of variational training. Second, unlike existing methods, AIQN learns to approximate the full continuous-valued distribution without discretizing values or making prior assumptions about the value range or underlying distribution."  
>	"A common perspective in generative modeling is that the choice of model should encode existing metric assumptions about the domain, combined with a generic likelihood-focused loss such as the KL divergence. Under this view, the KL’s general applicability and robust optimization properties make it a natural choice, and most of the methods attempt to, at least indirectly, minimize a version of the KL. On the other hand, as every model inevitably makes trade-offs when constrained by capacity or limited training, it is desirable for its optimization goal to incentivize trade-offs prioritizing approximately correct solutions, when the data space is endowed with a metric supporting a meaningful (albeit potentially subjective) notion of approximation. It has been argued that the KL may not always be appropriate from this perspective, by making sub-optimal trade-offs between likelihood and similarity."  
>	"Many limitations of existing models can be traced back to the use of KL, and the resulting trade-offs in approximate solutions it implies. For instance, its use appears to play a central role in one of the primary failure modes of VAEs, that of blurry samples. Zhao et al. (2017) argue that the Gaussian posterior pθ(x|z) implies an overly simple model, which, when unable to perfectly fit the data, is forced to average (thus creating blur), and is not incentivized by the KL towards an alternative notion of approximate solution. Theis et al. (2015) emphasized that an improvement of log-likelihood does not necessarily translate to higher perceptual quality, and that the KL loss is more likely to produce atypical samples than some other training criteria. We offer an alternative perspective: a good model should encode assumptions about the data distribution, whereas a good loss should encode the notion of similarity, that is, the underlying metric on the data space. From this point of view, the KL corresponds to an actual absence of explicit underlying metric, with complete focus on probability."  
>	"Wasserstein GAN reposes the two-player game as the estimation of the gradient of the 1-Wasserstein distance between the data and generator distributions. It reframes this in terms of the dual form of the 1-Wasserstein, with the critic estimating a function f which maximally separates the two distributions. It still faces limitations when the critic solution is approximate, i.e. when f* is not found before each update. In this case, due to insufficient training of the critic or limitations of the function approximator, the gradient direction produced can be arbitrarily bad. We are left with the question of how to minimize a distribution loss respecting an underlying metric. Recent work in distributional reinforcement learning has proposed the use of quantile regression as a method for minimizing the 1-Wasserstein in the univariate case when approximating using a mixture of Dirac functions."  
  - `video` <https://vimeo.com/287766947> (Dabney)

#### ["From Optimal Transport to Generative Modeling: the VEGAN Cookbook"](https://arxiv.org/abs/1705.07642) Bousquet, Gelly, Tolstikhin, Simon-Gabriel, Scholkopf
  `alternative to KL divergence` `unifying GANs and VAEs`
>	"The Optimal Transport cost is a way to measure a distance between probability distributions and provides a much weaker topology than many others, including f-divergences associated with the original GAN algorithms. This is particularly important in applications, where data is usually supported on low dimensional manifolds in the input space X. As a result, stronger notions of distances (such as f-divergences, which capture the density ratio between distributions) often max out, providing no useful gradients for training. In contrast, the Optimal Transport behave nicer and may lead to a more stable training."  
>	"We show that the Optimal Transport problem can be equivalently written in terms of probabilistic encoders, which are constrained to match the posterior and prior distributions over the latent space. When relaxed, this constrained optimization problem leads to a penalized optimal transport (POT) objective, which can be efficiently minimized using stochastic gradient descent by sampling from Px and Pg."  
>	"We show that POT for the 2-Wasserstein distance coincides with the objective heuristically employed in adversarial auto-encoders, which provides the first theoretical justification for AAEs known to the authors."  
>	"We also compare POT to other popular techniques like variational auto-encoders. Our theoretical results include (a) a better understanding of the commonly observed blurriness of images generated by VAEs, and (b) establishing duality between Wasserstein GAN and POT for the 1-Wasserstein distance."  
>	"WGAN and VAE are respectively dual and primal approximations with deep-nets of an optimal transport estimator (minimum Kantorovich distance estimator). The approximation is so crude however that OT theory is probably mostly of cosmetic use."  
>	"The optimal transport metrics Wc, for underlying metric c(x,x0), and in particular the p-Wasserstein distance, when c is an Lp metric, have frequently been proposed as being well-suited replacements to KL. Briefly, the advantages are (1) avoidance of mode collapse (no need to choose between spreading over modes or collapsing to a single mode as in KL), and (2) the ability to trade off errors and incentivize approximations that respect the underlying metric."  
  - `paper` ["GAN and VAE from an Optimal Transport Point of View"](https://arxiv.org/abs/1706.01807) by Genevay, Peyre, Cuturi
  - `paper` ["On Minimum Kantorovich Distance Estimators"](https://www.sciencedirect.com/science/article/pii/S0167715206000381) by Bassetti, Bodini, Regazzini

----
#### ["Variational Approaches for Auto-Encoding Generative Adversarial Networks"](https://arxiv.org/abs/1706.04987) Rosca, Lakshminarayanan, Warde-Farley, Mohamed
  `unifying GANs and VAEs` `α-GAN`
>	"Attempt to sample from the true latent distribution of a VAE-like latent variable model with sampling distribution trained using a GAN."  
>	"Use more general likelihoods than in VAE. Fight intractability using discriminators."  
  - `video` <https://youtu.be/ZHucm52V3Zw?t=5m5s> (Umnov)
  - `video` <https://youtu.be/jAI3rBI6poU?t=1h1m33s> (Ulyanov) `in russian`
  - `slides` <http://elarosca.net/slides/iccv_autoencoder_gans.pdf>
  - `slides` <http://www.gatsby.ucl.ac.uk/~balaji/Understanding-GANs.pdf>
  - `notes` <https://medium.com/@g789872001darren/paper-note-variational-approaches-for-auto-encoding-generative-adversarial-networks-fefc3b3841ff>
  - `code` <https://github.com/victor-shepardson/alpha-GAN>

#### ["On Unifying Deep Generative Models"](https://arxiv.org/abs/1706.00550) Hu, Yang, Salakhutdinov, Xing
  `unifying GANs and VAEs`
>	"We show that GANs and VAEs are essentially minimizing KL divergences of respective posterior and inference distributions with opposite directions, extending the two learning phases of classic wake-sleep algorithm, respectively. The unified view provides a powerful tool to analyze a diverse set of existing model variants, and enables to exchange ideas across research lines in a principled way. For example, we transfer the importance weighting method in VAE literatures for improved GAN learning, and enhance VAEs with an adversarial mechanism for leveraging generated samples."  

#### ["Hierarchical Implicit Models and Likelihood-Free Variational Inference"](http://arxiv.org/abs/1702.08896) Tran, Ranganath, Blei
  `unifying GANs and VAEs`
>	"We introduce hierarchical implicit models. HIMs combine the idea of implicit densities with hierarchical Bayesian modeling, thereby defining models via simulators of data with rich hidden structure."  
>	"We develop likelihood-free variational inference, a scalable variational inference algorithm for HIMs. Key to LFVI is specifying a variational family that is also implicit. This matches the model's flexibility and allows for accurate approximation of the posterior."  
  - `notes` <http://dustintran.com/papers/TranRanganathBlei2017_poster.pdf>
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
  - `slides` <http://www.gatsby.ucl.ac.uk/~balaji/Understanding-GANs.pdf>

#### ["Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks"](http://arxiv.org/abs/1701.04722) Mescheder, Nowozin, Geiger
  `unifying GANs and VAEs`
>	"Real NVP can be used as an encoder in order to measure the gap between the unbiased KL estimate log q(z|x) - log p(z) and its approximation from GAN. We show that Adversarial Variational Bayes underestimates the KL divergence."  
  - `video` <https://youtu.be/y7pUN2t5LrA?t=14m19s> (Nowozin)
  - `video` <https://youtu.be/xFCuXE1Nb8w?t=26m55s> (Nowozin)
  - `video` <https://youtu.be/m80Vp-jz-Io?t=1h28m34s> (Tolstikhin)
  - `post` <http://inference.vc/variational-inference-with-implicit-models-part-ii-amortised-inference-2/>
  - `post` <https://chrisorm.github.io/AVB-pyt.html>
  - `notes` <https://casmls.github.io/general/2017/02/23/modified-gans.html>
  - `code` <https://github.com/wiseodd/generative-models/tree/master/VAE/adversarial_vb>
  - `code` <https://gist.github.com/poolio/b71eb943d6537d01f46e7b20e9225149>



---
### generative models - flow models

[**interesting recent papers - generative models**](#generative-models)

----
#### ["Glow: Generative Flow with Invertible 1x1 Convolutions"](https://arxiv.org/abs/1807.03039) Kingma, Dhariwal
  `Glow`
>	"We demonstrate that a generative model optimized towards the plain log-likelihood objective is capable of efficient realistic-looking synthesis and manipulation of large images."  
  - `post` <https://blog.openai.com/glow>
  - `video` <https://youtube.com/watch?v=exJZOC3ZceA> (demo)
  - `post` <https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#glow>
  - `code` <https://github.com/openai/glow>
  - `code` <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/glow_ops.py>
  - `code` <https://github.com/ikostrikov/pytorch-flows>

#### ["Neural Autoregressive Flows"](https://arxiv.org/abs/1804.00779) Huang, Krueger, Lacoste, Courville
  `NAF`
>	"NAF unifies and generalizes MAF and IAF, replacing the (conditionally) affine univariate transformations of MAF/IAF with a more general class of invertible univariate transformations expressed as monotonic neural networks. We demonstrate that NAFs are universal approximators for continuous probability distributions, and their greater expressivity allows them to better capture multimodal target distributions."  
  - `post` <https://medium.com/element-ai-research-lab/neural-autoregressive-flows-f164d6b8e462>
  - `post` <https://habr.com/company/yandex/blog/418421> `in russian`
  - `code` <https://github.com/CW-Huang/NAF>

#### ["Parallel WaveNet: Fast High-Fidelity Speech Synthesis"](https://arxiv.org/abs/1711.10433) Oord et al.
  `Parallel WaveNet`
>	"Inverse autoregressive flows represent a kind of dual formulation of deep autoregressive modelling, in which sampling can be performed in parallel, while the inference procedure required for likelihood estimation is sequential and slow. The goal of this paper is to marry the best features of both models: the efficient training of WaveNet and the efficient sampling of IAF networks. The bridge between them is a new form of neural network distillation, which we refer to as Probability Density Distillation, where a trained WaveNet model is used as a teacher for training feedforward IAF model with no significant difference in quality."  
>	"WaveNet: efficient training, slow sampling"  
>	"IAF: efficient sampling, slow inference"  
>	"Distribution Distillation combines the advantages of both types of flows. It trains one model, which closely resembles MAF, for density estimation. Its role is just to evaluate probability of a data point, given that data point. Once this model is trained, the authors instantiate a second model parametrised by IAF. Now, we can draw samples from IAF and evaluate their probability under the MAF. This allows us to compute Monte-Carlo approximation of the KL-divergence between the two probability distributions, which we can use as a training objective for IAF. This way, MAF acts as a teacher and IAF as a student. This clever application of both types of flows allowed to improve efficiency of the original WaveNet by the factor of 300."
  - <https://deepmind.com/blog/wavenet-launches-google-assistant/> (demo)
  - `post` <https://deepmind.com/blog/high-fidelity-speech-synthesis-wavenet/>
  - `video` <https://vimeo.com/287766925> (Oord)
  - `video` <https://facebook.com/iclr.cc/videos/2125495797479475?t=493> (Kavukcuoglu)
  - `video` <https://youtu.be/YyUXG-BfDbE?t=26m19s> (Andrews)

#### ["Masked Autoregressive Flow for Density Estimation"](https://arxiv.org/abs/1705.07057) Papamakarios, Pavlakou, Murray
  `MAF`
>	"We describe an approach for increasing the flexibility of an autoregressive model, based on modelling the random numbers that the model uses internally when generating data. By constructing a stack of autoregressive models, each modelling the random numbers of the next model in the stack, we obtain a type of normalizing flow suitable for density estimation. This type of flow is closely related to Inverse Autoregressive Flow and is a generalization of Real NVP."  
>	"MAF:  
>	- fast to calculate p(x)  
>	- slow to sample from  
>
>	Inverse Autoregressive Flow:  
>	- slow to calculate p(x)  
>	- fast to sample from  
>
>	Real NVP:  
>	- fast to calculate p(x)  
>	- fast to sample from  
>	- limited capacity vs MAF"  
  - `video` <https://vimeo.com/252105837> (Papamakarios)
  - `audio` <https://youtube.com/watch?v=315xKcYX-1w> (Papamakarios)
  - `post` <http://blog.evjang.com/2018/01/nf2.html>
  - `post` <http://akosiorek.github.io/ml/2018/04/03/norm_flows.html>
  - `post` <https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#masked-autoregressive-flow>
  - `code` <https://github.com/ikostrikov/pytorch-flows>
  - `code` <https://github.com/gpapamak/maf>

#### ["Density Estimation using Real NVP"](http://arxiv.org/abs/1605.08803) Dinh, Sohl-Dickstein, Bengio
  `Real NVP` `RNVP`
>	"Real-valued Non Volume Preserving transform:  
>	- one-pass and exact inference and sampling  
>	- explicit learning of a latent representation  
>	- tractable log-likelihood  
>	- coherent and sharp samples"  
  - <https://laurent-dinh.github.io/2016/07/12/real-nvp-visualization.html> (demo)
  - `video` <https://channel9.msdn.com/events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-1> (08:19) (Dinh)
  - `video` <https://periscope.tv/hugo_larochelle/1ypKdAVmbEpGW> (Dinh)
  - `video` <https://cds.cern.ch/record/2302480> (43:54) (Rezende)
  - `post` <http://blog.evjang.com/2018/01/nf2.html>
  - `post` <http://akosiorek.github.io/ml/2018/04/03/norm_flows.html>
  - `post` <https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#realnvp>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.08803>
  - `code` <https://github.com/tensorflow/models/tree/master/research/real_nvp>
  - `code` <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distributions/python/ops/bijectors/real_nvp.py>
  - `code` <https://github.com/ikostrikov/pytorch-flows>



---
### generative models - generative adversarial networks

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---generative-adversarial-networks)

[**interesting recent papers - generative models**](#generative-models)

----
#### ["What Are GANs Useful For?"](https://openreview.net/forum?id=HkwrqtlR-) Olmos, Hitaj, Gasti, Ateniese, Perez-Cruz
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
  - `video` <https://smartech.gatech.edu/handle/1853/59407> (Arora)

#### ["Approximation and Convergence Properties of Generative Adversarial Learning"](https://arxiv.org/abs/1705.08991) Liu, Bousquet, Chaudhuri
  `GAN theory`
>	"Two very basic questions on how well GANs can approximate the target distribution µ, even in the presence of a very large number of samples and perfect optimization, remain largely unanswered.  
>	The first relates to the role of the discriminator in the quality of the approximation. In practice, the discriminator is usually restricted to belong to some family, and it is not understood in what sense this restriction affects the distribution output by the generator.  
>	The second question relates to convergence; different variants of GANs have been proposed that involve different objective functions (to be optimized by the generator and the discriminator). However, it is not understood under what conditions minimizing the objective function leads to a good approximation of the target distribution. More precisely, does a sequence of distributions output by the generator that converges to the global minimum under the objective function always converge to the target distribution µ under some standard notion of distributional convergence?"  
>	"We first characterize a very general class of objective functions that we call adversarial divergences, and we show that they capture the objective functions used by a variety of existing procedures that include the original GAN, f-GAN, MMD-GAN, WGAN, improved WGAN, as well as a class of entropic regularized optimal transport problems. We then define the class of strict adversarial divergences – a subclass of adversarial divergences where the minimizer of the objective function is uniquely the target distribution."  
>	"We show that if the objective function is an adversarial divergence that obeys certain conditions, then using a restricted class of discriminators has the effect of matching generalized moments. A concrete consequence of this result is that in linear f-GANs, where the discriminator family is the set of all affine functions over a vector ψ of features maps, and the objective function is an f-GAN, the optimal distribution ν output by the GAN will satisfy Ex\~µ[ψ(x)] = Ex\~ν[ψ(x)] regardless of the specific f-divergence chosen in the objective function. Furthermore, we show that a neural network GAN is just a supremum of linear GANs, therefore has the same moment-matching effect."  
>	"We show that convergence in an adversarial divergence implies some standard notion of topological convergence. Particularly, we show that provided an objective function is a strict adversarial divergence, convergence to µ in the objective function implies weak convergence of the output distribution to µ. An additional consequence of this result is the observation that as the Wasserstein distance metrizes weak convergence of probability distributions, Wasserstein-GANs have the weakest objective functions in the class of strict adversarial divergences."  
>
>	"Authors worry about two important aspects of GAN convergence: what how good the generative distribution approximates the real distribution; and, when does this convergence takes place. For the first question the answer is the discriminator forces some kind of moment matching between the real and fake distributions. In order to get full representation of the density we will need that the discriminator grows with the data. For the second question, they show a week convergence result. This result is somewhat complementary to Arora et al. (2017), because it indicates that the discriminator complexity needs to grow indefinitely to achieve convergence. The question that remains to be answer is the rate of convergence, as the moments need to be matched for complicated distributions might require large data and complex discriminators. So in practice, we cannot tell if the generated distribution is close enough to the distribution we are interested in."  
>	"Today’s GAN’s results can be explained in the light of these theoretical results. First, we are clearly matching some moments that relate to visual quality of natural images with finite size deep neural networks, but they might not be deep enough to capture all relevant moments and hence they will not be able to match the distribution in these images."  

#### ["Do GANs Actually Learn the Distribution? An Empirical Study"](https://arxiv.org/abs/1706.08224) Arora, Zhang
  `GAN theory`
>	"On the positive side, we can show the existence of an equilibrium where generator succeeds in fooling the discriminator. On the  negative side, we show that in this equilibrium, generator produces a distribution of fairly low support."  
>	"A recent analysis raised doubts whether GANs actually learn the target distribution when discriminator has finite size. It showed that the training objective can approach its optimum value even if the generated distribution has very low support ---in other words, the training objective is unable to prevent mode collapse. The current note reports experiments suggesting that such problems are not merely theoretical. It presents empirical evidence that well-known GANs approaches do learn distributions of fairly low support, and thus presumably are not learning the target distribution. The main technical contribution is a new proposed test, based upon the famous birthday paradox, for estimating the support size of the generated distribution."  
  - `video` <https://youtube.com/watch?v=qStuhkIHE6c> (Arora)
  - `video` <https://smartech.gatech.edu/handle/1853/59407> (Arora)
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
#### ["The Relativistic Discriminator: A Key Element Missing from Standard GAN"](https://arxiv.org/abs/1807.00734) Jolicoeur-Martineau
  `GAN objective`
>	"In standard generative adversarial network, the discriminator estimates the probability that the input data is real. The generator is trained to increase the probability that fake data is real. We argue that it should also simultaneously decrease the probability that real data is real because 1) this would account for a priori knowledge that half of the data in the mini-batch is fake, 2) this would be observed with divergence minimization, and 3) in optimal settings, SGAN would be equivalent to integral probability metric (IPM) GANs."  
>	"We show that this property can be induced by using a relativistic discriminator which estimate the probability that the given real data is more realistic than a randomly sampled fake data. We also present a variant in which the discriminator estimate the probability that the given real data is more realistic than fake data, on average. We generalize both approaches to non-standard GAN loss functions and we refer to them respectively as Relativistic GANs (RGANs) and Relativistic average GANs (RaGANs). We show that IPM-based GANs are a subset of RGANs which use the identity function."  
>	"Empirically, we observe that 1) RGANs and RaGANs are significantly more stable and generate higher quality data samples than their non-relativistic counterparts, 2) Standard RaGAN with gradient penalty generate data of better quality than WGAN-GP while only requiring a single discriminator update per generator update (reducing the time taken for reaching the state-of-the-art by 400%), and 3) RaGANs are able to generate plausible high resolutions images (256x256) from a very small sample (N=2011), while GAN and LSGAN cannot; these images are of significantly better quality than the ones generated by WGAN-GP and SGAN with spectral normalization."  
>	"Instead of D(x) = activation (C(x)), use 1) D(x_real,x_fake) = activation (C(x_real)-C(x_fake)) or 2) D(x_real, x_fake) = activation (C(x_real)-E[C(x_fake)]) + activation (E[C(x_real)]-C(x_fake))."  
  - `post` <https://ajolicoeur.wordpress.com/relativisticgan>
  - `video` <https://youtu.be/m9USSDtUy40?t=28m38s> (Chavdarova)
  - `code` <https://github.com/AlexiaJM/RelativisticGAN>

#### ["A Geometric View of Optimal Transportation and Generative Model"](https://arxiv.org/abs/1710.05488) Lei et al.
  `GAN objective`
>	"We show the intrinsic relations between optimal transportation and convex geometry, especially the variational approach to solve Alexandrov problem: constructing a convex polytope with prescribed face normals and volumes."  
>	"By using the optimal transportation view of GAN model, we show that the discriminator computes the Kantorovich potential, the generator calculates the transportation map. For a large class of transportation costs, the Kantorovich potential can give the optimal transportation map by a close-form formula. Therefore, it is sufficient to solely optimize the discriminator. This shows the adversarial competition can be avoided, and the computational architecture can be simplified."  
>	"Preliminary experimental results show the geometric method outperforms WGAN for approximating probability measures with multiple clusters in low dimensional space."  

#### ["Optimizing the Latent Space of Generative Networks"](https://arxiv.org/abs/1707.05776) Bojanowski, Joulin, Lopez-Paz, Szlam
  `GAN objective`
>	"Are GANs successful because of adversarial training or the use of ConvNets? We show that a ConvNet generator trained with a simple reconstruction loss and learnable noise vectors leads many of the desirable properties of a GAN."  
>	"We introduce Generative Latent Optimization, a framework to train deep convolutional generators without using discriminators, thus avoiding the instability of adversarial optimization problems. Throughout a variety of experiments, we show that GLO enjoys many of the desirable properties of GANs: learning from large data, synthesizing visually-appealing samples, interpolating meaningfully between samples, and performing linear arithmetic with noise vectors."  
  - `video` <https://youtu.be/r7oSmy_AtZY?t=29m4s> (Szlam)
  - `post` <https://facebook.com/yann.lecun/posts/10154646915277143>
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/BojanowskiJLS17>

#### ["Bayesian GAN"](https://arxiv.org/abs/1705.09558) Saatchi, Wilson
  `GAN objective`
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#bayesian-gan-saatchi-wilson>

#### ["AdaGAN: Boosting Generative Models"](https://arxiv.org/abs/1701.02386) Tolstikhin, Gelly, Bousquet, Simon-Gabriel, Scholkopf
  `GAN objective`
>	"We call it Adaptive GAN, but we could actually use any other generator: a Gaussian mixture model, a VAE, a WGAN, or even an unrolled or mode-regularized GAN, which were both already specifically developed to tackle the missing mode problem. Thus, we do not aim at improving the original GAN or any other generative algorithm. We rather propose and analyse a meta-algorithm that can be used on top of any of them. This meta-algorithm is similar in spirit to AdaBoost in the sense that each iteration corresponds to learning a “weak” generative model (e.g., GAN) with respect to a re-weighted data distribution. The weights change over time to focus on the “hard” examples, i.e. those that the mixture has not been able to properly generate so far."  
>
>	"New components are added to the mixture until the original distribution is recovered and authors show exponential convergence to the underlying density. Our concern with the proof is its practical applicability, as it requires that, at each step, the GAN estimated density, call it dQ, and the true underlying density of the data, call it dPd, satisfy that βdQ ≤ dPd. However, it is indeed unknown how to design a generative network that induces a density dQ that would guarantee βdQ ≤ dPd with a non-zero β when dPd is a high-dimensional structure generative process."  
  - `video` <https://youtube.com/watch?v=5EEaY_cVYkk>
  - `video` <https://youtube.com/watch?v=myvoMklo5Uc> (Tolstikhin)
  - `video` <https://youtube.com/watch?v=wPKGIIy4rtU> (Bousquet)

#### ["Improved Training of Wasserstein GANs"](https://arxiv.org/abs/1704.00028) Gulrajani, Ahmed, Arjovsky, Dumoulin, Courville
  `GAN objective` `WGAN-GP`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#improved-training-of-wasserstein-gans-gulrajani-ahmed-arjovsky-dumoulin-courville>

#### ["Wasserstein GAN"](https://arxiv.org/abs/1701.07875) Arjovsky, Chintala, Bottou
  `GAN objective` `WGAN`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#wasserstein-gan-arjovsky-chintala-bottou>

#### ["f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization"](http://arxiv.org/abs/1606.00709) Nowozin, Cseke, Tomioka
  `GAN objective`
>	"We show that the generative-adversarial approach is a special case of an existing more general variational divergence estimation approach. We show that any f-divergence can be used for training generative neural samplers. We discuss the benefits of various choices of divergence functions on training complexity and the quality of the obtained generative models."  
>	"For a given f-divergence, the model learns the composition of the density ratio (of data to model density) with the derivative of f, by comparing generator and data samples. This provides a lower bound on the “true” divergence that would be obtained if the density ratio were perfectly known. In the event that the model is in a smaller class than the true data distribution, this broader family of divergences implements a variety of different approximations: some focus on individual modes of the true sample density, others try to cover the support."  
  - `video` <https://youtube.com/watch?v=I1M_jGWp5n0>
  - `video` <https://youtube.com/watch?v=kQ1eEXgGsCU> (Nowozin)
  - `video` <https://youtube.com/watch?v=y7pUN2t5LrA> (Nowozin)
  - `video` <https://youtu.be/jAI3rBI6poU?t=14m31s> (Ulyanov) `in russian`
  - `post` <https://colinraffel.com/blog/gans-and-divergence-minimization.html#citation-nowozin2016>
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/f_gan>

----
#### ["Which Training Methods for GANs Do Actually Converge?"](https://arxiv.org/abs/1801.04406) Mescheder, Geiger, Nowozin
  `GAN training protocol`
>	"Recent work has shown local convergence of GAN training for absolutely continuous data and generator distributions. In this paper, we show that the requirement of absolute continuity is necessary: we describe a simple yet prototypical counterexample showing that in the more realistic case of distributions that are not absolutely continuous, unregularized GAN training is not always convergent. Furthermore, we discuss regularization strategies that were recently proposed to stabilize GAN training. Our analysis shows that GAN training with instance noise or zero-centered gradient penalties converges. On the other hand, we show that Wasserstein-GANs and WGAN-GP with a finite number of discriminator updates per generator update do not always converge to the equilibrium point."  
>	"We discuss these results, leading us to a new explanation for the stability problems of GAN training. Based on our analysis, we extend our convergence results to more general GANs and prove local convergence for simplified gradient penalties even if the generator and data distributions lie on lower dimensional manifolds. We find these penalties to work well in practice and use them to learn high-resolution generative image models for a variety of datasets with little hyperparameter tuning."  
  - `code` <https://github.com/LMescheder/GAN_stability>

#### ["GANs Trained by a Two Time-Scale Update Rule Converge to a Nash Equilibrium"](https://arxiv.org/abs/1706.08500) Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter
  `GAN training protocol`
>	"We propose a two time-scale update rule (TTUR) for training GANs with stochastic gradient descent that has an individual learning rate for both the discriminator and the generator."  
>	"For the evaluation of the performance of GANs at image generation, we introduce the "Frechet Inception Distance" (FID) which captures the similarity of generated images to real ones better than the Inception Score."  
>	"In experiments, TTUR improves learning for DCGANs, improved Wasserstein GANs, and BEGANs."  
>	"to the best of our knowledge this is the first convergence proof for GANs"  
  - `video` <https://youtu.be/h6eQrkkU9SA?t=21m6s> (Hochreiter)
  - `video` <https://youtu.be/NZEAqdepq0w?t=46m25s> (Hochreiter)
  - `code` <https://github.com/bioinf-jku/TTUR>

#### ["How to Train Your DRAGAN"](https://arxiv.org/abs/1705.07215) Kodali, Abernethy, Hays, Kira
  `GAN training protocol`
  - `post` <http://lernapparat.de/more-improved-wgan>
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
#### ["A Style-Based Generator Architecture for Generative Adversarial Networks"](https://arxiv.org/abs/1812.04948) Karras, Laine, Aila
  `GAN applications` `image synthesis` `StyleGAN`
>	"We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis."  
  - `post` <https://www.gwern.net/Faces>

#### ["Self-Attention Generative Adversarial Networks"](https://arxiv.org/abs/1805.08318) Zhang, Goodfellow, Metaxas, Odena
  `GAN applications` `image synthesis` `SAGAN`
>	"SAGAN allows attention-driven, long-range dependency modeling for image generation tasks. Traditional convolutional GANs generate high-resolution details as a function of only spatially local points in lower-resolution feature maps. In SAGAN, details can be generated using cues from all feature locations. Moreover, the discriminator can check that highly detailed features in distant portions of the image are consistent with each other."  
  - `video` <https://youtube.com/watch?v=FdeHlC4QiqA> (Chen)

#### ["Synthesizing Programs for Images using Reinforced Adversarial Learning"](https://arxiv.org/abs/1804.01118) Ganin, Kulkarni, Babuschkin, Eslami, Vinyals
  `GAN applications` `image synthesis` `SPIRAL`
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#synthesizing-programs-for-images-using-reinforced-adversarial-learning-ganin-kulkarni-babuschkin-eslami-vinyals>

#### ["High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"](https://arxiv.org/abs/1711.11585) Wang, Liu, Zhu, Tao, Kautz, Catanzaro
  `GAN applications` `image synthesis`
  - <https://tcwang0509.github.io/pix2pixHD/> (demo)
  - `code` <https://github.com/NVIDIA/pix2pixHD>

----
#### ["Learning from Simulated and Unsupervised Images through Adversarial Training"](http://arxiv.org/abs/1612.07828) Shrivastava, Pfister, Tuzel, Susskind, Wang, Webb
  `GAN applications` `domain adaptation`
  - `post` <https://machinelearning.apple.com/2017/07/07/GAN.html>
  - `video` <https://youtube.com/watch?v=P3ayMdNdokg> (Shrivastava)
  - `video` <https://youtube.com/watch?v=ukt_F1FTNBA> (Karazeev) `in russian`
  - `code` <https://github.com/carpedm20/simulated-unsupervised-tensorflow>

#### ["Unsupervised Pixel-Level Domain Adaptation with Generative Asversarial Networks"](http://arxiv.org/abs/1612.05424) Bousmalis, Silberman, Dohan, Erhan, Krishnan
  `GAN applications` `domain adaptation`
  - `video` <https://youtube.com/watch?v=VhsTrWPvjcA> (Bousmalis)

----
#### ["Learning to Discover Cross-Domain Relations with Generative Adversarial Networks"](https://arxiv.org/abs/1703.05192) Kim et al.
  `GAN applications` `domain translation`
  - `code` <https://github.com/carpedm20/DiscoGAN-pytorch>
  - `code` <https://github.com/SKTBrain/DiscoGAN>

#### ["DualGAN: Unsupervised Dual Learning for Image-to-Image Translation"](https://arxiv.org/abs/1704.02510) Yi et al.
  `GAN applications` `domain translation`
  - `code` <https://github.com/wiseodd/generative-models/tree/master/GAN/dual_gan>

#### ["Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"](https://arxiv.org/abs/1703.10593) Zhu, Park, Isola, Efros
  `GAN applications` `domain translation` `CycleGAN`
  - <https://junyanz.github.io/CycleGAN/> (demo)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1703.10593>
  - `code` <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/cycle_gan.py>

#### ["Unsupervised Image-to-Image Translation Networks"](http://arxiv.org/abs/1703.00848) Liu, Breuel, Kautz
  `GAN applications` `domain translation`
  - `video` <https://facebook.com/nipsfoundation/videos/1554402331317667?t=3919> (Liu)



---
### generative models - variational autoencoders

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---variational-autoencoder)

[**interesting recent papers - generative models**](#generative-models)  
[**interesting recent papers - bayesian deep learning**](#bayesian-deep-learning)  
[**interesting recent papers - unsupervised learning**](#unsupervised-learning)  

----
#### ["Tighter Variational Bounds are Not Necessarily Better"](https://arxiv.org/abs/1802.04537) Rainforth, Kosiorek, Le, Maddison, Igl, Wood, Teh
>	"We provide theoretical and empirical evidence that using tighter evidence lower bounds can be detrimental to the process of learning an inference network by reducing the signal-to-noise ratio of the gradient estimator. Our results call into question common implicit assumptions that tighter ELBOs are better variational objectives for simultaneous model learning and inference amortization schemes."  
  - `post` <http://akosiorek.github.io/ml/2018/03/14/what_is_wrong_with_vaes.html>

#### ["Inference Suboptimality in Variational Autoencoders"](https://arxiv.org/abs/1801.03558) Cremer, Li, Duvenaud
>	"The quality of posterior inference is largely determined by two factors: a) the ability of the variational distribution to model the true posterior and b) the capacity of the recognition network to generalize inference over all datapoints. We find that suboptimal inference is often due to amortizing inference rather than the limited complexity of the approximating distribution. We show that this is due partly to the generator learning to accommodate the choice of approximation. Furthermore, we show that the parameters used to increase the expressiveness of the approximation play a role in generalizing inference rather than simply improving the complexity of the approximation."  

#### ["Fixing a Broken ELBO"](https://arxiv.org/abs/1711.00464) Alemi, Poole, Fischer, Dillon, Saurous, Murphy
>	"Fitting deep directed latent-variable models by maximizing the marginal likelihood or evidence is typically intractable, thus a common approximation is to maximize the evidence lower bound (ELBO) instead. However, maximum likelihood training (whether exact or approximate) does not necessarily result in a good latent representation, as we demonstrate both theoretically and empirically. In particular, we derive variational lower and upper bounds on the mutual information between the input and the latent variable, and use these bounds to derive a rate-distortion curve that characterizes the tradeoff between compression and reconstruction accuracy. Using this framework, we demonstrate that there is a family of models with identical ELBO, but different quantitative and qualitative characteristics. Our framework also suggests a simple new method to ensure that latent variable models with powerful stochastic decoders do not ignore their latent code."  
>	"We have motivated the β-VAE objective on information theoretic grounds, and demonstrated that comparing model architectures in terms of the rate-distortion plot offers a much better look at their performance and tradeoffs than simply comparing their marginal log likelihoods."  
  - `notes` <https://medium.com/peltarion/generative-adversarial-nets-and-variational-autoencoders-at-icml-2018-6878416ebf22>
  - `notes` <https://habr.com/company/yandex/blog/418421> `in russian`

----
#### ["Wasserstein Auto-Encoders"](https://arxiv.org/abs/1711.01558) Tolstikhin, Bousquet, Gelly, Scholkopf
  `WAE`
>	"WAE minimizes a penalized form of the Wasserstein distance between the model distribution and the target distribution, which leads to a different regularizer than the one used by VAE."  
>	"Both VAE and WAE minimize two terms: the reconstruction cost and the regularizer penalizing discrepancy between Pz and distribution induced by the encoder Q. VAE forces Q(Z|X=x) to match Pz for all the different input examples x drawn from Px. Every single red ball is forced to match Pz depicted as the white shape. Red balls start intersecting, which leads to problems with reconstruction. In contrast, WAE forces the continuous mixture Qz:=∫Q(z|x)dPx to match Pz. As a result latent codes of different examples get a chance to stay far away from each other, promoting a better reconstruction."  
>	"WAE shares many of the properties of VAEs (stable training, encoder-decoder architecture, nice latent manifold structure) while generating samples of better quality, as measured by the FID score."  
>	"WAE is a generalization of adversarial auto-encoder."  
  - `video` <https://facebook.com/iclr.cc/videos/2123421684353553?t=2850> (Gelly)
  - `paper` ["Adversarial Autoencoders"](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#adversarial-autoencoders-makhzani-shlens-jaitly-goodfellow) by Makhzani et al. `summary`

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
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1711.00937>
  - `slides` <https://avdnoord.github.io/homepage/slides/SANE2017.pdf>
  - `code` <https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb>

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
  - `video` <https://pscp.tv/w/1BRJjyZjqppGw> `in russian`
  - `notes` <https://bayesgroup.github.io/sufficient-statistics/posts/improved-variational-autoencoders-for-text-modeling-using-dilated-convolutions/> `in russian`

#### ["A Hybrid Convolutional Variational Autoencoder for Text Generation"](http://arxiv.org/abs/1702.02390) Semeniuta, Severyn, Barth
  - `code` <https://github.com/stas-semeniuta/textvae>

#### ["Generating Sentences from a Continuous Space"](http://arxiv.org/abs/1511.06349) Bowman, Vilnis, Vinyals, Dai, Jozefowicz, Bengio
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1511.06349>
  - `code` <https://github.com/kefirski/pytorch_RVAE>
  - `code` <https://github.com/cheng6076/Variational-LSTM-Autoencoder>



---
### generative models - autoregressive models

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---autoregressive-models)

[**interesting recent papers - generative models**](#generative-models)

----
#### ["Universal Transformers"](https://arxiv.org/abs/1807.03819) Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser
  `Transformer`
  - `post` <https://ai.googleblog.com/2018/08/moving-beyond-translation-with.html>
  - `code` <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer.py>

#### ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) Vaswani et al.
  `Transformer`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#attention-is-all-you-need-vaswani-et-al>

----
#### ["WaveNet: A Generative Model for Raw Audio"](http://arxiv.org/abs/1609.03499) Oord et al.
  `WaveNet`
  - `post` <https://deepmind.com/blog/wavenet-generative-model-raw-audio/>
  - `video` <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Learning-Symposium-Session-1> (42:36) (van den Oord)
  - `video` <https://youtube.com/watch?v=leu286ciQcE> (Kalchbrenner)
  - `video` <https://youtube.com/watch?v=YyUXG-BfDbE> (Andrews)
  - `video` <https://youtube.com/watch?v=gUdyQ5Ocr0g> (Zakirov) `in russian`
  - `post` <https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#wavenet>
  - `code` <https://github.com/vincentherrmann/pytorch-wavenet>
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
  - `post` <https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#pixelrnn>
  - `post` <https://github.com/tensorflow/magenta/blob/master/magenta/reviews/pixelrnn.md>
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
  - `audio` <https://soundcloud.com/nlp-highlights/52-sequence-to-sequence-learning-as-beam-search-optimization-with-sam-wiseman> (Rush)
  - `notes` <http://shortscience.org/paper?bibtexKey=journals/corr/1606.02960>
  - `notes` <https://medium.com/@sharaf/a-paper-a-day-2-sequence-to-sequence-learning-as-beam-search-optimization-92424b490350>



---
### reinforcement learning - model-free methods

[**interesting older papers - value-based methods**](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---value-based-methods)  
[**interesting older papers - policy-based methods**](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---policy-based-methods)  

----
#### ["Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms?"](https://arxiv.org/abs/1811.02553) Ilyas et al.
>	"Deep reinforcement learning algorithms are rooted in a well-grounded framework of classical RL. However, as our investigations uncover, this framework fails to explain much of the behavior of these algorithms. This disconnect impedes our understanding of why these algorithms succeed (or fail). It also poses a major barrier to addressing key challenges facing deep RL, such as widespread brittleness and poor reproducibility. To close this gap, we need to either develop methods that adhere more closely to theory, or build theory that can capture what makes existing policy gradient methods successful. In both cases, the first step is to precisely pinpoint where theory and practice diverge.  
>	"Gradient estimation. Our analysis shows that the quality of gradient estimates that policy gradient algorithms use is rather poor. Indeed, even when agents are still improving, such gradient estimates are often virtually uncorrelated with the true gradient and with each other. Our results thus indicate that adhering to existing theory would require algorithms that obtain better estimates of the gradient. Alternatively, one might aim to broaden the theory to explain why modern policy gradient algorithms are successful despite relying on such poor gradient estimates.  
>	Value prediction. The findings presented identify two key issues. First, while the value network successfully solves the supervised learning task it is trained on, it does not accurately model the “true” value function. Second, employing the value network as a baseline does decrease the gradient variance (compared to the trivial (“zero”) baseline). However, this decrease is rather marginal compared to the variance reduction offered by the “true” value function, but employing a value network dramatically increases agent’s performance. These phenomena motivate us to ask: is this failure in modeling the true value function inevitable? And what is the real role of the value network in policy gradient methods?  
>	Optimization landscape. The optimization landscape induced by modern policy gradient algorithms is often not reflective of the underlying true reward landscape. In fact, in the sample-regime where policy gradient methods operate, the true reward landscape is noisy and the surrogate reward is often misleading. We thus need a better understanding of why the current methods succeed despite these issues, and, more broadly, how to navigate the true reward landscape more accurately.  
>	Trust region approximation. Our findings indicate that there may be a number of reasons why policies need to be locally similar. These include noisy gradient estimates, poor baseline functions and misalignment of the surrogate landscape. Not only is our theory surrounding trust region optimization oblivious to these factors, it is also notoriously difficult to translate this theory into efficient algorithms. Deep policy gradient methods thus resort to relaxations of trust region constraints, which makes their performance difficult to properly understand and analyze. Therefore, we need either techniques that enforce trust regions more strictly, or a rigorous theory of trust region relaxations."  

#### ["Simple Random Search Provides a Competitive Approach to Reinforcement Learning"](https://arxiv.org/abs/1803.07055) Mania, Guy, Recht
>	"We attempted to find the simplest algorithm for model-free RL that performs well on the continuous control benchmarks used in the RL literature. We demonstrated that with a few algorithmic augmentations, basic random search could be used to train linear policies that achieve state-of-theart sample efficiency on the MuJoCo locomotion tasks. We showed that linear policies match the performance of complex neural network policies and can be found through a simple algorithm."  
>	"For application to continuous control, we augment the basic random search method with three simple features. First, we scale each update step by the standard deviation of the rewards collected for computing that update step. Second, we normalize the system’s states by online estimates of their mean and standard deviation. Third, we discard from the computation of the update steps the directions that yield the least improvement of the reward."  
>	"Since the algorithm and policies are simple, we were able to perform extensive sensitivity studies, and observed that our method can find good solutions to highly nonconvex problems a large fraction of the time. Our results emphasize the high variance intrinsic to the training of policies for MuJoCo RL tasks. Therefore, it is not clear what is gained by evaluating RL algorithms on only a small numbers of random seeds, as is common in the RL literature. Evaluation on small numbers of random seeds does not capture performance adequately due to high variance."  
>	"Though many RL researchers are concerned about minimizing sample complexity, it does not make sense to optimize the running time of an algorithm on a single instance. The running time of an algorithm is only a meaningful notion if either (a) evaluated on a family of instances, or (b) when clearly restricting the class of algorithms. Common RL practice, however, does not follow either (a) or (b). Instead researchers run algorithm A on task T with a given hyperparameter configuration, and plot a “learning curve” showing the algorithm reaches a target reward after collecting X samples. Then the “sample complexity” of the method is reported as the number of samples required to reach a target reward threshold, with the given hyperparameter configuration. However, any number of hyperparameter configurations can be tried. Any number of algorithmic enhancements can be added or discarded and then tested in simulation. For a fair measurement of sample complexity, should we not count the number of rollouts used for every tested hyperparameters?"  
  - `post` <http://argmin.net/2018/03/20/mujocoloco/>
  - `post` <http://argmin.net/2018/03/26/outsider-rl/>
  - `code` <https://github.com/modestyachts/ARS>

#### ["Towards Generalization and Simplicity in Continuous Control"](https://arxiv.org/abs/1703.02660) Rajeswaran, Lowrey, Todorov, Kakade
>	"This work shows that policies with simple linear and RBF parameterizations can be trained to solve a variety of continuous control tasks, including the OpenAI gym benchmarks. The performance of these trained policies are competitive with state of the art results, obtained with more elaborate parameterizations such as fully connected neural networks. Furthermore, existing training and testing scenarios are shown to be very limited and prone to over-fitting, thus giving rise to only trajectory-centric policies. Training with a diverse initial state distribution is shown to produce more global policies with better generalization. This allows for interactive control scenarios where the system recovers from large on-line perturbations."  
  - `video` <https://youtube.com/watch?v=frojcskMkkY>
  - `code` <https://github.com/aravindr93/mjrl>

----
#### ["Optimizing Agent Behavior over Long Time Scales by Transporting Value"](https://arxiv.org/abs/1810.06721) Hung et al.
  `TVT` `credit assignment`
>	"Humans spend a remarkable fraction of waking life engaged in acts of "mental time travel". We dwell on our actions in the past and experience satisfaction or regret. More than merely autobiographical storytelling, we use these event recollections to change how we will act in similar scenarios in the future. This process endows us with a computationally important ability to link actions and consequences across long spans of time, which figures prominently in addressing the problem of long-term temporal credit assignment; in artificial intelligence this is the question of how to evaluate the utility of the actions within a long-duration behavioral sequence leading to success or failure in a task. Existing approaches to shorter-term credit assignment in AI cannot solve tasks with long delays between actions and consequences. Here, we introduce a new paradigm for reinforcement learning where agents use recall of specific memories to credit actions from the past, allowing them to solve problems that are intractable for existing algorithms."  
>	"Temporal Value Transpor is a credit assignment scheme similar to RUDDER. The critical insight here is that they use the agent's memory access to decide on credit assignment. So if the model uses a memory from 512 steps ago, the choice from 512 steps ago gets credit for the current reward."  
>	"RMA is a simplified version of MERLIN which uses model based RL and long term memory."  
>	"RMA has a long term memory and chooses to save and load working memory represented by the LSTM's hidden state."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1810.06721>

#### ["RUDDER: Return Decomposition for Delayed Rewards"](https://arxiv.org/abs/1806.07857) Arjona-Medina, Gillhofer, Widrich, Unterthiner, Hochreiter
  `RUDDER` `credit assignment`
>	"In this work, biases of temporal difference estimates are proved to be corrected only exponentially slowly in the number of delay steps. Furthermore, variances of Monte Carlo estimates are proved to increase the variance of other estimates, the number of which can exponentially grow in the number of delay steps."  
>	"We propose RUDDER, a novel reinforcement learning approach for finite Markov decision processes with delayed rewards, which creates a new MDP with same optimal policies as the original MDP but with redistributed rewards that have largely reduced delays. If the return decomposition is optimal, then the new MDP does not have delayed rewards and TD estimates are unbiased. In this case, the rewards track Q-values so that the future expected reward is always zero."  
>	"We experimentally confirm our theoretical results on bias and variance of TD and MC estimates. On artificial tasks with different lengths of reward delays, we show that RUDDER is exponentially faster than TD, MC, and MC Tree Search. RUDDER outperforms Rainbow, A3C, DDQN, Distributional DQN, Dueling DDQN, Noisy DQN, and Prioritized DDQN on the delayed reward Atari game Venture in only a fraction of the learning time."  
>	"The core of the paper breaks down to transforming the reward function of an MDP into one which is simpler to learn, because the delayed reward is redistributed to key events. Also, in non-deterministic cases, high variance in later states are moved back (via reward redistribution) to the key-events responsible for that variance."  
>	"The big idea of RUDDER is to use LSTM to predict return of an episode. To do this, the LSTM will have to recognize what actually causes the reward (e.g. shooting the gun in the right direction causes the reward, even if we get the reward only once the bullet hits the enemy after travelling along the screen). We then use a salience method (Layer-Wise Relevance Propagation or Integrated Gradients) to get that information out of LSTM, and redistribute reward accordingly (i.e., we then give reward already once the gun is shot in the right direction). Once the reward is redistributed this way, solving/learning the actual reinforcement learning problem is much, much easier and as we prove in the paper, the optimal policy does not change with this redistribution."  
>	"Theoretical results:  
>	- bias-variance treatment via exponential averages and arithmetic means of TD and MC  
>	- variance formula for sampling return from an MDP  
>	- deriving problem of exponentially small bias correction for TD in case of delayed rewards  
>	- deriving problem of single delayed reward increasing variance of exponentially many other action-value values  
>	- concept of return-equivalent MDPs  
>	- return-equivalent transformation of immediate reward MDP into delayed reward MDP which can be used to train LSTM which only sees reward at episode's end  
>	- return-equivalent transformation of delayed reward MDP into MDP with much reduced delay of reward using reward redistribution and return decomposition  
>	"  
  - `video` <https://youtube.com/playlist?list=PLDfrC-Vpg-CzVTqSjxVeLQZy3f7iv9vyY> (demo)
  - `video` <https://youtu.be/NZEAqdepq0w?t=56m34s> (Hochreiter)
  - `post` <https://widmi.github.io>
  - `code` <https://github.com/ml-jku/baselines-rudder>

#### ["TD or not TD: Analyzing the Role of Temporal Differencing in Deep Reinforcement Learning"](https://arxiv.org/abs/1806.01175) Amiranashvili, Dosovitskiy, Koltun, Brox
  `credit assignment`
>	"There is little understanding of when and why certain deep RL algorithms work well. Theoretical results are mainly based on tabular environments or linear function approximators. Their assumptions do not cover the typical application domains of deep RL, which feature extremely high input dimensionality (typically in the tens of thousands) and the use of nonlinear function approximators. Thus, our understanding of deep RL is based primarily on empirical results, and these empirical results guide the design of deep RL algorithms."  
>	"We perform a controlled experimental study aiming at better understanding the role of temporal differencing in modern deep reinforcement learning, which is characterized by essentially infinite-dimensional state spaces, extremely high observation dimensionality, partial observability, and deep nonlinear models used as function approximators. We focus on environments with visual inputs and discrete action sets, and algorithms that involve prediction of value or action-value functions."  
>	"By varying the parameters such as the balance between TD and MC in the learning update or the prediction horizon, we are able to clearly isolate the effect of these parameters on learning. Moreover, we designed a series of controlled scenarios that focus on specific characteristics of RL problems: reward sparsity, reward delay, perceptual complexity, and properties of terminal states."  
>	"Temporal differencing methods are generally considered superior to Monte Carlo methods in reinforcement learning. This opinion is largely based on empirical evidence from domains such as gridworlds, cart pole, and mountain car. Our results agree: in gridworlds and on Atari games we find that n-step Q learning outperforms QMC. We further find, similar to the TD(λ) experiments from the past, that a mixture of MC and TD achieves best results in n-step Q and A3C. However, the situation changes in perceptually complex environments. In our experiments in immersive three-dimensional simulations, a finite-horizon MC method matches or outperforms TD-based methods."  
>	"Our findings in modern deep RL settings both support and contradict past results on the merits of TD. On the one hand, value-based infinite-horizon methods perform best with a mixture of TD and MC; this is consistent with the TD(λ) results of Sutton (1988). On the other hand, in sharp contrast to prior beliefs, we observe that Monte Carlo algorithms can perform very well on challenging RL tasks. This is made possible by simply limiting the prediction to a finite horizon. Surprisingly, finite-horizon Monte Carlo training is successful in dealing with sparse and delayed rewards, which are generally assumed to impair this class of methods. Monte Carlo training is also more stable to noisy rewards and is particularly robust to perceptual complexity and variability."  
>	"While TD is at an advantage in tasks with simple perception, long planning horizons, or terminal rewards, MC training is more robust to noisy rewards, effective for training perception systems from raw sensory inputs, and surprisingly successful in dealing with sparse and delayed rewards."  
>	"What is the reason for this contrast between classic findings and our results? We believe that the key difference is in the complexity of perception in immersive three-dimensional environments, which was not present in gridworlds and other classic problems, and is only partially present in Atari games. In immersive simulation, the agent’s observation is a high-dimensional image that represents a partial view of a large (mostly hidden) three-dimensional environment. The dimensionality of the state space is essentially infinite: the underlying environment is specified by continuous surfaces in three-dimensional space. Memorizing all possible states is easy and routine in gridworlds and is also possible in some Atari games, but is not feasible in immersive three-dimensional simulations. Therefore, in order to successfully operate in such simulations, the agent has to learn to extract useful representations from the observations it receives. Encoding a meaningful representation from rich perceptual input is where Monte Carlo methods are at an advantage due to the reliability of their training signals. Monte Carlo methods train on ground-truth targets, not “guess from a guess”, as TD methods do."  

----
#### ["Diagnosing Bottlenecks in Deep Q-learning Algorithms"](https://arxiv.org/abs/1902.10250) Fu, Kumar, Soh, Levine
  `Q-learning`
>	"The behavior of Q-learning methods with function approximation is poorly understood, both theoretically and empirically. In this work, we aim to experimentally investigate potential issues in Q-learning, by means of a "unit testing" framework where we can utilize oracles to disentangle sources of error. Specifically, we investigate questions related to function approximation, sampling error and nonstationarity, and where available, verify if trends found in oracle settings hold true with modern deep RL methods. We find that large neural network architectures have many benefits with regards to learning stability; offer several practical compensations for overfitting; and develop a novel sampling method based on explicitly compensating for function approximation error that yields fair improvement on high-dimensional continuous control domains."  
>	"Ever wonder if Q-learning will converge? Should you use a big or small model? On-policy or off-policy data? Many or few grad steps?"  

#### ["Deep Reinforcement Learning and the Deadly Triad"](https://arxiv.org/abs/1812.02648) Hasselt, Doron, Strub, Hessel, Sonnerat, Modayil
  `Q-learning`
>	"We know from reinforcement learning theory that temporal difference learning can fail in certain cases. Sutton and Barto identify a deadly triad of function approximation, bootstrapping, and off-policy learning. When these three properties are combined, learning can diverge with the value estimates becoming unbounded. However, several algorithms successfully combine these three properties, which indicates that there is at least a partial gap in our understanding. In this work, we investigate the impact of the deadly triad in practice, in the context of a family of popular deep reinforcement learning models - deep Q-networks trained with experience replay - analysing how the components of this system play a role in the emergence of the deadly triad, and in the agent’s performance."  
>	"When combining TD learning with function approximation, updating the value at one state creates a risk of inappropriately changing the values of other states, including the state being bootstrapped upon. This is not a concern when the agent updates the values used for bootstrapping as often as they are used. However, if the agent is learning off-policy, it might not update these bootstrap values sufficiently often. This can create harmful learning dynamics that can lead to divergence of the function parameters. The combination of function approximation, off-policy learning, and bootstrapping has been called “the deadly triad” due to this possibility of divergence."  
>	"The Deep Q-network agent uses deep neural networks to approximate action values, which are updated by Q-learning, an off-policy algorithm. Moreover, DQN uses experience replay to sample transitions, thus the updates are computed from transitions sampled according to a mixture of past policies rather than the current policy. This causes the updates to be even more off-policy. Finally, since DQN uses one-step Q-learning as its learning algorithm, it relies on bootstrapping. Despite combining all these components of the deadly triad, DQN successfully learnt to play many Atari 2600 games."  

#### ["Non-delusional Q-learning and Value-iteration"](https://papers.nips.cc/paper/8200-non-delusional-q-learning-and-value-iteration) Lu, Schuurmans, Boutilier
  `PCQL` `PCVI` `Q-learning` `NeurIPS 2018`
>	"We identify a fundamental source of error in Q-learning and other forms of dynamic programming with function approximation. Delusional bias arises when the approximation architecture limits the class of expressible greedy policies. Since standard Q-updates make globally uncoordinated action choices with respect to the expressible policy class, inconsistent or even conflicting Q-value estimates can result, leading to pathological behaviour such as over/under-estimation, instability and even divergence. To solve this problem, we introduce a new notion of policy consistency and define a local backup process that ensures global consistency through the use of information sets---sets that record constraints on policies consistent with backed-up Q-values. We prove that both the model-based and model-free algorithms using this backup remove delusional bias, yielding the first known algorithms that guarantee optimal results under general conditions. These algorithms furthermore only require polynomially many information sets (from a potentially exponential support). Finally, we suggest other practical heuristics for value-iteration and Q-learning that attempt to reduce delusional bias."  
>	"New source of error in value-based RL with value approximation arises due to interaction of two factors:  
>	- restrictions on realizable greedy policies due to approximator  
>	- independent Bellman backups: oblivious to policy class"  
  - `video` <https://youtube.com/watch?v=PSfJ44C3-sU> (Lu)
  - `video` <http://www.fields.utoronto.ca/video-archive/2019/02/2509-19619> (Boutilier)

#### ["Is Q-learning Provably Efficient?"](https://arxiv.org/abs/1807.03765) Jin, Allen-Zhu, Bubeck, Jordan
  `Q-learning`
>	"The theoretical question of "whether model-free algorithms can be made sample efficient" is one of the most fundamental questions in RL, and remains unsolved even in the basic scenario with finitely many states and actions."  
>	"We prove that, in an episodic MDP setting, Q-learning with UCB exploration achieves regret O(sqrt(H^3SAT)), where S and A are the numbers of states and actions, H is the number of steps per episode, and T is the total number of steps. This sample efficiency matches the optimal regret that can be achieved by any model-based approach, up to a single √H factor. To the best of our knowledge, this is the first analysis in the model-free setting that establishes √T regret without requiring access to a "simulator."  
  - `video` <https://youtu.be/Tge7LPT9vGA?t=20m37s> (Jin)

#### ["Implicit Quantile Networks for Distributional Reinforcement Learning"](https://arxiv.org/abs/1806.06923) Dabney, Ostrovski, Silver, Munos
  `IQN` `distributional RL` `Q-learning`
>	"Using quantile regression to approximate the full quantile function for the state-action return distribution. By reparameterizing a distribution over the sample space, this yields an implicitly defined return distribution and gives rise to a large class of risk-sensitive policies."  
>	"There may be additional benefits to implicit quantile networks beyond the obvious increase in representational fidelity. As with UVFAs, we might hope that training over many different τ’s (goals in the case of the UVFA) leads to better generalization between values and improved sample complexity than attempting to train each separately."  
  - `video` <https://facebook.com/icml.imls/videos/429611280886726?t=2511> (Dabney)
  - `code` <https://github.com/google/dopamine/blob/master/dopamine/agents/implicit_quantile/implicit_quantile_agent.py>

#### ["An Analysis of Categorical Distributional Reinforcement Learning"](https://arxiv.org/abs/1802.08163) Rowland, Bellemare, Dabney, Munos, Teh
  `CDRL` `distributional RL` `Q-learning`
  - `video` <http://videocrm.ca/Machine18/Machine18-20180423-3-MarcBellemare> (22:23) (Bellemare)

#### ["Distributional Reinforcement Learning with Quantile Regression"](https://arxiv.org/abs/1710.10044) Dabney, Rowland, Bellemare, Munos
  `QR-DQN` `distributional RL` `Q-learning`
>	"One of the theoretical contributions of the C51 work was a proof that the distributional Bellman operator is a contraction in a maximal form of the Wasserstein metric between probability distributions. In this context, the Wasserstein metric is particularly interesting because it does not suffer from disjoint-support issues which arise when performing Bellman updates. Unfortunately, this result does not directly lead to a practical algorithm: the Wasserstein metric, viewed as a loss, cannot generally be minimized using stochastic gradient methods. This negative result left open the question as to whether it is possible to devise an online distributional reinforcement learning algorithm which takes advantage of the contraction result. Instead, the C51 algorithm first performs a heuristic projection step, followed by the minimization of a KL divergence between projected Bellman update and prediction. The work therefore leaves a theory-practice gap in our understanding of distributional reinforcement learning, which makes it difficult to explain the good performance of C51. Thus, the existence of a distributional algorithm that operates end-to-end on the Wasserstein metric remains an open question."  
>	"In this paper, we answer this question affirmatively. By appealing to the theory of quantile regression, we show that there exists an algorithm, applicable in a stochastic approximation setting, which can perform distributional reinforcement learning over the Wasserstein metric. Our method relies on the following techniques:  
>	- We “transpose” the parametrization from C51: whereas the former uses N fixed locations for its approximation distribution and adjusts their probabilities, we assign fixed, uniform probabilities to N adjustable locations  
>	- We show that quantile regression may be used to stochastically adjust the distributions’ locations so as to minimize the Wasserstein distance to a target distribution  
>	- We formally prove contraction mapping results for our overall algorithm, and use these results to conclude that our method performs distributional RL end-to-end under the Wasserstein metric, as desired"  
>	"Authors proposed the use of quantile regression as a method for minimizing the 1-Wasserstein in the univariate case when approximating using a mixture of Dirac functions."  
>	"The quantile regression loss for a quantile at τ∈ [0,1] and error u (positive for underestimation and negative for overestimation) is given by ρτ(u) = (τ − I{u ≤ 0})u. It is an asymmetric loss function penalizing underestimation by weight τ and overestimation by weight 1 − τ. For a given scalar distribution Z with c.d.f. Fz and a quantile τ, the inverse c.d.f. q = Fz−1(τ) minimizes the expected quantile regression loss E z∼ Z [ρτ(z − q)]. Using this loss allows one to train a neural network to approximate a scalar distribution represented by its inverse c.d.f. For this, the network can output a fixed grid of quantiles, with the respective quantile regression losses being applied to each output independently. A more effective approach is to provide the desired quantile τ as an additional input to the network, and train it to output the corresponding value of Fz−1(τ)."  
  - `post` <https://mtomassoli.github.io/2017/12/08/distributional_rl/>
  - `code` <https://github.com/higgsfield/RL-Adventure>
  - `code` <https://github.com/NervanaSystems/coach/blob/master/agents/qr_dqn_agent.py>

#### ["A Distributional Perspective on Reinforcement Learning"](https://arxiv.org/abs/1707.06887) Bellemare, Dabney, Munos
  `C51` `Categorical DQN` `distributional RL` Q-learning`
>	"The value function gives the expected future discounted return. This ignores variance and multi-modality. Authors argue for modelling the full distribution of the return."  
>	"Distributional Bellman equation"  
>	"It is unclear if method works because of modelling uncertainty over rewards, training network with richer signal (categorical loss) or using distributional Bellman update."  
>	"Bellman (1957): Bellman equation for mean  
>	Sobel (1982): ... for variance  
>	Engel (2003): ... for Bayesian uncertainty  
>	Azar et al. (2011), Lattimore & Hutter (2012): ... for higher moments  
>	Morimura et al. (2010, 2010b): ... for densities"  
  - `post` <https://deepmind.com/blog/going-beyond-average-reinforcement-learning/>
  - `post` <http://marcgbellemare.info/blog/eighteen-months-of-rl-research-at-google-brain-in-montreal>
  - `video` <https://youtube.com/watch?v=yFBwyPuO2Vg> (demo)
  - `video` <http://videocrm.ca/Machine18/Machine18-20180423-3-MarcBellemare> (Bellemare)
  - `video` <https://vimeo.com/235922311> (Bellemare)
  - `video` <https://vimeo.com/237274251> (Bellemare)
  - `video` <https://videolectures.net/DLRLsummerschool2018_bellemare_deep_RL/#t=2010> (Bellemare)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=4m45s> (Mnih)
  - `video` <https://youtu.be/fnwo3GCmyEo?t=6m45s> (Fritzler) `in russian`
  - `video` <https://youtu.be/5REJGbNu-Kk?t=11m45s> (Grinchuk) `in russian`
  - `post` <https://mtomassoli.github.io/2017/12/08/distributional_rl/>
  - `post` <https://flyyufelix.github.io/2017/10/24/distributional-bellman.html>
  - `code` <https://github.com/higgsfield/RL-Adventure>
  - `code` <https://github.com/floringogianu/categorical-dqn>
  - `code` <https://github.com/flyyufelix/C51-DDQN-Keras>
  - `code` <https://github.com/reinforceio/tensorforce/blob/master/tensorforce/models/categorical_dqn_model.py>

#### ["Discrete Sequential Prediction of Continuous Actions for Deep RL"](https://arxiv.org/abs/1705.05035) Metz, Ibarz, Jaitly, Davidson
  `SDQN` `Q-learning`
>	"We draw inspiration from the recent success of sequence-to-sequence models for structured prediction problems to develop policies over discretized spaces. Central to this method is the realization that complex functions over high dimensional spaces can be modeled by neural networks that use next step prediction. Specifically, we show how Q-values and policies over continuous spaces can be modeled using a next step prediction model over discretized dimensions. With this parameterization, it is possible to both leverage the compositional structure of action spaces during learning, as well as compute maxima over action spaces (approximately). On a simple example task we demonstrate empirically that our method can perform global search, which effectively gets around the local optimization issues that plague DDPG and NAF. We apply the technique to off-policy (Q-learning) methods and show that our method can achieve the state-of-the-art for off-policy methods on several continuous control tasks."  

#### ["Learning to Play in a Day: Faster Deep Reinforcement Learning by Optimality Tightening"](https://arxiv.org/abs/1611.01606) He, Liu, Schwing, Peng
  `Q-learning`
>	"We propose a novel training algorithm for reinforcement learning which combines the strength of deep Q-learning with a constrained optimization approach to tighten optimality and encourage faster reward propagation."  
>	"Optimality tightening introduces an objective based on the lower/upper bound of the optimal Q-function."  
  - `video` <https://yadi.sk/i/yBO0q4mI3GAxYd> (1:10:20) (Fritzler) `in russian`
  - `video` <https://youtu.be/mrj_hyH974o?t=16m13s> (Podoprikhin) `in russian`

#### ["Neural Episodic Control"](https://arxiv.org/abs/1703.01988) Pritzel, Uria, Srinivasan, Puigdomenech, Vinyals, Hassabis, Wierstra, Blundell
  `Q-learning` `episodic memory`
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
  - `code` <https://github.com/mjacar/pytorch-nec>
  - `code` <https://github.com/EndingCredits/Neural-Episodic-Control>
  - `code` <https://github.com/NervanaSystems/coach/blob/master/agents/nec_agent.py>

#### ["Model-Free Episodic Control"](http://arxiv.org/abs/1606.04460) Blundell, Uria, Pritzel, Li, Ruderman, Leibo, Rae, Wierstra, Hassabis
  `Q-learning` `episodic memory`
>	"This might be achieved by a dual system (hippocampus vs neocortex) where information are stored in alternated way such that new nonstationary experience is rapidly encoded in the hippocampus (most flexible region of the brain with the highest amount of plasticity and neurogenesis); long term memory in the cortex is updated in a separate phase where what is updated (both in terms of samples and targets) can be controlled and does not put the system at risk of instabilities."  
  - <https://sites.google.com/site/episodiccontrol/> (demo)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=42m10s> (Mnih)
  - `post` <http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-4-episodic-and-interactive-memory/>
  - `code` <https://github.com/ShibiHe/Model-Free-Episodic-Control>
  - `code` <https://github.com/sudeepraja/Model-Free-Episodic-Control>

----
#### ["Natural Value Approximators: Learning when to Trust Past Estimates"](http://papers.nips.cc/paper/6807-natural-value-approximators-learning-when-to-trust-past-estimates) Xu, Modayil, Hasselt, Barreto, Silver, Schaul
  `NVA` `value-based`
>	"Neural networks are most effective for value function approximation when the desired target function is smooth. However, value functions are, by their very nature, discontinuous functions with sharp variations over time. We introduce a representation of value that matches the natural temporal structure of value functions."  
>	"A value function represents the expected sum of future discounted rewards. If non-zero rewards occur infrequently but reliably, then an accurate prediction of the cumulative discounted reward rises as such rewarding moments approach and drops immediately after. This is a pervasive scenario because many domains associate positive or negative reinforcements to salient events (like picking up an object, hitting a wall, or reaching a goal position). The problem is that the agent’s observations tend to be smooth in time, so learning an accurate value estimate near those sharp drops puts strain on the function approximator - especially when employing differentiable function approximators such as neural networks that naturally make smooth maps from observations to outputs."  
>	"We incorporate the temporal structure of cumulative discounted rewards into the value function itself. The main idea is that, by default, the value function can respect the reward sequence. If no reward is observed, then the next value smoothly matches the previous value, but becomes a little larger due to the discount. If a reward is observed, it should be subtracted out from the previous value: in other words a reward that was expected has now been consumed. The natural value approximator combines the previous value with the observed rewards and discounts, which makes this sequence of values easy to represent by a smooth function approximator such as a neural network."  
  - `video` <https://facebook.com/nipsfoundation/videos/1554741347950432?t=4212> (Xu)

#### ["Regret Minimization for Partially Observable Deep Reinforcement Learning"](https://arxiv.org/abs/1710.11424) Jin, Levine, Keutzer
  `ARM` `value-based`
>	"Algorithm based on counterfactual regret minimization that iteratively updates an approximation to a cumulative clipped advantage function."  
>	"In contrast to prior methods, advantage-based regret minimization is well suited to partially observed or non-Markovian environments."  
  - `video` <https://vimeo.com/287803161> (Jin)

#### ["Multi-step Reinforcement Learning: A Unifying Algorithm"](https://arxiv.org/abs/1703.01327) De Asis, Hernandez-Garcia, Holland, Sutton
  `Q(σ)` `value-based`
>	"Currently, there are a multitude of algorithms that can be used to perform TD control, including Sarsa, Q-learning, and Expected Sarsa. These methods are often studied in the one-step case, but they can be extended across multiple time steps to achieve better performance. Each of these algorithms is seemingly distinct, and no one dominates the others for all problems. In this paper, we study a new multi-step action-value algorithm called Q(σ) which unifies and generalizes these existing algorithms, while subsuming them as special cases. A new parameter, σ, is introduced to allow the degree of sampling performed by the algorithm at each step during its backup to be continuously varied, with Sarsa existing at one extreme (full sampling), and Expected Sarsa existing at the other (pure expectation)."  
>	"With a constant value of sampling parameter σ, Q(σ) is a weighted average between tree backups and regular SARSA: σ varies the breadth of the tree backup, contrasted with TD(λ) where λ varies the depth. Q(σ) allows for interpolation in bias-variance tradeoff: if σ is dynamically adjusted, can enforce a desirable tradeoff."  
  - `video` <https://youtube.com/watch?v=MidZJ-oCpRk> (De Asis)
  - `video` <https://youtube.com/watch?v=_OP5g1gRP5s> (Hernandez-Garcia)
  - `video` <https://youtu.be/dZmCOIJ7Cyc?t=7m3s> (Bobyrev) `in russian`

#### ["Convergent Tree-Backup and Retrace with Function Approximation"](https://arxiv.org/abs/1705.09322) Touati, Bacon, Precup, Vincent
  `Retrace` `value-based` `off-policy evaluation`
>	"We show that Tree Backup and Retrace algorithms are unstable with linear function approximation, both in theory and with specific examples. We addressed these issues by formulating gradient-based versions of these algorithms which minimize the mean-square projected Bellman error. Using a saddle-point formulation, we were also able to provide convergence guarantees and characterize the convergence rate of our algorithms."  
>	"The design and analysis of off-policy algorithms using all the features of reinforcement learning, e.g. bootstrapping, multi-step updates (eligibility traces), and function approximation has been explored extensively over three decades. While off-policy learning and function approximation have been understood in isolation, their combination with multi-steps bootstrapping produces a so-called deadly triad, i.e., many algorithms in this category are unstable. A convergent approach to this triad is provided by importance sampling, which bends the behavior policy distribution onto the target one. However, as the length of the trajectories increases, the variance of importance sampling corrections tends to become very large. An alternative approach which was developed for tabular representations of the value function is the tree backup algorithm which, remarkably, does not rely on importance sampling directly. Tree Backup has recently been revisited by authors of Retrace(λ) algorithm. Both Tree Backup and Retrace(λ) were only shown to converge with a tabular value function representation, and whether they would also converge with function approximation was an open question, which we tackle in this paper."  
  - `video` <https://facebook.com/icml.imls/videos/430846900763164?t=303> (Touati)

#### ["Safe and Efficient Off-Policy Reinforcement Learning"](http://arxiv.org/abs/1606.02647) Munos, Stepleton, Harutyunyan, Bellemare
  `Retrace` `value-based` `off-policy evaluation`
>	"Retrace(λ) is a new strategy to weight a sample for off-policy learning, it provides low-variance, safe and efficient updates."  
>	"Our goal is to design a RL algorithm with two desired properties. Firstly, to use off-policy data, which is important for exploration, when we use memory replay, or observe log-data. Secondly, to use multi-steps returns in order to propagate rewards faster and avoid accumulation of approximation/estimation errors. Both properties are crucial in deep RL. We introduce the “Retrace” algorithm, which uses multi-steps returns and can safely and efficiently utilize any off-policy data."  
>	"open issue: off policy unbiased, low variance estimators for long horizon delayed reward problems"  
>	"As a corollary, we prove the convergence of Watkins’ Q(λ), which was an open problem since 1989."  
  - `video` <https://youtu.be/WuFMrk3ZbkE?t=35m30s> (Bellemare)
  - `video` <https://youtube.com/watch?v=8hK0NnG_DhY&t=25m27s> (Brunskill)
  - `video` <https://youtu.be/ggPGtMSoVN8?t=51m10s> (Petrenko) `in russian`

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
#### ["Self-Imitation Learning"](https://arxiv.org/abs/1806.05635) Oh, Guo, Singh, Lee
  `SIL` `lower-bound soft Q-learning`
>	"A simple off-policy actor-critic algorithm that learns to reproduce the agent’s past good decisions."  
>	"SIL stores experiences in a replay buffer and learns to imitate state-action pairs in the replay buffer only when the return in the past episode is greater than the agent’s value estimate."  
>	"A theoretical justification of the SIL objective is given by showing that the SIL objective is derived from the lower bound of the optimal Q-function."  
>	"SIL combined with advantage actor-critic is competitive to the state-of-the-art count-based exploration actor-critic methods on several hard exploration Atari games."  
>	"SIL improves the performance of PPO on MuJoCo continuous control tasks, demonstrating that SIL may be generally applicable to any actor-critic architecture."  
>	"Intuitively, A2C updates the policy in the direction of increasing the expected return of the learner policy and enforces consistency between the value and the policy from on-policy trajectories. On the other hand, SIL updates each of them directly towards optimal policies and values respectively from off-policy trajectories."  
  - `video` <https://facebook.com/icml.imls/videos/432572773923910?t=3600> (Oh)
  - `notes` <https://medium.com/intelligentunit/paper-notes-2-self-imitation-learning-b3a0fbdee351>
  - `code` <https://github.com/junhyukoh/self-imitation-learning>

#### ["Smoothed Action Value Functions for Learning Gaussian Policies"](https://arxiv.org/abs/1803.02348) Nachum, Norouzi, Tucker, Schuurmans
  `Smoothie` `smoothed Q-learning` `policy gradient`
>	"A new notion of action value defined by a Gaussian smoothed version of the expected Q-value."  
>	"Smoothed Q-values still satisfy a Bellman equation, making them learnable via function approximation and bootstrapping."  
>	"Gradients of expected reward with respect to the mean and covariance of a parameterized Gaussian policy can be recovered from the gradient and Hessian of the smoothed Q-value function."  
>	"In the spirit of DDPG, trains a policy using the derivatives of a trained smoothed Q-value function to learn a Gaussian policy."  
>	"Unlike DDPG, which is restricted to deterministic policies and is well-known to have poor exploratory behavior, Smoothie is able to utilize a non-deterministic Gaussian policy parameterized by both a mean and a covariance, thus allowing the policy to be exploratory by default and alleviating the need for excessive hyperparameter tuning."  
>	"Unlike DDPG, Smoothie can be adapted to incorporate proximal policy optimization techniques by augmenting the objective with a penalty on KL-divergence from a previous version of the policy."  
>	"Unlike standard policy gradient, Smoothie utilizes derivatives of a Q-value function to train a policy and thus avoids the high variance and sample inefficiency of stochastic updates."  
>	"Q-value at a state s and action a answers the question, “What would my future value from s be if I were to take an initial action a?”. Such information about a hypothetical action is helpful when learning a policy; we want to nudge the policy distribution to favor actions with potentially higher Q-values. We investigate the practicality and benefits of answering a more difficult, but more relevant, question: “What would my future value from s be if I were to sample my initial action from a distribution centered at a?”"  
  - `video` <https://facebook.com/icml.imls/videos/430993334081854?t=5335> (Nachum)

#### ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"](https://arxiv.org/abs/1801.01290) Haarnoja, Zhou, Abbeel, Levine
  `SAC` `soft Q-learning` `policy gradient` `maximum entropy policy` `on-policy + off-policy`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#soft-actor-critic-off-policy-maximum-entropy-deep-reinforcement-learning-with-a-stochastic-actor-haarnoja-zhou-abbeel-levine>

#### ["A Unified View of Entropy-Regularized Markov Decision Processes"](https://arxiv.org/abs/1705.07798) Neu, Gomez, Jonsson
  `soft Q-learning` `policy gradient` `maximum entropy policy`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#a-unified-view-of-entropy-regularized-markov-decision-processes-neu-gomez-jonsson>

#### ["Equivalence Between Policy Gradients and Soft Q-Learning"](https://arxiv.org/abs/1704.06440) Schulman, Chen, Abbeel
  `soft Q-learning` `policy gradient` `maximum entropy policy`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#equivalence-between-policy-gradients-and-soft-q-learning-schulman-chen-abbeel>

#### ["Reinforcement Learning with Deep Energy-Based Policies"](https://arxiv.org/abs/1702.08165) Haarnoja, Tang, Abbeel, Levine
  `SQL` `soft Q-learning` `policy gradient` `maximum entropy policy`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#reinforcement-learning-with-deep-energy-based-policies-haarnoja-tang-abbeel-levine>

#### ["Bridging the Gap Between Value and Policy Reinforcement Learning"](http://arxiv.org/abs/1702.08892) Nachum, Norouzi, Xu, Schuurmans
  `PCL` `policy gradient` `on-policy + off-policy`
>	"Softmax temporally consistent action values satisfy a strong consistency property with optimal entropy regularized policy probabilities along any action sequence, regardless of provenance."  
>	"Path Consistency Learning minimizes inconsistency measured along multi-step action sequences extracted from both on- and off-policy traces."  
>	"We show how a single model can be used to represent both a policy and its softmax action values. Beyond eliminating the need for a separate critic, the unification demonstrates how policy gradients can be stabilized via self-bootstrapping from both on- and off-policy data. An experimental evaluation demonstrates that both algorithms can significantly outperform strong actor-critic and Q-learning baselines across several benchmark tasks."  
>	"PCL optimizes the upper bound of the mean square consistency Bellman error."  
>	"Entropy-regularized A2C can be viewed as n-step online soft Q-learning or path consistency learning."  
  - `video` <https://youtu.be/fZNyHoXgV7M?t=1h16m17s> (Norouzi)
  - `notes` <https://github.com/ethancaballero/paper-notes/blob/master/Bridging%20the%20Gap%20Between%20Value%20and%20Policy%20Based%20Reinforcement%20Learning.md>
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Bridging_the_Gap_Between_Value_and_Policy_Based_Reinforcement_Learning.md>
  - `code` <https://github.com/tensorflow/models/tree/master/research/pcl_rl>
  - `code` <https://github.com/rarilurelo/pcl_keras>
  - `code` <https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/pcl.py>

#### ["Combining Policy Gradient and Q-learning"](http://arxiv.org/abs/1611.01626) O'Donoghue, Munos, Kavukcuoglu, Mnih
  `PGQL` `Q-learning` `policy gradient`
>	"A connection between the fixed points of the regularized policy gradient algorithm and the Q-values allows us to estimate the Q-values from the action preferences of the policy, to which we apply Q-learning updates."  
>	"We establish an equivalency between action-value fitting techniques and actor-critic algorithms, showing that regularized policy gradient techniques can be interpreted as advantage function learning algorithms."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/ODonoghueMKM16>
  - `code` <https://github.com/Fritz449/Asynchronous-RL-agent>
  - `code` <https://github.com/abhishm/PGQ>

----
#### ["Scalable Trust-region Method for Deep Reinforcement Learning using Kronecker-Factored Approximation"](https://arxiv.org/abs/1708.05144) Wu, Mansimov, Liao, Grosse, Ba
  `ACKTR` `policy gradient` `on-policy`
>	"A2C + K-FAC (Kronecker-Factored Approximate Curvature)"  
>	"2-3x improvement in sample efficiency over A2C with only 10-25% more computing time"  
>	"superior to PPO"  
>
>	"K-FAC is a scalable approximate natural gradient algorithm. Natural gradient is more efficient optimizer because it extracts more information from batches of samples."  
>	"TRPO doesn't scale to modern size neural networks. TRPO approximates natural gradient using conjugate gradient, similar to Hessian-free optimization, is very efficient in terms of number of parameter updates, requires expensive iterative procedure for each update due and only uses curvature information from current batch."  
>	"Core idea: approximate the I^th Fisher matrix blocks using a Kronecker product of two small matrices."  
>	"K-FAC for simple networks is equivalent to gradient descent where activation and backprop values are whitened."  
>
>	"For optimizing actor network, follows natural policy gradient, using policy distribution to define Fisher Matrix."  
>	"For optimizing critic network, assumes Gaussian distribution over output. This is equivalent to Gauss-Newton metric."  
>
>	"Applying K-FAC to A2C:  
>	- Fisher metric for actor network (same as in prior work)  
>	- Gauss-Newton metric for critic network (i.e. Euclidean metric on values)  
>	- rescale updates using trust region method, analogously to TRPO  
>	- approximate the KL using the Fisher metric  
>	- using moving average to accumulate Fisher statistics, in order to get a better estimate of the curvature"  
  - `video` <https://youtube.com/watch?v=0rrffaYuUi4> (Wu)
  - `video` <https://facebook.com/nipsfoundation/videos/1554654864625747?t=2703> (Wu)
  - `video` <https://youtu.be/eeJ1-bUnwRI?t=1h54m20s> (Sigaud)
  - `video` <https://youtu.be/xvRrgxcpaHY?t=34m54s> (Schulman)
  - `video` ["Optimizing Neural Networks using Structured Probabilistic Models of the Gradient Computation"](https://fields.utoronto.ca/video-archive/2017/02/2267-16498) (Grosse)
  - `video` ["Optimizing NN using Kronecker-factored Approximate Curvature"](https://youtube.com/watch?v=FLV-MLPt3sU) (Kropotov)
  - `slides` <https://csc2541-f17.github.io/slides/lec10.pdf#page=55> (Grosse)
  - `post` <https://medium.com/@yaroslavvb/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0>
  - `code` <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr>

#### ["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347) Schulman, Wolski, Dhariwal, Radford, Klimov
  `PPO` `policy gradient` `on-policy`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#proximal-policy-optimization-algorithms-schulman-wolski-dhariwal-radford-klimov>

#### ["Emergence of Locomotion Behaviours in Rich Environments"](https://arxiv.org/abs/1707.02286) Heess et al.
  `DPPO` `policy gradient` `on-policy`
>	"Force an agent to autonomously be creative by exposing him to ever changing environments and the will of going forward - agent's speed as a single, pure reward function."  
>	"parallelized Proximal Policy Optimization"  
  - `video` <https://youtube.com/watch?v=hx_bgoTF7bs> (demo)
  - `video` <https://youtu.be/ZX3l2whplz8?t=33m3s> (Riedmiller)
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
  - `video` <https://youtube.com/watch?v=8jKC95KklT0> (Karazeev) `in russian`
  - `post` <http://inference.vc/evolutionary-strategies-embarrassingly-parallelizable-optimization/> (Huszar)
  - `post` <http://inference.vc/evolution-strategies-variational-optimisation-and-natural-es-2/> (Huszar)
  - `post` <http://davidbarber.github.io/blog/2017/04/03/variational-optimisation/> (Barber)
  - `post` <http://argmin.net/2017/04/03/evolution/> (Recht)
  - `paper` ["Parameter-exploring Policy Gradients"](https://mediatum.ub.tum.de/doc/1287490/409330.pdf) by Sehnke et al.
  - `paper` ["Random Gradient-Free Minimization of Convex Functions"](https://mipt.ru/dcam/students/elective/a_5gc1te/RandomGradFree.PDF) by Nesterov
  - `code` <https://github.com/openai/evolution-strategies-starter>
  - `codd` <https://github.com/atgambardella/pytorch-es>

----
#### ["Maximum a Posteriori Policy Optimisation"](https://arxiv.org/abs/1806.06920) Abdolmaleki, Springenberg, Tassa, Munos, Heess, Riedmiller
  `MPO` `policy gradient` `on-policy + off-policy`
>	"To derive our algorithm, we take advantage of the duality between control and estimation by using Expectation Maximization, a powerful tool from the probabilistic estimation toolbox, in order to solve control problems. This duality can be understood as replacing the question “what are the actions which maximise future rewards?” with the question “assuming future success in maximising rewards, what are the actions most likely to have been taken?”. By using this estimation objective we have more control over the policy change in both E and M steps yielding robust learning. We show that several algorithms, including TRPO, can be directly related to this perspective."  
>	"We leverage the fast convergence properties of EM-style coordinate ascent by alternating a non-parametric data-based E-step which re-weights state-action samples, with a supervised, parametric M-step using deep neural networks. This process is stable enough to allow us to use full covariance matrices, rather than just diagonal, in our policies."  
>	"The derivation of our algorithm starts from the classical connection between RL and probabilistic inference. Rather than estimating a single optimal trajectory methods in this space attempt to identify a distribution of plausible solutions by trading off expected return against the (relative) entropy of this distribution. Building on classical work such as Dayan & Hinton (1997) we cast policy search as a particular instance of the class of Expectation Maximization algorithms. Our algorithm then combines properties of existing approaches in this family with properties of recent off-policy algorithms for neural networks."  
>	"Specifically we propose an alternating optimization scheme that leads us to a novel, off-policy algorithm that is (a) data efficient; (b) robust and effectively hyper-parameter free; (c) applicable to complex control problems that have so far been considered outside the realm of off-policy algorithms; (d) allows for effective parallelisation."  
>	"Our algorithm separates policy learning into two alternating phases which we refer to as E and M step in reference to the EM algorithm:  
>	- E-step: In the E-step we obtain an estimate of the distribution of return-weighted trajectories. We perform this step by re-weighting state action samples using a learned value function. This is akin to posterior inference step when estimating the parameters of a probabilistic models with latent variables.  
>	- M-step: In the M-step we update the parametric policy in a supervised learning step using the reweighted state-action samples from the E-step as targets. This corresponds to the update of the model parameters given the complete data log-likelihood when performing EM for a probabilistic model."  
>	"These choices lead to the following desirable properties: (a) low-variance estimates of the expected return via function approximation; (b) low-sample complexity of value function estimate via robust off-policy learning; (c) minimal parametric assumption about the form of the trajectory distribution in the E-step; (d) policy updates via supervised learning in the M step; (e) robust updates via hard trust-region constraints in both the E and the M step."  
  - `video` <https://youtube.com/watch?v=he_BPw32PwU>
  - `video` <http://dropbox.com/s/pgcmjst7t0zwm4y/MPO.mp4> + <https://vimeo.com/240200982> (demo)
  - `video` <https://youtu.be/ZX3l2whplz8?t=18m7s> (Riedmiller)

#### ["Addressing Function Approximation Error in Actor-Critic Methods"](https://arxiv.org/abs/1802.09477) Fujimoto, Hoof, Meger
  `TD3` `policy gradient` `on-policy + off-policy`
>	"In Q-learning function approximation errors lead to overestimated value estimates and suboptimal policies. We show that this problem persists in an actor-critic setting and propose novel mechanisms to minimize its effects on both the actor and the critic. Our algorithm builds on Double Q-learning, by taking the minimum value between a pair of critics to limit overestimation. We draw the connection between target networks and overestimation bias, and suggest delaying policy updates to reduce per-update error and further improve performance."  
  - `video` <https://facebook.com/icml.imls/videos/430993334081854?t=7107> (Fujimoto)
  - `video` <https://youtu.be/eeJ1-bUnwRI?t=1h22m44s> (Sigaud)

#### ["IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"](https://arxiv.org/abs/1802.01561) Espeholt, Soyer, Munos, Simonyan, Mnih, Ward, Doron, Firoiu, Harley, Dunning, Legg, Kavukcuoglu
  `IMPALA` `V-Trace` `policy gradient` `on-policy + off-policy` `multi-task`
>	"Authors achieve stable learning at high throughput by combining decoupled acting and learning with a novel off-policy correction method called V-trace, which was critical for achieving learning stability."  
>	"Synchronous batch learning is more robust to hyperparameters than asynchronous SGD."  
>	"Deep ResNets finally outperform 3 layer ConvNets on DMLab-30 - Atari was too simple."  
  - `post` <https://deepmind.com/blog/impala-scalable-distributed-deeprl-dmlab-30/>
  - `video` <https://youtube.com/playlist?list=PLqYmG7hTraZDRA9vW0zV8iIHlHnBSTBcC> (demo)
  - `video` <http://fields.utoronto.ca/video-archive/2018/01/2509-18003> (Mnih)
  - `video` <https://facebook.com/icml.imls/videos/432150780632776?t=1458> (Espeholt)
  - `video` <https://facebook.com/iclr.cc/videos/2125495797479475?t=1265> (Kavukcuoglu)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1802.01561>
  - `code` <https://github.com/deepmind/scalable_agent>

#### ["The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously"](https://arxiv.org/abs/1707.03300) Cabi, Colmenarejo, Hoffman, Denil, Wang, de Freitas
  `IUA` `policy gradient` `on-policy + off-policy` `multi-task`
>	"We hypothesize that a single stream of experience offers agents the opportunity to learn and perfect many policies both on purpose and incidentally, thus accelerating the acquisition of grounded knowledge. To investigate this hypothesis, we propose a deep actor-critic architecture, trained with DDPG, for learning several policies concurrently. The architecture enables the agent to attend to one task on-policy, while unintentionally learning to solve many other tasks off-policy. Importantly, the policies learned unintentionally can be harnessed for intentional use even if those policies were never followed before."  
>	"More precisely, this intentional-unintentional architecture consists of two neural networks. The actor neural network has multiple-heads representing different policies with shared lower-level representations. The critic network represents several state-action value functions, sharing a common representation for the observations."  
>	"Our experiments demonstrate that when acting according to the policy associated with one of the hardest tasks, we are able to learn all other tasks off-policy. The results for the playroom domain also showed that by increasing the number of tasks, all actors and critics learn faster. In fact, in some settings, learning with many goals was essential to solve hard many-body control tasks."  
  - `video` <https://youtu.be/xfyK03MEZ9Q?t=7h20m39s> (Cabi)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/#t=1240> (de Freitas)

#### ["Trust-PCL: An Off-Policy Trust Region Method for Continuous Control"](https://arxiv.org/abs/1707.01891) Nachum, Norouzi, Xu, Schuurmans
  `Trust-PCL` `policy gradient` `on-policy + off-policy`
>	"off-policy TRPO"  
>	"Optimal relative-entropy regularized policy satisfies path consistencies relating state values at ends of path to log-probabilities of actions along path. Trust-PCL implicitly optimizes trust region constraint using off-policy data by minimizing inconsistencies with maintained lagged policy along paths sampled from replay buffer."  
  - `video` <https://youtu.be/xfyK03MEZ9Q?t=7h38m11s> (Nachum)
  - `code` <https://github.com/tensorflow/models/tree/master/research/pcl_rl>

#### ["Expected Policy Gradients"](https://arxiv.org/abs/1706.05374) Ciosek, Whiteson
  `EPG` `policy gradient` `on-policy + off-policy`
>	"EPG unify stochastic policy gradients (SPG) and deterministic policy gradients (DPG) for reinforcement learning. Inspired by expected SARSA, EPG integrates across the action when estimating the gradient, instead of relying only on the action selected during the sampled trajectory. We establish a new general policy gradient theorem, of which the stochastic and deterministic policy gradient theorems are special cases."  
>	"We also prove that EPG reduces the variance of the gradient estimates without requiring deterministic policies and, for the Gaussian case, with no computational overhead. When the policy is Gaussian, we can now reinterpret deterministic policy gradients as an on-policy method: the deterministic policy of the original formulation is just the result of analytically integrating across actions in the Gaussian policy."  
>	"Both SPG and DPG approaches have significant shortcomings. For SPG, variance in the gradient estimates means that many trajectories are usually needed for learning. Since gathering trajectories is typically expensive, there is a great need for more sample efficient methods. DPG’s use of deterministic policies mitigates the problem of variance in the gradient but raises other difficulties. The theoretical support for DPG is limited since it assumes a critic that approximates ∇aQ when in practice it approximates Q instead. In addition, DPG learns off-policy, which is undesirable when we want learning to take the cost of exploration into account. More importantly, learning off-policy necessitates designing a suitable exploration policy, which is difficult in practice. In fact, efficient exploration in DPG is an open problem and most applications simply use independent Gaussian noise or the Ornstein-Uhlenbeck heuristic."  
>	"EPG also enables a practical contribution. Under certain conditions, we get an analytical expression for the covariance of the Gaussian that leads to a principled directed exploration strategy for continuous problems. We show that it is optimal in a certain sense to explore with a Gaussian policy such that the covariance is proportional to exp(H), where H is the scaled Hessian of the critic with respect to the actions. We present empirical results confirming that this new approach to exploration substantially outperforms DPG with Ornstein-Uhlenbeck exploration in four challenging MuJoCo domains."  
>	"EPG learns a stochastic policy but for a Gaussian policy is equivalent to DPG with a specific form of exploration, implying that it'd be fine to use it deterministically once trained."  
  - `video` <https://youtube.com/watch?v=x2NFiP6cuXI> (Ciosek)

#### ["Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic"](http://arxiv.org/abs/1611.02247) Gu, Lillicrap, Ghahramani, Turner, Levine
  `Q-Prop` `policy gradient` `on-policy + off-policy`
>	"Batch policy gradient methods offer stable learning, but at the cost of high variance, which often requires large batches. TD-style methods, such as off-policy actor-critic and Q-learning, are more sample-efficient but biased, and often require costly hyperparameter sweeps to stabilize. In this work, we aim to develop methods that combine the stability of policy gradients with the efficiency of off-policy RL."  
>	"A policy gradient method that uses a Taylor expansion of the off-policy critic as a control variate. Q-Prop is both sample efficient and stable, and effectively combines the benefits of on-policy and off-policy methods."  
>	"  
>	- unbiased gradient  
>	- combine PG and AC gradients  
>	- learns critic from off-policy data  
>	- learns policy from on-policy data  
>	"  
>	"Q-Prop works with smaller batch size than TRPO-GAE"  
>	"Q-Prop is significantly more sample-efficient than TRPO-GAE"  
>	"policy gradient algorithm that is as fast as value estimation"  
>	"take off-policy algorithm and correct it with on-policy algorithm on residuals"  
>	"can be understood as REINFORCE with state-action-dependent baseline with bias correction term instead of unbiased state-dependent baseline"  
  - `video` <https://facebook.com/iclr.cc/videos/1712224178806641?t=5807> (Gu)
  - `video` <https://youtu.be/M6nfipCxQBc?t=16m11s> (Lillicrap)
  - `video` <https://youtu.be/ggPGtMSoVN8?t=8m28s> (Petrenko) `in russian`
  - `notes` <http://www.alexirpan.com/rl-derivations/#q-prop>
  - `code` <https://github.com/shaneshixiang/rllabplusplus>
  - `paper` ["The Mirage of Action-Dependent Baselines in Reinforcement Learning"](https://arxiv.org/abs/1802.10031) by Tucker et al. ([talk](https://facebook.com/icml.imls/videos/430993334081854?t=4087) by Tucker `video`)

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
  - `video` <https://youtu.be/ggPGtMSoVN8?t=1h9m45s> (Petrenko) `in russian`
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2FWangBHMMKF16>
  - `code` <https://github.com/openai/baselines/tree/master/baselines/acer>
  - `code` <https://github.com/higgsfield/RL-Adventure-2>
  - `code` <https://github.com/hercky/ACER_tf>
  - `code` <https://github.com/Kaixhin/ACER>

----
#### ["RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning"](http://arxiv.org/abs/1611.02779) Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel
  `RL^2` `meta-learning`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#rl2-fast-reinforcement-learning-via-slow-reinforcement-learning-duan-schulman-chen-bartlett-sutskever-abbeel>

#### ["Learning to Reinforcement Learn"](http://arxiv.org/abs/1611.05763) Wang et al.
  `meta-learning`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#learning-to-reinforcement-learn-wang-et-al>



---
### reinforcement learning - model-based methods

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---model-based-methods)

----
#### ["Unsupervised Predictive Memory in Goal-Directed Agent"](https://arxiv.org/abs/1803.10760) Wayne et al.
  `partial observability` `MERLIN`
>	"We demonstrate that contemporary RL algorithms struggle to solve simple tasks when enough information is concealed from the sensors of the agent, a property called "partial observability". An obvious requirement for handling partially observed tasks is access to extensive memory, but we show memory is not enough; it is critical that the right information be stored in the right format. We develop a model, the Memory, RL, and Inference Network, in which memory formation is guided by a process of predictive modeling."  
  - `video` <https://youtu.be/9z3_tJAu7MQ?t=22m45s> (Wayne)
  - `video` <https://slideslive.com/38915863/learning-models-for-representations-and-planning> (7:00) (Lillicrap)
  - `video` <https://youtu.be/aV4wz7FAXmo?t=1h18m26s> (Shvechikov)

#### ["Neural Map: Structured Memory for Deep Reinforcement Learning"](https://arxiv.org/abs/1702.08360) Parisotto, Salakhutdinov
  `partial observability`
>	"Spatially structured 2D memory to learn to store arbitrary information about the environment over long time lags."  
>	"Memory was given a 2D structure in order to resemble a spatial map to address specific problems such as 2D or 3D navigation."  
>	"Size and computational cost doesn't grow with time horizon of environment."  
  - `video` <https://youtube.com/watch?v=cUW99V5x7fE> (Salakhutdinov)
  - `video` <https://vimeo.com/252185932> (Salakhutdinov)
  - `video` <https://youtu.be/x_kK4Pc4qow?t=18m3s> (Salakhutdinov)
  - `video` <https://youtu.be/bTWlFiF4Kns?t=6m55s> (Salakhutdinov)
  - `video` <https://yadi.sk/i/pMdw-_uI3Gke7Z> (Shvechikov) `in russian`

----
#### ["Composable Planning with Attributes"](https://arxiv.org/abs/1803.00512) Zhang, Lerer, Sukhbaatar, Fergus, Szlam
  `planning` `using available environment model`
>	"The tasks that an agent will need to solve often are not known during training. However, if the agent knows which properties of the environment are important then, after learning how its actions affect those properties, it may be able to use this knowledge to solve complex tasks without training specifically for them. Towards this end, we consider a setup in which an environment is augmented with a set of user defined attributes that parameterize the features of interest. We propose a method that learns a policy for transitioning between “nearby” sets of attributes, and maintains a graph of possible transitions. Given a task at test time that can be expressed in terms of a target set of attributes, and a current state, our model infers the attributes of the current state and searches over paths through attribute space to get a high level plan, and then uses its low level policy to execute the plan. We show in 3D block stacking, gridworld games, and StarCraft that our model is able to generalize to longer, more complex tasks at test time by composing simpler learned policies."  
  - `video` <https://youtube.com/watch?v=r7oSmy_AtZY> (Szlam)
  - `video` <https://vimeo.com/287812698> (Zhang, Lerer)

#### ["Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"](https://arxiv.org/abs/1708.02596) Nagabandi, Kahn, Fearing, Levine
  `planning` `using available environment model`
>	"Calculating optimal plan is difficult due to the dynamics and reward functions being nonlinear, but many techniques exist for obtaining approximate solutions to finite-horizon control problems that are sufficient for succeeding at the desired task. We use a simple random-sampling shooting method in which K candidate action sequences are randomly generated, the corresponding state sequences are predicted using the learned dynamics model, the rewards for all sequences are calculated, and the candidate action sequence with the highest expected cumulative reward is chosen. Rather than have the policy execute this action sequence in open-loop, we use model predictive control: the policy executes only the first action, receives updated state information, and recalculates the optimal action sequence at the next time step. This combination of predictive dynamics model plus controller is beneficial in that the model is trained only once, but by simply changing the reward function, we can accomplish a variety of goals at run-time, without a need for live task-specific retraining."  
  - <https://sites.google.com/view/mbmf> (demo)
  - `post` <http://bair.berkeley.edu/blog/2017/11/30/model-based-rl/>
  - `video` <https://youtube.com/watch?v=G7lXiuEC8x0>
  - `video` <https://vimeo.com/252186751> (Nagabandi)
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
#### ["Understanding and Generalizing AlphaGo Zero"](https://openreview.net/forum?id=rkxtl3C5YX)
  `learning from planning` `using available environment model` `expert iteration`
>	"We provide a formal framework to study AGZ. We argue that the self-play in AGZ corresponds to learning a Nash equilibrium for the two-player game; and the supervised learning with MCTS is attempting to learn the policy corresponding to the Nash equilibrium, by establishing a novel bound on the difference between the expected return achieved by two policies in terms of the expected KL divergence (cross-entropy) of their induced distributions."  
>	"To extend AGZ to generic sequential decision-making problems, we introduce a robust MDP framework, in which the agent and nature effectively play a zero-sum game: the agent aims to take actions to maximize reward while nature seeks state transitions, subject to the constraints of that environment, that minimize the agent’s reward. For a challenging network scheduling domain, we find that AGZ within the robust MDP framework provides near-optimal performance, matching one of the best known scheduling policies that has taken the networking community three decades of intensive research to develop."  
>	"The questions. We are interested in three key unanswered questions about AGZ:  
>	Question 1. What is the optimal policy that AGZ is trying to learn? We would like a formal model to understand how learning via self-play impacts the policy to which AGZ converges.  
>	Question 2. Why is minimizing cross-entropy (or KL divergence) the right objective? We seek to precisely characterize the distribution under which cross-entropy should be evaluated, and derive quantitative performance guarantees that explain what optimizing for this metric achieves.  
>	Question 3. How does AGZ extend to generic sequential decision-making problems? In generic sequential decision-making, the agent chooses actions and earns rewards while the state evolves per the environment. This is different from two-player zero-sum games for which AGZ was designed."  

#### ["Dual Policy Iteration"](https://arxiv.org/abs/1805.10755) Sun, Gordon, Boots, Bagnell
  `learning from planning` `using available environment model` `expert iteration`
>	"Recently, a novel class of Approximate Policy Iteration algorithms such as ExIt and AlphaGo-Zero have demonstrated impressive practical performance. This new family of algorithms maintains, and alternately optimizes, two policies: a fast, reactive policy (e.g., a deep neural network) deployed at test time, and a slow, non-reactive policy (e.g., Tree Search), that can plan multiple steps ahead. The reactive policy is updated under supervision from the non-reactive policy, while the non-reactive policy is improved with guidance from the reactive policy. In this work we study this Dual Policy Iteration strategy in an alternating optimization framework and provide a convergence analysis that extends existing API theory."  
>	"We also develop a special instance of this framework which reduces the update of non-reactive policies to model-based optimal control using learned local models, and provides a theoretically sound way of unifying model-free and model-based RL approaches with unknown dynamics. We demonstrate the efficacy of our approach on various continuous control Markov Decision Processes."  

#### ["ExpIt-OOS: Towards Learning from Planning in Imperfect Information Games"](https://arxiv.org/abs/1808.10120) Kitchen, Benedetti
  `learning from planning` `using available environment model` `expert iteration` `ExpIt-OOS`
>	"A novel approach to playing imperfect information games within the Expert Iteration framework and inspired by AlphaZero. We use Online Outcome Sampling, an online search algorithm for imperfect information games in place of MCTS. While training online, our neural strategy is used to improve the accuracy of playouts in OOS, allowing a learning and planning feedback loop for imperfect information games."  

#### ["Thinking Fast and Slow with Deep Learning and Tree Search"](https://arxiv.org/abs/1705.08439) Anthony, Tian, Barber
  `learning from planning` `using available environment model` `expert iteration` `ExIt`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#thinking-fast-and-slow-with-deep-learning-and-tree-search-anthony-tian-barber>

#### ["Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"](https://arxiv.org/abs/1712.01815) Silver et al.
  `learning from planning` `using available environment model` `expert iteration`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#mastering-chess-and-shogi-by-self-play-with-a-general-reinforcement-learning-algorithm-silver-et-al>

#### ["Mastering the Game of Go without Human Knowledge"](https://deepmind.com/documents/119/agz_unformatted_nature.pdf) Silver et al.
  `learning from planning` `using available environment model` `expert iteration`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#mastering-the-game-of-go-without-human-knowledge-silver-et-al>

#### ["Deep Counterfactual Regret Minimization"](https://arxiv.org/abs/1811.00164) Brown, Lerer, Gross, Sandholm

#### ["DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker"](http://arxiv.org/abs/1701.01724) Moravcik et al.
  `learning from planning` `using available environment model`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#deepstack-expert-level-artificial-intelligence-in-no-limit-poker-moravcik-et-al>

#### ["Learning Generalized Reactive Policies using Deep Neural Networks"](https://arxiv.org/abs/1708.07280) Groshev, Tamar, Srivastava, Abbeel
  `learning from planning` `using available environment model`
>	"learning a reactive policy that imitates execution traces produced by a planner"  
>	"We consider the problem of learning for planning, where knowledge acquired while planning is reused to plan faster in new problem instances. For robotic tasks, among others, plan execution can be captured as a sequence of visual images."  
>	"We investigate architectural properties of deep networks that are suitable for learning long-horizon planning behavior, and explore how to learn, in addition to the policy, a heuristic function that can be used with classical planners or search algorithms such as A*."  
>	"Results on the challenging Sokoban domain suggest that DNNs have the capability to extract powerful features from observations, and the potential to learn the type of ‘visual thinking’ that makes some planning problems easy to humans but very hard for automatic planners."  
  - <https://sites.google.com/site/learn2plannips/> (demo)

----
#### ["Learning to Search with MCTSnets"](https://arxiv.org/abs/1802.04697) Guez, Weber, Antonoglou, Simonyan, Vinyals, Wierstra, Munos, Silver
  `learning to plan` `using available environment model`
>	"Planning problems are most typically solved by tree search algorithms that simulate ahead into the future, evaluate future states, and back-up those evaluations to the root of a search tree. Among these algorithms, Monte-Carlo tree search is one of the most general, powerful and widely used. A typical implementation of MCTS uses cleverly designed rules, optimised to the particular characteristics of the domain. These rules control where the simulation traverses, what to evaluate in the states that are reached, and how to back-up those evaluations. In this paper we instead learn where, what and how to search. Our architecture, which we call an MCTSnet, incorporates simulation-based search inside a neural network, by expanding, evaluating and backing-up a vector embedding. The parameters of the network are trained end-to-end using gradient-based optimisation. When applied to small searches in the well-known planning problem Sokoban, the learned search algorithm significantly outperformed MCTS baselines."  
>	"Although we have focused on a supervised learning setup, our approach could easily be extended to a reinforcement learning setup by leveraging policy iteration with MCTS. We have focused on small searches, more similar in scale to the plans that are processed by the human brain, than to the massive-scale searches in high-performance games or planning applications. In fact, our learned search performed better than a standard MCTS with more than an order-of-magnitude more computation, suggesting that neural approaches to search may ultimately replace their handcrafted counterparts."  
>	"We present a neural network architecture that includes the same processing stages as a typical MCTS, but inside the neural network itself, as a dynamic computational graph. The key idea is to represent the internal state of the search, at each node, by a memory vector. The computation of the network proceeds forwards from the root state, just like a simulation of MCTS, using a simulation policy based on the memory vector to select the trajectory to traverse. The leaf state is then processed by an embedding network to initialize the memory vector at the leaf. The network proceeds backwards up the trajectory, updating the memory at each visited state according to a backup network that propagates from child to parent. Finally, the root memory vector is used to compute an overall prediction of value or action."  
>	"The major benefit of our planning architecture, compared to more traditional planning algorithms, is that it can be exposed to gradient-based optimisation. This allows us to replace every component of MCTS with a richer, learnable equivalent - while maintaining the desirable structural properties of MCTS such as the use of a model, iterative local computations, and structured memory. We jointly train the parameters of the evaluation network, backup network and simulation policy so as to optimise the overall predictions of the MCTS network. The majority of the network is fully differentiable, allowing for efficient training by gradient descent. Still, internal action sequences directing the control flow of the network cannot be differentiated, and learning this internal policy presents a challenging credit assignment problem. To address this, we propose a novel, generally applicable approximate scheme for credit assignment that leverages the anytime property of our computational graph, allowing us to also effectively learn this part of the search network from data."  
  - `video` <https://drive.google.com/drive/folders/0B8hmSuYkl6xuVE1Id25kZ2swUVU> (demo)
  - `video` <https://facebook.com/icml.imls/videos/429607650887089?t=1421> (Weber)

#### ["Learning and Querying Fast Generative Models for Reinforcement Learning"](https://arxiv.org/abs/1802.03006) Buesing, Weber, Racaniere, Eslami, Rezende, Reichert, Viola, Besse, Gregor, Hassabis, Wierstra
  `learning to plan` `using available environment model` `I2A`
>	"We have shown that state-space models directly learned from raw pixel observations are good candidates for model-based RL: 1) they are powerful enough to capture complex environment dynamics, exhibiting similar accuracy to frame-auto-regressive models; 2) they allow for computationally efficient Monte-Carlo rollouts; 3) their learned dynamic state-representations are excellent features for evaluating and anticipating future outcomes compared to raw pixels. This enabled Imagination Augemented Agents to outperform strong model-free baselines on MS PACMAN."  
>	"On a conceptual level, we present (to the best of our knowledge) the first results on what we termed learning-to-query. We show learning a rollout policy by backpropagating policy gradients leads to consistent (if modest) improvements."  
>	"We address, the Imagination-Augmented Agent framework, the main challenge posed by model-based RL: training accurate, computationally efficient models on more complex domains and using them with agents. First, we consider computationally efficient state-space environment models that make predictions at a higher level of abstraction, both spatially and temporally, than at the level of raw pixel observations. Such models substantially reduce the amount of computation required to perform rollouts, as future states can be represented much more compactly. Second, in order to increase model accuracy, we examine the benefits of explicitly modeling uncertainty in the transitions between these abstract states. Finally, we explore different strategies of learning rollout policies that define the interface between agent and environment model: We consider the possibility of learning to query the internal model, for guiding the Monte-Carlo rollouts of the model towards informative outcomes."  
>	"Here, we adopted the I2A assumption of having access to a pre-trained envronment model. In future work, we plan to drop this assumption and jointly learn the model and the agent."  

#### ["Imagination-Augmented Agents for Deep Reinforcement Learning"](https://arxiv.org/abs/1707.06203) Weber et al.
  `learning to plan` `using available environment model` `I2A`
>	"In contrast to most existing model-based reinforcement learning and planning methods, which prescribe how a model should be used to arrive at a policy, I2As learn to interpret predictions from an imperfect learned environment model to construct implicit plans in arbitrary ways, by using predictions as additional context in deep policy networks."  
>	"I2A's policy and value functions are informed by the outputs of two separate pathways: 1) a model-free path, that tries to estimate the value and which action to take directly from the latest observation ot using a CNN; and 2) a model-based path, designed in the following way. The I2A is endowed with a pretrained, fixed environment model. At every time, conditioned on past observations and actions, it uses the model to simulate possible futures (Monte-Carlo rollouts) represented by imagnations over some horizon, under a rollout policy. It then extracts informative features from the rollout imaginations, and uses these, together with the results from the model-free path, to compute policy and value functions. It has been shown that I2As are robust to model imperfections: they learn to interpret imaginations produced from the internal models in order to inform decision making as part of standard return maximization."  
  - `post` <https://deepmind.com/blog/agents-imagine-and-plan/>
  - `video` <https://drive.google.com/drive/folders/0B4tKsKnCCZtQY2tTOThucHVxUTQ> (demo)
  - `video` <https://facebook.com/nipsfoundation/videos/1554654864625747?t=1107> (Weber)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=39m27s> (Mnih)
  - `video` <https://youtube.com/watch?v=agXIYMCICcc> (Kilcher)
  - `slides` <https://mltrain.cc/wp-content/uploads/2017/10/sebastien-racaniere.pdf>
  - `code` <https://github.com/vasiloglou/mltrain-nips-2017/blob/master/sebastien_racaniere/I2A%20-%20NIPS%20workshop.ipynb>

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
#### ["Temporal Difference Models: Model-Free Deep RL for Model-Based Control"](https://arxiv.org/abs/1802.09081) Pong, Gu, Dalal, Levine
  `learning to plan` `learning abstract environment model` `TDM`
>	"A family of goal-conditioned value functions that can be trained with model-free learning and used for model-based control. TDMs combine the benefits of model-free and model-based RL: they leverage the rich information in state transitions to learn very efficiently, while still attaining asymptotic performance that exceeds that of direct model-based RL methods."  
  - `post` <https://bair.berkeley.edu/blog/2018/04/26/tdm>
  - `video` <https://youtube.com/watch?v=j-3nUkzMFA8> (Gu)
  - `code` <https://github.com/vitchyr/rlkit>

#### ["Self-supervised Deep Reinforcement Learning with Generalized Computation Graphs for Robot Navigation"](https://arxiv.org/abs/1709.10489) Kahn, Villaflor, Ding, Abbeel, Levine
  `learning to plan` `learning abstract environment model`
>	"generalized computation graph that subsumes value-based model-free methods and model-based methods, with specific instantiations interpolating between model-free and model-based"  
  - `video` <https://youtube.com/watch?v=vgiW0HlQWVE> (demo)
  - `code` <http://github.com/gkahn13/gcg>

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

#### ["Value Prediction Network"](https://arxiv.org/abs/1707.03497) Oh, Singh, Lee
  `learning to plan` `learning abstract environment model` `VPN`
>	"VPN combines model-based RL (i.e., learning the dynamics of an abstract state space sufficient for computing future rewards and values) and model-free RL (i.e., mapping the learned abstract states to rewards and values) in a unified framework. In order to train a VPN, we propose a combination of temporal-difference search (TD search) and n-step Q-learning. In brief, VPNs learn to predict values via Q-learning and rewards via supervised learning. At the same time, VPNs perform lookahead planning to choose actions and compute bootstrapped target Q-values."  
>	"Extends the Predictron model from policy evaluation to optimal control."  
>	"Uses the model to construct a look-ahead tree only when constructing bootstrap targets and selecting actions, similarly to TD-search. Crucially, the model is not embedded in a planning algorithm during optimisation."  
  - `video` <http://videolectures.net/deeplearning2017_singh_reinforcement_learning/#t=4366> (Singh)
  - `video` <https://youtu.be/PRQ8-FwDPRE?t=16m> (Holland)
  - `video` <https://youtu.be/RwLTrQUyDvA?t=14m58s> (Diaz Rodriguez)
  - `notes` <https://medium.com/arxiv-bytes/summary-value-prediction-networks-vpn-474c0b080b2a>

#### ["The Predictron: End-to-End Learning and Planning"](https://arxiv.org/abs/1612.08810) Silver et al.
  `learning to plan` `learning abstract environment model`
>	"The Predictron consists of a fully abstract model, represented by a Markov reward process, that can be rolled forward multiple “imagined" planning steps. Each forward pass of the predictron accumulates internal rewards and values over multiple planning depths. The predictron is trained end-to-end so as to make these accumulated values accurately approximate the true value function."  
>	"trains deep network to implicitly plan via iterative rollouts"  
>	"uses implicit environment model which does not capture dynamics"  
>	"only applied to learning Markov reward processes rather than solving Markov decision processes"  
  - `video` <https://youtube.com/watch?v=BeaLdaN2C3Q> (demo)
  - `video` <https://vimeo.com/238243832>
  - `video` <https://youtube.com/watch?v=ID150Tl-MMw&t=55m9s> (Abbeel)
  - `video` <https://youtu.be/bsuvM1jO-4w?t=38m6s> (Mnih)
  - `video` <http://videolectures.net/deeplearning2017_singh_reinforcement_learning/#t=4366> (Singh)
  - `code` <https://github.com/zhongwen/predictron>
  - `code` <https://github.com/muupan/predictron>

#### ["Cognitive Mapping and Planning for Visual Navigation"](https://arxiv.org/abs/1702.03920) Gupta, Davidson, Levine, Sukthankar, Malik
  `learning to plan` `learning abstract environment model`
>	"1st person mapping + navigation with VIN"  
  - <https://sites.google.com/view/cognitive-mapping-and-planning>
  - `video` <https://youtu.be/ID150Tl-MMw?t=54m24s> (demo)
  - `code` <https://github.com/tensorflow/models/tree/master/research/cognitive_mapping_and_planning>

#### ["Value Iteration Networks"](http://arxiv.org/abs/1602.02867) Tamar, Wu, Thomas, Levine, Abbeel
  `learning to plan` `learning abstract environment model`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#value-iteration-networks-tamar-wu-thomas-levine-abbeel>

----
#### ["Visual Interaction Networks"](https://arxiv.org/abs/1706.01433) Watters, Tacchetti, Weber, Pascanu, Battaglia, Zoran
  `learning to simulate` `learning abstract environment model`
>	"general-purpose model for learning dynamics of physical system from raw visual observations without prior knowledge"  
>	"Model consists of a perceptual front-end based on convolutional neural networks and a dynamics predictor based on interaction networks. Through joint training, the perceptual front-end learns to parse a dynamic visual scene into a set of factored latent object representations. The dynamics predictor learns to roll these states forward in time by computing their interactions and dynamics, producing a predicted physical trajectory of arbitrary length."  
>	"We found that from just six input video frames the Visual Interaction Network can generate accurate future trajectories of hundreds of time steps on a wide range of physical systems. Our model can also be applied to scenes with invisible objects, inferring their future states from their effects on the visible objects, and can implicitly infer the unknown mass of objects."  
>	"The VIN is learnable and can be trained from supervised data sequences which consist of input image frames and target object state values. It can learn to approximate a range of different physical systems which involve interacting entities by implicitly internalizing the rules necessary for simulating their dynamics and interactions."  
  - `video` <https://goo.gl/FD1XX5> + <https://goo.gl/4SSGP0> (demo)
  - `post` <https://deepmind.com/blog/neural-approach-relational-reasoning/>
  - `code` <https://github.com/Mrgemy95/visual-interaction-networks-pytorch>

#### ["Interaction Networks for Learning about Objects, Relations and Physics"](http://arxiv.org/abs/1612.00222) Battaglia, Pascanu, Lai, Rezende, Kavukcuoglu
  `learning to simulate` `learning abstract environment model`
>	"Model which can reason about how objects in complex systems interact, supporting dynamical predictions, as well as inferences about the abstract properties of the system."  
>	"Model takes graphs as input, performs object- and relation-centric reasoning in a way that is analogous to a simulation, and is implemented using deep neural networks."  
>	"Our results show it can be trained to accurately simulate the physical trajectories of dozens of objects over thousands of time steps, estimate abstract quantities such as energy, and generalize automatically to systems with different numbers and configurations of objects and relations."  
>	"Our interaction network implementation is the first general-purpose, learnable physics engine, and a powerful general framework for reasoning about object and relations in a wide variety of complex real-world domains."  
>	"The interaction network may also serve as a powerful model for model-predictive control inputting active control signals as external effects – because it is differentiable, it naturally supports gradient-based planning."  
>	"Graph based framework for dynamic systems which is able to simulate the physical trajectories of n-body, bouncing ball, and non-rigid string systems accurately over thousands of time steps, after training only on single step predictions."  
  - `video` <https://youtube.com/watch?v=zsvYr5tyj9M> (Erzat) `in russian`
  - `notes` <https://blog.acolyer.org/2017/01/02/interaction-networks-for-learning-about-objects-relations-and-physics/>
  - `code` <https://github.com/jaesik817/Interaction-networks_tensorflow>
  - `code` <https://github.com/higgsfield/interaction_network_pytorch>

----
#### ["Model Based Reinforcement Learning for Atari"](https://arxiv.org/abs/1903.00374) Kaiser et al.
  `learning to simulate` `learning environment model` `video prediction` `SimPLe`
>	"Simulated Policy Learning, a complete model-based deep RL algorithm based on video prediction models."  
>	"SimPLe consists of alternating between learning a model, and then using this model to optimize a policy by using model-free reinforcement learning within the model. Variants of this basic algorithm have been proposed in a number of prior works, starting from Dyna."  
>	"SimPLe can learn to play many of the games with just 100K transitions, corresponding to 2 hours of play time. In many cases, the number of samples required for prior methods to learn to reach the same reward value is several times larger."  
>	"The final scores are on the whole substantially lower than the best state-of-the-art model-free methods. This is generally common with model-based RL algorithms, which excel more in learning efficiency rather than final performance."  
>	"The performance of our method generally varied substantially between different runs on the same game. The complex interactions between the model, policy, and data collection were likely responsible for this: at a fundamental level, the model makes guesses when it extrapolates the behavior of the game under a new policy. When these guesses are correct, the resulting policy performs well in the final game. In future work, models that capture uncertainty via Bayesian parameter posteriors or ensembles may further improve robustness."  
  - <https://sites.google.com/view/modelbasedrlatari>
  - `notes` <https://medium.com/arxiv-bytes/summary-simple-ae74ae934c4a>

#### ["Learning Latent Dynamics for Planning from Pixels"](https://arxiv.org/abs/1811.04551) Hafner et al.
  `learning to simulate` `learning environment model` `video prediction` `PlaNet`
>	"Deep Planning Network is a purely model-based agent that learns the environment dynamics from images and chooses actions through fast online planning in latent space. To achieve high performance, the dynamics model must accurately predict the rewards ahead for multiple time steps. We approach this using a latent dynamics model with both deterministic and stochastic transition components. Moreover, we  propose  a multi-step variational inference objective that we name latent overshooting. Using only pixel observations, our agent solves continuous control tasks with contact dynamics, partial observability, and sparse rewards, which exceed the difficulty of tasks that were previously solved by planning with learned models. PlaNet uses substantially fewer episodes and reaches final performance close to and sometimes higher than strong model-free algorithms."  
  - <https://planetrl.github.io>
  - `post` <https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html>
  - `video` <https://youtube.com/watch?v=C7Dmu0GtrSw>
  - `video` <https://slideslive.com/38915863/learning-models-for-representations-and-planning> (14:20) (Lillicrap)
  - `code` <https://github.com/google-research/planet>

#### ["Stochastic Variational Video Prediction"](https://arxiv.org/abs/1710.11252) Babaeizadeh, Finn, Erhan, Campbell, Levine
  `learning to simulate` `learning environment model` `video prediction`
>	"Real-world events can be stochastic and unpredictable, and the high dimensionality and complexity of natural images requires the predictive model to build an intricate understanding of the natural world. Many existing methods tackle this problem by making simplifying assumptions about the environment. One common assumption is that the outcome is deterministic and there is only one plausible future. This can lead to low-quality predictions in real-world settings with stochastic dynamics."  
>	"SV2P predicts a different possible future for each sample of its latent variables. Our model is the first to provide effective stochastic multi-frame prediction for real-world video. We demonstrate the capability of the proposed method in predicting detailed future frames of videos on multiple real-world datasets, both action-free and action-conditioned."  
  - <https://sites.google.com/site/stochasticvideoprediction/> (demo)

#### ["Self-Supervised Visual Planning with Temporal Skip Connections"](https://arxiv.org/abs/1710.05268) Ebert, Finn, Lee, Levine
  `learning to simulate` `learning environment model` `video prediction`
  - <https://sites.google.com/view/sna-visual-mpc> (demo)
  - `video` <https://youtube.com/watch?v=6k7GHG4IUCY>
  - `video` <https://youtu.be/UDLI9K6b9G8?t=1h14m56s> (Ebert)
  - `code` <https://github.com/febert/visual_mpc>

#### ["Recurrent Environment Simulators"](https://arxiv.org/abs/1704.02254) Chiappa, Racaniere, Wierstra, Mohamed
  `learning to simulate` `learning environment model` `video prediction`
>	"We improve on previous environment simulators from high-dimensional pixel observations by introducing recurrent neural networks that are able to make temporally and spatially coherent predictions for hundreds of time-steps into the future."  
>	"We address the issue of computationally inefficiency with a model that does not need to generate a high-dimensional image at each time-step."  
>	"It is a deterministic model designed for deterministic environments. Clearly most real world environments involve noisy state transitions."  
  - `video` <https://sites.google.com/site/resvideos1729> (demo)
  - `code` <https://github.com/KokoMind/Recurrent-Environment-Simulators>

----
#### ["Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models"](https://arxiv.org/abs/1805.12114) Chua, Calandra, McAllister, Levine
  `learning to simulate` `learning environment model` `PETS` `NIPS 2018`
>	"Model-based reinforcement learning algorithms can attain excellent sample efficiency, but often lag behind the best model-free algorithms in terms of asymptotic performance. This is especially true with high-capacity parametric function approximators, such as deep networks. In this paper, we study how to bridge this gap, by employing uncertainty-aware dynamics models."  
>	"Probabilistic ensembles with trajectory sampling algorithm combines uncertainty-aware deep network dynamics models with sampling-based uncertainty propagation. Our comparison to state-of-the-art model-based and model-free deep RL algorithms shows that our approach matches the asymptotic performance of model-free algorithms on several challenging benchmark tasks, while requiring significantly fewer samples (e.g., 8 and 125 times fewer samples than Soft Actor Critic and Proximal Policy Optimization respectively on the half-cheetah task)."  
>	"dynamics model + trajectory propagation + planning via Model Predictive Control"  
  - `video` <https://sites.google.com/view/drl-in-a-handful-of-trials>
  - `video` <https://slideslive.com/38915863/learning-models-for-representations-and-planning> (12:17) (Lillicrap)

#### ["Model-Ensemble Trust-Region Policy Optimization"](https://arxiv.org/abs/1802.10592) Kurutach, Clavera, Duan, Tamar, Abbeel
  `learning to simulate` `learning environment model` `ME-TRPO` `ICLR 2018`
>	"We analyze the behavior of vanilla model-based reinforcement learning methods when deep neural networks are used to learn both the model and the policy, and show that the learned policy tends to exploit regions where insufficient data is available for the model to be learned, causing instability in training. To overcome this issue, we propose to use an ensemble of models to maintain the model uncertainty and regularize the learning process. We further show that the use of likelihood ratio derivatives yields much more stable learning than backpropagation through time. Altogether, our approach Model-Ensemble Trust-Region Policy Optimization significantly reduces the sample complexity compared to model-free deep RL methods on challenging continuous control benchmark tasks."  
>	"The dynamics model maintains uncertainty due to limited data through an ensemble of models. The algorithm alternates among adding transitions to a replay buffer, optimizing the dynamics models given the buffer, and optimizing the policy given the dynamics models in Dyna's style. This algorithm significantly helps alleviating the model bias problem in model-based RL, when the policy exploits the error in the dynamics model. In many Mujoco domains, we show that it can achieve the same final performance as model-free approaches while using 100x less data."  
  - `code` <https://github.com/thanard/me-trpo>

#### ["Learning Multimodal Transition Dynamics for Model-Based Reinforcement Learning"](https://arxiv.org/abs/1705.00470) Moerland, Broekens, Jonker
  `learning to simulate` `learning environment model`
>	"So why is model-based RL not the standard approach? Model-based RL consists of two steps: 1) transition function estimation through supervised learning, and 2) (sample-based) planning over the learned model. Each step has a particular challenging aspect. For this work we focus on a key challenge of the first step: stochasticity in the transition dynamics. Stochasticity is an inherent property of many environments, and increases in real-world settings due to sensor noise. Transition dynamics usually combine both deterministic aspects (such as the falling trajectory of an object due to gravity) and stochastic elements (such as the behaviour of another car on the road). Our goal is to learn to jointly predict these. Note that stochasticity has many forms, both homoscedastic versus heteroscedastic, and unimodal versus multimodal. In this work we specifically focus on multimodal stochasticity, as this should theoretically pose the largest challenge."  
>	"We focus on deep generative models as they can approximate complex distributions and scale to high-dimensional domains. For model-based RL we have additional requirements, as we are ultimately interested in using the model for sample-based planning. This usually requires sampling a lot of traces, so we require models that are 1) easy to sample from, 2) ideally allow planning at an abstract level. Implicit density models, like Generative Adverserial Networks lack a clear probabilistic objective function, which was the focus of this work. Among the explicit density models, there are two categories. Change of variable formula models, like Real NVP, have the drawback that the latent space dimension must equal the observation space. Fully visible belief nets like pixelCNN, which factorize the likelihood in an auto-regressive fashion, hold state-of-the-art likelihood results. However, they have the drawback that sampling is a sequential operation (e.g. pixel-by-pixel, which is computationally expensive), and they do not allow for latent level planning either. Therefore, most suitable for model-based RL seem approximate density models, most noteworthy the Variational Auto-Encoder framework. These models can estimate stochasticity at a latent level, allow for latent planning, are easy to sample from, and have a clear probabilistic interpretation."  
>	"An important challenge is planning under uncertainty. RL initially provides correlated data from a limited part of state-space. When planning over this model, we should not extrapolate too much, nor trust our model to early with limited data. Note that ‘uncertainty’ (due to limited data) is fundamentally different from the ‘stochasticity’ (true probabilistic nature of the domain) discussed in this paper."  
  - `code` <http://github.com/tmoer/multimodal_varinf>

#### ["Prediction and Control with Temporal Segment Models"](https://arxiv.org/abs/1703.04070) Mishra, Abbeel, Mordatch
  `learning to simulate` `learning environment model`
>	"We learn the distribution over future state trajectories conditioned on past state, past action, and planned future action trajectories, as well as a latent prior over action trajectories. Our approach is based on convolutional autoregressive models and variational autoencoders. It makes stable and accurate predictions over long horizons for complex, stochastic systems, effectively expressing uncertainty and modeling the effects of collisions, sensory noise, and action delays."  
  - `video` <https://vimeo.com/237267784> (Mishra)

#### ["Learning and Policy Search in Stochastic Dynamic Systems with Bayesian Neural Networks"](https://arxiv.org/abs/1605.07127) Depeweg, Hernandez-Lobato, Doshi-Velez, Udluft
  `learning to simulate` `learning environment model`
>	"Monte-Carlo model-based policy gradient technique in continuous stochastic systems"  
>	"Proposed approach enables automatic identification of arbitrary stochastic patterns such as multimodality and heteroskedasticity, without having to manually incorporate these into the model."  
>	"We have extended Bayesian neural network with addition of a random input noise source z. This enables principled Bayesian inference over complex stochastic functions. We have also presented an algorithm that uses random roll-outs and stochastic optimization for learning a parameterized policy in a batch scenario. Our BNNs with random inputs have allowed us to solve a challenging benchmark problem where model-based approaches usually fail."  
>	"For safety, we believe having uncertainty over the underlying stochastic functions will allow us to optimize policies by focusing on worst case results instead of on average performance. For exploration, having uncertainty on the stochastic functions will be useful for efficient data collection."  
>	"The optimal policy can be significantly affected by the noise present in the state transitions. This is illustrated by the drunken spider story, in which a spider has two possible paths to go home: either by crossing the bridge or by walking around the lake. In the absence of noise, the bridge option is prefered since it is shorter. However, after heavily drinking alcohol, the spider’s movements may randomly deviate left or right. Since the bridge is narrow, and spiders do not like swimming, the prefered trajectory is now to walk around the lake. The previous example shows how noise can significantly affect optimal control. For example, the optimal policy may change depending on whether the level of noise is high or low. Therefore, we expect to obtain significant improvements in model-based reinforcement learning by capturing with high accuracy any noise patterns present in the state transition data."  
  - `post` <https://medium.com/towards-data-science/bayesian-neural-networks-with-random-inputs-for-model-based-reinforcement-learning-36606a9399b4> (Hernandez-Lobato)
  - `video` <https://youtube.com/watch?v=0H3EkUPENSY> (Hernandez-Lobato)
  - `video` <https://youtube.com/watch?v=J4KLWjZ1QVM> (Hernandez-Lobato)
  - `slides` <https://yadi.sk/i/NCrM4cEb3NhQux>
  - `code` <https://github.com/siemens/policy_search_bb-alpha>



---
### reinforcement learning - exploration and intrinsic motivation

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---exploration-and-intrinsic-motivation)  
[**interesting older papers - artificial curiosity and creativity**](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#interesting-papers---artificial-curiosity-and-creativity)  

[papers](https://sites.google.com/view/erl-2018/accepted-papers) from ICML 2018 workshop

----
#### ["A Contextual Bandit Bake-off"](https://arxiv.org/abs/1802.04064) Bietti, Agarwal, Langford
>	"We leverage the availability of large numbers of supervised learning datasets to compare and empirically optimize contextual bandit algorithms, focusing on practical methods that learn by relying on optimization oracles from supervised learning. We find that a recent method (Foster et al., 2018) using optimism under uncertainty works the best overall. A surprisingly close second is a simple greedy baseline that only explores implicitly through the diversity of contexts, followed by a variant of Online Cover (Agarwal et al., 2014) which tends to be more conservative but robust to problem specification by design."  
  - `video` <https://youtu.be/zr6H4kR8vTg?t=50m36s> (Langford)

#### ["Contextual Decision Processes with Low Bellman Rank are PAC-Learnable"](https://arxiv.org/abs/1610.09512) Jiang, Krishnamurthy, Agarwal, Langford, Schapire
  `provably correct and sample efficient exploration`
>	"This paper studies systematic exploration for reinforcement learning with rich observations and function approximation. We introduce a new model called contextual decision processes, that unifies and generalizes most prior settings. Our first contribution is a complexity measure, the Bellman rank, that we show enables tractable learning of near-optimal behavior in these processes and is naturally small for many well-studied reinforcement learning settings. Our second contribution is a new reinforcement learning algorithm that engages in systematic exploration to learn contextual decision processes with low Bellman rank. Our algorithm provably learns near-optimal behavior with a number of samples that is polynomial in all relevant parameters but independent of the number of unique observations. The approach uses Bellman error minimization with optimistic exploration and provides new insights into efficient exploration for reinforcement learning with function approximation."  
>	"Approximation of value function with function from some class is a powerful practical approach with implicit assumption that true value function is approximately in class.  
>	Even with this assumption:  
>	- no guarantee methods will work  
>	- no bound on how much data needed  
>	- no theory on how to explore in large spaces"  
  - `video` <https://vimeo.com/238228755> (Jiang)
  - `video` <https://youtube.com/watch?v=VBkUmD5Em2k> (Agarwal)
  - `video` <https://youtube.com/watch?v=L5Q4Y3omnrY> (Agarwal)
  - `video` <https://vimeo.com/235929810> (Schapire)

----
#### ["Unsupervised Control through Non-Parametric Discriminative Rewards"](https://arxiv.org/abs/1811.11359) Warde-Farley, Wiele, Kulkarni, Ionescu, Hansen, Mnih
  `learning reward function` `intrinsic motivation` `DISCERN`
>	"Learning to control an environment without hand-crafted rewards or expert data remains challenging and is at the frontier of reinforcement learning research. We present an unsupervised learning algorithm to train agents to achieve perceptually specified goals using only a stream of observations and actions. Our agent simultaneously learns a goal-conditioned policy and a goal achievement reward function that measures how similar a state is to the goal state. This dual optimization leads to a co-operative game, giving rise to a learned reward function that reflects similarity in controllable aspects of the environment instead of distance in the space of observations. We demonstrate the efficacy of our agent to learn, in an unsupervised manner, to reach a diverse set of goals on three domains – Atari, the DeepMind Control Suite and DeepMind Lab."  

#### ["On Learning Intrinsic Rewards for Policy Gradient Methods"](https://arxiv.org/abs/1804.06459) Zheng, Oh, Singh
  `learning reward function` `intrinsic motivation`
>	"Optimal Rewards Framework defines the optimal intrinsic reward function as one that when used by an agent achieves behavior that optimizes the task-specifying or extrinsic reward function. Previous work in this framework has shown how good intrinsic reward functions can be learned for lookahead search based planning agents. Whether it is possible to learn intrinsic reward functions for learning agents remains an open problem. In this paper we derive a novel algorithm for learning intrinsic rewards for policy-gradient based learning agents."  
  - `video` <https://youtu.be/_4oL3DDCwCw?t=38m> (Singh)

#### ["Deep Learning for Reward Design to Improve Monte Carlo Tree Search in ATARI Games"](https://arxiv.org/abs/1604.07095) Guo, Singh, Lewis, Lee
  `learning reward function` `intrinsic motivation`
>	"Monte Carlo Tree Search methods have proven powerful in planning for sequential decision-making problems such as Go and video games, but their performance can be poor when the planning depth and sampling trajectories are limited or when the rewards are sparse. We present an adaptation of PGRD (policy-gradient for reward design) for learning a reward-bonus function to improve UCT (a MCTS algorithm). Unlike previous applications of PGRD in which the space of reward-bonus functions was limited to linear functions of hand-coded state-action-features, we use PGRD with a multi-layer convolutional neural network to automatically learn features from raw perception as well as to adapt the non-linear reward-bonus function parameters. We also adopt a variance-reducing gradient method to improve PGRD’s performance. The new method improves UCT’s performance on multiple ATARI games compared to UCT without the reward bonus. Combining PGRD and Deep Learning in this way should make adapting rewards for MCTS algorithms far more widely and practically applicable than before."  
  - `video` <https://youtu.be/_4oL3DDCwCw?t=11m47s> (Singh)
  - `video` <https://vimeo.com/250399421> (Singh)
  - `video` <http://videolectures.net/deeplearning2017_singh_reinforcement_learning/#t=1177> (Singh)
  - `video` <https://youtube.com/watch?v=MhIP1SOqlS8> (Singh)
  - `paper` ["Where Do Rewards Come From?"](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.151.8250) by Singh, Lewis, Barto

#### ["Evolved Policy Gradients"](https://arxiv.org/abs/1802.04821) Houthooft, Chen, Isola, Stadie, Wolski, Ho, Abbeel
  `learning loss function` `intrinsic motivation` `EPG`
>	"Method evolves a differentiable loss function, such that an agent, which optimizes its policy to minimize this loss, will achieve high rewards. The loss is parametrized via temporal convolutions over the agent’s experience. Because this loss is highly flexible in its ability to take into account the agent’s history, it enables fast task learning and eliminates the need for reward shaping at test time. At test time, the learner optimizes only its learned loss function, and requires no explicit reward signal."  
>	"Method is capable of learning a loss function over thousands of sequential environmental actions. Crucially, this learned loss is both highly adaptive (allowing for quicker learning of new tasks) and highly instructive (sometimes eliminating the need for environmental rewards at test time)."  
>	"Our loss’ instructive nature – which allows it to operate at test time without environmental rewards – is interesting and desirable. This instructive nature can be understood as the loss function’s internalization of the reward structures it has previously encountered under the training task distribution. We see this internalization as a step toward learning intrinsic motivation. A good intrinsically motivated agent would successfully infer useful actions in new situations by using heuristics it developed over its entire lifetime. This ability is likely required to achieve truly intelligent agents."  
>	"In addition to internalizing environment rewards, learned loss could, in principle, have several other positive effects. For example, by examining the agent’s history, the loss could incentivize desirable extended behaviors, such as exploration. Further, the loss could perform a form of system identification, inferring environment parameters and adapting how it guides the agent as a function of these parameters (e.g., by adjusting the effective learning rate of the agent)."  
  - `post` <https://blog.openai.com/evolved-policy-gradients>
  - `video` <https://youtu.be/JX5E0Tt7K10?t=11m20s> (Sutskever)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1802.04821>
  - `code` <https://github.com/openai/EPG>

----
#### ["Meta-Reinforcement Learning of Structured Exploration Strategies"](https://arxiv.org/abs/1802.07245) Gupta, Mendonca, Liu, Abbeel, Levine
  `learning to explore` `MAESN`
>	"Many of the current exploration methods for deep RL use task-agnostic objectives, such as information gain or bonuses based on state visitation. However, many practical applications of RL involve learning more than a single task, and prior tasks can be used to inform how exploration should be performed in new tasks. In this work, we explore how prior tasks can inform an agent about how to explore effectively in new situations. We introduce a novel gradient-based fast adaptation algorithm -- model agnostic exploration with structured noise -- to learn exploration strategies from prior experience. The prior experience is used both to initialize a policy and to acquire a latent exploration space that can inject structured stochasticity into a policy, producing exploration strategies that are informed by prior knowledge and are more effective than random action-space noise."  
  - `video` <https://youtube.com/watch?v=Tge7LPT9vGA> (Gupta)

----
#### ["Randomized Prior Functions for Deep Reinforcement Learning"](https://arxiv.org/abs/1806.03335) Osband, Aslanides, Cassirer
  `approximate bayesian exploration` `approximate posterior sampling`
>	"A simple modification where each member of the ensemble is initialized together with a random but fixed prior function. Predictions are then taken as the sum of the trainable neural network and the prior function. We show that this approach passes a sanity check by demonstrating an equivalence to Bayesian inference with linear models. We also present a series of simple experiments designed to extend this intuition to deep learning. We show that many of the most popular approaches for uncertainty estimation in deep RL do not pass these sanity checks, and crystallize these shortcomings in a series of lemmas and small examples. We demonstrate that our simple modification can facilitate aspiration in difficult tasks where previous approaches for deep RL fail. We believe that this work presents a simple and practical approach to encoding prior knowledge with deep reinforcement learning."  
  - <https://sites.google.com/view/randomized-prior-nips-2018> (demo)

#### ["Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling"](https://arxiv.org/abs/1802.09127) Riquelme, Tucker, Snoek
  `approximate bayesian exploration` `posterior sampling`
>	"empirical comparison of bayesian deep networks for Thompson sampling"  
>	"We study an algorithm, which we call NeuralLinear, that is remarkably simple, and combines two classic ideas (NNs and Bayesian linear regression). In our evaluation, NeuralLinear performs well across datasets. Our insight is that, once the learned representation is of decent quality, being able to exactly compute the posterior in closed form with something as simple as a linear model already leads to better decisions than most of the other methods. NeuralLinear is based on a standard deep neural network. However, decisions are made according to a Bayesian linear regression applied to the features at the last layer of the network. Note that the last hidden layer representation determines the final output of the network via a linear function, so we can expect a representation that explains the expected value of an action with a linear model. For all the training contexts, their deep representation is computed, and then uncertainty estimates on linear parameters for each action are derived via standard formulas. Thompson sampling will sample from this distribution, say \beta_t,i at time t for action i, and the next context will be pushed through the network until the last layer, leading to its representation c_t. Then, the sampled beta’s will predict an expected value, and the action with the highest prediction will be taken. Importantly, the algorithm does not use any uncertainty estimates on the representation itself (as opposed to variational methods, for example). On the other hand, the way the algorithm handles uncertainty conditional on the representation and the linear assumption is exact, which seems to be key to its success."  
>	"Variational approaches to estimate uncertainty in neural networks are an active area of research, however, there is no study that systematically benchmarks variational approaches in decision-making scenarios. We find that Bayes by Backprop underperforms even with a linear model. We demonstrate that because the method is simultaneously learning the representation and the uncertainty level. When faced with a limited optimization budget (for online learning), slow convergence becomes a serious concern. In particular, when the fitted model is linear, we evaluate the performance of a mean field model which we can solve in closed form for the variational objective. We find that as we increase number of training iterations for BBB, it slowly converges to the performance of this exact method. This is not a problem in the supervised learning setting, where we can train until convergence. Unfortunately, in the online learning setting, this is problematic, as we cannot train for an unreasonable number of iterations at each step, so poor uncertainty estimates lead to bad decisions. Additionally, tricks to speed up convergence of BBB, such as initializing the variance parameters to a small value, distort uncertainty estimates and thus are not applicable in the online decision making setting. We believe that these insights into the problems with variational approaches are of value to the community, and highlight the need for new ways to estimate uncertainty for online scenarios (i.e., without requiring great computational power)."  
>	"An interesting observation is that in many cases the stochasticity induced by stochastic gradient descent is enough to perform an implicit Thompson sampling. The greedy approach sometimes suffices (or conversely is equally bad as approximate inference). However, we also proposed the wheel problem, where the need for exploration is smoothly parameterized. In this case, we see that all greedy approaches fail."  

#### ["Efficient Exploration through Bayesian Deep Q-Networks"](https://arxiv.org/abs/1802.04412) Azizzadenesheli, Brunskill, Anandkumar
  `approximate bayesian exploration` `approximate posterior sampling`

#### ["The Uncertainty Bellman Equation and Exploration"](https://arxiv.org/abs/1709.05380) O'Donoghue, Osband, Munos, Mnih
  `approximate bayesian exploration` `approximate posterior sampling`
>	"We consider uncertainty Bellman equation which connects the uncertainty at any time-step to the expected uncertainties at subsequent time-steps, thereby extending the potential exploratory benefit of a policy beyond individual time-steps. We prove that the unique fixed point of the UBE yields an upper bound on the variance of the estimated value of any fixed policy. This bound can be much tighter than traditional count-based bonuses that compound standard deviation rather than variance. Importantly, and unlike several existing approaches to optimism, this method scales naturally to large systems with complex generalization."  
>	"Posterior over Q values provides efficient exploration but has complicated distribution in general. Bayesian Central Limit Theorem allows to approximate posterior with Normal distribution. One just needs first and second moments."  
  - `video` <https://facebook.com/icml.imls/videos/432572773923910?t=7326> (O'Donoghue)

#### ["Noisy Networks for Exploration"](https://arxiv.org/abs/1706.10295) Fortunato, Azar, Piot, Menick, Osband, Graves, Mnih, Munos, Hassabis, Pietquin, Blundell, Legg
  `approximate bayesian exploration` `approximate posterior sampling` `NoisyNet`
>	"scale of perturbation to parameters is learned along with original objective function"  
  - `video` <https://youtu.be/fevMOp5TDQs?t=1h27s> (Mnih)
  - `video` <https://youtu.be/fnwo3GCmyEo?t=49m46s> (Fritzler) `in russian`
  - `code` <https://github.com/higgsfield/RL-Adventure>
  - `code` <https://github.com/Kaixhin/NoisyNet-A3C>
  - `code` <https://github.com/andrewliao11/NoisyNet-DQN>

#### ["Parameter Space Noise for Exploration"](https://arxiv.org/abs/1706.01905) Plappert, Houthooft, Dhariwal, Sidor, Chen, Chen, Asfour, Abbeel, Andrychowicz
  `approximate bayesian exploration` `approximate posterior sampling`
>	"Deep reinforcement learning methods generally engage in exploratory behavior through noise injection in the action space. An alternative is to add noise directly to the agent's parameters, which can lead to more consistent exploration and a richer set of behaviors. Methods such as evolutionary strategies use parameter perturbations, but discard all temporal structure in the process and require significantly more samples. Combining parameter noise with traditional RL methods allows to combine the best of both worlds."  
>	"We demonstrate that both off- and on-policy methods benefit from this approach through experimental comparison of DQN, DDPG, and TRPO on high-dimensional discrete action environments as well as continuous control tasks. Our results show that RL with parameter noise learns more efficiently than traditional RL with action space noise and evolutionary strategies individually."  
>	"The training updates for the network are unchanged, but when selecting actions, the network weights are perturbed with isotropic Gaussian noise. Crucially, the network uses layer normalization, which ensures that all weights are on the same scale."  
  - `video` <https://vimeo.com/252185862> (Dhariwal)
  - `post` <https://blog.openai.com/better-exploration-with-parameter-noise/>

#### ["Deep Exploration via Randomized Value Functions"](https://arxiv.org/abs/1703.07608) Osband, Russo, Wen, van Roy
  `approximate bayesian exploration` `approximate posterior sampling`
>	"A very recent thread of work builds on count-based (or upper-confidence-bound-based) exploration schemes that operate with value function learning. These methods maintain a density over the state-action space of pseudo-counts, which represent the quantity of data gathered that is relevant to each state-action pair. Such algorithms may offer a viable approach to deep exploration with generalization. There are, however, some potential drawbacks. One is that a separate representation is required to generalize counts, and it's not clear how to design an effective approach to this. As opposed to the optimal value function, which is fixed by the environment, counts are generated by the agent’s choices, so there is no single target function to learn. Second, the count model generates reward bonuses that distort data used to fit the value function, so the value function representation needs to be designed to not only capture properties of the true optimal value function but also such distorted versions. Finally, these approaches treat uncertainties as uncoupled across state-action pairs, and this can incur a substantial negative impact on statistical efficiency."  
  - `video` <https://youtube.com/watch?v=lfQEPWj97jk> (Osband)
  - `video` <http://techtalks.tv/talks/generalization-and-exploration-via-randomized-value-functions/62467/> (Osband)
  - `video` <https://youtu.be/ck4GixLs4ZQ?t=33m7s> (Osband)
  - `video` <https://vimeo.com/252186381> (van Roy)

#### ["Why is Posterior Sampling Better than Optimism for Reinforcement Learning?"](http://arxiv.org/abs/1607.00215) Osband, van Roy
  `approximate bayesian exploration`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#why-is-posterior-sampling-better-than-optimism-for-reinforcement-learning-osband-van-roy>

#### ["BBQ-Networks: Efficient Exploration in Deep Reinforcement Learning for Task-Oriented Dialogue Systems"](https://arxiv.org/abs/1608.05081) Lipton, Li, Gao, Li, Ahmed, Deng
  `approximate bayesian exploration` `approximate posterior sampling`
  - `video` <https://youtube.com/watch?v=DmasOMKbczg> (Lipton)

#### ["UCB and InfoGain Exploration via Q-Ensembles"](https://arxiv.org/abs/1706.01502) Chen, Sidor, Abbeel, Schulman
  `approximate bayesian exploration` `information gain`

#### ["Exploration Potential"](http://arxiv.org/abs/1609.04994) Leike
  `approximate bayesian exploration` `information gain`
>	"We introduce exploration potential, a quantity that measures how much a reinforcement learning agent has explored its environment class. In contrast to information gain, exploration potential takes the problem's reward structure into account. This leads to an exploration criterion that is both necessary and sufficient for asymptotic optimality (learning to act optimally across the entire environment class). Our experiments in multi-armed bandits use exploration potential to illustrate how different algorithms make the tradeoff between exploration and exploitation."  

----
#### ["Count-Based Exploration with the Successor Representation"](https://arxiv.org/abs/1807.11622) Machado, Bellemare, Bowling
  `exploration guided by observation novelty`
>	"While the traditional successor representation is a representation that defines state generalization by the similarity of successor states, the substochastic successor representation is also able to implicitly count the number of times each state (or feature) has been observed. This extension connects two until now disjoint areas of research."  
  - `video` <https://youtube.com/watch?v=bp3l8BNrefk> (Machado)
  - `video` <https://youtu.be/Tge7LPT9vGA?t=9m36s> (Machado)

#### ["Count-Based Exploration with Neural Density Models"](http://arxiv.org/abs/1703.01310) Ostrovski, Bellemare, Oord, Munos
  `exploration guided by observation novelty` `Reactor-PixelCNN`
>	"PixelCNN for exploration, neural alternative to Context Tree Switching"  
  - `video` <http://youtube.com/watch?v=qSfd27AgcEk> (Bellemare)
  - `video` <https://vimeo.com/238243932> (Bellemare)
  - `video` <http://videolectures.net/DLRLsummerschool2018_bellemare_deep_RL/#t=3474> (Bellemare)

#### ["EX2: Exploration with Exemplar Models for Deep Reinforcement Learning"](https://arxiv.org/abs/1703.01260) Fu, Co-Reyes, Levine
  `exploration guided by observation novelty`
>	"Many of the most effective exploration techniques rely on tabular representations, or on the ability to construct a generative model over states and actions. This paper introduces a novel approach, EX2, which approximates state visitation densities by training an ensemble of discriminators, and assigns reward bonuses to rarely visited states."  
  - <https://sites.google.com/view/ex2exploration> (demo)
  - `video` <https://facebook.com/nipsfoundation/videos/1554741347950432?t=4515> (Fu)
  - `code` <https://github.com/jcoreyes/ex2>
  - `code` <https://github.com/justinjfu/exemplar_models>

#### ["#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning"](http://arxiv.org/abs/1611.04717) Tang, Houthooft, Foote, Stooke, Chen, Duan, Schulman, Turck, Abbeel
  `exploration guided by observation novelty` `SimHash`
>	"The authors encourage exploration by adding a pseudo-reward of the form beta/sqrt(count(state)) for infrequently visited states. State visits are counted using Locality Sensitive Hashing (LSH) based on an environment-specific feature representation like raw pixels or autoencoder representations. The authors show that this simple technique achieves gains in various classic RL control tasks and several games in the ATARI domain. While the algorithm itself is simple there are now several more hyperaprameters to tune: The bonus coefficient beta, the LSH hashing granularity (how many bits to use for hashing) as well as the type of feature representation based on which the hash is computed, which itself may have more parameters. The experiments don't paint a consistent picture and different environments seem to need vastly different hyperparameter settings, which in my opinion will make this technique difficult to use in practice."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1611.04717>
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/%23Exploration:_A_Study_of_Count-Based_Exploration_for_Deep_Reinforcement_Learning.md>

----
#### ["Empowerment-driven Exploration using Mutual Information Estimation"](https://arxiv.org/abs/1810.05533) Kumar
  `exploration guided by empowerment`
>	"using Mutual Information Neural Estimation to calculate empowerment"  
  - `post` <https://navneet-nmk.github.io/2018-08-26-empowerment>
  - `code` <https://github.com/navneet-nmk/pytorch-rl/blob/master/models/empowerment_models.py>

#### ["Unsupervised Real-Time Control through Variational Empowerment"](https://arxiv.org/abs/1710.05101) Karl, Soelch, Becker-Ehmck, Benbouzid, Smagt, Bayer
  `exploration guided by empowerment`

#### ["Variational Intrinsic Control"](http://arxiv.org/abs/1611.07507) Gregor, Rezende, Wierstra
  `exploration guided by empowerment`
>	"The second scenario is that in which the long-term goal of the agent is to get to a state with a maximal set of available intrinsic options – the objective of empowerment (Salge et al., 2014). This set of options consists of those that the agent knows how to use. Note that this is not the theoretical set of all options: it is of no use to the agent that it is possible to do something if it is unable to learn how  to do it. Thus, to maximize empowerment, the agent needs to simultaneously learn how to control the environment as well – it needs to discover the options available to it. The agent should in fact not aim for states where it has the most control according to its current abilities, but for states where it expects it will achieve the most control after learning. Being able to learn available options is thus fundamental to  becoming empowered."  
>	"Let us compare this to the commonly used intrinsic motivation objective of maximizing the amount of model-learning progress, measured as the difference in compression of its experience before and after learning (Schmidhuber, 1991; 2010; Bellemare et al., 2016; Houthooft et al., 2016). The empowerment objective differs from this in a fundamental manner: the primary goal is not to understand or predict the observations but to control the environment. This is an important point – agents can often control an environment perfectly well without much understanding, as exemplified by canonical model-free reinforcement learning algorithms (Sutton & Barto, 1998), where agents only model action-conditioned expected returns. Focusing on such understanding might significantly distract and impair the agent, as such reducing the control it achieves."  

----
#### ["Exploration by Random Network Distillation"](https://arxiv.org/abs/1810.12894) Burda, Edwards, Storkey, Klimov
  `exploration guided by prediction error` `RND`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#exploration-by-random-network-distillation-burda-edwards-storkey-klimov>

#### ["Large-Scale Study of Curiosity-Driven Learning"](https://arxiv.org/abs/1808.04355) Burda, Edwards, Pathak, Storkey, Darrell, Efros
  `exploration guided by prediction error`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#large-scale-study-of-curiosity-driven-learning-burda-edwards-pathak-storkey-darrell-efros>

#### ["Learning to Play with Intrinsically-Motivated Self-Aware Agents"](https://arxiv.org/abs/1802.07442) Haber, Mrowca, Fei-Fei, Yamins
  `exploration guided by prediction error`

#### ["Curiosity-driven Reinforcement Learning with Homeostatic Regulation"](https://arxiv.org/abs/1801.07440) Abril, Kanai
  `exploration guided by prediction error`
>	"Authors extend existing approach by compensating the heterostacity drive encouraged by the curiosity reward with an additional homeostatic drive. The first component implements the heterostatic motivation (same as in Pathak et al 17). It refers to the tendency to push away agent from its habitual state. The second component implements the homeostatic motivation. It encourages taking actions at that lead to future states st+1 where the corresponding future action at+1 gives additional information about st+1."  
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1801.07440>

#### ["Curiosity-driven Exploration by Self-supervised Prediction"](https://arxiv.org/abs/1705.05363) Pathak, Agrawal, Efros, Darrell
  `exploration guided by prediction error`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#curiosity-driven-exploration-by-self-supervised-prediction-pathak-agrawal-efros-darrell>

#### ["Surprise-Based Intrinsic Motivation for Deep Reinforcement Learning"](http://arxiv.org/abs/1703.01732) Achiam, Sastry
  `exploration guided by prediction error`
>	"Authors present two tractable approximations to their framework - one which ignores the stochasticity of the true environmental dynamics, and one which approximates the rate of information gain (somewhat similar to Schmidhuber's formal theory of creativity, fun and intrinsic motivation)."  
>	"Stadie et al. learn deterministic dynamics models by minimizing Euclidean loss - whereas in our work, we learn stochastic dynamics with cross entropy loss - and use L2 prediction errors for intrinsic motivation."  
>	"Our results suggest that surprisal is a viable alternative to VIME in terms of performance, and is highly favorable in terms of computational cost. In VIME, a backwards pass through the dynamics model must be computed for every transition tuple separately to compute the intrinsic rewards, whereas our surprisal bonus only requires forward passes through the dynamics model for intrinsic reward computation. Furthermore, our dynamics model is substantially simpler than the Bayesian neural network dynamics model of VIME. In our speed test, our bonus had a per-iteration speedup of a factor of 3 over VIME."  

#### ["Learning to Perform Physics Experiments via Deep Reinforcement Learning"](http://arxiv.org/abs/1611.01843) Denil, Agrawal, Kulkarni, Erez, Battaglia, de Freitas
  `exploration guided by prediction error`
>	"By letting our agents conduct physical experiments in an interactive simulated environment, they learn to manipulate objects and observe the consequences to infer hidden object properties."  
>	"By systematically manipulating the problem difficulty and the cost incurred by the agent for performing experiments, we found that agents learn different strategies that balance the cost of gathering information against the cost of making mistakes in different situations."  
>	"Exploration bonus can be defined as the prediction error for a problem related to the agent’s transitions. Non-generic prediction problems can be used if specialized information about the environment is available, like predicting physical properties of objects the agent interacts with."  
  - `video` <https://youtu.be/SAcHyzMdbXc?t=16m6s> (de Freitas)

----
#### ["Diversity is All You Need: Learning Skills without a Reward Function"](https://arxiv.org/abs/1802.06070) Eysenbach, Gupta, Ibarz, Levine
  `DIAYN` `exploration guided by additional tasks`
>	"Intelligent creatures can explore their environments and learn useful skills without supervision. In this paper, we propose DIAYN ('Diversity is All You Need'), a method for learning useful skills without a reward function. Our proposed method learns skills by maximizing an information theoretic objective using a maximum entropy policy. On a variety of simulated robotic tasks, we show that this simple objective results in the unsupervised emergence of diverse skills, such as walking and jumping. In a number of reinforcement learning benchmark environments, our method is able to learn a skill that solves the benchmark task despite never receiving the true task reward. We show how pretrained skills can provide a good parameter initialization for downstream tasks, and can be composed hierarchically to solve complex, sparse reward tasks. Our results suggest that unsupervised discovery of skills can serve as an effective pretraining mechanism for overcoming challenges of exploration and data efficiency in reinforcement learning."  

#### ["Learning by Playing - Solving Sparse Reward Tasks from Scratch"](https://arxiv.org/abs/1802.10567) Riedmiller, Hafner, Lampe, Neunert, Degrave, Wiele, Mnih, Heess, Springenberg
  `SAC-X` `exploration guided by additional tasks`
>	"SAC-X simultaneously learns intention policies on a set of auxiliary tasks, and actively schedules and executes these to explore its observation space - in search for sparse rewards of externally defined target tasks. Utilizing simple auxiliary tasks enables SAC-X to learn complicated target tasks from rewards defined in a ’pure’, sparse, manner: only the end goal is specified, but not the solution path."  
>	"It can be interpreted as a generalization of the IUA and UNREAL objectives to stochastic continuous controls – in combination with active execution of auxiliary tasks and (potentially learned) scheduling within an episode."  
>	"It can also be understood as a hierarchical extension of Hindsight Experience Replay, where the agent behaves according to a fixed set of semantically grounded auxiliary tasks – instead of following random goals – and optimizes over the task selection."  
>	"While IUA, UNREAL and HER mainly consider using auxiliary tasks to provide additional learning signals – and additional exploration by following random sensory goals – we here make active use of the auxiliary tasks by switching between them throughout individual episodes (to achieve exploration for the main task)."  
  - `post` <https://deepmind.com/blog/learning-playing>
  - `video` <https://youtu.be/ZX3l2whplz8?t=22m6s> (Riedmiller)
  - `video` <https://facebook.com/icml.imls/videos/429963197518201?t=1401> (Hafner)

#### ["The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously"](https://arxiv.org/abs/1707.03300) Cabi, Colmenarejo, Hoffman, Denil, Wang, de Freitas
  `IUA` `exploration guided by additional tasks`
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#the-intentional-unintentional-agent-learning-to-solve-many-continuous-control-tasks-simultaneously-cabi-colmenarejo-hoffman-denil-wang-de-freitas>

#### ["Reinforcement Learning with Unsupervised Auxiliary Tasks"](http://arxiv.org/abs/1611.05397) Jaderberg, Mnih, Czarnecki, Schaul, Leibo, Silver, Kavukcuoglu
  `UNREAL` `exploration guided by additional tasks`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#reinforcement-learning-with-unsupervised-auxiliary-tasks-jaderberg-mnih-czarnecki-schaul-leibo-silver-kavukcuoglu>

#### ["Learning to Navigate in Complex Environments"](http://arxiv.org/abs/1611.03673) Mirowski, Pascanu, Viola, Soyer, Ballard, Banino, Denil, Goroshin, Sifre, Kavukcuoglu, Kumaran, Hadsell
  `exploration guided by additional tasks`
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
  `exploration guided by additional tasks`

#### ["Feature Control as Intrinsic Motivation for Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1705.06769) Dilokthanakul, Kaplanis, Pawlowski, Shanahan
  `exploration guided by additional tasks`
>	"Extracting reward from features makes sparse-reward environment into dense one. Authors incorporate this strategy to solve hierarchical RL."  
>	"Authors solve Montezuma with Options framework (but without dynamic terminal condition). The agent is encouraged to change pixel or visual-feature given the option from meta-control and this dense feedback makes agent to learn basic skill under high sparsity."  

----
#### ["Overcoming Exploration in Reinforcement Learning with Demonstrations"](https://arxiv.org/abs/1709.10089) Nair, McGrew, Andrychowicz, Zaremba, Abbeel
  `exploration guided by learning progress and demonstrations` `Q-filter`
>	"We use demonstrations to overcome the exploration problem and successfully learn to perform long-horizon, multi-step robotics tasks with continuous control such as stacking blocks with a robot arm."  
>	"Our method, which builds on top of Deep Deterministic Policy Gradients and Hindsight Experience Replay, provides an order of magnitude of speedup over RL on simulated robotics tasks."  
>	"Our method is able to solve tasks not solvable by either RL or behavior cloning alone, and often ends up outperforming the demonstrator policy."  
  - <http://ashvin.me/demoddpg-website/> (demo)
  - `video` <https://vimeo.com/252186855> (Nair)

#### ["Hindsight Experience Replay"](https://arxiv.org/abs/1707.01495) Andrychowicz, Wolski, Ray, Schneider, Fong, Welinder, McGrew, Tobin, Abbeel, Zaremba
  `exploration guided by learning progress` `HER`
>	"Get reward signal from any experience by simply assuming the goal equals whatever happened."  
>	"HER may be seen as a form of implicit curriculum as the goals used for replay naturally shift from ones which are simple to achieve even by a random agent to more difficult ones. However, in contrast to explicit curriculum, HER does not require having any control over the distribution of initial environment states."  
>	"Not only does HER learn with extremely sparse rewards, in our experiments it also performs better with sparse rewards than with shaped ones. These results are indicative of the practical challenges with reward shaping, and that shaped rewards would often constitute a compromise on the metric we truly care about (such as binary success/failure)."  
  - `post` <https://blog.openai.com/ingredients-for-robotics-research/>
  - <https://sites.google.com/site/hindsightexperiencereplay/> (demo)
  - `video` <https://facebook.com/nipsfoundation/videos/1554594181298482?t=2280> (Abbeel)
  - `video` <https://youtu.be/TERCdog1ddE?t=50m45s> (Abbeel)
  - `video` <https://youtu.be/JX5E0Tt7K10?t=3m50s> (Sutskever)
  - `video` <https://youtu.be/BCzFs9Xb9_o?t=21m2s> (Sutskever)
  - `video` <https://youtu.be/RvEwFvl-TrY?t=19m18s> (Sutskever)
  - `video` <https://youtu.be/BXe2A5i4ESw?t=10m42s> (Fournier)
  - `video` <https://youtu.be/0Ey02HT_1Ho?t=10m57s> (Steenbrugge)
  - `notes` <https://yobibyte.github.io/files/paper_notes/her.pdf>
  - `post` <https://jangirrishabh.github.io/2018/03/25/Overcoming-exploration-demos.html>
  - `code` <https://github.com/openai/baselines/tree/master/baselines/her>
  - `code` <https://github.com/vitchyr/rlkit>
  - `code` <https://github.com/higgsfield/RL-Adventure-2>
  - `paper` ["Universal Value Function Approximators"](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#schaul-horgan-gregor-silver---universal-value-function-approximators) by Schaul et al. `summary`

#### ["Reverse Curriculum Generation for Reinforcement Learning"](https://arxiv.org/abs/1707.05300) Florensa, Held, Wulfmeier, Zhang, Abbeel
  `exploration guided by learning progress`
>	"Many tasks require to reach a desired configuration (goal) from everywhere."  
>	"Challenging for current RL: inherently sparse rewards, most start positions get 0 reward."  
>
>	"Solve the task in reverse, first training from positions closer to the goal and then bootstrap this knowledge to solve from further."  
>	"Sample more start states from where you succeed sometimes but not always (for best efficiency)."  
  - `post` <http://bair.berkeley.edu/blog/2017/12/20/reverse-curriculum/>
  - `video` <https://youtu.be/xfyK03MEZ9Q?t=7h32m35s> (Florensa)

#### ["Teacher-Student Curriculum Learning"](https://arxiv.org/abs/1707.00183) Matiisen, Oliver, Cohen, Schulman
  `exploration guided by learning progress`

#### ["Automated Curriculum Learning for Neural Networks"](https://arxiv.org/abs/1704.03003) Graves, Bellemare, Menick, Munos, Kavukcuoglu
  `exploration guided by learning progress`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#automated-curriculum-learning-for-neural-networks-graves-bellemare-menick-munos-kavukcuoglu>

#### ["Automatic Goal Generation for Reinforcement Learning Agents"](https://arxiv.org/abs/1705.06366) Held, Geng, Florensa, Abbeel
  `exploration guided by learning progress`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#automatic-goal-generation-for-reinforcement-learning-agents-held-geng-florensa-abbeel>

#### ["Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play"](http://arxiv.org/abs/1703.05407) Sukhbaatar, Lin, Kostrikov, Synnaeve, Szlam, Fergus
  `exploration guided by learning progress`
>	"self-play between the policy and a task-setter in order to automatically generate goal states which are on the border of what the current policy can achieve"  
>	"A separate policy to find new slightly harder goals. A goal-generating policy may only generate goals in a small region of the goal-space, having difficulties to quickly cover the full set of goals. Inevitable differences in improvement rate of the goal-generating and the goal-learning agents leads to instabilities and local optima."  
  - `video` <https://youtube.com/watch?v=EHHiFwStqaA> (demo)
  - `video` <https://youtube.com/watch?v=X1O21ziUqUY> (Fergus)
  - `video` <https://youtube.com/watch?v=5dNAnCYBFN4> (Szlam)
  - `post` <http://giorgiopatrini.org/posts/2017/09/06/in-search-of-the-missing-signals/>

#### ["Towards Information-Seeking Agents"](http://arxiv.org/abs/1612.02605) Bachman, Sordoni, Trischler
  `exploration guided by learning progress`
  - `video` <https://youtube.com/watch?v=3bSquT1zqj8> (demo)



---
### reinforcement learning - hierarchical

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---hierarchical-reinforcement-learning)

----
#### ["The Termination Critic"](https://arxiv.org/abs/1902.09996) Harutyunyan, Dabney, Borsa, Heess, Munos, Precup
>	"We consider the problem of autonomously discovering behavioral abstractions, or options, for reinforcement learning agents. We propose an algorithm that focuses on the termination condition, as opposed to -- as is common -- the policy. The termination condition is usually trained to optimize a control objective: an option ought to terminate if another has better value. We offer a different, information-theoretic perspective, and propose that terminations should focus instead on the compressibility of the option's encoding -- arguably a key reason for using abstractions. To achieve this algorithmically, we leverage the classical options framework, and learn the option transition model as a "critic" for the termination condition."  
  - `notes` <https://pbs.twimg.com/media/DtxEP5BUUAAVP4h.jpg>

#### ["Human-level Performance in First-person Multiplayer Games with Population-based Deep Reinforcement Learning"](https://arxiv.org/abs/1807.01281) Jaderberg et al.
  `FTW`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#human-level-performance-in-first-person-multiplayer-games-with-population-based-deep-reinforcement-learning-jaderberg-et-al>

#### ["Data-Efficient Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1805.08296) Nachum, Gu, Lee, Levine

#### ["Self-Consistent Trajectory Autoencoder: Hierarchical Reinforcement Learning with Trajectory Embeddings"](https://arxiv.org/abs/1806.02813) Co-Reyes, Liu, Gupta, Eysenbach, Abbeel, Levine
  `SeCTAR`
>	"The problem of learning lower layers in a hierarchy is transformed into the problem of learning trajectory-level generative models."  
>	"Continuous latent representations of trajectories are effective in solving temporally extended and multi-stage problems."  
>	"State decoder learns to model a continuous space of behaviors. Policy decoder learns to realize behaviors in the environment."  
>	"Given the same latent, policy decoder generates a trajectory which should match the trajectory predicted by state decoder."  
>	"SeCTAR combines model-based planning via model predictive control in latent space with unsupervised exploration objective."  
  - <https://sites.google.com/view/sectar> (demo)
  - `video` <https://facebook.com/icml.imls/videos/429761510871703?t=6122> (Liu)
  - `code` <https://github.com/wyndwarrior/Sectar>

#### ["Latent Space Policies for Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1804.02808) Haarnoja, Hartikainen, Abbeel, Levine
>	"Higher levels in the hierarchy can directly make use of the latent space of the lower levels as their action space, which allows to train the entire hierarchy in a layerwise fashion. This approach to hierarchical reinforcement learning has a number of conceptual and practical benefits. First, each layer in the hierarchy can be trained with exactly the same algorithm. Second, by using an invertible mapping from latent variables to actions, each layer becomes invertible, which means that the higher layer can always perfectly invert any behavior of the lower layer. This makes it possible to train lower layers on heuristic shaping rewards, while higher layers can still optimize task-specific rewards with good asymptotic performance. Finally, our method has a natural interpretation as an iterative procedure for constructing graphical models that gradually simplify the task dynamics."  
  - `video` <https://sites.google.com/view/latent-space-deep-rl> (demo)
  - `video` <https://facebook.com/icml.imls/videos/429761510871703?t=4970> (Haarnoja)
  - `video` <https://youtu.be/IAJ1LywY6Zg?t=25m55s> (Levine)
  - `code` <https://github.com/haarnoja/sac/blob/master/sac/policies/latent_space_policy.py>

#### ["Learning to Compose Skills"](https://arxiv.org/abs/1711.11289) Sahni, Kumar, Tejani, Isbell
  `ComposeNet`
>	"A major distinction between our work and recent attempts to learn an optimal sequence of subgoals is that our framework can learn a much richer set of compositions of skills. Our main contribution in this work is the expression of these compositions as differentiable functions. Representations of the individual skill policies are fed to this function as inputs and a representation for the composed task policy is produced. Skill policies are learned only once, and a wide variety of compositions can be created after the fact. We show that learning to compose skills is more efficient than learning to sequence those skills as is typically done in hierarchical RL. Moreover, we show how recursive compositions can be used to create rich hierarchies for more complicated behavior."  
>	"For example, in the game of Pacman, an agent must learn to collect food pellets while also avoiding enemy ghosts. In the usual view of hierarchical RL, a subgoal or option, such as "navigate to food pellet A" or "evade enemy ghost", would be activated one at a time and the agent must learn to alternate between them to complete the overall task. A better approach is to learn a policy that composes both subgoals, i.e. identifies food pellets that also keep Pacman far away from ghosts and prioritizes their collection."  
>	"In this work, we consider a subset of compositions defined by Linear Temporal Logic. A wide variety of common RL tasks can be specified using the temporal modal operators defined in LTL: next (O), always (□ ), eventually (♦ ), and until (U), along with the basic logic connectives: negation (¬), disjunction (∨ ), conjunction (∧ ) and implication (→ ). The Pacman task above can be translated into LTL as ¬G U (♦ F1 ∧ ♦ F2 ∧ ... ♦ Fn), where G is the proposition that the Pacman occupies the same location as any of the ghosts and F1 through Fn are the corresponding propositions for all the food pellets. Thus, the LTL sentence can be interpreted as “do not get eaten by a ghost until all the food pellets have been collected”."  
>	"We consider four types of compositions in Pacman task:  
>	1. ¬p U q, collect object q while evading enemy p  
>	2. ♦ p ∨ ♦ q, collect object p or q  
>	3. □ ¬p ∧ □ ¬q, always evade enemy p and enemy q  
>	4. ♦ (p ∧ ♦ q), collect object q then object p"  
  - `post` <https://himanshusahni.github.io/2017/12/26/reusability-in-ai.html>
  - `code` <https://github.com/himanshusahni/ComposeNet>

#### ["Learning with Options that Terminate Off-Policy"](https://arxiv.org/abs/1711.03817) Harutyunyan, Vrancx, Bacon, Precup, Nowe
>	"Generally, learning with longer options (like learning with multi-step returns) is known to be more efficient. However, if the option set for the task is not ideal, and cannot express the primitive optimal policy exactly, shorter options offer more flexibility and can yield a better solution. Thus, the termination condition puts learning efficiency at odds with solution quality. We propose to resolve this dilemma by decoupling the behavior and target terminations, just like it is done with policies in off-policy learning. To this end, we give a new algorithm, Q(beta), that learns the solution with respect to any termination condition, regardless of how the options actually terminate. We derive Q(beta) by casting learning with options into a common framework with well-studied multi-step off-policy learning."  
  - `video` <https://vimeo.com/249558377> (Harutyunyan)

#### ["Meta Learning Shared Hierarchies"](https://arxiv.org/abs/1710.09767) Frans, Ho, Chen, Abbeel, Schulman
>	"Meta-learning formulation of hierarchical RL: Agent has to solve a distribution of related long-horizon tasks, with the goal of learning new tasks in the distribution quickly."  
  - `post` <https://blog.openai.com/learning-a-hierarchy/> (demo)
  - `video` <https://vimeo.com/249558183> (Abbeel)
  - `video` <https://facebook.com/nipsfoundation/videos/1554594181298482?t=1282> (Abbeel)
  - `video` <https://youtu.be/BCzFs9Xb9_o?t=32m35s> (Sutskever)
  - `video` <https://youtu.be/JX5E0Tt7K10?t=9m50s> (Sutskever)
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Meta_Learning_Shared_Hierarchies.md>
  - `code` <https://github.com/openai/mlsh>

#### ["Eigenoption Discovery Through The Deep Successor Rerpesentation"](https://arxiv.org/abs/1710.11089) Machado, Rosenbaum, Guo, Liu, Tesauro, Campbell
  - `post` <https://manantomar.github.io/2018/04/10/blogpost.html>

#### ["Stochastic Neural Networks for Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1704.03012) Florensa, Duan, Abbeel
>	"SNN approach maximizes the mutual information of the top-level actions and the state distribution."  
>	"SNN approach outperforms state-of-the-art intrinsic motivation results like VIME (Houthooft et al., 2016)."  
  - `video` <https://youtube.com/playlist?list=PLEbdzN4PXRGVB8NsPffxsBSOCcWFBMQx3> (demo)
  - `video` <https://youtu.be/ARfpQzRCWT4?t=50m3s> (Nikishin)
  - `notes` <https://medium.com/syncedreview/stochastic-neural-networks-for-hierarchical-reinforcement-learning-7f9133cc18aa>
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Stochastic_Neural_Networks_for_Hierarchical_Reinforcement_Learning.md>
  - `code` <https://github.com/florensacc/snn4hrl>

#### ["FeUdal Networks for Hierarchical Reinforcement Learning"](http://arxiv.org/abs/1703.01161) Vezhnevets, Osindero, Schaul, Heess, Jaderberg, Silver, Kavukcuoglu
>	"A novel architecture that formulates sub-goals as directions in latent state space, which, if followed, translates into a meaningful behavioural primitives. FuN clearly separates the module that discovers and sets sub-goals from the module that generates behaviour through primitive actions. This creates a natural hierarchy that is stable and allows both modules to learn in complementary ways."  
>	"Agent with two level hierarchy: manager and worker."  
>	"Manager does not act in environment directly, sets goals for worker and gets rewarded for setting good goals with true reward."  
>	"Worker acts in environment and gets rewarded for achieving goals by manager - this is potentially much richer learning signal."  
>	"Manager selects subgoal direction that maximises reward. Worker selects actions that maximise cosine similarity with direction."  
>	"Manager is encouraged to predict advantageous direction in latent space and to provide reward to worker to follow the direction."  
>	"Key problems: how to represent goals and determine when they've been achieved."  
>
>	"Options framework:  
>	- bottom level: option, a sub-policy with terminal condition  
>	- top level: policy over options  
>	FeUdal framework:  
>	- bottom level: action  
>	- top level: provide meaningful and explicit goal for bottom level  
>	- sub-goal: a direction in latent space"  
  - `video` <https://youtube.com/watch?v=0e_uGa7ic74&t=29m20s> (demo)
  - `video` <https://vimeo.com/249557775> (Silver)
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

#### ["An Inference-Based Policy Gradient Method for Learning Options"](http://proceedings.mlr.press/v80/smith18a.html) Smith, Hoof, Pineau
>	"A novel policy gradient method for the automatic learning of policies with options. It uses inference methods to simultaneously improve all of the options available to an agent, and thus can be employed in an off-policy manner, without observing option labels. The differentiable inference procedure employed yields options that can be easily interpreted."  
  - `video` <https://facebook.com/icml.imls/videos/429761510871703?t=6774> (Smith)

#### ["The Option-Critic Architecture"](http://arxiv.org/abs/1609.05140) Bacon, Harb, Precup
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#the-option-critic-architecture-bacon-harb-precup>

#### ["Probabilistic Inference for Determining Options in Reinforcement Learning"](https://link.springer.com/article/10.1007/s10994-016-5580-x) Daniel, Hoof, Peters, Neumann
>	"Tasks that require many sequential decisions or complex solutions are hard to solve using conventional reinforcement learning algorithms. Based on the semi Markov decision process setting (SMDP) and the option framework, we propose a model which aims to alleviate these concerns. Instead of learning a single monolithic policy, the agent learns a set of simpler sub-policies as well as the initiation and termination probabilities for each of those sub-policies. While existing option learning algorithms frequently require manual specification of components such as the sub-policies, we present an algorithm which infers all relevant components of the option framework from data."  
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Probabilistic_Inference_for_Determining_Options_in_Reinforcement_Learning.md>

#### ["Strategic Attentive Writer for Learning Macro-Actions"](http://arxiv.org/abs/1606.04695) Vezhnevets, Mnih, Agapiou, Osindero, Graves, Vinyals, Kavukcuoglu
  `STRAW`
  - `video` <https://youtube.com/watch?v=niMOdSu3yio> (demo)
  - `video` <http://videolectures.net/deeplearning2016_mohamed_generative_models/#t=4435> (Mohamed)
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
  - `video` <https://youtu.be/XNGo9xqpgMo?t=16m46s> (Higgins)

#### ["Robust and Efficient Transfer Learning with Hidden Parameter Markov Decision Processes"](https://arxiv.org/abs/1706.06544) Killian, Daulton, Konidaris, Doshi-Velez
  `semantic representation`
>	"A new framework for modeling families of related tasks using low-dimensional latent embeddings, which correctly models the joint uncertainty in the latent parameters and the state space."  
>	"Define a new class of MDPs that includes a parameter θ which defines a parameterized transition function. Then, learning is done in the parameterized space; if the agent effectively learns the parameter, it can transfer knowledge to any MDP in the class."  
  - `video` <https://vimeo.com/248527846#t=14m49s> (Killian)

#### ["Towards Deep Symbolic Reinforcement Learning"](http://arxiv.org/abs/1609.05518) Garnelo, Arulkumaran, Shanahan
  `semantic representation`
>	"Contemporary DRL systems require very large datasets to work effectively, entailing that they are slow to learn even when such datasets are available. Moreover, they lack the ability to reason on an abstract level, which makes it difficult to implement high-level cognitive functions such as transfer learning, analogical reasoning, and hypothesis-based reasoning. Finally, their operation is largely opaque to humans, rendering them unsuitable for domains in which verifiability is important. We propose an end-to-end RL architecture comprising a neural back end and a symbolic front end with the potential to overcome each of these shortcomings."  
>	"Resulting system learns effectively and, by acquiring a set of symbolic rules that are easily comprehensible to humans, dramatically outperforms a conventional, fully neural DRL system on a stochastic variant of the game."  
>	"We tested the transfer learning capabilities of our algorithm by training an agent only on games of the grid variant then testing it on games of the random variant. After training, the unsupervised neural back end of the system is able to form a symbolic representation of any given frame within the micro-world of the game. In effect it has acquired the ontology of that micro-world, and this capability can be applied to any game within that micro-world irrespective of its specific rules. In the present case, no re-training of the back end was required when the system was applied to new variants of the game."  
>	"The key is for the system to understand when a new situation is analogous to one previously encountered or, more potently, to hypothesise that a new situation contains elements of several previously encountered situations combined in a novel way. In the present system, this capability is barely exploited."  
  - `video` <https://youtube.com/watch?v=_9dsx4tyzJ8> (Garnelo)
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
#### ["Kickstarting Deep Reinforcement Learning"](https://arxiv.org/abs/1803.03835) Schmitt et al.
  `policy distillation`
>	"The idea of having experts which can be used to train new agents through matching the output distributions was adapted for multitask reinforcement learning. Typically one gathers experience from expert policies, which are then used to train a student model using supervised learning. Consequently the focus has hitherto been on compression and teacher-matching, rather than the ultimate goal of reward maximisation. Although it is not explored in these papers, after performing distillation one could fine-tune the student policy using rewards. Kickstarting can be seen as a continuous version of such two-phase learning, with a focus on reward maximisation from the very beginning (which does not require arbitrary stopping criteria for any of the phases, as it is a joint optimisation problem)."  
>	"The main idea is to employ an auxiliary loss function which encourages the student policy to be close to the teacher policy on the trajectories sampled by the student. Importantly, the weight of this loss in the overall learning objective is allowed to change over time, so that the student can gradually focus more on maximising rewards it receives from the environment, potentially surpassing the teacher (which might indeed have an architecture with less learning capacity). In multi-task problems, it is also straightforward to extend this approach to the case of multiple teachers, each of which is an expert on a particular task: in this case the student will learn from an appropriate teacher on each task using an analogous formulation."  
>	"Our auxiliary loss can also be seen from the perspective of entropy regularisation. In the A3C method one adds the negated entropy H(πS(a|xt,ω)) as an auxiliary loss to encourage exploration. But minimisation of negated entropy is equivalent to minimising the KL divergence DKL(πS(a|xt,ω),U), where U is a uniform distribution over actions. Similarly the kickstarter loss is equivalent to the KL divergence between the teacher and the student policies. In this sense, the kickstarter loss can be seen as encouraging behaviour similar to the teacher, but just as entropy regularisation is not supposed to lead to convergence to a uniform policy, the goal of kickstarting is not to converge to the teacher’s policy. The aim of both is to provide a helpful auxiliary loss, based on what is a sensible behaviour – for the case of entropy regularization it is just sampling a random action, while for kickstarting it is following the teacher."  
  - `notes` <https://yobibyte.github.io/files/paper_notes/Kickstarting_Deep_Reinforcement_Learning_Simon_Schmitt__Jonathan_J__Hudson__Augustin_Zidek_et_al___2018.pdf>

#### ["Distral: Robust Multitask Reinforcement Learning"](https://arxiv.org/abs/1707.04175) Teh, Bapst, Czarnecki, Quan, Kirkpatrick, Hadsell, Heess, Pascanu
  `policy distillation`
>	"The assumption is that the tasks are related to each other (e.g. being in the same environment or having the same physics) and so good action sequences tend to recur across tasks. Our method achieves this by simultaneously distilling task-specific policies into a common default policy, and transferring this common knowledge across tasks by regularising all task-specific policies towards the default policy."  
>	"We observe that distillation arises naturally as one half of an optimization procedure when using KL divergences to regularize the output of task models towards a distilled model. The other half corresponds to using the distilled model as a regularizer for training the task models."  
>	"Another observation is that parameters in deep networks do not typically by themselves have any semantic meaning, so instead of regularizing networks in parameter space, it is worthwhile considering regularizing networks in a more semantically meaningful space, e.g. of policies."  
  - `video` <http://www.fields.utoronto.ca/video-archive/2017/11/2509-17850> (Teh)
  - `video` <https://youtube.com/watch?v=scf7Przmh7c> (Teh)
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
#### ["Transfer in Deep Reinforcement Learning Using Successor Features and Generalised Policy Improvement"](http://proceedings.mlr.press/v80/barreto18a) Barreto et al.
  `successor features`
>	"Extension to SF&GPI (Successor Features + Generalized Policy Improvement) framework in two ways."  
>	"First, its applicability is broader than initially shown. SF&GPI was designed for the scenario where each task corresponds to a different reward function; one of the basic assumptions in the original formulation was that the rewards of all tasks can be computed as a linear combination of a fixed set of features. Such an assumption is not strictly necessary, and in fact it is possible to have guarantees on the performance of the transferred policy even on tasks that are not in the span of the features. The realisation above adds some flexibility to the problem of computing features that are useful for transfer."  
>	"Second, by looking at the associated approximation from a slightly different angle, we show that one can replace the features with actual rewards. This makes it possible to apply SF&GPI online at scale."  
>	"We show that the transfer promoted by SF&GPI leads to good policies on unseen tasks almost instantaneously. Furthermore, we show how to learn policies that are specialised to the new tasks in a way that allows them to be added to the agent’s ever-growing set of skills, a crucial ability for continual learning."  
  - `video` <https://youtu.be/-dTnqfwTRMI> (demo)

#### ["Transfer in Deep Reinforcement Learning Using Successor Features and Generalised Policy Improvement"](http://arxiv.org/abs/1606.05312) Barreto et al.
  `successor features`
>	"Our approach rests on two key ideas: "successor features", a value function representation that decouples the dynamics of the environment from the rewards, and "generalised policy improvement", a generalisation of dynamic programming’s policy improvement step that considers a set of policies rather than a single one. Put together, the two ideas lead to an approach that integrates seamlessly within the reinforcement learning framework and allows transfer to take place between tasks without any restriction."  
  - `video` <https://facebook.com/nipsfoundation/videos/1554741347950432?t=5074> (Barreto)

#### ["Learning to Act by Predicting the Future"](https://arxiv.org/abs/1611.01779) Dosovitskiy, Koltun
  `successor features`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#learning-to-act-by-predicting-the-future-dosovitskiy-koltun>

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
  - `video` <https://youtu.be/x1kf4Zojtb0?t=41m4s> (de Freitas)
  - `notes` <https://blog.acolyer.org/2016/10/11/progressive-neural-networks/>
  - `code` <https://github.com/synpon/prog_nn>



---
### reinforcement learning - imitation

[**interesting older papers - behavioral cloning**](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---behavioral-cloning)  
[**interesting older papers - inverse reinforcement learning**](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---inverse-reinforcement-learning)  

----
#### ["Deep Reinforcement Learning from Human Preferences"](https://arxiv.org/abs/1706.03741) Christiano, Leike, Brown, Martic, Legg, Amodei
  `reinforcement learning from preferences`
>	"Algorithm provides two possible solutions for task to human who indicates which one is better. The process is repeated and the algorithm learns from 900 bits of feedback how to solve the problem."  
  - `video` <https://drive.google.com/drive/folders/0BwcFziBYuA8RM2NTdllSNVNTWTg> (demo)
  - `video` <https://youtube.com/watch?v=DekQm9pBbOE> (Shavkunov) `in russian`
  - `video` <https://youtube.com/watch?v=6h3_lTDFMb0> (Yagudin) `in russian`
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
>	"Learn a self-supervised understanding of the world and use it to quickly learn real-world tasks, entirely from 3rd person videos of humans (addressing correspondence problem, no labels, no reward function design, providing sample-efficiency of RL, quickly learning tasks, no kinesthetic demonstrations)."  
  - `video` <https://youtube.com/watch?v=b1UTUQpxPSY>
  - `video` <https://vimeo.com/252185872> (Lynch)
  - `code` <https://github.com/tensorflow/models/tree/master/research/tcn>

#### ["Learning from Demonstrations for Real World Reinforcement Learning"](https://arxiv.org/abs/1704.03732) Hester et al.
  `reinforcement learning from demonstrations` `DQfD`
>	"DQfD leverages small sets of demonstration data to massively accelerate the learning process even from relatively small amounts of demonstration data and is able to automatically assess the necessary ratio of demonstration data while learning thanks to a prioritized replay mechanism. DQfD works by combining temporal difference updates with supervised classification of the demonstrator's actions."  
  - `video` <https://youtube.com/playlist?list=PLdjpGm3xcO-0aqVf--sBZHxCKg-RZfa5T> (demo)
  - `code` <https://github.com/reinforceio/tensorforce/blob/master/tensorforce/models/dqfd_model.py>
  - `code` <https://github.com/go2sea/DQfD>

#### ["Deeply AggreVaTeD: Differentiable Imitation Learning for Sequential Prediction"](https://arxiv.org/abs/1703.01030) Sun, Venkatraman, Gordon, Boots, Bagnell
  `reinforcement learning from demonstrations` `AggreVaTeD`
>	"policy gradient extension of DAgger"  
  - `video` <https://vimeo.com/238243230> (Sun)
  - `paper` ["Convergence of Value Aggregation for Imitation Learning"](https://arxiv.org/abs/1801.07292) by Cheng and Boots

#### ["Query-Efficient Imitation Learning for End-to-End Autonomous Driving"](https://arxiv.org/abs/1605.06450) Zhang, Cho
  `reinforcement learning from demonstrations` `SafeDAgger`
  - `video` <https://youtu.be/soZXAH3leeQ?t=15m51s> (Cho)

----
#### ["One-Shot Visual Imitation Learning via Meta-Learning"](https://arxiv.org/abs/1709.04905) Finn, Yu, Zhang, Abbeel, Levine
  `imitation learning from visual observations` `meta-learning`
  - `video` <https://vimeo.com/252186304> (Finn, Yu)
  - `video` <https://youtu.be/lYU5nq0dAQQ?t=51m> (Levine)
  - `code` <https://github.com/tianheyu927/mil>

#### ["One-Shot Imitation Learning"](http://arxiv.org/abs/1703.07326) Duan, Andrychowicz, Stadie, Ho, Schneider, Sutskever, Abbeel, Zaremba
  `imitation learning` `meta-learning`
  - `video` <http://bit.ly/one-shot-imitation> (demo)
  - `post` <https://blog.openai.com/robots-that-learn/>
  - `video` <https://facebook.com/nipsfoundation/videos/1554594181298482?t=1543> (Abbeel)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/#t=3790> (de Freitas)
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
  `adversarial imitation learning` `InfoGAIL`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#inferring-the-latent-structure-of-human-decision-making-from-raw-visual-inputs-li-song-ermon>

#### ["Robust Imitation of Diverse Behaviors"](https://arxiv.org/abs/1707.02747) Wang, Merel, Reed, Wayne, Freitas, Heess
  `adversarial imitation learning`
>	"We develop a new version of GAIL that (1) is much more robust than the purely-supervised controller, especially with few demonstrations, and (2) avoids mode collapse, capturing many diverse behaviors when GAIL on its own does not."  
>	"The base of our model is a new type of variational autoencoder on demonstration trajectories that learns semantic policy embeddings, which can be smoothly interpolated with a resulting smooth interpolation of reaching behavior."  
  - `post` <https://deepmind.com/blog/producing-flexible-behaviours-simulated-environments/>
  - `video` <https://youtube.com/watch?v=necs0XfnFno> (demo)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/#t=4560> (de Freitas)
  - `notes` <https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Robust_Imitation_of_Diverse_Behaviors.md>

#### ["Generative Adversarial Imitation Learning"](http://arxiv.org/abs/1606.03476) Ho, Ermon
  `adversarial imitation learning` `GAIL`
>	"Uses a GAN framework to discriminate between teacher and student experience and force the student to behave close to the teacher."  
  - `video` <https://youtube.com/watch?v=bcnCo9RxhB8> (Ermon)
  - `video` <https://youtu.be/d9DlQSJQAoI?t=22m12s> (Finn)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/#t=4183> (de Freitas)
  - `notes` <http://tsong.me/blog/gail/>
  - `notes` <https://yobibyte.github.io/files/paper_notes/Generative_Adversarial_Imitation_Learning__Ho_Ermon__2017.pdf>
  - `code` <https://github.com/openai/imitation>
  - `code` <https://github.com/higgsfield/RL-Adventure-2>
  - `code` <https://github.com/DanielTakeshi/rl_algorithms>

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
#### ["Open-ended Learning in Symmetric Zero-sum Games"](https://arxiv.org/abs/1901.08106) Balduzzi, Garnelo, Bachrach, Czarnecki, Perolat, Jaderberg, Graepel
>	"PSROrn = Rectified Nash matchmaking, upper left"  
>	"Zero-sum games such as chess and poker are, abstractly, functions that evaluate pairs of agents, for example labeling them ‘winner’ and ‘loser’. If the game is approximately transitive, then selfplay generates sequences of agents of increasing strength. However, nontransitive games, such as rock-paper-scissors, can exhibit strategic cycles, and there is no longer a clear objective – we want agents to increase in strength, but against whom is unclear. In this paper, we introduce a geometric framework for formulating agent objectives in zero-sum games, in order to construct adaptive sequences of objectives that yield openended learning. The framework allows us to reason about population performance in nontransitive games, and enables the development of a new algorithm (rectified Nash response, PSROrN) that uses game-theoretic niching to construct diverse populations of effective agents, producing a stronger set of agents than existing algorithms."  
>	"The paper is about formulating useful objectives in nontransitive games (e.g. poker or StarCraft), which turns out to be a surprisingly subtle problem. Usually, the learning objective is *given*: minimize a loss or maximize rewards. In nontransitive games, the objective is unclear. Yes, to win, but against whom? Beating paper and beating scissors in rock-paper-scissors are different objectives, that pull in different directions. Blizzard has painstakingly embedded many rock-paper-scissor cycles into SC2. For example, ground units, void rays and phoenixes have this kind of dynamic. These endless exploits are a large part of why humans find the game is so rich and interesting. Nontransitivity has also been linked to biodiversity. Which makes sense! If there are lots of ways of “winning” in an ecosystem, then there’ll be lots of niches for organisms to evolve into. The problem is that there’s no clear way to define the fitness of individuals in nontransitive games — which is better, rock or paper? And if you don’t have a clear objective, then all the compute in the world won’t save you. Our solution is to formulate population-level objectives, using tools like Nash equilibria. Rather than trying to find a single dominant agent, which may not exist, the goal is to find all the underlying strategic dimensions of the game, and the best ways of executing them. Doing this right requires some cool geometry: we extend the idea of a 1-dim fitness landscape to multi-dim gamescapes that represent the latent objectives in a game."  
>	"In a game with finite sets of strategies, the right thing to do is to find mixed Nash. If strategies are parametrized by (say) neural nets, you've got no hope of computing Nash over all of them. So you have to find "the right" finite set of agents/strategies to work with. That's what we mean when we talk about growing the gamescape in useful directions: finding a good (finite) set of agents in this vast continuum of potential strategies. Once we've got them, Nash is a good way to go."  

#### ["A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning"](https://arxiv.org/abs/1711.00832) Lanctot, Zambaldi, Gruslys, Lazaridou, Tuyls, Perolat, Silver, Graepel
>	"We first observe that independent reinforcement learners produce policies that can be jointly correlated, failing to generalize well during execution with other agents. We quantify this effect by proposing a new metric called joint policy correlation. We then propose an algorithm motivated by game-theoretic foundations, which generalises several previous approaches such as fictitious play, iterated best response, independent RL, and double oracle. We show that our algorithm can reduce joint policy correlation significantly in first-person coordination games, and finds robust counter-strategies in a common poker benchmark game."  

#### ["Learning Nash Equilibrium for General-Sum Markov Games from Batch Data"](https://arxiv.org/abs/1606.08718) Perolat, Strub, Piot, Pietquin
>	"We address the problem of learning a Nash equilibrium in γ-discounted multiplayer general-sum Markov Games in a batch setting. As the number of players increases in MG, the agents may either collaborate or team apart to increase their final rewards. One solution to address this problem is to look for a Nash equilibrium. Although, several techniques were found for the subcase of two-player zero-sum MGs, those techniques fail to find a Nash equilibrium in general-sum Markov Games."  
>	"We introduce a new (weaker) definition of ε-Nash equilibrium in MGs which grasps the strategy’s quality for multiplayer games. We prove that minimizing the norm of two Bellman-like residuals implies to learn such an ε-Nash equilibrium. Then, we show that minimizing an empirical estimate of the Lp norm of these Bellman-like residuals allows learning for general-sum games within the batch setting. Finally, we introduce a neural network architecture that successfully learns a Nash equilibrium in generic multiplayer general-sum turn-based MGs."  

----
#### ["Bayesian Action Decoder for Deep Multi-Agent Reinforcement Learning"](https://arxiv.org/abs/1811.01458) Foerster et al.

----
#### ["QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"](https://arxiv.org/abs/1803.11485) Rashid, Samvelyan, Witt, Farquhar, Foerster, Whiteson
  `QMIX`
>	"Learning joint action-values conditioned on extra state information is an attractive way to exploit centralised learning, where global state information is available and communication constraints are lifted, but the best strategy for then extracting decentralised policies is unclear. QMIX trains decentralised policies in a centralised end-to-end fashion. QMIX employs a network that estimates joint action-values as a complex non-linear combination of per-agent values that condition only on local observations. QMIX structurally enforces that the joint-action value is monotonic in the per-agent values, which allows tractable maximisation of the joint action-value in off-policy learning, and guarantees consistency between the centralised and decentralised policies."  
  - `video` <https://vimeo.com/287801892> (Rashid)
  - `post` <https://medium.com/@gema.parreno.piqueras/qmix-paper-ripped-monotonic-value-function-factorization-for-deep-multi-agent-reinforcement-7e03998f61e7>
  - `code` <https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py>

#### ["Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"](https://arxiv.org/abs/1706.02275) Lowe, Wu, Tamar, Harb, Abbeel, Mordatch
  `MADDPG`
>	"Uses learning rule from DDPG to learn a central off-policy critic based on Q-learning and uses certain on-policy policy gradient estimator to learn policies for each agent."  
  - `post` <https://blog.openai.com/learning-to-cooperate-compete-and-communicate/>
  - `video` <https://youtube.com/watch?v=QCmBo91Wy64> (demo)
  - `code` <https://github.com/openai/maddpg>
  - `code` <https://github.com/openai/multiagent-particle-envs>

#### ["Counterfactual Multi-Agent Policy Gradients"](https://arxiv.org/abs/1705.08926) Foerster, Farquhar, Afouras, Nardelli, Whiteson
  `COMA`
>	"One of the great challenges when training multi-agent policies is the credit assignment problem. Just like in a football team, the reward achieved depends on the actions of all of the different agents. Given that all agents are constantly improving their policies, it is difficult for any given agent to evaluate the impact of their individual action on the overall performance of the team."  
>	"We evaluate COMA in the testbed of StarCraft unit micromanagement, using a decentralised variant with significant partial observability. COMA significantly improves average performance over other multi-agent actor-critic methods in this setting, and the best performing agents are competitive with state-of-the-art centralised controllers that get access to the full state."  
>	"COMA uses a centralised critic to train decentralised actors, estimating a counterfactual advantage function for each agent in order to address multi-agent credit assignment. COMA learns a fully centralised state-action value function and then uses it to guide the optimisation of decentralised policies in an actor-critic framework. This requires on-policy learning, which can be sample-inefficient, and training the fully centralised critic becomes impractical when there are more than a handful of agents."  
  - `video` <https://youtube.com/watch?v=3OVvjE5B9LU> (Whiteson)
  - `code` <https://github.com/oxwhirl/pymarl/blob/master/src/learners/coma_learner.py>

#### ["Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning"](https://arxiv.org/abs/1702.08887) Foerster, Nardelli, Farquhar, Afouras, Torr, Kohli, Whiteson
  - `video` <https://vimeo.com/238243859> (Foerster)
  - `post` <https://parnian.ghost.io/understanding-stabilising-experience-replay-for-deep-multi-agent-reinforcement-learning>

----
#### ["Emergent Complexity via Multi-Agent Competition"](https://arxiv.org/abs/1710.03748) Bansal, Pachocki, Sidor, Sutskever, Mordatch
  - `post` <https://blog.openai.com/competitive-self-play/> (demo)
  - <https://sites.google.com/view/multi-agent-competition> (demo)
  - `video` <https://vimeo.com/250399465#t=7m56s> (Sutskever)
  - `video` <https://youtu.be/JX5E0Tt7K10?t=17m42s> (Sutskever)
  - `notes` <https://blog.acolyer.org/2018/01/11/emergent-complexity-via-multi-agent-competition/>
  - `code` <https://github.com/openai/multiagent-competition>

#### ["Learning with Opponent-Learning Awareness"](https://arxiv.org/abs/1709.04326) Foerster, Chen, Al-Shedivat, Whiteson, Abbeel, Mordatch
>	"LOLA modifies the learning objective by predicting and differentiating through opponent learning steps. This is intuitively appealing and experimentally successful, encouraging cooperation in settings like the Iterated Prisoner’s Dilemma where more ‘stable’ algorithms like SGA defect. However, LOLA has no guarantees of converging or even preserving fixed points of the game."  
  - `post` <https://blog.openai.com/learning-to-model-other-minds/> (demo)
  - `video` <https://facebook.com/icml.imls/videos/429607650887089?t=900> (Foerster)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1709.04326>
  - `code` <http://github.com/alshedivat/lola>
  - `code` <https://github.com/alexis-jacq/LOLA_DiCE>

----
#### ["Learning to Communicate with Deep Multi-Agent Reinforcement Learning"](http://arxiv.org/abs/1605.06676) Foerster, Assael, de Freitas, Whiteson
  - `video` <https://youtu.be/SAcHyzMdbXc?t=19m> (de Freitas)
  - `video` <https://youtube.com/watch?v=xL-GKD49FXs> (Foerster)
  - `video` <http://videolectures.net/deeplearning2016_foerster_learning_communicate/> (Foerster)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.07133>
  - `code` <https://github.com/iassael/learning-to-communicate>

#### ["Learning Multiagent Communication with Backpropagation"](http://arxiv.org/abs/1605.07736) Sukhbaatar, Szlam, Fergus
  - `video` <https://youtube.com/watch?v=9fZ8JiDZqCA> (Sukhbaatar)
  - `video` <https://youtu.be/_iVVXWkoEAs?t=30m6s> (Fergus)
  - `video` <https://youtu.be/SAcHyzMdbXc?t=19m> (de Freitas)
  - `slides` <https://uclmr.github.io/nampi/talk_slides/rob-nampi.pdf>
  - `code` <https://github.com/facebookresearch/CommNet>
  - `code` <https://github.com/rickyhan/CommNet>



---
### program synthesis

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Machine%20Learning.md#program-synthesis)

[overview](https://alexpolozov.com/blog/program-synthesis-2018) of recent papers by Oleksandr Polozov

----
#### ["Evolving Simple Programs for Playing Atari Game"](https://arxiv.org/abs/1806.05695) Wilson, Cussat-Blanc, Luga, Miller
>	"Programs are evolved using mixed type Cartesian Genetic Programming with a function set suited for matrix operations, including image processing, but allowing for controller behavior to emerge."  
>	"While the programs are relatively small, many controllers are competitive with state of the art methods for the Atari benchmark set and require less training time."  

#### ["Synthesizing Programs for Images using Reinforced Adversarial Learning"](https://arxiv.org/abs/1804.01118) Ganin, Kulkarni, Babuschkin, Eslami, Vinyals
  `SPIRAL`
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#synthesizing-programs-for-images-using-reinforced-adversarial-learning-ganin-kulkarni-babuschkin-eslami-vinyals>

#### ["Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis"](https://arxiv.org/abs/1805.04276) Bunel, Hausknecht, Devlin, Singh, Kohli
>	"Using the DSL grammar and reinforcement learning to improve synthesis of programs with complex control flow."  
>	"Sequence-to-sequence generation models are trained to maximize the likelihood of known reference programs. This strategy has two key limitations. First, it ignores Program Aliasing: the fact that many different programs may satisfy a given specification (especially with incomplete specifications such as a few input-output examples). By maximizing the likelihood of only a single reference program, it penalizes many semantically correct programs, which can adversely affect the synthesizer performance. Second, this strategy overlooks the fact that programs have a strict syntax that can be efficiently checked."  
  - `code` <https://github.com/carpedm20/program-synthesis-rl-tensorflow>
  - `code` <https://github.com/carpedm20/karel>

#### ["Learning Explanatory Rules from Noisy Data"](https://arxiv.org/abs/1711.04574) Evans, Grefenstette
>	"We demonstrate it is possible for systems to combine intuitive perceptual with conceptual interpretable reasoning. The system we describe, ∂ILP, is robust to noise, data-efficient, and produces interpretable rules."  
>	"∂ILP differs from standard neural nets because it is able to generalise symbolically, and it differs from standard symbolic programs because it is able to generalise visually. It learns explicit programs from examples that are readable, interpretable, and verifiable. ∂ILP is given a partial set of examples (the desired results) and produces a program that satisfies them. It searches through the space of programs using gradient descent. If the outputs of the program conflict with the desired outputs from the reference data, the system revises the program to better match the data."  
  - `post` <https://deepmind.com/blog/learning-explanatory-rules-noisy-data/>
  - `video` <https://youtube.com/watch?v=AcpbZF4gy7Y> (Evans)
  - `video` <https://youtube.com/watch?v=_wuFBF_Cgm0> (Evans)
  - `post` <https://reddit.com/r/MachineLearning/comments/7tthm3/r_learning_explanatory_rules_from_noisy_data/dtgu2uw/>

#### ["Learning to Select Examples for Program Synthesis"](https://arxiv.org/abs/1711.03243) Pu, Miranda, Solar-Lezama, Kaelbling
>	"Due to its precise and combinatorial nature, program synthesis is commonly formulated as a constraint satisfaction problem, where input-output examples are encoded as constraints and solved with a constraint solver. A key challenge of this formulation is scalability: while constraint solvers work well with few well-chosen examples, a large set of examples can incur significant overhead in both time and memory. We address this challenge by constructing a representative subset of examples that is both small and able to constrain the solver sufficiently. We build the subset one example at a time, using a neural network to predict the probability of unchosen input-output examples conditioned on the chosen input-output examples, and adding the least probable example to the subset."  

#### ["Neural Program Meta-Induction"](https://arxiv.org/abs/1710.04157) Devlin, Bunel, Singh, Hausknecht, Kohli
  - `post` <https://microsoft.com/en-us/research/blog/neural-program-induction>
  - `code` <https://github.com/carpedm20/karel>

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

#### ["Making Neural Programming Architectures Generalize via Recursion"](https://arxiv.org/abs/1704.06611) Cai, Shin, Song
>	"We implement recursion in the Neural Programmer-Interpreter framework on four tasks: grade-school addition, bubble sort, topological sort, and quicksort."  
  - `video` <https://facebook.com/iclr.cc/videos/1713144705381255?t=2999> (Cai)
  - `notes` <https://theneuralperspective.com/2017/03/14/making-neural-programming-architecture-generalize-via-recursion/>

#### ["RobustFill: Neural Program Learning under Noisy I/O"](https://arxiv.org/abs/1703.07469) Devlin, Uesato, Bhupatiraju, Singh, Mohamed, Kohli
  - `video` <https://vimeo.com/238227939> (Uesato, Bhupatiraju)
  - `video` <https://facebook.com/nipsfoundation/videos/1552060484885185?t=5885> (Reed)

#### ["Differentiable Programs with Neural Libraries"](https://arxiv.org/abs/1611.02109) Gaunt, Brockschmidt, Kushman, Tarlow
  - `video` <https://vimeo.com/238227833> (Gaunt)

#### ["Neuro-Symbolic Program Synthesis"](https://arxiv.org/abs/1611.01855) Parisotto, Mohamed, Singh, Li, Zhou, Kohli

#### ["TerpreT: A Probabilistic Programming Language for Program Induction"](http://arxiv.org/abs/1608.04428) Gaunt, Brockschmidt, Singh, Kushman, Kohli, Taylor, Tarlow
>	"These works raise questions of (a) whether new models can be designed specifically to synthesize interpretable source code that may contain looping and branching structures, and (b) whether searching over program space using techniques developed for training deep neural networks is a useful alternative to the combinatorial search methods used in traditional IPS. In this work, we make several contributions in both of these directions."  
>	"Shows that differentiable interpreter-based program induction is inferior to discrete search-based techniques used by the programming languages community. We are then left with the question of how to make progress on program induction using machine learning techniques."  
  - `video` <https://youtu.be/vzDuVhFMB9Q?t=2m40s> (Gaunt)
  - `code` <https://github.com/51alg/TerpreT>

#### ["Programming with a Differentiable Forth Interpreter"](http://arxiv.org/abs/1605.06640) Bošnjak, Rocktaschel, Naradowsky, Riedel
  `learning details of probabilistic program`
>	"The paper talks about a certain class of neural networks that incorporate procedural knowledge. The way they are constructed is by compiling Forth code (procedural) to TensorFlow expressions (linear algebra) to be able to train slots (missing pieces in the code) end-to-end from input-output pairs using backpropagation."  
  - `video` <https://vimeo.com/238227890> (Bosnjak)
  - `video` <https://youtu.be/LsLPp7gqwA4?t=27m8s> (Minervini)
  - `video` <https://facebook.com/nipsfoundation/videos/1552060484885185?t=5637> (Reed)
  - `code` <https://github.com/uclmr/d4>

#### ["Adaptive Neural Compilation"](http://arxiv.org/abs/1605.07969) Bunel, Desmaison, Kohli, Torr, Kumar



---
### reasoning

[**interesting older papers - reasoning**](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---reasoning)  
[**interesting older papers - question answering over knowledge bases**](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-knowledge-bases)  
[**interesting older papers - question answering over texts**](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-texts)  

----
#### ["Towards Neural Theorem Proving at Scale"](https://arxiv.org/abs/1807.08204) Minervini, Bosnjak, Rocktaschel, Riedel
  `learning logic` `NTP`
  - `video` <https://youtu.be/LsLPp7gqwA4?t=43m46s> (Minervini)

#### ["End-to-end Differentiable Proving"](https://arxiv.org/abs/1705.11040) Rocktaschel, Riedel
  `learning logic` `NTP`
  - `video` <https://youtu.be/LsLPp7gqwA4?t=32m45s> (Minervini)
  - <https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#end-to-end-differentiable-proving-rocktaschel-riedel>

#### ["Differentiable Learning of Logical Rules for Knowledge Base Reasoning"](https://arxiv.org/abs/1702.08367) Yang, Yang, Cohen
  `learning logic` `TensorLog`
>	"We study the problem of learning probabilistic first-order logical rules for knowledge base reasoning. This learning problem is difficult because it requires learning the parameters in a continuous space as well as the structure in a discrete space. We propose a framework, Neural Logic Programming, that combines the parameter and structure learning of first-order logical rules in an end-to-end differentiable model. This approach is inspired by a recently-developed differentiable logic called TensorLog, where inference tasks can be compiled into sequences of differentiable operations. We design a neural controller system that learns to compose these operations."  
  - `video` <https://youtu.be/bVWDxyydyEM?t=43m31s> (Neubig)
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
#### ["ReinforceWalk: Learning to Walk in Graph with Monte Carlo Tree Search"](https://arxiv.org/abs/1802.04394) Shen, Chen, Huang, Guo, Gao
  `knowledge graph completion` `expert iteration`
>	"MINERVA uses policy gradient method to explore paths in knowledge graphs during training and test. Our proposed model further exploits state transition information by integrating with the MCTS algorithm. Empirically, our proposed algorithm outperforms both MINERVA in the knowledge base completion benchmark."  
>	"This work shares a similar spirit with AlphaZero in that it also uses MCTS and policy network to iteratively improve each other. However, the method in AlphaZero improves the policy network from the MCTS probabilities of move, while this method improves the policy from the trajectories generated by MCTS. Note that the MCTS probabilities of move in AlphaZero is constructed from the visit counts of all the edges connected to the MCTS root node, meaning that it only uses information near the root node to improve the policy network. While this work improves the policy network by learning from the trajectories generated by MCTS. Therefore, the information over the entire MCTS search tree is used."  

#### ["Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning"](https://arxiv.org/abs/1711.05851) Das, Dhuliawala, Zaheer, Vilnis, Durugkar, Krishnamurthy, Smola, McCallum
  `question answering over knowledge bases` `MINERVA`
>	"RL agent which answers queries in a knowledge graph of entities and relations. Starting from an entity node, MINERVA learns to navigate the graph conditioned on the input query till it reaches the answer entity."  
  - `code` <https://github.com/shehzaadzd/MINERVA>

#### ["Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks"](https://arxiv.org/abs/1704.08384) Das, Zaheer, Reddy, McCallum
  `question answering over knowledge bases`
  - `video` <https://youtu.be/lc68_d_DnYs?t=7m28s> (Neelakantan)
  - `code` <https://github.com/rajarshd/TextKBQA>

#### ["Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision"](http://arxiv.org/abs/1611.00020) Liang, Berant, Le, Forbus, Lao
  `question answering over knowledge bases`
>	"We propose the Manager-Programmer-Computer framework, which integrates neural networks with non-differentiable memory to support abstract, scalable and precise operations through a friendly neural computer interface. Specifically, we introduce a Neural Symbolic Machine, which contains a sequence-to-sequence neural "programmer", and a non-differentiable "computer" that is a Lisp interpreter with code assist."  
  - `video` <https://youtube.com/watch?v=KFkqosOGTxM> (Liang)
  - `video` <https://vimeo.com/234953110> (Liang)
  - `notes` <https://northanapon.github.io/papers/2017/01/16/neural-symbolic-machine.html>
  - `notes` <https://github.com/carpedm20/paper-notes/blob/master/notes/neural-symbolic-machine.md>
  - `code` <https://github.com/crazydonkey200/neural-symbolic-machines>

#### ["Learning a Natural Language Interface with Neural Programmer"](http://arxiv.org/abs/1611.08945) Neelakantan, Le, Abadi, McCallum, Amodei
  `question answering over knowledge bases`
  - `video` <http://youtu.be/lc68_d_DnYs?t=24m44s> (Neelakantan)
  - `code` <https://github.com/tensorflow/models/tree/master/research/neural_programmer>

----
#### ["Multi-Mention Learning for Reading Comprehension with Neural Cascades"](https://arxiv.org/abs/1711.00894) Swayamdipta, Parikh, Kwiatkowski
  `question answering over texts` `documents collection`

#### ["Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering"](https://arxiv.org/abs/1711.05116) Wang et al.
  `question answering over texts` `documents collection`

#### ["R^3: Reinforced Reader-Ranker for Open-Domain Question Answering"](https://arxiv.org/abs/1709.00023) Wang et al.
  `question answering over texts` `documents collection`
>	"First, we propose a new pipeline for open-domain QA with a Ranker component, which learns to rank retrieved passages in terms of likelihood of extracting the ground-truth answer to a given question. Second, we propose a novel method that jointly trains the Ranker along with an answer-extraction Reader model, based on reinforcement learning."  
  - `post` <https://ibm.com/blogs/research/2018/02/open-domain-qa/>

#### ["Reading Wikipedia to Answer Open-Domain Questions"](https://arxiv.org/abs/1704.00051) Chen, Fisch, Weston, Bordes
  `question answering over texts` `documents collection` `DrQA`
  - `code` <https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa>
  - `code` <https://github.com/hitvoice/DrQA>

#### ["Simple and Effective Multi-Paragraph Reading Comprehension"](https://arxiv.org/abs/1710.10723) Clark, Gardner
  `question answering over texts` `multi-paragraph document`

#### ["Coarse-to-Fine Question Answering for Long Documents"](http://arxiv.org/abs/1611.01839) Choi, Hewlett, Lacoste, Polosukhin, Uszkoreit, Berant
  `question answering over texts` `multi-paragraph document`
  - `video` <https://youtu.be/fpycaHd1Z08?t=36m14s> (Neubig)

#### ["Key-Value Memory Networks for Directly Reading Documents"](http://arxiv.org/abs/1606.03126) Miller, Fisch, Dodge, Amir-Hossein Karimi, Bordes, Weston
  `question answering over texts` `multi-paragraph document`
  - `video` <https://youtu.be/x1kf4Zojtb0?t=25m46s> (de Freitas)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1606.03126>
  - `code` <https://github.com/facebook/MemNN/blob/master/KVmemnn>

#### ["QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension"](https://arxiv.org/abs/1804.09541) Yu, Dohan, Luong, Zhao, Chen, Norouzi, Le
  `question answering over texts` `single paragraph`
>	"Paper is the first work to achieve both fast and accurate reading comprehension model, by discarding the recurrent networks in favor of feed forward architectures."  
>	"Paper is the first to mix self-attention and convolutions."  

----
#### ["A Generative Vision Model that Trains with High Data Efficiency and Breaks Text-based CAPTCHAs"](http://science.sciencemag.org/content/early/2017/10/26/science.aag2612.full) George et al.
  `question answering over images`
  - <https://github.com/brylevkirill/notes/blob/master/Bayesian%20Inference%20and%20Learning.md#a-generative-vision-model-that-trains-with-high-data-efficiency-and-breaks-text-based-captchas-george-et-al>

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
  - `video` <https://facebook.com/nipsfoundation/videos/1554654864625747?t=2390> (Santoro)
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

#### ["From Language to Goals: Inverse Reinforcement Learning for Vision-Based Instruction Following"](https://arxiv.org/abs/1902.07742) Fu, Korattikara, Levine, Guadarrama
  `goal-driven language learning` `LC-RL` `ICLR 2019`
  - <https://sites.google.com/view/language-irl>
>	"We investigate the problem of grounding language commands as reward functions using inverse reinforcement learning, and argue that language-conditioned rewards are more transferable than language-conditioned policies to new environments. We propose language-conditioned reward learning, which grounds language commands as a reward function represented by a deep neural network. We demonstrate that our model learns rewards that transfer to novel tasks and environments on realistic, high-dimensional visual environments with natural language commands, whereas directly learning a language-conditioned policy leads to poor performance."  
>	 "Reward-learning enables an agent to learn and interact within the test environment rather than relying on zero-shot policy transfer."  

#### ["Learning to Follow Language Instructions with Adversarial Reward Induction"](https://arxiv.org/abs/1806.01946) Bahdanau, Hill, Leike, Hughes, Kohli, Grefenstette
  `goal-driven language learning` `AGILE`
>	"AGILE, an approach to training instruction-following agents from examples of corresponding goal-states rather than explicit reward functions. This opens up new possibilities for training language-aware agents, because in the real world, and even in rich simulated environments, acquiring such data via human annotation would often be much more viable than defining and implementing reward functions programmatically. Indeed, programming rewards to teach robust and general instruction-following may ultimately be as challenging as writing a program to interpret language directly, an endeavour that is notoriously laborious, and some say, ultimately futile. As well as a means to learn from a potentially more prevalent form of data, our experiments demonstrate that AGILE performs comparably with and can learn as fast as RL with an auxiliary task."  
>	"An attractive aspect of AGILE is that learning “what should be done” and “how it should be done” is performed by two different model components. Our experiments confirm that the “what” kind of knowledge generalizes better to different environments. When the dynamics of the environment changed at test time, fine-tuning against a frozen discriminator allowed to the policy recover some of its original capability in the new setting."  
>	"AGILE is strongly inspired by Inverse Reinforcement Learning methods in general, and Generative Adversarial Imitation Learning in particular. However, it develops these methods to enable language learning; the policy and the discriminator are conditioned on an instruction, and that the training data contains goal-states - rather than complete trajectories. An appealing advantage of AGILE is the fact that the discriminator Dφ and the policy πθ learn two related but distinct aspects of an instruction: the discriminator focuses on recognizing the goal-states (what should be done), whereas the policy learns what to do in order to get to a goal-state (how it should be done). The intuition motivating this design is that the knowledge about how instructions define goals should generalize more strongly than the knowledge about which behavior is needed to execute instructions. Following this intuition, we propose to reuse a trained AGILE’s discriminator as a reward function for training or fine-tuning policies."  
>	"AGILE modifies GAIL to use a goal-based discriminator and false negative filtering, using DQN as a policy optimizer and ρ=0.25."  
  - `video` <https://youtube.com/watch?v=07S-x3MkEoQ>

#### ["Learning with Latent Language"](https://arxiv.org/abs/1711.00482) Andreas, Klein, Levine
  `goal-driven language learning`
>	"optimizing models in a space parameterized by natural language"  
>	"turning an instruction following model into a model for few-shot policy learning"  
>	"learning with natural language parameters provides structure, efficiency, interpretability"  
>	"Using standard neural encoder–decoder components to build models for representation and search in this space, we demonstrated that our approach outperforms strong baselines on classification, structured prediction and reinforcement learning tasks."  
>	"The approach outperforms both multi-task and meta-learning approaches that map directly from training examples to outputs by way of a real-valued parameterization, as well as approaches that make use of natural language annotations as an additional supervisory signal rather than an explicit latent parameter. The natural language concept descriptions inferred by our approach often agree with human annotations when they are correct, and provide an interpretable debugging signal when incorrect. In short, by equipping models with the ability to “think out loud” when learning, they become both more comprehensible and more accurate."  
>	"- Language encourages compositional generalization: standard deep learning architectures are good at recognizing new instances of previously encountered concepts, but not always at generalizing to new ones. By forcing decisions to pass through a linguistic bottleneck in which the underlying compositional structure of concepts is explicitly expressed, stronger generalization becomes possible.  
>	- Language simplifies structured exploration: relatedly, linguistic scaffolding can provide dramatic advantages in problems like reinforcement learning that require exploration: models with latent linguistic parameterizations can sample in this space, and thus limit exploration to a class of behaviors that are likely a priori to be goal-directed and interpretable.  
>	- Language can help learning: in multitask settings, it can even improve learning on tasks for which no language data is available at training or test time. While some of these advantages are also provided by techniques like program synthesis that are built on top of formal languages, natural language is at once more expressive and easier to obtain than formal supervision."  
  - `video` <https://vimeo.com/252185410> (Andreas)
  - `post` <http://blog.jacobandreas.net/fake-language.html>
  - `code` <http://github.com/jacobandreas/l3>

#### ["Grounded Language Learning in a Simulated 3D World"](https://arxiv.org/abs/1706.06551) Hermann et al.
  `goal-driven language learning`
>	"The agent learns simple language by making predictions about the world in which that language occurs, and by discovering which combinations of words, perceptual cues and action decisions result in positive outcomes. Its knowledge is distributed across language, vision and policy networks, and pertains to modifiers, relational concepts and actions, as well as concrete objects. Its semantic representations enable the agent to productively interpret novel word combinations, to apply known relations and modifiers to unfamiliar objects and to re-use knowledge pertinent to the concepts it already has in the process of acquiring new concepts."  
>	"While our simulations focus on language, the outcomes are relevant to machine learning in a more general sense. In particular, the agent exhibits active, multi-modal concept induction, the ability to transfer its learning and apply its knowledge representations in unfamiliar settings, a facility for learning multiple, distinct tasks, and the effective synthesis of unsupervised and reinforcement learning. At the same time, learning in the agent reflects various effects that are characteristic of human development, such as rapidly accelerating rates of vocabulary growth, the ability to learn from both rewarded interactions and predictions about the world, a natural tendency to generalise and re-use semantic knowledge, and improved outcomes when learning is moderated by curricula."  
  - `video` <https://youtube.com/watch?v=wJjdu1bPJ04> (demo)
  - `video` <http://videolectures.net/deeplearning2017_blunsom_language_processing/#t=2934> (Blunsom)
  - `code` <https://github.com/dai-dao/Grounded-Language-Learning-in-Pytorch>

#### ["Programmable Agents"](https://arxiv.org/abs/1706.06383) Denil, Colmenarejo, Cabi, Saxton, Freitas
  `goal-driven language learning`
>	"Agents that execute declarative programs and can generalize to a wide variety of zero-shot semantic tasks."  
  - `video` <https://youtube.com/playlist?list=PLs1LSEoK_daRDnPUB2u7VAXSonlNU7IcV> (demo)
  - `video` <http://videolectures.net/deeplearning2017_de_freitas_deep_control/#t=1977> (de Freitas)
  - `video` <https://youtu.be/zsvYr5tyj9M?t=50m27s> (Erzat) `in russian`
  - `code` <https://github.com/jaesik817/programmable-agents_tensorflow>

#### ["Gated-Attention Architectures for Task-Oriented Language Grounding"](https://arxiv.org/abs/1706.07230) Chaplot, Sathyendra, Pasumarthi, Rajagopal, Salakhutdinov
  `goal-driven language learning`
  - `video` <https://vimeo.com/252185932> (21:57) (Salakhutdinov)
  - `code` <https://github.com/devendrachaplot/DeepRL-Grounding>

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
  - `video` <https://facebook.com/iclr.cc/videos/1712966538732405> (Peysakhovich)

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

[**interesting older papers**](https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#interesting-papers)

----
#### ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) Radford, Wu, Child, Luan, Amodei, Sutskever
  `language modeling` `GPT-2`
  - <https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#language-models-are-unsupervised-multitask-learners-radford-wu-child-luan-amodei-sutskever>

#### ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805) Devlin, Chang, Lee, Toutanova
  `language modeling` `BERT`
  - <https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-devlin-chang-lee-toutanova>

----
#### ["A Simple but Tough-to-Beat Baseline for Sentence Embeddings"](https://openreview.net/pdf?id=SyK00v5xx) Arora, Liang, Ma
  `text embedding`
  - <https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#a-simple-but-tough-to-beat-baseline-for-sentence-embeddings-arora-liang-ma>

----
#### ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
  `sequence transduction` `Transformer`
  - <https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#attention-is-all-you-need-vaswani-shazeer-parmar-uszkoreit-jones-gomez-kaiser-polosukhin>

#### ["Non-Autoregressive Neural Machine Translation"](https://arxiv.org/abs/1711.02281) Gu, Bradbury, Xiong, Li, Socher
  `sequence transduction` `Transformer`
>	"Existing approaches to neural machine translation condition each output word on previously generated outputs. We introduce a model that avoids this autoregressive property and produces its outputs in parallel, allowing an order of magnitude lower latency during inference. Through knowledge distillation, the use of input token fertilities as a latent variable, and policy gradient fine-tuning, we achieve this at a cost of as little as 2.0 BLEU points relative to the autoregressive Transformer network used as a teacher."  
>	"parallel decoder in addition to parallel encoder - up to 8x speed-up"  
  - `post` <https://einstein.ai/research/non-autoregressive-neural-machine-translation>

#### ["Generating Wikipedia by Summarizing Long Sequences"](https://arxiv.org/abs/1801.10198) Liu, Saleh, Pot, Goodrich, Sepassi, Kaiser, Shazeer
  `sequence transduction` `Transformer`
>	"Extractive summarization to coarsely identify salient information and a neural abstractive model to generate the article. For the abstractive model, decoder-only Transformer architecture that can scalably attend to very long sequences, much longer than typical encoder-decoder architectures used in sequence transduction."  

---
#### ["Phrase-Based & Neural Unsupervised Machine Translation"](https://arxiv.org/abs/1804.07755) Lample, Ott, Conneau, Denoyer, Ranzato
  `translation`
  - <https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#word-translation-without-parallel-data-conneau-lample-ranzato-denoyer-jegou>

#### ["Unsupervised Machine Translation Using Monolingual Corpora Only"](https://arxiv.org/abs/1711.00043) Lample, Denoyer, Ranzato
  `translation`
  - <https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#unsupervised-machine-translation-using-monolingual-corpora-only-lample-denoyer-ranzato>

#### ["Unsupervised Neural Machine Translation"](https://arxiv.org/abs/1710.11041) Artetxe, Labaka, Agirre, Cho
  `translation`
  - <https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#unsupervised-neural-machine-translation-artetxe-labaka-agirre-cho>

#### ["Word Translation Without Parallel Data"](https://arxiv.org/abs/1710.04087) Conneau, Lample, Ranzato, Denoyer, Jegou
  `translation`
  - <https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#word-translation-without-parallel-data-conneau-lample-ranzato-denoyer-jegou>

#### ["Style Transfer from Non-Parallel Text by Cross-Alignment"](https://arxiv.org/abs/1705.09655) Shen, Lei, Barzilay, Jaakkola
  `translation`
  - `video` <https://facebook.com/nipsfoundation/videos/1554741734617060?t=4850> (Shen)

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
