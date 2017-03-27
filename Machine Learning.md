Machine Learning is the subfield of computer science that gives computers the ability to learn without being explicitly programmed.  
A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.  
Anywhere there is a function y = f(x), where f(x) has some parameters used to make a decision, prediction, or estimate, has the potential to be replaced by a machine learning algorithm.  


  * [overview](#overview)
  * [study](#study)
  * [theory](#theory)
  * [methods](#methods)
  * [representation learning](#representation-learning)
  * [inductive programming](#inductive-programming)
  * [automated machine learning](#automated-machine-learning)
  * [interesting quotes](#interesting-quotes)
  * [interesting papers](#interesting-papers)
    - [automated machine learning](#interesting-papers---automated-machine-learning)
    - [systems](#interesting-papers---systems)

----

  [deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md)  
  [reinforcement learning](https://dropbox.com/s/dexryjnmxujdynd/Reinforcement%20Learning.txt)  
  [bayesian inference and learning](https://dropbox.com/s/7vlg0vhb51rd6c1/Bayesian%20Inference%20and%20Learning.txt)  
  [probabilistic programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md)  


[recent breakthroughs](https://dropbox.com/s/cfjjubx8ocfx748/breakthroughs.md) (in russian)



---
### overview

#### non-technical overview

  "What Is Machine Learning" by Pedro Domingos - <https://class.coursera.org/machlearning-001/lecture/145>

  "Machine Learning is the new algorithms" by Hal Daume - <http://nlpers.blogspot.ru/2014/10/machine-learning-is-new-algorithms.html>  
  "When is Machine Learning Worth It?" - <http://inference.vc/when-is-machine-learning-worth-it/>  

  "Deep Learning - Introduction" by Nando de Freitas - <https://youtube.com/watch?v=PlhFWT7vAEw>  
  "Machine Learning - Introduction" by Nando de Freitas - <https://youtube.com/watch?v=w2OtwL5T1ow>  

  <http://machinelearningmastery.com/start-here/>  


#### technical overview

  introduction by Pedro Domingos - <https://class.coursera.org/machlearning-001/lecture/149>  
  introduction (in russian) by Dmitry Vetrov - <http://youtube.com/watch?v=srIcbDBAJBo> + <http://youtube.com/watch?v=ftlbxFypW74>  

  <http://bugra.github.io/work/notes/2014-08-23/on-machine-learning/>  
  <https://thebeautyofml.wordpress.com/2016/04/03/in-a-nutshell-learning/>  

  "Machine Learning: Trends, Perspectives and Prospects" by Jordan and Mitchell - <https://goo.gl/U8552O>  

  overview (in russian) by Dmitry Vetrov - <http://youtube.com/watch?v=lkh7bLUc30g>  
  overview (in russian) by Igor Kuralenok - <http://youtube.com/watch?v=ynS7XvkAdLU> + <http://youtube.com/watch?v=jiyD0r2SC-g>  

  <http://thetalkingmachines.com/blog/>  (podcasts)


#### applications

  artificial intelligence    - <https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md> + <https://dropbox.com/sh/gmo2hort07gsydj/AACK0XoOOLsrzCyYC1eLwyOwa>  
  knowledge representation and reasoning    - <https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md> + <https://dropbox.com/sh/9ytscy4pwegbvhb/AACtB7tQGj-vigo0yExfciu0a>  
  natural language processing    - <https://dropbox.com/s/0kw1s9mrrcwct0u/Natural%20Language%20Processing.txt> + <https://dropbox.com/sh/rb7u9nwb16bg5xq/AADV3d_bS6-mqFW0_jaec1sZa>  
  information retrieval    - <https://dropbox.com/s/21ugi2p9uy1shvt/Information%20Retrieval.txt> + <https://dropbox.com/sh/pvpzyxfcpy39j8p/AACduJ-pVF9Lh-gn3_SExj1va>  
  personal assistants    - <https://dropbox.com/s/0fyarlwcfb8mjdq/Personal%20Assistants.txt> + <https://dropbox.com/sh/veqe3c800ztpkxe/AABwH6camduJrsTUJpeKobWUa>  



---
### study

#### tutorials

  <http://frnsys.com/ai_notes/>

  <https://github.com/ujjwalkarn/Machine-Learning-Tutorials>

  <https://github.com/rasbt/python-machine-learning-book> + <https://github.com/rasbt/python-machine-learning-book/tree/master/faq>

  Python notebooks for many algorithms - <http://nbtest.herokuapp.com/github/fonnesbeck/Bios366/tree/master/notebooks/>

  <http://ciml.info/dl/v0_8/ciml-v0_8-all.pdf>  
  <http://metacademy.org>  
  <http://machinelearning.ru/wiki/>  

  <https://en.wikipedia.org/wiki/Machine_learning>
  "Introduction to Machine Learning - The Wikipedia Guide" - <https://github.com/Nixonite/open-source-machine-learning-degree/blob/master/Introduction%20to%20Machine%20Learning%20-%20Wikipedia.pdf>


#### guides

  things to know - <http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf>  
  Google's rules - <http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf>  
  common pitfalls - <http://danielnee.com/?p=155&utm_content=buffer163ed>  
  methods of overfitting - <http://hunch.net/?p=22>  
  more data or better algorithms - <http://kdnuggets.com/2015/06/machine-learning-more-data-better-algorithms.html>  
  fitting models with more parameters than data points - <https://jakevdp.github.io/blog/2015/07/06/model-complexity-myth/>  
  cross-validation - <http://robjhyndman.com/hyndsight/crossvalidation/>  
  priors for modelling invariances - <http://inference.vc/the-holy-gr/>  
  baseline - <http://nlpers.blogspot.ru/2014/11/the-myth-of-strong-baseline.html>  
  feature selection - <http://machinelearningmastery.com/an-introduction-to-feature-selection/>  
  algorithm selection - <http://machinelearningmastery.com/a-data-driven-approach-to-machine-learning/>  
  dimensionality reduction - <https://colah.github.io/posts/2014-10-Visualizing-MNIST/>  
  hyper-parameter selection - <http://startup.ml/blog/hyperparam>  
  hyper-parameter search - <http://nlpers.blogspot.ru/2014/10/hyperparameter-search-bayesian.html>  
  distributed machine learning - <http://fastml.com/the-emperors-new-clothes-distributed-machine-learning/>  
  deception of supervised learning - <http://kdnuggets.com/2016/09/deception-of-supervised-learning.html>  
  classification vs prediction - <http://fharrell.com/2017/01/classification-vs-prediction.html>  
  machine learning and statistics - <https://www.ics.uci.edu/~welling/publications/papers/WhyMLneedsStatistics.pdf>  
  machine learning and engineering - <http://machinedlearnings.com/2017_02_01_archive.html>  


#### courses

  <http://youtube.com/playlist?list=PLE6Wd9FR--Ecf_5nCbnSQMHqORpiChfJf>  (Nando de Freitas - undergraduate course)  
  <http://www.cs.ubc.ca/~nando/540-2013/lectures.html>  (Nando de Freitas - graduate course)  
  <http://coursera.org/course/machlearning>  (Pedro Domingos)  
  <http://alex.smola.org/teaching/10-701-15/>  (Alex Smola)  
  <http://youtube.com/playlist?list=PLD0F06AA0D2E8FFBA>  (MathematicalMonk)  
  <http://www.cs.toronto.edu/~rsalakhu/STA4273_2015/lectures.html>  (Ruslan Salakhutdinov)  
  <https://see.stanford.edu/Course/CS229/47>  (Andrew Ng)  
  <http://videolectures.net/course_information_theory_pattern_recognition/>  (David MacKay)  
  <http://homepages.inf.ed.ac.uk/vlavrenk/iaml.html>  (Victor Lavrenko, 60 short lectures on single page)  
  <https://www.inf.ed.ac.uk/teaching/courses/iaml/>  (Victor Lavrenko)  

  <http://coursera.org/specializations/machine-learning-data-analysis/>  (Yandex, in russian)

  <http://coursera.org/learn/introduction-machine-learning>  (Konstantin Vorontsov, in russian)  
  <http://youtube.com/playlist?list=PLJOzdkh8T5kp99tGTEFjH_b9zqEQiiBtC>  (Konstantin Vorontsov, in russian)  
  <http://youtube.com/playlist?list=PLJOzdkh8T5kp99tGTEFjH_b9zqEQiiBtC>  (Igor Kuralenok, in russian)  
  <http://lektorium.tv/course/22975>  (Igor Kuralenok, in russian)  
  <http://habrahabr.ru/company/mailru/blog/254897/>  (in russian)  

  <http://dataschool.io/15-hours-of-expert-machine-learning-videos/>  
  <https://cs.cmu.edu/~tom/10701_sp11/lectures.shtml> + <http://cs.cmu.edu/~aarti/Class/10601/lectures.shtml>  
  <https://edx.org/course/artificial-intelligence-uc-berkeleyx-cs188-1x>  
  <https://edx.org/course/caltechx/caltechx-cs1156x-learning-data-2516>  
  <http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/>  
  <http://www.cs.nyu.edu/~mohri/ml14/>  

  final projects from Stanford CS231 Deep Learning class - <http://cs231n.stanford.edu/reports2016.html> + <http://cs231n.stanford.edu/reports.html>  
  final projects from Stanford CS224d Deep Learning for Natural Language Processing - <http://cs224d.stanford.edu/reports.html>  
  final projects from CMU 10-701 - <https://youtube.com/playlist?list=PLZSO_6-bSqHTw6zw5XhxawMgYGRItZ4Zg>  


#### schools

  <http://mlss.cc>  
  <http://videolectures.net/site/search/?q=MLSS>  
  <http://youtube.com/channel/UCty-pPOWlWUk4gXNm5pydcg/videos>  (Tubingen 2015)  
  <http://youtube.com/channel/UCT1k2e63pqm_VSXmaF21n6g>  (Sydney 2015)  
  <http://youtube.com/playlist?list=PLZSO_6-bSqHQCIYxE3ycGLXHMjK3XV7Iz>  (Pittsburgh 2014)  


#### books

  Pedro Domingos - ["The Master Algorithm"](http://basicbooks.com/full-details?isbn=9780465065707)  
  John Winn, Chris Bishop - ["Model-Based Machine Learning"](http://mbmlbook.com/toc.html)  
  Max Welling - ["A First Encounter with Machine Learning"](https://www.ics.uci.edu/~welling/teaching/ICS273Afall11/IntroMLBook.pdf)  
  Ian Goodfellow, Yoshua Bengio, Aaron Courville - ["Deep Learning"](http://deeplearningbook.org)  
  Richard Sutton, Andrew Barto - ["Reinforcement Learning: An Introduction"](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) +
	[second edition](https://dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf)  
  Tom Mitchell - ["Machine Learning"](https://goo.gl/tyNHMH)  
  Shai Shalev-Shwartz, Shai Ben-David - ["Understanding Machine Learning: From Theory to Algorithms"](https://goo.gl/tHmsdD)  
  Chris Bishop - ["Pattern Recognition and Machine Learning"](https://goo.gl/58Yvvp)  
  Trevor Hastie, Robert Tibshirani, Jerome Friedman - ["The Elements of Statistical Learning"](http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf)  
  Kevin Murphy - ["Machine Learning - A Probabilistic Perspective"](https://goo.gl/Vh7Jje)  
  David MacKay - ["Information Theory, Inference, and Learning Algorithms"](http://users.aims.ac.za/~mackay/itila/book.html)  
  David Barber - ["Bayesian Reasoning and Machine Learning"](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.Online)  
  Mehryar Mohri - ["Foundations of Machine Learning"](http://www.cs.nyu.edu/~mohri/mlbook/)  
  Ron Bekkerman, Mikhail Bilenko, John Langford - ["Scaling Up Machine Learning: Parallel and Distributed Approaches"](https://goo.gl/dE7jPb)  


#### blogs

  <http://inference.vc>  
  <http://blog.shakirm.com>  
  <http://offconvex.org>  
  <http://argmin.net>  
  <http://hunch.net>  
  <http://machinedlearnings.com>
  <http://nlpers.blogspot.com>  
  <http://inverseprobability.com/blog.html>  
  <http://blogs.princeton.edu/imabandit>  
  <http://fastml.com>  
  <http://wildml.com>  
  <http://thetalkingmachines.com>  


#### news and discussions

  <https://reddit.com/r/MachineLearning/>


#### code

  <https://github.com/josephmisiti/awesome-machine-learning>


#### datasets

  <https://github.com/caesar0301/awesome-public-datasets>



---
### theory

**goals**
  - prediction with high accuracy
  - prediction of "rare" events (outlier detection)
  - interpretable modeling
  - hypothesis testing
  - visualization
  - drawing causal conclusions
  - quantifying uncertainty/risk
  - generating new samples
  - clustering
  - online/active/reinforcement learning

**data**
  - text
  - networks
  - time series
  - streaming data
  - bag data or full distributions
  - paired samples
  - clinical trial
  - A/B testing
  - spatial data
  - high-dimensional data
  - low-sample-size data 
  - non-metric-space data

**challenges**
  - How to decide autonomously which representation is best for target knowledge?
  - How to tell genuine regularities from chance occurrences?
  - How to exploit pre-existing domain knowledge knowledge?
  - How to learn with limited computational resources?
  - How to learn with limited data?
  - How to make learned results understandable?
  - How to quantify uncertainty?
  - How to take into account the costs of decisions?
  - How to handle non-indepedent and non-stationary data?

[**things to know**](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
  - it's generalization that counts
  - data alone is not enough
  - overfitting has many faces
  - intuition fails in high dimensions
  - theoretical guarantees are not what they seem
  - feature engineering is the key
  - more data beats a cleverer algorithm
  - learn many models, not just one (ensembles)
  - simplicity does not imply accuracy
  - representable does not imply learnable
  - correlation does not imply causation

----

#### bayesian framework

  [Bayesian Inference and Learning](https://dropbox.com/s/7vlg0vhb51rd6c1/Bayesian%20Inference%20and%20Learning.txt)

  - probability as measure of uncertainty, encodes ignorance in terms of distributions
  - treats everything as random variables, no difference between random and unknown variables
  - makes use of Bayes theorem: posterior = likelihood * prior / evidence
  - possible to compute the estimate p(U|O) for arbitrary unknown variable (U) given observed data (O) and not having any knowledge about latent variables (L) from the joint distribution p(U, O, L)
  - possibility to use posterior distributions as priors, combination of multiple models

"From a Bayesian point of view, we should be integrating over likelihoods instead of using optimization methods to select a point estimate of model parameters (usually with ad hoc regularization tuned by cross validation)."

"No free lunch theorem relies on unrealistic uniform sampling (which results mostly in white noise for real-world phenomena). Sampling from Solomonoff's universal distribution permits free lunch."

----

#### learning theory

  - what does it mean to learn  
  - when is a concept/function learnable  
  - how much data do we need to learn something  
  - how can we make sure what we learn will generalize to future data  

frameworks:

  - Statistical learning theory  
  - PAC learning or PAC-Bayes  
  - Minimax estimation (estimation/decision theory)  

  <http://kdnuggets.com/2015/07/deep-learning-triumph-empiricism-over-theoretical-mathematical-guarantees.html>  
  <https://hips.seas.harvard.edu/blog/2013/02/15/learning-theory-purely-theoretical/>  

  <https://mostafa-samir.github.io/ml-theory-pt1/>  
  <https://mostafa-samir.github.io/ml-theory-pt2/>  
  <https://mostafa-samir.github.io/ml-theory-pt3/>  

  <http://jeremykun.com/2014/01/02/probably-approximately-correct-a-formal-theory-of-learning/>  
  <http://jeremykun.com/2014/04/21/an-un-pac-learnable-problem/>  
  <http://jeremykun.com/2014/09/19/occams-razor-and-pac-learning/>  

  <https://blogs.princeton.edu/imabandit/2015/10/13/crash-course-on-learning-theory-part-1/>  
  <https://blogs.princeton.edu/imabandit/2015/10/22/crash-course-on-learning-theory-part-2/>  

  <https://web.stanford.edu/class/cs229t/notes.pdf> (course by Percy Liang)

----

#### bias/variance

  "The bias is error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).  
   The variance is error from sensitivity to small fluctuations in the training set. High variance can cause overfitting: modeling the random noise in the training data, rather than the intended outputs."  

  "The bias–variance tradeoff is a central problem in supervised learning. Ideally, one wants to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. High-variance learning methods may be able to represent their training set well, but are at risk of overfitting to noisy or unrepresentative training data. In contrast, algorithms with high bias typically produce simpler models that don't tend to overfit, but may underfit their training data, failing to capture important regularities."

  "Bias is how well the best hypothesis in your hypothesis class would perform in reality, whereas variance is how much performance degradation is introduced from having finite training data."

  "Bias is a learner's tendency to consistently learn the same wrong thing."  
  "Variance is the tendency to learn random things irrespective of the real signal."  

  "Bias-variance decomposition in understanding the prediction error incurred by statistical models:
  - the bias component of prediction error reflects the inability of a model to represent the systematic patterns that govern the observations
  - the variance component of prediction error reflects the sensitivity of the model’s predictions to different observations of the same problem

Together, bias and variance additively contribute to the total prediction error:  mean squared error = (bias)^2 + error variance + noise  
  Bias-variance tradeoff characterizes how robust algorithm is to errors in its modeling assumptions (bias) or to errors in the training data (variance)."  



---
### methods

  *machine learning* = *representation* + *evaluation* + *optimization*

  *representation*:  A classifier/regressor must be represented in some formal language that the computer can handle. Conversely, choosing a representation for a learner is tantamount to choosing the set of classifiers that it can possibly learn. This set is called the hypothesis space of the learner. If a classifier is not in the hypothesis space, it cannot be learned. A related question is how to represent the input, i.e., what features to use.

  *evaluation*:  An evaluation function (also called objective function or scoring function) is needed to distinguish good classifiers from bad ones. The evaluation function used internally by the algorithm may differ from the external one that we want the classifier to optimize, for ease of optimization and other issues.

  *optimization*:  Finally, we need a method to search among the classifiers in the language for the highest-scoring one. The choice of optimization technique is key to the efficiency of the learner, and also helps determine the classifier produced if the evaluation function has more than one optimum. It is common for new learners to start out using off-the-shelf optimizers, which are later replaced by custom-designed ones.

*representation*:
  - instances
    * k-nearest neighbor
    * support vector machines
  - hyperplanes
    * naive Bayes
    * logistic regression
  - decision trees
  - sets of rules
    * propositional rules
    * logic programs
  - neural networks
  - graphical models
    * bayesian networks
    * conditional random fields

*evaluation*:
  - accuracy/error rate
  - precision/recall
  - squared error
  - likelihood
  - posterior probability
  - information gain
  - K-L divergence
  - cost/utility
  - margin

*optimization*:
  - combinatorial optimization
    * greedy search
    * beam search
    * branch-and-bound
  - unconstrained continuous optimization
    * gradient descent
    * conjugate gradient
    * quasi-Newton methods
  - constrained continuous optimization
    * linear programming
    * quadratic programming

----

  *machine learning* = *experience obtaining* + *cost function* + *decision function*

*experience obtaining*:
  - transductive learning
  - inductive learning
  - stochastic optimization
  - active learning
  - budget learning
  - online learning
  - multi-armed bandits
  - reinforcement learning

*cost function*:
  - supervised
    * classification
    * regression
    * learning to rank
    * metric learning
  - unsupervised
    * cluster analysis
    * dimensionality reduction
    * representation learning
  - semi-supervised
    * conditional clustering
    * transfer learning

*decision function*:
  - linear desions
    * linear regression, logistic regression
    * LDA/QDA
    * LASSO 
    * SVM
    * LSI
  - graphs
    * Markov chains, Hidden Markov Models
    * Probabilistic Graphical Models
    * Conditional Random Fields
  - artificial neural networks
    * Multilayer Perceptron
    * Hopfield net
    * Kohonen net
  - parametric family functions
    * sampling
    * genetic algorithms
    * PLSI
    * LDA
  - instance based learning
    * KNN
    * DANN
  - predicates
    * logic rules
    * decision trees
  - ensembles
    * bagging    
    * boosting
    * bayesian averaging
    * stacking

----

  <http://en.wikipedia.org/wiki/List_of_machine_learning_algorithms>

  <https://justindomke.wordpress.com/2015/09/14/algorithmic-dimensions/>  
  <https://www.cs.jhu.edu/~jason/tutorials/ml-simplex.html>  

  "All Models of Learning have Flaws" by John Langford - <http://hunch.net/?p=224>

  <https://azure.microsoft.com/en-us/documentation/articles/machine-learning-algorithm-cheat-sheet/>

  <http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/>

  <http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html>  
  <http://eferm.com/wp-content/uploads/2011/05/cheat3.pdf>  
  <https://github.com/soulmachine/machine-learning-cheat-sheet/blob/master/machine-learning-cheat-sheet.pdf>  

  <http://dataschool.io/comparing-supervised-learning-algorithms/>

----

  [tutorials](#tutorials) [guides](#guides) [courses](#courses) [books](#books)



---
### representation learning

  "Reresentation is a formal system which makes explicit certain entities and types of information, and which can be operated on by an algorithm in order to achieve some information processing goal. Representations differ in terms of what information they make explicit and in terms of what algorithms they support. As example, Arabic and Roman numerals - the fact that operations can be applied to particular columns of Arabic numerals in meaningful ways allows for simple and efficient algorithms for addition and multiplication."

  "In representation learning, our goal isn’t to predict observables, but to learn something about the underlying structure. In cognitive science and AI, a representation is a formal system which maps to some domain of interest in systematic ways. A good representation allows us to answer queries about the domain by manipulating that system. In machine learning, representations often take the form of vectors, either real- or binary-valued, and we can manipulate these representations with operations like Euclidean distance and matrix multiplication."

  "In representation learning, the goal isn’t to make predictions about observables, but to learn a representation which would later help us to answer various queries. Sometimes the representations are meant for people, such as when we visualize data as a two-dimensional embedding. Sometimes they’re meant for machines, such as when the binary vector representations learned by deep Boltzmann machines are fed into a supervised classifier. In either case, what’s important is that mathematical operations map to the underlying relationships in the data in systematic ways."

  <https://hips.seas.harvard.edu/blog/2013/02/04/predictive-learning-vs-representation-learning/>  
  <https://hips.seas.harvard.edu/blog/2013/02/25/what-is-representation-learning/>  

----

  [Deep Learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md)  
  [Knowledge Representation and Reasoning](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md)  
  [Probabilistic Programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md)  



---
### inductive programming

  "The essence of programmatic representations is that they are well-specified, compact, combinatorial and hierarchical.
  - well-specified:  Unlike sentences in natural language, programs are unambiguous, although two distinct programs can be precisely equivalent.
  - compact:  Programs allow us to compress data on the basis of their regularities.
  - combinatorial:  Programs can access the results of running other programs, as well as delete, duplicate, and rearrange these results.
  - hierarchical:  Programs have an intrinsic hierarchical organization and may be decomposed into subprograms."

"Alternative representations for procedural abstractions such as neural networks have serious downsides including opacity and inefficiency."

"Challenges with programmatic representations:
  - open-endedness:  In contrast to other knowledge representations in machine learning, programs may vary in size and “shape”, and there is no obvious problem-independent upper bound on program size. This makes it difficult to represent programs as points in a fixed-dimensional space, or learn programs with algorithms that assume such a space.
  - over-representation:  Often syntactically distinct programs will be semantically identical (i.e. represent the same underlying behavior or functional mapping). Lacking prior knowledge, many algorithms will inefficiently sample semantically identical programs repeatedly.
  - chaotic execution: Programs that are very similar, syntactically, may be very different, semantically. This presents difficulty for many heuristic search algorithms, which require syntactic and semantic distance to be correlated.
  - high resource-variance:  Programs in the same space may vary greatly in the space and time they require to execute."

"Limitations of program learning:
  - can't overrule no-free-lunch
    * averaged over all possible scoring functions
  - can't learn to model "arbitrary" Turing machines
  - can't scale up to large programs
    * without external guidance
    * without strong structural inductive bias
    * without relatedness to past problems"

----

  (Nando de Freitas) "For me there are two types of generalisation, which I will refer to as Symbolic and Connectionist generalisation. If we teach a machine to sort sequences of numbers of up to length 10 or 100, we should expect them to sort sequences of length 1000 say. Obviously symbolic approaches have no problem with this form of generalisation, but neural nets do poorly. On the other hand, neural nets are very good at generalising from data (such as images), but symbolic approaches do poorly here. One of the holy grails is to build machines that are capable of both symbolic and connectionist generalisation. Neural Programmer Interpreters is a very early step toward this. NPI can do symbolic operations such as sorting and addition, but it can also plan by taking images as input and it's able to generalise the plans to different images (e.g. in the NPI car example, the cars are test set cars not seen before)."

----

  Neural Abstract Machines & Program Induction workshop @ NIPS 2016 - <https://uclmr.github.io/nampi/>


  <http://cacm.acm.org/magazines/2015/11/193326-inductive-programming-meets-the-real-world/fulltext>  
  <http://homes.cs.washington.edu/~bornholt/post/synthesis-for-architects.html>  

  <http://languagengine.co/blog/symbolic-machine-learning/>

----

  [Probabilistic Programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md)  
  [selected papers](https://dropbox.com/sh/vrr1gs798zy02n1/AACj7hlXOiRt1nXltXVC-2Wca)  



---
### automated machine learning

AutoML aims to automate many different stages of the machine learning process:
  - model selection, hyper-parameter optimization, and model search
  - meta learning and transfer learning
  - representation learning and automatic feature extraction / construction
  - demonstrations (demos) of working AutoML systems
  - automatic generation of workflows / workflow reuse
  - automatic problem "ingestion" (from raw data and miscellaneous formats)
  - automatic feature transformation to match algorithm requirements
  - automatic detection and handling of skewed data and/or missing values
  - automatic acquisition of new data (active learning, experimental design)
  - automatic report writing (providing insight on automatic data analysis)
  - automatic selection of evaluation metrics / validation procedures
  - automatic selection of algorithms under time/space/power constraints
  - automatic prediction post-processing and calibration
  - automatic leakage detection
  - automatic inference and differentiation
  - user interfaces for AutoML

problems:
  - different data distributions: the intrinsic/geometrical complexity of the dataset.
  - different tasks: regression, binary classification, multi-class classification, multi-label classification.
  - different scoring metrics: AUC, BAC, MSE, F1, etc
  - class balance: Balanced or unbalanced class proportions.
  - sparsity: Full matrices or sparse matrices.
  - missing values: Presence or absence of missing values.
  - categorical variables: Presence or absence of categorical variables.
  - irrelevant variables: Presence or absence of additional irrelevant variables (distractors).
  - number Ptr of training examples: Small or large number of training examples.
  - number N of variables/features: Small or large number of variables.
  - aspect ratio Ptr/N of the training data matrix: Ptr >> N, Ptr = N or Ptr << N.

----

  "We can put a unified framework around the various approaches. Borrowing from the conventional classification of feature selection methods, model search strategies can be categorized into filters, wrappers, and embedded methods.  
  Filters are methods for narrowing down the model space, without training the learning machine. Such methods include preprocessing, feature construction, kernel design, architecture design, choice of prior or regularizers, choice of a noise model, and filter methods for feature selection. Although some filters use training data, many incorporate human prior knowledge of the task or knowledge compiled from previous tasks (a form of meta learning or transfer learning). Recently, it has been proposed to apply collaborative filtering methods to model search.  
  Wrapper methods consider the learning machine as a black-box capable of learning from examples and making predictions once trained. They operate with a search algorithm in hyper-parameter space (for example grid search or stochastic search) and an evaluation function assessing the trained learning machine performances (for example the cross-validation error or the Bayesian evidence).  
  Embedded methods are similar to wrappers, but they exploit the knowledge of the learning machine algorithm to make the search more efficient. For instance, some embedded methods compute the leave-one-out solution in a closed form, without leaving anything out, i.e., by performing a single model training on all the training data. Other embedded methods jointly optimize parameters and hyperparameters."  

----

  TPOT Data Science Assistant - <http://rhiever.github.io/tpot/>

  "The Automatic Statistician" by Zoubin Ghahramani - <https://youtu.be/H7AMB0oo__4?t=53m20s> + <http://webdav.tuebingen.mpg.de/mlss2013/2015/slides/ghahramani/mlss15future.pdf>

  DARPA's Data-Driven Discovery of Models (D3M) - <http://www.darpa.mil/news-events/2016-06-17>
>	"The goal of D3M is to help overcome the data-science expertise gap by enabling non-experts to construct complex empirical models through automation of large parts of the model-creation process. If successful, researchers using D3M tools will effectively have access to an army of “virtual data scientists”."

>	"The construction of empirical models today is largely a manual process, requiring data experts to translate stochastic elements, such as weather and traffic, into models that engineers and scientists can then ask questions of. We have an urgent need to develop machine-based modeling for users with no data-science background. We believe it’s possible to automate certain aspects of data science, and specifically to have machines learn from prior example how to construct new models."

----

  "Why Tool AIs Want to Be Agent AIs" - <http://www.gwern.net/Tool%20AI>  

  "Roughly, we can try to categorize the different kinds of agentiness by level of neural network they work on. There are:  

  - actions internal to a computation
    * inputs
    * intermediate states
    * accessing the external environment
    * amount of computation
    * enforcing constraints/finetuning quality of output
    * changing the loss function applied to output
  - actions internal to training the neural network
    * the gradient itself
    * size & direction of gradient descent steps on each parameter
    * overall gradient descent learning rate and learning rate schedule
    * choice of data samples to train on
  - internal to the neural network design step
    * hyperparameter optimization
    * neural network architecture
  - internal to the dataset
    * active learning
    * optimal experiment design
  - internal to interaction with environment
    * adaptive experiment
    * multi-armed bandit
    * exploration for reinforcement learning"

"The logical extension of these neural networks all the way down papers is that an actor like Google/Baidu/Facebook/MS could effectively turn neural networks into a black box: a user/developer uploads through an API a dataset of input/output pairs of a specified type and a monetary loss function, and a top-level neural network running on a large GPU cluster starts autonomously optimizing over architectures & hyperparameters for the neural network design which balances GPU cost and the monetary loss, interleaved with further optimization over the thousands of previous submitted tasks, sharing its learning across all of the datasets/loss functions/architectures/hyperparameters, and the original user simply submits future data through the API for processing by the best neural network so far."



---
### interesting quotes

  ():
  > "The huge role played by random (or seemingly random due to incomplete available information) events as fundamental forces which dictate our life experience clearly demonstrates the universality and importance of randomness. Just as classical physics is the precise (i.e. mathematical) language used to describe our world at the macro-level, probability is the precise language used to deal with such uncertainty. Now, as human beings without direct access to the underlying forces behind different phenomena, we can only observe/sample events, from which we may try and construct "models" which capture some elements of the underlying probability distributions of interest. Call this problem statistics, ML, data science/mining or whatever you want, but it is simply the extension of the previous scientific paradigm (using differential equations to deterministically explain & predict natural phenomena in a precise mathematical manner) to more complicated problems in which uncertainty is inherent; typically because we cannot measure all relevant quantities (the # of quantities relevant to the phenomena tends to increase with the complexity of the system). For example, if we wish to predict how far a thrown ball travels from the force/angle of the toss, Newtonian physics offers a diff-eq-based formula which most would deem adequate, but given data on a huge number of throws, a learning algorithm could actually offer better performance. This is because it would properly account for the uncertainty in distance-traveled due to spin of the ball, air resistance, and other unmeasured quantities, while simultaneously learning a distance-traveled vs force/angle function which would be similar to the theoretical one obtained from classical mechanics."

  ():
  > "Imagine if back in Newton's day, they were analyzing data from physical random variables with deep nets. Sure, they might get great prediction accuracy on how far a ball will go given measurements of its weight, initial force/angle, and some other irrelevant variables, but would this really be the best approach to discover all of the useful laws of physics such as f = ma and the conversion from potential to kinetic energy via the gravitational constant? Probably not, in fact the predictions might be in some sense "too good" incorporating other confounding effects such as air drag and the shape / spin of the ball which obfuscate the desired law. In many settings where an interpretation of what is going on in the data is desired, a clear model is necessary with simple knobs that have clear effects when turned. This may also be a requirement not only for human interpretation, but an also AI system which is able to learn and combine facts about the world (rather than only storing the complex functions which represent the relationships between things as inferred by a deep-net)."

  Ferenc Huszar:
  > "My favourite theoretical machine learning papers are ones that interpret heuristic learning algorithms in a probabilistic framework, and uncover that they in fact are doing something profound and meaningful. Being trained as a Bayesian, what I mean by profound typically means statistical inference or fitting statistical models. An example would be the k-means algorithm. K-means intuitively makes sense as an algorithm for clustering. But we only really understand what it does when we make the observation that it actually is a special case of expectation-maximisation in gaussian mixture models. This interpretation as special case of something allows us to understand the expected behaviour of the algorithm better. It will allow us to make predictions about the situations in which it's likely to fail, and to meaningfully extend it to situations it doesn't handle well."

  Yann LeCun:
  > "I think if it were true that P=NP or if we had no limitations on memory and computation, AI would be a piece of cake. We could just brute-force any problem. We could go "full Bayesian" on everything (no need for learning anymore - everything becomes Bayesian marginalization). But the world is what it is."

  Ferenc Huszar:
  > "There is no such thing as learning without priors. In the simplest form, the objective function of the optimisation is a prior - you tell the machine that it's goal is to minimise mean squared error for example. The machine solves the optimisation problem (typically) you tell it to solve, and good machine learning is about figuring out what that problem is. Priors are part of that. Secondly, if you think about it, it is actually a tiny portion of machine learning problems where you actually have enough data to get away without engineering better priors or architectures by just using a model which is highly flexible. Today, you can do this in visual, audio, video domain because you can collect and learn from tonnes of examples and particularly because you can use unsupervised or semi-supervised learning to learn natural invariances. An example is chemistry: if you want to predict certain properties of chemicals, it almost doesn't make sense to use data only to make the machine learn what a chemical is, and what the invariances are - doing that would be less accurate and a lot harder than giving it the required context. Un- and semi-supervised learning doesn't make sense because in many cases learning about the natural distribution of chemicals (even if you had a large dataset of this) may be uninformative of the prediction tasks you want to solve."

  Ferenc Huszar:
  > "My belief is that speeding up computation is not fast enough, you do need priors to beat the curse of dimensionality. Think rotational invariance. Yes, you can model that by allowing enough flexibility in a neural netowrk to learn separate representations for all possible rotations of an object, but you're exponentially more efficient if you can somehow 'integrate out' the invariance by designing the architecture/maths cleverly. By modeling invariances correctly, you can make exponential leaps in representational capacity of the network - on top of the exponential growth in computing power that'd kind of a given. I don't think the growth in computing power is fast enough to make progress in machine learning for real-world hard tasks. You need that, combined with exponential leaps on top of that, made possible by building in prior knowledge correcltly."

  Ilya Sutskever:
  > "Generalization means that the gap between the training and the test error is small. So for example, a very bad model that has similar training and test errors does not overfit, and hence generalizes, according to the way I use these concepts. It follows that generalization is easy to achieve whenever the capacity of the model (as measured by the number of parameters or its VC-dimension) is limited - we merely need to use more training cases than the model has parameters / VC dimension. Thus, the difficult part is to get a low training error."

  Jason Brownlee:
  > "Model performance is estimated in terms of its accuracy to predict the occurrence of an event on unseen data. A more accurate model is seen as a more valuable model. Model interpretability provides insight into the relationship between in the inputs and the output. An interpreted model can answer questions as to why the independent features predict the dependent attribute. The issue arrises because as model accuracy increases so does model complexity, at the cost of interpretability. The optimization of accuracy leads to further increases in the complexity of models in the form of additional model parameters (and resources required to tune those parameters). A model with fewer parameters is easier to interpret. A linear regression model has a coefficient per input feature and an intercept term. Moving to logistic regression gives more power in terms of the underlying relationships that can be modeled at the expense of a function transform to the that now too must be understood along with the coefficients. A decision tree (of modest size) may be understandable, a bagged decision tree requires a different perspective to interpret why an event is predicted to occur. Pushing further, the optimized blend of multiple models into a single prediction may be beyond meaningful or timely interpretation."

  John D. Cook:
  > "Simple models often outperform complex models in complex situations. He cites as examples sports prediction, diagnosing heart attacks, locating serial criminals, picking stocks, and  understanding spending patterns. Complex environments often instead call for simple decision rules. That is because these rules are more robust to ignorance. And yet behind every complex set of rules is a paper showing that it outperforms simple rules, under conditions of its author’s choosing. That is, the person proposing the complex model picks the scenarios for comparison. Unfortunately, the world throws at us scenarios not of our choosing. Simpler methods may perform better when model assumptions are violated. And model assumptions are always violated, at least to some extent."

  Yoshua Bengio:
  > "Whereas other nonparametric learning algorithms also suffer from the curse of dimensionality, the way in which the problem appears in the case of decision trees is different and helps to focus on the fundamental difficulty. The general problem is not really dimensionality, nor is it about a predictor that is a sum of purely local terms (like kernel machines). The problem arises from dividing the input space in regions (in a hard way in the case of decision trees) and having separate parameters for each region. Unless the parameters are tied in some way or regularized using strong prior knowledge, the number of available examples thus limits the complexity one can capture, that is, the number of independent regions that can be distinguished. Decision trees and many other machine learning algorithms are doomed to generalize poorly because they partition the input space and then allocate separate parameters to each region. Thus no generalization to new regions or across regions. No way you can learn a function which needs to vary across a number of distinguished regions that is greater than the number of training examples. Neural nets do not suffer from that and can generalize "non-locally" because each parameter is re-used over many regions (typically HALF of all the input space, in a regular neural net)."

  Juergen Schmidhuber:
  > "A naive Bayes classifier will assume data elements are statistically independent random variables and therefore fail to produce good results. If the data are first encoded in a factorial way, then the naive Bayes classifier will achieve its optimal performance. Thus, factorial code can be seen as an ultimate unsupervised learning approach - predictors and binary feature detectors, each receiving the raw data as an input. For each detector there is a predictor that sees the other detectors and learns to predict the output of its own detector in response to the various input vectors or raw data. But each detector uses a machine learning algorithm to become as unpredictable as possible. The global optimum of this objective function corresponds to a factorial code represented in a distributed fashion across the outputs of the feature detectors."

  Jonathan Huggins:
  > "There are two main flavors of learning theory, statistical learning theory (StatLT) and computational learning (CompLT). StatLT originated with Vladimir Vapnik, while the canonical example of CompLT, PAC learning, was formulated by Leslie Valiant. StatLT, in line with its “statistical” descriptor, focuses on asymptotic questions (though generally based on useful non-asymptotic bounds). It is less concerned with computational efficiency, which is where CompLT comes in. Computer scientists are all about efficient algorithms (which for the purposes of theory essentially means polynomial vs. super-polynomial time). Generally, StatLT results apply to a wider variety of hypothesis classes, with few or no assumptions made about the concept class (a concept class refers to the class of functions to which the data generating mechanism belongs). CompLT results apply to very specific concept classes but have stronger performance guarantees, often using polynomial time algorithms."

  Michael I. Jordan:
  > "Throughout the eighties and nineties, it was striking how many times people working within the "ML community" realized that their ideas had had a lengthy pre-history in statistics. Decision trees, nearest neighbor, logistic regression, kernels, PCA, canonical correlation, graphical models, K means and discriminant analysis come to mind, and also many general methodological principles (e.g., method of moments, which is having a mini-renaissance, Bayesian inference methods of all kinds, M estimation, bootstrap, cross-validation, ROC, and of course stochastic gradient descent, whose pre-history goes back to the 50s and beyond), and many many theoretical tools (large deviations, concentrations, empirical processes, Bernstein-von Mises, U statistics, etc). Of course, the "statistics community" was also not ever that well defined, and while ideas such as Kalman filters, HMMs and factor analysis originated outside of the "statistics community" narrowly defined, there were absorbed within statistics because they're clearly about inference. Similarly, layered neural networks can and should be viewed as nonparametric function estimators, objects to be analyzed statistically."

  Michael I. Jordan:
  > "In a classical database, you have maybe a few thousand people in them. You can think of those as the rows of the database. And the columns would be the features of those people: their age, height, weight, income, et cetera. Now, the number of combinations of these columns grows exponentially with the number of columns. So if you have many, many columns—and we do in modern databases—you’ll get up into millions and millions of attributes for each person. Now, if I start allowing myself to look at all of the combinations of these features—if you live in Beijing, and you ride bike to work, and you work in a certain job, and are a certain age—what’s the probability you will have a certain disease or you will like my advertisement? Now I’m getting combinations of millions of attributes, and the number of such combinations is exponential; it gets to be the size of the number of atoms in the universe. Those are the hypotheses that I’m willing to consider. And for any particular database, I will find some combination of columns that will predict perfectly any outcome, just by chance alone. If I just look at all the people who have a heart attack and compare them to all the people that don’t have a heart attack, and I’m looking for combinations of the columns that predict heart attacks, I will find all kinds of spurious combinations of columns, because there are huge numbers of them. So it’s like having billions of monkeys typing. One of them will write Shakespeare."

  Leon Bottou:
  > "When attainable, theoretical guarantees are beautiful. They reflect clear thinking and provide deep insight to the structure of a problem. Given a working algorithm, a theory which explains its performance deepens understanding and provides a basis for further intuition. Given the absence of a working algorithm, theory offers a path of attack. However, there is also beauty in the idea that well-founded intuitions paired with rigorous empirical study can yield consistently functioning systems that outperform better-understood models, and sometimes even humans at many important tasks. Empiricism offers a path forward for applications where formal analysis is stifled, and potentially opens new directions that might eventually admit deeper theoretical understanding in the future."

  Dustin Tran:
  > "It's important to reiterate that titles of "learning" and "evolving" are only titles. Just as with neural networks, the naming scheme is very loosely tied to the actual workings of the methods. At the end of the day, both are simply optimization routines with different assumptions about the underlying cost function. "Evolutionary" algorithms are often zero-order methods which apply search heuristics, and "learning" algorithms are often at least first-order in which they use gradient information. Another difference can be thought of as a similar one between MCMC and variational inference. The former guarantees finding the global optima (asymptotically, and under other "nice" assumptions) whereas the other can get stuck in local optima but can at least reach some solution very quickly."

  Claudia Perlich:
  > "If the signal to noise ratio is high, trees and neural networks tend to win logistic regression. But, if you have very noisy problems and the best model has an AUC<0.8 - logistic beats trees and neural networks almost always. Ultimately not very surprising: if the signal is too weak, high variance models get lost in the weeds. So what does this mean in practice? The type of problems I tend to deal with are super noisy with low level of predictability. Think of it in the terms of deterministic (chess) all the way to random (supposedly the stock market). Some problems are just more predictable (given the data you have) than others. And this is not a question of the algorithms but rather a conceptual statement about the world. Deep learning is really great on the other end - “Is this picture showing a cat?”. In the world of uncertainty, the bias variance tradeoff still often ends up being favorable on the side of more bias - meaning, you want a ‘simple’ very constrained model. And this is where logistic regression comes in. I personally have found it much easier to ‘beef up’ a simple linear model by adding complicated features than trying to constrain a very powerful (high variance) model class."



---
### interesting papers


[selected papers on machine learning](https://dropbox.com/sh/kpd5tvfnc29lstj/AAD3oGVCUdkoMS56_2g8Oj7_a)

[selected papers on deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers)  
[selected papers on bayesian inference and learning](https://dropbox.com/s/7vlg0vhb51rd6c1/Bayesian%20Inference%20and%20Learning.txt#interesting-papers)  
[selected papers on probabilistic programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md#interesting-papers)  
[selected papers on reinforcement learning](https://dropbox.com/s/dexryjnmxujdynd/Reinforcement%20Learning.txt#interesting-papers)  


[recent interesting papers](https://github.com/brylevkirill/notes/blob/master/recent%20papers.md)


interesting papers (see below):
  - [automated machine learning](#interesting-papers---automated-machine-learning)
  - [systems](#interesting-papers---systems)


----

#### Valiant - ["A Theory of the Learnable"](https://people.mpi-inf.mpg.de/~mehlhorn/SeminarEvolvability/ValiantLearnable.pdf)
>	"Humans appear to be able to learn new concepts without needing to be programmed explicitly in any conventional sense. In this paper we regard learning as the phenomenon of knowledge acquisition in the absence of explicit programming. We give a precise methodology for studying this phenomenon from a computational viewpoint. It consists of choosing an appropriate information gathering mechanism, the learning protocol, and exploring the class of concepts that can be learned using it in a reasonable (polynomial) number of steps. Although inherent algorithmic complexity appears to set serious limits to the range of concepts that can be learned, we show that there are some important nontrivial classes of propositional concepts that can be learned in a realistic sense."

>	"Proof that if you have a finite number of functions, say N, then every training error will be close to every test error once you have more than log N training cases by a small constant factor. Clearly, if every training error is close to its test error, then overfitting is basically impossible (overfitting occurs when the gap between the training and the test error is large)."


#### Breiman - ["Statistical Modeling: The Two Cultures"](http://projecteuclid.org/euclid.ss/1009213726)
>	"There are two cultures in the use of statistical modeling to reach conclusions from data. One assumes that the data are generated by a given stochastic data model. The other uses algorithmic models and treats the data mechanism as unknown. The statistical community has been committed to the almost exclusive use of data models. This commitment has led to irrelevant theory, questionable conclusions, and has kept statisticians from working on a large range of interesting current problems. Algorithmic modeling, both in theory and practice, has developed rapidly in fields outside statistics. It can be used both on large complex data sets and as a more accurate and informative alternative to data modeling on smaller data sets. If our goal as a field is to use data to solve problems, then we need to move away from exclusive dependence on data models and adopt a more diverse set of tools."


#### Domingos - ["A Few Useful Things to Know about Machine Learning"](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
>	"Machine learning algorithms can figure out how to perform important tasks by generalizing from examples. This is often feasible and cost-effective where manual programming is not. As more data becomes available, more ambitious problems can be tackled. As a result, machine learning is widely used in computer science and other fields. However, developing successful machine learning applications requires a substantial amount of “black art” that is hard to find in textbooks. This article summarizes twelve key lessons that machine learning researchers and practitioners have learned. These include pitfalls to avoid, important issues to focus on, and answers to common questions."


#### Vapnik, Izmailov - ["Learning with Intelligent Teacher: Similarity Control and Knowledge Transfer"](http://link.springer.com/chapter/10.1007/978-3-319-17091-6_1)
>	"This paper introduces an advanced setting of machine learning problem in which an Intelligent Teacher is involved. During training stage, Intelligent Teacher provides Student with information that contains, along with classification of each example, additional privileged information (explanation) of this example. The paper describes two mechanisms that can be used for significantly accelerating the speed of Student’s training: (1) correction of Student’s concepts of similarity between examples, and (2) direct Teacher-Student knowledge transfer."

>	"During last fifty years a strong machine learning theory has been developed. This theory includes: 1. The necessary and sufficient conditions for consistency of learning processes. 2. The bounds on the rate of convergence which in general cannot be improved. 3. The new inductive principle (SRM) which always achieves the smallest risk. 4. The effective algorithms, (such as SVM), that realize consistency property of SRM principle. It looked like general learning theory has been complied: it answered almost all standard questions that is asked in the statistical theory of inference. Meantime, the common observation was that human students require much less examples for training than learning machine. Why? The talk is an attempt to answer this question. The answer is that it is because the human students have an Intelligent Teacher and that Teacher-Student interactions are based not only on the brute force methods of function estimation from observations. Speed of learning also based on Teacher-Student interactions which have additional mechanisms that boost learning process. To learn from smaller number of observations learning machine has to use these mechanisms. In the talk I will introduce a model of learning that includes the so called Intelligent Teacher who during a training session supplies a Student with intelligent (privileged) information in contrast to the classical model where a student is given only outcomes y for events x. Based on additional privileged information x* for event x two mechanisms of Teacher-Student interactions (special and general) are introduced: 1. The Special Mechanism: To control Student's concept of similarity between training examples. and 2. The General Mechanism: To transfer knowledge that can be obtained in space of privileged information to the desired space of decision rules. Both mechanisms can be considered as special forms of capacity control in the universally consistent SRM inductive principle. Privileged information exists for almost any inference problem and can make a big difference in speed of learning processes."

  - <https://video.ias.edu/csdm/2015/0330-VladimirVapnik>
  - <http://learningtheory.org/learning-has-just-started-an-interview-with-prof-vladimir-vapnik/>


#### Sculley, Brodley - ["Compression and Machine Learning: A New Perspective on Feature Space Vectors"](http://www.eecs.tufts.edu/~dsculley/papers/compressionAndVectors.pdf)
>	"The use of compression algorithms in machine learning tasks such as clustering and classification has appeared in a variety of fields, sometimes with the promise of reducing problems of explicit feature selection. The theoretical justification for such methods has been founded on an upper bound on Kolmogorov complexity and an idealized information space. An alternate view shows compression algorithms implicitly map strings into implicit feature space vectors, and compression-based similarity measures compute similarity within these feature spaces. Thus, compression-based methods are not a “parameter free” magic bullet for feature selection and data representation, but are instead concrete similarity measures within defined feature spaces, and are therefore akin to explicit feature vector models used in standard machine learning algorithms. To underscore this point, we find theoretical and empirical connections between traditional machine learning vector models and compression, encouraging cross-fertilization in future work."



---
### interesting papers - automated machine learning

#### Guyon, Bennett, Cawley, Escalante, Escalera, Ho, Macia, Ray, Saeed, Statnikov, Viegas - ["Design of the 2015 ChaLearn AutoML Challenge"](http://www.causality.inf.ethz.ch/AutoML/automl_ijcnn15.pdf)
>	"ChaLearn is organizing for IJCNN 2015 an Automatic Machine Learning challenge (AutoML) to solve classification and regression problems from given feature representations, without any human intervention. This is a challenge with code submission: the code submitted can be executed automatically on the challenge servers to train and test learning machines on new datasets. However, there is no obligation to submit code. Half of the prizes can be won by just submitting prediction results. There are six rounds (Prep, Novice, Intermediate, Advanced, Expert, and Master) in which datasets of progressive difficulty are introduced (5 per round). There is no requirement to participate in previous rounds to enter a new round. The rounds alternate AutoML phases in which submitted code is “blind tested” on datasets the participants have never seen before, and Tweakathon phases giving time (~1 month) to the participants to improve their methods by tweaking their code on those datasets. This challenge will push the state-of-the-art in fully automatic machine learning on a wide range of problems taken from real world applications."


#### Shahriari, Swersky, Wang, Adams, de Freitas - ["Taking the Human Out of the Loop: A Review of Bayesian Optimization"](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
>	"Big data applications are typically associated with systems involving large numbers of users, massive complex software systems, and large-scale heterogeneous computing and storage architectures. The construction of such systems involves many distributed design choices. The end products (e.g., recommendation systems, medical analysis tools, real-time game engines, speech recognizers) thus involves many tunable configuration parameters. These parameters are often specified and hard-coded into the software by various developers or teams. If optimized jointly, these parameters can result in significant improvements. Bayesian optimization is a powerful tool for the joint optimization of design choices that is gaining great popularity in recent years. It promises greater automation so as to increase both product quality and human productivity. This review paper introduces Bayesian optimization, highlights some of its methodological aspects, and showcases a wide range of applications."


#### Ratner, Sa, Wu, Selsam, Re - ["Data Programming: Creating Large Training Sets, Quickly"](https://arxiv.org/abs/1605.07723)
>	"Large labeled training sets are the critical building blocks of supervised learning methods and are key enablers of deep learning techniques. For some applications, creating labeled training sets is the most time-consuming and expensive part of applying machine learning. We therefore propose a paradigm for the programmatic creation of training sets called data programming in which users provide a set of labeling functions, which are programs that heuristically label large subsets of data points, albeit noisily. By viewing these labeling functions as implicitly describing a generative model for this noise, we show that we can recover the parameters of this model to “denoise” the training set. Then, we show how to modify a discriminative loss function to make it noise-aware. We demonstrate our method over a range of discriminative models including logistic regression and LSTMs. We establish theoretically that we can recover the parameters of these generative models in a handful of settings. Experimentally, on the 2014 TAC-KBP relation extraction challenge, we show that data programming would have obtained a winning score, and also show that applying data programming to an LSTM model leads to a TAC-KBP score almost 6 F1 points over a supervised LSTM baseline (and into second place in the competition). Additionally, in initial user studies we observed that data programming may be an easier way to create machine learning models for non-experts."

>	"In the data programming approach to developing a machine learning system, the developer focuses on writing a set of labeling functions, which create a large but noisy training set. Snorkel then learns a generative model of this noise - learning, essentially, which labeling functions are more accurate than others - and uses this to train a discriminative classifier. At a high level, the idea is that developers can focus on writing labeling functions - which are just (Python) functions that provide a label for some subset of data points - and not think about algorithms or features!"

  - <https://youtube.com/watch?v=iSQHelJ1xxU>
  - <http://hazyresearch.github.io/snorkel/blog/weak_supervision.html>
  - <https://github.com/HazyResearch/snorkel>
  - "Fonduer: Knowledge Base Construction from Richly Formatted Data" - <https://arxiv.org/abs/1703.05028>


#### Varma, He, Iter, Xu, Yu, de Sa, Re - [Socratic Learning: Empowering the Generative Model](http://arxiv.org/abs/1610.08123)
>	"A challenge in training discriminative models like neural networks is obtaining enough labeled training data. Recent approaches have leveraged generative models to denoise weak supervision sources that a discriminative model can learn from. These generative models directly encode the users' background knowledge. Therefore, these models may be incompletely specified and fail to model latent classes in the data. We present Socratic learning to systematically correct such generative model misspecification by utilizing feedback from the discriminative model. We prove that under mild conditions, Socratic learning can recover features from the discriminator that informs the generative model about these latent classes. Experimentally, we show that without any hand-labeled data, the corrected generative model improves discriminative performance by up to 4.47 points and reduces error for an image classification task by 80% compared to a state-of-the-art weak supervision modeling technique."

  - <https://youtube.com/watch?v=0gRNochbK9c>


#### Platanios, Blum, Mitchell - ["Estimating Accuracy from Unlabeled Data"](http://auai.org/uai2014/proceedings/individuals/313.pdf)
>	"We consider the question of how unlabeled data can be used to estimate the true accuracy of learned classifiers. This is an important question for any autonomous learning system that must estimate its accuracy without supervision, and also when classifiers trained from one data distribution must be applied to a new distribution (e.g., document classifiers trained on one text corpus are to be applied to a second corpus). We first show how to estimate error rates exactly from unlabeled data when given a collection of competing classifiers that make independent errors, based on the agreement rates between subsets of these classifiers. We further show that even when the competing classifiers do not make independent errors, both their accuracies and error dependencies can be estimated by making certain relaxed assumptions. Experiments on two real-world data sets produce estimates within a few percent of the true accuracy, using solely unlabeled data. These results are of practical significance in situations where labeled data is scarce and shed light on the more general question of how the consistency among multiple functions is related to their true accuracies."

>	"Estimating accuracy of classifiers is central to machine learning and many other fields. Traditionally, one estimates accuracy of a function based on its performance over a set of labeled test examples. This paper considers the question of under what conditions is it possible to estimate accuracy based instead on unlabeled data. We show that accuracy can be estimated exactly from unlabeled data in the case that at least three different approximations to the same function are available, so long as these functions make independent errors and have better than chance accuracy. More interestingly, we show that even if one does not assume independent errors, one can still estimate accuracy given a sufficient number of competing approximations to the same function, by viewing the degree of independence of those approximations as an optimization criterion. We present experimental results demonstrating the success of this approach in estimating classification accuracies to within a few percentage points of their true value, in two diverse domains."

>	"We consider a “multiple approximations” problem setting in which we have several different approximations, to some target boolean classification function, and we wish to know the true accuracies of each of these different approximations, using only unlabeled data. The multiple functions can be from any source - learned or manually constructed. One example of this setting that we consider here is taken from NELL. NELL learns classifiers that map noun phrases to boolean categories such as fruit, food and vehicle. For each such boolean classification function, NELL learns several different approximations based on different views of the NP. One approximation is based on the orthographic features of the NP (e.g., if the NP ends with the letter string “burgh”, it may be a city), whereas another uses phrases surrounding the NP (e.g., if the NP follows the word sequence “mayor of”, it may be a city). Our aim in this paper is to find a way to estimate the error rates of each of the competing approximations, using only unlabeled data (e.g., many unlabeled NPs in the case of NELL)."

>	"The main contributions of this paper include: (1) formulating the problem of estimating the error rate of each of several approximations to the same function, based on their agreement rates over unlabeled data, as an optimization problem, (2) providing two different analytical methods that estimate error rates from agreement rates in this setting, one based on a set of simultaneous equations relating accuracies, agreements, and error dependencies, and a second, based on maximizing data likelihood, and (3) demonstrating the success of these two methods in two very different real-world problems."

>	"We have introduced the concept of estimating the error rate of each of several approximations to the same function, based on their agreement rates over unlabeled data and we have provided three different analytical methods to do so: the AR method, the MLE method and the MAP method. Our experiments showed that the AR method performs significantly better than the other two methods for both data sets we considered. Our results are very encouraging and suggest that function agreement rates are indeed very useful in estimating function error rates. We consider this work to be a first step towards developing a self-reflection framework for autonomous learning systems. There are several directions we would like to pursue to further improve upon the methods introduced here. Firstly, we wish to explore other interesting natural objectives one can aim to optimize. It would also be very interesting to explore possible generalizations of our models to non-boolean, discrete-valued functions, or even to real-valued functions. Finally, apart from simply estimating function error rates, we want to explore how the obtained error rate estimates can be used to improve the learning ability of a system such as NELL, for example. In this context, we could try using our estimates in order to develop a more robust co-training framework. One very direct application of our methods would be to use the estimated error rates and their dependencies in order to combine the functions’ outputs and obtain one final output."

  - <https://youtu.be/PF6ViL5pcGs?t=5m3s>



---
### interesting papers - systems

#### Sculley, Holt, Golovin, Davydov, Phillips, Ebner, Chaudhary, Young - ["Machine Learning: The High-Interest Credit Card of Technical Debt"](http://research.google.com/pubs/pub43146.html)
>	"Machine learning offers a fantastically powerful toolkit for building complex systems quickly. This paper argues that it is dangerous to think of these quick wins as coming for free. Using the framework of technical debt, we note that it is remarkably easy to incur massive ongoing maintenance costs at the system level when applying machine learning. The goal of this paper is highlight several machine learning specific risk factors and design patterns to be avoided or refactored where possible. These include boundary erosion, entanglement, hidden feedback loops, undeclared consumers, data dependencies, changes in the external world, and a variety of system-level anti-patterns."

  - <http://john-foreman.com/blog/the-perilous-world-of-machine-learning-for-fun-and-profit-pipeline-jungles-and-hidden-feedback-loops>


#### Google Brain - ["TensorFlow: A system for large-scale machine learning"](https://arxiv.org/abs/1605.08695)
>	"TensorFlow is a machine learning system that operates at large scale and in heterogeneous environments. TensorFlow uses dataflow graphs to represent computation, shared state, and the operations that mutate that state. It maps the nodes of a dataflow graph across many machines in a cluster, and within a machine across multiple computational devices, including multicore CPUs, general-purpose GPUs, and custom designed ASICs known as Tensor Processing Units (TPUs). This architecture gives flexibility to the application developer: whereas in previous “parameter server” designs the management of shared state is built into the system, TensorFlow enables developers to experiment with novel optimizations and training algorithms. TensorFlow supports a variety of applications, with particularly strong support for training and inference on deep neural networks. Several Google services use TensorFlow in production, we have released it as an open-source project, and it has become widely used for machine learning research. In this paper, we describe the TensorFlow dataflow model in contrast to existing systems, and demonstrate the compelling performance that TensorFlow achieves for several real-world applications."

  - <https://github.com/nlintz/TensorFlow-Tutorials>


#### Agarwal, Chapelle, Dudik, Langford - ["A Reliable Effective Terascale Linear Learning System"](http://arxiv.org/abs/1110.4198) (Vowpal Wabbit)
>	"We present a system and a set of techniques for learning linear predictors with convex losses on terascale data sets, with trillions of features, 1 billions of training examples and millions of parameters in an hour using a cluster of 1000 machines. Individually none of the component techniques are new, but the careful synthesis required to obtain an efficient implementation is. The result is, up to our knowledge, the most scalable and efficient linear learning system reported in the literature. We describe and thoroughly evaluate the components of the system, showing the importance of the various design choices."

>	"- Online by default
>	 - Hashing, raw text is fine
>	 - Most scalable public algorithm
>	 - Reduction to simple problems
>	 - Causation instead of correlation
>	 - Learn to control based on feedback"

  - <http://youtube.com/watch?v=wwlKkFhEhxE> (Langford)
  - <https://github.com/JohnLangford/vowpal_wabbit/wiki>
  - "Bring The Noise: Embracing Randomness Is the Key to Scaling Up Machine Learning Algorithms" - <http://online.liebertpub.com/doi/pdf/10.1089/big.2013.0010>


#### Agarwal et al. - ["A Multiworld Testing Decision Service"](http://arxiv.org/abs/1606.03966)
>	"Applications and systems are constantly faced with decisions to make, often using a policy to pick from a set of actions based on some contextual information. We create a service that uses machine learning to accomplish this goal. The service uses exploration, logging, and online learning to create a counterfactually sound system supporting a full data lifecycle. The system is general: it works for any discrete choices, with respect to any reward metric, and can work with many learning algorithms and feature representations. The service has a simple API, and was designed to be modular and reproducible to ease deployment and debugging, respectively. We demonstrate how these properties enable learning systems that are robust and safe. Our evaluation shows that the Decision Service makes decisions in real time and incorporates new data quickly into learned policies. A large-scale deployment for a personalized news website has been handling all traffic since Jan. 2016, resulting in a 25% relative lift in clicks. By making the Decision Service externally available, we hope to make optimal decision making available to all."

>	"We have presented the Decision Service: a powerful tool to support the complete data lifecycle, which automates many of the burdensome tasks that data scientists face such as gathering the right data and deploying in an appropriate manner. Instead, a data scientist can focus on more core tasks such as finding the right features, representation, or signal to optimize against. The data lifecycle support also makes basic application of the Decision Service feasible without a data scientist. To assist in lowering the barrier to entry, we are exploring techniques based on expert learning and hyperparameter search that may further automate the process. Since the policy evaluation techniques can provide accurate predictions of online performance, such automations are guaranteed to be statistically sound. We are also focusing on making the decision service easy to deploy and use because we believe this is key to goal of democratizing machine learning for everyone. The Decision Service can also naturally be extended to a greater variety of problems, all of which can benefit from data lifecycle support. Plausible extensions might address advanced variants like reinforcement and active learning, and simpler ones like supervised learning."

----
>	"It is the first general purpose reinforcement-based learning system. Wouldn’t it be great if Reinforcement Learning algorithms could easily be used to solve all reinforcement learning problems? But there is a well-known problem: It’s very easy to create natural RL problems for which all standard RL algorithms (epsilon-greedy Q-learning, SARSA, etc) fail catastrophically. That’s a serious limitation which both inspires research and which I suspect many people need to learn the hard way. Removing the credit assignment problem from reinforcement learning yields the Contextual Bandit setting which we know is generically solvable in the same manner as common supervised learning problems."

>	"Many people have tried to create online learning system that do not take into account the biasing effects of decisions. These fail near-universally. For example they might be very good at predicting what was shown (and hence clicked on) rather that what should be shown to generate the most interest."

>	"We need a system that explores over appropriate choices with logging of features, actions, probabilities of actions, and outcomes. These must then be fed into an appropriate learning algorithm which trains a policy and then deploys the policy at the point of decision. The system enables a fully automatic causally sound learning loop for contextual control of a small number of actions. It is strongly scalable, for example a version of this is in use for personalized news on MSN."

  - <http://hunch.net/?p=4464948> (Langford)
  - <http://research.microsoft.com/en-us/projects/mwt/>
  - <https://mwtds.azurewebsites.net>


#### Simard, Chickering, Lakshmiratan, Charles, Bottou, Suarez, Grangier, Amershi, Verwey, Suh, - ["ICE: Enabling Non-Experts to Build Models Interactively for Large-Scale Lopsided Problems"](http://arxiv.org/abs/1409.4814)
>	"Quick interaction between a human teacher and a learning machine presents numerous benefits and challenges when working with web-scale data. The human teacher guides the machine towards accomplishing the task of interest. The learning machine leverages big data to find examples that maximize the training value of its interaction with the teacher. When the teacher is restricted to labeling examples selected by the machine, this problem is an instance of active learning. When the teacher can provide additional information to the machine (e.g., suggestions on what examples or predictive features should be used) as the learning task progresses, then the problem becomes one of interactive learning. To accommodate the two-way communication channel needed for efficient interactive learning, the teacher and the machine need an environment that supports an interaction language. The machine can access, process, and summarize more examples than the teacher can see in a lifetime. Based on the machine’s output, the teacher can revise the definition of the task or make it more precise. Both the teacher and the machine continuously learn and benefit from the interaction. We have built a platform to (1) produce valuable and deployable models and (2) support research on both the machine learning and user interface challenges of the interactive learning problem. The platform relies on a dedicated, low-latency, distributed, in-memory architecture that allows us to construct web-scale learning machines with quick interaction speed. The purpose of this paper is to describe this architecture and demonstrate how it supports our research efforts. Preliminary results are presented as illustrations of the architecture but are not the primary focus of the paper."

  - used by Microsoft LUIS according to <http://arxiv.org/abs/1606.03966>




<brylevkirill (at) gmail.com>
