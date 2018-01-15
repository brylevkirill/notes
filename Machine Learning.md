  A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.


  * [overview](#overview)
  * [study](#study)
  * [theory](#theory)
  * [methods](#methods)
  * [representation learning](#representation-learning)
  * [program induction](#program-induction)
  * [meta-learning](#meta-learning)
  * [automated machine learning](#automated-machine-learning)
  * [interesting quotes](#interesting-quotes)
  * [interesting papers](#interesting-papers)
    - [theory](#interesting-papers---theory)
    - [automated machine learning](#interesting-papers---automated-machine-learning)
    - [systems](#interesting-papers---systems)

----

  [deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md)  
  [reinforcement learning](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md)  
  [bayesian inference and learning](https://github.com/brylevkirill/notes/blob/master/Bayesian%20Inference%20and%20Learning.md)  
  [probabilistic programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md)  



----
#### applications

  [artificial intelligence](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md)  
  [knowledge representation and reasoning](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md)  
  [natural language processing](https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md)  
  [information retrieval](https://github.com/brylevkirill/notes/blob/master/Information%20Retrieval.md)  


  Any source code for expression y = f(x), where f(x) has some parameters and is used to make decision, prediction or estimate, has potential to be replaced by machine learning algorithm.



---
### overview

  ["Machine Learning: Trends, Perspectives and Prospects"](https://goo.gl/U8552O) by Michael I. Jordan and Tom Mitchell  
  ["When is Machine Learning Worth It?"](http://inference.vc/when-is-machine-learning-worth-it/) by Ferenc Huszar  
  ["Machine Learning is the new algorithms"](http://nlpers.blogspot.ru/2014/10/machine-learning-is-new-algorithms.html) by Hal Daume  

----

  ["The Talking Machines"](http://thetalkingmachines.com/blog/) podcast `audio`

----

  ["Machine Learning Basics"](http://www.deeplearningbook.org/contents/ml.html) by Ian Goodfellow, Yoshua Bengio, Aaron Courville

----

  overview by Dmitry Vetrov ([first](https://youtu.be/srIcbDBAJBo), [second](https://youtu.be/ftlbxFypW74)) `video` `in russian`  
  [overview](http://youtube.com/watch?v=lkh7bLUc30g) by Dmitry Vetrov `video` `in russian`  
  overview by Igor Kuralenok ([first](https://youtu.be/ynS7XvkAdLU?t=12m5s), [second](https://youtu.be/jiyD0r2SC-g?t=12m55s)) `video` `in russian`  



---
### study

#### knowledge bases

  <http://metacademy.org>  
  <http://machinelearning.ru/wiki/> `in russian`  

  <https://en.wikipedia.org/wiki/Machine_learning>  
  ["Introduction to Machine Learning - The Wikipedia Guide"](https://github.com/Nixonite/open-source-machine-learning-degree/blob/master/Introduction%20to%20Machine%20Learning%20-%20Wikipedia.pdf)  


#### guides

  ["A Few Useful Things to Know about Machine Learning"](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) by Pedro Domingos  
  ["Expressivity, Trainability, and Generalization in Machine Learning"](http://blog.evjang.com/2017/11/exp-train-gen.html) by Eric Jang  
  ["Classification vs. Prediction"](http://fharrell.com/2017/01/classification-vs-prediction.html) by Frank Harrell  
  ["Causality in Machine Learning"](http://unofficialgoogledatascience.com/2017/01/causality-in-machine-learning.html) by Muralidharan et al.  
  ["Are ML and Statistics Complementary?"](https://www.ics.uci.edu/~welling/publications/papers/WhyMLneedsStatistics.pdf) by Max Welling  
  ["Introduction to Information Theory and Why You Should Care"](https://blog.recast.ai/introduction-information-theory-care/) by Gil Katz  
  ["Ideas on Interpreting Machine Learning"](https://oreilly.com/ideas/ideas-on-interpreting-machine-learning) by Hall et al.  
  ["Clever Methods of Overfitting"](http://hunch.net/?p=22) by John Langford  
  ["Common Pitfalls in Machine Learning"](http://danielnee.com/?p=155) by Daniel Nee  
  ["Software Engineering vs Machine Learning Concepts"](http://machinedlearnings.com/2017_02_01_archive.html) by Paul Mineiro  
  ["Rules of Machine Learning: Best Practices for ML Engineering"](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) by Martin Zinkevich  


#### courses

  [course](https://youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN) by Andrew Ng `video` ([Coursera](http://coursera.org/learn/machine-learning))  
  [course](http://youtube.com/playlist?list=PLE6Wd9FR--Ecf_5nCbnSQMHqORpiChfJf) by Nando de Freitas `video`  
  [course](http://youtube.com/playlist?list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6) by Nando de Freitas `video`  
  [course](http://youtube.com/playlist?list=PLTPQEx-31JXgtDaC6-3HxWcp7fq4N8YGr) by Pedro Domingos `video`  
  [course](http://youtube.com/playlist?list=PLZSO_6-bSqHTTV7w9u7grTXBHMH-mw3qn) by Alex Smola `video`  
  [course](http://dataschool.io/15-hours-of-expert-machine-learning-videos/) by Trevor Hastie and Rob Tibshirani `video`  
  [course](https://youtube.com/playlist?list=PLD0F06AA0D2E8FFBA) by Jeff Miller `video`  

  [course](http://ciml.info) by Hal Daume

  [course](http://coursera.org/specializations/machine-learning-data-analysis) by Yandex `video` `in russian`  
  [course](https://github.com/Yorko/mlcourse_open) by OpenDataScience `video` `in russian`  
  [course](http://youtube.com/playlist?list=PLJOzdkh8T5kp99tGTEFjH_b9zqEQiiBtC) by Konstantin Vorontsov `video` `in russian`  
  [course](http://youtube.com/playlist?list=PLlb7e2G7aSpTd91sd82VxWNdtTZ8QnFne) by Igor Kuralenok `video` `in russian`  
  [course](http://youtube.com/playlist?list=PLlb7e2G7aSpSWVExpq74FnwFnWgLby56L) by Igor Kuralenok `video` `in russian`  
  [course](http://youtube.com/playlist?list=PLlb7e2G7aSpSSsCeUMLN-RxYOLAI9l2ld) by Igor Kuralenok `video` `in russian`  
  [course](http://lektorium.tv/course/22975) by Igor Kuralenok `video` `in russian`  


#### books

  ["The Master Algorithm"](http://basicbooks.com/full-details?isbn=9780465065707) by Pedro Domingos  
  ["A First Encounter with Machine Learning"](https://www.ics.uci.edu/~welling/teaching/ICS273Afall11/IntroMLBook.pdf) by Max Welling  
  ["Deep Learning"](http://www.deeplearningbook.org) by Ian Goodfellow, Yoshua Bengio, Aaron Courville  
  ["Reinforcement Learning: An Introduction"](http://incompleteideas.net/sutton/book/ebook/the-book.html)
	([second edition](http://incompleteideas.net/sutton/book/the-book-2nd.html)) by Richard Sutton and Andrew Barto  
  ["Machine Learning"](https://goo.gl/tyNHMH) by Tom Mitchell  
  ["Understanding Machine Learning: From Theory to Algorithms"](http://cs.huji.ac.il/~shais/UnderstandingMachineLearning/) by Shai Shalev-Shwartz and Shai Ben-David  
  ["Pattern Recognition and Machine Learning"](https://goo.gl/58Yvvp) by Chris Bishop  
  ["Computer Age Statistical Inference"](https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf) by Bradley Efron and Trevor Hastie  
  ["The Elements of Statistical Learning"](http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf) by Trevor Hastie, Robert Tibshirani, Jerome Friedman  
  ["Machine Learning - A Probabilistic Perspective"](https://goo.gl/Vh7Jje) by Kevin Murphy  
  ["Information Theory, Inference, and Learning Algorithms"](http://users.aims.ac.za/~mackay/itila/book.html) by David MacKay  
  ["Bayesian Reasoning and Machine Learning"](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.Online) by David Barber  
  ["Foundations of Machine Learning"](http://www.cs.nyu.edu/~mohri/mlbook/) by Mehryar Mohri  
  ["Scaling Up Machine Learning: Parallel and Distributed Approaches"](https://goo.gl/dE7jPb) by Ron Bekkerman, Mikhail Bilenko, John Langford  


#### blogs

  <http://inference.vc>  
  <http://offconvex.org>  
  <http://argmin.net>  
  <http://blog.shakirm.com>  
  <http://hunch.net>  
  <http://machinedlearnings.com>  
  <http://nlpers.blogspot.com>  
  <http://fastml.com>  
  <http://wildml.com>  
  <http://blogs.princeton.edu/imabandit>  
  <http://timvieira.github.io/blog/>  
  <http://r2rt.com>  
  <http://danieltakeshi.github.io>  
  <http://theneuralperspective.com>  


#### news and discussions

  <https://jack-clark.net/import-ai/>  
  <https://www.getrevue.co/profile/wildml>  
  <https://deeplearningweekly.com>  

  <https://reddit.com/r/MachineLearning/>  


#### conferences

  - ICLR 2018  
	<http://search.iclr2018.smerity.com/>  
	<http://iclr2018.mmanukyan.io>  

  - NIPS 2017  
	<https://nips.cc/Conferences/2017/Videos> `video`  
	<https://facebook.com/pg/nipsfoundation/videos/> `video`  

	<https://nips17.ml>

	<https://github.com/hindupuravinash/nips2017>  
	<https://github.com/kihosuh/nips_2017>  
	<https://github.com/sbarratt/nips2017>  

  - ICML 2017  
	<https://vimeo.com/user72337760> `video`

	<http://artem.sobolev.name/posts/2017-08-14-icml-2017.html>  
	<https://olgalitech.wordpress.com/tag/icml2017/>  

  - ICLR 2017  
	<https://facebook.com/pg/iclr.cc/videos/> `video`

	<https://medium.com/@karpathy/iclr-2017-vs-arxiv-sanity-d1488ac5c131>

  - NIPS 2016  
	<https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016> `video`  
	<https://nips.cc/Conferences/2016/SpotlightVideos> `video`  

	<http://people.idsia.ch/~rupesh/rnnsymposium2016/program.html> + <https://youtube.com/playlist?list=PLPwzH56Rdmq4hcuEMtvBGxUrcQ4cAkoSc>  
	<https://sites.google.com/site/nips2016adversarial/> + <https://youtube.com/playlist?list=PLJscN9YDD1buxCitmej1pjJkR5PMhenTF>  
	<http://bayesiandeeplearning.org> + <https://youtube.com/channel/UC_LBLWLfKk5rMKDOHoO7vPQ>  
	<https://uclmr.github.io/nampi/> + <https://youtube.com/playlist?list=PLzTDea_cM27LVPSTdK9RypSyqBHZWPywt>  

	<https://github.com/hindupuravinash/nips2016>  
	<http://artem.sobolev.name/posts/2016-12-31-nips-2016-summaries.html>  

  - ICML 2016  
	<http://techtalks.tv/icml/2016/> `video`

  - ICLR 2016  
	<http://videolectures.net/iclr2016_san_juan/> `video`

	<http://www.computervisionblog.com/2016/06/deep-learning-trends-iclr-2016.html>  

  - NIPS 2015  
	<https://youtube.com/playlist?list=PLD7HFcN7LXRdvgfR6qNbuvzxIwG0ecE9Q> `video`  
	<https://youtube.com/user/NeuralInformationPro/search?query=NIPS+2015> `video`  
 
	<http://reddit.com/r/MachineLearning/comments/3x2ueg/nips_2015_overviews_collection/>  
	<http://cinrizasti.blogspot.ru/2015/12/a-blog-post-about-blog-posts-about-nips.html>  

  - ICML 2015  
	<https://youtube.com/playlist?list=PLdH9u0f1XKW8cUM3vIVjnpBfk_FKzviCu> `video`  
	<http://dpkingma.com/?page_id=483> `video`  

  - ICLR 2015  
	<http://youtube.com/channel/UCqxFGrNL5nX10lS62bswp9w> `video`

  - NIPS 2014  
	<https://youtube.com/user/NeuralInformationPro/search?query=NIPS+2014> `video`



---
### theory

  [machine learning has become alchemy](https://youtube.com/watch?v=Qi1Yry33TQE&t=11m2s) by Ali Rahimi `video`  
  [statistics in machine learning](https://youtube.com/watch?v=uyZOcUDhIbY&t=17m27s) by Michael I. Jordan `video`  
  [theory in machine learning](https://youtube.com/watch?v=uyZOcUDhIbY&t=23m1s) by Michael I. Jordan `video`  

  ["Learning Theory: Purely Theoretical?"](https://hips.seas.harvard.edu/blog/2013/02/15/learning-theory-purely-theoretical/) by Jonathan Huggins

----

  problems:  
  - What does it mean to learn?  
  - When is a concept/function learnable?  
  - How much data do we need to learn something?  
  - How can we make sure what we learn will generalize to future data?  

  theory helps to:  
  - design algorithms  
  - understand behaviour of algorithms  
  - quantify knowledge/uncertainty  
  - identify new and refine old challenges  

  frameworks:  
  - [statistical learning theory](#theory---statistical-learning-theory)  
  - [computational learning theory](#theory---computational-learning-theory) (PAC learning or PAC-Bayes)  


----
#### theory - statistical learning theory

  ingredients:
  - distributions
  - i.i.d. samples
  - learning algorithms
  - predictors
  - loss functions

  *A priori analysis*: How well a learning algorithm will perform on new data?  
  - (Vapnik's learning theory) Can we compete with best hypothesis from a given set of hypotheses?  
  - (statistics) Can we match the best possible loss assuming data generating distribution belongs to known family?  

  *A posteriori analysis*: How well is a learning algorithm doing on some data? Quantify uncertainty left

  *Fundamental theorem of statistical learning theory*:  
  In binary classification, to match the loss of hypothesis in class H up to accuracy ε, one needs O(VC(H)/ε^2) observations.

  *Theorem (computational complexity of learning linear classifiers)*:  
  Unless NP=RP, linear classifiers (hyperplanes) cannot be learned in polynomial time.

----

  ["Machine Learning Theory"](https://mostafa-samir.github.io/ml-theory-pt1/) by Mostafa Samir  
  ["Crash Course on Learning Theory"](https://blogs.princeton.edu/imabandit/2015/10/13/crash-course-on-learning-theory-part-1/) by Sebastien Bubeck  
  ["Statistical Learning Theory"](https://web.stanford.edu/class/cs229t/Lectures/percy-notes.pdf) by Percy Liang  

  [course](http://work.caltech.edu/telecourse.html) by Yaser Abu-Mostafa `video`


----
#### theory - computational learning theory

  ["Computational Learning Theory, AI and Beyond"](https://www.math.ias.edu/files/mathandcomp.pdf) chapter of "Mathematics and Computation" book by Avi Wigderson

  ["Probably Approximately Correct - A Formal Theory of Learning"](http://jeremykun.com/2014/01/02/probably-approximately-correct-a-formal-theory-of-learning/) by Jeremy Kun  
  ["A Problem That is Not (Properly) PAC-learnable"](http://jeremykun.com/2014/04/21/an-un-pac-learnable-problem/) by Jeremy Kun  
  ["Occam’s Razor and PAC-learning"](http://jeremykun.com/2014/09/19/occams-razor-and-pac-learning/) by Jeremy Kun  


----
#### theory - applications

  [bayesian inference and learning](https://github.com/brylevkirill/notes/blob/master/Bayesian%20Inference%20and%20Learning.md#theory)

  [deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#theory)

  [reinforcement learning](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#problems)  
  ["Theory of Reinforcement Learning"](http://videolectures.net/deeplearning2017_szepesvari_theory_of_rl/) by Csaba Szepesvari `video`  



---
### methods

  **challenges**  
  - How to decide which representation is best for target knowledge?  
  - How to tell genuine regularities from chance occurrences?  
  - How to exploit pre-existing domain knowledge knowledge?  
  - How to learn with limited computational resources?  
  - How to learn with limited data?  
  - How to make learned results understandable?  
  - How to quantify uncertainty?  
  - How to take into account the costs of decisions?  
  - How to handle non-indepedent and non-stationary data?  

----

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
    * bayesian model averaging  
    * stacking  

----

  ["The Three Cultures of Machine Learning"](https://www.cs.jhu.edu/~jason/tutorials/ml-simplex.html) by Jason Eisner  
  ["Algorithmic Dimensions"](https://justindomke.wordpress.com/2015/09/14/algorithmic-dimensions/) by Justin Domke  
  ["All Models of Learning Have Flaws"](http://hunch.net/?p=224) by John Langford  

----

  <http://en.wikipedia.org/wiki/List_of_machine_learning_algorithms>

  <http://eferm.com/wp-content/uploads/2011/05/cheat3.pdf>  
  <http://github.com/soulmachine/machine-learning-cheat-sheet/blob/master/machine-learning-cheat-sheet.pdf>  

  <http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html>  



---
### representation learning

  "Representation is a formal system which makes explicit certain entities and types of information, and which can be operated on by an algorithm in order to achieve some information processing goal. Representations differ in terms of what information they make explicit and in terms of what algorithms they support. As example, Arabic and Roman numerals - the fact that operations can be applied to particular columns of Arabic numerals in meaningful ways allows for simple and efficient algorithms for addition and multiplication."

  "In representation learning, our goal isn’t to predict observables, but to learn something about the underlying structure. In cognitive science and AI, a representation is a formal system which maps to some domain of interest in systematic ways. A good representation allows us to answer queries about the domain by manipulating that system. In machine learning, representations often take the form of vectors, either real- or binary-valued, and we can manipulate these representations with operations like Euclidean distance and matrix multiplication."

  "In representation learning, the goal isn’t to make predictions about observables, but to learn a representation which would later help us to answer various queries. Sometimes the representations are meant for people, such as when we visualize data as a two-dimensional embedding. Sometimes they’re meant for machines, such as when the binary vector representations learned by deep Boltzmann machines are fed into a supervised classifier. In either case, what’s important is that mathematical operations map to the underlying relationships in the data in systematic ways."

----

  ["What is representation learning?"](https://hips.seas.harvard.edu/blog/2013/02/25/what-is-representation-learning/) by Roger Grosse  
  ["Predictive learning vs. representation learning"](https://hips.seas.harvard.edu/blog/2013/02/04/predictive-learning-vs-representation-learning/) by Roger Grosse  

----

  [deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md)  
  [probabilistic programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md)  
  [knowledge representation](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#knowledge-representation)  



---
### program induction

  "The essence of programmatic representations is that they are well-specified, compact, combinatorial and hierarchical.  
  - *well-specified*:  Unlike sentences in natural language, programs are unambiguous, although two distinct programs can be precisely equivalent.  
  - *compact*:  Programs allow us to compress data on the basis of their regularities.  
  - *combinatorial*:  Programs can access the results of running other programs, as well as delete, duplicate, and rearrange these results.  
  - *hierarchical*:  Programs have an intrinsic hierarchical organization and may be decomposed into subprograms."  

  "Alternative representations for procedural abstractions such as neural networks have serious downsides including opacity and inefficiency."

  "Challenges with programmatic representations:  
  - *open-endedness*:  In contrast to other knowledge representations in machine learning, programs may vary in size and “shape”, and there is no obvious problem-independent upper bound on program size. This makes it difficult to represent programs as points in a fixed-dimensional space, or learn programs with algorithms that assume such a space.  
  - *over-representation*:  Often syntactically distinct programs will be semantically identical (i.e. represent the same underlying behavior or functional mapping). Lacking prior knowledge, many algorithms will inefficiently sample semantically identical programs repeatedly.  
  - *chaotic execution*: Programs that are very similar, syntactically, may be very different, semantically. This presents difficulty for many heuristic search algorithms, which require syntactic and semantic distance to be correlated.  
  - *high resource-variance*:  Programs in the same space may vary greatly in the space and time they require to execute."  

----

  "For me there are two types of generalisation, which I will refer to as Symbolic and Connectionist generalisation. If we teach a machine to sort sequences of numbers of up to length 10 or 100, we should expect them to sort sequences of length 1000 say. Obviously symbolic approaches have no problem with this form of generalisation, but neural nets do poorly. On the other hand, neural nets are very good at generalising from data (such as images), but symbolic approaches do poorly here. One of the holy grails is to build machines that are capable of both symbolic and connectionist generalisation. Neural Programmer Interpreters is a very early step toward this. NPI can do symbolic operations such as sorting and addition, but it can also plan by taking images as input and it's able to generalise the plans to different images (e.g. in the NPI car example, the cars are test set cars not seen before)."

  *(Nando de Freitas)*

----

  [inductive programming](https://en.wikipedia.org/wiki/Inductive_programming)

  ["Program Synthesis Explained"](http://homes.cs.washington.edu/~bornholt/post/synthesis-for-architects.html) by James Bornholt  
  ["Inductive Programming Meets the Real World"](https://microsoft.com/en-us/research/publication/inductive-programming-meets-real-world/) by Gulwani et al. `paper`  

  ["Symbolic Machine Learning"](http://languagengine.co/blog/symbolic-machine-learning/) by Darryl McAdams

----

  ["The Future of Deep Learning"](https://blog.keras.io/the-future-of-deep-learning.html) by Francois Chollet ([talk](https://youtu.be/MUF32XHqM34) `video`)

----

  ["Deep Learning Trends: Program Induction"](https://facebook.com/nipsfoundation/videos/1552060484885185/) (1:30:12) by Scott Reed `video`  
  ["Learning to Code: Machine Learning for Program Induction"](https://youtu.be/vzDuVhFMB9Q?t=2m40s) by Alex Gaunt `video`  

  ["Neural Abstract Machines & Program Induction"](https://uclmr.github.io/nampi) workshop at NIPS 2016
	([videos](https://youtube.com/playlist?list=PLzTDea_cM27LVPSTdK9RypSyqBHZWPywt))  
  [panel](https://youtu.be/nqiUFc52g78?t=1h29m3s) at NAMPI workshop
	with Percy Liang, Juergen Schmidhuber, Joshua Tenenbaum, Martin Vechev, Daniel Tarlow, Dawn Song `video`  

----

  ["TerpreT: A Probabilistic Programming Language for Program Induction"](https://arxiv.org/abs/1608.04428) by Gaunt et al. `paper`  
  ["Learning Explanatory Rules from Noisy Data"](https://arxiv.org/abs/1711.04574) by Evans and Grefenstette `paper`  

  [interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#program-induction)  
  [selected papers](https://dropbox.com/sh/vrr1gs798zy02n1/AACj7hlXOiRt1nXltXVC-2Wca)  



---
### meta-learning

  [learning to learn](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#meta-learning)



---
### automated machine learning

  AutoML aims to automate many different stages of the machine learning process:  
  - model selection, hyper-parameter optimization, and model search  
  - meta learning and transfer learning  
  - representation learning and automatic feature extraction / construction  
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
  - different data distributions: the intrinsic/geometrical complexity of the dataset  
  - different tasks: regression, binary classification, multi-class classification, multi-label classification  
  - different scoring metrics: AUC, BAC, MSE, F1, etc  
  - class balance: Balanced or unbalanced class proportions  
  - sparsity: Full matrices or sparse matrices  
  - missing values: Presence or absence of missing values  
  - categorical variables: Presence or absence of categorical variables  
  - irrelevant variables: Presence or absence of additional irrelevant variables (distractors)  
  - number Ptr of training examples: Small or large number of training examples  
  - number N of variables/features: Small or large number of variables  
  - aspect ratio Ptr/N of the training data matrix: Ptr >> N, Ptr = N or Ptr << N  

----

  "We can put a unified framework around the various approaches. Borrowing from the conventional classification of feature selection methods, model search strategies can be categorized into filters, wrappers, and embedded methods.  
  Filters are methods for narrowing down the model space, without training the learning machine. Such methods include preprocessing, feature construction, kernel design, architecture design, choice of prior or regularizers, choice of a noise model, and filter methods for feature selection. Although some filters use training data, many incorporate human prior knowledge of the task or knowledge compiled from previous tasks (a form of meta learning or transfer learning). Recently, it has been proposed to apply collaborative filtering methods to model search.  
  Wrapper methods consider the learning machine as a black-box capable of learning from examples and making predictions once trained. They operate with a search algorithm in hyper-parameter space (for example grid search or stochastic search) and an evaluation function assessing the trained learning machine performances (for example the cross-validation error or the Bayesian evidence).  
  Embedded methods are similar to wrappers, but they exploit the knowledge of the learning machine algorithm to make the search more efficient. For instance, some embedded methods compute the leave-one-out solution in a closed form, without leaving anything out, i.e., by performing a single model training on all the training data. Other embedded methods jointly optimize parameters and hyperparameters."  

----

  ["Automated Machine Learning"](https://youtube.com/watch?v=AFeozhAD9xE) by Andreas Mueller `video`

  ["Automated Machine Learning: A Short History"](https://datarobot.com/blog/automated-machine-learning-short-history/) by Thomas Dinsmore

----

  ["AutoML for large scale image classification and object detection"](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html) by Zoph et al.

----

  [TPOT](http://rhiever.github.io/tpot/) project

  [auto_ml](http://auto-ml.readthedocs.io/en/latest/) project

  [The Automatic Statistician](https://automaticstatistician.com) project  
  - ["The Automatic Statistician"](https://youtu.be/H7AMB0oo__4?t=53m20s) by Zoubin Ghahramani `video`  
  - ["The Automatic Statistician: A Project Update"](https://youtube.com/watch?v=WW2eunuApAU) by Zoubin Ghahramani `video`  

  [BayesDB](http://probcomp.org/bayesdb/) project  
  - [overview](https://youtube.com/watch?v=-8QMqSWU76Q) by Vikash Mansinghka `video`  
  - ["BayesDB: Query the Probable Implications of Data"](https://youtube.com/watch?v=7_m7JCLKmTY) by Richard Tibbetts `video`  

----

  [AutoML](http://automl.chalearn.org) challenge  
>	"Design the perfect machine learning “black box” capable of performing all model selection and hyper-parameter tuning without any human intervention"  

----

  ["The Future of Deep Learning"](https://blog.keras.io/the-future-of-deep-learning.html) by Francois Chollet ([talk](https://youtu.be/MUF32XHqM34?t=11m43s) `video`)

----

  ["Why Tool AIs Want to Be Agent AIs"](http://www.gwern.net/Tool%20AI) by Gwern Branwen:  

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

  "The logical extension of these neural networks all the way down papers is that an actor like Google / Baidu / Facebook / MS could effectively turn neural networks into a black box: a user/developer uploads through an API a dataset of input/output pairs of a specified type and a monetary loss function, and a top-level neural network running on a large GPU cluster starts autonomously optimizing over architectures & hyperparameters for the neural network design which balances GPU cost and the monetary loss, interleaved with further optimization over the thousands of previous submitted tasks, sharing its learning across all of the datasets / loss functions / architectures / hyperparameters, and the original user simply submits future data through the API for processing by the best neural network so far."

  *(Gwern Branwen)*



---
### interesting quotes

  - [architectures](#interesting-quotes---architectures)  
  - [representation](#interesting-quotes---representation)  
  - [learning and generalization](#interesting-quotes---learning-and-generalization)  
  - [symbolic approach](#interesting-quotes---symbolic-approach)  
  - [theory and black box](#interesting-quotes---theory-and-black-box)  
  - [unsupervised learning](#interesting-quotes---unsupervised-learning)  
  - [loss function and grounding](#interesting-quotes---loss-function-and-grounding)  
  - [bayesian inference and learning](#interesting-quotes---bayesian-inference-and-learning)  


  ----
  #### interesting quotes - architectures

  Juergen Schmidhuber:
  > "A search for solution-computing, perturbation-resistant, low-complexity neural networks describable by few bits of information can reduce overfitting and improve learning, including reinforcement learning in the case of partially observable environments. Deep learning often create hierarchies of more and more abstract representations of stationary data, sequential data or reinforcement learning policies. Unlike these systems, humans learn to actively perceive patterns by sequentially directing attention to relevant parts of the available data. Near future deep NNs will do so, too, extending previous work on neural networks that learn selective attention through reinforcement learning of (a) motor actions such as saccade control and (b) internal actions controlling spotlights of attention within RNNs, thus closing the general sensorimotor loop through both external and internal feedback. Many future deep neural networks will also take into account that it costs energy to activate neurons, and to send signals between them. Brains seem to minimize such computational costs during problem solving in at least two ways: (1) At a given time, only a small fraction of all neurons is active because local competition through winner-take-all mechanisms shuts down many neighbouring neurons, and only winners can activate other neurons through outgoing connections. (2) Numerous neurons are sparsely connected in a compact 3D volume by many short-range and few long-range connections (much like microchips in traditional supercomputers). Often neighbouring neurons are allocated to solve a single task, thus reducing communication costs. Physics seems to dictate that any efficient computational hardware will in the future also have to be brain-like in keeping with these two constraints. The most successful current deep recurrent neural networks, however, are not. Unlike certain spiking neural networks, they usually activate all units at least slightly, and tend to be strongly connected, ignoring natural constraints of 3D hardware. It should be possible to improve them by adopting (1) and (2), and by minimizing non-differentiable energy and communication costs through direct search in program (weight) space. These more brain-like RNNs will allocate neighboring RNN parts to related behaviors, and distant RNN parts to less related ones, thus self-modularizing in a way more general than that of traditional self-organizing maps in feedforward neural networks. They will also implement Occam’s razor as a by-product of energy minimization, by finding simple (highly generalizing) problem solutions that require few active neurons and few, mostly short connections. The more distant future may belong to general purpose learning algorithms that improve themselves in provably optimal ways, but these are not yet practical or commercially relevant."

  Geoffrey Hinton:
  > "Dumb stuff like stochastic gradient descent working so well raises huge problems for GOFAI advocates. These techniques are always going to beat a smart system which can't learn, provided they can learn a huge number of parameters. So the real lesson here is that dumb systems that learn are better than smart systems that don't. And the other lesson is, of course, that smart systems that learn billions of parameters are going to be even better. Models should be bigger than the data. Ex: your brain has many more synapses than experiences."

  Paul Mineiro:
  > "Gerald Tesauro dusted off his old Neurogammon code, ran it on a more powerful computer (his current laptop), and got much better results. Unfortunately, we cannot conclude that NVIDIA will solve AI for us if we wait long enough. In 2 player games or in simulated environments more generally, computational power equates to sample complexity, because you can simulate more. In the real world we have sample complexity constraints: you have to perform actual actions to get actual rewards. However, in the same way that cars and planes are faster than people because they have unfair energetic advantages (we are 100W machines; airplanes are much higher), I think “superhuman AI”, should it come about, will be because of sample complexity advantages, i.e., a distributed collection of robots that can perform more actions and experience more rewards (and remember and share all of them with each other). So really Boston Dynamics, not NVIDIA, is the key."

  Raia Hadsell:
  > "Biological brains are amazing. We have watched Lee Sedol deliberately change his style of play over this week, fluidly and consciously adapting and exploring. Presumably he will use the experience of winning Game 4 to further adapt to try to gain an advantage in Game 5. This points to one of the largest differences between human learning and modern machine learning. Deep networks, such as AlphaGo's policy and value nets, learn with lots of data and are generalists. They do not retain and refer back to individual examples, nor can they learn meaningfully from single examples. Moreover, if trained on data from a changing distribution, they will forget previous skills, quickly and catastrophically."

  Adam Ierymenko:
  > "The brain is not a neural network, and a neuron is not a switch. The brain contains a neural network. But saying that the brain “is” a neural network is like saying a city “is” buildings and roads and that’s all there is to it. The brain is not a simple network. It’s at the very least a nested set of networks of networks of networks with other complexities like epigenetics and hormonal systems sprinkled on top. You can’t just make a blind and sloppy analogy between neurons and transistors, peg the number of neurons in the brain on a Moore’s Law plot, and argue that human-level AI is coming Real Soon Now."

  Nando de Freitas:
  > "I think we are still missing good environments. I believe intelligent agents are mirrors of their environments. Our brain is the way it is because of being on planet earth. It is a consequence of evolution. However, we'd like to do things faster this time, so we need to make more progress in memory architectures, attention, concept and program induction, continual learning, teaching and social learning."


  ----
  #### interesting quotes - representation

  Juergen Schmidhuber:
  > "Artificial recursive neural networks are general computers which can learn algorithms to map input sequences to output sequences, with or without a teacher. They are computationally more powerful and biologically more plausible than other adaptive approaches such as Hidden Markov Models (no continuous internal states), feedforward networks and Support Vector Machines (no internal states at all). The program of an RNN is its weight matrix. Unlike feedforward NNs, RNNs can implement while loops, recursion, you name it. While FNNs are traditionally linked to concepts of statistical mechanics and traditional information theory, the programs of RNNs call for the framework of algorithmic information theory (or Kolmogorov complexity theory)."

  Antonio Valerio Miceli Barone:
  > "RNNs are Turing Complete in the limit of either unbounded numerical precision or infinite nodes. RNNs of finite size with finite precision can represent arbitrary finite state machines, just like any physical computer. It's a nice property to have, but it doesn't guarantee any good performance by itself. It's easy to come up with other schemes that are also Turing-complete in the limit but don't support efficient learning. Conversely, non-Turing-complete sequence learning methods (e.g. sliding window/n-gram methods) can be practically useful."

  Olivier Grisel:
  > "RNNs can't learn any algorithm. They can approximate any algorithm but there is no known and proven way to learn the weights for an arbitrary algorithm. They have the representation power but that does not mean that we can train them successfully for all tasks. When I say "can train them" I mean finding an algorithm that can optimize the weights till the error is zero on the training set irrespective of their generalization ability as measured on a validation set. It's the same for the MLP: we can prove that for any smooth function there exists suitable MLP weights to approximate that function with the MLP to an arbitrary precision level (universal approximator) provided enough hidden units. However we don't know how to set the weights for an arbitrary function. SGD with momentum on a least square objective function seem to work in many interesting cases but there is no proof it works for all the cases."

  Ilya Sutskever:
  > "Conventional statistical models learn simple patterns or clusters. In contrast, deep neural networks learn computation, albeit a massively parallel computation with a modest number of steps. Indeed, this is the key difference between DNNs and other statistical models. To elaborate further: it is well known that any algorithm can be implemented by an appropriate very deep circuit (with a layer for each timestep of the algorithm’s execution). What’s more, the deeper the circuit, the more expensive are the algorithms that can be implemented by the circuit (in terms of runtime). And given that neural networks are circuits as well, deeper neural networks can implement algorithms with more steps - which is why depth = more power. Surprisingly, neural networks are actually more efficient than boolean circuits. By more efficient, I mean that a fairly shallow DNN can solve problems that require many more layers of boolean circuits. For a specific example, consider the highly surprising fact that a DNN with 2 hidden layer and a modest number of units can sort N N-bit numbers! I found the result shocking when I heard about it, so I implemented a small neural network and trained it to sort 10 6-bit numbers, which was easy to do to my surprise. It is impossible to sort N N-bit numbers with a boolean circuit that has two hidden layers and that are not gigantic. The reason DNNs are more efficient than boolean circuits is because neurons perform a threshold operation, which cannot be done with a tiny boolean circuit."

  Ilya Sutskever:
  > "I don't see a particular difference between a shallow net with a reasonable number of neurons and a kernel machine with a reasonable number of support vectors (it's not useful to consider Kernel machines with exponentially many support vectors just like there isn't a point in considering the universal approximation theorem as both require exponential resources) - both of these models are nearly identical, and thus equally unpowerful. Both of these models will be inferior to an large deep neural network with a comparable number of parameters precisely because the DNN can do computation and the shallow models cannot. The DNN can sort, do integer-multiplication, compute analytic functions, decompose an input into small pieces and recombine it later in a higher level representation, partition the input space into an exponential number of non-arbitrary tiny regions, etc. Ultimately, if the DNN has 10,000 layers, then it can, in principle, execute any parallel algorithm that runs in fewer than 10,000 steps, giving this DNN an incredible expressive power. Now why is it that models that can do computation are in some sense "right" compared to models that cannot? Why is the inductive bias captured by DNNs "good", or even "correct"? Why do DNNs succeed on the natural problems that we often want to solve in practice? I think that it is a very nontrivial fact about the universe, and is a bit like asking "why are typical recognition problems solvable by an efficient computer program". I don't know the answer but I have two theories: 1) if they weren't solvable by an efficient computer program, then humans and animals wouldn't be solving them in the first place; and 2) there is something about the nature of physics and possibly even evolution that gives raise to problems that can usually be solvable by efficient algorithms."

  Yann LeCun:
  > "Take a binary input vector with N bits. There are 2^2^N possible boolean functions of these N bits. For any decent-size N, it's a ridiculously large number. Among all those functions, only a tiny, tiny proportion can be computed by a 2-layer network with a non-exponential number of hidden units. A less tiny (but still small) proportion can be computed by a multi-layer network with a less-than-exponential number of units. Among all the possible functions out there, the ones we are likely to want to learn are a tiny subset."

  Kevin Murphy:
  > "If by "deep learning" you mean "nested composition of functions", then that describes pretty much all of computing. However, the main problem (in my mind) is that current deep learning methods need too much time and data. This seems inconsistent with the ability of people to learn much more quickly from much smaller sample sizes (e.g., there are 100x more words in the NYT corpus than a child hears by the time they are 3). The key question is: what is the best form of representation (inductive bias) for learning? This of course depends on the task. Humans seem to use multiple forms of knowledge representation. For example, see Liz Spelke's work on "core knowledge" in children and also work by Josh Tenenbaum. This high level knowledge is of course represented by patterns of neuronal firing, but it might be statistically (and possibly computationally) more efficient to do learning by manipulating these more structured representations (e.g., in terms of objects and agents and their attributes and relations) rather than taking tiny steps in a super high dimensional continuous parameter space (although the latter approach does seem to be killing it right now)."

  Chris Olah:
  > "I’m really excited about the idea of having more “structured” representations. Right now, vectors are kind of the lingua franca of neural networks. Convolutional neural nets pass tensors, though, not just vectors. And recurrent neural nets lists of vectors. You can think of these as big vectors with metadata. That makes me wonder what other kinds of metadata we can add."

  Demis Hassabis:
  > "Intuition is an implicit knowledge acquired through experience but not consciously expressible or accessible. The existence and quality of this knowledge can be verified behaviourally. Creativity is an ability to synthesize knowledge to produce a novel or original idea."


  ----
  #### interesting quotes - learning and generalization

  Ilya Sutskever:
  > "The success of Deep Learning hinges on a very fortunate fact: that well-tuned and carefully-initialized stochastic gradient descent can train deep neural networks on problems that occur in practice. It is not a trivial fact since the training error of a neural network as a function of its weights is highly non-convex. And when it comes to non-convex optimization, we were taught that all bets are off. Only convex is good, and non-convex is bad. And yet, somehow, stochastic gradient descent seems to be very good at training those large deep neural networks on the tasks that we care about. The problem of training neural networks is NP-hard, and in fact there exists a family of datasets such that the problem of finding the best neural network with three hidden units is NP-hard. And yet, SGD just solves it in practice. My hypothesis (which is shared by many other scientists) is that neural networks start their learning process by noticing the most “blatant” correlations between the input and the output, and once they notice them they introduce several hidden units to detect them, which enables the neural network to see more complicated correlations."

  Ilya Sutskever:
  > "While it is very difficult to say anything specific about the precise nature of the optimization of neural networks (except near a local minimum where everything becomes convex and uninteresting), we can say something nontrivial and specific about generalization. And the thing we can say is the following: in his famous 1984 paper called "A Theory of the Learnable", Valiant proved, roughly speaking, that if you have a finite number of functions, say N, then every training error will be close to every test error once you have more than log N training cases by a small constant factor. Clearly, if every training error is close to its test error, then overfitting is basically impossible (overfitting occurs when the gap between the training and the test error is large). (I am also told that this result was given in Vapnik’s book as small exercise). But this very simple result has a genuine implication to any implementation of neural networks. Suppose I have a neural network with N parameters. Each parameter will be a float32. So a neural network is specified with 32N bits, which means that we have no more than 2^32N distinct neural networks, and probably much less. This means that we won’t overfit much once we have more than 32N training cases. Which is nice. It means that it’s theoretically OK to count parameters. What’s more, if we are quite confident that each weight only requires 4 bits (say), and that everything else is just noise, then we can be fairly confident that the number of training cases will be a small constant factor of 4N rather than 32N."

  Ilya Sutskever:
  > "We know that most machine learning algorithms are consistent: that is, they will solve the problem given enough data. But consistency generally requires an exponentially large amount of data. For example, the nearest neighbor algorithm can definitely solve any problem by memorizing the correct answer to every conceivable input. The same is true for a support vector machine - we’d have a support vector for almost every possible training case for very hard problems. The same is also true for a neural network with a single hidden layer: if we have a neuron for every conceivable training case, so that neuron fires for that training case and but not for any other, then we could also learn and represent every conceivable function from inputs to outputs. Everything can be done given exponential resources, but it is never ever going to be relevant in our limited physical universe. And it is in this point that deep neural networks differ from previous methods: we can be reasonably certain that a large but not huge network will achieve good results on a surprising variety of problems that we may want to solve. If a problem can be solved by a human in a fraction of a second, then we have a very non-exponential super-pessimistic upper bound on the size of the smallest neural network that can achieve very good performance. But I must admit that it is impossible to predict whether a given problem will be solvable by a deep neural network ahead of time, although it is often possible to tell whenever we know that a similar problem can be solved by an neural network of a manageable size. So that’s it, then. Given a problem, such as visual object recognition, all we need is to train a giant convolutional neural network with 50 layers. Clearly a giant convnet with 50 layers can be configured to achieve human-level performance on object recognition, right? So we simply need to find these weights."

  John Cook:
  > "Simple models often outperform complex models in complex situations. As examples, sports prediction, diagnosing heart attacks, locating serial criminals, picking stocks, and  understanding spending patterns. Complex environments often instead call for simple decision rules. That is because these rules are more robust to ignorance. And yet behind every complex set of rules is a paper showing that it outperforms simple rules, under conditions of its author’s choosing. That is, the person proposing the complex model picks the scenarios for comparison. Unfortunately, the world throws at us scenarios not of our choosing. Simpler methods may perform better when model assumptions are violated. And model assumptions are always violated, at least to some extent."

  Claudia Perlich:
  > "If the signal to noise ratio is high, trees and neural networks tend to win logistic regression. But, if you have very noisy problems and the best model has an AUC<0.8 - logistic beats trees and neural networks almost always. Ultimately not very surprising: if the signal is too weak, high variance models get lost in the weeds. So what does this mean in practice? The type of problems I tend to deal with are super noisy with low level of predictability. Think of it in the terms of deterministic (chess) all the way to random (supposedly the stock market). Some problems are just more predictable (given the data you have) than others. And this is not a question of the algorithms but rather a conceptual statement about the world. Deep learning is really great on the other end - “Is this picture showing a cat?”. In the world of uncertainty, the bias variance tradeoff still often ends up being favorable on the side of more bias - meaning, you want a ‘simple’ very constrained model. And this is where logistic regression comes in. I personally have found it much easier to ‘beef up’ a simple linear model by adding complicated features than trying to constrain a very powerful (high variance) model class."

  Yoshua Bengio:
  > "So long as our machine learning models cheat by relying only on surface statistical regularities, they remain vulnerable to out-of-distribution examples. Humans generalize better than other animals by implicitly having a more accurate internal model of the underlying causal relationships. This allows one to predict future situations (e.g. the effect of planned actions) that are far from anything seen before, an essential component of reasoning, intelligence and science."

  Yoshua Bengio:
  > "I am sorry to say that model selection and regularization are NOT hacky. They are crucial ways of expressing your priors and make all the difference between a poor model and a good one. However, we want BROAD priors, or if you want, general-purpose hacks (which I would not call hacks, of course). Check back on the No-Free-Lunch theorem!"

  Chris Olah:
  > "One criticism of deep learning is that by using massive amounts of data, the network has effectively memorised all possible inputs, how do you counter that? I wish network could memorize all possible inputs. Sadly, there’s a problem called the curse of dimensionality. For high-dimensional inputs, there’s such a vast, exponentially large space of possible inputs, that you’ll never be able to fill it."

  > "Parity is an adverserial learning task for neural nets that Minsky and Papart came up with like an eternity ago. Basically, since parity is a global function of the input vector (any feature detector that only recieves input from a strict subset of the input vector wil carry zero information on the parity of the input) it's incredibly difficult (as in statistically impossible) for a neural net to learn parity, since learining rules are local to each edge weight (as in an update to a edge weight doesn't depend on what the other edges are updating to) in the current learing algorithms for neural nets."

  Luke Vilnis:
  > "Modeling probability distributions over somewhat unordered (or exchangeable) data is difficult for RNNs due to the need to pick some sequential factorization of the probability distribution. Modeling probability distributions invariant to ordering of variables with neural networks is an interesting research topic. Along this line, there are combinatorial potentials (like "at most k variables of this type can be on at once") that are difficult to model with this sort of directed factorization. Generally, RNNs are very powerful models in the style of directed graphical models ("Bayes nets"), and traditional strengths of undirected models (direction-agnostic dependencies) are not their strong suit."

  Christian Szegedy:
  > "What has been discovered is that a single neuron's feature is no more interpretable as a meaningful feature than a random set of neurons. That is, if you pick a random set of neurons and find the images that produce the maximum output on the set then these images are just as semantically similar as in the single neuron case. This means that neural networks do not "unscramble" the data by mapping features to individual neurons in say the final layer. The information that the network extracts is just as much distributed across all of the neurons as it is localized in a single neuron. Every deep neural network has "blind spots" in the sense that there are inputs that are very close to correctly classified examples that are misclassified. For all the networks we studied, for each sample, we always manage to generate very close, visually indistinguishable, adversarial examples that are misclassified by the original network. What is even more shocking is that the adversarial examples seem to have some sort of universality. That is a large fraction were misclassified by different network architectures trained on the same data and by networks trained on a different data set. The above observations suggest that adversarial examples are somewhat universal and not just the results of overfitting to a particular model or to the specific selection of the training set. One possible explanation is that this is another manifestation of the curse of dimensionality. As the dimension of a space increases it is well known that the volume of a hypersphere becomes increasingly concentrated at its surface. (The volume that is not near the surface drops exponentially with increasing dimension.) Given that the decision boundaries of a deep neural network are in a very high dimensional space it seems reasonable that most correctly classified examples are going to be close to the decision boundary - hence the ability to find a misclassified example close to the correct one, you simply have to work out the direction to the closest boundary."

  Ian Goodfellow:
  > "The criticism of deep networks as vulnerable to adversarial examples is misguided, because unlike shallow linear models, deep networks are at least able to represent functions that resist adversarial perturbation. The universal approximator theorem (Horniket al, 1989) guarantees that a neural network with at least one hidden layer can represent any function to an arbitrary degree of accuracy so long as its hidden layer is permitted to have enough units."

  Yoshua Bengio:
  > "My conjecture is that *good* unsupervised learning should generally be much more robust to adversarial examples because it tries to discriminate the data manifold from its surroundings, in ALL non-manifold directions (at every point on the manifold). This is in contrast with supervised learning, which only needs to worry about the directions that discriminate between the observed classes. Because the number of classes is much less than the dimensionality of the space, for image data, supervised learning is therefore highly underconstrained, leaving many directions of changed "unchecked" (i.e. to which the model is either insensitive when it should not or too sensitive in the wrong way)."

  Ian Goodfellow:
  > "Model-based optimization, or as I like to call it, “the automatic inventor”, is a huge future application. Right now we make models that take some input, and produce some output. We put in a photo, the model outputs a value saying that it is a cat. In the future (and to a limited extent, now), we will be able to use optimization algorithms to search for the input to the model that yields the optimal output. Suppose we have a model that looks at the blueprints for a car and predicts how fast the car will go. We can then use gradient descent on a continuous representation of the blueprint to optimize for the fastest car. Right now, this approach doesn’t work very well, because you don’t get an input that is actually optimal in the real world. Instead, you get an adversarial example that the model thinks will perform great but turns out to perform poorly in the real world. For example, if you start your optimization with a photo of an airplane, then use gradient descent to search for an image that is classified as a cat, gradient descent will find an image that still looks like an airplane to a human observer but is classified as a cat by the model. In the future, when we have fixed the adversarial example problem, we’ll be able to build deep nets that estimate the effectiveness of medicinal drugs, genes, and other things that are too complex for people to design efficiently. We’ll then be able to invent new drugs and discover new useful genes by using gradient descent on a continuous representation of the design space."


  ----
  #### interesting quotes - symbolic approach

  Geoffrey Hinton:
  > "The fathers of AI believed that formal logic provided insight into how human reasoning must work. For implications to travel from one sentence to the next, there had to be rules of inference containing variables that got bound to symbols in the first sentence and carried the implications to the second sentence. I shall demonstrate that this belief is as incorrect as the belief that a lightwave can only travel through space by causing disturbances in the luminiferous aether. In both cases, scientists were misled by compelling but incorrect analogies to the only systems they knew that had the required properties. Arguments have little impact on such strongly held beliefs. What is needed is a demonstration that it is possible to propagate implications in some quite different way that does not involve rules of inference and has no resemblance to formal logic. Recent results in machine translation using recurrent neural networks show that the meaning of a sentence can be captured by a "thought vector" which is simply the hidden state vector of a recurrent net that has read the sentence one word at a time. In future, it will be possible to predict thought vectors from the sequence of previous thought vectors and this will capture natural human reasoning. With sufficient effort, it may even be possible to train such a system to ignore nearly all of the contents of its thoughts and to make predictions based purely on those features of the thoughts that capture the logical form of the sentences used to express them."

  Geoffrey Hinton:
  > "If we can convert a sentence into a vector that captures the meaning of the sentence, then google can do much better searches, they can search based on what is being said in a document. Also, if you can convert each sentence in a document into a vector, you can then take that sequence of vectors and try and model why you get this vector after you get these vectors, that's called reasoning, that's natural reasoning, and that was kind of the core of good old fashioned AI and something they could never do because natural reasoning is a complicated business, and logic isn't a very good model of it, here we can say, well, look, if we can read every english document on the web, and turn each sentence into a thought vector, we've got plenty of data for training a system that can reason like people do. Now, you might not want to reason like people do on the web, but at least we can see what they would think."

  Geoffrey Hinton:
  > "Most people fall for the traditional AI fallacy that thought in the brain must somehow resemble lisp expressions. You can tell someone what thought you are having by producing a string of words that would normally give rise to that thought but this doesn't mean the thought is a string of symbols in some unambiguous internal language. The new recurrent network translation models make it clear that you can get a very long way by treating a thought as a big state vector. Traditional AI researchers will be horrified by the view that thoughts are merely the hidden states of a recurrent net and even more horrified by the idea that reasoning is just sequences of such state vectors. That's why I think its currently very important to get our critics to state, in a clearly decideable way, what it is they think these nets won't be able to learn to do. Otherwise each advance of neural networks will be met by a new reason for why that advance does not really count. So far, I have got both Garry Marcus and Hector Levesque to agree that they will be impressed if neural nets can correctly answer questions about "Winograd" sentences such as "The city councilmen refused to give the demonstrators a licence because they feared violence." Who feared the violence?"

  Geoffrey Hinton:
  > "There are no symbols inside the encoder and decoder neural nets for machine translation. The only symbols are at the input and output. Processing pixel arrays is not done by manipulating internal pixels. Maybe processing symbol strings is not done by manipulating internal symbol strings. It was obvious to physicists that light waves must have an aether to propagate from one place to the next. They thought there was no other possibility. It was obvious to AI researchers that people must use formal rules of inference to propagate implications from one proposition to the next. They thought there was no other possibility. What is inside the black box is not necessarily what goes in or what comes out. The physical symbol system hypothesis is probably false. Get over it."

  Juergen Schmidhuber:
  > "Where do the symbols and self-symbols underlying consciousness and sentience come from? I think they come from data compression during problem solving. While a problem solver is interacting with the world, it should store the entire raw history of actions and sensory observations including reward signals. The data is ‘holy’ as it is the only basis of all that can be known about the world. If you can store the data, do not throw it away! Brains may have enough storage capacity to store 100 years of lifetime at reasonable resolution. As we interact with the world to achieve goals, we are constructing internal models of the world, predicting and thus partially compressing the data history we are observing. If the predictor/compressor is a biological or artificial recurrent neural network (RNN), it will automatically create feature hierarchies, lower level neurons corresponding to simple feature detectors similar to those found in human brains, higher layer neurons typically corresponding to more abstract features, but fine-grained where necessary. Like any good compressor, the RNN will learn to identify shared regularities among different already existing internal data structures, and generate prototype encodings (across neuron populations) or symbols for frequently occurring observation sub-sequences, to shrink the storage space needed for the whole (we see this in our artificial RNNs all the time). Self-symbols may be viewed as a by-product of this, since there is one thing that is involved in all actions and sensory inputs of the agent, namely, the agent itself. To efficiently encode the entire data history through predictive coding, it will profit from creating some sort of internal prototype symbol or code (e.g. a neural activity pattern) representing itself. Whenever this representation becomes activated above a certain threshold, say, by activating the corresponding neurons through new incoming sensory inputs or an internal ‘search light’ or otherwise, the agent could be called self-aware. No need to see this as a mysterious process - it is just a natural by-product of partially compressing the observation history by efficiently encoding frequent observations."

  Adam Ierymenko:
  > "Imagine if back in Newton's day, they were analyzing data from physical random variables with deep neural networks. Sure, they might get great prediction accuracy on how far a ball will go given measurements of its weight, initial force/angle, and some other irrelevant variables, but would this really be the best approach to discover all of the useful laws of physics such as f = ma and the conversion from potential to kinetic energy via the gravitational constant? Probably not, in fact the predictions might be in some sense "too good" incorporating other confounding effects such as air drag and the shape / spin of the ball which obfuscate the desired law. In many settings where an interpretation of what is going on in the data is desired, a clear model is necessary with simple knobs that have clear effects when turned. This may also be a requirement not only for human interpretation, but an also AI system which is able to learn and combine facts about the world (rather than only storing the complex functions which represent the relationships between things as inferred by a neural network)."

  Nando de Freitas:
  > "For me there are two types of generalisation, which I will refer to as Symbolic and Connectionist generalisation. If we teach a machine to sort sequences of numbers of up to length 10 or 100, we should expect them to sort sequences of length 1000 say. Obviously symbolic approaches have no problem with this form of generalisation, but neural nets do poorly. On the other hand, neural nets are very good at generalising from data (such as images), but symbolic approaches do poorly here. One of the holy grails is to build machines that are capable of both symbolic and connectionist generalisation. Neural Programmer Interpreters is a very early step toward this. NPI can do symbolic operations such as sorting and addition, but it can also plan by taking images as input and it's able to generalise the plans to different images (e.g. in the NPI car example, the cars are test set cars not seen before)."

  Christian Szegedy:
  > "The inroads of machine learning will transform all of information technologies. Most prominently, the way we program our computers will slowly shift from prescribing how to solve problems to just specifying them and let machines learn to cope with them. We could even have them distill their solution to formal procedures akin to our current programs. In order to truly get there, the most exciting developments will come from the synergy of currently disjoint areas: the marriage of formal, discrete methods and fuzzy, probabilistic approaches, like deep neural networks."

  Josh Tenenbaum:
  > "From early infancy, human thought is structured around a basic understanding of physical objects, intentional agents, and their causal interactions. Reverse-engineering core KRR is easier than, and a valuable (essential?) precursor for getting later, language-based KRR right. Probabilistic programs will let us build quantitative, reverse-engineering models of core KRR, and later language-base KRR as well, capturing these key features of common-sense thought:
  > - probabilistic
  > - causal
  > - compositional
  > - enabled by built-in primitives (objects, forces, agents, goals)
  > - inference by simulation, more flexible than neural networks (pattern matching, vector spaces), more robust than logic"

  Josh Tenenbaum:
  > "Intelligence is not about pattern recognition. It's about modelling the world:
  > - explaining and understanding what we see
  > - imagining things we could see but haven't yet
  > - problem solving and planning to make these things real
  > - building new models as we learn more about the world"

  Josh Tenenbaum:
  > "There is no integration of neural and symbolic approaches to common-sense reasoning. Common-sense reasoning is symbolic (and many other things that integrate naturally with symbols: probabilistic, causal, object and agent-based). The idea that neural nets (in any of their current forms) are going to be able to read all the text on the web and then perform common-sense reasoning is ridiculous. The knowledge representation and reasoning mechanisms that are being explored are too weak. My guess: Neural nets could play a role but not in reasoning. Yet neural nets might still be very helpful, in mapping between natural language and a probabilistic-logical language of thought."


  ----
  #### interesting quotes - theory and black box

  Jonathan Huggins:
  > "There are two main flavors of learning theory, statistical learning theory (StatLT) and computational learning (CompLT). StatLT originated with Vladimir Vapnik, while the canonical example of CompLT, PAC learning, was formulated by Leslie Valiant. StatLT, in line with its “statistical” descriptor, focuses on asymptotic questions (though generally based on useful non-asymptotic bounds). It is less concerned with computational efficiency, which is where CompLT comes in. Computer scientists are all about efficient algorithms (which for the purposes of theory essentially means polynomial vs. super-polynomial time). Generally, StatLT results apply to a wider variety of hypothesis classes, with few or no assumptions made about the concept class (a concept class refers to the class of functions to which the data generating mechanism belongs). CompLT results apply to very specific concept classes but have stronger performance guarantees, often using polynomial time algorithms."

  Michael I. Jordan:
  > "Throughout the eighties and nineties, it was striking how many times people working within the "ML community" realized that their ideas had had a lengthy pre-history in statistics. Decision trees, nearest neighbor, logistic regression, kernels, PCA, canonical correlation, graphical models, K means and discriminant analysis come to mind, and also many general methodological principles (e.g., method of moments, which is having a mini-renaissance, Bayesian inference methods of all kinds, M estimation, bootstrap, cross-validation, ROC, and of course stochastic gradient descent, whose pre-history goes back to the 50s and beyond), and many many theoretical tools (large deviations, concentrations, empirical processes, Bernstein-von Mises, U statistics, etc). Of course, the "statistics community" was also not ever that well defined, and while ideas such as Kalman filters, HMMs and factor analysis originated outside of the "statistics community" narrowly defined, there were absorbed within statistics because they're clearly about inference. Similarly, layered neural networks can and should be viewed as nonparametric function estimators, objects to be analyzed statistically."

  Michael Nielsen:
  > "Maybe the real problem is that our 30 hidden neuron network will never work well, no matter how the other hyper-parameters are chosen? Maybe we really need at least 100 hidden neurons? Or 300 hidden neurons? Or multiple hidden layers? Or a different approach to encoding the output? Maybe our network is learning, but we need to train for more epochs? Maybe the mini-batches are too small? Maybe we'd do better switching back to the quadratic cost function? Maybe we need to try a different approach to weight initialization? And so on, on and on and on. In many parts of science - especially those parts that deal with simple phenomena - it's possible to obtain very solid, very reliable evidence for quite general hypotheses. But in neural networks there are large numbers of parameters and hyper-parameters, and extremely complex interactions between them. In such extraordinarily complex systems it's exceedingly difficult to establish reliable general statements. Understanding neural networks in their full generality is a problem that, like quantum foundations, tests the limits of the human mind. Instead, we often make do with evidence for or against a few specific instances of a general statement. As a result those statements sometimes later need to be modified or abandoned, when new evidence comes to light."

  Paul Mineiro:
  > "Paper on neural machine translation by jointly learning to align and translate excels as an example of the learned representation design process. Deep learning is not merely the application of highly flexible model classes to large amounts of data: if it were that simple, the Gaussian kernel would have solved AI. Instead, deep learning is like the rest of machine learning: navigating the delicate balance between model complexity and data resources, subject to computational constraints. In particular, more data and a faster GPU would not create these kinds of improvements in the standard neural encoder/decoder architecture because of the mismatch between the latent vector representation and the sequence-to-sequence mapping being approximated. A much better approach is to judiciously increase model complexity in a manner that better matches the target. Furthermore, the “art” is not in knowing that alignments are important per se (the inspiration is clearly from existing statistical translation systems), but in figuring out how to incorporate alignment-like operations into the architecture without destroying the ability to optimize using SGD. Note that while a representation is being learned from data, clearly the human designers have gifted the system with a strong prior via the specification of the architecture (as with deep convolutional networks). We should anticipate this will continue to be the case for the near future, as we will always be data impoverished relative to the complexity of the hypothesis classes we'd like to consider. Anybody who says to you “I'm using deep learning because I want to learn from the raw data without making any assumptions” doesn't get it. If they also use the phrase “universal approximator”, exit the conversation and run away as fast as possible, because nothing is more dangerous than an incorrect intuition expressed with high precision."

  Nando de Freitas:
  > "Many recent developments blur the distinction between model and algorithm. This is profound - at least for someone with training in statistics. Ziyu Wang recently replaced the convnet of DQN (DeepMind's Atari reinforcement learning agent) and re-run exactly the same algorithm but with a different net (a slight modification of the old net with two streams which he calls the dueling architecture). That is, everything is the same, but only the representation (neural net) changed slightly to allow for computation of not only the Q function, but also the value and advantage functions. The simple modification resulted in a massive performance boost. For example, for the Seaquest game, the DQN of the Nature paper scored 4,216 points, while the modified net of Ziyu leads to a score of 37,361 points. For comparison, the best human we have found scores 40,425 points. Importantly, many modifications of DQN only improve on the 4,216 score by a few hundred points, while the Ziyu's network change using the old vanilla DQN code and gradient clipping increases the score by nearly a factor of 10. I emphasize that what Ziyu did was he changed the network. He did not change the algorithm. However, the computations performed by the agent changed remarkably. Moreover, the modified net could be used by any other Q learning algorithm. Reinforcement learning people typically try to change equations and write new algorithms, instead here the thing that changed was the net. The equations are implicit in the network. One can either construct networks or play with equations to achieve similar goals."

  Leon Bottou:
  > "When attainable, theoretical guarantees are beautiful. They reflect clear thinking and provide deep insight to the structure of a problem. Given a working algorithm, a theory which explains its performance deepens understanding and provides a basis for further intuition. Given the absence of a working algorithm, theory offers a path of attack. However, there is also beauty in the idea that well-founded intuitions paired with rigorous empirical study can yield consistently functioning systems that outperform better-understood models, and sometimes even humans at many important tasks. Empiricism offers a path forward for applications where formal analysis is stifled, and potentially opens new directions that might eventually admit deeper theoretical understanding in the future."

  Stephen Hsu:
  > "In many parts of science - especially those parts that deal with simple phenomena - it's possible to obtain very solid, very reliable evidence for quite general hypotheses. But in neural networks there are large numbers of parameters and hyper-parameters, and extremely complex interactions between them. In such extraordinarily complex systems it's exceedingly difficult to establish reliable general statements. Understanding neural networks in their full generality is a problem that, like quantum foundations, tests the limits of the human mind. Instead, we often make do with evidence for or against a few specific instances of a general statement. As a result those statements sometimes later need to be modified or abandoned, when new evidence comes to light. Any heuristic story about neural networks carries with it an implied challenge. For example, consider the statement, explaining why dropout works: "This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons." This is a rich, provocative statement, and one could build a fruitful research program entirely around unpacking the statement, figuring out what in it is true, what is false, what needs variation and refinement. Indeed, there is now a small industry of researchers who are investigating dropout (and many variations), trying to understand how it works, and what its limits are. And so it goes with many of the heuristics we've discussed. Each heuristic is not just a (potential) explanation, it's also a challenge to investigate and understand in more detail. Of course, there is not time for any single person to investigate all these heuristic explanations in depth. It's going to take decades (or longer) for the community of neural networks researchers to develop a really powerful, evidence-based theory of how neural networks learn. Does this mean you should reject heuristic explanations as unrigorous, and not sufficiently evidence-based? No! In fact, we need such heuristics to inspire and guide our thinking. It's like the great age of exploration: the early explorers sometimes explored (and made new discoveries) on the basis of beliefs which were wrong in important ways. Later, those mistakes were corrected as we filled in our knowledge of geography. When you understand something poorly - as the explorers understood geography, and as we understand neural nets today - it's more important to explore boldly than it is to be rigorously correct in every step of your thinking. And so you should view these stories as a useful guide to how to think about neural nets, while retaining a healthy awareness of the limitations of such stories, and carefully keeping track of just how strong the evidence is for any given line of reasoning. Put another way, we need good stories to help motivate and inspire us, and rigorous in-depth investigation in order to uncover the real facts of the matter."

  Geoffrey Hinton:
  > "I suspect that in the end, understanding how big artificial neural networks work after they have learned will be quite like trying to understand how the brain works but with some very important differences:
  > - We know exactly what each neuron computes.
  > - We know the learning algorithm they are using.
  > - We know exactly how they are connected.
  > - We can control the input and observe the behaviour of any subset of the neurons for as long as we like.
  > - We can interfere in all sorts of ways without filling in forms."

  Yoshua Bengio:
  > "There are 4 factors that explain the success of deep learning: (1) computing power, (2) large datasets, (3) large flexible models and (4) powerful biases (preferences in the space of functions, or priors in Bayesian parlance). Deep nets benefit from built-in assumptions about the underlying data, including: assumption of multiple underlying factors (distributed representations, causality), assumption of composition of factors (depth), equivariance and temporal coherence assumptions (in convolutional nets), temporal stationarity (in recurrent nets), etc. Although the first 3 factors are mostly in the realm of computer science, the last and probably most interesting one clearly involves thinking in a statistical way. There is little hope to interpret the billions of parameters that large deep nets are learning, but there is hope to understand the priors implicitly or explicitly introduced in these networks."

  Yann LeCun:
  > "I do think that there is a need for better theoretical understanding of deep learning. But if a method works, it should not be abandoned nor dismissed just because theorists haven’t yet figured out how to explain it. The field of machine learning made that mistake in the mid 1990s, largely dismissing neural nets (and sometimes even making fun of it). The reasons for this are complicated, but that clearly was a bad collective mistake in that the field was set back by at least a decade. One theoretical puzzle is why the type of non-convex optimization that needs to be done when training deep neural nets seems to work reliably. A naive intuition would suggest that optimizing a non-convex function is difficult because we can get trapped in local minima and get slowed down by plateaus and saddle points. While plateaus and saddle points can be a problem, local minima never seem to cause problems. Our intuition is wrong, because we picture an energy landscape in low dimension (e.g. 2 or 3). But the objective function of deep neural nets is often in 100 million dimensions or more. It’s hard to build a box in 100 million dimensions. That’s a lot of walls. By working strictly on methods that you can fully analyze theoretically, you confine yourself to using excessively naive methods. Physicists don’t work like that. They don’t get to choose the complexity of the systems they study: the physical world is what it is. To them, complex systems are more interesting. For example, a lot of interesting mathematics and theoretical physics methods were developed in the context of studying spin glasses and other “disordered” systems. Physicists couldn’t simply choose to not study these systems because they were too complicated. On our engineering-oriented field, in which the systems we study are artifact of our own creation, we can be tempted to simplify those creations in order to analyze them more easily. But if we over-simplify them in the process in such a way that they no longer work, we have thrown the baby with the bath water."

  Yann LeCun:
  > "I don’t think there is a choice to make between performance and theory. If there is performance, there will be theory to explain it. Also, what kind of theory are we talking about? Is it a generalization bound? Convnets have a finite VC dimension, hence they are consistent and admit the classical VC bounds. What more do you want? Do you want a tighter bound, like what you get for SVMs? No theoretical bound that I know of is tight enough to be useful in practice. So I really don’t understand the point. Sure, generic VC bounds are atrociously non tight, but non-generic bounds (like for SVMs) are only slightly less atrociously non tight. No one uses generalization bounds to do model selection. Everyone in their right mind use (cross)validation. If what you desire are convergence proofs (or guarantees), that’s a little more complicated. The loss function of multi-layer nets is non-convex, so the easy proofs that assume convexity are out the window. But we all know that in practice, a convnet will almost always converge to the same level of performance, regardless of the starting point (if the initialization is done properly). There is theoretical evidence that there are lots and lots of equivalent local minima and a very small number of “bad” local minima. Hence convergence is rarely a problem."

  Yann LeCun:
  > "Simple and general theorems are good. Thermodynamics gave us principles that prevented us from wasting our time looking for perfectly efficient thermal machines or perpetual motion. We already have such theorems in ML that apply to just about every learning machine, including neural networks (e.g. VC theory consistency/capacity, no-free-lunch, etc). But it could very well be that we won't have "simple" theorems that are more specific to neural networks, for the same reasons we don't have analytical solutions of Navier-Stokes or the 3-body problem."


  ----
  #### interesting quotes - unsupervised learning

  Vincent van Houcke:
  > "I think of deep learning as being to machine learning what something like matrices are to math: it's a small, foundational part of machine learning, it provides a basic unifying vocabulary and a convenient elementary building block: anywhere you have X, Y data, you can throw a deep net at it an reasonably expect predict Y from X; bonus: the mapping is differentiable. The real interesting question in ML is what having this elementary building block enables. True learning is not about mapping X to Ys: there is in general no Y to begin with."

  Yann LeCun:
  > "Unsupervised learning is about discovering the internal structure of the data, discovering mutual dependencies between input variables, and disentangling the independent explanatory factors of variations. Generally, unsupervised learning is a means to an end. There are four main uses for unsupervised learning: (1) learning features (or representations); (2) visualization/exploration; (3) compression; (4) synthesis."

  Yann LeCun:
  > "Unsupervised learning is crucial to approach AI for a number of fundamental reasons, including the abundance of unlabeled data and the observed fact that representation learning (whether supervised or unsupervised) allows transfer learning, allowing to learn from very few labelled examples some new categories. Of course this is only possible if the learner has previously learned good representations from related categories, but with the AlexNet, it has clearly been shown that from 1000 object categories you can generalize to new categories with just a few examples. This has been demonstrated in many papers using unsupervised transfer learning. More recently, Socher showed that you can even get some decent generalization from zero examples simply because you know things from multiple modalities (e.g., that 'dog' and 'cat' are semantically similar in sentences, so that you can guess that something in an image could be a dog even if you have only seen images of cats). So you can't use deep learning on a new field for which there is very little data if there is no relationship with what the learner has learned previously, but that is also true of humans."

  > "Generative models are important for most current conceptions of how general AI could/should work. You learn a mostly unsupervised generative model of the future, you then sample from that to create predicted future sequences, and then you can feed those into a planning engine. For a simpler world like go you can use something like MCTS and get superhuman results already. That doesn't scale well for more complex environments. So basically, figuring out to learn efficient deep generative models in a scalable unsupervised way is a key unsolved problem for general AI."

  > "The trick is that the neural networks we use as generative models have a number of parameters significantly smaller than the amount of data we train them on, so the models are forced to discover and efficiently internalize the essence of the data in order to generate it. These models usually have only about 100 million parameters, so a network trained on ImageNet has to (lossily) compress 200GB of pixel data into 100MB of weights. This incentivizes it to discover the most salient features of the data: for example, it will likely learn that pixels nearby are likely to have the same color, or that the world is made up of horizontal or vertical edges, or blobs of different colors. Eventually, the model may discover many more complex regularities: that there are certain types of backgrounds, objects, textures, that they occur in certain likely arrangements, or that they transform in certain ways over time in videos, etc. In the long run, they hold the potential to automatically learn the natural features of a dataset, whether categories or dimensions or something else entirely."

  Kevin Murphy:
  > "The most important unresolved problem is unsupervised learning. In particular, what objective function should we use? Maximizing likelihood of the observed data, or even of future observed data, seems like the wrong thing to aim for. Consider, for example, predicting every pixel in the next N frames of video. Do we care about the exact intensity values? No, we care about predicting what the world is going to do next (will the car turn left or right? will the glass break if I drop it?). Somehow humans and animals seem to learn to predict at this higher level of abstraction, in terms of objects and relations, without ever receiving any such labeled data. Multi-task reinforcement learning will help, but learning from scalar reward alone seems too limited. Learning to predict the outcome of one's actions seems like it might help (and this can be used in goal-based planning)."

  Andrej Karpathy:
  > "But wait, humans learn unsupervised - why give up? We might just be missing something conceptually!,- I've heard some of my friends argue. The premise may, unfortunately be false: humans have temporally contiguous RGBD perception and take heavy advantage of Active Learning, Curriculum Learning, and Reinforcement Learning, with help from various pre-wired neural circuits. Imagine a (gruesome) experiment in which we'd sit a toddler in front of a monitor and flash random internet images at him/her for months. Would we expect them to develop the same understanding of the visual world? Because that's what we're currently trying to get working with computers. The strengths, weaknesses and types of data practically available to humans and computers are fundamentally misaligned."

  Kyle Kastner:
  > "Weak labels and other tricks seem to me a better and more direct angle than going straight into reinforcement learning. There are a lot of weak labels we can make for tons of inputs, with or without adding domain expertise to make even stronger "weak" losses that are much easier than learning a generative model. Maybe using these types losses to complement/bootstrap the generative process makes sense? Reinforcement Learning is neat and makes a ton of sense for control related problems, but there is a lot of work in trying to stabilize these types of techniques for even relatively local tasks - long term dependencies/credit assignment is still brutal in supervised models, let alone ones with extremely noisy gradients. Human learning is guided by large amounts of weak labels that are present (through the underlying physical laws, actually a very powerful supervisor) in our learning environment. Therefore, saying that 'most of human learning is unsupervised' (as it is often done) is in my opinion wrong. As another side note, the (huge) set of weak label-types itself has a learneable structure which could also be exploited."

  Nando de Freitas:
  > "For me, learning is never unsupervised. Whether predicting the current data (autoencoders), next frames, other data modalities, etc., there always appears to be a target. The real question is how do we come up with good target signals (labels) automatically for learning? This question is currently being answered by people who spend a lot of time labelling datasets like ImageNet. Also I think unsupervised learning can be a trap. The Neocognitron had convolution, pooling, contrast normalization and ReLUs already in the 70s. This is precisely the architecture that so many of us now use. The key difference is that we learn these models in supervised fashion with backprop. Fukushima focused more on trying to come up with biologically plausible algorithms and unsupervised learning schemes."

  Juergen Schmidhuber:
  > "There was a time when I thought unsupervised learning is indispensable. My first deep learner of 1991 used unsupervised learning-based pre-training for a stack of recurrent neural networks. Each RNN is trained for a while by unsupervised learning to predict its next input. From then on, only unexpected inputs (errors) convey new information and get fed into next higher RNN which thus ticks on a slower, self-organising time scale. We get less and less redundant input sequence encodings in deeper and deeper levels of this hierarchical temporal memory, which compresses data in both space (like feedforward NN) and time. In one ancient illustrative experiment of 1993 the top level code got so compact that subsequent supervised learning across 1200 time steps (= 1200 virtual layers) became trivial. With the advent of LSTM RNN, however, pure supervised learning without any unsupervised learning could perform similar feats. And today it is mainly pure supervised learning systems (RNN and feedforward NN) that are winning the competitions. Some say that in case of small data sets we still need unsupervised learning. But even then it may be enough to start with nets pretrained by supervised learning on different data sets, to get useful codes of new data in deep layers - ideally factorial codes, an ultimate goal of unsupervised learning for NN. Note that supervised learning on top of a factorial code is trivial - a naive Bayes classifier will yield optimal results. But even near-factorial codes are often good enough. For example, when we use supervised learning to train a deep NN on lots of image data, it will develop pretty good general visual feature detectors. These will usually also work well for different image sets. Learning just a simple extra mapping on top of the deep supervised learning-based code may yield excellent transfer results."

  Juergen Schmidhuber:
  > "A naive Bayes classifier will assume data elements are statistically independent random variables and therefore fail to produce good results. If the data are first encoded in a factorial way, then the naive Bayes classifier will achieve its optimal performance. Thus, factorial code can be seen as an ultimate unsupervised learning approach - predictors and binary feature detectors, each receiving the raw data as an input. For each detector there is a predictor that sees the other detectors and learns to predict the output of its own detector in response to the various input vectors or raw data. But each detector uses a machine learning algorithm to become as unpredictable as possible. The global optimum of this objective function corresponds to a factorial code represented in a distributed fashion across the outputs of the feature detectors."

  Juergen Schmidhuber:
  > "Unsupervised learning is basically nothing but compression."

  Juergen Schmidhuber:
  > "True AI goes beyond imitating teachers. This explains the interest in unsupervised learning. There are two types of unsupervised learning: passive and active. Passive unsupervised learning is simply about detecting regularities in observation streams. This means learning to encode data with fewer computational resources, such as space and time and energy, or data compression through predictive coding, which can be achieved to a certain extent by backpropagation, and can facilitate supervised learning. Active unsupervised learning is more sophisticated than passive unsupervised learning: it is about learning to shape the observation stream through action sequences that help the learning agent figure out how the world works and what can be done in it. Active unsupervised learning explains all kinds of curious and creative behaviour in art and music and science and comedy."

  Juergen Schmidhuber:
  > "The most general type of unsupervised learning comes up in the general reinforcement learning case. Which unsupervised experiments should an agent's reinforcement learning controller C conduct to collect data that quickly improves its predictive world model M, which could be an unsupervised RNN trained on the history of actions and observations so far? The simple formal theory of curiosity and creativity says: Use the learning progress of M (typically compression progress in the Maximum Description Length sense) as the intrinsic reward or fun of C. I believe this general principle of active unsupervised learning explains all kinds of curious and creative behaviour in art and science."

  Nando de Freitas:
  > "Is a scalar reward enough? Hmmm, I don't know. Certainly for most supervised learning - e.g. think ImageNet, there is a single scalar reward. Note that the reward happens at every time step - i.e. it is very informative for ImageNet. Most of what people dub as unsupervised learning can also be cast as reinforcement learning. It is a very general and broad framework, with huge variation depending on whether the reward is rare, whether we have mathematical expressions for the reward function, whether actions are continuous or discrete, etc."


  ----
  #### interesting quotes - loss function and grounding

  Francois Chollet:
  > "Arguably, intelligence is not about learning the latent manifold of some data, it is about taking control of the process that generated it. The human mind has this remarkable property of filtering out the massive fraction of its input that is irrelevant to what it can control. Learning doesn't just mean compressing data; it's mainly about discarding data. This requires a supervision signal. Control is that signal."

  Yoshua Bengio:
  > "Maximum likelihood can be improved upon, it is not necessarily the best objective when learning in complex high-dimensional domains (as arises in unsupervised learning and structured output scenarios)."

  Richard Sutton:
  > "The history of AI is marked by increasing automation. First people hand designed systems to answer hand designed questions. Now they use lots of data to train statistical systems to answer hand designed questions. The next step is to automate asking the questions."

  Ilya Sutskever:
  > "Learning complex cost function for optimizing neural network is likely required for truly sophisticated behavior - cost function can be learned by viewing video."

  François Chollet:
  > "The biggest problem in deep learning is grounding, especially for natural language understanding. Essentially that you cannot reverse-engineer the mental models of a society of agents merely by modeling their communications."

  François Chollet:
  > "All existing NLP is about mapping the internal statistical dependencies of language, missing the point that language is a *communication protocol*. You cannot study language without considering *agents* communicating *about something*. The only reason language even has any statistical dependencies to study is because it's imperfect. A maximally efficient communication protocol would look like random noise, out of context (besides error correction mechanisms). All culture is a form of communication, so "understanding" art requires grounding. Mimicking what humans do isn't enough. You can't understand language without considering it in context: agents communicating about something. An analogy could be trying to understand an economy by looking at statistical structure in stock prices only."

  > "The examples of progress are cases where static networks are trained by an outside device on a (fairly) well-defined problem. The strength of human intelligence derives not just from being able to perform well at a particular task --which can be done already by making specialized neural networks-- but from its ability to generalize and learn never-before-seen tasks. This requires a level of adaptability that the  deep learning paradigm doesn't currently allow for. A threshold that hasn't been reached yet, but is pivotal to creating any kind of human level AI, is the off-loading of centralized learning techniques (such as reinforcement or supervised learning) onto the neural network itself in the form of distributed, local learning rules. The human brain is not a plastic network its connectivity changes and its capacity for learning is driven by these local rules. Even the expression of reinforcement-like learning (through dopamine) and supervised-like learning (through executive function and attention) are still emergent manifestations of lower-level rules. A human-like AI should be able to, without any outside training, pick-up and learn novel tasks simply by interacting with the task environment."


  ----
  #### interesting quotes - bayesian inference and learning

  > "The huge role played by random (or seemingly random due to incomplete available information) events as fundamental forces which dictate our life experience clearly demonstrates the universality and importance of randomness. Just as classical physics is the precise (i.e. mathematical) language used to describe our world at the macro-level, probability is the precise language used to deal with such uncertainty. Now, as human beings without direct access to the underlying forces behind different phenomena, we can only observe/sample events, from which we may try and construct "models" which capture some elements of the underlying probability distributions of interest. Call this problem statistics, ML, data science/mining or whatever you want, but it is simply the extension of the previous scientific paradigm (using differential equations to deterministically explain & predict natural phenomena in a precise mathematical manner) to more complicated problems in which uncertainty is inherent; typically because we cannot measure all relevant quantities (the number of quantities relevant to the phenomena tends to increase with the complexity of the system). For example, if we wish to predict how far a thrown ball travels from the force/angle of the toss, Newtonian physics offers a diff-eq-based formula which most would deem adequate, but given data on a huge number of throws, a learning algorithm could actually offer better performance. This is because it would properly account for the uncertainty in distance-traveled due to spin of the ball, air resistance, and other unmeasured quantities, while simultaneously learning a distance-traveled vs force/angle function which would be similar to the theoretical one obtained from classical mechanics."

  David Barber:
  > "For me Bayesian Reasoning is probability theory extended to treating parameters and models as variables. In this sense, for me the question is essentially the same as `what makes probabiltiy appealing?'. Probability is a (some people would say 'the') logical calculus of uncertainty. There are many aspects of machine learning in which we naturally need to deal with uncertainty. I like the probability approach since it naturally enables one to integrate prior knowledge about a problem into the solution. It does this also in a way that requires one to be explicit about the assumptions being made about the model. People have to be clear about their model specification some people might not agree with that model, but at least they know what the assumptions of the model are."

  Zoubin Ghahramani:
  > "The key ingredient of Bayesian methods is not the prior, it's the idea of averaging over different possibilities."

  > "Bayesian methods have a nice intuitive flow to them. You have a belief (formulated into a prior), you observe data and evaluate it in the context of a likelihood function that you think fits the data generation process well, you have a new updated belief. Nice, elegant, intuitive. I thought this, I saw that, now I think this. Compared to like a maximum likelihood method that will answer the question of what parameters with this likelihood function best fit my data. Which doesn't really answer your actual research question. If I flip a coin one time and get heads, and do a maximum likelihood approach, then it's going to tell me that the type of coin most likely to have given me that result is a double-headed coin. That's probably not the question you had, you probably wanted to know "what's the probability that this comes up heads?" not "what type of coin would give me this result with the highest probability?"."

  > "The frequentist vs. Bayesian debate that raged for decades in statistics before sputtering out in the 90s had more of a philosophical flavor. Starting with Fisher, frequentists argued that unless a priori probabilities were known exactly, they should not be "guessed" or "intuited", and they created many tools that did not require the specification of a prior. Starting with Laplace, Bayesians quantified lack of information by means of a "uninformative" or "objective" uniform prior, using Bayes theorem to update their information as more data came in. Once it became clear that this uniform prior was not invariant under transformation, Bayesian methods fell out of mainstream use. Jeffreys led a Bayesian renaissance with his invariant prior, and Lindley and Savage poked holes in frequentist theory. Statisticians realized that things weren't quite so black and white, and the rise of MCMC methods and computational statistics made Bayesian inference feasible in many, many new domains of science. Nowadays, few statisticians balk at priors, and the two strands have effectively merged (consider the popularity of empirical Bayes methods, which combine the best of both schools). There are still some Bayesians that consider Bayes theorem the be-all-end-all approach to inference, and will criticize model selection and posterior predictive checks on philosophical grounds. However, the vast majority of statisticians will use whatever method is appropriate. The problem is that many scientists aren't yet aware of/trained in Bayesian methods and will use null hypothesis testing and p-values as if they're still the gold standard in statistics."

  > "Bayesian modelling is more elegant, but requires more story telling, which is bad. For instance the recent paper about bayesian program induction requires an entire multilevel story about how strokes are created and how they interact. Just flipping a coin requires a story about a mean and prior distribution over the mean and the hyperparameters describing the prior. It's great but I am a simple man and I just want input output. The other criticism is bayesian cares little for actual computational resources. I just want a simple neural net that runs in linear/polytime, has a simple input-output interpretation, no stories required, to heck if its operation is statistically theoretically unjustified or really even outside of the purview of human understanding to begin with, as long as it vaguely seems to do cool stuff."

  > "An approximate answer to the right problem is worth a good deal more than an exact answer to an approximate problem."

  > "The choice is between A) finding a point estimate of parameters that minimizes some ad hoc cost function that balances the true cost and some other cost designed to reduce overfitting, and Bayes) integrating over a range of models with respect to how well they fit the data. Optimization isn't fundamentally what modeling data is about. Optimization is what you do when you can't integrate. Unfortunately you're left with hyperparameters to tune and you often fall back on weak forms of integration: cross validation and model averaging."

  > "The rule-based system, scientifically speaking, was on the wrong track. They modeled the experts instead of modeling the disease. The problems were that the rules created by the programmers did not combine properly. When you added more rules, you had to undo the old ones. It was a very brittle system. A new thinking came about in the early '80s when we changed from rule-based systems to a Bayesian network. Bayesian networks are probabilistic reasoning systems. An expert will put in his or her perception of the domain. A domain can be a disease, or an oil field—the same target that we had for expert systems. The idea was to model the domain rather than the procedures that were applied to it. In other words, you would put in local chunks of probabilistic knowledge about a disease and its various manifestations and, if you observe some evidence, the computer will take those chunks, activate them when needed and compute for you the revised probabilities warranted by the new evidence. It's an engine for evidence. It is fed a probabilistic description of the domain and, when new evidence arrives, the system just shuffles things around and gives you your revised belief in all the propositions, revised to reflect the new evidence."

  > "If you are able to create a successful generative model, than you now understand more about the underlying science of the problem. You're not just able to fit data well, but you have a model for how the process that generates the data works. If you're just trying to build the best classifier you can with the resources you have, this might not be that useful, but if you're interested in the science of the system that generated this data, this is crucial. What is also crucial is that you can often sacrifice some accuracy to simplify a lot your generative model and obtain really simple mechanisms that tell you a lot about the basic science of the system. Of course it's difficult to see how this transfers to problems like generative deep models for computer vision, where your model is a huge neural network that is not as transparent to read as a simple bayesian model. But I think part of the goal is this: hey, look at this particular filter the network learned - it can generate X or Y objects when I turn it on and off. Now we understand a little more of how we perceive objects X and Y. There's also a feeling that generative models will eventually be more accurate if you can find the "true" generative process that created the data and that nothing could be more accurate than this (after all, this is the true process)."

  John Cook:
  > "The primary way to quantify uncertainty is to use probability. Subject to certain axioms that aim to capture common-sense rules for quantifying uncertainty, probability theory is essentially the only way. (This is Cox’s theorem.) Other methods, such as fuzzy logic, may be useful, though they must violate common sense (at least as defined by Cox’s theorem) under some circumstances. They may be still useful when they provide approximately the results that probability would have provided and at less effort and stay away from edge cases that deviate too far from common sense. There are various kinds of uncertainty, principally epistemic uncertainty (lack of knowledge) and aleatory uncertainty (randomness), and various philosophies for how to apply probability. One advantage to the Bayesian approach is that it handles epistemic and aleatory uncertainty in a unified way."

  Abram Demski:
  > "A Bayesian learning system has a space of possible models of the world, each with a specific weight, the prior probability. The system can converge to the correct model given enough evidence: as observations come in, the weights of different theories get adjusted, so that the theory which is predicting observations best gets the highest scores. These scores don't rise too fast, though, because there will always be very complex models that predict the data perfectly; simpler models have higher prior weight, and we want to find models with a good balance of simplicity and predictive accuracy to have the best chance of correctly predicting the future."

  Yann LeCun:
  > "I think if it were true that P=NP or if we had no limitations on memory and computation, AI would be a piece of cake. We could just brute-force any problem. We could go "full Bayesian" on everything (no need for learning anymore. Everything becomes Bayesian marginalization). But the world is what it is."

  > "Imagine if back in Newton's day, they were analyzing data from physical random variables with deep nets. Sure, they might get great prediction accuracy on how far a ball will go given measurements of its weight, initial force/angle, and some other irrelevant variables, but would this really be the best approach to discover all of the useful laws of physics such as f = ma and the conversion from potential to kinetic energy via the gravitational constant? Probably not, in fact the predictions might be in some sense "too good" incorporating other confounding effects such as air drag and the shape / spin of the ball which obfuscate the desired law. In many settings where an interpretation of what is going on in the data is desired, a clear model is necessary with simple knobs that have clear effects when turned. This may also be a requirement not only for human interpretation, but an also AI system which is able to learn and combine facts about the world (rather than only storing the complex functions which represent the relationships between things as inferred by a deep-net)."

  Daphne Koller:
  > "Uncertainty is unavoidable in real-world applications: we can almost never predict with certainty what will happen in the future, and even in the present and the past, many important aspects of the world are not observed with certainty. Probability theory gives us the basic foundation to model our beliefs about the different possible states of the world, and to update these beliefs as new evidence is obtained. These beliefs can be combined with individual preferences to help guide our actions, and even in selecting which observations to make. While probability theory has existed since the 17th century, our ability to use it effectively on large problems involving many inter-related variables is fairly recent, and is due largely to the development of a framework known as Probabilistic Graphical Models. This framework, which spans methods such as Bayesian networks and Markov random fields, uses ideas from discrete data structures in computer science to efficiently encode and manipulate probability distributions over high-dimensional spaces, often involving hundreds or even many thousands of variables."

  Michael I. Jordan:
  > "Probabilistic graphical models are one way to express structural aspects of joint probability distributions, specifically in terms of conditional independence relationships and other factorizations. That's a useful way to capture some kinds of structure, but there are lots of other structural aspects of joint probability distributions that one might want to capture, and PGMs are not necessarily going to be helpful in general. There is not ever going to be one general tool that is dominant; each tool has its domain in which its appropriate. On the other hand, despite having limitations (a good thing!), there is still lots to explore in PGM land. Note that many of the most widely-used graphical models are chains - the HMM is an example, as is the CRF. But beyond chains there are trees and there is still much to do with trees. There's no reason that one can't allow the nodes in graphical models to represent random sets, or random combinatorial general structures, or general stochastic processes; factorizations can be just as useful in such settings as they are in the classical settings of random vectors. There's still lots to explore there."

  Ferenc Huszar:
  > "My favourite theoretical machine learning papers are ones that interpret heuristic learning algorithms in a probabilistic framework, and uncover that they in fact are doing something profound and meaningful. Being trained as a Bayesian, what I mean by profound typically means statistical inference or fitting statistical models. An example would be the k-means algorithm. K-means intuitively makes sense as an algorithm for clustering. But we only really understand what it does when we make the observation that it actually is a special case of expectation-maximisation in gaussian mixture models. This interpretation as special case of something allows us to understand the expected behaviour of the algorithm better. It will allow us to make predictions about the situations in which it's likely to fail, and to meaningfully extend it to situations it doesn't handle well."

  Ferenc Huszar:
  > "There is no such thing as learning without priors. In the simplest form, the objective function of the optimisation is a prior - you tell the machine that it's goal is to minimise mean squared error for example. The machine solves the optimisation problem (typically) you tell it to solve, and good machine learning is about figuring out what that problem is. Priors are part of that. Secondly, if you think about it, it is actually a tiny portion of machine learning problems where you actually have enough data to get away without engineering better priors or architectures by just using a model which is highly flexible. Today, you can do this in visual, audio, video domain because you can collect and learn from tonnes of examples and particularly because you can use unsupervised or semi-supervised learning to learn natural invariances. An example is chemistry: if you want to predict certain properties of chemicals, it almost doesn't make sense to use data only to make the machine learn what a chemical is, and what the invariances are - doing that would be less accurate and a lot harder than giving it the required context. Un- and semi-supervised learning doesn't make sense because in many cases learning about the natural distribution of chemicals (even if you had a large dataset of this) may be uninformative of the prediction tasks you want to solve."

  Ferenc Huszar:
  > "My belief is that speeding up computation is not fast enough, you do need priors to beat the curse of dimensionality. Think rotational invariance. Yes, you can model that by allowing enough flexibility in a neural netowrk to learn separate representations for all possible rotations of an object, but you're exponentially more efficient if you can somehow 'integrate out' the invariance by designing the architecture/maths cleverly. By modeling invariances correctly, you can make exponential leaps in representational capacity of the network - on top of the exponential growth in computing power that'd kind of a given. I don't think the growth in computing power is fast enough to make progress in machine learning for real-world hard tasks. You need that, combined with exponential leaps on top of that, made possible by building in prior knowledge correcltly."

  > "Many labelling problems are probably better solved by (conditional) generative models. Multi-label problems where the labels are not independent are an obvious example. Even in the single label case, I bet it's probably better to represent uncertainty in the appropriate label via multiple modes in the internal behavior of the model, rather than relegating all entropy to the final prediction."

  Yann LeCun:
  > "I'm a big fan of the conceptual framework of factor graphs as a way to describe learning and inference models. But I think in their simplest/classical form (variable nodes, and factor nodes), factor graphs are insufficient to capture the computational issues. There are factors, say between two variables X and Y, that allow you to easily compute Y from X, but not X from Y, or vice versa. Imagine that X is an image, Y a description of the image, and the factor contains a giant convolutional net that computes a description of the image and measures how well Y matches the computed description. It's easy to infer Y from X, but essentially impossible to infer X from Y. In the real world, where variables are high dimensional and continuous, and where dependencies are complicated, factors are directional."

  > "It's interesting that many summarize Bayesian methods as being about priors; but real power is its focus on integrals and expectations over maximas and modes."

  > "Broadly speaking there are two ways of doing inference in ML. One is integration (which tends to be Bayesian) and the other is optimization (which is usually not). A lot of things that "aren't" Bayesian turn out to be the same algorithm with a different interpretation when you view them from a Bayesian perspective (like ridge regression being a MAP estimate for linear regression with a Gaussian prior). However, there are plenty of things people do that don't fit easily into a Bayesian framework. A few of them that come to mind are random forests, energy based models (in the Yann LeCun sense), and the Apriori and DBSCAN algorithms."

  Yann LeCun:
  > "There is no opposition between "deep" and "Bayesian". Many deep learning methods are Bayesian, and many more can be made Bayesian if you find that useful. David Mackay, myself and a few colleagues at Bell Labs have worked in the 90s on variational Bayesian methods for getting probabilities out of the neural nets (by integrating over a Gaussian approximation of the weight posterior), RBMs are Bayesian, Variational Auto-Encoders are Bayesian, the view of neural nets as factor graphs is Bayesian."

  Nando de Freitas:
  > "Some folks use information theory to learn autoencoders - it's not clear what the value of the prior is in this setting. Some are using Bayesian ideas to obtain confidence intervals - but the bootstrap could have been equally used. Where it becomes interesting is where people use ideas of deep learning to do Bayesian inference. An example of this is Kevin Murphy and colleagues using distillation (aka dark knowledge) for reducing the cost of Bayesian model averaging. I also think deep nets have enough power to implement Bayes rule and sampling rules. I strongly believe that Bayesian updating, Bayesian filtering and other forms of computation can be approximated by the type of networks we use these days. A new way of thinking is in the air."

  Ian Osband:
  > "In sequential decision problems there is an important distinction between risk and uncertainty. We identify risk as inherent stochasticity in a model and uncertainty as the confusion over which model parameters apply. For example, a coin may have a fixed p = 0.5 of heads and so the outcome of any single flip holds some risk; a learning agent may also be uncertain of p. The demarcation between risk and uncertainty is tied to the specific model class, in this case a Bernoulli random variable; with a more detailed model of flip dynamics even the outcome of a coin may not be risky at all. Our distinction is that unlike risk, uncertainty captures the variability of an agent’s posterior belief which can be resolved through statistical analysis of the appropriate data. For a learning agent looking to maximize cumulative utility through time, this distinction represents a crucial dichotomy. Consider the reinforcement learning problem of an agent interacting with its environment while trying to maximize cumulative utility through time. At each timestep, the agent faces a fundamental tradeoff: by exploring uncertain states and actions the agent can learn to improve its future performance, but it may attain better short-run performance by exploiting its existing knowledge. At a high level this effect means uncertain states are more attractive since they can provide important information to the agent going forward. On the other hand, states and action with high risk are actually less attractive for an agent in both exploration and exploitation. For exploitation, any concave utility will naturally penalize risk. For exploration, risk also makes any single observation less informative. Although colloquially similar, risk and uncertainty can require radically different treatment."

  > "While Bayesian inference can capture uncertainty about parameters, it relies on the model being correctly specified. However, in practice, all models are wrong. And in fact, this model mismatch can be often be large enough that we should be more concerned with calibrating our inferences to correct for the mismatch than to produce uncertainty estimates from incorrect assumptions."

  > "While in principle it is nice that we can build models separate from our choice of inference, we often need to combine the two in practice. (The whole naming behind the popular model-inference classes of “variational auto-encoders” and “generative adversarial networks” are one example.) That is, we often choose our model based on what we know enables fast inferences, or we select hyperparameters in our model from data. This goes against the Bayesian paradigm."

  Dustin Tran:
  > "Parameter uncertainty is a key advantage of Bayesian methods. But I think it's overemphasized. What about inference of latent variables? In many cases, including big data, I can see utility for imposing inductive biases via latent vars. Not sure about parameter uncertainty."

  Ferenc Huszar:
  > "There are actually only a handful of use-cases where uncertainty estimates are actually used for anything: active learning, reinforcement learning, control, decision making. I predict that the first mainstream application of Bayesian neural nets will be in active learning for labelling new concepts or object categories. Bayesian neural nets (to some degree) are one way to understand Elastic Weight Consolidation in the Catastrophic Forgetting paper by DeepMind, that's another brilliant application of Bayesian reasoning. So applications are starting to appear, representing uncertainty is just not as absolutely essential in most scenarios as Bayesians like to believe. The other reson is the choice of techniques available: I think a lot of people focus on variational-style inference for Bayesian neural nets, which I personally think is a pretty ugly thing to tackle. Neural network are horribly non-naturally parametrised, parameters are non-identifiable, there are many trivial reparametrisations that capture the same input-output relationship. Approximating posteriors as Gaussians in actual NN parameter space seems like it's not going to be much better than just doing MAP or ML."



---
### interesting papers

  - [theory](#interesting-papers---theory)
  - [automated machine learning](#interesting-papers---automated-machine-learning)
  - [systems](#interesting-papers---systems)


[interesting papers - deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers)  
[interesting papers - reinforcement learning](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers)  
[interesting papers - bayesian inference and learning](https://github.com/brylevkirill/notes/blob/master/Bayesian%20Inference%20and%20Learning.md#interesting-papers)  
[interesting papers - probabilistic programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md#interesting-papers)  


[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md)



---
### interesting papers - theory


#### ["A Theory of the Learnable"](https://people.mpi-inf.mpg.de/~mehlhorn/SeminarEvolvability/ValiantLearnable.pdf) Valiant
>	"Humans appear to be able to learn new concepts without needing to be programmed explicitly in any conventional sense. In this paper we regard learning as the phenomenon of knowledge acquisition in the absence of explicit programming. We give a precise methodology for studying this phenomenon from a computational viewpoint. It consists of choosing an appropriate information gathering mechanism, the learning protocol, and exploring the class of concepts that can be learned using it in a reasonable (polynomial) number of steps. Although inherent algorithmic complexity appears to set serious limits to the range of concepts that can be learned, we show that there are some important nontrivial classes of propositional concepts that can be learned in a realistic sense."

>	"Proof that if you have a finite number of functions, say N, then every training error will be close to every test error once you have more than log N training cases by a small constant factor. Clearly, if every training error is close to its test error, then overfitting is basically impossible (overfitting occurs when the gap between the training and the test error is large)."


#### ["Statistical Modeling: The Two Cultures"](http://projecteuclid.org/euclid.ss/1009213726) Breiman
>	"There are two cultures in the use of statistical modeling to reach conclusions from data. One assumes that the data are generated by a given stochastic data model. The other uses algorithmic models and treats the data mechanism as unknown. The statistical community has been committed to the almost exclusive use of data models. This commitment has led to irrelevant theory, questionable conclusions, and has kept statisticians from working on a large range of interesting current problems. Algorithmic modeling, both in theory and practice, has developed rapidly in fields outside statistics. It can be used both on large complex data sets and as a more accurate and informative alternative to data modeling on smaller data sets. If our goal as a field is to use data to solve problems, then we need to move away from exclusive dependence on data models and adopt a more diverse set of tools."


#### ["A Few Useful Things to Know about Machine Learning"](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) Domingos
>	"Machine learning algorithms can figure out how to perform important tasks by generalizing from examples. This is often feasible and cost-effective where manual programming is not. As more data becomes available, more ambitious problems can be tackled. As a result, machine learning is widely used in computer science and other fields. However, developing successful machine learning applications requires a substantial amount of “black art” that is hard to find in textbooks. This article summarizes twelve key lessons that machine learning researchers and practitioners have learned. These include pitfalls to avoid, important issues to focus on, and answers to common questions."


#### ["Learning with Intelligent Teacher: Similarity Control and Knowledge Transfer"](http://link.springer.com/chapter/10.1007/978-3-319-17091-6_1) Vapnik, Izmailov
>	"This paper introduces an advanced setting of machine learning problem in which an Intelligent Teacher is involved. During training stage, Intelligent Teacher provides Student with information that contains, along with classification of each example, additional privileged information (explanation) of this example. The paper describes two mechanisms that can be used for significantly accelerating the speed of Student’s training: (1) correction of Student’s concepts of similarity between examples, and (2) direct Teacher-Student knowledge transfer."

>	"During last fifty years a strong machine learning theory has been developed. This theory includes: 1. The necessary and sufficient conditions for consistency of learning processes. 2. The bounds on the rate of convergence which in general cannot be improved. 3. The new inductive principle (SRM) which always achieves the smallest risk. 4. The effective algorithms, (such as SVM), that realize consistency property of SRM principle. It looked like general learning theory has been complied: it answered almost all standard questions that is asked in the statistical theory of inference. Meantime, the common observation was that human students require much less examples for training than learning machine. Why? The talk is an attempt to answer this question. The answer is that it is because the human students have an Intelligent Teacher and that Teacher-Student interactions are based not only on the brute force methods of function estimation from observations. Speed of learning also based on Teacher-Student interactions which have additional mechanisms that boost learning process. To learn from smaller number of observations learning machine has to use these mechanisms. In the talk I will introduce a model of learning that includes the so called Intelligent Teacher who during a training session supplies a Student with intelligent (privileged) information in contrast to the classical model where a student is given only outcomes y for events x. Based on additional privileged information x* for event x two mechanisms of Teacher-Student interactions (special and general) are introduced: 1. The Special Mechanism: To control Student's concept of similarity between training examples. and 2. The General Mechanism: To transfer knowledge that can be obtained in space of privileged information to the desired space of decision rules. Both mechanisms can be considered as special forms of capacity control in the universally consistent SRM inductive principle. Privileged information exists for almost any inference problem and can make a big difference in speed of learning processes."

  - `video` <https://video.ias.edu/csdm/2015/0330-VladimirVapnik> (Vapnik)
  - `press` <http://learningtheory.org/learning-has-just-started-an-interview-with-prof-vladimir-vapnik/>


#### ["Compression and Machine Learning: A New Perspective on Feature Space Vectors"](http://www.eecs.tufts.edu/~dsculley/papers/compressionAndVectors.pdf) Sculley, Brodley
>	"The use of compression algorithms in machine learning tasks such as clustering and classification has appeared in a variety of fields, sometimes with the promise of reducing problems of explicit feature selection. The theoretical justification for such methods has been founded on an upper bound on Kolmogorov complexity and an idealized information space. An alternate view shows compression algorithms implicitly map strings into implicit feature space vectors, and compression-based similarity measures compute similarity within these feature spaces. Thus, compression-based methods are not a “parameter free” magic bullet for feature selection and data representation, but are instead concrete similarity measures within defined feature spaces, and are therefore akin to explicit feature vector models used in standard machine learning algorithms. To underscore this point, we find theoretical and empirical connections between traditional machine learning vector models and compression, encouraging cross-fertilization in future work."



---
### interesting papers - automated machine learning


#### ["Design of the 2015 ChaLearn AutoML Challenge"](http://www.causality.inf.ethz.ch/AutoML/automl_ijcnn15.pdf) Guyon et al.
>	"ChaLearn is organizing for IJCNN 2015 an Automatic Machine Learning challenge (AutoML) to solve classification and regression problems from given feature representations, without any human intervention. This is a challenge with code submission: the code submitted can be executed automatically on the challenge servers to train and test learning machines on new datasets. However, there is no obligation to submit code. Half of the prizes can be won by just submitting prediction results. There are six rounds (Prep, Novice, Intermediate, Advanced, Expert, and Master) in which datasets of progressive difficulty are introduced (5 per round). There is no requirement to participate in previous rounds to enter a new round. The rounds alternate AutoML phases in which submitted code is “blind tested” on datasets the participants have never seen before, and Tweakathon phases giving time (~1 month) to the participants to improve their methods by tweaking their code on those datasets. This challenge will push the state-of-the-art in fully automatic machine learning on a wide range of problems taken from real world applications."


#### ["Population Based Training of Neural Networks"](https://arxiv.org/abs/1711.09846) Jaderberg et al.
>	"Neural networks dominate the modern machine learning landscape, but their training and success still suffer from sensitivity to empirical choices of hyperparameters such as model architecture, loss function, and optimisation algorithm. In this work we present Population Based Training, a simple asynchronous optimisation algorithm which effectively utilises a fixed computational budget to jointly optimise a population of models and their hyperparameters to maximise performance. Importantly, PBT discovers a schedule of hyperparameter settings rather than following the generally sub-optimal strategy of trying to find a single fixed set to use for the whole course of training. With just a small modification to a typical distributed hyperparameter training framework, our method allows robust and reliable training of models. We demonstrate the effectiveness of PBT on deep reinforcement learning problems, showing faster wall-clock convergence and higher final performance of agents by optimising over a suite of hyperparameters. In addition, we show the same method can be applied to supervised learning for machine translation, where PBT is used to maximise the BLEU score directly, and also to training of Generative Adversarial Networks to maximise the Inception score of generated images. In all cases PBT results in the automatic discovery of hyperparameter schedules and model selection which results in stable training and better final performance."

>	"Two common tracks for the tuning of hyperparameters exist: parallel search and sequential optimisation, which trade-off concurrently used computational resources with the time required to achieve optimal results. Parallel search performs many parallel optimisation processes (by optimisation process we refer to neural network training runs), each with different hyperparameters, with a view to finding a single best output from one of the optimisation processes – examples of this are grid search and random search. Sequential optimisation performs few optimisation processes in parallel, but does so many times sequentially, to gradually perform hyperparameter optimisation using information obtained from earlier training runs to inform later ones – examples of this are hand tuning and Bayesian optimisation. Sequential optimisation will in general provide the best solutions, but requires multiple sequential training runs, which is often unfeasible for lengthy optimisation processes."

>	"In this work, we present a simple method, Population Based Training which bridges and extends parallel search methods and sequential optimisation methods. Advantageously, our proposal has a wallclock run time that is no greater than that of a single optimisation process, does not require sequential runs, and is also able to use fewer computational resources than naive search methods such as random or grid search. Our approach leverages information sharing across a population of concurrently running optimisation processes, and allows for online propagation/transfer of parameters and hyperparameters between members of the population based on their performance."

>	"Furthermore, unlike most other adaptation schemes, our method is capable of performing online adaptation of hyperparameters – which can be particularly important in problems with highly non-stationary learning dynamics, such as reinforcement learning settings, where the learning problem itself can be highly non-stationary (e.g. dependent on which parts of an environment an agent is currently able to explore). As a consequence, it might be the case that the ideal hyperparameters for such learning problems are themselves highly non-stationary, and should vary in a way that precludes setting their schedule in advance."

  - `post` <https://deepmind.com/blog/population-based-training-neural-networks/>


#### ["Data Programming: Creating Large Training Sets, Quickly"](https://arxiv.org/abs/1605.07723) Ratner, Sa, Wu, Selsam, Re
>	"Large labeled training sets are the critical building blocks of supervised learning methods and are key enablers of deep learning techniques. For some applications, creating labeled training sets is the most time-consuming and expensive part of applying machine learning. We therefore propose a paradigm for the programmatic creation of training sets called data programming in which users provide a set of labeling functions, which are programs that heuristically label large subsets of data points, albeit noisily. By viewing these labeling functions as implicitly describing a generative model for this noise, we show that we can recover the parameters of this model to “denoise” the training set. Then, we show how to modify a discriminative loss function to make it noise-aware. We demonstrate our method over a range of discriminative models including logistic regression and LSTMs. We establish theoretically that we can recover the parameters of these generative models in a handful of settings. Experimentally, on the 2014 TAC-KBP relation extraction challenge, we show that data programming would have obtained a winning score, and also show that applying data programming to an LSTM model leads to a TAC-KBP score almost 6 F1 points over a supervised LSTM baseline (and into second place in the competition). Additionally, in initial user studies we observed that data programming may be an easier way to create machine learning models for non-experts."

>	"In the data programming approach to developing a machine learning system, the developer focuses on writing a set of labeling functions, which create a large but noisy training set. Snorkel then learns a generative model of this noise - learning, essentially, which labeling functions are more accurate than others - and uses this to train a discriminative classifier. At a high level, the idea is that developers can focus on writing labeling functions - which are just (Python) functions that provide a label for some subset of data points - and not think about algorithms or features!"

  - `video` <https://youtube.com/watch?v=iSQHelJ1xxU>
  - `video` <https://youtube.com/watch?v=HmocI2b5YfA> (Re)
  - `post` <http://hazyresearch.github.io/snorkel/blog/weak_supervision.html>
  - `post` <http://hazyresearch.github.io/snorkel/blog/dp_with_tf_blog_post.html>
  - `audio` <https://soundcloud.com/nlp-highlights/28-data-programming-creating-large-training-sets-quickly> (Ratner)
  - `notes` <https://github.com/b12io/reading-group/blob/master/data-programming-snorkel.md>
  - `code` <https://github.com/HazyResearch/snorkel>
  - [Snorkel](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#snorkel) project


#### ["Socratic Learning: Empowering the Generative Model"](http://arxiv.org/abs/1610.08123) Varma et al.
>	"A challenge in training discriminative models like neural networks is obtaining enough labeled training data. Recent approaches have leveraged generative models to denoise weak supervision sources that a discriminative model can learn from. These generative models directly encode the users' background knowledge. Therefore, these models may be incompletely specified and fail to model latent classes in the data. We present Socratic learning to systematically correct such generative model misspecification by utilizing feedback from the discriminative model. We prove that under mild conditions, Socratic learning can recover features from the discriminator that informs the generative model about these latent classes. Experimentally, we show that without any hand-labeled data, the corrected generative model improves discriminative performance by up to 4.47 points and reduces error for an image classification task by 80% compared to a state-of-the-art weak supervision modeling technique."

  - `video` <https://youtube.com/watch?v=0gRNochbK9c>
  - `post` <http://hazyresearch.github.io/snorkel/blog/socratic_learning.html>
  - `code` <https://github.com/HazyResearch/snorkel>
  - [Snorkel](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#snorkel) project


#### ["Snorkel: Rapid Training Data Creation with Weak Supervision"](https://arxiv.org/abs/1711.10160) Ratner, Bach, Ehrenberg, Fries, Wu, Re
>	"Labeling training data is increasingly the largest bottleneck in deploying machine learning systems. We present Snorkel, a first-of-its-kind system that enables users to train stateof-the-art models without hand labeling any training data. Instead, users write labeling functions that express arbitrary heuristics, which can have unknown accuracies and correlations. Snorkel denoises their outputs without access to ground truth by incorporating the first end-to-end implementation of our recently proposed machine learning paradigm, data programming. We present a flexible interface layer for writing labeling functions based on our experience over the past year collaborating with companies, agencies, and research labs. In a user study, subject matter experts build models 2.8× faster and increase predictive performance an average 45.5% versus seven hours of hand labeling. We study the modeling tradeoffs in this new setting and propose an optimizer for automating tradeoff decisions that gives up to 1.8× speedup per pipeline execution. In two collaborations, with the U.S. Department of Veterans Affairs and the U.S. Food and Drug Administration, and on four open-source text and image data sets representative of other deployments, Snorkel provides 132% average improvements to predictive performance over prior heuristic approaches and comes within an average 3.60% of the predictive performance of large hand-curated training sets."

  - `video` <https://youtube.com/watch?v=HmocI2b5YfA> (Re)
  - [Snorkel](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#snorkel) project



---
### interesting papers - systems


#### ["Machine Learning: The High-Interest Credit Card of Technical Debt"](http://research.google.com/pubs/pub43146.html) Sculley et al.
>	"Machine learning offers a fantastically powerful toolkit for building complex systems quickly. This paper argues that it is dangerous to think of these quick wins as coming for free. Using the framework of technical debt, we note that it is remarkably easy to incur massive ongoing maintenance costs at the system level when applying machine learning. The goal of this paper is highlight several machine learning specific risk factors and design patterns to be avoided or refactored where possible. These include boundary erosion, entanglement, hidden feedback loops, undeclared consumers, data dependencies, changes in the external world, and a variety of system-level anti-patterns."

  - `post` <http://john-foreman.com/blog/the-perilous-world-of-machine-learning-for-fun-and-profit-pipeline-jungles-and-hidden-feedback-loops>


#### ["TensorFlow: A system for large-scale machine learning"](https://arxiv.org/abs/1605.08695) Abadi et al.
>	"TensorFlow is a machine learning system that operates at large scale and in heterogeneous environments. TensorFlow uses dataflow graphs to represent computation, shared state, and the operations that mutate that state. It maps the nodes of a dataflow graph across many machines in a cluster, and within a machine across multiple computational devices, including multicore CPUs, general-purpose GPUs, and custom designed ASICs known as Tensor Processing Units (TPUs). This architecture gives flexibility to the application developer: whereas in previous “parameter server” designs the management of shared state is built into the system, TensorFlow enables developers to experiment with novel optimizations and training algorithms. TensorFlow supports a variety of applications, with particularly strong support for training and inference on deep neural networks. Several Google services use TensorFlow in production, we have released it as an open-source project, and it has become widely used for machine learning research. In this paper, we describe the TensorFlow dataflow model in contrast to existing systems, and demonstrate the compelling performance that TensorFlow achieves for several real-world applications."


#### ["A Reliable Effective Terascale Linear Learning System"](http://arxiv.org/abs/1110.4198) Agarwal, Chapelle, Dudik, Langford
  `Vowpal Wabbit`
>	"We present a system and a set of techniques for learning linear predictors with convex losses on terascale data sets, with trillions of features, 1 billions of training examples and millions of parameters in an hour using a cluster of 1000 machines. Individually none of the component techniques are new, but the careful synthesis required to obtain an efficient implementation is. The result is, up to our knowledge, the most scalable and efficient linear learning system reported in the literature. We describe and thoroughly evaluate the components of the system, showing the importance of the various design choices."

>	"
>	 - Online by default  
>	 - Hashing, raw text is fine  
>	 - Most scalable public algorithm  
>	 - Reduction to simple problems  
>	 - Causation instead of correlation  
>	 - Learn to control based on feedback"  

  - <https://github.com/JohnLangford/vowpal_wabbit/wiki>
  - `video` <http://youtube.com/watch?v=wwlKkFhEhxE> (Langford)
  - `paper` ["Bring The Noise: Embracing Randomness Is the Key to Scaling Up Machine Learning Algorithms"](http://online.liebertpub.com/doi/pdf/10.1089/big.2013.0010) by Brian Dalessandro


#### ["Making Contextual Decisions with Low Technical Debt"](http://arxiv.org/abs/1606.03966) Agarwal et al.
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#making-contextual-decisions-with-low-technical-debt-agarwal-et-al>


#### ["Communication-Efficient Learning of Deep Networks from Decentralized Data"](https://arxiv.org/abs/1602.05629) McMahan, Moore, Ramage, Hampson, Arcas
>	"Modern mobile devices have access to a wealth of data suitable for learning models, which in turn can greatly improve the user experience on the device. For example, language models can improve speech recognition and text entry, and image models can automatically select good photos. However, this rich data is often privacy sensitive, large in quantity, or both, which may preclude logging to the data center and training there using conventional approaches. We advocate an alternative that leaves the training data distributed on the mobile devices, and learns a shared model by aggregating locally-computed updates. We term this decentralized approach Federated Learning. We present a practical method for the federated learning of deep networks based on iterative model averaging, and conduct an extensive empirical evaluation, considering five different model architectures and four datasets. These experiments demonstrate the approach is robust to the unbalanced and non-IID data distributions that are a defining characteristic of this setting. Communication costs are the principal constraint, and we show a reduction in required communication rounds by 10-100x as compared to synchronized stochastic gradient descent."

  - `post` <https://research.googleblog.com/2017/04/federated-learning-collaborative.html>
