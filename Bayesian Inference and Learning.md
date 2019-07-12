  Statistical learning frames models as distributions over data and latent variables, allowing models to address a broad array of downstream tasks, and underlying methodology of latent variable models is typically Bayesian.  
  A central problem involves modeling complex data sets using highly flexible families of probability distributions in which learning, sampling, inference, and evaluation are still analytically or computationally tractable.  


  * [**introduction**](#introduction)
  * [**overview**](#overview)
    - [**bayesian deep learning**](#overview)
    - [**bayesian reinforcement learning**](#overview)
  * [**theory**](#theory)
  * [**models**](#models)
    - [**graphical models**](#graphical-models)
    - [**non-parametric models**](#non-parametric-models)
    - [**probabilistic programming**](#probabilistic-programming)
  * [**inference and learning**](#inference-and-learning)
    - [**expectation maximization**](#expectation-maximization)
    - [**variational inference**](#variational-inference)
    - [**monte carlo inference**](#monte-carlo-inference)
    - [**likelihood-free inference**](#likelihood-free-inference)
    - [**belief propagation**](#belief-propagation)
  * [**interesting papers**](#interesting-papers)



---
### introduction

  "Bayesian inference is application of Bayes' theorem to update probability of hypothesis as more evidence or information becomes available."

  "Probability is the representation of uncertain or partial knowledge about the truth of statements."

  "Logical inference is about what is certain to be true. Statistical inference is about what is likely to be true."

  "How do you extend classical logic to reason with uncertain propositions? Suppose we agree to represent degrees of plausibility with real numbers, larger numbers indicating greater plausibility. If we also agree to a few axioms to quantify what we mean by consistency and common sense, there is a unique and inevitable system for plausible reasoning that satisfies the axioms, which is probability theory. And this has been proven over 60 years ago. The important implication is that all other systems of plausible reasoning - fuzzy logic, neural networks, artificial intelligence, etc. - must either lead to the same conclusions as probability theory, or violate one of the axioms used to derive probability theory."

  "In Bayesian approach, probability is used not only to describe “physical” randomness, such as errors in labeling, but also uncertainty regarding the true values of the parameters. These prior and posterior probabilities represent degrees of belief, before and after seeing the data. The Bayesian approach takes modeling seriously. A Bayesian model includes a suitable prior distribution for model parameters. If the model/prior are chosen without regard for the actual situation, there is no justification for believing the results of Bayesian inference. The model and prior are chosen based on our knowledge of the problem. These choices are not, in theory, affected by the amount of data collected, or by the question we are interested in answering. We do not, for example, restrict the complexity of the model just because we have only a small amount of data. Pragmatic compromises are inevitable in practice - no model and prior perfectly express to our knowledge of the situation. The Bayesian approach relies on reducing such flaws to a level where we think they won’t seriously affect the results."

----

  [introduction](http://mlg.eng.cam.ac.uk/zoubin/bayesian.html) by Zoubin Ghahramani

  ["Why probability models?"](http://johndcook.com/blog/probability-modeling/) by John Cook  
  ["What is randomness? What is a random variable?"](http://johndcook.com/blog/2012/04/19/random-is-as-random-does/) by John Cook  
  ["Plausible reasoning"](http://johndcook.com/blog/2008/03/19/plausible-reasoning/) by John Cook  

  ["Embracing Uncertainty - The Role of Probabilities"](http://blogs.technet.com/b/machinelearning/archive/2014/10/22/embracing-uncertainty-the-role-of-probabilities.aspx) by Chris Bishop  
  ["Embracing Uncertainty - Probabilistic Inference"](http://blogs.technet.com/b/machinelearning/archive/2014/10/30/embracing-uncertainty-probabilistic-inference.aspx) by Chris Bishop  

  ["Where Priors Come From"](http://zinkov.com/posts/2015-06-09-where-priors-come-from/) by Rob Zinkov

----

  ["Probability as Extended Logic"](http://bjlkeng.github.io/posts/probability-the-logic-of-science/) by Brian Keng

  [definitions](https://youtube.com/watch?v=Ihud7yG2iKs) of probability by Andrey Kolmogorov and E.T. Jaynes `video` `in russian`  
  [definition](https://youtube.com/watch?v=X0Lo5IWLjko) of randomness in algorithmic information theory `video` `in russian`  



---
### overview

  ["The Three Faces of Bayes"](https://slackprop.wordpress.com/2016/08/28/the-three-faces-of-bayes/) by Burr Settles  
  ["Bayesian Machine Learning"](http://fastml.com/bayesian-machine-learning/) by Zygmunt Zajac  

  ["Bayesian Methods for Machine Learning"](http://www.cs.toronto.edu/~radford/ftp/bayes-tut.pdf) by Radford Neal  

  [overview](http://metacademy.org/roadmaps/rgrosse/bayesian_machine_learning) by Roger Grosse  
  [overview](http://frnsys.com/ai_notes/machine_learning/bayesian_learning.html) by Francis Tseng  

  ["Probabilistic Learning and Reasoning"](https://www.cs.toronto.edu/~duvenaud/courses/csc412/index.html) course by David Duvenaud

----

  ["Probabilistic Machine Learning - Foundations and Frontiers"](https://youtube.com/watch?v=3foXO9noKj8) by Zoubin Ghahramani `video`  
  ["Probabilistic Modelling and Bayesian Inference"](https://youtube.com/watch?v=kjo9Y_Vrgn4) by Zoubin Ghahramani `video`  
  ["Introduction to Bayesian Inference"](http://videolectures.net/mlss09uk_bishop_ibi/) by Chris Bishop `video`  

  [Coursera](https://coursera.org/learn/bayesian-methods-in-machine-learning) course by Daniil Polykovskiy and Alexander Novikov `video`

  ["Information Theory, Pattern Recognition and Neural Networks"](http://videolectures.net/course_information_theory_pattern_recognition/) course by David MacKay `video`

----

  ["Latent Variable Models"](https://youtube.com/watch?v=7yLOF07Mv5I) by Dmitry Vetrov `video` `in russian`
	([slides](https://drive.google.com/open?id=0BwU8otKU0BqQSVoyN295Y0doRTg) `in english`)  
  ["Scalable Bayesian Methods"](https://youtube.com/watch?v=if9bTlZOiO8) by Dmitry Vetrov `video` `in russian`
	([slides](https://drive.google.com/open?id=0BwU8otKU0BqQOGdzYTdMem1UTEk) `in english`)  

  [course](https://youtube.com/playlist?list=PLEqoHzpnmTfCiJpMPccTWXD9DB4ERQkyw) by Dmitry Vetrov `video` `in russian`  
  [course](https://lektorium.tv/lecture/30977) by Sergey Nikolenko `video` `in russian`  

----

  [**bayesian deep learning**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#bayesian-deep-learning)

  [**bayesian reinforcement learning**](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#bayesian-reinforcement-learning)

  [**Solomonoff induction**](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#solomonoff-induction)  *(bayesian optimal prediction)*  
  [**AIXI**](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#universal-artificial-intelligence---aixi)  *(bayesian optimal decision making)*  

----

  misconceptions:  
  - *Bayesian methods make assumptions and others don't.*  
	All methods make assumptions otherwise it would be impossible to learn. Bayesian methods are transparent in assumptions while others are opaque.  
  - *If you don't have the right prior you won't do well.*  
	No such thing as the right prior. Choose vague prior such as nonparametrics if in doubt.  
  - *As dataset grows infinitely, Bayes converges to maximum likelihood, prior washes out, integration becomes unnecessary.*  
	This assumes learning simple model from large set of i.i.d. data points, while big data is more like large set of little data sets with structure.  
  - *Bayesian models are generative.*  
	Also can be used for discriminative learning such as in gaussian process classification.  
  - *Bayesian methods don't have theoretical guarantees.*  
	Frequentist style generalization error bounds such as PAC-Bayes can be applied, it is often possible to prove convergence, consistency and rates.  

  advantages:  
  - learn from limited, noisy, missing data  
  - deal with small sample size  
  - marginalize over latent variables  
  - compute error bars  
  - establish causal relationships  
  - produce explanations for decisions  
  - integrate knowledge  

  disadvantages:  
  - learning can be wrong if model is wrong  
  - not all prior knowledge can be encoded as joint distributions  
  - simple analytic forms of conditional distributions  

  applications:  
  - data-efficient learning  
  - exploration  
  - relational learning  
  - semiparametric learning  
  - hypothesis formation  
  - causal reasoning  
  - macro-actions and planning  
  - visual concept learning  
  - world simulation  
  - scene understanding  

  research directions:  
  - probabilistic programming languages  
  - bayesian optimization  
  - rational allocation of computational resources  
  - efficient data compression  
  - automating model discovery and experimental design  



---
### theory

#### books

  ["Pattern Recognition and Machine Learning"](https://dropbox.com/s/pwtiuqs27lblvjz/Bishop%20-%20Pattern%20Recognition%20and%20Machine%20Learning.pdf) by Chris Bishop  
  ["Machine Learning - A Probabilistic Perspective"](https://dropbox.com/s/jdly520i5irx1h6/Murphy%20-%20Machine%20Learning%20-%20A%20Probabilistic%20Perspective.pdf) by Kevin Murphy  
  ["Information Theory, Inference and Learning Algorithms"](http://www.inference.phy.cam.ac.uk/mackay/itila/book.html) by David MacKay  
  ["Bayesian Reasoning and Machine Learning"](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.Online) by David Barber  
  ["Probabilistic Graphical Models: Principles and Techniques"](https://dropbox.com/s/cc3mafx3wp0ad1t/Daphne%20Koller%20and%20Nir%20Friedman%20-%20Probabilistic%20Graphical%20Models%20-%20Principles%20and%20Techniques.pdf) by Daphne Koller and Nir Friedman  
  ["Graphical Models, Exponential Families, and Variational Inference"](https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf) by Martin Wainwright and Michael Jordan  
  ["Elements of Causal Inference"](https://mitpress.mit.edu/books/elements-causal-inference) by Jonas Peters, Dominik Janzing, Bernhard Scholkopf  
  ["Probability Theory: The Logic of Science"](https://dropbox.com/s/pt5tpm9i5wofbl5/Jaynes%20-%20Probability%20Theory%20-%20The%20Logic%20of%20Science.pdf) by E.T. Jaynes  

----

  **contrast with the "learning machine" approach**

  "One view of machine learning pictures a "learning machine", which takes in inputs for a training/test case at one end, and outputs a prediction at the other. The machine has various "knobs", whose settings change how a prediction is made from the inputs. Learning is seen as a procedure for twiddling the knobs in the hopes of making better predictions on test cases - for instance, we might use the knob settings that minimize prediction error on training cases. This approach differs profoundly from the Bayesian view:  
  - The choice of learning machine is essentially arbitrary - unlike a model, the machine has no meaningful semantics, that we could compare with our beliefs.  
  - The “knobs” on the machine do not correspond to the parameters of a Bayesian model - Bayesian predictions, found by averaging, usually cannot be reproduced using any single value of the model parameters."  


  **contrast with "learning theory"**

  "An aim of “learning theory” is to prove that certain learning machines “generalize” well. One can sometimes prove that if you adjust the knobs on the learning machine to minimize training error, then apply it to test cases, the training error rates and test error rates are unlikely to be far apart:  P(|test error rate − training error rate| > ε) < δ  , where δ and ε have certain small values, which depend on the training set size. Such a result would be of little interest, if it weren’t usually interpreted as guaranteeing that, for instance:  P(|test error rate − 0.02| > ε | training error rate = 0.02) < δ.  
  This is a fallacy, however - no valid probabilistic statement about test error rates conditional on the observed error rate on training cases is possible without assuming some prior distribution over possible situations. This fallacy is analogous to the common misinterpretation of a frequentist p-value as the probability that the null hypothesis is true, or of a frequentist confidence interval as an interval that likely contains the true value."  


  **what about "bias" and "variance"**

  "Another approach to analysing learning methods (especially for predicting real-valued quantities) looks at the following two indicators of how well a method predicts some quantity:  
  - Bias: how much predictions depart from the truth on average.  
  - Variance: the average squared difference of predictions from their average.  

  The average squared error for the method can be decomposed as the sum of the squared bias and the variance. This leads to a strategy: choose a method that minimizes this sum, possibly trading off increased bias for reduced variance, or vice versa, by adjusting complexity, or introducing some form of “regularization”.  
  There are two problems with this strategy:  
  - The bias and variance depend on the true situation, which is unknown.  
  - There is no reason to think that trying nevertheless to minimize squared bias plus variance produces a unique answer.  

  Assessments of bias and variance play no role in the Bayesian approach."  


  **limitations of bayesian approach**

  - problems requiring specific priors in vague situations

    "An example: We have a sample of points that we know come from a convex polyhedron, whose volume we wish to estimate. A Bayesian method will need a prior over possible polyhedra - which could be done, but probably requires a lot of thought. But a simple non-Bayesian estimate based on cross validation is (usually) available."

  - problems where the likelihood has an intractable normalizing constant

    "Boltzmann machines are an example - even maximum likelihood is hard, and Bayesian inference seems out of the question at the moment."

  - problems with complex, unknown error distributions

    "We can try to model the error, but it may be difficult. A bad model may lead to “overfitting” data where the model thinks the error is less than it is. A cross-validation approach to regularization can sometimes work better in such situations."

  *(Radford Neal)*



---
### models

  - [**graphical models**](#graphical-models)
  - [**non-parametric models**](#non-parametric-models)
  - [**probabilistic programming**](#probabilistic-programming)



---
### graphical models

  The biggest advantage of graphical models is relatively simple way to distinguish conditionally independent variables, which simplify further analysis and allows to significantly lower number of factors given variable depends on.

  overview by Roger Grosse
	([first part](https://metacademy.org/roadmaps/rgrosse/bayesian_machine_learning#models),
	[second part](https://metacademy.org/roadmaps/rgrosse/bayesian_machine_learning#advanced-topics))

  [notes](https://ermongroup.github.io/cs228-notes/) by Volodymyr Kuleshov et al.  
  ["Graphical Models"](http://www.deeplearningbook.org/contents/graphical_models.html) chapter of "Deep Learning" book by Goodfellow, Bengio, Courville  

  [overview](http://youtube.com/watch?v=ju1Grt2hdko) by Chris Bishop `video`  
  [overview](http://youtube.com/watch?v=W6XyXeB3Cko) by Alex Smola `video`  
  [overview](http://youtube.com/watch?v=D_dNxrIazco) by Dmitry Vetrov `video` `in russian`  

  ["Probabilistic Graphical Models"](https://coursera.org/course/pgm) course by Daphne Koller ([videos](https://youtube.com/playlist?list=PL50E6E80E8525B59C))

----

  [**variational autoencoder**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#generative-models---variational-autoencoder)  
  [**bayesian neural network**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#bayesian-deep-learning)  



---
### non-parametric models

  The basic point of non-parametric models is that they provide a prior distribution on real-valued functions. This lets you do regression as Bayesian inference: given observed data, Bayes rule turns your prior on functions into a posterior distribution. Having a posterior distribution on functions, rather than just a single learned function, means you can reason about uncertainty of your predictions at any set of points.

  [overview](https://metacademy.org/roadmaps/rgrosse/bayesian_machine_learning#bayesian-nonparametrics) by Roger Grosse

  ["Bayesian Nonparametric Models"](http://www.stats.ox.ac.uk/~teh/research/npbayes/OrbTeh2010a.pdf) by Peter Orbanz and Yee Whye Teh `paper`

  ["Gaussian Processes for Machine Learning"](http://gaussianprocess.org/gpml) by Carl Rasmussen and Christopher Williams `book`

----

  [overview](https://youtu.be/H7AMB0oo__4?t=21m51s) by Zoubin Ghahramani `video`

  [Gaussian Processes](https://youtube.com/watch?v=4vGiHC35j9s) by Nando de Freitas `video`  
  [Gaussian Processes](https://youtube.com/watch?v=50Vgw11qn0o) by Philipp Hennig `video`  
  [Gaussian Processes](https://youtube.com/watch?v=S9RbSCpy_pg) by Neil Lawrence `video`  

  ["Gaussian Processes and Bayesian Optimization"](https://youtube.com/watch?v=PgJMLpIfIc8) by Evgeny Burnaev `video` `in russian`  
  ["Scalable and Deep Gaussian Processes"](https://youtube.com/watch?v=NqOBWLUYBm4) by Dmitry Kropotov `video` `in russian`  

  ["Nonparametric Bayesian Methods: Models, Algorithms, and Applications"](https://youtube.com/watch?v=I7bgrZjoRhM) course by Tamara Broderick and Michael I. Jordan `video`  
  ["Bayesian Nonparametrics"](https://youtube.com/watch?v=kKZkNUvsJ4M) course by Tamara Broderick `video`  
  ["Bayesian Nonparametrics"](https://youtube.com/watch?v=FUL1DcjOjwo) course by Tamara Broderick `video`  

----

  "Many real phenomena are of essentially unlimited complexity. Suppose we model consumer behaviour by categorizing consumers into various “types” (mixture components). There is no reason to think that there are only (say) five types of consumer. Surely there are an unlimited number of types, though some may be rare. Suppose we model the growth rate of trees as a function of climate, soil type, genetic characteristics, disease suppression measures taken, etc. There is no reason to think any simple functional form (eg, linear, low-order polynomial) will capture the many ways these factors interact to determine tree growth. How can we build a model that accommodates such complexity?

  One approach:  
  - Define models that can have any finite amount of complexity (e.g., a finite number of mixture components, or of hidden units).  
  - Define priors for these models that make sense.  
  - See if the limit as the complexity goes to infinity is sensible.  

  If the limit makes sense, we can use a model that is as large as we can handle computationally. And sometimes, we can figure out how to actually implement the infinite model on a finite computer."  

  *(Radford Neal)*



---
### probabilistic programming

  [**probabilistic programming**](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md)



---
### inference and learning

  [overview](https://metacademy.org/roadmaps/rgrosse/bayesian_machine_learning#basic-inference-algorithms) by Roger Grosse

----

  "Imagine you have a distribution p(x) and you want to compute the integral ∫p(x)F(x)dx for some function F(x) of interest. We call the computation of this integral as inference. Examples include Bayesian inference where now p(x) is some posterior distribution and F(x) is the likelihood function of x on unseen data. Or if p(x) is unnormalised, taking F(x)=1 would return the integral as the normalising constant (or partition function) of p. Unfortunately for many complicated models we are fancy on now (say neural networks) this integral is intractable, and here intractability means you can't compute the exact value of the integral due to computational constraints (say running time, memory usage, precision, etc). So instead we use approximate inference to approximate that integral. There are mainly two ways to do approximate inference: directly approximating the integral you want, or, finding an accurate approximation q to the target distribution p and using it for integration later. The first approach is mainly dominated by Monte Carlo methods while the second one is dominated by variational inference methods."

  ["Topics in Approximate Inference"](http://yingzhenli.net/home/en/?page_id=895) by Yingzhen Li

----

  applications:
  - prediction:  p(x(t+1,...,∞)|x(-∞,...,t))  *(inference)*
  - parameter estimation:  p(θ|x(0,...,N))  *(learning as inference)*
  - planning:  J = Ep[∫dtC(xt)|x0,u]  *([policy evaluation / optimization](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#bayesian-reinforcement-learning))*
  - experiment design:  EIG = DKL[p(f(x(t,...,∞))|u);p(f(x(-∞,...,t)))]  *(expected information gain)*
  - hypothesis testing:  p(f(x(-∞,...,t))|H0) / p(f(x(-∞,...,t))|H1)  *(bayes factor)*

----

  - [**expectation maximization**](#expectation-maximization)
  - [**variational inference**](#variational-inference)
  - [**monte carlo methods**](#monte-carlo-methods)
  - [**likelihood-free inference**](#likelihood-free-inference)
  - [**belief propagation**](#belief-propagation)



---
### expectation maximization

  EM algorithm estimates parameters of model iteratively, starting from some initial guess. Each iteration consists of Expectation step, which finds distribution for unobserved variables, given known values for observed variables and current estimate of parameters, and Maximization step, which re-estimates the parameters with maximum likelihood, under assumption that distribution found on E step is correct.

  EM algorithm can be interpreted as coordinate ascent procedure which optimizes variational lower bound on the likelihood function. This connects it with variational inference algorithms and justifies various generalizations and approximations to the algorithm.

----

  [overview](https://metacademy.org/graphs/concepts/expectation_maximization)

  ["EM Algorithm and Variants: an Informal Tutorial"](http://arxiv.org/abs/1105.1476) by Alexis Roche `paper`

----

  [overview](https://youtube.com/watch?v=cn4sI39uD_Q) by Dmitry Vetrov `video`  
  [overview](https://youtube.com/watch?v=CYeCeQ-pULE) by Ekaterina Lobacheva `video`  
  overview by Dmitry Vetrov ([part 1](http://youtu.be/U0LylVL-zJM?t=35m59s), [part 2](http://youtube.com/watch?v=CqjqTbUgbOo)) `video` `in russian`  
  [overview](https://youtube.com/watch?v=vPRphQh1eGQ&t=32m54s) by Konstantin Vorontsov `video` `in russian`  

----

  [EM and VI](https://youtu.be/yzNbaAPKXA8?t=19m45s) by Zoubin Ghahramani `video`  ([**variational inference**](#variational-inference))  
  ["VAE = EM"](https://machinethoughts.wordpress.com/2017/10/02/vae-em/) by David McAllester  ([**variational auto-encoder**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#generative-models---variational-autoencoder))  

----

  "Perhaps the most salient feature of EM as an optimization algorithm is that it works iteratively by maximizing successive local approximations of the likelihood function. Therefore, each iteration consists of two steps: one that performs the approximation (the E-step) and one that maximizes it (the M-step). What essentially defines an EM algorithm is the philosophy underlying the local approximation scheme - which, for instance, doesn't rely on differential calculus.  

  The key idea underlying EM is to introduce a latent variable Z whose PDF depends on θ with the property that maximizing p(z|θ) is easy or, say, easier than maximizing p(y|θ). Loosely speaking, we somewhat enhance the incomplete data by guessing some useful additional information. Technically, Z can be any variable such that θ -> Z -> Y is a Markov chain, i.e. we assume that p(y|z,θ) is independent from θ: p(z,y|θ) = p(z|θ)p(y|z).  

  Original EM formulation stems from a very simple variational argument. Under almost no assumption regarding the complete variable Z, except its PDF doesn't vanish to zero, we can bound the variation of the log-likelihood function L(θ) = log p(y|θ) as follows:  
  L(θ) - L(θ') = log (p(y|θ)/p(y|θ')) = log ∫ (p(z,y|θ)/p(y|θ'))dz = log ∫ (p(z,y|θ)/p(z,y|θ'))p(z|y,θ')dz = [step 1] log ∫ (p(z|θ)/p(z|θ'))p(z|y,θ')dz >= [step 2] ∫ log(p(z|θ)/p(z|θ')p(z|y,θ')dz = Q(θ,θ')  

  Step 1 results from the fact that p(y|z,θ) is indepedent from θ because of p(z,y|θ) = p(z|θ)p(y|z).  
  Step 2 follows from Jensen's inequality and well-known concavity property of the logarithm function.  

  Therefore Q(θ,θ') is an auxiliary function for the log-likelihood, in the sense that: (i) the likelihood variation from θ' to θ is always greater than Q(θ,θ'), and (ii) we have Q(θ,θ') = 0.  
  Hence, starting from an initial guess θ', we are guaranteed to increase the likelihood value if we can find a θ such that Q(θ,θ') > 0. Iterating such a process defines an EM algorithm.  

  There is no general convergence theorem for EM, but thanks to the above mentioned monotonicity property, convergence results may be proved under mild regularity conditions. Typically, convergence towards a non-global likelihood maximizer, or a saddle point, is a worst-case scenario. Still, the only trick behind EM is to exploit the concavity of the logarithm function."

----

  "It's easier to understand EM through lens of variational inference.  
  We want to maximize log p(x|θ) = log ∫ p(x,z|θ)dz but don't know z.  
  log ∫ p(x,z|θ)dz = log ∫ (q(z)/q(z))p(x,z|θ)dz ≥ <Jensen's inequality> ∫ q(z)log(p(x,z|θ)/q(z))dz = E[log p(x,z|θ)] + H[q(z)],  
  where q(z) is variational distribution and H[q(z)] is independent of θ and can be dropped.  
  When EM is derived, q(z) is almost always set as q(z) = p(z|x,θ) but this is not necessary.  
  The above will be true for any distribution q. Different choices will alter tightness of bound.  
  Q = E[log p(x,z|θ)] or Q = E[log p(x|θ)] in case q(z) = p(z|x,θ) because of p(x,z|θ) = p(x|θ)p(z|x,θ)  
  EM is usually written as first computing Q then optimizing wrt θ but can be written in single step: argmax_θ E[log p(x,z|θ)]."



---
### variational inference

  Variational inference is an umbrella term for algorithms which cast Bayesian inference as optimization.

  Variational inference approximates Bayesian posterior distribution over a set of latent variables W by optimising the evidence lower bound (ELBO) L(q) = Eq(W)[log p(Y|X,W)] − DKL(q(W)||p(W)) with respect to approximate posterior q(W).

----

  [overview](https://metacademy.org/roadmaps/rgrosse/bayesian_machine_learning#variational-inference) by Roger Grosse

  ["Variational Inference in 5 Minutes"](http://davmre.github.io/inference/2015/11/13/elbo-in-5min/) by Dave Moore  
  ["General Purpose Variational Inference"](http://davmre.github.io/inference/2015/11/13/general_purpose_variational_inference/) by Dave Moore  

  ["Neural Variational Inference"](http://artem.sobolev.name/tags/modern%20variational%20inference%20series.html) by Artem Sobolev:  
  - ["Classical Theory"](http://artem.sobolev.name/posts/2016-07-01-neural-variational-inference-classical-theory.html)  
  - ["Scaling Up"](http://artem.sobolev.name/posts/2016-07-04-neural-variational-inference-stochastic-variational-inference.html)  
  - ["Blackbox Mode"](http://artem.sobolev.name/posts/2016-07-05-neural-variational-inference-blackbox.html)  
  - ["Variational Autoencoders and Helmholtz machines"](http://artem.sobolev.name/posts/2016-07-11-neural-variational-inference-variational-autoencoders-and-Helmholtz-machines.html)  
  - ["Importance Weighted Autoencoders"](http://artem.sobolev.name/posts/2016-07-14-neural-variational-importance-weighted-autoencoders.html)  

  ["Variational Inference: A Review for Statisticians"](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#variational-inference-a-review-for-statisticians-blei-kucukelbir-mcauliffe) by Blei et al. `paper` `summary`  
  ["Advances in Variational Inference"](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#advances-in-variational-inference-zhang-butepage-kjellstrom-mandt) by Zhang et al. `paper` `summary`  
  ["Variational Inference"](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf) by David Blei `paper`  
  ["An Introduction to Variational Methods for Graphical Model"](https://www.cs.berkeley.edu/~jordan/papers/variational-intro.pdf) by Jordan et al. `paper`  

----

  "Variational Inference: Foundations and Innovations" by David Blei ([1](https://youtube.com/watch?v=DaqNNLidswA), [2](https://youtube.com/watch?v=Wd7R_YX4PcQ)) `video`  
  ["Variational Inference: Foundations and Innovations"](https://youtube.com/watch?v=Dv86zdWjJKQ) by David Blei `video`  
  ["Variational Inference: Foundations and Modern Methods"](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Variational-Inference-Foundations-and-Modern-Methods)
	tutorial by David Blei, Rajesh Ranganath, Shakir Mohamed `video` ([slides](http://www.cs.columbia.edu/~blei/talks/2016_NIPS_VI_tutorial.pdf))  

  [overview](https://youtube.com/watch?v=d0LZE6Drqyc) by Dmitry Kropotov `video`  
  [overview](https://youtu.be/tqGEX_Ucu04?t=48m42s) of alternatives to variational inference by Dmitry Molchanov `video` `in russian`  

----

  applications:
  - [**variational autoencoder**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#generative-models---variational-autoencoder)  *(approximating posterior over latent variables given a datapoint)*
  - [**bayesian neural network**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#bayesian-deep-learning)  *(approximating posterior over model parameters given a dataset)*

----

  At a high level, probabilistic graphical models have two kinds of variables: visible and hidden. Visible variables are the ones we observe, hidden variables are ones that we use as part of our model to explain relationships between visible variables or describe hidden causes behind the observations. These hidden variables may not correspond to observable quantities. For example, when modelling faces, observable variables might be raw pixel intensities in an image, hidden variables might be things that describe things like lighting, eye colour, face orientation, skin tone. Hidden variables, and the relationships between variables correspond to our model of how the world might work.

  Generally, we want to do be able to do two things with such models:  
  - *inference* - determining the value (or conditional probability distribution) of hidden variables, given the observations.  
	"Given a particular image with its pixel values, what are probable values of face orientation?"  
  - *learning* - adjusting parameters of the model so it fits our dataset better.  
	"How should we find the parameters that are most consistent with our observations?”  
	This is particularly important in the deep learning flavour of probabilistic models where the relationship between hidden variables might be described by a deep neural network with several layers and millions of parameters.  

  To solve these two problems, we often need the ability to marginalise, to calculate marginal probability distributions of subsets of the variables. In particular, we often want to calculate (and maximise) the marginal likelihood, or model evidence, which is the probability of observable variables, but with the hidden variables averaged out. Equivalently, one might phrase the learning and inference problems as evaluating normalisations constants or partition functions. Evaluating these quantities generally involves intractable integrals or enumerating and summing over exponentially many possibilities, so exact inference and learning in most models are practically impossible.  
  One approach is to try to approximate that integrals by sampling, but often we're faced with a distribution we can't even easily obtain unbiased samples from, and we have to do use Markov chains, which may take a long time to visit all the places they need to visit for our estimate to be any good.  
  Variational inference sidesteps the problem of calculating normalisation constants by constructing a lower bound to the marginal likelihood. For that we use an approximate posterior distribution, with a bunch of little knobs inside of it that we can adjust even per data point to make it as close to the real posterior as possible. Note that this optimization problem (of matching one distribution with another approximate one) doesn't involve the original intractable integrals we try to avoid. With some math we can show that this can give a lower bound on the thing we'd like to be maximizing (the probability of the data under our model), and so if we can optimize the parameters of our model with respect to the lower bound, maybe we'll be able to do something useful with respect to the thing we actually care about.  

  Variational inference is a paradigm where instead of trying to compute exactly the posterior distribution one searches through a parametric family for the closest (in relative entropy) distribution to the true posterior. The key observation is that one can perform stochastic gradient descent for this problem without having to compute the normalization constant in the posterior distribution (which is often an intractable problem). The only catch is that in order to compute the required gradients one needs to be able to use sample from variational posterior (sample an element of the parametric family under consideration conditioned on the observed data), and this might itself be a difficult problem in large-scale applications.

  Variational inference provides an optimization-based alternative to the sampling-based Monte Carlo methods, and tend to be more efficient. They involve approximating the exact posterior using a distribution from a more tractable family, often a fully factored one, by maximizing a variational lower bound on the log-likelihood w.r.t. the parameters of the distribution. For a small class of models, using such variational posteriors allows the expectations that specify the parameter updates to be computed analytically. However, for highly expressive models such as the ones we are interested in, these expectations are intractable even with the simplest variational posteriors. This difficulty is usually dealt with by lower bounding the intractable expectations with tractable one by introducing more variational parameters. However, this technique increases the gap between the bound being optimized and the log-likelihood, potentially resulting in a poorer fit to the data. In general, variational methods tend to be more model-dependent than sampling-based methods, often requiring non-trivial model-specific derivations.

  Traditional unbiased inference schemes such as Markov Chain Monte Carlo are often slow to run and difficult to evaluate in finite time. In contrast, variational inference allows for competitive run times and more reliable convergence diagnostics on large-scale and streaming data - while continuing to allow for complex, hierarchical modelling. The recent resurgence of interest in variational methods includes new methods for scalability using stochastic gradient methods, extensions to the streaming variational setting, improved local variational methods, inference in non-linear dynamical systems, principled regularisation in deep neural networks, and inference-based decision making in reinforcement learning, amongst others. Variational methods have clearly emerged as a preferred way to allow for tractable Bayesian inference. Despite this interest, there remain significant trade-offs in speed, accuracy, simplicity, applicability, and learned model complexity between variational inference and other approximative schemes such as MCMC and point estimation.

----

  "Variational inference is useful for dealing with latent variable models. Let's assume that for each observation x we assign a hidden variable z. Our model pθ describes the joint distribution between x and z. In such a model, typically:  
  - pθ(z) is very easy  ( 🐣 )  
  - pθ(x|z) is easy  ( 🐹 )  
  - pθ(x,z) is easy  ( 🐨 )  
  - pθ(x) is super-hard  ( 🐍 )  
  - pθ(z|x) is mega-hard  ( 🐲 )   
  to evaluate.  
  Unfortunately, in machine learning the things we want to calculate are exactly the bad guys, 🐍  and 🐲:
  - inference is evaluating pθ(z|x)  ( 🐲 )  
  - learning (via maximum likelihood) involves pθ(x)  ( 🐍 )  

  Variational lower bounds give us ways to approximately perform both inference and maximum likelihood parameter learning, by approximating the posterior 🐲  with a simpler, tamer distribution, qψ(z|x) ( 🐰 ) called the approximate posterior or recognition model. Variational inference and learning involves maximising the evidence lower bound (ELBO):

  ELBO(θ,ψ) = ∑n log p(xn) − KL[qψ(z|xn)∥pθ(z|xn)]  
  or  
  💪  = ∑n log🐍 - KL[ 🐰 || 🐲 ]

  This expression is still full of 🐍  and 🐲s but the nice thing about it is that it can be writtein more convenient forms which only contain the good guys
        🐣 🐹 🐨 🐰:

  💪  = − ∑nE🐰 log ( 🐰 / 🐨 ) + constant = ∑n E🐰 log 🐹 - E🐰 KL[ 🐰 || 🐣  ]

  Both expressions only contain nice, tame distributions and do not need explicit evaluation of either the marginal likelihood 🐍  or the posterior 🐲.

  ELBO is - as the name suggests - a lower bound to the model evidence or log likelihood. Therefore, maximising it with respect to θ and ψ approximates maximum likelihood learning, while you can use the recognition model 🐰 instead of 🐉 to perform tractable approximate inference."

  *([Ferenc Huszar](http://inference.vc/variational-renyi-lower-bound/))*

----

  ["Variational Inference: Tricks of the Trade"](http://blog.shakirm.com/2015/01/variational-inference-tricks-of-the-trade/) by Shakir Mohamed

----
#### pathwise derivative estimator for gradient of ELBO

  [reparametrization trick](http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/)

----
#### likelihood ratio estimator for gradient of ELBO

  [log derivative trick](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/)



---
### monte carlo inference

  Monte Carlo methods are a diverse class of algorithms that rely on repeated random sampling to compute the solution to problems whose solution space is too large to explore systematically or whose systemic behavior is too complex to model.

----

  [overview](https://metacademy.org/roadmaps/rgrosse/bayesian_machine_learning#sampling-algorithms) by Roger Grosse

  ["Introduction to MCMC"](http://johndcook.com/blog/2016/01/23/introduction-to-mcmc/) by John Cook  
  ["Markov Chain Monte Carlo Without all the Bullshit"](http://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/) by Jeremy Kun  

  ["Markov Chains: Why Walk When You Can Flow?"](http://elevanth.org/blog/2017/11/28/build-a-better-markov-chain/) by Richard McElreath

  ["Why is it hard to directly sample from certain statistical distributions"](https://quora.com/Why-is-it-hard-to-directly-sample-from-certain-statistical-distributions/answer/Charles-Yang-Zheng)

----

  [overview](https://youtube.com/watch?v=TNZk8lo4e-Q) by Nando de Freitas `video`  
  [overview](https://youtube.com/watch?v=M6aoDSsq2ig) by Alex Smola `video`  
  [overview](https://youtu.be/qQFF4tPgeWI?t=1h55m39s) by Bob Carpenter `video`  
  [overview](https://youtube.com/watch?v=4qfTUF9LudY) by Igor Kuralenok `video` `in russian`  
  overview by Igor Kuralenok ([first part](https://youtube.com/watch?v=q-J-wh74OJA), [second part](https://youtube.com/watch?v=6Q1YdWP92mo)) `video` `in russian`  

  [tutorial](http://research.microsoft.com/apps/video/default.aspx?id=259575) by Iain Murray `video`  
  [tutorial](http://nowozin.net/sebastian/blog/history-of-monte-carlo-methods-part-1.html) by Sebastian Nowozin `video`  

----

  ["Monte Carlo Theory, Methods and Examples"](http://statweb.stanford.edu/~owen/mc/) by Art Owen `book`

  [visualization](https://chi-feng.github.io/mcmc-demo/app.html)

  [implementations](https://github.com/wiseodd/MCMC)  
  [implementations](https://plot.ly/ipython-notebooks/computational-bayesian-analysis/)  



---
### likelihood-free inference

  Some statistical models are specified via data generating process for which likelihood function is intractable and cannot be evaluated numerically in a practical time.  
  Standard likelihood-based inference is then not feasible but model parameters can be inferred by finding values which yield simulated data that resemble observed data.  

----

  [overview](http://dennisprangle.github.io/research/2016/06/07/bayesian-inference-by-neural-networks) by Dennis Prangle  
  [overview](https://casmls.github.io/general/2016/10/02/abc.html) by Scott Linderman  

  ["Machine Learning and Likelihood-Free Inference in Particle Physics"](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Machine-Learning-and-Likelihood-Free-Inference-in-Particle-Physics) by Kyle Cranmer `video`  

----

  [history and key papers](http://dennisprangle.github.io/research/2016/01/03/LFtimeline)

  ["Likelihood-free Inference via Classification"](#likelihood-free-inference-via-classification-gutmann-dutta-kaski-corander) by Gutmann et al. `paper` `summary`  
  ["Fast Epsilon-free Inference of Simulation Models with Bayesian Conditional Density Estimation"](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#fast-epsilon-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation-papamakarios-murray) by Papamakarios et al. `paper` `summary`  
  ["Sequential Neural Likelihood: Fast Likelihood-free Inference with Autoregressive Flows"](https://arxiv.org/abs/1805.07226) by Papamakarios et al. `paper`  
  ["Likelihood-free Inference with Emulator Networks"](https://arxiv.org/abs/1805.09294) by Lueckmann et al. `paper`  

----

  [**generative adversarial networks**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#generative-models---generative-adversarial-networks) as implicit models with likelihood-free inference



---
### belief propagation

  [overview](https://metacademy.org/roadmaps/rgrosse/bayesian_machine_learning#belief-propagation) by Roger Grosse

  [expectation propagation](https://metacademy.org/graphs/concepts/expectation_propagation)



---
### interesting papers

[**interesting papers - bayesian deep learning**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---bayesian-deep-learning)  
[**interesting papers - variational autoencoder**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers---variational-autoencoder)  
[**interesting papers - probabilistic programming**](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md#interesting-papers)  


[**interesting recent papers - bayesian deep learning**](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#bayesian-deep-learning)  
[**interesting recent papers - variational autoencoders**](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#generative-models---variational-autoencoders)  
[**interesting recent papers - unsupervised learning**](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#unsupervised-learning)  



----
#### ["A Contemporary Overview of Probabilistic Latent Variable Models"](https://arxiv.org/abs/1706.08137) Farouni
>	"In this paper we provide a conceptual overview of latent variable models within a probabilistic modeling framework, an overview that emphasizes the compositional nature and the interconnectedness of the seemingly disparate models commonly encountered in statistical practice."


#### ["Inside-Outside and Forward-Backward Algorithms Are Just Backprop"](https://www.cs.jhu.edu/~jason/papers/eisner.spnlp16.pdf) Eisner
>	"A probabilistic or weighted grammar implies a posterior probability distribution over possible parses of a given input sentence. One often needs to extract information from this distribution, by computing the expected counts (in the unknown parse) of various grammar rules, constituents, transitions, or states. This requires an algorithm such as inside-outside or forward-backward that is tailored to the grammar formalism. Conveniently, each such algorithm can be obtained by automatically differentiating an “inside” algorithm that merely computes the log-probability of the evidence (the sentence). This mechanical procedure produces correct and efficient code. As for any other instance of back-propagation, it can be carried out manually or by software. This pedagogical paper carefully spells out the construction and relates it to traditional and nontraditional views of these algorithms."


#### ["The Markov Chain Monte Carlo Revolution"](http://math.uchicago.edu/~shmuel/Network-course-readings/MCMCRev.pdf) Diaconis
>	"The use of simulation for high dimensional intractable computations has revolutionized applied mathematics. Designing, improving and understanding the new tools leads to (and leans on) fascinating mathematics, from representation theory through micro-local analysis."


#### ["Markov Chain Monte Carlo and Variational Inference: Bridging the Gap"](http://jmlr.org/proceedings/papers/v37/salimans15.pdf) Salimans, Kingma, Welling
>	"Recent advances in stochastic gradient variational inference have made it possible to perform variational Bayesian inference with posterior approximations containing auxiliary random variables. This enables us to explore a new synthesis of variational inference and Monte Carlo methods where we incorporate one or more steps of MCMC into our variational approximation. By doing so we obtain a rich class of inference algorithms bridging the gap between variational methods and MCMC, and offering the best of both worlds: fast posterior approximation through the maximization of an explicit objective, with the option of trading off additional computation for additional accuracy. We describe the theoretical foundations that make this possible and show some promising first results."

  - `video` <http://videolectures.net/icml2015_salimans_variational_inference/> (Salimans)


#### ["Likelihood-free Inference via Classification"](https://arxiv.org/abs/1407.4981) Gutmann, Dutta, Kaski, Corander
>	"Increasingly complex generative models are being used across disciplines as they allow for realistic characterization of data, but a common difficulty with them is the prohibitively large computational cost to evaluate the likelihood function and thus to perform likelihood-based statistical inference. A likelihood-free inference framework has emerged where the parameters are identified by finding values that yield simulated data resembling the observed data. While widely applicable, a major difficulty in this framework is how to measure the discrepancy between the simulated and observed data. Transforming the original problem into a problem of classifying the data into simulated versus observed, we find that classification accuracy can be used to assess the discrepancy. The complete arsenal of classification methods becomes thereby available for inference of intractable generative models. We validate our approach using theory and simulations for both point estimation and Bayesian inference, and demonstrate its use on real data by inferring an individual-based epidemiological model for bacterial infections in child care centers."

>	"At about the same time we first presented our work, Goodfellow et al (2014) proposed to use nonlinear logistic regression to train a neural network (Generative Adversarial Networks) such that it transforms “noise” samples into samples approximately following the same distribution as some given data set. The main difference to our work is that the method of Goodfellow et al (2014) is a method for producing random samples while ours is a method for statistical inference."


#### ["AIDE: An Algorithm for Measuring the Accuracy of Probabilistic Inference Algorithms"](https://arxiv.org/abs/1705.07224) Cusumano-Towner, Mansinghka
>	"Approximate probabilistic inference algorithms are central to many fields. Examples include sequential Monte Carlo inference in robotics, variational inference in machine learning, and Markov chain Monte Carlo inference in statistics. A key problem faced by practitioners is measuring the accuracy of an approximate inference algorithm on a specific data set. This paper introduces the auxiliary inference divergence estimator (AIDE), an algorithm for measuring the accuracy of approximate inference algorithms. AIDE is based on the observation that inference algorithms can be treated as probabilistic models and the random variables used within the inference algorithm can be viewed as auxiliary variables. This view leads to a new estimator for the symmetric KL divergence between the approximating distributions of two inference algorithms. The paper illustrates application of AIDE to algorithms for inference in regression, hidden Markov, and Dirichlet process mixture models. The experiments show that AIDE captures the qualitative behavior of a broad class of inference algorithms and can detect failure modes of inference algorithms that are missed by standard heuristics."



---
### interesting papers - applications

[**interesting papers - probabilistic programming - applications**](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md#interesting-papers---applications)  


#### ["A Generative Vision Model that Trains with High Data Efficiency and Breaks Text-based CAPTCHAs"](http://science.sciencemag.org/content/early/2017/10/26/science.aag2612.full) George et al.
>	"Learning from few examples and generalizing to dramatically different situations are capabilities of human visual intelligence that are yet to be matched by leading machine learning models. By drawing inspiration from systems neuroscience, we introduce a probabilistic generative model for vision in which message-passing based inference handles recognition, segmentation and reasoning in a unified way. The model demonstrates excellent generalization and occlusion-reasoning capabilities, and outperforms deep neural networks on a challenging scene text recognition benchmark while being 300-fold more data efficient. In addition, the model fundamentally breaks the defense of modern text-based CAPTCHAs by generatively segmenting characters without CAPTCHA-specific heuristics. Our model emphasizes aspects like data efficiency and compositionality that may be important in the path toward general artificial intelligence."

  - `post` <https://vicarious.com/2017/10/26/common-sense-cortex-and-captcha/>
  - `video` <https://slideslive.com/38909792/building-machines-that-work-like-the-brain> (George)
  - `video` <https://youtube.com/watch?v=y459Yip5vRQ> (George)
  - `video` <https://youtube.com/watch?v=lmeZIHkep7c> (George)
  - `code` <https://github.com/vicariousinc/science_rcn>


#### ["Unsupervised Learning of 3D Structure from Images"](http://arxiv.org/abs/1607.00662) Rezende, Eslami, Mohamed, Battaglia, Jaderberg, Heess
>	"A key goal of computer vision is to recover the underlying 3D structure from 2D observations of the world. In this paper we learn strong deep generative models of 3D structures, and recover these structures from 3D and 2D images via probabilistic inference. We demonstrate high-quality samples and report log-likelihoods on several datasets, including ShapeNet, and establish the first benchmarks in the literature. We also show how these models and their inference networks can be trained end-to-end from 2D images. This demonstrates for the first time the feasibility of learning to infer 3D representations of the world in a purely unsupervised manner."

>	"A key goal of computer vision is that of recovering the underlying 3D structure that gives rise to these 2D observations. The 2D projection of a scene is a complex function of the attributes and positions of the camera, lights and objects that make up the scene. If endowed with 3D understanding, agents can abstract away from this complexity to form stable, disentangled representations, e.g., recognizing that a chair is a chair whether seen from above or from the side, under different lighting conditions, or under partial occlusion. Moreover, such representations would allow agents to determine downstream properties of these elements more easily and with less training, e.g., enabling intuitive physical reasoning about the stability of the chair, planning a path to approach it, or figuring out how best to pick it up or sit on it. Models of 3D representations also have applications in scene completion, denoising, compression and generative virtual reality."

>	"There have been many attempts at performing this kind of reasoning, dating back to the earliest years of the field. Despite this, progress has been slow for several reasons: First, the task is inherently ill-posed. Objects always appear under self-occlusion, and there are an infinite number of 3D structures that could give rise to a particular 2D observation. The natural way to address this problem is by learning statistical models that recognize which 3D structures are likely and which are not. Second, even when endowed with such a statistical model, inference is intractable. This includes the sub-tasks of mapping image pixels to 3D representations, detecting and establishing correspondences between different images of the same structures, and that of handling the multi-modality of the representations in this 3D space. Third, it is unclear how 3D structures are best represented, e.g., via dense volumes of voxels, via a collection of vertices, edges and faces that define a polyhedral mesh, or some other kind of representation. Finally, ground-truth 3D data is difficult and expensive to collect and therefore datasets have so far been relatively limited in size and scope."

>	"(a) We design a strong generative model of 3D structures, defined over the space of volumes and meshes, using ideas from state-of-the-art generative models of images.  
>	(b) We show that our models produce high-quality samples, can effectively capture uncertainty and are amenable to probabilistic inference, allowing for applications in 3D generation and simulation. We report log-likelihoods on a dataset of shape primitives, a 3D version of MNIST, and on ShapeNet, which to the best of our knowledge, constitutes the first quantitative benchmark for 3D density modeling.  
>	(c) We show how complex inference tasks, e.g., that of inferring plausible 3D structures given a 2D image, can be achieved using conditional training of the models. We demonstrate that such models recover 3D representations in one forward pass of a neural network and they accurately capture the multi-modality of the posterior.  
>	(d) We explore both volumetric and mesh-based representations of 3D structure. The latter is achieved by flexible inclusion of off-the-shelf renders such as OpenGL. This allows us to build in further knowledge of the rendering process, e.g., how light bounces of surfaces and interacts with its material’s attributes.  
>	(e) We show how the aforementioned models and inference networks can be trained end-to-end directly from 2D images without any use of ground-truth 3D labels. This demonstrates for the first time the feasibility of learning to infer 3D representations of the world in a purely unsupervised manner."  

>	"In this paper we introduced a powerful family of 3D generative models inspired by recent advances in image modeling. We showed that when trained on ground-truth volumes, they can produce high-quality samples that capture the multi-modality of the data. We further showed how common inference tasks, such as that of inferring a posterior over 3D structures given a 2D image, can be performed efficiently via conditional training. We also demonstrated end-to-end training of such models directly from 2D images through the use of differentiable renderers. We experimented with two kinds of 3D representations: volumes and meshes. Volumes are flexible and can capture a diverse range of structures, however they introduce modeling and computational challenges due to their high dimensionality. Conversely, meshes can be much lower dimensional and therefore easier to work with, and they are the data-type of choice for common rendering engines, however standard paramaterizations can be restrictive in the range of shapes they can capture."

  - `video` <https://youtube.com/watch?v=stvDAGQwL5c> + <https://goo.gl/9hCkxs> (demo)
  - `video` <https://docs.google.com/presentation/d/12uZQ_Vbvt3tzQYhWR3BexqOzbZ-8AeT_jZjuuYjPJiY/pub?start=true&loop=true&delayms=30000#slide=id.g1329951dde_0_0> (demo)


#### ["Human-level Concept Learning Through Probabilistic Program Induction"](http://web.mit.edu/cocosci/Papers/Science-2015-Lake-1332-8.pdf) Lake, Salakhutdinov, Tenenbaum
>	"People learning new concepts can often generalize successfully from just a single example, yet machine learning algorithms typically require tens or hundreds of examples to perform with similar accuracy. People can also use learned concepts in richer ways than conventional algorithms - for action, imagination, and explanation. We present a computational model that captures these human learning abilities for a large class of simple visual concepts: handwritten characters from the world’s alphabets. The model represents concepts as simple programs that best explain observed examples under a Bayesian criterion. On a challenging one-shot classification task, the model achieves human-level performance while outperforming recent deep learning approaches. We also present several “visual Turing tests” probing the model’s creative generalization abilities, which in many cases are indistinguishable from human behavior."

----
>	"Vision program outperformed humans in identifying handwritten characters, given single training example"

>	"This work brings together three key ideas -- compositionality, causality, and learning-to-learn --- challenging (in a good way) the traditional deep learning approach"

  - `video` <http://youtube.com/watch?v=kzl8Bn4VtR8> (Lake)
  - `video` <http://techtalks.tv/talks/one-shot-learning-of-simple-fractal-concepts/63049/> (Lake)
  - `video` <http://youtu.be/quPN7Hpk014?t=21m5s> (Tenenbaum)
  - `notes` <https://casmls.github.io/general/2017/02/08/oneshot.html>
  - `code` <https://github.com/brendenlake/BPL>


#### ["Variational Autoencoders for Collaborative Filtering"](https://arxiv.org/abs/1802.05814) Liang, Krishnan, Hoffman, Jebara
>	"We extend variational autoencoders to collaborative filtering for implicit feedback. This non-linear probabilistic model enables us to go beyond the limited modeling capacity of linear factor models which still largely dominate collaborative filtering research. We introduce a generative model with multinomial likelihood and use Bayesian inference for parameter estimation. Despite widespread use in language modeling and economics, the multinomial likelihood receives less attention in the recommender systems literature. We introduce a different regularization parameter for the learning objective, which proves to be crucial for achieving competitive performance. Remarkably, there is an efficient way to tune the parameter using annealing. The resulting model and learning algorithm has information-theoretic connections to maximum entropy discrimination and the information bottleneck principle. Empirically, we show that the proposed approach significantly outperforms several state-of-the-art baselines, including two recently-proposed neural network approaches, on several real-world datasets. We also provide extended experiments comparing the multinomial likelihood with other commonly used likelihood functions in the latent factor collaborative filtering literature and show favorable results. Finally, we identify the pros and cons of employing a principled Bayesian inference approach and characterize settings where it provides the most significant improvements."

>	"Recommender systems is more of a "small data" than a "big data" problem."  
>	"VAE generalizes linear latent factor model and recovers Gaussian matrix factorization as a special linear case. No iterative procedure required to rank all the items given a user's watch history - only need to evaluate inference and generative functions."  
>	"We introduce a regularization parameter for the learning objective to trade-off the generative power for better predictive recommendation performance. For recommender systems, we don't necessarily need all the statistical property of a generative model. We trade off the ability of performing ancestral sampling for better fitting the data."  

  - `video` <https://youtube.com/watch?v=gRvxr47Gj3k> (Liang)
  - `code` <https://github.com/dawenl/vae_cf>


#### ["TrueSkill(TM): A Bayesian Skill Rating System"](http://research.microsoft.com/apps/pubs/default.aspx?id=67956) Herbrich, Minka, Graepel
>	"We present a new Bayesian skill rating system which can be viewed as a generalisation of the Elo system used in Chess. The new system tracks the uncertainty about player skills, explicitly models draws, can deal with any number of competing entities and can infer individual skills from team results. Inference is performed by approximate message passing on a factor graph representation of the model. We present experimental evidence on the increased accuracy and convergence speed of the system compared to Elo and report on our experience with the new rating system running in a large-scale commercial online gaming service under the name of TrueSkill."

  - <http://trueskill.org>
  - `video` <https://youtube.com/watch?v=Y3obG7F1crw&t=32m28s> (Bishop)
  - `video` <http://videolectures.net/ecmlpkdd2010_graepel_mlm/> (Graepel)
  - `post` <http://moserware.com/2010/03/computing-your-skill.html>
  - `paper` ["The Math Behind TrueSkill"](http://www.moserware.com/assets/computing-your-skill/The%20Math%20Behind%20TrueSkill.pdf) by Jeff Moser
  - `paper` ["TrueSkill 2: An Improved Bayesian Skill Rating System"](https://microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system) by Minka, Cleven, Zaykov


#### ["Matchbox: Large Scale Bayesian Recommendations"](http://research.microsoft.com/apps/pubs/default.aspx?id=79460) Stern, Herbrich, Graepel
>	"We present a probabilistic model for generating personalised recommendations of items to users of a web service. The Matchbox system makes use of content information in the form of user and item meta data in combination with collaborative filtering information from previous user behavior in order to predict the value of an item for a user. Users and items are represented by feature vectors which are mapped into a low-dimensional ‘trait space’ in which similarity is measured in terms of inner products. The model can be trained from different types of feedback in order to learn user-item preferences. Here we present three alternatives: direct observation of an absolute rating each user gives to some items, observation of a binary preference (like/don’t like) and observation of a set of ordinal ratings on a user-specific scale. Efficient inference is achieved by approximate message passing involving a combination of Expectation Propagation and Variational Message Passing. We also include a dynamics model which allows an item’s popularity, a user’s taste or a user’s personal rating scale to drift over time. By using Assumed-Density Filtering for training, the model requires only a single pass through the training data. This is an on-line learning algorithm capable of incrementally taking account of new data so the system can immediately reflect the latest user preferences. We evaluate the performance of the algorithm on the MovieLens and Netflix data sets consisting of approximately 1,000,000 and 100,000,000 ratings respectively. This demonstrates that training the model using the on-line ADF approach yields state-of-the-art performance with the option of improving performance further if computational resources are available by performing multiple EP passes over the training data."

  - <https://dotnet.github.io/infer/userguide/Learners/Matchbox%20recommender.html>
  - `video` <http://videolectures.net/ecmlpkdd2010_graepel_mlm/#t=1265> (Graepel)


#### ["Bayesian Optimization in AlphaGo"](https://arxiv.org/abs/1812.06855) Chen, Huang, Wang, Antonoglou, Schrittwieser, Silver, Freitas
>	"During the development of AlphaGo, its many hyper-parameters were tuned with Bayesian optimization multiple times. This automatic tuning process resulted in substantial improvements in playing strength. For example, prior to the match with Lee Sedol, we tuned the latest AlphaGo agent and this improved its win-rate from 50% to 66.5% in self-play games. This tuned version was deployed in the final match. Of course, since we tuned AlphaGo many times during its development cycle, the compounded contribution was even higher than this percentage. It is our hope that this brief case study will be of interest to Go fans, and also provide Bayesian optimization practitioners with some insights and inspiration."


#### ["Inverting a Steady-State"](http://theory.stanford.edu/~sergei/papers/wsdm15-cset.pdf) Kumar, Tomkins, Vassilvitskii, Vee
>	"We consider the problem of inferring choices made by users based only on aggregate data containing the relative popularity of each item. We propose a framework that models the problem as that of inferring a Markov chain given a stationary distribution. Formally, we are given a graph and a target steady-state distribution on its nodes. We are also given a mapping from per-node scores to a transition matrix, from a broad family of such mappings. The goal is to set the scores of each node such that the resulting transition matrix induces the desired steady state. We prove sufficient conditions under which this problem is feasible and, for the feasible instances, obtain a simple algorithm for a generic version of the problem. This iterative algorithm provably finds the unique solution to this problem and has a polynomial rate of convergence; in practice we find that the algorithm converges after fewer than ten iterations. We then apply this framework to choice problems in online settings and show that our algorithm is able to explain the observed data and predict the user choices much better than other competing baselines across a variety of diverse datasets."

  - `post` <http://cm.cecs.anu.edu.au/post/invert_random_walk_problem/>
  - `post` <https://jfmackay.wordpress.com/2015/05/14/inverting-the-steady-state-of-a-markov-chain/>


#### ["The Human Kernel"](http://arxiv.org/abs/1510.07389) Wilson, Dann, Lucas, Xing
>	"Bayesian nonparametric models, such as Gaussian processes, provide a compelling framework for automatic statistical modelling: these models have a high degree of flexibility, and automatically calibrated complexity. However, automating human expertise remains elusive; for example, Gaussian processes with standard kernels struggle on function extrapolation problems that are trivial for human learners. In this paper, we create function extrapolation problems and acquire human responses, and then design a kernel learning framework to reverse engineer the inductive biases of human learners across a set of behavioral experiments. We use the learned kernels to gain psychological insights and to extrapolate in humanlike ways that go beyond traditional stationary and polynomial kernels. Finally, we investigate Occam’s razor in human and Gaussian process based function learning."

>	"We have shown that (1) human learners have systematic expectations about smooth functions that deviate from the inductive biases inherent in the RBF kernels that have been used in past models of function learning; (2) it is possible to extract kernels that reproduce qualitative features of human inductive biases, including the variable sawtooth and step patterns; (3) that human learners favour smoother or simpler functions, even in comparison to GP models that tend to over-penalize complexity; and (4) that it is possible to build models that extrapolate in human-like ways which go beyond traditional stationary and polynomial kernels."

>	"We have focused on human extrapolation from noise-free nonparametric relationships. This approach complements past work emphasizing simple parametric functions and the role of noise, but kernel learning might also be applied in these other settings. In particular, iterated learning experiments provide a way to draw samples that reflect human learners’ a priori expectations. Like most function learning experiments, past IL experiments have presented learners with sequential data. Our approach, following Little and Shiffrin, instead presents learners with plots of functions. This method is useful in reducing the effects of memory limitations and other sources of noise (e.g., in perception). It is possible that people show different inductive biases across these two presentation modes. Future work, using multiple presentation formats with the same underlying relationships, will help resolve these questions. Finally, the ideas discussed in this paper could be applied more generally, to discover interpretable properties of unknown models from their predictions. Here one encounters fascinating questions at the intersection of active learning, experimental design, and information theory."

  - `video` <http://research.microsoft.com/apps/video/default.aspx?id=259610> (11:30) (Wilson)
