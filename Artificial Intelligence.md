  Intelligence is information processing necessary to achieve goals with limited resources.  


  * [overview](#overview)
  * [benchmarks and competitions](#benchmarks-and-competitions)
  * [problems](#problems)
  * [knowledge representation](#knowledge-representation)
  * [inference/reasoning](#inferencereasoning)
  * [decisions/actions](#decisionsactions)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Meta-learning](#meta-learning)
    - [Artificial Curiosity and Creativity](#artificial-curiosity-and-creativity)
    - [Solomonoff Induction and AIXI](#solomonoff-induction-and-aixi)
  * [interesting papers](#interesting-papers)
    - [definitions and measures of intelligence](#interesting-papers---definitions-and-measures-of-intelligence)
    - [Solomonoff Induction and AIXI](#interesting-papers---solomonoff-induction-and-aixi)
    - [Artificial Curiosity and Creativity](#interesting-papers---artificial-curiosity-and-creativity)

----

  [knowledge representation and reasoning](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md)  
  [machine learning](https://github.com/brylevkirill/notes/blob/master/Machine%20Learning.md)  
  [deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md)  
  [reinforcement learning](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md)  
  [bayesian inference and learning](https://dropbox.com/s/7vlg0vhb51rd6c1/Bayesian%20Inference%20and%20Learning.txt)  
  [probabilistic programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md)  

  [natural language processing](https://dropbox.com/s/0kw1s9mrrcwct0u/Natural%20Language%20Processing.txt)  
  [information retrieval](https://dropbox.com/s/21ugi2p9uy1shvt/Information%20Retrieval.txt)  
  [personal assistants](https://dropbox.com/s/0fyarlwcfb8mjdq/Personal%20Assistants.txt)  
  [semantic web](https://github.com/brylevkirill/tech/blob/master/RDF/RDF.txt)  



---
### overview

[recent breakthroughs](https://github.com/brylevkirill/posts/blob/master/breakthroughs.md) (in russian)


  - Juergen Schmidhuber  
	["Compression Progress: The Algorithmic Principle Behind Curiosity and Creativity"](https://vimeo.com/7441291)  
	["Formal Theory of Fun & Creativity"](http://videolectures.net/ecmlpkdd2010_schmidhuber_ftf/)  

  - Demis Hassabis  
	["Towards General Artificial Intelligence"](https://youtube.com/watch?v=vQXAsdMa_8A)  

  - Richard Sutton  
	["The Future of Artificial Intelligence Belongs to Search and Learning"](http://www.fields.utoronto.ca/talks/advances-reinforcement-learning)  

  - Geoffrey Hinton  
	["Deep Learning and AI"](https://youtube.com/watch?v=izrG86jycck)  

  - Yann LeCun  
	["Obstacles to Progress in AI"](https://youtube.com/watch?v=9Kgk4s7yG1c)  

  - Yoshua Bengio  
	["Deep Learning and General AI"](https://youtube.com/watch?v=exhdfIPzj24)  

  - David Ferucci  
	["AI: A Return To Meaning"](https://youtube.com/watch?v=F_0hpnLdNjk)  

  - Joshua Tenenbaum  
	["Building Machines That Learn like Humans"](https://youtube.com/watch?v=quPN7Hpk014)  


  "What you wanted to know about AI" (informal) -
	[first part](http://fastml.com/what-you-wanted-to-know-about-ai/) +
	[second part](http://fastml.com/what-you-wanted-to-know-about-ai-part-ii/)


  [introductory course](https://youtube.com/channel/UCHBzJsIcRIVuzzHVYabikTQ/videos) from UC Berkeley



---
### benchmarks and competitions

  ["Tests of Machine Intelligence"](http://arxiv.org/abs/0712.3825) by Legg and Hutter

----
#### universal problem solving

  ["A Formal Measure of Machine Intelligence"](https://arxiv.org/abs/cs/0605024) by Legg and Hutter  
  ["An Approximation of the Universal Intelligence Measure"](http://arxiv.org/abs/1109.5951) by Legg and Veness  
  ["Space-Time Embedded Intelligence"](http://frontiersinai.com/turingfiles/December/orseau.pdf) by Orseau and Ring  

----
#### automated science

  [Automated Machine Learning](https://github.com/brylevkirill/notes/blob/master/Machine%20Learning.md#automated-machine-learning)

----
#### dialog interaction

  ["A Paradigm for Situated and Goal-Driven Language Learning"](https://arxiv.org/abs/1610.03585) by Gauthier and Mordatch  
  ["A Roadmap Towards Machine Intelligence"](http://arxiv.org/abs/1511.08130) by Mikolov, Joulin, Baroni  

----
#### games

  ["Measuring Intelligence through Games"](http://arxiv.org/abs/1109.1314) by Schaul, Togelius, Schmidhuber  

  ["A 'Brief' History of Game AI Up To AlphaGo"](http://andreykurenkov.com/writing/a-brief-history-of-game-ai/) by Andrey Kurenkov  
  ["AI for classic games"](http://youtube.com/watch?v=kZ_AUmFcZtk) by David Silver  

  ["DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker"](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#moravcik-et-al---deepstack-expert-level-artificial-intelligence-in-no-limit-poker) by Michael Bowling and others  
  ["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#silver-et-al---mastering-the-game-of-go-with-deep-neural-networks-and-tree-search) by David Silver and others  
  ["Giraffe: Using Deep Reinforcement Learning to Play Chess"](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#lai---giraffe-using-deep-reinforcement-learning-to-play-chess) by Matthew Lai  

  ["Why video games are essential for inventing artificial intelligence"](http://togelius.blogspot.ru/2016/01/why-video-games-are-essential-for.html) by Julian Togelius  
  ["Deep Learning for Video Game Playing"](https://arxiv.org/abs/1708.07902) by Justesen et al.  

  [games](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#games)  

----
#### open world problems

  - OpenAI Universe  
	<https://universe.openai.com>  
	<https://openai.com/blog/universe/>  
	<https://youtube.com/watch?v=BskhUBPRrqE> (Ilya Sutskever)  

  - OpenAI World of Bits  
	<http://alpha.openai.com/miniwob/index.html>  
	<http://proceedings.mlr.press/v70/shi17a.html>  
	<http://alpha.openai.com/ak_rework_2017.pdf> (Andrej Karpathy)  
	<https://youtube.com/watch?v=w6PCLESiExQ> (Jim Fan)  
	<http://alpha.openai.com/kalite_exercises/index.html>  

  - OpenAI Gym  
	<https://gym.openai.com>  
	<https://gym.openai.com/envs>  
	<https://openai.com/blog/openai-gym-beta/>  

  - DeepMind Lab  
	<https://github.com/deepmind/lab>  
	<https://deepmind.com/blog/open-sourcing-deepmind-lab/>  

  - Microsoft Malmo  
	<https://microsoft.com/en-us/research/project/project-malmo/>  
	<https://youtube.com/watch?v=KkVj_ddseO8>  
	<https://youtube.com/watch?v=399qJUBRA0o> (Katja Hofmann)  

----
#### [machine reading](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#machine-reading-benchmarks)  

----

  ["Measuring the Progress of AI Research"](https://eff.org/files/AI-progress-metrics.html)

----

  ["Unsolved Problems in AI"](https://medium.com/ai-roadmap-institute/unsolved-problems-in-ai-38f4ce18921d)  
  ["Concrete AI tasks for forecasting"](http://aiimpacts.org/concrete-ai-tasks-for-forecasting/)  



---
### problems

  - [*knowledge representation*](#knowledge-representation)  
	What is space of possible beliefs? How do beliefs interact?  

  - [*inference/reasoning*](#inferencereasoning)  
	How can inference and reasoning be performed over beliefs?  

  - [*decisions/actions*](#decisionsactions)  
	How can beliefs conspire to produce decisions and actions?  



---
### knowledge representation

  [Knowledge Representation](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md)  
  [Deep Learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md)  
  [Probabilistic Programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md)  


  "Knowledge is in our minds and language is just orienting us within our common experiences."  
  "Language is complex indexing system that points to shared experiences of people on which meaning is grounded."  
  "Communicating using language is possible only after lining up experiences."  
  "Language is very flexible thing and not a formal mathematical structure at all."  
  "Computers may be programmed to analyze data and detect meaning but humans are the source of meaning."  

  - *symbolic knowledge* ("dog as word") - logic networks
  - *conceptual knowledge* ("dog as mammal, a good companion, a good guardian") - open research area
  - *perceptual knowledge* ("dog as something with certain physical appearance") - deep learning

  symbolic -> conceptual (words have to be grounded in real world and actions to be not recursive)  
  perceptual -> conceptual (prediction and classification are not enough for some tasks without reasoning over concepts)  



---
### inference/reasoning

  [Reasoning](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md)  
  [Bayesian Inference and Learning](https://dropbox.com/s/7vlg0vhb51rd6c1/Bayesian%20Inference%20and%20Learning.txt)  
  [Machine Learning](https://github.com/brylevkirill/notes/blob/master/Machine%20Learning.md)  


  approaches:
   - logical/symbolic/computationalism/logical/causational/theory-driven/relational vs statistical/non-symbolic/connectionism/correlational/data-driven/numerical
   - deductive vs inductive reasoning


----
#### logical vs statistical approaches

  "Beginning with Leibniz, scholars have attempted to unify logic and probability. For “classical” AI, based largely on first-order logic, the purpose of such a unification is to handle uncertainty and facilitate learning from real data; for “modern” AI, based largely on probability theory, the purpose is to acquire formal languages with sufficient expressive power to handle complex domains and incorporate prior knowledge. The world is uncertain and it has things in it. To deal with this, we have to unify logic and probability."

  - *knowledge representation*:  
	logical - first-order logic  
	statistical - [probabilistic programs](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md)  
  - *reasoning*:  
	logical - satisfiability testing/proving  
	statistical - [bayesian inference](https://dropbox.com/s/7vlg0vhb51rd6c1/Bayesian%20Inference%20and%20Learning.txt)  
  - *learning*:  
	logical - inductive logic programming  
	statistical - [bayesian deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#bayesian-deep-learning)


["On Chomsky and the Two Cultures of Statistical Learning"](http://norvig.com/chomsky.html) by Peter Norvig  
>	"mathematical model specifies a relation among variables, either in functional or in relational form"  
>	"statistical model is a mathematical model which is modified or trained by the input of data points"  
>	"probabilistic model specifies a probability distribution over possible values of random variables"  
>	"statistical models are often but not always probabilistic and probabilistic ones are statistical"  

["Unifying Logic and Probability"](https://www.cs.berkeley.edu/~russell/papers/ipmu14-oupm.pdf) by Stuart Russell
	([video](http://video.upmc.fr/differe.php?collec=S_C_colloquium_lip6_2012&video=3))
>	"Beginning with Leibniz, scholars have attempted to unify logic and probability. For “classical” AI, based largely on first-order logic, the purpose of such a unification is to handle uncertainty and facilitate learning from real data; for “modern” AI, based largely on probability theory, the purpose is to acquire formal languages with sufficient expressive power to handle complex domains and incorporate prior knowledge."

["Unifying Logical and Statistical AI"](http://homes.cs.washington.edu/~pedrod/papers/aaai06c.pdf) by Pedro Domingos
	([video](http://youtube.com/watch?v=bW5DzNZgGxY))
>	"logic handles complexity and statistics handles uncertainty"


----
#### symbolic vs non-symbolic approaches

  ["Aetherial Symbols"](https://drive.google.com/file/d/0B8i61jl8OE3XdHRCSkV1VFNqTWc) by Geoffrey Hinton
>	"The fathers of AI believed that formal logic provided insight into how human reasoning must work. For implications to travel from one sentence to the next, there had to be rules of inference containing variables that got bound to symbols in the first sentence and carried the implications to the second sentence. I shall demonstrate that this belief is as incorrect as the belief that a lightwave can only travel through space by causing disturbances in the luminiferous aether. In both cases, scientists were misled by compelling but incorrect analogies to the only systems they knew that had the required properties. Arguments have little impact on such strongly held beliefs. What is needed is a demonstration that it is possible to propagate implications in some quite different way that does not involve rules of inference and has no resemblance to formal logic. Recent results in machine translation using recurrent neural networks show that the meaning of a sentence can be captured by a "thought vector" which is simply the hidden state vector of a recurrent net that has read the sentence one word at a time. In future, it will be possible to predict thought vectors from the sequence of previous thought vectors and this will capture natural human reasoning. With sufficient effort, it may even be possible to train such a system to ignore nearly all of the contents of its thoughts and to make predictions based purely on those features of the thoughts that capture the logical form of the sentences used to express them."

>	"If we can convert a sentence into a vector that captures the meaning of the sentence, then google can do much better searches, they can search based on what is being said in a document. Also, if you can convert each sentence in a document into a vector, you can then take that sequence of vectors and try and model why you get this vector after you get these vectors, that's called reasoning, that's natural reasoning, and that was kind of the core of good old fashioned AI and something they could never do because natural reasoning is a complicated business, and logic isn't a very good model of it, here we can say, well, look, if we can read every english document on the web, and turn each sentence into a thought vector, we've got plenty of data for training a system that can reason like people do. Now, you might not want to reason like people do on the web, but at least we can see what they would think."

>	"Most people fall for the traditional AI fallacy that thought in the brain must somehow resemble lisp expressions. You can tell someone what thought you are having by producing a string of words that would normally give rise to that thought but this doesn't mean the thought is a string of symbols in some unambiguous internal language. The new recurrent network translation models make it clear that you can get a very long way by treating a thought as a big state vector. Traditional AI researchers will be horrified by the view that thoughts are merely the hidden states of a recurrent net and even more horrified by the idea that reasoning is just sequences of such state vectors. That's why I think its currently very important to get our critics to state, in a clearly decideable way, what it is they think these nets won't be able to learn to do. Otherwise each advance of neural networks will be met by a new reason for why that advance does not really count. So far, I have got both Garry Marcus and Hector Levesque to agree that they will be impressed if neural nets can correctly answer questions about "Winograd" sentences such as "The city councilmen refused to give the demonstrators a licence because they feared violence." Who feared the violence?"

>	"There are no symbols inside the encoder and decoder neural nets for machine translation. The only symbols are at the input and output. Processing pixel arrays is not done by manipulating internal pixels. Maybe processing symbol strings is not done by manipulating internal symbol strings. It was obvious to physicists that light waves must have an aether to propagate from one place to the next. They thought there was no other possibility. It was obvious to AI researchers that people must use formal rules of inference to propagate implications from one proposition to the next. They thought there was no other possibility. What is inside the black box is not necessarily what goes in or what comes out. The physical symbol system hypothesis is probably false."

>	"Most of our reasoning is by analogy; it's not logical reasoning. The early AI guys thought we had to use logic as a model and so they couldn't cope with reasoning by analogy. The honest ones, like Allen Newell, realized that reasoning by analogy was a huge problem for them, but they weren't willing to say that reasoning by analogy is the core kind of reasoning we do, and logic is just a sort of superficial thing on top of it that happens much later."

  "Neural Programmer Interpreters" by Nando de Freitas - <https://youtu.be/SAcHyzMdbXc?t=5m13s> + <http://videolectures.net/kdd2016_de_freitas_recurrent_neural/#t=3130>
>	"For me there are two types of generalisation, which I will refer to as Symbolic and Connectionist generalisation. If we teach a machine to sort sequences of numbers of up to length 10 or 100, we should expect them to sort sequences of length 1000 say. Obviously symbolic approaches have no problem with this form of generalisation, but neural nets do poorly. On the other hand, neural nets are very good at generalising from data (such as images), but symbolic approaches do poorly here. One of the holy grails is to build machines that are capable of both symbolic and connectionist generalisation. Neural Programmer Interpreters is a very early step toward this. NPI can do symbolic operations such as sorting and addition, but it can also plan by taking images as input and it's able to generalise the plans to different images (e.g. in the NPI car example, the cars are test set cars not seen before)."


---- 
#### deductive vs inductive reasoning

  ["Probability Theory - The Logic of Science"](https://goo.gl/zqDwOF) by E. T. Jaynes

  >  reasoning is deductive when enough information is at hand to permit it and inductive/plausible when necessary information is not available

  >  rules for deductive reasoning:  
  >  (1)  (if A is true, then B is true), (A is true) |- (B is true)  
  >  (2)  (if A is true, then B is true), (B is false) |- (A is false)  

  >  rules for inductive/plausible reasoning:  
  >  (3)  (if A is true, then B is true), (B is true) |- (A becomes more plausible)  
  >  (4)  (if A is true, then B is true), (A is false) |- (B becomes less plausible)  
  >  (5)  (if A is true, then B becomes more plausible), (B is true) |- (A becomes more plausible)  
 
  >  The reasoning of a scientist, by which he accepts or rejects his theories, consists almost entirely of syllogisms of the kinds (2) and (3).

  >  Evidently, the deductive reasoning described above has the property that we can go through long chains of reasoning of the type (1) and (2) and the conclusions have just as much certainty as the premises. With the other kinds of reasoning, (3)–(5), the reliability of the conclusion changes as we go through several stages. But in their quantitative form we shall find that in many cases our conclusions can still approach the certainty of deductive reasoning. Pólya showed that even a pure mathematician actually uses these weaker forms of reasoning most of the time. Of course, on publishing a new theorem, the mathematician will try very hard to invent an argument which uses only the first kind; but the reasoning process which led to the theorem in the first place almost always involves one of the weaker forms (based, for example, on
following up conjectures suggested by analogies). Good mathematicians see analogies between theorems; great mathematicians see analogies between analogies.

 
  - *type of inference*:  
	deduction - specialization/derivation  
    	induction - generalization/prediction  
  - *framework*:  
    	deduction - logic axioms  
    	induction - probability theory  
  - *assumptions*:  
    	deduction - non-logical axioms  
    	induction - prior  
  - *inference rule*:  
    	deduction - Modus Ponens  
    	induction - Bayes rule  
  - *results*:  
    	deduction - theorems  
    	induction - posterior  
  - *universal scheme*:  
    	deduction - Zermelo-Fraenkel set theory  
    	induction - Solomonoff probability distribution  
  - *universal inference*:  
    	deduction - universal theorem prover  
    	induction - universal induction  
  - *limitation*:  
    	deduction - incomplete (Goedel)  
    	induction - incomputable  
  - *in practice*:  
    	deduction - semi-formal proofs  
    	induction - approximations  
  - *operation*:  
    	deduction - proof  
    	induction - computation  



---
### decisions/actions

  - [Reinforcement Learning](#reinforcement-learning)
  - [Meta-learning](#meta-learning)
  - [Artificial Curiosity and Creativity](#artificial-curiosity-and-creativity)
  - [Solomonoff Induction and AIXI](#solomonoff-induction-and-aixi)



---
### Reinforcement Learning

  [Reinforcement Learning](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md)

  "Learning in Brains and Machines" by Shakir Mohamed:  
	["Temporal Differences"](http://blog.shakirm.com/2016/02/learning-in-brains-and-machines-1/)  
	["The Dogma of Sparsity"](http://blog.shakirm.com/2016/04/learning-in-brains-and-machines-2/)  
	["Synergistic and Modular Action"](http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-3-synergistic-and-modular-action/)  
	["Episodic and Interactive Memory"](http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-4-episodic-and-interactive-memory/)  



---
### Meta-learning

  (Juergen Schmidhuber) "Current commercial AI algorithms are still missing something fundamental. They are no self-referential general purpose learning algorithms. They improve some system’s performance in a given limited domain, but they are unable to inspect and improve their own learning algorithm. They do not learn the way they learn, and the way they learn the way they learn, and so on (limited only by the fundamental limits of computability)."

----

  ["Why Tool AIs Want to Be Agent AIs"](http://www.gwern.net/Tool%20AI) by Gwern Branwen:

  "Roughly, we can try to categorize the different kinds of agentiness by the level of the neural network they work on. There are:  
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

  *(Gwern Branwen)*

----

  [overview](http://scholarpedia.org/article/Metalearning) by Tom Schaul and Juergen Schmidhuber


  ["Learning to Learn, to Program, to Explore and to Seek Knowledge"](https://youtube.com/watch?v=SAcHyzMdbXc) by Nando de Freitas  
  ["The Future of Machines that Learn Algorithms"](https://youtube.com/watch?v=zSNkbhgMkzQ) panel with Juergen Schmidhuber, Ilya Sutskever, Alex Graves  


  [interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#meta-learning)

----

  [Learning to Learn](http://people.idsia.ch/~juergen/metalearner.html) by Juergen Schmidhuber

  [Goedel Machine](https://en.wikipedia.org/wiki/G%C3%B6del_machine) by Juergen Schmidhuber

  [Goedel Machine vs AIXI](http://people.idsia.ch/~juergen/gmweb2/node21.html) by Juergen Schmidhuber

  ["Towards An Actual Goedel Machine Implementation: A Lesson in Self-Reflective Systems"](http://people.idsia.ch/~juergen/selfreflection.pdf) by Steunebrink and Schmidhuber


  ["Learning how to Learn Learning Algorithms: Recursive Self-Improvement"](https://youtube.com/watch?v=nqiUFc52g78) by Juergen Schmidhuber


  "Self-improving universal methods have also been defined, including some that justify self-changes (including changes of the learning algorithm) through empirical evidence in a lifelong learning context and the Goedel Machine that self-improves via proving theorems about itself, and can improve any part of its software (including the learning algorithm itself) in a way that is provably time-optimal in a sense that takes constant overheads into account and goes beyond asymptotic optimality. At each step of the way, the Goedel Machine takes the action that it can prove, according to its axiom system and its perceptual data, will be the best way to achieve its goals. The current versions of the Goedel Machine are not computationally tractable in complex environments, however."

  "Goedel machines are limited by the basic limits of math and computation identified by the founder of modern theoretical computer science himself, Kurt Goedel: some theorems are true but cannot be proven by any computational theorem proving procedure (unless the axiomatic system itself is flawed). That is, in some situations the GM may never find a proof of the benefits of some change to its own code."

  "It’s currently not sufficiently formalized, so it’s difficult to state if and how it really works. Searching for proofs is extremely complicated: Making a parallel with Levin search, where given a goal output string (an improvement in the Goedel Machine), you enumerate programs (propositions in the GM) and run them to see if they output the goal string (search for a proof of improvement in GM). This last part is the problem: in LS, the programs are fast to run, whereas in GM there is an additional search step for each proposition, so this looks very roughly like going from exponential (LS) to double-exponential (GM). And LS is already not really practical. Theorem proving is even more complicated when you need to prove that there will be an improvement of the system at an unknown future step. Maybe it would work better if the kinds of proofs were limited to some class, for example use simulation of the future steps up to some horizon given a model of the world. These kinds of proofs are easier to check and have a guaranteed termination, e.g. if the model class for the environment is based on Schmidhuber’s Speed Prior. But this starts to look pretty much like an approximation of AIXI."

  "We looked into existing proof verification systems. Some of them may turn out to be useful for limited AI applications. Unfortunately, however, off-the-shelf verifiers make implicit assumptions that are broken in self-referential proof-based Goedel Machines. They assume that (1) the thing being reasoned about is static (not an active and running program) and (2) the thing being reasoned about does not contain the reasoner itself."

  "MC-AIXI is a probabilistic approximation of AIXI. What might be the equivalent for the self-referential proof searcher of a GM? One possibility comes to mind: Holographic proofs, where errors in the derivation of a theorem are 'apparent after checking just a negligible fraction of bits of the proof'."

  "A Goedel Machine may indeed change its utility function and target theorem, but not in some arbitrary way. It can do so only if the change is provably useful according to its initial utility function. E.g., it may be useful to replace some complex-looking utility function by an equivalent simpler one. In certain environments, a Goedel Machine may even prove the usefulness of deleting its own proof searcher, and stop proving utility-related theorems, e.g., when the expected computational costs of proof search exceed the expected reward."

  *(Juergen Schmidhuber)*



---
### Artificial Curiosity and Creativity

  ["Formal Theory of Creativity and Fun and Intrinsic Motivation"](http://people.idsia.ch/~juergen/creativity.html) by Juergen Schmidhuber

  Intrinsic motivation objective for an artificially curious/creative agent is to maximize the amount of model-learning progress, measured as the difference in compression of agent's experience before and after learning.

  "What experiments should an agent’s reinforcement learning controller, C, conduct to generate data that quickly improves agent's adaptive, predictive world model, M, which in turn can help to plan ahead? The theory says: use the learning progress of M (typically compression progress and speed-ups) as the intrinsic reward or fun for C. This motivates C to create action sequences (experiments) such that M can quickly discover new, previously unknown regularities."

----

  "Humans, even as infants, invent their own tasks in a curious and creative fashion, continually increasing their problem solving repertoire even without an external reward or teacher. They seem to get intrinsic reward for creating experiments leading to observations that obey a previously unknown law that allows for better compression of the observations—corresponding to the discovery of a temporarily interesting, subjectively novel regularity. For example, a video of 100 falling apples can be greatly compressed via predictive coding once the law of gravity is discovered. Likewise, the video-like image sequence perceived while moving through an office can be greatly compressed by constructing an internal 3D model of the office space. The 3D model allows for re-computing the entire high-resolution video from a compact sequence of very low-dimensional eye coordinates and eye directions. The model itself can be specified by far fewer bits of information than needed to store the raw pixel data of a long video. Even if the 3D model is not precise, only relatively few extra bits will be required to encode the observed deviations from the predictions of the model. Even mirror neurons are easily explained as by-products of history compression. They fire both when an animal acts and when the animal observes the same action performed by another. Due to mutual algorithmic information shared by perceptions of similar actions performed by various animals, efficient predictive coding profits from using the same feature detectors (neurons) to encode the shared information, thus saving storage space."

  "In the real world external rewards are rare. Agents using additional intrinsic rewards for model-learning progress will be motivated to learn many useful behaviors even in absence of external rewards, behaviors that lead to predictable or compressible results and thus reflect regularities in the environment, such as repeatable patterns in the world’s reactions to certain action sequences. Often a bias towards exploring previously unknown environmental regularities through artificial curiosity/creativity is a priori desirable because goal-directed learning may greatly profit from it, as behaviors leading to external reward may often be rather easy to compose from previously learnt curiosity-driven behaviors. It may be possible to formally quantify this bias towards novel patterns in form of a mixture-based prior, a weighted sum of probability distributions on sequences of actions and resulting inputs, and derive precise conditions for improved expected external reward intake. Intrinsic reward may be viewed as analogous to a regularizer in supervised learning, where the prior distribution on possible hypotheses greatly influences the most probable interpretation of the data in a Bayesian framework."

  *(Juergen Schmidhuber)*

----

  overview by Juergen Schmidhuber:  
	<https://vimeo.com/7441291#t=7m12s>  
	<http://videolectures.net/ecmlpkdd2010_schmidhuber_ftf/>  
	<https://archive.org/details/Redwood_Center_2014_08_15_Jurgen_Schmidhuber>  
	<https://vimeo.com/28759091>  

  ["Driven by compression progress: A simple principle explains essential aspects of subjective beauty, novelty, surprise, interestingness, attention, curiosity, creativity, art, science, music, jokes"](http://arxiv.org/abs/0812.4360) by Juergen Schmidhuber
>	"I argue that data becomes temporarily interesting by itself to some self-improving, but computationally limited, subjective observer once he learns to predict or compress the data in a better way, thus making it subjectively simpler and more beautiful. Curiosity is the desire to create or discover more non-random, non-arbitrary, regular data that is novel and surprising not in the traditional sense of Boltzmann and Shannon but in the sense that it allows for compression progress because its regularity was not yet known. This drive maximizes interestingness, the first derivative of subjective beauty or compressibility, that is, the steepness of the learning curve. It motivates exploring infants, pure mathematicians, composers, artists, dancers, comedians, yourself, and artificial systems."

  ["Formal Theory of Creativity, Fun, and Intrinsic Motivation"](http://people.idsia.ch/~juergen/ieeecreative.pdf) by Juergen Schmidhuber
>	"The simple but general formal theory of fun & intrinsic motivation & creativity is based on the concept of maximizing intrinsic reward for the active creation or discovery of novel, surprising patterns allowing for improved prediction or data compression. It generalizes the traditional field of active learning, and is related to old but less formal ideas in aesthetics theory and developmental psychology. It has been argued that the theory explains many essential aspects of intelligence including autonomous development, science, art, music, humor. This overview first describes theoretically optimal (but not necessarily practical) ways of implementing the basic computational principles on exploratory, intrinsically motivated agents or robots, encouraging them to provoke event sequences exhibiting previously unknown but learnable algorithmic regularities."

  ["On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models"](http://arxiv.org/abs/1511.09249) by Juergen Schmidhuber
>	"Humans, even as infants, invent their own tasks in a curious and creative fashion, continually increasing their problem solving repertoire even without an external reward or teacher. They seem to get intrinsic reward for creating experiments leading to observations that obey a previously unknown law that allows for better compression of the observations - corresponding to the discovery of a temporarily interesting, subjectively novel regularity. For example, a video of 100 falling apples can be greatly compressed via predictive coding once the law of gravity is discovered. Likewise, the video-like image sequence perceived while moving through an office can be greatly compressed by constructing an internal 3D model of the office space. The 3D model allows for re-computing the entire high-resolution video from a compact sequence of very low-dimensional eye coordinates and eye directions. The model itself can be specified by far fewer bits of information than needed to store the raw pixel data of a long video. Even if the 3D model is not precise, only relatively few extra bits will be required to encode the observed deviations from the predictions of the model."

>	"Even mirror neurons are easily explained as by-products of history compression. They fire both when an animal acts and when the animal observes the same action performed by another. Due to mutual algorithmic information shared by perceptions of similar actions performed by various animals, efficient RNN-based predictive coding profits from using the same feature detectors (neurons) to encode the shared information, thus saving storage space."

>	"Given combination of controller and environment model, we motivate controller to become an efficient explorer and an artificial scientist, by adding to its standard external reward (or fitness) for solving user-given tasks another intrinsic reward for generating novel action sequences (or experiments) that allow environment model to improve its compression performance on the resulting data."

----

  **approximations**

  ["VIME: Variational Information Maximizing Exploration"](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#houthooft-chen-duan-schulman-turck-abbeel---vime-variational-information-maximizing-exploration) by Houthooft et al.  
>	"Variational inference is used to approximate the posterior distribution of a Bayesian neural network that represents the environment dynamics. Using information gain in this learned dynamics model as intrinsic rewards allows the agent to optimize for both external reward and intrinsic surprise simultaneously."  
>	"r'(st,at,st+1) = r(st,at) + μ * Dkl[p(θ|ξt,at,st+1)||p(θ|ξt)]"  

>	"It is possible to derive an interesting relationship between compression improvement - an intrinsic reward objective defined in Schmidhuber's Artificial Curiosity and Creativity theory, and the information gain. The agent’s curiosity is equated with compression improvement, measured through C(ξt; φt-1) - C(ξt; φt), where C(ξ; φ) is the description length of ξ using φ as a model. Furthermore, it is known that the negative variational lower bound can be viewed as the description length. Hence, we can write compression improvement as L[q(θ; φt), ξt] - L[q(θ; φt-1), ξt]. In addition, due to alternative formulation of the variational lower bound, compression improvement can be written as (log p(ξt) - Dkl[q(θ; φt)||p(θ|ξt)]) - (log p(ξt) - Dkl[q(θ; φt-1)||p(θ|ξt)]). If we assume that φt perfectly optimizes the variational lower bound for the history ξt, then Dkl[q(θ; φt)||p(θ|ξt)] = 0, which occurs when the approximation equals the true posterior, i.e., q(θ; φt) = p(θ|ξt). Hence, compression improvement becomes Dkl[p(θ|ξt-1) || p(θ|ξt)]. Therefore, optimizing for compression improvement comes down to optimizing the KL divergence from the posterior given the past history ξt-1 to the posterior given the total history ξt. As such, we arrive at an alternative way to encode curiosity than information gain, namely Dkl[p(θ|ξt)||p(θ|ξt,at,st+1)], its reversed KL divergence. In experiments, we noticed no significant difference between the two KL divergence variants. This can be explained as both variants are locally equal when introducing small changes to the parameter distributions."  

  - [demo](https://goo.gl/fyxLvI)  
  - [overview](https://youtube.com/watch?v=nbbMSMv3v5k)  

  ["Automated Curriculum Learning for Neural Networks"](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md##graves-bellemare-menick-munos-kavukcuoglu---automated-curriculum-learning-for-neural-networks) by Graves et al.  
>	"We focus on variants of prediction gain, and also introduce a novel class of progress signals which we refer to as complexity gain. Derived from minimum description length principles, complexity gain equates acquisition of knowledge with an increase in effective information encoded in the network weights."  

>	"VIME uses a reward signal that is closely related to variational complexity gain. The difference is that while VIME measures the KL between the posterior before and after a step in parameter space, we consider the change in KL between the posterior and prior induced by the step. Therefore, while VIME looks for any change to the posterior, we focus only on changes that alter the divergence from the prior. Further research will be needed to assess the relative merits of the two signals."  

----

  **applications**

  ["Toward Intelligent Humanoids"](https://vimeo.com/51011081) demo by Schmidhuber's group

  ["Curiosity Driven Reinforcement Learning Motion Planning on Humanoids"](http://goo.gl/FKNPth) by Frank, Schmidhuber and others  
  ["PowerPlay: Training an Increasingly General Problem Solver by Continually Searching for the Simplest Still Unsolvable Problem"](https://arxiv.org/abs/1112.5309) by Schmidhuber  
  ["Continually Adding Self-Invented Problems to the Repertoire: First Experiments with PowerPlay"](http://goo.gl/lTYm6o) by Srivastava, Schmidhuber and others  

----

  [interesting papers](#interesting-papers---artificial-curiosity-and-creativity)

  [exploration and intrinsic motivation](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#exploration-and-intrinsic-motivation)



---
### Solomonoff Induction and AIXI

  [introduction to Solomonoff Induction](http://lesswrong.com/lw/dhg/an_intuitive_explanation_of_solomonoff_induction/) by Luke Muehlhauser and Alex Altair  
  [Algorithmic Probability](http://scholarpedia.org/article/Algorithmic_probability)  
  ["How Bayes theorem is consistent with Solomonoff induction"](http://lesswrong.com/r/discussion/lw/di3/how_bayes_theorem_is_consistent_with_solomonoff/)  

  [introduction to AIXI](http://youtube.com/watch?v=F2bQ5TSB-cE) by Marcus Hutter  
  [introduction to AIXI](http://jan.leike.name/AIXI.html) by Jan Leike  
  [General Reinforcement Learning Agent Zoo](http://aslanides.io/aixijs/) by John Aslanides
	([demo](http://aslanides.io/aixijs/demo.html), [code](https://github.com/aslanides/aixijs), [paper](https://arxiv.org/abs/1705.07615))

  [interesting papers](#solomonoff-induction-and-aixi)

----

  "AI systems have to learn from experience, build models of the environment from the acquired knowledge, and use these models for prediction (and action). In philosophy this is called inductive inference, in statistics it is called estimation and prediction, and in computer science it is addressed by machine learning. There were several unsuccessful attempts and unsuitable approaches towards a general theory of uncertainty and induction, including Popper’s denial of induction, frequentist statistics, much of statistical learning theory, subjective Bayesianism, Carnap’s confirmation theory, the big data paradigm, eliminative induction, pluralism, and deductive and other approaches. I argue that Solomonoff’s formal, general, complete, consistent, and essentially unique theory provably solves most issues that have plagued other approaches."

  "Sequential decision theory formally solves the problem of rational agents in uncertain worlds if the true environmental prior probability distribution is known. Solomonoff's theory of universal induction formally solves the problem of sequence prediction for unknown prior distribution. AIXI combines both ideas and develop an elegant parameter-free theory of an optimal reinforcement learning agent embedded in an arbitrary unknown environment that possesses essentially all aspects of rational intelligence. The theory reduces all conceptual AI problems to pure computational ones. There are strong arguments that the resulting AIXI model is the most intelligent unbiased agent possible."

  "No free lunch myth relies on unrealistic uniform sampling (which results mostly in white noise for real-world phenomena). Sampling from Solomonoff's universal distribution permits free lunch."

  "AIXI applications (environments):  
  - *sequence prediction*:  predict weather or stock market (strong result - upper bound for approximation of true distribution)  
  - *strategic games*:  learn play well (minimax) zero-sum games (like chess) or even exploit limited capabilities of opponent  
  - *optimization*:  find (approximate) minimum of function with as few function calls as possible (difficult exploration versus exploitation problem)  
  - *supervised learning*:  learn functions by presenting (z, f(z)) pairs and ask for function values of z' presenting (z', ?) pairs (supervised learning is much faster than reinforcement learning)"  

  *(Marcus Hutter)*

----

  AIXI = learning + prediction + planning + acting

  AIXI is an algorithmic probability theory extended to the case of sequential decision theory.  
  AIXI defines universal agent which effectively simulates all possible programs which output agrees with agent’s set of observations.  
  AIXI is not computable but related agent AIXI-tl may be computed and is superior to any other agent bounded by time t and space l.  

  components of AIXI:  
  - Ockhams' razor (simplicity) principle  
  - Epicurus' principle of multiple explanations  
  - Bayes' rule for conditional probabilities  
  - Turing's universal machine  
  - Kolmogorov's complexity  
  - Solomonoff's universal prior distribution = Ockham + Epicurus + Bayes + Turing + Kolmogorov  
  - Bellman equations + Solomonoff = Universal Artificial Intelligence  


  "AIXI operates within the following agent model: There is an agent, and an environment, which is a computable function unknown to the agent. Thus the agent will need to have a probability distribution on the range of possible environments. On each clock tick, the agent receives an observation (a bitstring/number) from the environment, as well as a reward (another number). The agent then outputs an action (another number). To do this, AIXI guesses at a probability distribution for its environment, using Solomonoff induction, a formalization of Occam's razor: Simpler computations are more likely a priori to describe the environment than more complex ones. This probability distribution is then Bayes-updated by how well each model fits the evidence (or more precisely, by throwing out all computations which have not exactly fit the environmental data so far, but for technical reasons this is roughly equivalent as a model). AIXI then calculates the expected reward of each action it might choose--weighting the likelihood of possible environments as mentioned. It chooses the best action by extrapolating its actions into its future time horizon recursively, using the assumption that at each step into the future it will again choose the best possible action using the same procedure. Then, on each iteration, the environment provides an observation and reward as a function of the full history of the interaction; the agent likewise is choosing its action as a function of the full history. The agent's intelligence is defined by its expected reward across all environments, weighting their likelihood by their complexity."

  ![AIXI formula](http://www.hutter1.net/ai/aixi1linel.gif)

  "AIXI is an agent that interacts with an environment in cycles k=1,2,...,m. In cycle k, AIXI takes action ak (e.g. a limb movement) based on past perceptions o1 r1...ok-1 rk-1. Thereafter, the environment provides a (regular) observation ok (e.g. a camera image) to AIXI and a real-valued reward rk. The reward can be very scarce, e.g. just +1 (-1) for winning (losing) a chess game, and 0 at all other times. Then the next cycle k+1 starts. The expression shows that AIXI tries to maximize its total future reward rk+...+rm. If the environment is modeled by a deterministic program q, then the future perceptions ...okrk...omrm = U(q,a1..am) can be computed, where U is a universal (monotone Turing) machine executing q given a1..am. Since q is unknown, AIXI has to maximize its expected reward, i.e. average rk+...+rm over all possible future perceptions created by all possible environments q that are consistent with past perceptions. The simpler an environment, the higher is its a-priori contribution 2^-l(q), where simplicity is measured by the length l of program q. AIXI effectively learns by eliminating Turing machines q once they become inconsistent with the progressing history. Since noisy environments are just mixtures of deterministic environments, they are automatically included. The sums in the formula constitute the averaging process. Averaging and maximization have to be performed in chronological order, hence the interleaving of max and Σ (similarly to minimax for games). One can fix any finite action and perception space, any reasonable U, and any large finite lifetime m. This completely and uniquely defines AIXI's actions ak, which are limit-computable via the expression above (all quantities are known)."

----

  overview by Marcus Hutter  
	<http://youtube.com/watch?v=vUUeHZJFN2Q>  
	<http://vimeo.com/14888930>  
	<http://youtube.com/watch?v=gb4oXRsw3yA>  

  tutorial by Marcus Hutter  
	<http://videolectures.net/ssll09_hutter_uai/>  
	<http://videolectures.net/mlss08au_hutter_fund/>  

  tutorial by Tom Everitt  
	<https://youtube.com/watch?v=BP7vhBaBDyk>  

  overview by Alexey Potapov (in russian)  
	<http://geektimes.ru/post/148002/>  
	<http://geektimes.ru/post/150056/>  
	<http://geektimes.ru/post/150902/>  
	<http://geektimes.ru/post/151838/>  

  [book about AIXI](http://hutter1.net/ai/uaibook.htm) by Marcus Hutter  

----

  **limitations**
  
  "AIXI it is not a feasible AI, because Solomonoff induction is not computable."

  "AIXI only works over some finite time horizon, though any finite horizon can be chosen. But some environments may not interact over finite time horizons."

  "AIXI lacks a self-model. It extrapolates its own actions into the future indefinitely, on the assumption that it will keep working in the same way in the future. Though AIXI is an abstraction, any real AI would have a physical embodiment that could be damaged, and an implementation which could change its behavior due to bugs; and the AIXI formalism completely ignores these possibilities."

  "Solomonoff induction treats the world as a sort of qualia factory, a complicated mechanism that outputs experiences for the inductor. Its hypothesis space tacitly assumes a Cartesian barrier separating the inductor's cognition from the hypothesized programs generating the perceptions. Through that barrier, only sensory bits and action bits can pass. Real agents, on the other hand, will be in the world they're trying to learn about. A computable approximation of AIXI, like AIXI-tl, would be a physical object. Its environment would affect it in unseen and sometimes drastic ways; and it would have involuntary effects on its environment, and on itself. Solomonoff induction doesn't appear to be a viable conceptual foundation for artificial intelligence - not because it's an uncomputable idealization, but because it's Cartesian."

----

  **approximations**

  "Even without considering AIXI approximations, AIXI still is very important for AGI research because it unifies all important properties of cognition, like agency (interaction with an environment), knowledge representation and memory, understanding, reasoning, goals, problem solving, planning and action selection, abstraction, generalization without overfitting, multiple hypotheses, creativity, exploration and curiosity, optimization and utility maximization, prediction, uncertainty, with incremental, on-line, lifelong, continual learning in arbitrarily complex environments, without a restart state, no i.i.d. or stationarity assumption, etc. and does all this in a very simple, elegant and precise manner."

  *(Laurent Orseau)*


#### A Monte-Carlo AIXI Approximation (MC-AIXI-CTW)

  ["A Monte Carlo AIXI Approximation"](https://arxiv.org/abs/0909.0801) by Veness, Ng, Hutter, Uther, Silver  
  ["Approximate Universal Artificial Intelligence and Self-Play Learning for Games"](http://jveness.info/publications/veness_phd_thesis_final.pdf) by Joel Veness  

  [Ms. Pac-Man demo](http://youtube.com/watch?v=yfsMHtmGDKE)


#### Formal Theory of Fun and Creativity and Intrinsic Motivation

  ["Artificial Curiosity and Creativity"](#artificial-curiosity-and-creativity) by Juergen Schmidhuber

  "The ultimate optimal Bayesian approach to machine learning is embodied by the AIXI model. Any computational problem can be phrased as the maximization of a reward function. AIXI is based on Solomonoff's universal mixture M of all computable probability distributions. If the probabilities of the world's responses to some reinforcement learning agent's actions are computable (there is no physical evidence against that), then the agent may predict its future sensory inputs and rewards using M instead of the true but unknown distribution. The agent can indeed act optimally by choosing those action sequences that maximize M-predicted reward. This may be dubbed the unbeatable, ultimate statistical approach to AI - it demonstrates the mathematical limits of what's possible. However, AIXI’s notion of optimality ignores computation time, which is the reason why we are still in business with less universal but more practically feasible approaches such as deep learning based on more limited local search techniques such as gradient descent."

  "Generally speaking, when it comes to Reinforcement Learning, it is indeed a good idea to train a recurrent neural network called M to become a predictive model of the world, and use M to train a separate controller network C which is supposed to generate reward-maximising action sequences. Marcus Hutter’s mathematically optimal universal AIXI also has a predictive world model M, and a controller C that uses M to maximise expected reward. Ignoring limited storage size, RNNs are general computers just like your laptop. That is, AIXI’s M is related to the RNN-based M above in the sense that both consider a very general space of predictive programs. AIXI’s M, however, really looks at all those programs simultaneously, while the RNN-based M uses a limited local search method such as gradient descent in program space (also known as backpropagation through time) to find a single reasonable predictive program (an RNN weight matrix). AIXI’s C always picks the action that starts the action sequence that yields maximal predicted reward, given the current M, which in a Bayes-optimal way reflects all the observations so far. The RNN-based C, however, uses a local search method (backpropagation through time) to optimise its program or weight matrix, using gradients derived from M. So in a way, my old RNN-based CM system of 1990 may be viewed as a limited, downscaled, sub-optimal, but at least computationally feasible approximation of AIXI."

  *(Juergen Schmidhuber)*

  ["On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models"](http://arxiv.org/abs/1511.09249) by Juergen Schmidhuber



---
### interesting papers


[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md)


interesting papers:  
  - [knowledge representation and reasoning](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers)  
  - [machine learning](https://github.com/brylevkirill/notes/blob/master/Machine%20Learning.md#interesting-papers)  
  - [deep learning](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#interesting-papers)  
  - [reinforcement learning](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers)  
  - [bayesian inference and learning](https://dropbox.com/s/7vlg0vhb51rd6c1/Bayesian%20Inference%20and%20Learning.txt#interesting-papers)  
  - [probabilistic programming](https://github.com/brylevkirill/notes/blob/master/Probabilistic%20Programming.md#interesting-papers)  


interesting papers (see below):  
  - [definitions and measures of intelligence](#interesting-papers---definitions-and-measures-of-intelligence)  
  - [Solomonoff Induction and AIXI](#interesting-papers---solomonoff-induction-and-aixi)  
  - [Artificial Curiosity and Creativity](#interesting-papers---artificial-curiosity-and-creativity)  



---
### interesting papers - definitions and measures of intelligence


#### Lake, Ullman, Tenenbaum, Gershman - ["Building Machines That Learn and Think Like People"](http://arxiv.org/abs/1604.00289)
>	"Recent progress in artificial intelligence has renewed interest in building systems that learn and think like people. Many advances have come from using deep neural networks trained end-to-end in tasks such as object recognition, video games, and board games, achieving performance that equals or even beats humans in some respects. Despite their biological inspiration and performance achievements, these systems differ from human intelligence in crucial ways. We review progress in cognitive science suggesting that truly human-like learning and thinking machines will have to reach beyond current engineering trends in both what they learn, and how they learn it. Specifically, we argue that these machines should (a) build causal models of the world that support explanation and understanding, rather than merely solving pattern recognition problems; (b) ground learning in intuitive theories of physics and psychology, to support and enrich the knowledge that is learned; and (c) harness compositionality and learning-to-learn to rapidly acquire and generalize knowledge to new tasks and situations. We suggest concrete challenges and promising routes towards these goals that can combine the strengths of recent neural network advances with more structured cognitive models."

  - <http://pemami4911.github.io/paper-summaries/2016/05/13/learning-to-think.html>


#### Legg, Hutter - ["Universal Intelligence: A Definition of Machine Intelligence"](http://arxiv.org/abs/0712.3329)
>	"A fundamental problem in artificial intelligence is that nobody really knows what intelligence is. The problem is especially acute when we need to consider artificial systems which are significantly different to humans. In this paper we approach this problem in the following way: We take a number of well known informal definitions of human intelligence that have been given by experts, and extract their essential features. These are then mathematically formalised to produce a general measure of intelligence for arbitrary machines. We believe that this equation formally captures the concept of machine intelligence in the broadest reasonable sense. We then show how this formal definition is related to the theory of universal optimal learning agents. Finally, we survey the many other tests and definitions of intelligence that have been proposed for machines."


#### Orseau, Ring - ["Space-Time Embedded Intelligence"](http://frontiersinai.com/turingfiles/December/orseau.pdf)
>	"This paper presents the first formal measure of intelligence for agents fully embedded within their environment. Whereas previous measures such as Legg’s universal intelligence measure and Russell’s bounded optimality provide theoretical insights into agents that interact with an external world, ours describes an intelligence that is computed by, can be modified by, and is subject to the time and space constraints of the environment with which it interacts. Our measure merges and goes beyond Legg’s and Russell’s, leading to a new, more realistic definition of artificial intelligence that we call Space-Time Embedded Intelligence."


#### Schaul, Togelius, Schmidhuber - ["Measuring Intelligence through Games"](http://arxiv.org/abs/1109.1314)
>	"Artificial general intelligence refers to research aimed at tackling the full problem of artificial intelligence, that is, create truly intelligent agents. This sets it apart from most AI research which aims at solving relatively narrow domains, such as character recognition, motion planning, or increasing player satisfaction in games. But how do we know when an agent is truly intelligent? A common point of reference in the AGI community is Legg and Hutter’s formal definition of universal intelligence, which has the appeal of simplicity and generality but is unfortunately incomputable. Games of various kinds are commonly used as benchmarks for “narrow” AI research, as they are considered to have many important properties. We argue that many of these properties carry over to the testing of general intelligence as well. We then sketch how such testing could practically be carried out. The central part of this sketch is an extension of universal intelligence to deal with finite time, and the use of sampling of the space of games expressed in a suitably biased game description language."


#### Rocki - ["Thinking Required"](http://arxiv.org/abs/1512.01926)
>	"There exists a theory of a single general-purpose learning algorithm which could explain the principles of its operation. It assumes the initial rough architecture, a small library of simple innate circuits which are prewired at birth and proposes that all significant mental algorithms are learned. Given current understanding and observations, this paper reviews and lists the ingredients of such an algorithm from architectural and functional perspectives."



---
### interesting papers - Solomonoff Induction and AIXI


#### Hutter - ["On Universal Prediction and Bayesian Confirmation"](http://arxiv.org/abs/0709.1516)
>	"The Bayesian framework is a well-studied and successful framework for inductive reasoning, which includes hypothesis testing and confirmation, parameter estimation, sequence prediction, classification, and regression. But standard statistical guidelines for choosing the model class and prior are not always available or fail, in particular in complex situations. Solomonoff completed the Bayesian framework by providing a rigorous, unique, formal, and universal choice for the model class and the prior. We discuss in breadth how and in which sense universal (non-i.i.d.) sequence prediction solves various (philosophical) problems of traditional Bayesian sequence prediction. We show that Solomonoff's model possesses many desirable properties: Strong total and weak instantaneous bounds, and in contrast to most classical continuous prior densities has no zero p(oste)rior problem, i.e. can confirm universal hypotheses, is reparametrization and regrouping invariant, and avoids the old-evidence and updating problem. It even performs well (actually better) in non-computable environments."

>	"The goal of the paper was to establish a single, universal theory for (sequence) prediction and (hypothesis) confirmation, applicable to all inductive inference problems. I started by showing that Bayesian prediction is consistent for any countable model class, provided it contains the true distribution. The major (agonizing) problem Bayesian statistics leaves open is how to choose the model class and the prior. Solomonoff’s theory fills this gap by choosing the class of all computable (stochastic) models, and a universal prior inspired by Ockham and Epicurus, and quantified by Kolmogorov complexity. I discussed in breadth how and in which sense this theory solves the inductive inference problem, by studying a plethora of problems other approaches suffer from. In one line: All you need for universal prediction is Ockham, Epicurus, Bayes, Solomonoff, Kolmogorov, and Turing. By including Bellman, one can extend this theory to universal decisions in reactive environments."


#### Hutter - ["Universal Algorithmic Intelligence: A Mathematical Top-down Approach"](http://arxiv.org/abs/cs/0701125) (AIXI agent)
>	"Sequential decision theory formally solves the problem of rational agents in uncertain worlds if the true environmental prior probability distribution is known. Solomonoff’s theory of universal induction formally solves the problem of sequence prediction for unknown prior distribution. We combine both ideas and get a parameter-free theory of universal Artificial Intelligence. We give strong arguments that the resulting AIXI model is the most intelligent unbiased agent possible. We outline how the AIXI model can formally solve a number of problem classes, including sequence prediction, strategic games, function minimization, reinforcement and supervised learning. The major drawback of the AIXI model is that it is uncomputable. To overcome this problem, we construct a modified algorithm AIXItl that is still effectively more intelligent than any other time t and length l bounded agent. The computation time of AIXItl is of the order t·2^l. The discussion includes formal definitions of intelligence order relations, the horizon problem and relations of the AIXI theory to other AI approaches."

  - ["A Theory of Universal Artificial Intelligence based on Algorithmic Complexity"](https://arxiv.org/abs/cs/0004001) by Hutter


#### Hutter - ["Open Problems in Universal Induction & Intelligence"](http://arxiv.org/abs/0907.0746)
>	"Specialized intelligent systems can be found everywhere: finger print, hand-writing, speech, and face recognition, spam filtering, chess and other game programs, robots, et al. This decade the first presumably complete mathematical theory of artificial intelligence based on universal induction-prediction-decision-action has been proposed. This information-theoretic approach solidifies the foundations of inductive inference and artificial intelligence. Getting the foundations right usually marks a significant progress and maturing of a field. The theory provides a gold standard and guidance for researchers working on intelligent algorithms. The roots of universal induction have been laid exactly half-a-century ago and the roots of universal intelligence exactly one decade ago. So it is timely to take stock of what has been achieved and what remains to be done. Since there are already good recent surveys, I describe the state-of-the-art only in passing and refer the reader to the literature. This article concentrates on the open problems in universal induction and its extension to universal intelligence."


#### Veness, Ng, Hutter, Uther, Silver - ["A Monte-Carlo AIXI Approximation"](https://www.jair.org/media/3125/live-3125-5397-jair.pdf) (MC-AIXI-CTW agent)
>	"This paper introduces a principled approach for the design of a scalable general reinforcement learning agent. Our approach is based on a direct approximation of AIXI, a Bayesian optimality notion for general reinforcement learning agents. Previously, it has been unclear whether the theory of AIXI could motivate the design of practical algorithms. We answer this hitherto open question in the affirmative, by providing the first computationally feasible approximation to the AIXI agent. To develop our approximation, we introduce a new Monte-Carlo Tree Search algorithm along with an agent-specific extension to the Context Tree Weighting algorithm. Empirically, we present a set of encouraging results on a variety of stochastic and partially observable domains. We conclude by proposing a number of directions for future research."

>	"This paper presents the first computationally feasible general reinforcement learning agent that directly and scalably approximates the AIXI ideal. Although well established theoretically, it has previously been unclear whether the AIXI theory could inspire the design of practical agent algorithms. Our work answers this question in the affirmative: empirically, our approximation achieves strong performance and theoretically, we can characterise the range of environments in which our agent is expected to perform well. To develop our approximation, we introduced two new algorithms: ρUCT, a Monte-Carlo expectimax approximation technique that can be used with any online Bayesian approach to the general reinforcement learning problem and FAC-CTW, a generalisation of the powerful CTW algorithm to the agent setting. In addition, we highlighted a number of interesting research directions that could improve the performance of our current agent; in particular, model class expansion and the online learning of heuristic rollout policies for ρUCT."

  - <http://youtube.com/watch?v=yfsMHtmGDKE> (Ms. Pac-Man demo)
  - <http://aslanides.io/aixijs/demo.html> (demo)
  - <http://videolectures.net/nips09_veness_bfg/> (Veness)
  - <http://jveness.info/publications/veness_phd_thesis_final.pdf>
  - <http://jveness.info/software/mcaixi_jair_2010.zip>
  - <https://github.com/moridinamael/mc-aixi>
  - <https://github.com/gkassel/pyaixi>
  - <https://github.com/GoodAI/SummerCamp/tree/master/AIXIModule>


#### Legg - ["Is there an Elegant Universal Theory of Prediction?"](http://arxiv.org/pdf/cs/0606070.pdf)
>	"Solomonoff induction is an elegant and extremely general model of inductive learning. It neatly brings together the philosophical principles of Occam’s razor, Epicurus’ principle of multiple explanations, Bayes theorem and Turing’s model of universal computation into a theoretical sequence predictor with astonishingly powerful properties. If theoretical models of prediction can have such elegance and power, one cannot help but wonder whether similarly beautiful and highly general computable theories of prediction are also possible.
>	What we have shown here is that there does not exist an elegant constructive theory of prediction for computable sequences, even if we assume unbounded computational resources, unbounded data and learning time, and place moderate bounds on the Kolmogorov complexity of the sequences to be predicted. Very powerful computable predictors are therefore necessarily complex. We have further shown that the source of this problem is computable sequences which are extremely expensive to compute. While we have proven that very powerful prediction algorithms which can learn to predict these sequences exist, we have also proven that, unfortunately, mathematical analysis cannot be used to discover these algorithms due to problems of Goedel incompleteness.
>	These results can be extended to more general settings, specifically to those problems which are equivalent to, or depend on, sequence prediction. Consider, for example, a reinforcement learning agent interacting with an environment. In each interaction cycle the agent must choose its actions so as to maximise the future rewards that it receives from the environment. Of course the agent cannot know for certain whether or not some action will lead to rewards in the future, thus it must predict these. Clearly, at the heart of reinforcement learning lies a prediction problem, and so the results for computable predictors presented in this paper also apply to computable reinforcement learners. More specifically, it follows that very powerful computable reinforcement learners are necessarily complex, and it follows that it is impossible to discover extremely powerful reinforcement learning algorithms mathematically."

  - <http://lo-tho.blogspot.ru/2012/08/truth-and-ai.html>


#### Leike, Hutter - ["On the Computability of AIXI"](http://arxiv.org/abs/1510.05572)
>	"How could we solve the machine learning and the artificial intelligence problem if we had infinite computation? Solomonoff induction and the reinforcement learning agent AIXI are proposed answers to this question. Both are known to be incomputable. In this paper, we quantify this using the arithmetical hierarchy, and prove upper and corresponding lower bounds for incomputability. We show that AIXI is not limit computable, thus it cannot be approximated using finite computation. Our main result is a limit-computable ε-optimal version of AIXI with infinite horizon that maximizes expected rewards."

  - ["On the Computability of Solomonoff Induction and Knowledge-Seeking"](http://arxiv.org/abs/1507.04124) by Leike and Hutter


#### Leike, Hutter - ["Bad Universal Priors and Notions of Optimality"](http://jmlr.csail.mit.edu/proceedings/papers/v40/Leike15.pdf)
>	"A big open question of algorithmic information theory is the choice of the universal Turing machine. For Kolmogorov complexity and Solomonoff induction we have invariance theorems: the choice of the UTM changes bounds only by a constant. For the universally intelligent agent AIXI no invariance theorem is known. Our results are entirely negative: we discuss cases in which unlucky or adversarial choices of the UTM cause AIXI to misbehave drastically. We show that Legg-Hutter intelligence and thus balanced Pareto optimality is entirely subjective, and that every policy is Pareto optimal in the class of all computable environments. This undermines all existing optimality properties for AIXI. While it may still serve as a gold standard for AI, our results imply that AIXI is a relative theory, dependent on the choice of the UTM."

>	"The choice of the universal Turing machine has been a big open question in algorithmic information theory for a long time. While attempts have been made no answer is in sight. The Kolmogorov complexity of a string, the length of the shortest program that prints this string, depends on this choice. However, there are invariance theorems which state that changing the UTM changes Kolmogorov complexity only by a constant. When using the universal prior M introduced by Solomonoff to predict any deterministic computable binary sequence, the number of wrong predictions is bounded by (a multiple of) the Kolmogorov complexity of the sequence. Due to the invariance theorem, changing the UTM changes the number of errors only by a constant. In this sense, compression and prediction work for any choice of UTM. Hutter defines the universally intelligent agent AIXI, which is targeted at the general reinforcement learning problem. It extends Solomonoff induction to the interactive setting. AIXI is a Bayesian agent, using a universal prior on the set of all computable environments; actions are taken according to the maximization of expected future discounted rewards. Closely related is the intelligence measure defined by Legg and Hutter, a mathematical performance measure for general reinforcement learning agents: defined as the discounted rewards achieved across all computable environments, weighted by the universal prior. There are several known positive results about AIXI. It has been proven to be Pareto optimal, balanced Pareto optimal, and has maximal Legg-Hutter intelligence. Furthermore, AIXI asymptotically learns to predict the environment perfectly and with a small total number of errors analogously to Solomonoff induction, but only on policy: AIXI learns to correctly predict the value (expected future rewards) of its own actions, but generally not the value of counterfactual actions that it does not take. Orseau showed that AIXI does not achieve asymptotic optimality in all computable environments. So instead, we may ask the following weaker questions. Does AIXI succeed in every partially observable Markov decision process/(ergodic) Markov decision process/bandit problem/sequence prediction task? In this paper we show that without further assumptions on the UTM, we cannot answer any of the preceding questions in the affirmative. More generally, there can be no invariance theorem for AIXI. As a reinforcement learning agent, AIXI has to balance between exploration and exploitation. Acting according to any (universal) prior does not lead to enough exploration, and the bias of AIXI’s prior is retained indefinitely. For bad priors this can cause serious malfunctions. However, this problem can be alleviated by adding an extra exploration component to AIXI, similar to knowledge-seeking agents, or by the use of optimism. We give two examples of universal priors that cause AIXI to misbehave drastically. In case of a finite lifetime, the indifference prior makes all actions equally preferable to AIXI. Furthermore, for any computable policy π the dogmatic prior makes AIXI stick to the policy π as long as expected future rewards do not fall too close to zero. This has profound implications. We show that if we measure Legg-Hutter intelligence with respect to a different universal prior, AIXI scores arbitrarily close to the minimal intelligence while any computable policy can score arbitrarily close to the maximal intelligence. This makes the Legg-Hutter intelligence score and thus balanced Pareto optimality relative to the choice of the UTM. Moreover, we show that in the class of all computable environments, every policy is Pareto optimal. This undermines all existing optimality results for AIXI. We discuss the implications of these results for the quest for a natural universal Turing machine and optimality notions of general reinforcement learners."

  - <http://aslanides.io/aixijs/demo.html> (demo)


#### Leike - ["Nonparametric General Reinforcement Learning"](https://jan.leike.name/publications/Nonparametric%20General%20Reinforcement%20Learning%20-%20Leike%202016.pdf)
>	"Reinforcement learning problems are often phrased in terms of Markov decision processes. In this thesis we go beyond MDPs and consider reinforcement learning in environments that are non-Markovian, non-ergodic and only partially observable. Our focus is not on practical algorithms, but rather on the fundamental underlying problems: How do we balance exploration and exploitation? How do we explore optimally? When is an agent optimal? We follow the nonparametric realizable paradigm: we assume the data is drawn from an unknown source that belongs to a known countable class of candidates.
>	First, we consider the passive (sequence prediction) setting, learning from data that is not independent and identically distributed. We collect results from artificial intelligence, algorithmic information theory, and game theory and put them in a reinforcement learning context: they demonstrate how agent can learn the value of its own policy. Next, we establish negative results on Bayesian reinforcement learning agents, in particular AIXI. We show that unlucky or adversarial choices of the prior cause the agent to misbehave drastically. Therefore Legg-Hutter intelligence and balanced Pareto optimality, which depend crucially on the choice of the prior, are entirely subjective. Moreover, in the class of all computable environments every policy is Pareto optimal. This undermines all existing optimality properties for AIXI.
>	However, there are Bayesian approaches to general reinforcement learning that satisfy objective optimality guarantees: We prove that Thompson sampling is asymptotically optimal in stochastic environments in the sense that its value converges to the value of the optimal policy. We connect asymptotic optimality to regret given a recoverability assumption on the environment that allows the agent to recover from mistakes. Hence Thompson sampling achieves sublinear regret in these environments.
>	AIXI is known to be incomputable. We quantify this using the arithmetical hierarchy, and establish upper and corresponding lower bounds for incomputability. Further, we show that AIXI is not limit computable, thus cannot be approximated using finite computation. However there are limit computable ε-optimal approximations to AIXI. We also derive computability bounds for knowledge-seeking agents, and give a limit computable weakly asymptotically optimal reinforcement learning agent.
>	Finally, our results culminate in a formal solution to the grain of truth problem: A Bayesian agent acting in a multi-agent environment learns to predict the other agents’ policies if its prior assigns positive probability to them (the prior contains a grain of truth). We construct a large but limit computable class containing a grain of truth and show that agents based on Thompson sampling over this class converge to play ε-Nash equilibria in arbitrary unknown computable multi-agent environments."

----
>	"Recently it was revealed that these optimality notions are trivial or subjective: a Bayesian agent does not explore enough to lose the prior’s bias, and a particularly bad prior can make the agent conform to any arbitrarily bad policy as long as this policy yields some rewards. These negative results put the Bayesian approach to (general) RL into question. We remedy the situation by showing that using Bayesian techniques an agent can indeed be optimal in an objective sense."
>	"The agent we consider is known as Thompson sampling or posterior sampling. It samples an environment ρ from the posterior, follows the ρ-optimal policy for one effective horizon (a lookahead long enough to encompass most of the discount function’s mass), and then repeats. We show that this agent’s policy is asymptotically optimal in mean (and, equivalently, in probability). Furthermore, using a recoverability assumption on the environment, and some (minor) assumptions on the discount function, we prove that the worst-case regret is sublinear. This is the first time convergence and regret bounds of Thompson sampling have been shown under such general conditions."

  - ["Thompson Sampling is Asymptotically Optimal in General Environments"](https://arxiv.org/abs/1602.07905) by Leike, Lattimore, Orseau, Hutter
  - <https://youtube.com/watch?v=hSiuJuvTBoE> (Leike)


#### Schmidhuber - ["On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models"](http://arxiv.org/abs/1511.09249)
>	"This paper addresses the general problem of reinforcement learning in partially observable environments. In 2013, our large RL recurrent neural networks learned from scratch to drive simulated cars from high-dimensional video input. However, real brains are more powerful in many ways. In particular, they learn a predictive model of their initially unknown environment, and somehow use it for abstract (e.g., hierarchical) planning and reasoning. Guided by algorithmic information theory, we describe RNN-based AIs designed to do the same. Such an RNNAI can be trained on never-ending sequences of tasks, some of them provided by the user, others invented by the RNNAI itself in a curious, playful fashion, to improve its RNN-based world model. Unlike our previous model-building RNN-based RL machines dating back to 1990, the RNNAI learns to actively query its model for abstract reasoning and planning and decision making, essentially “learning to think.” The basic ideas of this report can be applied to many other cases where one RNN-like system exploits the algorithmic information content of another."

>	"Real brains seem to be learning a predictive model of their initially unknown environment, but are still far superior to present artificial systems in many ways. They seem to exploit the model in smarter ways, e.g., to plan action sequences in hierarchical fashion, or through other types of abstract reasoning, continually building on earlier acquired skills, becoming increasingly general problem solvers able to deal with a large number of diverse and complex task."

>	"We introduced novel combinations of a RNNs-based reinforcement learning controller, C, and an RNN-based predictive world model, M. In a series of trials, an RNN controller C steers an agent interacting with an initially unknown, partially observable environment. The entire lifelong interaction history is stored, and used to train an RNN world model M, which learns to predict new inputs of C (including rewards) from histories of previous inputs and actions, using predictive coding to compress the history. Controller C may uses M to achieve its goals more efficiently, e.g., through cheap, “mental” M-based trials, as opposed to expensive trials in the real world. M is temporarily used as a surrogate for the environment: M and C form a coupled RNN where M’s outputs become inputs of C, whose outputs (actions) in turn become inputs of M. Now a gradient descent technique can be used to learn and plan ahead by training C in a series of M-simulated trials to produce output action sequences achieving desired input events, such as high real-valued reward signals (while the weights of M remain fixed). Given an RL problem, C may speed up its search for rewarding behavior by learning programs that address/query/exploit M’s program-encoded knowledge about predictable regularities, e.g., through extra connections from and to (a copy of) M. This may be much cheaper than learning reward-generating programs from scratch. C also may get intrinsic reward for creating experiments causing data with yet unknown regularities that improve M."

>	"The most general CM systems implement principles of algorithmic as opposed to traditional information theory. M is actively exploited in arbitrary computable ways by C, whose program search space is typically much smaller, and which may learn to selectively probe and reuse M’s internal programs to plan and reason. The basic principles are not limited to RL, but apply to all kinds of active algorithmic transfer learning from one RNN to another. By combining gradient-based RNNs and RL RNNs, we create a qualitatively new type of self-improving, general purpose, connectionist control architecture. This RNNAI may continually build upon previously acquired problem solving procedures, some of them self-invented in a way that resembles a scientist’s search for novel data with unknown regularities, preferring still-unsolved but quickly learnable tasks over others."

>	"Early CM systems did not yet use powerful RNNs such as LSTM. A more fundamental problem is that if the environment is too noisy, M will usually only learn to approximate the conditional expectations of predicted values, given parts of the history. In certain noisy environments, Monte Carlo Tree Sampling and similar techniques may be applied to M to plan successful future action sequences for C. All such methods, however, are about simulating possible futures time step by time step, without profiting from human-like hierarchical planning or abstract reasoning, which often ignores irrelevant details."

>	"This approach is different from other, previous combinations of traditional RL and RNNs which use RNNs only as value function approximators that directly predict cumulative expected reward, instead of trying to predict all sensations time step by time step. The CM system in the present section separates the hard task of prediction in partially observable environments from the comparatively simple task of RL under the Markovian assumption that the current input to C (which is M’s state) contains all information relevant for achieving the goal."

>	"Our RNN-based CM systems of the early 1990s could in principle plan ahead by performing numerous fast mental experiments on a predictive RNN world model, M, instead of time-consuming real experiments, extending earlier work on reactive systems without memory. However, this can work well only in (near-)deterministic environments, and, even there, M would have to simulate many entire alternative futures, time step by time step, to find an action sequence for C that maximizes reward. This method seems very different from the much smarter hierarchical planning methods of humans, who apparently can learn to identify and exploit a few relevant problem-specific abstractions of possible future events; reasoning abstractly, and efficiently ignoring irrelevant spatio-temporal details."

>	"According to Algorithmic Information Theory, given some universal computer, U, whose programs are encoded as bit strings, the mutual information between two programs p and q is expressed as K(q | p), the length of the shortest program w that computes q, given p, ignoring an additive constant of O(1) depending on U (in practical applications the computation will be time-bounded). That is, if p is a solution to problem P, and q is a fast (say, linear time) solution to problem Q, and if K(q | p) is small, and w is both fast and much shorter than q, then asymptotically optimal universal search for a solution to Q, given p, will generally find w first (to compute q and solve Q), and thus solve Q much faster than search for q from scratch."

>	"Let both C and M be RNNs or similar general parallel-sequential computers. M’s vector of learnable real-valued parameters wM is trained by any SL or UL or RL algorithm to perform a certain well-defined task in some environment. Then wM is frozen. Now the goal is to train C’s parameters wC by some learning algorithm to perform another well-defined task whose solution may share mutual algorithmic information with the solution to M’s task. To facilitate this, we simply allow C to learn to actively inspect and reuse (in essentially arbitrary computable fashion) the algorithmic information conveyed by M and wM."

>	"It means that now C’s relatively small candidate programs are given time to “think” by feeding sequences of activations into M, and reading activations out of M, before and while interacting with the environment. Since C and M are general computers, C’s programs may query, edit or invoke subprograms of M in arbitrary, computable ways through the new connections. Given some RL problem, according to the AIT argument, this can greatly accelerate C’s search for a problem-solving weight vector wˆ, provided the (time-bounded) mutual algorithmic information between wˆ and M’s program is high, as is to be expected in many cases since M’s environment-modeling program should reflect many regularities useful not only for prediction and coding, but also for decision making."

>	"This simple but novel approach is much more general than previous computable, but restricted, ways of letting a feedforward C use a model M, by simulating entire possible futures step by step, then propagating error signals or temporal difference errors backwards. Instead, we give C’s program search an opportunity to discover sophisticated computable ways of exploiting M’s code, such as abstract hierarchical planning and analogy-based reasoning. For example, to represent previous observations, an M implemented as an LSTM network will develop high-level, abstract, spatio-temporal feature detectors that may be active for thousands of time steps, as long as those memories are useful to predict (and thus compress) future observations. However, C may learn to directly invoke the corresponding “abstract” units in M by inserting appropriate pattern sequences into M. C might then short-cut from there to typical subsequent abstract representations, ignoring the long input sequences normally required to invoke them in M, thus quickly anticipating a few possible positive outcomes to be pursued (plus computable ways of achieving them), or negative outcomes to be avoided."

>	"Note that M (and by extension M) does not at all have to be a perfect predictor. For example, it won’t be able to predict noise. Instead M will have learned to approximate conditional expectations of future inputs, given the history so far. A naive way of exploiting M’s probabilistic knowledge would be to plan ahead through naive step-by-step Monte-Carlo simulations of possible M-predicted futures, to find and execute action sequences that maximize expected reward predicted by those simulations. However, we won’t limit the system to this naive approach. Instead it will be the task of C to learn to address useful problem-specific parts of the current M, and reuse them for problem solving. Sure, C will have to intelligently exploit M, which will cost bits of information (and thus search time for appropriate weight changes of C), but this is often still much cheaper in the AIT sense than learning a good C program from scratch."

>	"While M’s weights are frozen, the weights of C can learn when to make C attend to history information represented by M’s state, and when to ignore such information, and instead use M’s innards in other computable ways. This can be further facilitated by introducing a special unit, uˆ, to C, where uˆ(t)all(t) instead of all(t) is fed into M at time t, such that C can easily (by setting uˆ(t) = 0) force M to completely ignore environmental inputs, to use M for “thinking” in other ways."

>	"Given a new task and a C trained on several previous tasks, such hierarchical/incremental methods may freeze the current weights of C, then enlarge C by adding new units and connections which are trained on the new task. This process reduces the size of the search space for the new task, giving the new weights the opportunity to learn to use the frozen parts of C as subprograms."

>	"We motivate C to become an efficient explorer and an artificial scientist, by adding to its standard external reward (or fitness) for solving user-given tasks another intrinsic reward for generating novel action sequences (= experiments) that allow M to improve its compression performance on the resulting data. At first glance, repeatedly evaluating M’s compression performance on the entire history seems impractical. A heuristic to overcome this is to focus on M’s improvements on the most recent trial, while regularly re-training M on randomly selected previous trials, to avoid catastrophic forgetting. A related problem is that C’s incremental program search may find it difficult to identify (and assign credit to) those parts of C responsible for improvements of a huge, black box-like, monolithic M. But we can implement M as a self-modularizing, computation cost-minimizing, winner-take-all RNN. Then it is possible to keep track of which parts of M are used to encode which parts of the history. That is, to evaluate weight changes of M, only the affected parts of the stored history have to be re-tested. Then C’s search can be facilitated by tracking which parts of C affected those parts of M. By penalizing C’s programs for the time consumed by such tests, the search for C is biased to prefer programs that conduct experiments causing data yielding quickly verifiable compression progress of M. That is, the program search will prefer to change weights of M that are not used to compress large parts of the history that are expensive to verify. The first implementations of this simple principle were described in our work on the POWERPLAY framework, which incrementally searches the space of possible pairs of new tasks and modifications of the current program, until it finds a more powerful program that, unlike the unmodified program, solves all previously learned tasks plus the new one, or simplifies/compresses/speeds up previous solutions, without forgetting any. Under certain conditions this can accelerate the acquisition of external reward specified by user-defined tasks."

----
>	"What you describe is my other old RNN-based CM system from 1990: a recurrent controller C and a recurrent world model M, where C can use M to simulate the environment step by step and plan ahead. But the new stuff is different and much less limited - now C can learn to ask all kinds of computable questions to M (e.g., about abstract long-term consequences of certain subprograms), and get computable answers back. No need to simulate the world millisecond by millisecond (humans apparently don’t do that either, but learn to jump ahead to important abstract subgoals)."

  - <https://youtube.com/watch?v=JJj4allguoU> (Schmidhuber)



---
### interesting papers - Artificial Curiosity and Creativity

[Artificial Curiosity and Creativity](#artificial-curiosity-and-creativity) theory by Juergen Schmidhuber

[interesting papers](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#interesting-papers---exploration-and-intrinsic-motivation) on exploration and intrinsic motivation


#### Schmidhuber - ["Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes"](http://arxiv.org/abs/0812.4360)
>	"I argue that data becomes temporarily interesting by itself to some self-improving, but computationally limited, subjective observer once he learns to predict or compress the data in a better way, thus making it subjectively simpler and more beautiful. Curiosity is the desire to create or discover more non-random, non-arbitrary, regular data that is novel and surprising not in the traditional sense of Boltzmann and Shannon but in the sense that it allows for compression progress because its regularity was not yet known. This drive maximizes interestingness, the first derivative of subjective beauty or compressibility, that is, the steepness of the learning curve. It motivates exploring infants, pure mathematicians, composers, artists, dancers, comedians, yourself, and artificial systems."


#### Schmidhuber - ["Formal Theory of Creativity, Fun, and Intrinsic Motivation"](http://people.idsia.ch/~juergen/ieeecreative.pdf)
>	"The simple but general formal theory of fun & intrinsic motivation & creativity is based on the concept of maximizing intrinsic reward for the active creation or discovery of novel, surprising patterns allowing for improved prediction or data compression. It generalizes the traditional field of active learning, and is related to old but less formal ideas in aesthetics theory and developmental psychology. It has been argued that the theory explains many essential aspects of intelligence including autonomous development, science, art, music, humor. This overview first describes theoretically optimal (but not necessarily practical) ways of implementing the basic computational principles on exploratory, intrinsically motivated agents or robots, encouraging them to provoke event sequences exhibiting previously unknown but learnable algorithmic regularities. Emphasis is put on the importance of limited computational resources for online prediction and compression. Discrete and continuous time formulations are given. Previous practical but non-optimal implementations (1991, 1995, 1997-2002) are reviewed, as well as several recent variants by others (2005-). A simplified typology addresses current confusion concerning the precise nature of intrinsic motivation."

>	"I have argued that a simple but general formal theory of creativity based on reward for creating or finding novel patterns allowing for data compression progress explains many essential aspects of intelligence including science, art, music, humor. Here I discuss what kind of general bias towards algorithmic regularities we insert into our robots by implementing the principle, why that bias is good, and how the approach greatly generalizes the field of active learning. I provide discrete and continuous time formulations for ongoing work on building an Artificial General Intelligence based on variants of the artificial creativity framework."

>	"In the real world external rewards are rare. But unsupervised AGIs using additional intrinsic rewards as described in this paper will be motivated to learn many useful behaviors even in absence of external rewards, behaviors that lead to predictable or compressible results and thus reflect regularities in the environment, such as repeatable patterns in the world’s reactions to certain action sequences. Often a bias towards exploring previously unknown environmental regularities through artificial curiosity / creativity is a priori desirable because goal-directed learning may greatly profit from it, as behaviors leading to external reward may often be rather easy to compose from previously learnt curiosity-driven behaviors. It may be possible to formally quantify this bias towards novel patterns in form of a mixture-based prior, a weighted sum of probability distributions on sequences of actions and resulting inputs, and derive precise conditions for improved expected external reward intake. Intrinsic reward may be viewed as analogous to a regularizer in supervised learning, where the prior distribution on possible hypotheses greatly influences the most probable interpretation of the data in a Bayesian framework (for example, the well-known weight decay term of neural networks is a consequence of a Gaussian prior with zero mean for each weight). Following the introductory discussion, some of the AGIs based on the creativity principle will become scientists, artists, or comedians."

  - <http://idsia.ch/~juergen/creativity.html>
  - <https://vimeo.com/7441291#t=7m12s> (Schmidhuber)  
  - <http://videolectures.net/ecmlpkdd2010_schmidhuber_ftf/> (Schmidhuber)
  - <https://archive.org/details/Redwood_Center_2014_08_15_Jurgen_Schmidhuber> (Schmidhuber)
  - <https://vimeo.com/28759091> (Schmidhuber)


#### Houthooft, Chen, Duan, Schulman, Turck, Abbeel - ["VIME: Variational Information Maximizing Exploration"](http://arxiv.org/abs/1605.09674)
>	approximation of [Artificial Curiosity and Creativity](#artificial-curiosity-and-creativity) theory of Juergen Schmidhuber

  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#houthooft-chen-duan-schulman-turck-abbeel---vime-variational-information-maximizing-exploration>


#### Graves, Bellemare, Menick, Munos, Kavukcuoglu - ["Automated Curriculum Learning for Neural Networks"](https://arxiv.org/abs/1704.03003)
>	approximation of [Artificial Curiosity and Creativity](#artificial-curiosity-and-creativity) theory of Juergen Schmidhuber

  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#graves-bellemare-menick-munos-kavukcuoglu---automated-curriculum-learning-for-neural-networks>


#### Frank, Leitner, Stollenga, Forster, Schmidhuber - ["Curiosity Driven Reinforcement Learning for Motion Planning on Humanoids"](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3881010/pdf/fnbot-07-00025.pdf)
>	"Most previous work on artificial curiosity and intrinsic motivation focuses on basic concepts and theory. Experimental results are generally limited to toy scenarios, such as navigation in a simulated maze, or control of a simple mechanical system with one or two degrees of freedom. To study AC in a more realistic setting, we embody a curious agent in the complex iCub humanoid robot. Our novel reinforcement learning framework consists of a state-of-the-art, low-level, reactive control layer, which controls the iCub while respecting constraints, and a high-level curious agent, which explores the iCub’s state-action space through information gain maximization, learning a world model from experience, controlling the actual iCub hardware in real-time. To the best of our knowledge, this is the first ever embodied, curious agent for real-time motion planning on a humanoid. We demonstrate that it can learn compact Markov models to represent large regions of the iCub’s configuration space, and that the iCub explores intelligently, showing interest in its physical constraints as well as in objects it finds in its environment."

  - <http://vimeo.com/51011081> (demo)
  - <http://videolectures.net/ecmlpkdd2010_schmidhuber_ftf/> (Schmidhuber)


#### Schmidhuber - ["PowerPlay: Training an Increasingly General Problem Solver by Continually Searching for the Simplest Still Unsolvable Problem"](http://arxiv.org/pdf/1112.5309v2.pdf)
>	"Most of computer science focuses on automatically solving given computational problems. I focus on automatically inventing or discovering problems in a way inspired by the playful behavior of animals and humans, to train a more and more general problem solver from scratch in an unsupervised fashion. Consider the infinite set of all computable descriptions of tasks with possibly computable solutions. Given a general problem solving architecture, at any given time, the novel algorithmic framework PowerPlay searches the space of possible pairs of new tasks and modifications of the current problem solver, until it finds a more powerful problem solver that provably solves all previously learned tasks plus the new one, while the unmodified predecessor does not. Newly invented tasks may require to achieve a wow-effect by making previously learned skills more efficient such that they require less time and space. New skills may (partially) re-use previously learned skills. The greedy search of typical PowerPlay variants uses time-optimal program search to order candidate pairs of tasks and solver modifications by their conditional computational (time & space) complexity, given the stored experience so far. The new task and its corresponding task-solving skill are those first found and validated. This biases the search towards pairs that can be described compactly and validated quickly. The computational costs of validating new tasks need not grow with task repertoire size. Standard problem solver architectures of personal computers or neural networks tend to generalize by solving numerous tasks outside the self-invented training set; PowerPlay’s ongoing search for novelty keeps breaking the generalization abilities of its present solver. This is related to Goedel’s sequence of increasingly powerful formal theories based on adding formerly unprovable statements to the axioms without affecting previously provable theorems. The continually increasing repertoire of problem solving procedures can be exploited by a parallel search for solutions to additional externally posed tasks. PowerPlay may be viewed as a greedy but practical implementation of basic principles of creativity."

  - <https://youtu.be/SAcHyzMdbXc?t=16m6s> (de Freitas)


#### Srivastava, Steunebrink, Stollenga, Schmidhuber - ["Continually Adding Self-Invented Problems to the Repertoire: First Experiments with PowerPlay"](http://people.idsia.ch/~steunebrink/Publications/ICDL12_powerplay.pdf)
>	"Pure scientists do not only invent new methods to solve given problems. They also invent new problems. The recent PowerPlay framework formalizes this type of curiosity and creativity in a new, general, yet practical way. To acquire problem solving prowess through playing, PowerPlay-based artificial explorers by design continually come up with the fastest to find, initially novel, but eventually solvable problems. They also continually simplify or speed up solutions to previous problems. We report on results of first experiments with PowerPlay. A self-delimiting recurrent neural network (SLIM RNN) is used as a general computational architecture to implement the system’s solver. Its weights can encode arbitrary, self-delimiting, halting or non-halting programs affecting both environment (through effectors) and internal states encoding abstractions of event sequences. In open-ended fashion, our PowerPlay-driven RNNs learn to become increasingly general problem solvers, continually adding new problem solving procedures to the growing repertoire, exhibiting interesting developmental stages."

  - <https://youtu.be/SAcHyzMdbXc?t=16m6s> (de Freitas)


#### Orseau - ["Universal Knowledge-Seeking Agents"](http://www.agroparistech.fr/mmip/maths/laurent_orseau/papers/orseau-TCS-2014-uksa.pdf)
>	"Reinforcement learning agents like Hutter’s universal, Pareto optimal, incomputable AIXI heavily rely on the definition of the rewards, which are necessarily given by some “teacher” to define the tasks to solve. Therefore, as is, AIXI cannot be said to be a fully autonomous agent. From the point of view of artificial general intelligence, this can be argued to be an incomplete definition of a generally intelligent agent. Furthermore, it has recently been shown that AIXI can converge to a suboptimal behavior in certain situations, hence showing the intrinsic difficulty of RL, with its non-obvious pitfalls. We propose a new model of intelligence, the knowledge-seeking agent, halfway between Solomonoff induction and AIXI, that defines a completely autonomous agent that does not require a teacher. The goal of this agent is not to maximize arbitrary rewards, but to entirely explore its world in an optimal way. A proof of strong asymptotic optimality for a class of horizon functions shows that this agent behaves according to expectation. Some implications of such an unusual agent are proposed."

>	"We defined a new kind of universal intelligent agents, named knowledge-seeking agents, which differ significantly from the traditional reinforcement learning framework and its associated universal optimal learner AIXI: Their purpose is not to solve particular, narrow tasks, given or defined by experts like humans, but to be fully autonomous and to depend on no external intelligent entity. Full autonomy is an important property if we are to create generally intelligent agents, that should match or surpass (collective) human intelligence. We believe such agents (or their computational variants) should turn out to be useful to humanity in a different way than RL agents, since they should constantly be creative and solve interesting problems that we may not yet know. It seems that this kind of agent can still be directed to some extent, either by using pieces of knowledge as rewards, or by controlling the parts of the environment the agent interacts with, or by giving it prior knowledge. But these are only temporary biases that decrease in strength as the agent acquires knowledge, in the convergence to optimality. In the real world, where all agents are mortal in some way, it is unlikely that a KSA would be too curious so as to threaten its own life, since a (predicted) death would prevent it from acquiring more knowledge. From a game theory perspective, knowledge-seeking is a positive-sum game, and all the players should cooperate to maximize their knowledge. This is an interesting property regarding the safety of other intelligent agents, as long as they are interested in knowledge (which humans seem to be, at least to some point). However, this alone cannot ensure complete safety because, as soon as several agents have limited resources, a conflict situation can occur, which is not good news for the less “powerful” of the agents."

>	"We proved convergence of Square-KSA to the optimal non-learning variant of this agent for a class of horizon functions, meaning that it behaves according to expectation, even in the limit, in all computable environments. If the horizon function grows incomputably fast, we also proved that any environment that is different from the true one sufficiently often is eventually discarded. If the horizon function grows only in a computable way, we showed that environments that may not be discarded tend to be indistinguishable from the true environment in the limit."

>	"The related agent, Shannon-KSA, based on Shannon entropy, has interesting properties and its value function can be interpreted in terms of how many bits of complexity (information) the agent can expect to gain by doing a particular string of actions. Its asymptotic optimality has been proven but with more restrictions than for Square-KSA, and it is not clear if it can be extended beyond the current limitations."

>	"As for AIXI, we hope that the various KSA properties to extend nicely to stochastic computable environments. We also expect Square-KSAρ or Shannon-KSAρ (or both) to be Pareto optimal, to converge quickly, in an optimal way to the true environment, i.e., no learning agent should acquire knowledge faster. We are currently trying to get rid of the horizon function in an optimal way. One possibility could be to choose an horizon function based on ρ(h), although having asymptotic optimality for a geometric discounting would also be a good property, as such discounting is time consistent and widely used in RL. Another important concern is how to optimally (for some definition of optimality) scale down SquareKSAρ to a computable agent."

  - <http://aslanides.io/aixijs/#ksa>


#### Schmidhuber - ["Goedel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements"](http://arxiv.org/abs/cs/0309048)
>	"We present the first class of mathematically rigorous, general, fully self-referential, self-improving, optimally efficient problem solvers. Inspired by Kurt Goedel's celebrated self-referential formulas, such a problem solver rewrites any part of its own code as soon as it has found a proof that the rewrite is useful, where the problem-dependent utility function and the hardware and the entire initial code are described by axioms encoded in an initial proof searcher which is also part of the initial code. The searcher systematically and efficiently tests computable proof techniques (programs whose outputs are proofs) until it finds a provably useful, computable self-rewrite. We show that such a self-rewrite is globally optimal - no local maxima! - since the code first had to prove that it is not useful to continue the proof search for alternative self-rewrites. Unlike previous non-self-referential methods based on hardwired proof searchers, ours not only boasts an optimal order of complexity but can optimally reduce any slowdowns hidden by the O()-notation, provided the utility of such speed-ups is provable at all."

>	"The initial software p(1) of our Goedel machine runs an initial, typically sub-optimal problem solver, e.g., one of Hutter’s approaches which have at least an optimal order of complexity, or some less general method. Simultaneously, it runs an O()-optimal initial proof searcher using an online variant of Universal Search to test proof techniques, which are programs able to compute proofs concerning the system’s own future performance, based on an axiomatic system A encoded in p(1), describing a formal utility function u, the hardware and p(1) itself. If there is no provably good, globally optimal way of rewriting p(1) at all, then humans will not find one either. But if there is one, then p(1) itself can find and exploit it. This approach yields the first class of theoretically sound, fully self-referential, optimally efficient, general problem solvers. After the theoretical discussion, one practical question remains: to build a particular, especially practical Goedel machine with small initial constant overhead, which generally useful theorems should one add as axioms to A (as initial bias) such that the initial searcher does not have to prove them from scratch?"

  - <https://youtube.com/watch?v=nqiUFc52g78> (Schmidhuber)
