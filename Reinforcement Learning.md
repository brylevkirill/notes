  Reinforcement Learning is learning to maximize expected sum of future rewards for sequence of actions made by agent in environment with stochastic state unknown to agent and dependent on its actions.


  * [introduction](#introduction)
  * [applications](#applications)
  * [overview](#overview)
  * [deep reinforcement learning](#deep-reinforcement-learning)
  * [theory](#theory)
  * [exploration and intrinsic motivation](#exploration-and-intrinsic-motivation)
  * [bandits](#bandits)
  * [model-based methods](#model-based-methods)
  * [value-based methods](#value-based-methods)
  * [policy-based methods](#policy-based-methods)
  * [interesting papers](#interesting-papers)
    - [applications](#interesting-papers---applications)
    - [exploration and intrinsic motivation](#interesting-papers---exploration-and-intrinsic-motivation)
    - [abstractions for states and actions](#interesting-papers---abstractions-for-states-and-actions)
    - [model-based methods](#interesting-papers---model-based-methods)
    - [value-based methods](#interesting-papers---value-based-methods)
    - [policy-based methods](#interesting-papers---policy-based-methods)
    - [behavioral cloning](#interesting-papers---behavioral-cloning)
    - [inverse reinforcement learning](#interesting-papers---inverse-reinforcement-learning)



---
### introduction

  Reinforcement Learning in general case is learning to act through trial and error with no provided models, labels, demonstrations or supervision signals other than delayed rewards for agent's actions.

  [definition](https://youtube.com/watch?v=kl_G95uKTHw&t=1h9m30s) by Sergey Levine

  ![relations with other fields](https://goo.gl/XlgPJu)

----

  "Reinforcement Learning is as hard as any problem in computer science, since any task with a computable description can be formulated in it."

  "Reinforcement Learning is a general-purpose framework for decision-making:  
   - Is for an agent with the capacity to act  
   - Each action influences the agent's future state  
   - Success is measured by a scalar reward signal  
   - Goal: select actions to maximize future reward"  

  "Deep Learning is a general-purpose framework for representation learning:  
   - Given an objective  
   - Learn representation that is required to achieve objective  
   - Directly from raw inputs  
   - Using minimal domain knowledge"  

  "We seek a single agent which can solve any human-level task:  
   - Reinforcement Learning defines the objective  
   - Deep Learning gives the mechanism  
   - Reinforcement Learning + Deep Learning = general intelligence"  

  *(David Silver)*



---
### applications

#### industry

  [recommender systems](https://deepmind.com/blog/deep-reinforcement-learning/) at Google

  [datacenter cooling](https://deepmind.com/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-40/) at Google

  [personalized news](http://thenewstack.io/reinforcement-learning-ready-real-world/) at Microsoft  
>	"For MSN homepage 25% relative lift in click-through rate while no previous learning method had been more successful than humans curating the content manually."  
  - <https://mwtds.azurewebsites.net>  
  - <http://research.microsoft.com/en-us/projects/mwt/>  
  - <http://hunch.net/?p=4464948>  
  - <https://youtube.com/watch?v=7ic_d5TeIUk> (Langford)  
  - <https://youtu.be/N5x48g2sp8M?t=52m> (Schapire)  


  [interview with Demis Hassabis](https://www.technologyreview.com/s/601139/how-google-plans-to-solve-artificial-intelligence/) from Google
>	"Technology could surface in virtual assistants or improve recommendation systems, which are crucial to products such as YouTube (similar systems also power some of Google’s advertising products)."

  [patent on Deep Q-Network](http://google.com/patents/US20150100530) by Google  
>	"Further applications of the techniques we describe, which are merely given by way of example, include: robot control (such as bipedal or quadrupedal walking or running, navigation, grasping, and other control skills); vehicle control (autonomous vehicle control, steering control, airborne vehicle control such as helicopter or plane control, autonomous mobile robot control); machine control; control of wired or wireless communication systems; control of laboratory or industrial equipment; control of real or virtual resources (such as memory management, inventory management and the like); drug discovery (where the controlled action is, say, the definition of DNA sequence of a drug and the states are defined by states of a living entity to which the drug is applied); application to a system in which the state of or output from the system is defined by words (text and/or audio and/or image), such as a system employing natural language; application to a trading system such as a stock market (although the actions taken may have little effect on such a system, very small effects can be sufficient to achieve useful overall rewards); and others."

----
#### games

  - *Go*  
	["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)  

	[history of ideas](http://youtube.com/watch?v=UMm0XaCFTJQ) by Sutton, Szepesvari, Bowling, Hayward, Muller  
	"Google AlphaGo is a historical tour of AI ideas: 70s (Alpha-Beta), 80s/90s (RL & self-play), 00's (Monte-Carlo), 10's (deep neural networks)."  

	<https://youtu.be/i3lEG6aRGm8?t=16m> (Hassabis)  
	<https://youtu.be/yCALyQRN3hw?t=15m18s> + <https://youtu.be/qUAmTYHEyM8?t=15m31s> + <https://youtu.be/mzpW10DPHeQ?t=10m22s> (Silver, Maddison)  
	<https://youtube.com/watch?v=4D5yGiYe8p4> (Silver)  
	<https://youtu.be/LX8Knl0g0LE?t=4m41s> (Huang)  

	<https://xcorr.net/2016/02/03/5-easy-pieces-how-deepmind-mastered-go/>  
	<https://github.com/Rochester-NRT/RocAlphaGo/wiki>  

	game 1: [video](https://youtube.com/watch?v=bIQxOsRAXCo) +
		[analysis](https://gogameguru.com/alphago-defeats-lee-sedol-game-1/) +
		[analysis](http://deeplearningskysthelimit.blogspot.ru/2016/04/part-6-review-of-game-1-lee-sedol.html)  
	game 2: [video](https://youtube.com/watch?v=1aMt7ulL6EI) +
		[analysis](https://gogameguru.com/alphago-races-ahead-2-0-lee-sedol/) +
		[analysis](http://deeplearningskysthelimit.blogspot.ru/2016/04/part-7-review-of-game-2-alphagos-new.html)  
	game 3: [video](https://youtube.com/watch?v=6hROM_bxZ9E) +
		[analysis](https://gogameguru.com/alphago-shows-true-strength-3rd-victory-lee-sedol/) +
		[analysis](http://deeplearningskysthelimit.blogspot.ru/2016/04/part-8-review-of-game-3-lee-sedols.html)  
	game 4: [video](https://youtube.com/watch?v=G5gJ-pVo1gs) +
		[analysis](https://gogameguru.com/lee-sedol-defeats-alphago-masterful-comeback-game-4/) +
		[analysis](http://deeplearningskysthelimit.blogspot.ru/2016/04/part-9-review-of-game-4-lee-sedols.html)  
	game 5: [video](https://youtube.com/watch?v=QxHdPdRcMhw) +
		[analysis](https://gogameguru.com/alphago-defeats-lee-sedol-4-1/) +
		[analysis](http://deeplearningskysthelimit.blogspot.ru/2016/05/part-10-review-of-game-5-alphago.html)  

  - *Poker*  
	["DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker"](http://arxiv.org/abs/1701.01724)  
	[Science paper](http://science.sciencemag.org/content/early/2017/03/01/science.aam6960)  

	<http://deepstack.ai>  
	<http://twitter.com/DeepStackAI>  

	<https://youtube.com/watch?v=qndXrHcV1sM> (Bowling)  
	<http://thinkingpoker.net/2017/03/episode-208-michael-bowling-of-cprg/> (Bowling)  

	<http://natemeyvis.com/blog/2017/3/30/my-heads-up-matches-against-deepstack>  
	<http://thinkingpoker.net/2017/03/battling-deepstack/>  

	[demo matches](https://youtube.com/playlist?list=PLX7NnbJAq7PlA2XpynViLOigzWtmr6QVZ)

  - *Chess*  
	["Giraffe: Using Deep Reinforcement Learning to Play Chess"](http://arxiv.org/abs/1509.01549)

  - *Atari video games*  
	["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602)

	<http://youtube.com/watch?v=dV80NAlEins> (de Freitas)

	<http://youtube.com/watch?v=EfGD2qveGdQ> (demo)  
	<http://youtu.be/XAbLn66iHcQ?t=1h41m21s> + <http://youtube.com/watch?v=0xo1Ldx3L5Q> (3D racing demo)  
	<http://youtube.com/watch?v=nMR5mjCFZCw> (3D labyrinth demo)  
	<http://youtube.com/watch?v=re6hkcTWVUY> (Doom gameplay demo)  
	<http://youtube.com/watch?v=6jlaBD9LCnM> + <http://youtube.com/watch?v=6JT6_dRcKAw> (blockworld demo)  
	<http://youtu.be/XAbLn66iHcQ?t=1h41m21s> (demo)  
	<http://youtube.com/user/eldubro/videos> (demos)  
	<http://youtube.com/watch?v=iqXKQf2BOSE> (demo)  

  - *Doom*  
	<http://vizdoom.cs.put.edu.pl>  
	<https://youtube.com/watch?v=Qv4esGWOg7w>  
	<https://youtube.com/watch?v=947bSUtuSQ0>  
	<https://youtube.com/watch?v=tDRdgpkleXI>  

----
#### robotics

  "Making Robots Learn" by Pieter Abbeel:  
	<https://youtu.be/xe-z4i3l-iQ?t=30m35s>  
	<http://on-demand.gputechconf.com/gtc/2016/video/S6812.html>  
	<http://youtube.com/watch?v=xMHjkZBvnfU>  

  "Deep Robotic Learning" by Sergey Levine:  
	<http://videolectures.net/iclr2016_levine_deep_learning/>  
	<https://youtube.com/watch?v=f41JXf-ojrM>  
	<https://youtube.com/watch?v=EtMyH_--vnU>  
	<https://video.seas.harvard.edu/media/ME+Sergey+Levine+2015+-04-01/1_gqqp9r3o/23375211>  

----

  [**open world problems**](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#open-world-problems)

----

  ["Why Tool AIs Want to Be Agent AIs"](http://gwern.net/Tool%20AI) by Gwern Branwen:  
>	"The logical extension of these neural networks all the way down papers is that an actor like Google/Baidu/Facebook/MS could effectively turn neural networks into a black box: a user/developer uploads through an API a dataset of input/output pairs of a specified type and a monetary loss function, and a top-level neural network running on a large GPU cluster starts autonomously optimizing over architectures & hyperparameters for the neural network design which balances GPU cost and the monetary loss, interleaved with further optimization over the thousands of previous submitted tasks, sharing its learning across all of the datasets/loss functions/architectures/hyperparameters, and the original user simply submits future data through the API for processing by the best neural network so far."



---
### overview

  [introduction](https://youtube.com/watch?v=2pWv7GOvuf0) by David Silver

  introduction by Kevin Frans:  
  - [basics](http://kvfrans.com/reinforcement-learning-basics/)  
  - [Markov processes](http://kvfrans.com/markov-processes-in-reinforcement-learning/)  
  - [planning](http://kvfrans.com/planning-policy-evaluation-policy-iteration-value-iteration/)  
  - [model-free methods](http://kvfrans.com/model-free-prediction-and-control/)  
  - [policy gradient methods](http://kvfrans.com/the-policy-gradient/)  
  - [model-based methods](http://kvfrans.com/making-use-of-the-model/)  

  introduction by Massimiliano Patacchiola:  
  - [Dynamic Programming](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html)  
  - [Monte Carlo](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html)  
  - [Temporal Difference](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html)  
  - [Actor-Critic](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html)  
  - [Genetic Algorithms](https://mpatacchiola.github.io/blog/2017/03/14/dissecting-reinforcement-learning-5.html)  

  introduction by Shakir Mohamed:  
  - ["Learning in Brains and Machines: Temporal Differences"](http://blog.shakirm.com/2016/02/learning-in-brains-and-machines-1/)  
  - ["Synergistic and Modular Action"](http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-3-synergistic-and-modular-action/)  

  [introduction](http://readcube.com/articles/10.1038%2Fnature14540) by Michael Littman

----

  overview by David Silver:  
	<http://techtalks.tv/talks/deep-reinforcement-learning/62360/>  
	<http://videolectures.net/rldm2015_silver_reinforcement_learning/>  
	<http://youtube.com/watch?v=qLaDWKd61Ig>  
	<http://youtube.com/watch?v=3hWn5vMnpiM>  

  overview by Pieter Abbeel:  
	<http://youtube.com/watch?v=evq4p1zhS7Q>  
	<http://research.microsoft.com/apps/video/default.aspx?id=260045>  

----

  [tutorial](https://youtube.com/watch?v=Fsh1qMTg1xI) by Rich Sutton ([write-up](https://goo.gl/PxHMLK))  
  [tutorial](http://videolectures.net/deeplearning2016_pineau_reinforcement_learning/) by Joelle Pineau  
  [tutorial](https://youtube.com/watch?v=fIKkhoI1kF4) by Emma Brunskill  

----

  [course](http://youtube.com/playlist?list=PL5X3mDkKaJrL42i_jhE4N-p6E2Ol62Ofa) by David Silver
	([slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html))  
  [course](https://udacity.com/course/reinforcement-learning--ud600) by Michael Littman  

----

  ["Reinforcement Learning: An Introduction"](http://incompleteideas.net/sutton/book/the-book-2nd.html) book by Rich Sutton and Andrew Barto (second edition, draft)  
  ["Reinforcement Learning: An Introduction"](http://incompleteideas.net/sutton/book/ebook/the-book.html) book by Rich Sutton and Andrew Barto (first edition)  
  ["Algorithms for Reinforcement Learning"](http://www.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) book by Csaba Szepesvari  

----

  ["MDP Cheatsheet Reference"](http://rll.berkeley.edu/deeprlcourse/docs/mdp-cheatsheet.pdf) by John Schulman  

  [course notes](https://web.stanford.edu/class/msande338/lecture_notes.html) by Ben Van Roy  
  [course notes](http://dustintran.com/notes/cs282r.pdf) by Dustin Tran  
  [course slides](http://incompleteideas.net/sutton/609%20dropbox/slides%20(pdf%20and%20keynote)) by Rich Sutton  

  [exercises and solutions](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) by Shangtong Zhang  
  [exercises and solutions](http://wildml.com/2016/10/learning-reinforcement-learning/) by Denny Britz  
  [course with exercises and solutions](https://github.com/yandexdataschool/Practical_RL/) from Yandex  

  [implementations of algorithms](https://github.com/rlcode/reinforcement-learning) from RLCode team  
  [implementations of algorithms](https://github.com/openai/rllab/tree/master/rllab/algos) from OpenAI  



---
### deep reinforcement learning

  ["Deep Reinforcement Learning: An Overview"](https://arxiv.org/abs/1701.07274) by Yuxi Li

  [course](http://rll.berkeley.edu/deeprlcourse/) by Sergey Levine, John Schulman and Chelsea Finn
	([videos](https://youtube.com/playlist?list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX))

  ["The Nuts and Bolts of Deep RL Research"](http://rll.berkeley.edu/deeprlcourse/docs/nuts-and-bolts.pdf) by John Schulman


  ["Deep Reinforcement Learning"](https://sites.google.com/site/deeprlnips2016/) workshop at NIPS 2016  
  ["Abstraction in RL"](http://rlabstraction2016.wix.com/icml) workshop at ICML 2016  
  ["Deep Reinforcement Learning: Frontiers and Challenges"](https://sites.google.com/site/deeprlijcai16/) workshop at IJCAI 2016  
  ["Deep Reinforcement Learning"](http://rll.berkeley.edu/deeprlworkshop/) workshop at NIPS 2015  
  ["Novel Trends and Applications in RL"](https://tcrl14.wordpress.com/videos/) workshop at NIPS 2014  



---
### theory

  differences between reinforcement learning and other learning paradigms  ([overview](https://youtube.com/watch?v=2pWv7GOvuf0&t=9m37s) by David Silver):  
  - there is no supervisor, only a reward signal  
  - feedback is delayed, not instantaneous  
  - time really matters (sequential, not i.i.d. data)  
  - agent's actions affect subsequent data it receives  

  differences between reinforcement learning and supervised learning  ([overview](https://youtube.com/watch?v=8jQIKgTzQd4&t=50m28s) by John Schulman):  
  - no full access to analytic representation of loss function being optimized - value has to be queried by interaction with environment  
  - interacting with stateful environment (unknown, nonlinear, stochastic, arbitrarily complex) - next input depends on previous actions  

  characteristics:  
  - can learn any function  
  - inherently handles uncertainty  
    * uncertainty in actions (the world)  
    * uncertainty in observations (sensors)  
  - directly maximise criteria we care about (instead of loss function on samples)  
  - copes with delayed feedback  
    * temporal credit assignment problem  

----

  challenges:  
  - stability (non-stationary, fleeting nature of time and online data)  
  - credit assigment (delayed rewards and consequences)  
  - exploration vs exploitation (need for trial and error)  
  - using learned model of environment  

  open problems:  
  - adaptive methods which work under large number of conditions  
  - addressing exploration problem in large MDPs  
  - large-scale empirical evaluations  
  - learning and acting under partial information  
  - modular and hierarchical learning over multiple time scales  
  - sample efficiency  
  - improving existing value-function and policy search methods  
  - algorithms that work well with large or continuous action spaces  
  - transfer learning  
  - lifelong learning  
  - efficient sample-based planning (e.g., based on Monte-Carlo tree search)  
  - multiagent or distributed learning  
  - learning from demostrations  

----

  components of reinforcement learning algorithm  ([overview](https://youtube.com/watch?v=_UVYhuATS9E&t=2m44s) by Sergey Levine):  
  - generate samples / run the policy  
  - fit a model / estimate the return  
  - improve the policy  

  dimensions for classification of methods  ([overview](http://incompleteideas.net/sutton/book/ebook/node105.html) by Sutton and Barto):  
  - prediction vs control  
  - MDPs vs bandits  
  - model-based vs value-based vs policy-based  ([overview](http://youtube.com/watch?v=P_agNaSrVhc) by Michael Littman)  
  - on-policy vs off-policy  
  - bootstrapping vs Monte Carlo  

----

  [**model-based methods**](#model-based-methods):  
  - build prediction model for next state and reward after action  
  - space complexity asymptotically less than space required to store MDP  
  - define objective function measuring goodness of model (e.g. number of bits to reconstruct next state)  
  - plan using model (e.g. by lookahead)  
  - allows reasoning about task-independent aspects of environment  
  - allows for transfer learning across domains and faster learning  

  [**value-based methods**](#value-based-methods):  
  - estimate the optimal value function Q*(s,a) (expected total reward from state s and action a under policy π)  
  - this is the maximum value achievable under any policy  

  [**policy-based methods**](#policy-based-methods):  
  - search directly for the optimal policy (behaviour function selecting actions given states) achieving maximum expected reward  
  - often simpler to represent and learn good policies than good state value or action value functions (such as for robot grasping an object)  
  - state value function doesn't prescribe actions (dynamics model becomes necessary)  
  - action value function requires to solve maximization problem over actions (challenge for continuous / high-dimensional action spaces)  
  - focus on discriminating between several actions instead of estimating values for every state-action  
  - true objectives of expected cost is optimized (vs a surrogate like Bellman error)  
  - suboptimal values does not necessarily give suboptimal actions in every state (but optimal values do)  
  - easier generalization to continuous action spaces  

----

  **off-policy methods**:  
  - evaluate target policy to compute control while following another policy  
  - learn from observing humans or other agents (imperfect expert) 
  - re-use experience generated from old policies  
  - learn about optimal policy while following exploratory policy  
  - learn about multiple policies (options, waypoints) while following one policy  
  - learning from sessions (recorded data)

----

  **forms of supervision**  ([overview](https://youtube.com/watch?v=hKeSPnvNNJ8) by Sergey Levine):  
  - scalar rewards  
  - demonstrated behavior (imitation, inferring intention)  
  - self-supervision, prediction (model-based control)  
  - auxiliary objectives  
    * additional sensing modalities  
    * learning related tasks  
    * task-relevant properties of environment  

----

  **imitation learning** / **behavioral cloning**:

  ["Supervised Learning of Behaviors: Deep Learning, Dynamical Systems, and Behavior Cloning"](https://youtube.com/watch?v=kl_G95uKTHw) by Sergey Levine  
  ["Learning Policies by Imitating Optimal Control"](https://youtube.com/watch?v=o0Ebur3aNMo) by Sergey Levine  
  ["Advanced Topics in Imitation Learning and Safety"](https://youtube.com/watch?v=UClw47acYnw) by Chelsea Finn  

  [interesting papers](#interesting-papers---behavioral-cloning)

----

  **inverse reinforcement learning**:  
  - infer underlying reward structure guiding agent’s behavior based on observations and model of environment  
  - learn reward structure for modelling purposes or for imitation of demonstrator's behavior (apprenticeship)  

  [introduction](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/) (part 2, 20:40) by Pieter Abbeel  
  [overview](https://youtube.com/watch?v=J2blDuU3X1I) by Chelsea Finn  

  ["Apprenticeship Learning and Reinforcement Learning with Application to Robotic Control"](http://ai.stanford.edu/~pabbeel/thesis/thesis.pdf) by Pieter Abbeel

  [interesting papers](#interesting-papers---inverse-reinforcement-learning)

----

  [**exploration and intrinsic motivation**](#exploration-and-intrinsic-motivation)

----

  **abstractions for states and actions**:  
  - simplify dimensionality of the action spaces over which we need to reason  
  - enable quick planning and execution of low-level actions (such as robot movements)  
  - provide a simple mechanism that connects plans and intentions to commands at the level of execution  
  - support rapid learning and generalisation (that humans are capable of)  

  ["Learning in Brains and Machines: Synergistic and Modular Action"](http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-3-synergistic-and-modular-action/) by Shakir Mohamed

  ["Combining State and Temporal Abstractions"](https://youtube.com/watch?v=iLSUByYY6so) by George Konidaris  
  ["Towards Representations for Efficient Reinforcement Learning"](https://youtube.com/watch?v=Pk3E5zqhl9k) by Emma Brunskill  

  Options framework:  
	[introduction](http://videolectures.net/deeplearning2016_precup_advanced_lr/) by Doina Precup  
	["Temporal Abstraction in Reinforcement Learning"](https://youtube.com/watch?v=GntIVgNKkCI) by Doina Precup  
	["Advances in Option Construction: The Option-Critic Architecture"](https://youtube.com/watch?v=8r_EoYnPjGk) by Pierre-Luc Bacon  

  [model-based methods](#model-based-methods)  
  [interesting papers](#interesting-papers---abstractions-for-states-and-actions)  



---
### exploration and intrinsic motivation

  exploration:  
  - How to search through space of possible strategies for agent to avoid getting stuck in local optima of behavior?  
  - Given a long-running learning agent, how to balance exploration and exploitation to maximize long-term rewards?  

  intrinsic motivation:  
  - avoid handcrafting special-purpose utility functions  
  - faster training if external rewards are sparse  
  - transferrable skills (hierachy of skills, discovering and combining skills)  


  [overview](http://youtube.com/watch?v=SfCa1HQMkuw) by John Schulman

  [interesting papers](#interesting-papers---exploration-and-intrinsic-motivation)

----

  **approximate bayesian exploration models**:

  ["Weight Uncertainty in Neural Networks"](#blundell-cornebise-kavukcuoglu-wierstra---weight-uncertainty-in-neural-networks) by Blundell et al.  (training bayesian neural network to predict reward, sampling particular network weights from posterior and choosing action with highest predicted reward)  
  ["Deep Exploration via Bootstrapped DQN"](#osband-blundell-pritzel-van-roy---deep-exploration-via-bootstrapped-dqn) by Osband et al.  (training multiple value function networks with shared bottom layers using bootstrapping, sampling value function network and running episode using it)  
  ["Deep Exploration via Randomized Value Functions"](https://arxiv.org/abs/1703.07608) by Osband et al.  
  ["RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning"](http://arxiv.org/abs/1611.02779) by Duan et al.  
  ["Learning to Reinforcement Learn"](http://arxiv.org/abs/1611.05763) by Wang et al.  
  ["Nonparametric General Reinforcement Learning"](#leike---nonparametric-general-reinforcement-learning) by Leike  (estimating reward by sampling environment model from posterior distribution and running episode using it)

----

  ["How Can We Define Intrinsic Motivation"](http://pyoudeyer.com/epirob08OudeyerKaplan.pdf) by Oudeyer and Kaplan:  


**information theoretic and distributional models**:  
>	"This approach is based on the use of representations, built by an agent, that estimate the distributions of probabilities of observing certain events ek in particular contexts, defined as mathematical configurations in the sensorimotor flow. There are several types of such events, but the probabilities that are measured are typically either the probability of observing a certain state SMk in the sensorimotor flow, denoted P(SMk), or the probability of observing particular transitions between states, such as P(SMk(t),SMl(t+1)), or the probability of observing a particular state after having observed a given state P(SMk(t+1)|SMl(t)). Here, the states SMk can either be direct numerical prototypes or complete regions within the sensorimotor space (and it may involve a mechanism for discretizing the space). We assume that the agent possesses a mechanism that allows it to build internally, and as it experiences the world, an estimation of the probability distribution of events across the whole space E of possible events (but the space of possible events is not predefined and should also be discovered by the agent, so typically this is an initially empty space that grows with experience)."

  - *uncertainty motivation*  
	reward for every event inversely proportional to its probability of observation  

	["Action-Conditional Video Prediction using Deep Networks in Atari Games"](#oh-guo-lee-lewis-singh---action-conditional-video-prediction-using-deep-networks-in-atari-games) by Oh et al.  (approximate visitation counting in a learned state embedding using Gaussian kernels)  
	["Unifying Count-Based Exploration and Intrinsic Motivation"](#bellemare-srinivasan-ostrovski-schaul-saxton-munos---unifying-count-based-exploration-and-intrinsic-motivation) by Bellemare et al.  (relationship between the pseudo-count, a variant of Schmidhuber’s compression progress or prediction gain, and Bayesian information gain)  
	["Count-Based Exploration with Neural Density Models"](http://arxiv.org/abs/1703.01310) by Ostrovski et al.  
	["#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning"](http://arxiv.org/abs/1611.04717) by Tang et al.  
	["EX2: Exploration with Exemplar Models for Deep Reinforcement Learning"](http://arxiv.org/abs/1703.01260) by Fu et al.  

  - *information gain motivation*  
	decrease of uncertainty in knowledge that an agent has of environment after an event has happened  

	["An Information-theoretic Approach to Curiosity-driven Reinforcement Learning"](##still-precup---an-information-theoretic-approach-to-curiosity-driven-reinforcement-learning) by Still and Precup  
	["VIME: Variational Information Maximizing Exploration"](#houthooft-chen-duan-schulman-turck-abbeel---vime-variational-information-maximizing-exploration) by Houthooft et al.  
	["Exploration Potential"](http://arxiv.org/abs/1609.04994) by Leike  

  - *empowerment*  
	agent's ability to affect its environment, reward for sequences of actions that can transfer maximal amount of information to its sensors through environment  
	maximizing mutual information between actions and future states, i.e. information contained in a about s' or information that can be "injected" into s' by a  

	["Empowerment - An Introduction"](#salge-glackin-polani---empowerment---an-introduction) by Salge et al.  
	["Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning"](#mohamed-rezende---variational-information-maximisation-for-intrinsically-motivated-reinforcement-learning) by Mohamed and Rezende  
	["Variational Intrinsic Control"](http://arxiv.org/abs/1611.07507) by Gregor et al.  (the primary goal is not to understand or predict the observations but to control the environment - agents can often control an environment perfectly well without much understanding, and focusing on understanding might significantly distract and impair the agent, as such reducing the control it achieves)  


**predictive models**:  
>	"Often, knowledge and expectations in agent are not represented by complete probability distributions, but rather based on the use of predictors such as neural networks that make direct predictions about future events. These predictors, denoted Π, are typically used to predict some properties or sensorimotor states that will happen in the future (close or far) given the current sensorimotor context SM(t) and possibly the past sensorimotor context."

  - *predictive novelty motivation*  
	interesting situations are those for which the prediction errors are highest  

	["Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models"](#stadie-levine-abbeel---incentivizing-exploration-in-reinforcement-learning-with-deep-predictive-models) by Stadie et al.  

  - *learning progress motivation*  
	reward for prediction progress, i.e. decrease of prediction errors  
	difference in prediction error of the predictor, about the same sensorimotor context, between the first prediction and a second prediction made just after the predictor has been updated with a learning rule  

	[Artificial Curiosity and Creativity](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#artificial-curiosity-and-creativity) by Schmidhuber  
	["VIME: Variational Information Maximizing Exploration"](#houthooft-chen-duan-schulman-turck-abbeel---vime-variational-information-maximizing-exploration) by Houthooft et al.  (similarity between Schmidhuber's compression progress and information gain)  
	["Surprise-Based Intrinsic Motivation for Deep Reinforcement Learning"](http://arxiv.org/abs/1703.01732) by Achiam and Sastry  

  - *predictive familiarity motivation*  
	motivation to search for situations which are very predictable and thus familiar  


**competence-based models**:  
>	"A third major computational approach to intrinsic motivation is based on measures of competence that an agent has for achieving self-determined results or goals. Central here is the concept of “challenge”, with associated measures of difficulty as well as measures of actual performance. A “challenge” or “goal” here will be any sensorimotor configuration, or any set of properties of a sensorimotor configuration, that an agent sets by itself and that it tries to achieve through action. It is the properties of the achievement process, rather than the “meaning” of the particular goal being achieved, that will determine the level of interestingness of the associated activity. While prediction mechanisms or probability models, as used in previous sections, can be used in the goal-reaching architecture, they are not mandatory. The capacity to predict what happens in a situation can be sometimes only loosely coupled to the capacity to modify a situation in order to achieve a given self-determined goal."

  - *maximizing incompetence motivation*  
	reward measure that pushes an agent to set challenges/goals for which its performance is lowest  

  - *maximizing competence progress*  
	interestingness of a challenge as the competence progress that is experienced as an agent repeatedly tries to achieve it  

	["Reinforcement Learning with Unsupervised Auxiliary Tasks"](http://arxiv.org/abs/1611.05397) by Jaderberg et al.  (by using auxiliary tasks of pixel control, reward prediction and value function replay the agent is forced to learn about the controllability of its environment and the sorts of sequences which lead to rewards)


**morphological models**:
>	"The three previous computational approaches to motivation were based on measures comparing information characterizing a stimulus perceived in the present and information characterizing stimuli perceived in the past and represented in memory. A fourth approach that can be taken is based on the comparison of information characterizing several pieces of stimuli perceived at the same time in several parts of the perceptive field. Pragmatically, this approach consists in attributing interest depending on morphological mathematical properties of the current flow of sensorimotor values, irrespective of what the internal cognitive system might predict or master."

  - *synchronicity motivation*  
	high short-term correlation between a maximally large number of sensorimotor channels  

----

  (Ian Osband) "In sequential decision problems there is an important distinction between risk and uncertainty. We identify risk as inherent stochasticity in a model and uncertainty as the confusion over which model parameters apply. For example, a coin may have a fixed p = 0.5 of heads and so the outcome of any single flip holds some risk; a learning agent may also be uncertain of p. The demarcation between risk and uncertainty is tied to the specific model class, in this case a Bernoulli random variable; with a more detailed model of flip dynamics even the outcome of a coin may not be risky at all. Our distinction is that unlike risk, uncertainty captures the variability of an agent’s posterior belief which can be resolved through statistical analysis of the appropriate data. For a learning agent looking to maximize cumulative utility through time, this distinction represents a crucial dichotomy. Consider the reinforcement learning problem of an agent interacting with its environment while trying to maximize cumulative utility through time. At each timestep, the agent faces a fundamental tradeoff: by exploring uncertain states and actions the agent can learn to improve its future performance, but it may attain better short-run performance by exploiting its existing knowledge. At a high level this effect means uncertain states are more attractive since they can provide important information to the agent going forward. On the other hand, states and action with high risk are actually less attractive for an agent in both exploration and exploitation. For exploitation, any concave utility will naturally penalize risk. For exploration, risk also makes any single observation less informative. Although colloquially similar, risk and uncertainty can require radically different treatment."



---
### bandits

  [introduction](http://youtube.com/watch?v=sGuiWX07sKw) by David Silver


  introduction by Csaba Szepesvari:  
	<http://banditalgs.com/2016/09/04/bandits-a-new-beginning/>

  introduction by Ian Osband:  
	<http://iosband.github.io/2015/07/19/Efficient-experimentation-and-multi-armed-bandits.html>  
	<http://iosband.github.io/2015/07/28/Beat-the-bandit.html> (demo and implementations)  

  introduction by Jeremy Kun:  
	<http://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/>  
	<http://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/>  
	<http://jeremykun.com/2013/12/09/bandits-and-stocks/>  

----

  tutorial by Csaba Szepesvari:  
	<https://youtube.com/watch?v=VVcLnAoU9Gw>  
	<https://youtube.com/watch?v=cknukHreMdI>  
	<https://youtube.com/watch?v=ruIO79C2IQc>  

  course by Csaba Szepesvari and Tor Lattimore:  
	["Bandits: A new beginning"](http://banditalgs.com/2016/09/04/bandits-a-new-beginning/)  
	["Finite-armed stochastic bandits: Warming up"](http://banditalgs.com/2016/09/04/stochastic-bandits-warm-up/)  
	["First steps: Explore-then-Commit"](http://banditalgs.com/2016/09/14/first-steps-explore-then-commit/)  
	["The Upper Confidence Bound (UCB) Algorithm"](http://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/)  
	["Optimality concepts and information theory"](http://banditalgs.com/2016/09/22/optimality-concepts-and-information-theory/)  
	["More information theory and minimax lower bounds"](http://banditalgs.com/2016/09/28/more-information-theory-and-minimax-lower-bounds/)  
	["Instance dependent lower bounds"](http://banditalgs.com/2016/09/30/instance-dependent-lower-bounds/)  
	["Adversarial bandits"](http://banditalgs.com/2016/10/01/adversarial-bandits/)  
	["High probability lower bounds"](http://banditalgs.com/2016/10/14/high-probability-lower-bounds/)  
	["Contextual bandits, prediction with expert advice and Exp4"](http://banditalgs.com/2016/10/14/exp4/)  
	["Stochastic Linear Bandits and UCB"](http://banditalgs.com/2016/10/19/stochastic-linear-bandits/)  
	["Ellipsoidal confidence sets for least-squares estimators"](http://banditalgs.com/2016/10/20/lower-bounds-for-stochastic-linear-bandits/)  
	["Sparse linear bandits"](http://banditalgs.com/2016/11/13/ellipsoidal-confidence-bounds-for-least-squares-estimators/)  
	["Lower bounds for stochastic linear bandits"](http://banditalgs.com/2016/11/21/sparse-stochastic-linear-bandits/)  
	["Adversarial linear bandits"](http://banditalgs.com/2016/11/24/adversarial-linear-bandits/)  
	["Adversarial linear bandits and the curious case of linear bandits on the unit ball"](http://banditalgs.com/2016/11/25/adversarial-linear-bandits-and-the-curious-case-of-the-unit-ball/)  

  course by Sebastien Bubeck:  
	<https://blogs.princeton.edu/imabandit/2016/05/11/bandit-theory-part-i/>  
	<https://blogs.princeton.edu/imabandit/2016/05/13/bandit-theory-part-ii/>  

  ["Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems"](https://arxiv.org/abs/1204.5721) by Bubeck and Cesa-Bianchi

----

  "Bandit feedback means we only observe the feedback δ(x,y) for the specific y that was predicted, but not for any other possible (counterfactual) predictions Y\{y} unlike in supervised learning. The feedback is just a number, called the loss δ: X × Y → R. Smaller numbers are desirable. In general, the loss is the (noisy) realization of a stochastic random variable. The expected loss - called risk - of a hypothesis R(h) is R(h) = Ex∼Pr(X) Ey∼h(x) [δ(x,y)] = Eh [δ(x,y)]. The aim of learning is to find a hypothesis h∈ H that has minimum risk."

  "Most interactive systems (e.g. search engines, recommender systems, ad platforms) record large quantities of log data which contain valuable information about the system’s performance and user experience. For example, the logs of an ad-placement system record which ad was presented in a given context and whether the user clicked on it. While these logs contain information that should inform the design of future systems, the log entries do not provide supervised training data in the conventional sense. This prevents us from directly employing supervised learning algorithms to improve these systems. In particular, each entry only provides bandit feedback since the loss/reward is only observed for the particular action chosen by the system (e.g. the presented ad) but not for all the other actions the system could have taken. Moreover, the log entries are biased since actions that are systematically favored by the system will by over-represented in the logs. Learning from historical logs data can be formalized as batch learning from logged bandit feedback. Unlike the well-studied problem of online learning from bandit feedback, this setting does not require the learner to have interactive control over the system. Learning in such a setting is closely related to the problem of off-policy evaluation in reinforcement learning - we would like to know how well a new system (policy) would perform if it had been used in the past. This motivates the use of counterfactual estimators. Following an approach analogous to Empirical Risk Minimization, it was shown that such estimators can be used to design learning algorithms for batch learning from logged bandit feedback. Batch learning from logged bandit feedback is an instance of causal inference."

  "Supervised learning is essentially observational: some data has been collected and subsequently algorithms are run on it. Online supervised learning doesn't necessarily work this way, but mostly online techniques have been used for computational reasons after data collection. In contrast, counterfactual learning is very difficult do to observationally. Diverse fields such as economics, political science, and epidemiology all attempt to make counterfactual conclusions using observational data, essentially because this is the only data available (at an affordable cost). When testing a new medicine, however, the standard is to run a controlled experiment, because with control over the data collection more complicated conclusions can be made with higher confidence. Analogously, reinforcement learning is best done “in the loop”, with the algorithm controlling the collection of data which is used for learning."

----

  "The UCB family of algorithms use the problem structure to derive tight optimistic upper bounds. While these algorithms are simple and have been used in various applications with success, they lack the ability to incorporate structured prior information such as arm dependency or different reward policies without requiring complex and difficult re-analysis of the upper bounds. Bayes-UCB is a Bayesian index policy that improves on UCB in Bayesian bandits by taking advantage of the prior distribution.

  Thompson sampling works by choosing an arm based on its probability of being the best arm. The method draws a sample from the decision maker’s current belief distribution for each arm and then chooses the arm that yielded the highest sample. The performance of Thompson sampling has been proved to be near optimal, and it is simple and efficient to implement. Thompson sampling can easily be adapted to a wide range of problem structures and prior distributions. For example, one can reject sets of samples that contradict contextual information. However, the simplicity of the method makes it also difficult to improve its performance.

  Gittins indices exploit the weak dependence between actions to compute the optimal action in time that is linear in the number of arms. Gittins indices, however, are guaranteed to be optimal only for the basic multi-armed bandit problem, require a discounted infinite-horizon objective, and provably cannot be extended to most interesting and practical problems which involve correlations between arms or an additional context."


  ["Multi Armed Bandits and Exploration Strategies"](http://sudeepraja.github.io/Bandits/)



---
### contextual bandits

  [history of contextual bandits](https://youtu.be/7ic_d5TeIUk?t=6m41s) by John Langford

  [overview](http://youtube.com/watch?v=N5x48g2sp8M) by Robert Schapire

  [overview](http://youtube.com/watch?v=fIKkhoI1kF4&t=19m22s) by Emma Brunskill


  ["Counterfactual Reasoning and Learning from Logged Data"](http://timvieira.github.io/blog/post/2016/12/19/counterfactual-reasoning-and-learning-from-logged-data/) by Tim Vieira

----

  "Contextual bandits are simple reinforcement learning problems without persistent state. At each step an agent is presented with a context x and a choice of one of K possible actions a. Different actions yield different unknown rewards r. The agent must pick the action that yields the highest expected reward. The context is assumed to be presented independent of any previous actions, rewards or contexts. An agent builds a model of the distribution of the rewards conditioned upon the action and the context: P(r|x,a,w). It then uses this model to pick its action. Note, importantly, that an agent does not know what reward it could have received for an action that it did not pick, a difficulty often known as “the absence of counterfactual”. As the agent’s model P(r|x,a,w) is trained online, based upon the actions chosen, unless exploratory actions are taken, the agent may perform suboptimally."

  "Removing credit assignment problem from reinforcement learning yields contextual bandit setting which is tractable similar to supervised learning problems."

  "In supervised learning you know how good actions you didn't take are as well, which is not the case in bandits and in reinforcement learning in general."

----

  goal:  
  - learn through experimentation (in policies from given class, not in states of environment) to do (almost) as well as best policy from policy class  
  - assume policy class finite, but typically extremely large  
  - policies may be very complex and expressive  

  problems:  
  - need to be learning about all policies simultaneously while also performing as well as the best one  
  - when action selected, only observe reward for policies that would have chosen same action  
  - exploration vs exploitation on gigantic scale (exploration in space of policies)  

  challenges:  
  - computational efficiency  
  - very large policy space  
  - optimal statistical performance (regret)  
  - adversarial setting  

  effective methods:  
  - obtain sound statistical estimates from biased samples  
  - learn highly complex behaviors (i.e. policies from very large space)  
  - attain efficiency using previously developed methods (i.e. oracle)  
  - harness power of combining policies  
  - achieve explicit conditions using an optimization approach  

----

  "Given experience (xt,at,pt,rt)* generated using some policy, it is possible to evaluate another policy π: x -> a using estimator:

  V(π) = 1/n * Σ (xt,at,pt,rt) [rt * I(π(x) = at) / pt]

  This estimator (inverse propensity scoring) has three important properties. First, it is data-efficient. Each interaction on which π matches the exploration data can be used in evaluating π, regardless of the policy collecting the data. In contrast, A/B testing only uses data collected using π to evaluate π. Second, the division by pt makes it statistically unbiased: it converges to the true reward as n → ∞. Third, the estimator can be recomputed incrementally when new data arrives.

  Thus, using a fixed exploration dataset, accurate counterfactual estimates of how arbitrary policies would have performed can be computed without actually running them in real time. This is precisely the question A/B testing attempts to answer, except A/B testing must run a live experiment to test each policy.

  Let us compare the statistical efficiency of MWT to that of A/B testing. Suppose N data points are collected using an exploration policy which places probability at least on each action (for EpsilonGreedy, ε = ε0/#actions), and we wish to evaluate K different policies. Then the ips estimators for all K policies have confidence intervals whose width is (C/(εN)\*log(K/δ))^1/2, with probability at least 1−δ, where C is a small absolute constant and δ > 0 and N > 1/ε. This is an exponential (in K) improvement over A/B testing since an A/B test of K policies collecting N data points has confidence intervals of width C\*(K/N\*log(K/δ))^1/2. This also shows the necessity of exploration for policy learning. If ε = 0, we cannot correctly evaluate arbitrary policies."

  "Contextual bandits allow testing and optimization over exponentially more policies for a given number of events. In one realistic scenario, one can handle 1 billion policies for the data collection cost of 21 A/B tests. The essential reason for such an improvement is that each data point can be used to evaluate all the policies picking the same action for the same context (i.e., make the same decision for the same input features rather than just a single policy as in A/B testing). An important property is that policies being tested do not need to be approved, implemented in production, and run live for a period of time (thus saving much business and engineering effort). Furthermore, the policies do not even need to be known during data collection."

----

  (John Langford) "The Atari results are very fun but obviously unimpressive on about 1/4 of the games. My hypothesis for why is that the solution does only local (epsilon-greedy style) exploration rather than global exploration so they can only learn policies addressing either very short credit assignment problems or with greedily accessible polices. Global exploration strategies are known to result in exponentially more efficient strategies in general for [deterministic decision processes](http://idm-lab.org/bib/abstracts/papers/aaai93.pdf) (1993), [Markov Decision Processes](http://www.cis.upenn.edu/~mkearns/papers/KearnsSinghE3.pdf) (1998), and for [MDPs without modelling](http://research.microsoft.com/pubs/178886/published.pdf) (2006). The reason these strategies are not used is because they are based on tabular learning rather than function fitting. That’s why I shifted to Contextual Bandit research after the 2006 paper. We’ve learned quite a bit there, enough to start tackling a [Contextual Deterministic Decision Process](http://arxiv.org/abs/1602.02722) (2016), but that solution is still far from practical."

  [global exploration strategies](#exploration-and-intrinsic-motivation)

----

  Multiworld Testing Decision Service:  
  - <https://mwtds.azurewebsites.net>  
  - <http://research.microsoft.com/en-us/projects/mwt/>  
  - <http://hunch.net/?p=4464948> (John Langford)  
  - <http://machinedlearnings.com/2017/01/reinforcement-learning-as-service.html> (Paul Mineiro)  
  - <https://youtube.com/watch?v=7ic_d5TeIUk> (John Langford)  
  - <https://youtu.be/N5x48g2sp8M?t=52m> (Robert Schapire)  

  ["Bayesian Multi-armed Bandits vs A/B Tests"](https://habrahabr.ru/company/ods/blog/325416/) (in russian)

  [example implementation](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c)



---
### model-based methods

  using transition model of environment p(r,s0|s,a)


----
#### Guided Policy Search

  [overview](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/) by Pieter Abbeel (part 2)  
  [overview](http://youtube.com/watch?v=EtMyH_--vnU) by Sergey Levine  
  [overview](https://video.seas.harvard.edu/media/ME+Sergey+Levine+2015+-04-01/1_gqqp9r3o/23375211) by Sergey Levine  
  [overview](http://youtube.com/watch?v=xMHjkZBvnfU) by Pieter Abbeel  

  ["Optimal Control and Trajectory Optimization"](https://youtube.com/watch?v=mZtlW_xtarI) by Sergey Levine  
  ["Learning Policies by Imitating Optimal Control"](https://youtube.com/watch?v=o0Ebur3aNMo) by Sergey Levine  


  "GPS relies on learning an underlying dynamical model of the environment and then, at each iteration of the algorithm, using that model to gradually improve the policy."  

  "Suppose you want to train a neural net policy that can solve a fairly broad class of problems. Here’s one approach:  
  - Sample 10 instances of the problem, and solve each of the instances using a problem-specific method, e.g. a method that fits and uses an instance-specific model.  
  - Train the neural net to agree with all of the per-instance solutions.  
  But if you’re going to do that, you might do even better by constraining the specific solutions and what the neural net policy would do to be close to each other from the start, fitting both simultaneously."  

  "GPS trains a policy for accomplishing a given task by guiding the learning with multiple guiding distributions."  
  "GPS uses modified importance sampling to get gradients, where samples are obtained via trajectory optimization."  


  challenges:  
  - much weaker supervision: cost for being in a state but current state is consequence of initial state, control, and noise (temporal credit assignment problem)  
  - distribution over observed states determined by agent's own actions (need for exploration)  

  solutions:  
  - optimal trajectories are not sufficiently informative -> distribution over near-optimal trajectories  
  - representational mismatch trajectory distribution vs neural net -> constrained guided policy search  


  implementations:  
  - <http://rll.berkeley.edu/gps/>  
  - <https://github.com/cbfinn/gps>  
  - <https://github.com/nivwusquorum/guided-policy-search/>  

  [interesting papers](#interesting-papers---behavioral-cloning)


----
#### Monte Carlo Tree Search

  [introduction](https://youtube.com/watch?v=ItMutbeOHtc&t=1h4m32s) by David Silver  
  [introduction](https://youtube.com/watch?v=mZtlW_xtarI&t=45m12s) by Sergey Levine  


  ["A Survey of Monte Carlo Tree Search Methods"](http://www.cameronius.com/cv/mcts-survey-master.pdf) by Browne et al.


  ["Mastering the Game of Go with Deep Neural Networks and Tree Search"](#silver-et-al---mastering-the-game-of-go-with-deep-neural-networks-and-tree-search) by Silver et al.  
  ["Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning"](#guo-singh-lee-lewis-wang---deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning) by Guo et al.  
  ["A Monte-Carlo AIXI Approximation"](https://github.com/brylevkirill/notes/blob/Artificial%20Intelligence.md#veness-ng-hutter-uther-silver---a-monte-carlo-aixi-approximation-mc-aixi-ctw-agent) by Veness et al.  

----
#### deep model-based learning

  challenges:  
  - compounding errors  
    * errors in the transition model compound over the trajectory  
    * by the end of a long trajectory rewards can be totally wrong  
    * no success yet in Atari  
  - deep networks of value/policy can plan implicitly  
    * each layer of network performs arbitrary computational step  
    * n-layer network can “lookahead” n steps  
    * are transition models required at all?  


  ["Learning Dynamical System Models from Data"](https://youtube.com/watch?v=qVsLk5CVy_c) by Sergey Levine  
  ["Advanced Model Learning"](https://youtube.com/watch?v=6EasN2FAIX0) by Chelsea Finn  

  ["Deep Recurrent Q-Network"](https://youtube.com/watch?v=bE5DIJvZexc) by Alexander Fritzler (in russian)  
  ["Deep Reinforcement Learning with Memory"](http://93.180.23.59/videos/video/2420/in/channel/1/) by Sergey Bartunov (in russian)  

  ["Deep AutoRegressive Networks"](https://youtu.be/-yX1SYeDHbg?t=49m25s) by Alex Graves  
  ["Deep AutoRegressive Networks"](https://youtu.be/P78QYjWh5sM?t=20m50s) by Karol Gregor  


----
#### bayesian reinforcement learning

  BRL agent aims to maximise expected sum of future rewards obtained when interacting with unknown Markov Decision Process while using some prior knowledge.

  BRL agent maintains a distribution over worlds and either, samples a world and acts as if it is real, or chooses action reasoning about full distribution.

  Belief-augmented Markov Decision Process is an MDP obtained when considering augmented states made of concatenation of actual state and posterior beliefs.

  Bayes-Adaptive Markov Decision Processes form a natural framework to deal with sequential decision-making problems when some of the information is hidden. In these problems, an agent navigates in an initially unknown environment and receives a numerical reward according to its actions. However, actions that yield the highest instant reward and actions that maximise the gathering of knowledge about the environment are often different. The BAMDP framework leads to a rigorous definition of an optimal solution to this learning problem, which is based on finding a policy that reaches an optimal balance between exploration and exploitation.

  bayesian policy search in variational MDP (variational decision making):  
  Fπ(θ) = E q(a,z|x) [R(a|x)] - α * Dkl[qθ(z|x) || p(z|x)] + α * H[πθ(a|z)]  


  [overview](https://youtu.be/AggqBRdz6CQ?t=9m53s) by Shakir Mohamed  
  [overview](https://youtube.com/watch?v=_dkaynuKUFEby) by Alexey Seleznev (in russian)  


  ["Efficient Bayes-Adaptive Reinforcement Learning using Sample-Based Search"](#guez-silver-dayan---efficient-bayes-adaptive-reinforcement-learning-using-sample-based-search) by Guez et al.  
  ["Approximate Bayes Optimal Policy Search using Neural Networks"](#castronovo-francois-lavet-fonteneau-ernst-couetoux---approximate-bayes-optimal-policy-search-using-neural-networks) by Castronovo et al.  


  [interesting papers](#interesting-papers---model-based-methods) on model-based methods  
  [interesting papers](#interesting-papers---abstractions-for-states-and-actions) on abstractions for states and actions  



---
### value-based methods

  ["Markov Decision Process"](https://youtube.com/watch?v=lfHX2hHRMVQ) by David Silver  
  ["Planning by Dynamic Programming"](https://youtube.com/watch?v=Nd1-UUMVfz4) by David Silver  
  ["Model-Free Prediction"](https://youtube.com/watch?v=PnHCvfgC_ZA) by David Silver  
  ["Model Free Control"](https://youtube.com/watch?v=0g4j2k_Ggc4) by David Silver  
  ["Value Function Approximation"](http://youtube.com/watch?v=UoPei5o4fps) by David Silver  

  ["Value Iteration and Policy Iteration"](https://youtube.com/watch?v=IL3gVyJMmhg) by John Schulman  
  "Q-Function Learning Methods" by John Schulman
	([first part](https://youtube.com/watch?v=Wnl-Qh2UHGg&t=19m06s),
	[second part](https://youtube.com/watch?v=h1-pj4Y9-kM))

  [introduction to deep Q-learning](http://youtube.com/watch?v=dV80NAlEins) by Nando de Freitas  
  [introduction to deep Q-learning](http://youtube.com/watch?v=HUmEbUkeQHg) by Nando de Freitas  

  [tutorial](http://techtalks.tv/talks/deep-reinforcement-learning/62360/) by David Silver  
  [tutorial](http://youtu.be/qLaDWKd61Ig?t=9m16s) by David Silver  
  [tutorial](http://videolectures.net/rldm2015_silver_reinforcement_learning/) by David Silver  

  [overview](http://youtube.com/watch?v=mrgJ53TIcQc) by Mikhail Pavlov (in russian)  


  - naive Q-learning oscillates or diverges with neural nets  
    * data is sequential  
    * successive samples are correlated, non-iid  
  - policy changes rapidly with slight changes to Q-values  
    * policy may oscillate  
    * distribution of data can swing from one extreme to another  
  - scale of rewards and Q-values is unknown  
    * naive Q-learning gradients can be large  
    * unstable when backpropagated  

**Deep Q-learning Network (DQN)**:  
  - use experience replay  
    * break correlations in data, bring us back to iid setting  
    * learn from all past policies  
    * using off-policy Q-learning  
  - freeze target Q-network  
    * avoid oscillations  
    * break correlations between Q-network and target  
  - clip rewards or normalize network adaptively to sensible range  
    * robust gradients  


  [interesting papers](#interesting-papers---value-based-methods)


  implementations:  
  - <https://github.com/khanhptnk/deep-q-tensorflow>  
  - <https://github.com/nivwusquorum/tensorflow-deepq>  
  - <https://github.com/devsisters/DQN-tensorflow>  
  - <https://github.com/carpedm20/deep-rl-tensorflow>  
  - <https://github.com/VinF/deer>  
  - <https://github.com/osh/kerlym>  
  - <https://github.com/Jabberwockyll/deep_rl_ale>  
  - <https://github.com/DanielTakeshi/rl_algorithms/blob/master/dqn/dqn.py>  



---
### policy-based methods

  [introduction](http://youtube.com/watch?v=kUiR0RLmGCo) by Nando de Freitas  
  [introduction](http://youtube.com/watch?v=KHZVXao4qXs) by David Silver  

  [introduction](http://karpathy.github.io/2016/05/31/rl/) by Andrej Karpathy  
  [introduction](https://dropbox.com/s/yefei7380x7jeo7/Deep%20Reinforcement%20Learning%20Tutorial%20%28OpenAI%29.html) by John Schulman  

  [tutorial](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Reinforcement-Learning-Through-Policy-Optimization) by Pieter Abbeel and John Schulman ([slides](http://people.eecs.berkeley.edu/~pabbeel/nips-tutorial-policy-optimization-Schulman-Abbeel.pdf))  
  [tutorial](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/) by Pieter Abbeel  
  [tutorial](https://youtube.com/watch?v=PtAIh9KSnjo) by John Schulman  

  [overview](https://youtu.be/mrgJ53TIcQc?t=41m35s) by Alexey Seleznev (in russian)

  course by John Schulman:  
	https://youtube.com/watch?v=BB-BhTn6DCM  
	https://youtube.com/watch?v=_t5fpZuuf-4  
	https://youtube.com/watch?v=Fauwwkiy-bo  
	https://youtube.com/watch?v=IDSA2wAACr0  

  course by John Schulman:  
	https://youtube.com/watch?v=aUrX-rP_ss4  
	https://youtube.com/watch?v=oPGVsoBonLM  
	https://youtube.com/watch?v=rO7Dx8pSJQw  
	https://youtube.com/watch?v=gb5Q2XL5c8A  


  ["Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs"](http://joschu.net/docs/thesis.pdf) by John Schulman


  [interesting papers](#interesting-papers---policy-based-methods)

----

  methods:  
  - derivative-free  
    * [Cross-Entropy Method](#cross-entropy-method-cem)  (no policy gradient estimation)  
    * [Evolution Strategies](#evolution-strategies-es)  (derivative-free policy gradient estimation using finite differences)  
  - likelihood ratio policy gradient  
    * [REINFORCE](#reinforce)  (policy gradient estimation using simple baseline for returns)  
    * [Trust Region Policy Optimization](#trust-region-policy-optimization-trpo)  (policy gradient estimation using natural gradient / trust region)  
    * [Actor-Critic](#actor-critic-ac), [Generalized Advantage Estimation](#generalized-advantage-estimation-gae), [Asynchronous Advantage Actor-Critic](#asynchronous-advantage-actor-critic-a3c)  (policy gradient estimation using critic as baseline for returns)  
  - pathwise derivative policy gradient  
    * [Deep Deterministic Policy Gradient](#deep-deterministic-policy-gradient-ddpg), [Stochastic Value Gradient](#stochastic-value-gradient-svg)  (policy gradient estimation using gradient of critic as model of returns)  
  - [stochastic computation graphs](https://arxiv.org/abs/1506.05254)  (policy gradient estimation using both likelihood ratio and pathwise derivative)  


  what's the right core model-free algorithm is not clear:  
  - *derivative-free policy optimization*:  scalable, very sample-inefficient, more robust, no off-policy  
  - *policy gradient optimization*:  scalable, not sample-efficient, not robust, no off-policy  
  - *trust region policy optimization*:  less scalable, more sample-efficient, more robust, no off-policy  
  - *value-based policy optimization*:  scalable in state space, more sample-efficient, not robust, more off-policy  


  ["Benchmarking Deep Reinforcement Learning for Continuous Control"](http://arxiv.org/abs/1604.06778) by Duan, Chen, Houthooft, Schulman, Abbeel
	([video](http://techtalks.tv/talks/benchmarking-deep-reinforcement-learning-for-continuous-control/62380/), [code](https://github.com/openai/rllab))

----

  "Policy gradient methods are attractive because they are end-to-end: there is explicit policy and principled approach that directly optimizes expected reward."  
  "Policy gradient methods work well only in settings where there are few discrete choices so that algorithm is not hopelessly sampling through huge search space."  

  limitations of policy gradient optimization:  
  - inefficient use of data, large number of samples required  
    * each experience is only used to compute one gradient (on-policy)  
    * given a batch of trajectories what's the most we can do with it?  
  - hard to choose reasonable stepsize that works for the whole optimization  
    * we have a gradient estimate, no objective for line search  
    * statistics of data (observations and rewards) change during learning  

  "For reinforcement learning there are two widely known ways of optimizing a policy based on sampled sequences of actions and outcomes: There’s (a) likelihood-ratio gradient estimator, which updates the policy such that action sequences that lead to higher scores happen more often and that doesn’t need gradients, and (b) pathwise derivative gradient estimator, which adjusts individual actions such that the policy results in a higher score and that needs gradients. Likelihood-ratio estimator changes probabilities of experienced paths by shifting probability mass towards better ones and, unlike pathwise estimator, does not try to change the paths. While pathwise methods may be more sample-efficient, they work less generally due to high bias and don’t scale up as well to very high-dimensional problems."



----
#### Cross-Entropy Method (CEM)

  no policy gradient estimation, evolutionary algorithm with selection buth without recombination and mutation

  "If your policy has a small number of parameters (say 20), and sometimes even if it has a moderate number (say 2000), you might be better off using the Cross-Entropy Method than any of the fancy methods. It works like this:  
  - Sample n sets of parameters from some prior that allows for closed-form updating, e.g. a multivariate Gaussian.  
  - For each parameter set, compute a noisy score by running your policy on the environment you care about.  
  - Take the top 20% percentile (say) of sampled parameter sets. Fit a Gaussian distribution to this set, then go to (1) and repeat using this as the new prior."  


  [overview](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement) by Pieter Abbeel (08:37)  
  [overview](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Reinforcement-Learning-Through-Policy-Optimization) by Pieter Abbeel (07:07)  
  [overview](https://youtu.be/aUrX-rP_ss4?t=27m20s) by John Schulman  


  ["The Cross Entropy method for Fast Policy Search"](http://aaai.org/Papers/ICML/2003/ICML03-068.pdf) by Mannor, Rubinstein, Gat


  implementations:  
  - <https://github.com/openai/gym/blob/master/examples/agents/cem.py>  
  - <https://github.com/openai/rllab/blob/master/rllab/algos/cem.py>  
  - <https://github.com/matthiasplappert/keras-rl/blob/master/rl/agents/cem.py>  



----
#### Evolution Strategies (ES)

  policy gradient estimation using finite differences instead of derivative of loss function

  (Juergen Schmidhuber) "Evolutionary computation is one of the most useful practical methods for direct search in policy space, especially when there is no teacher who knows which output actions the system should produce at which time. Especially in partially observable environments where memories of previous events are needed to disambiguate states, this often works much better than other reinforcement learning techniques based on dynamic programming. In case of teacher-given desired output actions or labels, gradient descent such as backpropagation (also through time) usually works much better, especially for NNs with many weights."


  [overview](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement) by Pieter Abbeel (13:04)


  ["Natural Evolution Strategies"](http://jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) by Wierstra, Schaul, Glasmachers, Sun, Peters, Schmidhuber

----

  ["Evolution Strategies as a Scalable Alternative to Reinforcement Learning"](#salimans-ho-chen-sutskever---evolution-strategies-as-a-scalable-alternative-to-reinforcement-learning) by Salimans, Ho, Chen, Sutskever

  [overview](https://www.technologyreview.com/s/603916/a-new-direction-for-artificial-intelligence/) by Ilya Sutskever  
  [overview](https://youtube.com/watch?v=Rd0UdJFYkqI) by Pavel Temirchev (in russian)  


  <https://blog.openai.com/evolution-strategies/> :  
>	"Our work demonstrates that ES achieves strong performance, dispelling the common belief that ES methods are impossible to apply to high dimensional problems."  
>	"ES rivals the performance of standard RL techniques on modern benchmarks, while overcoming many of RL’s inconveniences. ES is simpler to implement (there is no need for backpropagation), it is easier to scale in a distributed setting, it does not suffer in settings with sparse rewards, and has fewer hyperparameters. This outcome is surprising because ES resembles simple hill-climbing in a high-dimensional space based only on finite differences along a few random directions at each step."  
>
>	"Mathematically, you’ll notice that this is also equivalent to estimating the gradient of the expected reward in the parameter space using finite differences, except we only do it along 100 random directions. Yet another way to see it is that we’re still doing RL (Policy Gradients, or REINFORCE specifically), where the agent’s actions are to emit entire parameter vectors using a gaussian policy."  
>	"Notice that the objective is identical to the one that RL optimizes: the expected reward. However, RL injects noise in the action space and uses backpropagation to compute the parameter updates, while ES injects noise directly in the parameter space. Another way to describe this is that RL is a “guess and check” on actions, while ES is a “guess and check” on parameters. Since we’re injecting noise in the parameters, it is possible to use deterministic policies (and we do, in our experiments). It is also possible to add noise in both actions and parameters to potentially combine the two approaches."  
>
>	"ES enjoys multiple advantages over RL algorithms:  
>	No need for backpropagation. ES only requires the forward pass of the policy and does not require backpropagation (or value function estimation), which makes the code shorter and between 2-3 times faster in practice. On memory-constrained systems, it is also not necessary to keep a record of the episodes for a later update. There is also no need to worry about exploding gradients in RNNs. Lastly, we can explore a much larger function class of policies, including networks that are not differentiable (such as in binary networks), or ones that include complex modules (e.g. pathfinding, or various optimization layers).  
>	Highly parallelizable. ES only requires workers to communicate a few scalars between each other, while in RL it is necessary to synchronize entire parameter vectors (which can be millions of numbers). Intuitively, this is because we control the random seeds on each worker, so each worker can locally reconstruct the perturbations of the other workers. Thus, all that we need to communicate between workers is the reward of each perturbation. As a result, we observed linear speedups in our experiments as we added on the order of thousands of CPU cores to the optimization.  
>	Structured exploration. Some RL algorithms (especially policy gradients) initialize with random policies, which often manifests as random jitter on spot for a long time. This effect is mitigated in Q-Learning due to epsilon-greedy policies, where the max operation can cause the agents to perform some consistent action for a while (e.g. holding down a left arrow). This is more likely to do something in a game than if the agent jitters on spot, as is the case with policy gradients. Similar to Q-learning, ES does not suffer from these problems because we can use deterministic policies and achieve consistent exploration.  
>	Credit assignment over long time scales. By studying both ES and RL gradient estimators mathematically we can see that ES is an attractive choice especially when the number of time steps in an episode is big, where actions have longlasting effects, or if no good value function estimates are available."  


  [overview](http://inference.vc/evolutionary-strategies-embarrassingly-parallelizable-optimization/) by Ferenc Huszar  
  [overview](http://inference.vc/evolution-strategies-variational-optimisation-and-natural-es-2/) by Ferenc Huszar  
  [overview](http://davidbarber.github.io/blog/2017/04/03/variational-optimisation/) by David Barber  
  [overview](http://argmin.net/2017/04/03/evolution/) by Ben Recht and Roy Frostig  


  https://en.wikipedia.org/wiki/Simultaneous\_perturbation\_stochastic\_approximation  
  ["Stochastic Gradient Estimation with Finite Differences"](http://approximateinference.org/accepted/BuesingEtAl2016.pdf) by Buesing, Weber, Mohamed  


  implementations:  
  - <https://github.com/openai/evolution-strategies-starter>  
  - <https://github.com/atgambardella/pytorch-es>  
  - <https://github.com/mdibaiee/flappy-es>  
  - <https://gist.github.com/kashif/5748e199a3bec164a867c9b654e5ffe5>  
  - <https://github.com/DanielTakeshi/rl_algorithms/blob/master/es/basic_es.py>  



----
#### REINFORCE

  likelihood ratio policy gradient estimation


  ["The Useless Beauty of REINFORCE"](https://theneural.wordpress.com/2011/09/13/the-useless-beauty-of-reinforce/) by Ilya Sutskever

  [introduction](http://karpathy.github.io/2016/05/31/rl) by Andrej Karpathy  
  [introduction](http://kvfrans.com/simple-algoritms-for-solving-cartpole/) by Kevin Frans  

  [overview](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/) by Pieter Abbeel (16:43)  
  overview by John Schulman ([part 1](https://youtube.com/watch?v=oPGVsoBonLM), [part 2](https://youtube.com/watch?v=oPGVsoBonLM))  


  implementations:  
  - <https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5>  
  - <https://github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py> + <http://kvfrans.com/simple-algoritms-for-solving-cartpole/>  
  - <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/bayesflow/examples/reinforce_simple/reinforce_simple_example.py>  
  - <https://github.com/osh/kerlym/blob/master/kerlym/pg.py>  
  - <https://github.com/rllab/rllab/blob/master/rllab/algos/vpg.py>  
  - <https://github.com/DanielTakeshi/rl_algorithms/tree/master/vpg>  



----
#### Trust Region Policy Optimization (TRPO)

  second order policy gradient algorithm

  ["Trust Region Policy Optimization"](#schulman-levine-moritz-jordan-abbeel---trust-region-policy-optimization) by Schulman et al.

----

  "As you iteratively improve your policy, it’s important to avoid parameter updates that change your policy too much, as enforced by constraining the KL divergence between the distributions predicted by the old and the new policy on a batch of data to be less than some constant δ. This δ (in the unit of nats) is better than a fixed step size, since the meaning of the step size changes depending on what the rewards and problem structure look like at different points in training. This is called Trust Region Policy Optimization (or, in a first-order variant, Proximal Policy Optimization) and it matters more as we do more experience replay. Instead of conjugate gradients the simplest instantiation of this idea could be implemented by doing a line search and checking the KL along the way."

  "To improve its policy, TRPO attempts to maximize the expectation of Q-values over the distribution of states and actions given by θnew:

  maxθ [Σs pθ(s) * (Σa πθ(a|s) * Qθold(s,a))]  subject to  Dkl(pθold, pθ) ≤ δ

  This objective can be approximated by using an importance-sampled Monte Carlo estimate of Q-values, with a distribution of states sampled from policy θold. However, theres a constraint to updating θ: the average KL divergence between the new policy and old policy cannot be greater than a constant δ. This acts as a limiter on the step size we can take on each update, and can be compared to the natural gradient. The theory behind TRPO guarantees gradual improvement over the expected return of a policy."

  "One downside to TRPO algorithm is its on-policy nature, requiring new Q-values after every policy update. We cannot use methods such as experience replay which reuse past information, so that we must acquire new Monte Carlo estimates of Q for every new policy. Furthermore, Monte Carlo estimates are known to have higher variance than methods such as one-step TD updates, since the return is affected by independent future decisions. Bringing this variance down requires many episodes of experience per policy update, making TRPO a data-heavy algorithm."

----

  [overview](https://youtu.be/xe-z4i3l-iQ?t=30m35s) by Pieter Abbeel  
  [overview](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/) by Pieter Abbeel (27:10)  
  [overview](https://youtube.com/watch?v=gb5Q2XL5c8A) by John Schulman  

  [explanation of natural gradient in TRPO](http://kvfrans.com/what-is-the-natural-gradient-and-where-does-it-appear-in-trust-region-policy-optimization/) by Kevin Frans


  implementations:  
  - <https://github.com/joschu/modular_rl>  
  - <https://github.com/rll/deeprlhw2/blob/master/ppo.py>  
  - <https://github.com/wojzaremba/trpo>
  - <https://github.com/rllab/rllab/blob/master/rllab/algos/trpo.py>  
  - <https://github.com/kvfrans/parallel-trpo>  



----
#### Actor-Critic (AC)

  critic provides loss function for actor, gradient backpropagates from critic into actor


  [introduction](http://incompleteideas.net/sutton/book/ebook/node66.html) by Sutton and Barto

  [overview](http://videolectures.net/rldm2015_silver_reinforcement_learning) by David Silver (1:07:23)  
  [overivew](https://youtu.be/qLaDWKd61Ig?t=38m58s) by David Silver  
  [overview](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement) by Pieter Abbeel (0:49:45)  
  [overview](https://youtu.be/rO7Dx8pSJQw?t=50m) by John Schulman  
  [overview](https://youtu.be/mrgJ53TIcQc?t=1h3m2s) by Alexey Seleznev (in russian)  


  - [Generalized Advantage Estimation (GAE)](#generalized-advantage-estimation-gae)  
  - [Asynchronous Advantage Actor-Critic (A3C)](#asynchronous-advantage-actor-critic-a3c)  

----

  "In advantage learning one throws away information that is not needed for coming up with a good policy. The argument is that throwing away information allows you to focus your resources on learning what is important. As an example consider Tetris when you gain a unit reward for every time step you survive. Arguably the optimal value function takes on large values when the screen is near empty, while it takes on small values when the screen is near full. The range of differences can be enormous (from millions to zero). However, for optimal decision making how long you survive does not matter. What matters is the small differences in how the screen is filled up because this is what determines where to put the individual pieces. If you learn an action value function and your algorithm focuses on something like the mean square error, i.e., getting the magnitudes right, it is very plausible that most resources of the learning algorithm will be spent on capturing how big the values are, while little resource will be spent on capturing the value differences between the actions. This is what advantage learning can fix. The fix comes because advantage learning does not need to wait until the value magnitudes are properly captured before it can start learning the value differences. As can be seen from this example, advantage learning is expected to make a bigger difference where the span of optimal values is orders of magnitudes larger than action-value differences."

  *(Csaba Szepesvari)*


----
#### Generalized Advantage Estimation (GAE)

  ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](#schulman-moritz-levine-jordan-abbeel---high-dimensional-continuous-control-using-generalized-advantage-estimation) by Schulman et al.


  [overview](https://youtu.be/xe-z4i3l-iQ?t=30m35s) by Pieter Abbeel  
  [overview](https://youtu.be/rO7Dx8pSJQw?t=40m20s) by John Schulman  

  <https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/>


  implementations:  
  - <https://github.com/joschu/modular_rl>  
  - <https://github.com/rll/deeprlhw2/blob/master/ppo.py>  


----
#### Asynchronous Advantage Actor-Critic (A3C)

  ["Asynchronous Methods for Deep Reinforcement Learning"](#mnih-badia-mirza-graves-lillicrap-harley-silver-kavukcuoglu---asynchronous-methods-for-deep-reinforcement-learning) by Mnih et al.


  - critic learns only state value function V(s) rather than action value function Q(s,a) and thus cannot pass back to actor gradients of value function with respect to action  
  - critic approximates action value with rewards from several steps of experience and passes TD error to actor  
  - exploiting multithreading capabilities and executing many instances of agent in parallel using shared model  
  - alternative to experience replay since parallelization also diversifies and decorrelates experience data  


  [overview](https://youtube.com/watch?v=9sx1_u2qVhQ) by Andriy Mnih  
  [overview](http://techtalks.tv/talks/asynchronous-methods-for-deep-reinforcement-learning/62475/) by Andriy Mnih  


  implementations:  
  - <https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb>  
  - <https://github.com/yandexdataschool/AgentNet/blob/master/agentnet/learning/a2c_n_step.py>  
  - <https://github.com/Zeta36/Asynchronous-Methods-for-Deep-Reinforcement-Learning>  
  - <https://github.com/miyosuda/async_deep_reinforce>  
  - <https://github.com/muupan/async-rl>  
  - <https://github.com/coreylynch/async-rl>  
  - <https://github.com/carpedm20/deep-rl-tensorflow/blob/master/agents/async.py>  
  - <https://github.com/ikostrikov/pytorch-a3c>  
  - <https://github.com/danijar/mindpark/blob/master/mindpark/algorithm/a3c.py>  



----
#### pathwise derivative policy gradient

  "For reinforcement learning there are two widely known ways of optimizing a policy based on sampled sequences of actions and outcomes: There’s (a) likelihood-ratio gradient estimator, which updates the policy such that action sequences that lead to higher scores happen more often and that doesn’t need gradients, and (b) pathwise derivative gradient estimator, which adjusts individual actions such that the policy results in a higher score and that needs gradients. Likelihood-ratio estimator changes probabilities of experienced paths by shifting probability mass towards better ones and, unlike pathwise estimator, does not try to change the paths. While pathwise methods may be more sample-efficient, they work less generally due to high bias and don’t scale up as well to very high-dimensional problems."


  [overview](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/) by Pieter Abbeel (01:02:04)


  - [Deep Deterministic Policy Gradient (DDPG)](#deep-deterministic-policy-gradient-ddpg)  
  - [Stochastic Value Gradient (SVG)](#stochastic-value-gradient-svg)  


----
#### Deep Deterministic Policy Gradient (DDPG)

  ["Deterministic Policy Gradient Algorithms"](#silver-lever-heess-degris-wierstra-riedmiller---deterministic-policy-gradient-algorithms) by Silver et al.  
  ["Continuous Control with Deep Reinforcement Learning"](#lillicrap-hunt-pritzel-heess-erez-tassa-silver-wierstra---continuous-control-with-deep-reinforcement-learning) by Lillicrap et al.  


  - continuous analogue to DQN which exploits differentiability of Q-network  
  - instead of requiring samples from stochastic policy and encouraging samples with higher scores, use deterministic policy and get gradient information directly from second network that models score function  
  - policy determinism allows policy to be optimized more easily and more sample efficiently due to action no longer being a random variable which must be integrated over in expectation  
  - can be much more efficient in settings with very high-dimensional actions where sampling actions provides poor coverage of state-action space  

  in continuous multidimensional action space ∇aQμ(s,a) tells how to improve action:  
  ∇θJ(μθ) = ∫ ρμ(s)∇aQμ(s,a)|a=μθ(s)∇θμθ(s) ds = E s\~ρμ [∇aQμ(s,a)|a=μθ(s)∇θμθ(s)]  


  [overview](http://videolectures.net/rldm2015_silver_reinforcement_learning/) by David Silver (1:07:23)  
  [overview](http://youtu.be/qLaDWKd61Ig?t=39m) by David Silver  
  [overview](http://youtu.be/KHZVXao4qXs?t=52m58s) by David Silver  
  [overview](http://youtu.be/M6nfipCxQBc?t=7m45s) by Timothy Lillicrap  
  [overview](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/) by Pieter Abbeel (1:02:04)  
  [overview](https://youtu.be/rO7Dx8pSJQw?t=50m) by John Schulman  
  [overview](https://youtu.be/mrgJ53TIcQc?t=1h3m2s) by Alexey Seleznev (in russian)  


  implementations:  
  - <https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html>  
  - <http://pemami4911.github.io/blog_posts/2016/08/21/ddpg-rl.html>  
  - <https://github.com/matthiasplappert/keras-rl/blob/master/rl/agents/ddpg.py>  
  - <https://github.com/rllab/rllab/blob/master/rllab/algos/ddpg.py>  
  - <https://github.com/yandexdataschool/AgentNet/blob/master/agentnet/learning/dpg_n_step.py>  
  - <https://github.com/MOCR/DDPG>  


----
#### Stochastic Value Gradient (SVG)

  ["Learning Continuous Control Policies by Stochastic Value Gradients"](#heess-wayne-silver-lillicrap-tassa-erez---learning-continuous-control-policies-by-stochastic-value-gradients) by Heess et al.


  - generalizes DPG to stochastic policies in a number of ways, giving a spectrum from model-based to model-free algorithms  
  - while SVG(0) is a direct stochastic generalization of DPG, SVG(1) combines an actor, critic and dynamics model f  
  - the actor is trained through a combination of gradients from the critic, model and reward simultaneously  

  reparametrization trick: E p(y|x)[g(y)]=∫g(f(x,ξ))ρ(ξ)dξ where y=f(x,ξ) and ξ\~ρ(.) a fixed noise distribution


  [overview](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/) by Pieter Abbeel (1:02:04)  
  [overview](https://youtu.be/rO7Dx8pSJQw?t=50m) by John Schulman  
  [overview](https://youtu.be/mrgJ53TIcQc?t=1h10m31s) by Alexey Seleznev (in russian)  



---
### interesting papers

[selected papers and books](https://dropbox.com/sh/zc5qxqksgqmxs0a/AAA4C1y_6Y0-3dm3gPuQhb_va)


[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md)


interesting papers (see below):  
  - [applications](#interesting-papers---applications)  
  - [exploration and intrinsic motivation](#interesting-papers---exploration-and-intrinsic-motivation)  
  - [abstractions for states and actions](#interesting-papers---abstractions-for-states-and-actions)  
  - [model-based methods](#interesting-papers---model-based-methods)  
  - [value-based methods](#interesting-papers---value-based-methods)  
  - [policy-based methods](#interesting-papers---policy-based-methods)  
  - [behavioral cloning](#interesting-papers---behavioral-cloning)  
  - [inverse reinforcement learning](#interesting-papers---inverse-reinforcement-learning)  



---
### interesting papers - applications

[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reinforcement-learning---applications)


#### Li - ["Deep Reinforcement Learning: An Overview"](https://arxiv.org/abs/1701.07274)
>	"We give an overview of recent exciting achievements of deep reinforcement learning. We start with background of deep learning and reinforcement learning, as well as introduction of testbeds. Next we discuss Deep Q-Network and its extensions, asynchronous methods, policy optimization, reward, and planning. After that, we talk about attention and memory, unsupervised learning, and learning to learn. Then we discuss various applications of RL, including games, in particular, AlphaGo, robotics, spoken dialogue systems (a.k.a. chatbot), machine translation, text sequence prediction, neural architecture design, personalized web services, healthcare, finance, and music generation. We mention topics/papers not reviewed yet. After listing a collection of RL resources, we close with discussions."


#### Silver et al. - ["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
>	"The game of Go has long been viewed as the most challenging of classic games for artificial intelligence due to its enormous search space and the difficulty of evaluating board positions and moves. We introduce a new approach to computer Go that uses value networks to evaluate board positions and policy networks to select moves. These deep neural networks are trained by a novel combination of supervised learning from human expert games, and reinforcement learning from games of self-play. Without any lookahead search, the neural networks play Go at the level of state-of-the-art Monte-Carlo tree search programs that simulate thousands of random games of self-play. We also introduce a new search algorithm that combines Monte-Carlo simulation with value and policy networks. Using this search algorithm, our program AlphaGo achieved a 99.8% winning rate against other Go programs, and defeated the European Go champion by 5 games to 0. This is the first time that a computer program has defeated a human professional player in the full-sized game of Go, a feat previously thought to be at least a decade away."

----
>	"Google AlphaGo is a historical tour of AI ideas: 70s (Alpha-Beta), 80s/90s (reinforcement learning & self-play), 00's (Monte-Carlo), 10's (deep neural networks)."  
>	"The most important application of reinforcement learning here is to learn a value function which aims to predict with which probability a certain position will lead to winning the game. The learned expert moves are already good, but the network that produces them did not learn with the objective to win the game, but only to minimize the differences to the teacher values in the training data set."  

  - <http://youtube.com/watch?v=4D5yGiYe8p4> (Silver)
  - <http://youtube.com/watch?v=LX8Knl0g0LE> (Huang)
  - <http://youtube.com/watch?v=UMm0XaCFTJQ> (Sutton, Szepesvari, Bowling, Hayward, Muller)
  - <https://github.com/Rochester-NRT/RocAlphaGo/wiki> (overview)
  - <https://github.com/Rochester-NRT/AlphaGo/>


#### Moravcik et al. - ["DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker"](http://arxiv.org/abs/1701.01724)
>	"Artificial intelligence has seen a number of breakthroughs in recent years, with games often serving as significant milestones. A common feature of games with these successes is that they involve information symmetry among the players, where all players have identical information. This property of perfect information, though, is far more common in games than in real-world problems. Poker is the quintessential game of imperfect information, and it has been a longstanding challenge problem in artificial intelligence. In this paper we introduce DeepStack, a new algorithm for imperfect information settings such as poker. It combines recursive reasoning to handle information asymmetry, decomposition to focus computation on the relevant decision, and a form of intuition about arbitrary poker situations that is automatically learned from selfplay games using deep learning. In a study involving dozens of participants and 44,000 hands of poker, DeepStack becomes the first computer program to beat professional poker players in heads-up no-limit Texas hold’em. Furthermore, we show this approach dramatically reduces worst-case exploitability compared to the abstraction paradigm that has been favored for over a decade."

>	"DeepStack is the first computer program to defeat professional poker players at heads-up nolimit Texas Hold’em, an imperfect information game with 10160 decision points. Notably it achieves this goal with almost no domain knowledge or training from expert human games. The implications go beyond just being a significant milestone for artificial intelligence. DeepStack is a paradigmatic shift in approximating solutions to large, sequential imperfect information games. Abstraction and offline computation of complete strategies has been the dominant approach for almost 20 years. DeepStack allows computation to be focused on specific situations that arise when making decisions and the use of automatically trained value functions. These are two of the core principles that have powered successes in perfect information games, albeit conceptually simpler to implement in those settings. As a result, for the first time the gap between the largest perfect and imperfect information games to have been mastered is mostly closed. As “real life consists of bluffing... deception... asking yourself what is the other man going to think”, DeepStack also has implications for seeing powerful AI applied more in settings that do not fit the perfect information assumption. The old paradigm for handling imperfect information has shown promise in applications like defending strategic resources and robust decision making as needed for medical treatment recommendations. The new paradigm will hopefully open up many more possibilities."

----
>	"In the past, perfect information games (chess, checkers, go) have been easier algorithmically than imperfect information games like poker. Powerful techniques like Alpha-Beta search, heuristic functions, depth-limited lookahead & Monte Carlo Tree Search work with perfect information games. They allow an AI to ignore the past and focus its computation on the tiny, immediate subgame most relevant for choosing actions. Until now, these techniques didn't really work in imperfect info games like poker, due to uncertainty about what each player knows. There is no single state to search from, but a set of states for each player. Past actions reveal info about what cards they hold. In imperfect info games like poker, local search hasn’t performed well. Had to solve the whole game at once, not as small fragments. But poker games are huge, and except for smallest versions, can't be solved exactly. Dominant technique was to solve a simplified game. This was called Abstraction-Solving-Translation. Simplify a game, solve it, use the simplified strategy in the real game. That simplification introduces mistakes. In some games, this still worked well enough to beat pros. AST didn't work well in No-Limit poker. Humans exploited simplified betting, and in huge pots, fine details of cards matter. This is the DeepStack breakthrough: it reintroduces powerful local search techniques to the imperfect info setting. No abstraction! It only considers a small local subgame to pick actions, given the "public state" and summary info of earlier actions in the hand. Early in the game, Deep Learning supplies a heuristic function to avoid searching to the end of the game. On the turn & river, it solves from the current decision until the end of the game and re-solves after every opponent action. On the preflop & flop, it solves to the end of the round then consults a deep neural net for value estimate of playing the turn/river. This NN is trained from randomly-generated hands (no human data needed) and must return value of every hand for each player. Deep Stack doesn't abstract cards or have to translate opponents bets. It always gets these details exactly right. This means there are no easy exploits, & we developed a new exploitability measurement program, Local Best Response, to show this. Also lets DeepStack play with any stacks/blinds. Can play freezeouts, cash games, etc. Earlier programs were specific to 1 stack size!"

>	"Compared to Libratus, on the turn/river both programs are pretty similar: both use the same "continual resolving" trick (and counterfactual regret minimization). Biggest difference is preflop/flop. DeepStack uses continual resolving there too, so it can't get tricked by bet size attacks. Libratus used the old precomputed-strategy method for preflop/flop. It had holes they had to patch overnight, as pros found them. DeepStack can play any stack sizes, so can do freezeouts, cash games, etc. Libratus can only do 200bb stacks. Last big difference is resources. Libratus runs on a huge supercomputer, Cepheus only needs a laptop with a good GPU."

>	"DeepStack does not compute and store a complete strategy prior to play. DeepStack computes a strategy based on the current state of the game for only the remainder of the hand, not maintaining one for the full game, which leads to lower overall exploitability."

>	"Despite using ideas from abstraction, DeepStack is fundamentally different from abstraction-based approaches, which compute and store a strategy prior to play. While DeepStack restricts the number of actions in its lookahead trees, it has no need for explicit abstraction as each re-solve starts from the actual public state, meaning DeepStack always perfectly understands the current situation."

>	"DeepStack is the first theoretically sound application of heuristic search methods—which have been famously successful in games like checkers, chess, and Go - to imperfect information games."

>	"At a conceptual level, DeepStack’s continual re-solving, “intuitive” local search and sparse lookahead trees describe heuristic search, which is responsible for many AI successes in perfect information games. Until DeepStack, no theoretically sound application of heuristic search was known in imperfect information games."

>	"During re-solving, DeepStack doesn’t need to reason about the entire remainder of the game because it substitutes computation beyond a certain depth with a fast approximate estimate, DeepStack’s "intuition" - a gut feeling of the value of holding any possible private cards in any possible poker situation. Finally, DeepStack’s intuition, much like human intuition, needs to be trained. We train it with deep learning using examples generated from random poker situations."

>	"Part of DeepStack's development is also a technique designed to find flaws in poker strategies. Local Best Response (LBR) is one of the cool new algorithms in Science paper. LBR looks directly at the strategy, like what a human could get from playing millions of hands to know its range. Older programs get beat by LBR for 4x more than folding every hand! DeepStack has no holes exposed by the LBR algorithm."

  - <http://science.sciencemag.org/content/early/2017/03/01/science.aam6960>
  - <https://youtube.com/playlist?list=PLX7NnbJAq7PlA2XpynViLOigzWtmr6QVZ> (demo matches)
  - <http://deepstack.ai>
  - <http://twitter.com/DeepStackAI>
  - <https://youtube.com/watch?v=qndXrHcV1sM> (Bowling)
  - <https://github.com/lifrordi/DeepStack-Leduc>


#### Mnih et al. - ["Human-Level Control Through Deep Reinforcement Learning"](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)
>	"The theory of reinforcement learning provides a normative account, deeply rooted in psychological and neuroscientific perspectives on animal behaviour, of how agents may optimize their control of an environment. To use reinforcement learning successfully in situations approaching real-world complexity, however, agents are confronted with a difficult task: they must derive efficient representations of the environment from high-dimensional sensory inputs, and use these to generalize past experience to new situations. Remarkably, humans and other animals seem to solve this problem through a harmonious combination of reinforcement learning and hierarchical sensory processing systems, the former evidenced by a wealth of neural data revealing notable parallels between the phasic signals emitted by dopaminergic neurons and temporal difference reinforcement learning algorithms. While reinforcement learning agents have achieved some successes in a variety of domains, their applicability has previously been limited to domains in which useful features can be handcrafted, or to domains with fully observed, low-dimensional state spaces. Here we use recent advances in training deep neural networks to develop a novel artificial agent, termed a deep Q-network, that can learn successful policies directly from high-dimensional sensory inputs using end-to-end reinforcement learning. We tested this agent on the challenging domain of classic Atari 2600 games. We demonstrate that the deep Q-network agent, receiving only the pixels and the game score as inputs, was able to surpass the performance of all previous algorithms and achieve a level comparable to that of a professional human games tester across a set of 49 games, using the same algorithm, network architecture and hyperparameters. This work bridges the divide between high-dimensional sensory inputs and actions, resulting in the first artificial agent that is capable of learning to excel at a diverse array of challenging tasks."

  - <http://nature.com/nature/journal/v518/n7540/full/nature14236.html>
  - <http://youtube.com/watch?v=EfGD2qveGdQ> (demo)
  - <http://youtu.be/XAbLn66iHcQ?t=1h41m21s> + <http://youtube.com/watch?v=0xo1Ldx3L5Q> (3D racing demo)
  - <http://youtube.com/watch?v=nMR5mjCFZCw> (3D labyrinth demo)
  - <http://youtube.com/watch?v=re6hkcTWVUY> (Doom gameplay demo)
  - <http://youtube.com/watch?v=6jlaBD9LCnM> + <https://youtube.com/watch?v=6JT6_dRcKA> (blockworld demo)
  - <http://youtube.com/user/eldubro/videos> (demos)
  - <http://youtube.com/watch?v=iqXKQf2BOSE> (demo)
  - <http://videolectures.net/nipsworkshops2013_mnih_atari/> (Mnih)
  - <http://youtube.com/watch?v=xzM7eI7caRk> (Mnih)
  - <http://youtube.com/watch?v=dV80NAlEins> (de Freitas)
  - <http://youtube.com/watch?v=HUmEbUkeQHg> (de Freitas)
  - <http://youtube.com/watch?v=mrgJ53TIcQc> (Pavlov, in russian)
  - <http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html>
  - <https://github.com/khanhptnk/deep-q-tensorflow>
  - <https://github.com/nivwusquorum/tensorflow-deepq>
  - <https://github.com/devsisters/DQN-tensorflow>
  - <https://github.com/carpedm20/deep-rl-tensorflow>
  - <https://github.com/VinF/deer>
  - <https://github.com/osh/kerlym>
  - <https://github.com/Jabberwockyll/deep_rl_ale>
  - <https://github.com/DanielTakeshi/rl_algorithms/blob/master/dqn/dqn.py>


#### Lample, Chaplot - ["Playing FPS Games with Deep Reinforcement Learning"](https://arxiv.org/abs/1609.05521)
>	"Advances in deep reinforcement learning have allowed autonomous agents to perform well on Atari games, often outperforming humans, using only raw pixels to make their decisions. However, most of these games take place in 2D environments that are fully observable to the agent. In this paper, we present the first architecture to tackle 3D environments in first-person shooter games, that involve partially observable states. Typically, deep reinforcement learning methods only utilize visual input for training. We present a method to augment these models to exploit game feature information such as the presence of enemies or items, during the training phase. Our model is trained to simultaneously learn these features along with minimizing a Q-learning objective, which is shown to dramatically improve the training speed and performance of our agent. Our architecture is also modularized to allow different models to be independently trained for different phases of the game. We show that the proposed architecture substantially outperforms built-in AI agents of the game as well as humans in deathmatch scenarios."

>	"We introduced a method to augment a DRQN model with high-level game information, and modularized our architecture to incorporate independent networks responsible for different phases of the game. These methods lead to dramatic improvements over the standard DRQN model when applied to complicated tasks like a deathmatch. We showed that the proposed model is able to outperform built-in bots as well as human players and demonstrated the generalizability of our model to unknown maps."

  - <https://youtube.com/playlist?list=PLduGZax9wmiHg-XPFSgqGg8PEAV51q1FT> (demo)


#### Lai - ["Giraffe: Using Deep Reinforcement Learning to Play Chess"](http://arxiv.org/abs/1509.01549)
>	"This report presents Giraffe, a chess engine that uses self-play to discover all its domain-specific knowledge, with minimal hand-crafted knowledge given by the programmer. Unlike previous attempts using machine learning only to perform parameter tuning on hand-crafted evaluation functions, Giraffe’s learning system also performs automatic feature extraction and pattern recognition. The trained evaluation function performs comparably to the evaluation functions of state-of-the-art chess engines - all of which containing thousands of lines of carefully hand-crafted pattern recognizers, tuned over many years by both computer chess experts and human chess masters. Giraffe is the most successful attempt thus far at using end-to-end machine learning to play chess. We also investigated the possibility of using probability thresholds instead of depth to shape search trees. Depth-based searches form the backbone of virtually all chess engines in existence today, and is an algorithm that has become well-established over the past half century. Preliminary comparisons between a basic implementation of probability-based search and a basic implementation of depth-based search showed that our new probability-based approach performs moderately better than the established approach. There are also evidences suggesting that many successful ad-hoc add-ons to depth-based searches are generalized by switching to a probability-based search. We believe the probability-based search to be a more fundamentally correct way to perform minimax. Finally, we designed another machine learning system to shape search trees within the probability-based search framework. Given any position, this system estimates the probability of each of the moves being the best move without looking ahead. The system is highly effective - the actual best move is within the top 3 ranked moves 70% of the time, out of an average of approximately 35 legal moves from each position. This also resulted in a significant increase in playing strength. With the move evaluator guiding a probability-based search using the learned evaluator, Giraffe plays at approximately the level of an FIDE International Master (top 2.2% of tournament chess players with an official rating)."

>	"In this project, we investigated the use of deep reinforcement learning with automatic feature extraction in the game of chess. The results show that the learned system performs at least comparably to the best expert-designed counterparts in existence today, many of which have been fine tuned over the course of decades. The beauty of this approach is in its generality. While it was not explored in this project due to time constraint, it is likely that this approach can easily be ported to other zero-sum turn-based board games, and achieve state-of-art performance quickly, especially in games where there has not been decades of intense research into creating a strong AI player. In addition to the machine learning aspects of the project, we introduced and tested an alternative variant of the decades-old minimax algorithm, where we apply probability boundaries instead of depth boundaries to limit the search tree. We showed that this approach is at least comparable and quite possibly superior to the approach that has been in use for the past half century. We also showed that this formulation of minimax works especially well with our probability-based machine learning approach. Efficiency is always a major consideration when switching from an expert system to a machine learning approach, since expert systems are usually more efficient to evaluate than generic models. This is especially important for applications like a chess engine, where being able to search nodes quickly is strongly correlated with playing strength. Some earlier attempts at applying neural network to chess have been thwarted by large performance penalties. Giraffe’s optimized implementation of neural network, when combined with the much higher vector arithmetics throughput of modern processors and effective caching, allows it to search at a speed that is less than 1 order of magnitude slower than the best modern chess engines, thus making it quite competitive against many chess engines in gameplay without need for time handicap. With all our enhancements, Giraffe is able to play at the level of an FIDE International Master on a modern mainstream PC. While that is still a long way away from the top engines today that play at super-Grandmaster levels, it is able to defeat many lower-tier engines, most of which search an order of magnitude faster. One of the original goals of the project is to create a chess engine that is less reliant on brute-force than its contemporaries, and that goal has certainly been achieved. Unlike most chess engines in existence today, Giraffe derives its playing strength not from being able to see very far ahead, but from being able to evaluate tricky positions accurately, and understanding complicated positional concepts that are intuitive to humans, but have been elusive to chess engines for a long time. This is especially important in the opening and end game phases, where it plays exceptionally well."

>	"It is clear that Giraffe’s evaluation function has at least comparable positional understanding compared to evaluation functions of top engines in the world, which is remarkable because their evaluation functions are all carefully hand-designed behemoths with hundreds of parameters that have been tuned both manually and automatically over several years, and many of them have been worked on by human grandmasters. The test suite likely under-estimates the positional understanding of Giraffe compared to other engines, because most of the themes tested by the test suite are generally well-understood concepts in computer chess that are implemented by many engines, and since the test suite is famous, it is likely that at least some of the engines have been tuned specifically against the test suite. Since Giraffe discovered all the evaluation features through self-play, it is likely that it knows about patterns that have not yet been studied by humans, and hence not included in the test suite. As far as we are aware, this is the first successful attempt at using machine learning to create a chess evaluation function from self-play, including automatic feature extraction (many previous attempts are weight-tuning for hand-designed features), starting from minimal hand-coded knowledge, and achieving comparable performance to state-of-the-art expert-designed evaluation functions."

  - <https://bitbucket.org/waterreaction/giraffe>


#### Veness, Silver, Uther, Blair - ["Bootstrapping from Game Tree Search"](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Applications_files/bootstrapping.pdf) (Chess)
>	"In this paper we introduce a new algorithm for updating the parameters of a heuristic evaluation function, by updating the heuristic towards the values computed by an alpha-beta search. Our algorithm differs from previous approaches to learning from search, such as Samuel’s checkers player and the TD-Leaf algorithm, in two key ways. First, we update all nodes in the search tree, rather than a single node. Second, we use the outcome of a deep search, instead of the outcome of a subsequent search, as the training signal for the evaluation function. We implemented our algorithm in a chess program Meep, using a linear heuristic function. After initialising its weight vector to small random values, Meep was able to learn high quality weights from self-play alone. When tested online against human opponents, Meep played at a master level, the best performance of any chess program with a heuristic learned entirely from self-play."

  - <http://videolectures.net/nips09_veness_bfg/> (Veness)


#### Yeh, Lin - ["Automatic Bridge Bidding Using Deep Reinforcement Learning"](http://arxiv.org/abs/1607.03290)
>	"Bridge is among the zero-sum games for which artificial intelligence has not yet outperformed expert human players. The main difficulty lies in the bidding phase of bridge, which requires cooperative decision making under partial information. Existing artificial intelligence systems for bridge bidding rely on and are thus restricted by human-designed bidding systems or features. In this work, we propose a pioneering bridge bidding system without the aid of human domain knowledge. The system is based on a novel deep reinforcement learning model, which extracts sophisticated features and learns to bid automatically based on raw card data. The model includes an upper-confidence-bound algorithm and additional techniques to achieve a balance between exploration and exploitation. Our experiments validate the promising performance of our proposed model. In particular, the model advances from having no knowledge about bidding to achieving superior performance when compared with a champion-winning computer bridge program that implements a human-designed bidding system. To the best of our knowledge, our proposed model is the first to tackle automatic bridge bidding from raw data without additional human knowledge."


#### Jaskowski - ["Mastering 2048 with Delayed Temporal Coherence Learning Multi-State Weight Promotion, Redundant Encoding and Carousel Shaping"](https://arxiv.org/abs/1604.05085)
>	"2048 is an engaging single-player, nondeterministic video puzzle game, which, thanks to the simple rules and hard-to-master gameplay, has gained massive popularity in recent years. As 2048 can be conveniently embedded into the discrete-state Markov decision processes framework, we treat it as a testbed for evaluating existing and new methods in reinforcement learning. With the aim to develop a strong 2048 playing program, we employ temporal difference learning with systematic n-tuple networks. We show that this basic method can be significantly improved with temporal coherence learning, multi-stage function approximator with weight promotion, carousel shaping and redundant encoding. In addition, we demonstrate how to take advantage of the characteristics of the n-tuple network, to improve the algorithmic effectiveness of the learning process by i) delaying the (decayed) update and applying lock-free optimistic parallelism to effortlessly make advantage of multiple CPU cores. This way, we were able to develop the best known 2048 playing program to date, which confirms the effectiveness of the introduced methods for discrete-state Markov decision problems."

  - <https://github.com/wjaskowski/mastering-2048>
  - <https://github.com/aszczepanski/2048>


#### Peng, Berseth, van de Panne - ["Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning"](http://www.cs.ubc.ca/~van/papers/2016-TOG-deepRL/2016-TOG-deepRL.pdf)
>	"Reinforcement learning offers a promising methodology for developing skills for simulated characters, but typically requires working with sparse hand-crafted features. Building on recent progress in deep reinforcement learning, we introduce a mixture of actor-critic experts approach that learns terrain-adaptive dynamic locomotion skills using high-dimensional state and terrain descriptions as input, and parameterized leaps or steps as output actions. MACE learns more quickly than a single actor-critic approach and results in actor-critic experts that exhibit specialization. Additional elements of our solution that contribute towards efficient learning include Boltzmann exploration and the use of initial actor biases to encourage specialization. Results are demonstrated for multiple planar characters and terrain classes."

>	"We introduce a novel mixture of actor-critic experts architecture to enable accelerated learning. MACE develops n individual control policies and their associated value functions, which each then specialize in particular regimes of the overall motion. During final policy execution, the policy associated with the highest value function is executed, in a fashion analogous to Q-learning with discrete actions. We show the benefits of Boltzmann exploration and various algorithmic features for our problem domain."

  - <https://youtube.com/watch?v=KPfzRSBzNX4> + <https://youtube.com/watch?v=A0BmHoujP9k> (demo)
  - <https://youtube.com/watch?v=mazfn4dHPRM> + <https://youtube.com/watch?v=RTuSHI5FNzg> (overview)
  - <https://github.com/xbpeng/DeepTerrainRL>


#### Langford et al. - ["A Multiworld Testing Decision Service"](http://arxiv.org/abs/1606.03966)
>	"Applications and systems are constantly faced with decisions to make, often using a policy to pick from a set of actions based on some contextual information. We create a service that uses machine learning to accomplish this goal. The service uses exploration, logging, and online learning to create a counterfactually sound system supporting a full data lifecycle. The system is general: it works for any discrete choices, with respect to any reward metric, and can work with many learning algorithms and feature representations. The service has a simple API, and was designed to be modular and reproducible to ease deployment and debugging, respectively. We demonstrate how these properties enable learning systems that are robust and safe. Our evaluation shows that the Decision Service makes decisions in real time and incorporates new data quickly into learned policies. A large-scale deployment for a personalized news website has been handling all traffic since Jan. 2016, resulting in a 25% relative lift in clicks. By making the Decision Service externally available, we hope to make optimal decision making available to all."

>	"We have presented the Decision Service: a powerful tool to support the complete data lifecycle, which automates many of the burdensome tasks that data scientists face such as gathering the right data and deploying in an appropriate manner. Instead, a data scientist can focus on more core tasks such as finding the right features, representation, or signal to optimize against. The data lifecycle support also makes basic application of the Decision Service feasible without a data scientist. To assist in lowering the barrier to entry, we are exploring techniques based on expert learning and hyperparameter search that may further automate the process. Since the policy evaluation techniques can provide accurate predictions of online performance, such automations are guaranteed to be statistically sound. We are also focusing on making the decision service easy to deploy and use because we believe this is key to goal of democratizing machine learning for everyone. The Decision Service can also naturally be extended to a greater variety of problems, all of which can benefit from data lifecycle support. Plausible extensions might address advanced variants like reinforcement and active learning, and simpler ones like supervised learning."

----
>	"It is the first general purpose reinforcement-based learning system. Wouldn’t it be great if Reinforcement Learning algorithms could easily be used to solve all reinforcement learning problems? But there is a well-known problem: It’s very easy to create natural RL problems for which all standard RL algorithms (epsilon-greedy Q-learning, SARSA, etc) fail catastrophically. That’s a serious limitation which both inspires research and which I suspect many people need to learn the hard way. Removing the credit assignment problem from reinforcement learning yields the Contextual Bandit setting which we know is generically solvable in the same manner as common supervised learning problems."

>	"Many people have tried to create online learning system that do not take into account the biasing effects of decisions. These fail near-universally. For example they might be very good at predicting what was shown (and hence clicked on) rather that what should be shown to generate the most interest."

>	"We need a system that explores over appropriate choices with logging of features, actions, probabilities of actions, and outcomes. These must then be fed into an appropriate learning algorithm which trains a policy and then deploys the policy at the point of decision. The system enables a fully automatic causally sound learning loop for contextual control of a small number of actions. It is strongly scalable, for example a version of this is in use for personalized news on MSN."

  - <http://hunch.net/?p=4464948> (Langford)
  - <http://research.microsoft.com/en-us/projects/mwt/>
  - <https://mwtds.azurewebsites.net>
  - <https://youtube.com/watch?v=7ic_d5TeIUk> (Langford)
  - <https://youtu.be/N5x48g2sp8M?t=52m> (Schapire)
  - <http://machinedlearnings.com/2017/01/reinforcement-learning-as-service.html> (Mineiro)


#### Norouzi, Bengio, Chen, Jaitly, Schuster, Wu, Schuurmans - ["Reward Augmented Maximum Likelihood for Neural Structured Prediction"](https://arxiv.org/abs/1609.00150)
>	"A key problem in structured output prediction is direct optimization of the task reward function that matters for test evaluation. This paper presents a simple and computationally efficient approach to incorporate task reward into a maximum likelihood framework. We establish a connection between the log-likelihood and regularized expected reward objectives, showing that at a zero temperature, they are approximately equivalent in the vicinity of the optimal solution. We show that optimal regularized expected reward is achieved when the conditional distribution of the outputs given the inputs is proportional to their exponentiated (temperature adjusted) rewards. Based on this observation, we optimize conditional log-probability of edited outputs that are sampled proportionally to their scaled exponentiated reward. We apply this framework to optimize edit distance in the output label space. Experiments on speech recognition and machine translation for neural sequence to sequence models show notable improvements over a maximum likelihood baseline by using edit distance augmented maximum likelihood."

>	"Neural sequence models use a maximum likelihood framework to maximize the conditional probability of the ground-truth outputs given corresponding inputs. These models do not explicitly consider the task reward during training, hoping that conditional log-likelihood would serve as a good surrogate for the task reward. Such methods make no distinction between alternative incorrect outputs: log-probability is only measured on the ground-truth input-output pairs, and all alternative outputs are equally penalized, whether near or far from the ground-truth target. We believe that one can improve upon maximum likelihood sequence models, if the difference in the rewards of alternative outputs is taken into account. A key property of ML training for locally normalized RNN models is that the objective function factorizes into individual loss terms, which could be efficiently optimized using stochastic gradient descend. In particular, ML training does not require any form of inference or sampling from the model during training, which leads to computationally efficient and easy to implementations."

>	"Alternatively, one can use reinforcement learning algorithms, such as policy gradient, to optimize expected task reward during training. Even though expected task reward seems like a natural objective, direct policy optimization faces significant challenges: unlike ML, the gradient for a mini-batch of training examples is extremely noisy and has a high variance; gradients need to be estimated via sampling from the model, which is a non-stationary distribution; the reward is often sparse in a high-dimensional output space, which makes it difficult to find any high value predictions, preventing learning from getting off the ground; and, finally, maximizing reward does not explicitly consider the supervised labels, which seems inefficient. In fact, all previous attempts at direct policy optimization for structured output prediction has started by bootstrapping from a previously trained ML solution and they use several heuristics and tricks to make learning stable."

>	"This paper presents a new approach to task reward optimization that combines the computational efficiency and simplicity of ML with the conceptual advantages of expected reward maximization. Our algorithm called reward augmented maximum likelihood simply adds a sampling step on top of the typical likelihood objective. Instead of optimizing conditional log-likelihood on training input-output pairs, given each training input, we first sample an output proportional to its exponentiated scaled reward. Then, we optimize log-likelihood on such auxiliary output samples given corresponding inputs. When the reward for an output is defined as its similarity to a ground-truth output, then the output sampling distribution is peaked at the ground-truth output, and its concentration is controlled by a temperature hyper-parameter."

>	"Surprisingly, we find that the best performance is achieved with output sampling distributions that put a lot of the weight away from the ground-truth outputs. In fact, in our experiments, the training algorithm rarely sees the original unperturbed outputs. Our results give further evidence that models trained with imperfect outputs and their reward values can improve upon models that are only exposed to a single ground-truth output per input."

>	"There are several critical differences between gradient estimators for RML loss (reward augmented maximum likelihood) and RL loss (regularized expected reward) that make SGD optimization of RML loss more desirable. First, for RML loss, one has to sample from a stationary distribution, the so called exponentiated payoff distribution, whereas for RL loss one has to sample from the model distribution as it is evolving. Not only sampling from the model could slow down training, but also one needs to employ several tricks to get a better estimate of the gradient of RL loss. Further, the reward is often sparse in a high-dimensional output space, which makes finding any reasonable predictions challenging, when RL loss is used to refine a randomly initialized model. Thus, smart model initialization is needed. By contrast, we initialize the models randomly and refine them using RML loss."

----
>	"This reads as another way to use a world model to mitigate the sample complexity of reinforcement learning (e.g., what if edit distance was just the initial model of the reward?)."

----
>	"Andrej Karpathy provided another perspective: We can also view the process of optimizing LRML as distilling the exponentiated payoff distribution q(y|y*;τ) into the model pθ(y|x). The objective reaches a maximum when these two distributions are equivalent. From this distillation view, the question is clear: how can we distill more complex objects into pθ? Concretely, this means we should develop more complex reward distributions q to use in this setup. We have seen two examples so far: the exponentiated payoff from the paper and the label smoothing example of the previous section. We could define q to be a complex pre-trained model or a mixture of experts, and use this training process to distill them into a single model pθ. We just need to make sure that we can efficiently sample from the q we select."

----
>	"Alec Radford mentioned that the data augmentation suggested in the paper sounds similar in spirit to virtual adversarial training, where the current model is encouraged to make robust predictions not only for the examples in the training set but also for inputs “nearby” those that exist in the training set. A high-level comparison:  
>	- Adversarial training can be seen as data-augmentation in the input space X. The RML objective does data-augmentation in the output space Y.  
>	- Adversarial training performs model-based data augmentation: the examples generated are those for which the current model is maximally vulnerable. RML training performs data-based augmentation: the examples generated have outputs that are “near” the ground-truth outputs. (Here 'near' is defined by the reward function.)"  

  - <https://youtu.be/fZNyHoXgV7M?t=24m59s> (Norouzi)
  - <https://youtu.be/uohtFXD_39c?t=38m10s> (Samy Bengio)
  - <http://youtube.com/watch?v=agA-rc71Uec> (Samy Bengio)
  - <http://drive.google.com/file/d/0B3Rdm_P3VbRDVUQ4SVBRYW82dU0> (Gauthier)
  - <http://www.shortscience.org/paper?bibtexKey=journals/corr/1609.00150>
  - <http://www.shortscience.org/paper?bibtexKey=conf%2Fnips%2FNorouziBCJSWS16>


#### Xia, He, Qin, Wang, Yu, Liu, Ma - ["Dual Learning for Machine Translation"](https://arxiv.org/abs/1611.00179)
>	"While neural machine translation (NMT) is making good progress in the past two years, tens of millions of bilingual sentence pairs are needed for its training. However, human labeling is very costly. To tackle this training data bottleneck, we develop a dual-learning mechanism, which can enable an NMT system to automatically learn from unlabeled data through a dual-learning game. This mechanism is inspired by the following observation: any machine translation task has a dual task, e.g., English-to-French translation (primal) versus French-to-English translation (dual); the primal and dual tasks can form a closed loop, and generate informative feedback signals to train the translation models, even if without the involvement of a human labeler. In the dual-learning mechanism, we use one agent to represent the model for the primal task and the other agent to represent the model for the dual task, then ask them to teach each other through a reinforcement learning process. Based on the feedback signals generated during this process (e.g., the languagemodel likelihood of the output of a model, and the reconstruction error of the original sentence after the primal and dual translations), we can iteratively update the two models until convergence (e.g., using the policy gradient methods). We call the corresponding approach to neural machine translation dual-NMT. Experiments show that dual-NMT works very well on English↔French translation; especially, by learning from monolingual data (with 10% bilingual data for warm start), it achieves a comparable accuracy to NMT trained from the full bilingual data for the French-to-English translation task."

>	"First, although we have focused on machine translation in this work, the basic idea of dual learning is generally applicable: as long as two tasks are in dual form, we can apply the dual-learning mechanism to simultaneously learn both tasks from unlabeled data using reinforcement learning algorithms. Actually, many AI tasks are naturally in dual form, for example, speech recognition versus text to speech, image caption versus image generation, question answering versus question generation (e.g., Jeopardy!), search (matching queries to documents) versus keyword extraction (extracting keywords/queries for documents), so on and so forth. It would very be interesting to design and test dual-learning algorithms for more dual tasks beyond machine translation."

>	"Second, although we have focused on dual learning on two tasks, our technology is not restricted to two tasks only. Actually, our key idea is to form a closed loop so that we can extract feedback signals by comparing the original input data with the final output data. Therefore, if more than two associated tasks can form a closed loop, we can apply our technology to improve the model in each task from unlabeled data. For example, for an English sentence x, we can first translate it to a Chinese sentence y, then translate y to a French sentence z, and finally translate z back to an English sentence x 0 . The similarity between x and x 0 can indicate the effectiveness of the three translation models in the loop, and we can once again apply the policy gradient methods to update and improve these models based on the feedback signals during the loop. We would like to name this generalized dual learning as close-loop learning, and will test its effectiveness in the future."

>	"We plan to explore the following directions in the future. First, in the experiments we used bilingual data to warm start the training of dual-NMT. A more exciting direction is to learn from scratch, i.e., to learn translations directly from monolingual data of two languages (maybe plus lexical dictionary). Second, our dual-NMT was based on NMT systems in this work. Our basic idea can also be applied to phrase-based SMT systems and we will look into this direction. Third, we only considered a pair of languages in this paper. We will extend our approach to jointly train multiple translation models for a tuple of 3+ languages using monolingual data."

----
>	"The authors finetune an FR -> EN NMT model using a RL-based dual game. 1. Pick a French sentence from a monolingual corpus and translate it to EN. 2. Use an EN language model to get a reward for the translation 3. Translate the translation back into FR using an EN -> FR system. 4. Get a reward based on the consistency between original and reconstructed sentence. Training this architecture using Policy Gradient authors can make efficient use of monolingual data and show that a system trained on only 10% of parallel data and finetuned with monolingual data achieves comparable BLUE scores as a system trained on the full set of parallel data."

  - <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/dual-learning-mt.md>



---
### interesting papers - exploration and intrinsic motivation

[interesting recent papers](#interesting-papers---exploration-and-intrinsic-motivation)


#### Oudeyer, Kaplan - ["How Can We Define Intrinsic Motivation"](http://pyoudeyer.com/epirob08OudeyerKaplan.pdf)
>	"Intrinsic motivation is a crucial mechanism for open-ended cognitive development since it is the driver of spontaneous exploration and curiosity. Yet, it has so far only been conceptualized in ad hoc manners in the epigenetic robotics community. After reviewing different approaches to intrinsic motivation in psychology, this paper presents a unified definition of intrinsic motivation, based on the theory of Daniel Berlyne. Based on this definition, we propose a landscape of types of computational approaches, making it possible to position existing and future models relative to each other, and we show that important approaches are still to be explored."

  - [models of intrinsic motivation](#exploration-and-intrinsic-motivation) described by Oudeyer and Kaplan


#### Schmidhuber - ["Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes"](http://arxiv.org/abs/0812.4360)
>	"I argue that data becomes temporarily interesting by itself to some self-improving, but computationally limited, subjective observer once he learns to predict or compress the data in a better way, thus making it subjectively simpler and more beautiful. Curiosity is the desire to create or discover more non-random, non-arbitrary, regular data that is novel and surprising not in the traditional sense of Boltzmann and Shannon but in the sense that it allows for compression progress because its regularity was not yet known. This drive maximizes interestingness, the first derivative of subjective beauty or compressibility, that is, the steepness of the learning curve. It motivates exploring infants, pure mathematicians, composers, artists, dancers, comedians, yourself, and artificial systems."

  - [Artificial Curiosity and Creativity](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#artificial-curiosity-and-creativity)


#### Schmidhuber - ["Formal Theory of Creativity, Fun, and Intrinsic Motivation"](http://people.idsia.ch/~juergen/ieeecreative.pdf)
>	"The simple but general formal theory of fun & intrinsic motivation & creativity is based on the concept of maximizing intrinsic reward for the active creation or discovery of novel, surprising patterns allowing for improved prediction or data compression. It generalizes the traditional field of active learning, and is related to old but less formal ideas in aesthetics theory and developmental psychology. It has been argued that the theory explains many essential aspects of intelligence including autonomous development, science, art, music, humor. This overview first describes theoretically optimal (but not necessarily practical) ways of implementing the basic computational principles on exploratory, intrinsically motivated agents or robots, encouraging them to provoke event sequences exhibiting previously unknown but learnable algorithmic regularities. Emphasis is put on the importance of limited computational resources for online prediction and compression. Discrete and continuous time formulations are given. Previous practical but non-optimal implementations (1991, 1995, 1997-2002) are reviewed, as well as several recent variants by others (2005-). A simplified typology addresses current confusion concerning the precise nature of intrinsic motivation."

>	"I have argued that a simple but general formal theory of creativity based on reward for creating or finding novel patterns allowing for data compression progress explains many essential aspects of intelligence including science, art, music, humor. Here I discuss what kind of general bias towards algorithmic regularities we insert into our robots by implementing the principle, why that bias is good, and how the approach greatly generalizes the field of active learning. I provide discrete and continuous time formulations for ongoing work on building an Artificial General Intelligence based on variants of the artificial creativity framework."

>	"In the real world external rewards are rare. But unsupervised AGIs using additional intrinsic rewards as described in this paper will be motivated to learn many useful behaviors even in absence of external rewards, behaviors that lead to predictable or compressible results and thus reflect regularities in the environment, such as repeatable patterns in the world’s reactions to certain action sequences. Often a bias towards exploring previously unknown environmental regularities through artificial curiosity / creativity is a priori desirable because goal-directed learning may greatly profit from it, as behaviors leading to external reward may often be rather easy to compose from previously learnt curiosity-driven behaviors. It may be possible to formally quantify this bias towards novel patterns in form of a mixture-based prior, a weighted sum of probability distributions on sequences of actions and resulting inputs, and derive precise conditions for improved expected external reward intake. Intrinsic reward may be viewed as analogous to a regularizer in supervised learning, where the prior distribution on possible hypotheses greatly influences the most probable interpretation of the data in a Bayesian framework (for example, the well-known weight decay term of neural networks is a consequence of a Gaussian prior with zero mean for each weight). Following the introductory discussion, some of the AGIs based on the creativity principle will become scientists, artists, or comedians."

  - [Artificial Curiosity and Creativity](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#artificial-curiosity-and-creativity)  
  - <http://idsia.ch/~juergen/creativity.html>
  - <https://archive.org/details/Redwood_Center_2014_08_15_Jurgen_Schmidhuber> (Schmidhuber)
  - <https://vimeo.com/28759091> (Schmidhuber)
  - <http://videolectures.net/ecmlpkdd2010_schmidhuber_ftf/> (Schmidhuber)
  - <https://vimeo.com/7441291> (Schmidhuber)


#### Still, Precup - ["An Information-Theoretic Approach to Curiosity-Driven Reinforcement Learning"](http://www2.hawaii.edu/~sstill/StillPrecup2011.pdf)
>	"We provide a fresh look at the problem of exploration in reinforcement learning, drawing on ideas from information theory. First, we show that Boltzmann-style exploration, one of the main exploration methods used in reinforcement learning, is optimal from an information-theoretic point of view. Second, we address the problem of curiosity-driven learning. We propose that, in addition to maximizing the expected return, a learner should chose a policy that maximizes the predictive power of its own behavior, measured by the information that the most recent state-action pair carries about the future. This makes the world “interesting” and exploitable. The general result has the form of Boltzmann-style exploration with a bonus that contains a novel exploration-exploitation trade-off that emerges from the proposed optimization principle. Importantly, this exploration-exploitation trade-off is also present when the “temperature”-like parameter in the Boltzmann distribution tends to zero, i.e. when there is no exploration due to randomness. As a result, exploration emerges as a directed behavior that optimizes information gain, rather than being modeled solely as behavior randomization."

>	"We showed that a soft policy similar to Boltzmann exploration optimally trades return and the coding cost (or complexity) of the policy. By postulating that an agent should, in addition to maximizing the expected return, also maximize its predictive power, at a fixed policy complexity, we derived a trade-off between exploration and exploitation that does not rely on randomness in the action policy, and thereby may be more adequate to model exploration than previous schemes."


#### Houthooft, Chen, Duan, Schulman, Turck, Abbeel - ["VIME: Variational Information Maximizing Exploration"](http://arxiv.org/abs/1605.09674)
>	"Scalable and effective exploration remains a key challenge in reinforcement learning. While there are methods with optimality guarantees in the setting of discrete state and action spaces, these methods cannot be applied in high-dimensional greedy exploration or adding Gaussian noise to the controls. This paper introduces Variational Information Maximizing Exploration (VIME), an exploration strategy based on maximization of information gain about the agent’s belief of environment dynamics. We propose a practical implementation, using variational inference in Bayesian neural networks which efficiently handles continuous state and action spaces. VIME modifies the MDP reward function, and can be applied with several different underlying RL algorithms. We demonstrate that VIME achieves significantly better performance compared to heuristic exploration methods across a variety of continuous control tasks and algorithms, including tasks with very sparse rewards."

>	"We have proposed Variational Information Maximizing Exploration, a curiosity-driven exploration strategy for continuous control tasks. Variational inference is used to approximate the posterior distribution of a Bayesian neural network that represents the environment dynamics. Using information gain in this learned dynamics model as intrinsic rewards allows the agent to optimize for both external reward and intrinsic surprise simultaneously. Empirical results show that VIME performs significantly better than heuristic exploration methods across various continuous control tasks and algorithms. As future work, we would like to investigate measuring surprise in the value function and using the learned dynamics model for planning."

>	"This paper proposes a curiosity-driven exploration strategy, making use of information gain about the agent’s internal belief of the dynamics model as a driving force. This principle can be traced back to the concepts of curiosity and surprise (Schmidhuber). Within this framework, agents are encouraged to take actions that result in states they deem surprising - i.e., states that cause large updates to the dynamics model distribution. We propose a practical implementation of measuring information gain using variational inference. Herein, the agent’s current understanding of the environment dynamics is represented by a Bayesian neural networks. We also show how this can be interpreted as measuring compression improvement, a proposed model of curiosity (Schmidhuber). In contrast to previous curiosity-based approaches, our model scales naturally to continuous state and action spaces. The presented approach is evaluated on a range of continuous control tasks, and multiple underlying RL algorithms. Experimental results show that VIME achieves significantly better performance than naïve exploration strategies."

>	"Variational inference is used to approximate the posterior distribution of a Bayesian neural network that represents the environment dynamics. Using information gain in this learned dynamics model as intrinsic rewards allows the agent to optimize for both external reward and intrinsic surprise simultaneously."  
>	"r'(st,at,st+1) = r(st,at) + μ * Dkl[p(θ|ξt,at,st+1)||p(θ|ξt)]"  

>	"It is possible to derive an interesting relationship between compression improvement - an intrinsic reward objective defined in Schmidhuber's Artificial Curiosity and Creativity theory, and the information gain. The agent’s curiosity is equated with compression improvement, measured through C(ξt; φt-1) - C(ξt; φt), where C(ξ; φ) is the description length of ξ using φ as a model. Furthermore, it is known that the negative variational lower bound can be viewed as the description length. Hence, we can write compression improvement as L[q(θ; φt), ξt] - L[q(θ; φt-1), ξt]. In addition, due to alternative formulation of the variational lower bound, compression improvement can be written as (log p(ξt) - Dkl[q(θ; φt)||p(θ|ξt)]) - (log p(ξt) - Dkl[q(θ; φt-1)||p(θ|ξt)]). If we assume that φt perfectly optimizes the variational lower bound for the history ξt, then Dkl[q(θ; φt)||p(θ|ξt)] = 0, which occurs when the approximation equals the true posterior, i.e., q(θ; φt) = p(θ|ξt). Hence, compression improvement becomes Dkl[p(θ|ξt-1) || p(θ|ξt)]. Therefore, optimizing for compression improvement comes down to optimizing the KL divergence from the posterior given the past history ξt-1 to the posterior given the total history ξt. As such, we arrive at an alternative way to encode curiosity than information gain, namely Dkl[p(θ|ξt)||p(θ|ξt,at,st+1)], its reversed KL divergence. In experiments, we noticed no significant difference between the two KL divergence variants. This can be explained as both variants are locally equal when introducing small changes to the parameter distributions. Investigation of how to combine both information gain and compression improvement is deferred to future work."

  - <https://goo.gl/fyxLvI> (demo)
  - <https://youtu.be/WRFqzYWHsZA?t=18m38s> (Abbeel)
  - <https://youtube.com/watch?v=sRIjxxjVrnY> (Panin)
  - <http://pemami4911.github.io/paper-summaries/2016/09/04/VIME.html>
  - <https://github.com/openai/vime>
  - [Artificial Curiosity and Creativity](https://github.com/brylevkirill/notes/blob/master/Artificial%20Intelligence.md#artificial-curiosity-and-creativity) by Juergen Schmidhuber


#### Bellemare, Srinivasan, Ostrovski, Schaul, Saxton, Munos - ["Unifying Count-Based Exploration and Intrinsic Motivation"](http://arxiv.org/abs/1606.01868)
>	"We consider an agent's uncertainty about its environment and the problem of generalizing this uncertainty across observations. Specifically, we focus on the problem of exploration in non-tabular reinforcement learning. Drawing inspiration from the intrinsic motivation literature, we use density models to measure uncertainty, and propose a novel algorithm for deriving a pseudo-count from an arbitrary density model. This technique enables us to generalize count-based exploration algorithms to the non-tabular case. We apply our ideas to Atari 2600 games, providing sensible pseudo-counts from raw pixels. We transform these pseudo-counts into intrinsic rewards and obtain significantly improved exploration in a number of hard games, including the infamously difficult Montezuma's Revenge."

>	"Many of hard RL problems share one thing in common: rewards are few and far between. In reinforcement learning, exploration is the process by which an agent comes to understand its environment and discover where the reward is. Most practical RL applications still rely on crude algorithms, like epsilon-greedy (once in awhile, choose a random action), because more theoretically-motivated approaches don't scale. But epsilon-greedy is quite data inefficient, and often can't even get off the ground. In this paper we show that it's possible to use simple density models (assigning probabilities to states) to "count" the number of times we've visited a particular state. We call the output of our algorithm a pseudo-count. Pseudo-counts give us a handle on uncertainty: how confident are we that we've explored this part of the game?"

>	"Intrinsic motivation offers a different perspective on exploration. Intrinsic motivation algorithms typically use novelty signals - surrogates for extrinsic rewards - to drive curiosity within an agent, influenced by classic ideas from psychology. To sketch out some recurring themes, these novelty signals might be prediction error (Singh et al., 2004; Stadie et al., 2015), value error (Simsek and Barto, 2006), learning progress (Schmidhuber, 1991), or mutual information (Still and Precup, 2012; Mohamed and Rezende, 2015). The idea also finds roots in continual learning (Ring, 1997). In Thrun’s taxonomy, intrinsic motivation methods fall within the category of error-based exploration."

>	"We provide what we believe is the first formal evidence that intrinsic motivation and count-based exploration are but two sides of the same coin. Our main result is to derive a pseudo-count from a sequential density model over the state space. We only make the weak requirement that such a model should be learning-positive: observing x should not immediately decrease its density. In particular, counts in the usual sense correspond to the pseudo-counts implied by the data’s empirical distribution. We expose a tight relationship between the pseudo-count, a variant of Schmidhuber’s compression progress which we call prediction gain, and Bayesian information gain."

----
>	"Authors derived pseudo-counts from Context Tree Switching density models over states and used those to form intrinsic rewards."

>	"VIME computes the amount of information gained about the dynamics model due to the agent taking an action and seeing a certain following state. The authors show that the results should be similar, as maximizing the information gain also maximizes a lower bound on the inverse of the pseudo count."

>	"Although I'm a bit bothered by the assumption of the density model being "learning-positive", which seems central to their theoretical derivation of pseudo-counts: after you observe a state, your subjective probability of observing it again immediately should generally decrease unless you believe that the state is a fixed point attractor with high probability. I can see that in practice the assumption works well in their experimental setting since they use pixel-level factored models and, by the nature of the ATARI games they test on, most pixels don't change value from frame to frame, but in a more general setting, e.g. a side-scroller game or a 3D first-person game this assumption would not hold."

  - <https://youtube.com/watch?v=0yI2wJ6F8r0> (demo)
  - <https://youtube.com/watch?v=qSfd27AgcEk> (Bellemare)
  - <https://youtu.be/qduxl-vKz1E?t=1h16m30s> (Seleznev, in russian)
  - <https://youtube.com/watch?v=qKyOLNVpknQ> (Pavlov, in russian)
  - <http://pemami4911.github.io/paper-summaries/2016/10/08/unifying-count-based-exploration-and-intrinsic-motivation.html>
  - <https://github.com/lake4790k/pseudo-count-atari>


#### Salge, Glackin, Polani - ["Empowerment - An Introduction"](https://arxiv.org/abs/1310.1863)
>	"Is it better for you to own a corkscrew or not? If asked, you as a human being would likely say “yes”, but more importantly, you are somehow able to make this decision. You are able to decide this, even if your current acute problems or task do not include opening a wine bottle. Similarly, it is also unlikely that you evaluated several possible trajectories your life could take and looked at them with and without a corkscrew, and then measured your survival or reproductive fitness in each. When you, as a human cognitive agent, made this decision, you were likely relying on a behavioural “proxy”, an internal motivation that abstracts the problem of evaluating a decision impact on your overall life, but evaluating it in regard to some simple fitness function. One example would be the idea of curiosity, urging you to act so that your experience new sensations and learn about the environment. On average, this should lead to better and richer models of the world, which give you a better chance of reaching your ultimate goals of survival and reproduction."

>	"But how about questions such as, would you rather be rich than poor, sick or healthy, imprisoned or free? While each options offers some interesting new experience, there seems to be a consensus that rich, healthy and free is a preferable choice. We think that all these examples, in addition to the question of tool ownership above, share a common element of preparedness. Everything else being equal it is preferable to be prepared, to keep ones options open or to be in a state where ones actions have the greatest influence on ones direct environment."

>	"The concept of Empowerment, in a nutshell, is an attempt at formalizing and quantifying these degrees of freedom (or options) that an organism or agent has as a proxy for “preparedness”; preparedness, in turn, is considered a proxy for prospective fitness via the hypothesis that preparedness would be a good indicator to distinguish promising from less promising regions in the prospective fitness landscape, without actually having to evaluate the full fitness landscape."

>	"Empowerment aims to reformulate the options or degrees of freedom that an agent has as the agent’s control over its environment; and not only of its control - to be reproducible, the agent needs to be aware of its control influence and sense it. Thus, empowerment is a measure of both the control an agent has over its environment, as well as its ability to sense this control. Note that this already hints at two different perspectives to evaluate the empowerment of an agent. From the agent perspective empowerment can be a tool for decision making, serving as a behavioural proxy for the agent. This empowerment value can be skewed by the quality of the agent world model, so it should be more accurately described as the agent’s approximation of its own empowerment, based on its world model. The actual empowerment depends both on the agent’s embodiment, and the world the agent is situated in. More precisely, there is a specific empowerment value for the current state of the world (the agent’s current empowerment), and there is an averaged value over all possible states of the environment, weighted by their probability (the agent’s average empowerment)."

>	"Empowerment, as introduced by Klyubin et al. (2005), aims to formalize the combined notion of an agent controlling its environment and sensing this control in the language of information theory. The idea behind this is that this should provide us with a utility function that is inherently local, universal and task-independent.  
>	1. Local means that the knowledge of the local dynamics of the agent is enough to compute it, and that it is not necessary to know the whole system to determine one’s empowerment. Ideally, the information that the agent itself can acquire should be enough.  
>	2. Universal means that it should be possible to apply empowerment “universally” to every possible agent-world interaction. This is achieved by expressing it in the language of information theory and thus making it applicable for any system that can be probabilistically expressed.  
>	3. Task-independent means that empowerment is not evaluated in regard to a specific goal or external reward state. Instead, empowerment is determined by the agent’s embodiment in the world. In particular, apart from minor niche-dependent parameters, the empowerment formalism should have the very same structure in most situations."  

>	"More concretely, the proposed formulation of empowerment defines it via the concept of potential information flow, or channel capacity, between an agent’s actuator state at earlier times and their sensor state at a later time. The idea behind this is that empowerment would quantify how much an agent can reliably and perceptibly influence the world."

>	"The different scenarios presented here, and in the literature on empowerment in general, are highlighting an important aspect of the empowerment flavour of intrinsic motivation algorithms, namely its universality. The same principle that organizes a swarm of agents into a pattern can also swing the pendulum into an upright position, seek out a central location in a maze, be driven towards a manipulable object, or drive the evolution of sensors. The task-independent nature reflected in this list can be both a blessing and a curse. In many cases the resulting solution, such as swinging the pendulum into the upright position, is the goal implied by default by a human observer. However, if indeed a goal is desired that differs from this default, then empowerment will not be the best solution. At present, the question of how to integrate explicit non-default goals into empowerment is fully open."

>	"Let us conclude with a remark regarding the biological empowerment hypotheses in general: the fact that the default behaviours produced by empowerment seem often to match what intuitive expectations concerning default behaviour seem to imply, there is some relevance in investigating whether some of these behaviours are indeed approximating default behaviours observed in nature. A number of arguments in favour of why empowerment maximizing or similar behaviour could be relevant in biology have been made in (Klyubin et al. 2008), of which in this review we mainly highlighted its role as a measure of sensorimotor efficiency and the advantages that an evolutionary process would confer to more informationally efficient perception-action configurations."


#### Mohamed, Rezende - ["Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning"](http://arxiv.org/abs/1509.08731)
>	"The mutual information is a core statistical quantity that has applications in all areas of machine learning, whether this is in training of density models over multiple data modalities, in maximising the efficiency of noisy transmission channels, or when learning behaviour policies for exploration by artificial agents. Most learning algorithms that involve optimisation of the mutual information rely on the Blahut-Arimoto algorithm - an enumerative algorithm with exponential complexity that is not suitable for modern machine learning applications. This paper provides a new approach for scalable optimisation of the mutual information by merging techniques from variational inference and deep learning. We develop our approach by focusing on the problem of intrinsically-motivated learning, where the mutual information forms the definition of a well-known internal drive known as empowerment. Using a variational lower bound on the mutual information, combined with convolutional networks for handling visual input streams, we develop a stochastic optimisation algorithm that allows for scalable information maximisation and empowerment-based reasoning directly from pixels to actions."

>	"We have developed a new approach for scalable estimation of the mutual information by exploiting recent advances in deep learning and variational inference. We focussed specifically on intrinsic motivation with a reward measure known as empowerment, which requires at its core the efficient computation of the mutual information. By using a variational lower bound on the mutual information, we developed a scalable model and efficient algorithm that expands the applicability of empowerment to high-dimensional problems, with the complexity of our approach being extremely favourable when compared to the complexity of the Blahut-Arimoto algorithm that is currently the standard. The overall system does not require a generative model of the environment to be built, learns using only interactions with the environment, and allows the agent to learn directly from visual information or in continuous state-action spaces. While we chose to develop the algorithm in terms of intrinsic motivation, the mutual information has wide applications in other domains, all which stand to benefit from a scalable algorithm that allows them to exploit the abundance of data and be applied to large-scale problems."

----
>	"Authors developed a scalable method of approximating empowerment, the mutual information between an agent’s actions and the future state of the environment, using variational methods."

>	"This paper presents a variational approach to the maximisation of mutual information in the context of a reinforcement learning agent. Mutual information in this context can provide a learning signal to the agent that is "intrinsically motivated", because it relies solely on the agent's state/beliefs and does not require from the ("outside") user an explicit definition of rewards. Specifically, the learning objective, for a current state s, is the mutual information between the sequence of K actions a proposed by an exploration distribution w(a|s) and the final state s' of the agent after performing these actions. To understand what the properties of this objective, it is useful to consider the form of this mutual information as a difference of conditional entropies: I(a,s'|s) = H(a|s) - H(a|s',s) Where I(.|.) is the (conditional) mutual information and H(.|.) is the (conditional) entropy. This objective thus asks that the agent find an exploration distribution that explores as much as possible (i.e. has high H(a|s) entropy) but is such that these actions have predictable consequences (i.e. lead to predictable state s' so that H(a|s',s) is low). So one could think of the agent as trying to learn to have control of as much of the environment as possible, thus this objective has also been coined as "empowerment".

>	"Interestingly, the framework allows to also learn the state representation s as a function of some "raw" representation x of states."

>	"A major distinction with VIME is that empowerment doesn’t necessarily favor exploration - as stated by Mohamed and Rezende, agents are only ‘curious’ about parts of its environment that can be reached within its internal planning horizon."

  - <https://youtube.com/watch?v=tMiiKXPirAQ> + <https://youtube.com/watch?v=LV5jYY-JFpE> (demo)
  - <https://youtube.com/watch?v=qduxl-vKz1E> + <https://youtube.com/watch?v=DpQKpSAMauY> (Kretov, in russian)
  - <https://www.evernote.com/shard/s189/sh/8c7ff9d9-c321-4e83-a802-58f55ebed9ac/bfc614113180a5f4624390df56e73889> (Larochelle)


#### Stadie, Levine, Abbeel - ["Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models"](http://arxiv.org/abs/1507.00814)
>	"Achieving efficient and scalable exploration in complex domains poses a major challenge in reinforcement learning. While Bayesian and PAC-MDP approaches to the exploration problem offer strong formal guarantees, they are often impractical in higher dimensions due to their reliance on enumerating the state-action space. Hence, exploration in complex domains is often performed with simple epsilon-greedy methods. To achieve more efficient exploration, we develop a method for assigning exploration bonuses based on a concurrently learned model of the system dynamics. By parameterizing our learned model with a neural network, we are able to develop a scalable and efficient approach to exploration bonuses that can be applied to tasks with complex, high-dimensional state spaces. We demonstrate our approach on the task of learning to play Atari games from raw pixel inputs. In this domain, our method offers substantial improvements in exploration efficiency when compared with the standard epsilon greedy approach. As a result of our improved exploration strategy, we are able to achieve state-of-the-art results on several games that pose a major challenge for prior methods."

>	"In the field of reinforcement learning, agents acting in unknown environments face the exploration versus exploitation tradeoff. Without adequate exploration, the agent might fail to discover effective control strategies, particularly in complex domains. Both PAC-MDP algorithms and Bayesian algorithms have managed this tradeoff by assigning exploration bonuses to novel states. In these methods, the novelty of a state-action pair is derived from the number of times an agent has visited that pair. While these approaches offer strong formal guarantees, their requirement of an enumerable representation of the agent’s environment renders them impractical for large-scale tasks. As such, exploration in large RL tasks is still most often performed using simple heuristics, such as the epsilon-greedy strategy, which can be inadequate in more complex settings. To achieve better exploration, we develop a method for assigning exploration bonuses based on a learned model of the system dynamics. Rather than requiring an a priori and enumerable representation of the agent’s environment, we instead propose to learn a state representation from observations, and then optimize a dynamics model concurrently with the policy. The misprediction error in our learned dynamics model is then used to assess the novelty of a given state; since novel states are expected to disagree more strongly with the model than those states that have been visited frequently in the past. These exploration bonuses are motivated by Bayesian exploration bonuses, in which state-action counts serve to capture the uncertainty in the belief space over a model’s transition matrices. Though it is intractable to construct such transition matrices for complex, partially observed tasks with high-dimensional observations such as image pixels, our method captures a similar notion of uncertainty via the misprediction error in the learned dynamics model over the observation space."

>	"There are a number of directions for future work. Our method assumes unpredictability is a good indicator for the need for more exploration, but in highly stochastic environments there can be transitions that remain unpredictable even after sufficient exploration, and a distributional prediction (rather than a single next state prediction) would become important. An important question both in policy learning and model learning with complex, high-dimensional observations is the question of representation. In the case of our model, we learn a representation using an autoencoder trained on prior experience, in an unsupervised setting. More generally, one can imagine learning representations as part of the model learning process in a supervised way, or even sharing representations between the model and the policy. Furthermore, although our method learns a model of the dynamics, the reinforcement learning is performed in a model-free way. We make no attempt to incorporate the model into the policy update except via exploration bonuses. An interesting direction for future work is to incorporate model-based updates with the same type of predictive model."

>	"exploration bonus := error in next-state prediction"  
>	"stochastic environment leads to infinite exploration"  

  - <http://research.microsoft.com/apps/video/default.aspx?id=260045> (Abbeel, 12:30)


#### Nachum, Norouzi, Schuurmans - ["Improving Policy Gradient by Exploring Under-appreciated Rewards"](https://arxiv.org/abs/1611.09321)
>	"This paper presents a novel form of policy gradient for model-free reinforcement learning with improved exploration properties. Current policy-based methods use entropy regularization to encourage undirected exploration of the reward landscape, which is ineffective in high dimensional spaces with sparse rewards. We propose a more directed exploration strategy that promotes exploration of under-appreciated reward regions. An action sequence is considered under-appreciated if its log-probability under the current policy under-estimates its resulting reward. The proposed exploration strategy is easy to implement, requiring small modifications to an implementation of the REINFORCE algorithm. We evaluate the approach on a set of algorithmic tasks that have long challenged RL methods. Our approach reduces hyper-parameter sensitivity and demonstrates significant improvements over baseline methods. Our algorithm successfully solves a benchmark multi-digit addition task and generalizes to long sequences. This is, to our knowledge, the first time that a pure RL method has solved addition using only reward feedback."

>	"Prominent approaches to improving exploration beyond epsilon-greedy in value-based or model-based RL have focused on reducing uncertainty by prioritizing exploration toward states and actions where the agent knows the least. This basic intuition underlies work on counter and recency methods, exploration methods based on uncertainty estimates of values, methods that prioritize learning environment dynamics, and methods that provide an intrinsic motivation or curiosity bonus for exploring unknown states. We relate the concepts of value and policy in RL and propose an exploration strategy based on the discrepancy between the two."

>	"To confirm whether our method is able to find the correct algorithm for multi-digit addition, we investigate its generalization to longer input sequences than provided during training. We evaluate the trained models on inputs up to a length of 2000 digits, even though training sequences were at most 33 characters. For each length, we test the model on 100 randomly generated inputs, stopping when the accuracy falls below 100%. Out of the 60 models trained on addition with UREX, we find that 5 models generalize to numbers up to 2000 digits without any observed mistakes."

  - <https://youtu.be/fZNyHoXgV7M?t=55m45s> (Norouzi)


#### Blundell, Cornebise, Kavukcuoglu, Wierstra - ["Weight Uncertainty in Neural Networks"](https://arxiv.org/abs/1505.05424)
>	"We introduce a new, efficient, principled and backpropagation-compatible algorithm for learning a probability distribution on the weights of a neural network, called Bayes by Backprop. It regularises the weights by minimising a compression cost, known as the variational free energy or the expected lower bound on the marginal likelihood. We show that this principled kind of regularisation yields comparable performance to dropout on MNIST classification. We then demonstrate how the learnt uncertainty in the weights can be used to improve generalisation in non-linear regression problems, and how this weight uncertainty can be used to drive the exploration-exploitation trade-off in reinforcement learning."

>	"P(r|x,a,w) can be modelled by a neural network where w are the weights of the neural network. However if this network is simply fit to observations and the action with the highest expected reward taken at each time, the agent can under-explore, as it may miss more rewarding actions."

>	"Thompson sampling is a popular means of picking an action that trades-off between exploitation (picking the best known action) and exploration (picking what might be a suboptimal arm to learn more). Thompson sampling usually necessitates a Bayesian treatment of the model parameters. At each step, Thompson sampling draws a new set of parameters and then picks the action relative to those parameters. This can be seen as a kind of stochastic hypothesis testing: more probable parameters are drawn more often and thus refuted or confirmed the fastest. More concretely Thompson sampling proceeds as follows:  
>	1. Sample a new set of parameters for the model.  
>	2. Pick the action with the highest expected reward according to the sampled parameters.  
>	3. Update the model. Go to 1."  

>	"Thompson sampling is easily adapted to neural networks using the variational posterior:  
>	1. Sample weights from the variational posterior: w ∼ q(w|θ).  
>	2. Receive the context x.  
>	3. Pick the action a that maximizes E P(r|x,a,w) [r]  
>	4. Receive reward r.  
>	5. Update variational parameters θ. Go to 1."  

>	"Note that it is possible to decrease the variance of the gradient estimates, trading off for reduced exploration, by using more than one Monte Carlo sample, using the corresponding networks as an ensemble and picking the action by minimising the average of the expectations."  

>	"Initially the variational posterior will be close to the prior, and actions will be picked uniformly. As the agent takes actions, the variational posterior will begin to converge, and uncertainty on many parameters can decrease, and so action selection will become more deterministic, focusing on the high expected reward actions discovered so far. It is known that variational methods under-estimate uncertainty which could lead to under-exploration and premature convergence in practice, but we did not find this in practice."

  - <http://videolectures.net/icml2015_blundell_neural_network/> (Blundell)
  - <https://github.com/tabacof/bayesian-nn-uncertainty>
  - <https://github.com/blei-lab/edward/blob/master/examples/bayesian_nn.py>
  - <https://github.com/ferrine/gelato>
  - <https://gist.github.com/rocknrollnerd/c5af642cf217971d93f499e8f70fcb72>


#### Osband, Blundell, Pritzel, van Roy - ["Deep Exploration via Bootstrapped DQN"](http://arxiv.org/abs/1602.04621)
>	"Efficient exploration in complex environments remains a major challenge for reinforcement learning. We propose bootstrapped DQN, a simple algorithm that explores in a computationally and statistically efficient manner through use of randomized value functions. Unlike dithering strategies such as Epsilon-greedy exploration, bootstrapped DQN carries out temporally-extended (or deep) exploration; this can lead to exponentially faster learning. We demonstrate these benefits in complex stochastic MDPs and in the large-scale Arcade Learning Environment. Bootstrapped DQN substantially improves learning times and performance across most Atari games."

>	"One of the reasons deep RL algorithms learn so slowly is that they do not gather the right data to learn about the problem. These algorithms use dithering (taking random actions) to explore their environment - which can be exponentially less efficient that deep exploration which prioritizes potentially informative policies over multiple timesteps. There is a large literature on algorithms for deep exploration for statistically efficient reinforcement learning. The problem is that none of these algorithms are computationally tractable with deep learning. We present the first practical reinforcement learning algorithm that combines deep learning with deep exploration."

>	"In this paper we present bootstrapped DQN as an algorithm for efficient reinforcement learning in complex environments. We demonstrate that the bootstrap can produce useful uncertainty estimates for deep neural networks. Bootstrapped DQN can leverage these uncertainty estimates for deep exploration even in difficult stochastic systems; it also produces several state of the art results in Atari 2600. Bootstrapped DQN is computationally tractable and also naturally scalable to massive parallel systems as per (Nair et al., 2015). We believe that, beyond our specific implementation, randomized value functions represent a promising alternative to dithering for exploration. Bootstrapped DQN practically combines efficient generalization with exploration for complex nonlinear value functions.

>	"Our algorithm, bootstrapped DQN, modifies DQN to produce distribution over Q-values via the bootstrap. At the start of each episode, bootstrapped DQN samples a single Q-value function from its approximate posterior. The agent then follows the policy which is optimal for that sample for the duration of the episode. This is a natural extension of the Thompson sampling heuristic to RL that allows for temporally extended (or deep) exploration. Bootstrapped DQN exhibits deep exploration unlike the naive application of Thompson sampling to RL which resample every timestep."

>	"By contrast, Epsilon-greedy strategies are almost indistinguishable for small values of Epsilon and totally ineffectual for larger values. Our heads explore a diverse range of policies, but still manage to each perform well individually."

>	"Unlike vanilla DQN, bootstrapped DQN can know what it doesn’t know."

>	"Uncertainty estimates allow an agent to direct its exploration at potentially informative states and actions. In bandits, this choice of directed exploration rather than dithering generally categorizes efficient algorithms. The story in RL is not as simple, directed exploration is not enough to guarantee efficiency; the exploration must also be deep. Deep exploration means exploration which is directed over multiple time steps; it can also be called “planning to learn” or “far-sighted” exploration. Unlike bandit problems, which balance actions which are immediately rewarding or immediately informative, RL settings require planning over several time steps. For exploitation, this means that an efficient agent must consider the future rewards over several time steps and not simply the myopic rewards. In exactly the same way, efficient exploration may require taking actions which are neither immediately rewarding, nor immediately informative."

>	"Unlike bandit algorithms, an RL agent can plan to exploit future rewards. Only an RL agent with deep exploration can plan to learn."

>	"Bootstrapped DQN explores in a manner similar to the provably-efficient algorithm PSRL but it uses a bootstrapped neural network to approximate a posterior sample for the value. Unlike PSRL, bootstrapped DQN directly samples a value function and so does not require further planning steps. This algorithm is similar to RLSVI, which is also provably-efficient, but with a neural network instead of linear value function and bootstrap instead of Gaussian sampling. The analysis for the linear setting suggests that this nonlinear approach will work well so long as the distribution {Q1, .., QK} remains stochastically optimistic, or at least as spread out as the “correct” posterior."

  - <http://youtube.com/watch?v=Zm2KoT82O_M> + <http://youtube.com/watch?v=0jvEcC5JvGY> (demo)
  - <http://youtube.com/watch?v=6SAdmG3zAMg>
  - <https://youtu.be/ck4GixLs4ZQ?t=1h27m39s> (Osband) + [slides](https://docs.google.com/presentation/d/1lis0yBGT-uIXnAsi0vlP3SuWD2svMErJWy_LYtfzMOA/)
  - <http://videolectures.net/rldm2015_van_roy_function_randomization/> (30:30, van Roy)
  - <https://youtu.be/mrgJ53TIcQc?t=32m24s> (Pavlov, in russian)
  - <https://github.com/Kaixhin/Atari>
  - <https://github.com/iassael/torch-bootstrapped-dqn>
  - <https://github.com/carpedm20/deep-rl-tensorflow>


#### Osband - ["Risk versus Uncertainty in Deep Learning: Bayes, Bootstrap and the Dangers of Dropout"](http://bayesiandeeplearning.org/papers/BDL_4.pdf)
>	"In this paper we investigate several popular approaches for uncertainty estimation in neural networks. We find that several popular approximations to the uncertainty of a unknown neural net model are in fact approximations to the risk given a fixed model. We review that conflating risk with uncertainty can lead to arbitrarily poor performance in a sequential decision problem. We present a simple and practical solution to this problem based upon smoothed bootstrap sampling."

>	"In sequential decision problems there is an important distinction between risk and uncertainty. We identify risk as inherent stochasticity in a model and uncertainty as the confusion over which model parameters apply. For example, a coin may have a fixed p = 0.5 of heads and so the outcome of any single flip holds some risk; a learning agent may also be uncertain of p. The demarcation between risk and uncertainty is tied to the specific model class, in this case a Bernoulli random variable; with a more detailed model of flip dynamics even the outcome of a coin may not be risky at all. Our distinction is that unlike risk, uncertainty captures the variability of an agent’s posterior belief which can be resolved through statistical analysis of the appropriate data. For a learning agent looking to maximize cumulative utility through time, this distinction represents a crucial dichotomy. Consider the reinforcement learning problem of an agent interacting with its environment while trying to maximize cumulative utility through time. At each timestep, the agent faces a fundamental tradeoff: by exploring uncertain states and actions the agent can learn to improve its future performance, but it may attain better short-run performance by exploiting its existing knowledge. At a high level this effect means uncertain states are more attractive since they can provide important information to the agent going forward. On the other hand, states and actions with high risk are actually less attractive for an agent in both exploration and exploitation. For exploitation, any concave utility will naturally penalize risk. For exploration, risk also makes any single observation less informative. Although colloquially similar, risk and uncertainty can require radically different treatment."

>	"One of the most popular recent suggestions has been to use dropout sampling (where individual neurons are independently set to zero with probability p) to “get uncertainty information from these deep learning models for free – without changing a thing”. Unfortunately, as we now show, dropout sampling can be better thought of as an approximation the risk in y, rather than the uncertainty of the learned model. Further, using a fixed dropout rate p, rather than optimizing this variational parameter can lead an arbitrarily bad approximation to the risk."

>	"We extend the analysis to linear functions and argue that this behavior also carries over to deep learning; extensive computational results support this claim. We investigate the importance of risk and uncertainty in sequential decision problems and why this setting is crucially distinct from standard supervised learning tasks. We highlight the dangers of a naive applications of dropout (or any other approximate risk measure) as a proxy for uncertainty. We present analytical regret bounds for algorithms based upon smoothed bootstrapped uncertainty estimates that complement their strong performance in complex nonlinear domains."


#### Russo, van Roy - ["Learning to Optimize Via Posterior Sampling"](https://arxiv.org/abs/1301.2609)
>	"This paper considers the use of a simple posterior sampling algorithm to balance between exploration and exploitation when learning to optimize actions such as in multi-armed bandit problems. The algorithm, also known as Thompson Sampling and as probability matching, offers significant advantages over the popular upper confidence bound (UCB) approach, and can be applied to problems with finite or infinite action spaces and complicated relationships among action rewards. We make two theoretical contributions. The first establishes a connection between posterior sampling and UCB algorithms. This result lets us convert regret bounds developed for UCB algorithms into Bayesian regret bounds for posterior sampling. Our second theoretical contribution is a Bayesian regret bound for posterior sampling that applies broadly and can be specialized to many model classes. This bound depends on a new notion we refer to as the eluder dimension, which measures the degree of dependence among action rewards. Compared to UCB algorithm Bayesian regret bounds for specific model classes, our general bound matches the best available for linear models and is stronger than the best available for generalized linear models. Further, our analysis provides insight into performance advantages of posterior sampling, which are highlighted through simulation results that demonstrate performance surpassing recently proposed UCB algorithms."

>	"The Thompson Sampling algorithm randomly selects an action according to the probability it is optimal. Although posterior sampling was first proposed almost eighty years ago, it has until recently received little attention in the literature on multi-armed bandits. While its asymptotic convergence has been established in some generality, not much else is known about its theoretical properties in the case of dependent arms, or even in the case of independent arms with general prior distributions. Our work provides some of the first theoretical guarantees."

>	"Our interest in posterior sampling is motivated by several potential advantages over UCB algorithms. While particular UCB algorithms can be extremely effective, performance and computational tractability depends critically on the confidence sets used by the algorithm. For any given model, there is a great deal of design flexibility in choosing the structure of these sets. Because posterior sampling avoids the need for confidence bounds, its use greatly simplifies the design process and admits practical implementations in cases where UCB algorithms are computationally onerous. In addition, we show through simulations that posterior sampling outperforms various UCB algorithms that have been proposed in the literature."

>	"In this paper, we make two theoretical contributions. The first establishes a connection between posterior sampling and UCB algorithms. In particular, we show that while the regret of a UCB algorithm can be bounded in terms of the confidence bounds used by the algorithm, the Bayesian regret of posterior sampling can be bounded in an analogous way by any sequence of confidence bounds. In this sense, posterior sampling preserves many of the appealing theoretical properties of UCB algorithms without requiring explicit, designed, optimism. We show that, due to this connection, existing analysis available for specific UCB algorithms immediately translates to Bayesian regret bounds for posterior sampling."

>	"Our second theoretical contribution is a Bayesian regret bound for posterior sampling that applies broadly and can be specialized to many specific model classes. Our bound depends on a new notion of dimension that measures the degree of dependence among actions. We compare our notion of dimension to the Vapnik-Chervonenkis dimension and explain why that and other measures of dimension used in the supervised learning literature do not suffice when it comes to analyzing posterior sampling."

----
>	"Authors describe an approach to addressing the limitations of the optimistic approach that serves as the basis for the UCB family of algorithms. They describe a method that considers not only the immediate single-period regret but also the information gain to learn from the partial feedback and to optimize the exploration-exploitation trade online."

  - <http://videolectures.net/rldm2015_van_roy_function_randomization/> (van Roy)


#### Osband, van Roy - ["Why is Posterior Sampling Better than Optimism for Reinforcement Learning?"](http://arxiv.org/abs/1607.00215)
>	"Computational results demonstrate that posterior sampling for reinforcement learning (PSRL) dramatically outperforms algorithms driven by optimism, such as UCRL2. We provide insight into the extent of this performance boost and the phenomenon that drives it. We leverage this insight to establish an O(H√S√A√T) expected regret bound for PSRL in finite-horizon episodic Markov decision processes, where H is the horizon, S is the number of states, A is the number of actions and T is the time elapsed. This improves upon the best previous bound of O(HS√A√T) for any reinforcement learning algorithm."

>	"We consider a well-studied reinforcement learning problem in which an agent interacts with a Markov decision process with the aim of maximizing expected cumulative reward. Our focus is on the tabula rasa case, in which the agent has virtually no prior information about the MDP. As such, the agent is unable to generalize across state-action pairs and may have to gather data at each in order to learn an effective decision policy. Key to performance is how the agent balances between exploration to acquire information of long-term benefit and exploitation to maximize expected near-term rewards. In principle, dynamic programming can be applied to compute the so-called Bayes-optimal solution to this problem. However, this is computationally intractable for anything beyond the simplest of toy problems. As such, researchers have proposed and analyzed a number of heuristic reinforcement learning algorithms.

>	The literature on efficient reinforcement learning offers statistical efficiency guarantees for computationally tractable algorithms. These provably efficient algorithms predominantly address the exploration-exploitation trade-off via optimism in the face of uncertainty (OFU): when at a state, the agent assigns to each action an optimistically biased while statistically plausible estimate of future value and selects the action with the greatest estimate. If a selected action is not near-optimal, the estimate must be overly optimistic, in which case the agent learns from the experience. Efficiency relative to less sophisticated exploration strategies stems from the fact that the agent avoids actions that neither yield high value nor informative data.

>	An alternative approach, based on Thompson sampling, involves sampling a statistically plausibly set of action values and selecting the maximizing action. These values can be generated, for example, by sampling from the posterior distribution over MDPs and computing the state-action value function of the sampled MDP. This approach is called posterior sampling for reinforcement learning (PSRL). Computational results demonstrate that PSRL dramatically outperforms algorithms based on OFU. The primary aim of this paper is to provide insight into the extent of this performance boost and the phenomenon that drives it.

>	We argue that applying OFU in a manner that competes with PSRL in terms of statistical efficiency would require intractable computation. As such, OFU-based algorithms presented in the literature sacrifice statistical efficiency to attain computational tractability. We will explain how these algorithms are statistically inefficient. We will also leverage this insight to produce an O(H√S√A√T) expected regret bound for PSRL in finite-horizon episodic Markov decision processes, where H is the horizon, S is the number of states, A is the number of actions and T is the time elapsed. This improves upon the best previous bound of O(HS√A√T) for any reinforcement learning algorithm. We discuss why we believe PSRL satisfies a tighter O(√H√S√A√T), though we have not proved that. We present computational results chosen to enhance insight on how learning times scale with problem parameters. These empirical scalings match our theoretical predictions."

>	"PSRL is orders of magnitude more statistically efficient than UCRL and S-times less computationally expensive. In the future, we believe that analysts will be able to formally specify an OFU approach to RL whose statistical efficiency matches PSRL. However, we argue that the resulting confidence sets which address both the coupling over H and S will result in a computationally intractable optimization problem. For this reason, computationally efficient approaches to OFU RL will sacrifice statistical efficiency; this is why posterior sampling is better than optimism for reinforcement learning."

  - <http://videolectures.net/rldm2015_van_roy_function_randomization/> (van Roy)
  - <https://youtube.com/watch?v=ck4GixLs4ZQ> (Osband) + [slides](https://docs.google.com/presentation/d/1lis0yBGT-uIXnAsi0vlP3SuWD2svMErJWy_LYtfzMOA/)


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



---
### interesting papers - abstractions for states and actions

[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reinforcement-learning---abstractions-for-states-and-actions)


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


#### Tamar, Wu, Thomas, Levine, Abbeel - ["Value Iteration Networks"](http://arxiv.org/abs/1602.02867)
>	"We introduce the value iteration network (VIN): a fully differentiable neural network with a ‘planning module’ embedded within. VINs can learn to plan, and are suitable for predicting outcomes that involve planning-based reasoning, such as policies for reinforcement learning. Key to our approach is a novel differentiable approximation of the value-iteration algorithm, which can be represented as a convolutional neural network, and trained end-to-end using standard backpropagation. We evaluate VIN based policies on discrete and continuous path-planning domains, and on a natural-language based search task. We show that by learning an explicit planning computation, VIN policies generalize better to new, unseen domains."

>	"The introduction of powerful and scalable RL methods has opened up a range of new problems for deep learning. However, few recent works investigate policy architectures that are specifically tailored for planning under uncertainty, and current RL theory and benchmarks rarely investigate the generalization properties of a trained policy. This work takes a step in this direction, by exploring better generalizing policy representations. Our VIN policies learn an approximate planning computation relevant for solving the task, and we have shown that such a computation leads to better generalization in a diverse set of tasks, ranging from simple gridworlds that are amenable to value iteration, to continuous control, and even to navigation of Wikipedia links. In future work we intend to learn different planning computations, based on simulation, or optimal linear control, and combine them with reactive policies, to potentially develop RL solutions for task and motion planning"

>	"In our experiment in continuous control we used hierarchical policy: high-level policy solved low-resolution map and low-level policy executed it. This is very different from options/skills framework. There is one smooth policy that implements everything. We don't need to learn initiation sets or termination sets. But more importantly, the motivation for using hierarchy here was different. The motivation wasn't to increase learning rate or exploration - the motivation was to generalize. We understood that low-resolution map is sufficient for doing planning which promotes generalization, but low-level policy uses the fact that dynamics is similar across different tasks."

----
>	"Its contribution is to offer a new way to think about value iteration in the context of deep networks. It shows how the CNN architecture can be hijacked to implement the Bellman optimality operator, and how the backprop signal can be used to learn a deterministic model of the underlying MDP."

>	"Value iteration is similar enough to a sequence of convolutions and max-pooling layers that you can emulate an (unrolled) planning computation with a deep network. This allows neural nets to do planning, e.g. moving from start to goal in grid-world, or navigating a website to find query."

  - <https://youtube.com/watch?v=tXBHfbHHlKc> (Tamar) + [slides](http://technion.ac.il/~danielm/icml_slides/Talk7.pdf)
  - <https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Value-Iteration-Networks> (Tamar)
  - <https://github.com/karpathy/paper-notes/blob/master/vin.md>
  - <https://blog.acolyer.org/2017/02/09/value-iteration-networks/>
  - <https://github.com/avivt/VIN>
  - <https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks>
  - <https://github.com/zuoxingdong/VIN_TensorFlow>
  - <https://github.com/zuoxingdong/VIN_PyTorch_Visdom>
  - <https://github.com/onlytailei/Value-Iteration-Networks-PyTorch>
  - <https://github.com/kentsommer/pytorch-value-iteration-networks>


#### Florensa, Duan, Abbeel - ["Stochastic Neural Networks for Hierarchical Reinforcement Learning"](https://openreview.net/pdf?id=B1oK8aoxe)
>	"Deep reinforcement learning has achieved many impressive results in recent years. However, many of the deep RL algorithms still employ naive exploration strategies, and they have been shown to perform poorly in tasks with sparse rewards, and/or with long horizons. To tackle these challenges, there are two common approaches. The first approach is to design a hierarchy over the actions, which would require domain-specific knowledge and careful hand-engineering. A different line of work utilizes domain-agnostic intrinsic rewards to guide exploration, which has been shown to be effective in tasks with sparse rewards. However, it is unclear how the knowledge of solving a task can be utilized for other tasks, leading to a high sample complexity overall for the entire collection of tasks. In this paper, we propose a general framework for learning useful skills in a pre-training environment, which can then be utilized in downstream tasks by training a high-level policy over these skills. To learn these skills, we use stochastic neural networks (SNNs) combined with a proxy reward, the design of which requires very minimal domain knowledge about the downstream tasks. Our experiments show that this combination is effective in learning a wide span of interpretable skills in a sample-efficient way, and, when used on downstream tasks, can significantly boost the learning performance uniformly across all these tasks."

>	"We propose a framework for learning a diverse set of skills using stochastic neural networks with minimum supervision, and utilize these skills in a hierarchical architecture to solve challenging tasks with sparse rewards. Our framework successfully combines two parts, firstly an unsupervised procedure to learn a large span of skills using proxy rewards and secondly a hierarchical structure that encapsulates the latter span of skills and allows to re-use them in future tasks. The span of skills learning can be greatly improved by using stochastic neural networks as policies and their additional expressiveness and multimodality. The bilinear integration and the mutual information bonus are key to consistently yield a wide, interpretable span of skills. As for the hierarchical structure, we demonstrate how drastically it can boost the exploration of an agent in a new environment and we demonstrate its relevance for solving complex tasks as mazes or gathering."

>	"One limitations of our current approach is the switching between skills for unstable agents, as reported in the Appendix D.2 for the “Ant” robot. There are multiple future directions to make our framework more robust to such challenging robots, like learning a transition policy or integrating switching in the pretrain task. Other limitations of our current approach are having fixed sub-policies and fixed switch time T during the training of downstream tasks. The first issue can be alleviated by introducing end-to-end training, for example using the new straight-through gradient estimators for Stochastic Computations Graphs with discrete latents (Jang et al., 2016; Maddison et al., 2016). The second issue is not critical for static tasks like the ones used here, as studied in Appendix C.3. But in case of becoming a bottleneck for more complex dynamic tasks, a termination policy could be learned by the Manager, similar to the option framework. Finally, we only used feedforward architectures and hence the decision of what skill to use next only depends on the observation at the moment of switching, not using any sensory information gathered while the previous skill was active. This limitation could be eliminated by introducing a recurrent architecture at the Manager level."

>	"Our SNN hierarchical approach outperforms state-of-the-art intrinsic motivation results like VIME (Houthooft et al., 2016)."

  - <https://youtube.com/playlist?list=PLEbdzN4PXRGVB8NsPffxsBSOCcWFBMQx3> (demo)


#### Bacon, Harb, Precup - ["The Option-Critic Architecture"](http://arxiv.org/abs/1609.05140)
>	"Temporal abstraction is key to scaling up learning and planning in reinforcement learning. While planning with temporally extended actions is well understood, creating such abstractions autonomously from data has remained challenging. We tackle this problem in the framework of options. We derive policy gradient theorems for options and propose a new option-critic architecture capable of learning both the internal policies and the termination conditions of options, in tandem with the policy over options, and without the need to provide any additional rewards or subgoals. Experimental results in both discrete and continuous environments showcase the flexibility and efficiency of the framework."

>	"We developed a general, gradient-based approach for learning simultaneously the intra-option policies and termination conditions, as well as the policy over options, in order to optimize a performance objective for the task at hand. Our ALE experiments demonstrate successful end-to-end training of the options in the presence of nonlinear function approximation. As noted, our approach only requires specifying the number of options. However, if one wanted to use additional pseudo-rewards, the option-critic framework would easily accommodate it. The internal policies and termination function gradients would simply need to be taken with respect to the pseudo-rewards instead of the task reward. A simple instance of this idea, which we used in some of the experiments, is to use additional rewards to encourage options that are indeed temporally extended, by adding a penalty whenever a switching event occurs."

>	"Our approach can work seamlessly with any other heuristic for biasing the set of options towards some desirable property (e.g. compositionality or sparsity), as long as it can be expressed as an additive reward structure. However, as seen in the results, such biasing is not necessary to produce good results. The option-critic architecture relies on the policy gradient theorem, and as discussed in (Thomas 2014), the gradient estimators can be biased in the Qt discounted case. By introducing factors of the form γ^t Π i=1 [1 − βi] in our updates (Thomas 2014, eq (3)), it would be possible to obtain unbiased estimates. However, we do not recommend this approach, since the sample complexity of the unbiased estimators is generally too high and the biased estimators performed well in our experiments."

>	"Perhaps the biggest remaining limitation of our work is the assumption that all options apply everywhere. In the case of function approximation, a natural extension to initiation sets is to use a classifier over features, or some other form of function approximation. As a result, determining which options are allowed may have similar cost to evaluating a policy over options (unlike in the tabular setting, where options with sparse initiation sets lead to faster decisions). This is akin to eligibility traces, which are more expensive than using no trace in the tabular case, but have the same complexity with function approximation. If initiation sets are to be learned, the main constraint that needs to be added is that the options and the policy over them lead to an ergodic chain in the augmented state-option space. This can be expressed as a flow condition that links initiation sets with terminations. The precise description of this condition, as well as sparsity regularization for initiation sets, is left for future work."

  - <http://videolectures.net/deeplearning2016_precup_advanced_lr/> (Precup)
  - <http://youtube.com/watch?v=8r_EoYnPjGk> (Bacon)
  - <http://blog.shakirm.com/2016/07/learning-in-brains-and-machines-3-synergistic-and-modular-action/>


#### Schaul, Horgan, Gregor, Silver - ["Universal Value Function Approximators"](http://jmlr.org/proceedings/papers/v37/schaul15.pdf)
>	"Value functions are a core component of reinforcement learning systems. The main idea is to construct a single function approximator V(s; θ) that estimates the long-term reward from any state s, using parameters θ. In this paper we introduce universal value function approximators V(s, g; θ) that generalise not just over states s but also over goals g. We develop an efficient technique for supervised learning of UVFAs, by factoring observed values into separate embedding vectors for state and goal, and then learning a mapping from s and g to these factored embedding vectors. We show how this technique may be incorporated into a reinforcement learning algorithm that updates the UVFA solely from observed rewards. Finally, we demonstrate that a UVFA can successfully generalise to previously unseen goals."

>	"Value functions may be used to represent knowledge beyond the agent’s overall goal. General value functions Vg(s) represent the utility of any state s in achieving a given goal g (e.g. a waypoint), represented by a pseudo-reward function that takes the place of the real rewards in the problem. Each such value function represents a chunk of knowledge about the environment: how to evaluate or control a specific aspect of the environment (e.g. progress toward a waypoint). A collection of general value functions provides a powerful form of knowledge representation that can be utilised in several ways. For example, the Horde architecture consists of a discrete set of value functions (‘demons’), all of which may be learnt simultaneously from a single stream of experience, by bootstrapping off-policy from successive value estimates. Each value function may also be used to generate a policy or option, for example by acting greedily with respect to the values, and terminating at goal states. Such a collection of options can be used to provide a temporally abstract action-space for learning or planning. Finally, a collection of value functions can be used as a predictive representation of state, where the predicted values themselves are used as a feature vector. In large problems, the value function is typically represented by a function approximator V(s, θ), such as a linear combination of features or a neural network with parameters θ. The function approximator exploits the structure in the state space to efficiently learn the value of observed states and generalise to the value of similar, unseen states. However, the goal space often contains just as much structure as the state space. Consider for example the case where the agent’s goal is described by a single desired state: it is clear that there is just as much similarity between the value of nearby goals as there is between the value of nearby states. Our main idea is to extend the idea of value function approximation to both states s and goals g, using a universal value function approximator V(s, g, θ). A sufficiently expressive function approximator can in principle identify and exploit structure across both s and g. By universal, we mean that the value function can generalise to any goal g in a set G of possible goals: for example a discrete set of goal states; their power set; a set of continuous goal regions; or a vector representation of arbitrary pseudo-reward functions. This UVFA effectively represents an infinite Horde of demons that summarizes a whole class of predictions in a single object. Any system that enumerates separate value functions and learns each individually (like the Horde) is hampered in its scalability, as it cannot take advantage of any shared structure (unless the demons share parameters). In contrast, UVFAs can exploit two kinds of structure between goals: similarity encoded a priori in the goal representations g, and the structure in the induced value functions discovered bottom-up. Also, the complexity of UVFA learning does not depend on the number of demons but on the inherent domain complexity. This complexity is larger than standard value function approximation, and representing a UVFA may require a rich function approximator such as a deep neural network. Learning a UVFA poses special challenges. In general, the agent will only see a small subset of possible combinations of states and goals (s, g), but we would like to generalise in several ways. Even in a supervised learning context, when the true value Vg(s) is provided, this is a challenging regression problem."

>	"On the Atari game of Ms Pacman, we then demonstrate that UVFAs can scale to larger visual input spaces and different types of goals, and show they generalize across policies for obtaining possible pellets."

>	"This paper has developed a universal approximator for goal-directed knowledge. We have demonstrated that our UVFA model is learnable either from supervised targets, or directly from real experience; and that it generalises effectively to unseen goals. We conclude by discussing several ways in which UVFAs may be used. First, UVFAs can be used for transfer learning to new tasks with the same dynamics but different goals. Specifically, the values V(s, g; θ) in a UVFA can be used to initialise a new, single value function Vg(s) for a new task with unseen goal g. We demonstrate that an agent which starts from transferred values in this fashion can learn to solve the new task g considerably faster than random value initialization. Second, generalized value functions can be used as features to represent state; this is a form of predictive representation. A UVFA compresses a large or infinite number of predictions into a short feature vector. Specifically, the state embedding φ(s) can be used as a feature vector that represents state s. Furthermore, the goal embedding φ(g) can be used as a separate feature vector that represents state g. These features can capture non-trivial structure in the domain. Third, a UVFA could be used to generate temporally abstract options. For any goal g a corresponding option may be constructed that acts (soft-)greedily with respect to V(s, g; θ) and terminates e.g. upon reaching g. The UVFA then effectively provides a universal option that represents (approximately) optimal behaviour towards any goal g∈ G. This in turn allows a hierarchical policy to choose any goal g∈ G as a temporally abstract action. Finally, a UVFA could also be used as a universal option model. Specifically, if pseudorewards are defined by goal achievement, then V(s, g; θ) approximates the (discounted) probability of reaching g from s, under a policy that tries to reach it."

  - <http://videolectures.net/icml2015_schaul_universal_value/> (Schaul)
  - <http://schaul.site44.com/publications/uvfa-slides.pdf>



---
### interesting papers - model-based methods

[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reinforcement-learning---simulation-and-planning) on simulation and planning  
[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reinforcement-learning---memory) on memory  


#### Hausknecht, Stone - ["Deep Recurrent Q-Learning for Partially Observable MDPs"](http://arxiv.org/abs/1507.06527)
>	"Deep Reinforcement Learning has yielded proficient controllers for complex tasks. However, these controllers have limited memory and rely on being able to perceive the complete game screen at each decision point. To address these shortcomings, this article investigates the effects of adding recurrency to a Deep Q-Network by replacing the first post-convolutional fully-connected layer with a recurrent LSTM. The resulting Deep Recurrent Q-Network exhibits similar performance on standard Atari 2600 MDPs but better performance on equivalent partially observed domains featuring flickering game screens. Results indicate that given the same length of history, recurrency allows partial information to be integrated through time and is superior to alternatives such as stacking a history of frames in the network's input layer. We additionally show that when trained with partial observations, DRQN's performance at evaluation time scales as a function of observability. Similarly, when trained with full observations and evaluated with partial observations, DRQN's performance degrades more gracefully than that of DQN. We therefore conclude that when dealing with partially observed domains, the use of recurrency confers tangible benefits."

>	"Real-world tasks often feature incomplete and noisy state information, resulting from partial observability. We modify DQN to better deal with the noisy observations characteristic of POMDPs by leveraging advances in Recurrent Neural Networks. More specifically we combined a Long Short Term Memory with a Deep Q-Network and show the resulting Deep Recurrent Q-Network, despite the lack of convolutional velocity detection, is better equipped than a standard Deep Q-Network to handle the type of partial observability induced by flickering game screens. Further analysis shows that DRQN, when trained with partial observations, can generalize its policies to the case of complete observations. On the Flickering Pong domain, performance scales with the observability of the domain, reaching near-perfect performance when every game screen is observed. This result indicates that the recurrent network learns policies that are both robust enough to handle to missing game screens, and scalable enough to improve performance. Generalization also occurs in the opposite direction: when trained on unobscured Atari games and evaluated against obscured games, DRQN’s performance generalizes better than DQN’s at all levels of partial information. Our results indicate that given access to the same amount of history, processing observations in a recurrent layer (like DRQN) rather than stacking frames in the input layer (like DQN) yields better performance on POMDPs and better generalization for both POMDPs and MDPs."

----
>	"Demonstrated that recurrent Q learning can perform the required information integration to resolve short-term partial observability (e.g. to estimate velocities) that is achieved via stacks of frames in the original DQN architecture."

  - <https://youtube.com/watch?v=bE5DIJvZexc> (Fritzler, in russian)
  - <https://github.com/mhauskn/dqn/tree/recurrent>
  - <https://github.com/awjuliani/DeepRL-Agents/blob/master/Deep-Recurrent-Q-Network.ipynb>


#### Oh, Guo, Lee, Lewis, Singh - ["Action-Conditional Video Prediction using Deep Networks in Atari Games"](http://arxiv.org/abs/1507.08750)
>	"Motivated by vision-based reinforcement learning problems, in particular Atari games from the recent benchmark Aracade Learning Environment, we consider spatio-temporal prediction problems where future (image-)frames are dependent on control variables or actions as well as previous frames. While not composed of natural scenes, frames in Atari games are high-dimensional in size, can involve tens of objects with one or more objects being controlled by the actions directly and many other objects being influenced indirectly, can involve entry and departure of objects, and can involve deep partial observability. We propose and evaluate two deep neural network architectures that consist of encoding, action-conditional transformation, and decoding layers based on convolutional neural networks and recurrent neural networks. Experimental results show that the proposed architectures are able to generate visually-realistic frames that are also useful for control over approximately 100-step action-conditional futures in some games. To the best of our knowledge, this paper is the first to make and evaluate long-term predictions on high-dimensional video conditioned by control inputs."

>	"Modeling videos (i.e., building a generative model) is still a very challenging problem because it usually involves high-dimensional natural-scene data with complex temporal dynamics. Thus, recent studies have mostly focused on modeling simple video data, such as bouncing balls or small video patches, where the next frame is highly predictable based on the previous frames. In many applications, however, future frames are not only dependent on previous frames but also on additional control or action variables. For example, the first-person-view in a vehicle is affected by wheel-steering and acceleration actions. The camera observation of a robot is similarly dependent on its movement and changes of its camera angle. More generally, in vision-based reinforcement learning problems, learning to predict future images conditioned on future actions amounts to learning a model of the dynamics of the agent-environment interaction; such transition-models are an essential component of model-based learning approaches to RL."

>	"The encoding part computes high-level abstractions of input frames, the action-conditional transformation part predicts the abstraction of the next frame conditioned on the action, and finally the decoding part maps the predicted high-level abstraction to a detailed frame. The feedforward architecture takes the last 4 frames as input while the recurrent architecture takes just the last frame but has recurrent connections. Our experimental results on predicting images in Atari games show that our architectures are able to generate realistic frames over 100-step action-conditional future frames without diverging. We show that the representations learned by our architectures 1) approximately capture natural similarity among actions, and 2) discover which objects are directly controlled by the agent’s actions and which are only indirectly influenced or not controlled at all. We evaluated the usefulness of our architectures for control in two ways: 1) by replacing emulator frames with predicted frames in a previously-learned model-free controller (DQN; DeepMind’s state of the art Deep-Q-Network for Atari Games), and 2) by using the predicted frames to drive a more informed than random exploration strategy to improve a model-free controller (also DQN)."

  - <https://sites.google.com/a/umich.edu/junhyuk-oh/action-conditional-video-prediction> (demos)
  - <https://youtu.be/igm38BakyAg?t=15m26s> (Lee)
  - <http://research.microsoft.com/apps/video/default.aspx?id=259646> (17:30)
  - <https://github.com/junhyukoh/nips2015-action-conditional-video-prediction>


#### Guo, Singh, Lee, Lewis, Wang - ["Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning"](https://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning)
>	"The combination of modern Reinforcement Learning and Deep Learning approaches holds the promise of making significant progress on challenging applications requiring both rich perception and policy-selection. The Arcade Learning Environment provides a set of Atari games that represent a useful benchmark set of such applications. A recent breakthrough in combining model-free reinforcement learning with deep learning, called DQN, achieves the best realtime agents thus far. Planning-based approaches achieve far higher scores than the best model-free approaches, but they exploit information that is not available to human players, and they are orders of magnitude slower than needed for real-time play. Our main goal in this work is to build a better real-time Atari game playing agent than DQN. The central idea is to use the slow planning-based agents to provide training data for a deep-learning architecture capable of real-time play. We proposed new agents based on this idea and show that they outperform DQN."

----
>	"run planning algorithm on episodes to get dataset (screen, best action) + train CNN policy as classification task with cross entropy loss"  
>	"deterministic version of ALE"  
>	"Upper Confidence Bound 1 applied to trees (UCT) as planning algorithm"  
>	"DAGGER algorithm for data collection"  

  - <https://youtube.com/watch?v=B3b6NLUxN3U> (Singh)
  - <https://youtube.com/watch?v=igm38BakyAg> (Lee)
  - <https://youtube.com/watch?v=mZtlW_xtarI&t=59m25s> (Levine)
  - <https://youtu.be/ywzZJ4L32xc?t=6m39s> (Pavlov)


#### Guez, Silver, Dayan - ["Efficient Bayes-Adaptive Reinforcement Learning using Sample-Based Search"](https://arxiv.org/abs/1205.3109)
>	"Bayesian model-based reinforcement learning is a formally elegant approach to learning optimal behaviour under model uncertainty, trading off exploration and exploitation in an ideal way. Unfortunately, finding the resulting Bayes-optimal policies is notoriously taxing, since the search space becomes enormous. In this paper we introduce a tractable, sample-based method for approximate Bayes-optimal planning which exploits Monte-Carlo tree search. Our approach outperformed prior Bayesian model-based RL algorithms by a significant margin on several well-known benchmark problems – because it avoids expensive applications of Bayes rule within the search tree by lazily sampling models from the current beliefs. We illustrate the advantages of our approach by showing it working in an infinite state space domain which is qualitatively out of reach of almost all previous work in Bayesian exploration."

>	"We suggested a sample-based algorithm for Bayesian RL called BAMCP that significantly surpassed the performance of existing algorithms on several standard tasks. We showed that BAMCP can tackle larger and more complex tasks generated from a structured prior, where existing approaches scale poorly. In addition, BAMCP provably converges to the Bayes-optimal solution. The main idea is to employ Monte-Carlo tree search to explore the augmented Bayes-adaptive search space efficiently. The naive implementation of that idea is the proposed BA-UCT algorithm, which cannot scale for most priors due to expensive belief updates inside the search tree. We introduced three modifications to obtain a computationally tractable sample-based algorithm: root sampling, which only requires beliefs to be sampled at the start of each simulation; a model-free RL algorithm that learns a rollout policy; and the use of a lazy sampling scheme to sample the posterior beliefs cheaply."


#### Castronovo, Francois-Lavet, Fonteneau, Ernst, Couetoux - ["Approximate Bayes Optimal Policy Search using Neural Networks"](http://orbi.ulg.ac.be/bitstream/2268/204410/1/ANN-BRL.pdf)
>	"Bayesian Reinforcement Learning agents aim to maximise the expected collected rewards obtained when interacting with an unknown Markov Decision Process while using some prior knowledge. State-of-the-art BRL agents rely on frequent updates of the belief on the MDP, as new observations of the environment are made. This offers theoretical guarantees to converge to an optimum, but is computationally intractable, even on small-scale problems. In this paper, we present a method that circumvents this issue by training a parametric policy able to recommend an action directly from raw observations. Artificial Neural Networks (ANNs) are used to represent this policy, and are trained on the trajectories sampled from the prior. The trained model is then used online, and is able to act on the real MDP at a very low computational cost. Our new algorithm shows strong empirical performance, on a wide range of test problems, and is robust to inaccuracies of the prior distribution."

>	"State-of-the-art Bayesian algorithms generally do not use offline training. Instead, they rely on Bayes updates and sampling techniques during the interaction, which may be too computationally expensive, even on very small MDPs. In order to reduce significantly this cost, we propose a new practical algorithm to solve BAMDPs: Artificial Neural Networks for Bayesian Reinforcement Learning. Our algorithm aims at finding an optimal policy, i.e. a mapping from observations to actions, which maximises the rewards in a certain environment. This policy is trained to act optimally on some MDPs sampled from the prior distribution, and then it is used in the test environment. By design, our approach does not use any Bayes update, and is thus computationally inexpensive during online interactions."

>	"We developed ANN-BRL, an offline policy-search algorithm for addressing BAMDPs. As shown by our experiments, ANN-BRL obtained state-of-the-art performance on all benchmarks considered in this paper. In particular, on the most challenging benchmark 9, a score 4 times higher than the one measured for the second best algorithm has been observed. Moreover, ANN-BRL is able to make online decisions faster than most BRL algorithms. Our idea is to define a parametric policy as an ANN, and train it using backpropagation algorithm. This requires a training set made of observations-action pairs and in order to generate this dataset, several simulations have been performed on MDPs drawn from prior distribution. In theory, we should label each example with a Bayes optimal action. However, those are too expensive to compute for the whole dataset. Instead, we chose to use optimal actions under full observability hypothesis. Due to the modularity of our approach, a better labelling technique could easily be integrated in ANN-BRL, and may bring stronger empirical results. Moreover, two types of features have been considered for representing the current history: Q-values and transition counters. The use of Q-values allows to reach state-of-the-art performance on most benchmarks and outperfom all other algorithms on the most difficult one. On the contrary, computing a good policy from transition counters only is a difficult task to achieve, even for Artificial Neural Networks. Nevertheless, we found that the difference between this approach and state-of-the-art algorithms was much less noticeable when prior distribution differs from test distribution, which means that at least in some cases, it is possible to compute efficient policies without relying on online computationally expensive tools such as Q-values."



---
### interesting papers - value-based methods

[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reinforcement-learning---algorithms)


#### Mnih, Kavukcuoglu, Silver, Graves, Antonoglou, Wierstra, Riedmiller - ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602)
>	"We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. We apply our method to seven Atari 2600 games from the Arcade Learning Environment, with no adjustment of the architecture or learning algorithm. We find that it outperforms all previous approaches on six of the games and surpasses a human expert on three of them."

  - <http://youtube.com/watch?v=EfGD2qveGdQ> (demo)
  - <http://youtu.be/XAbLn66iHcQ?t=1h41m21s> + <http://youtube.com/watch?v=0xo1Ldx3L5Q> (3D racing demo)
  - <http://youtube.com/watch?v=nMR5mjCFZCw> (3D labyrinth demo)
  - <http://youtube.com/watch?v=re6hkcTWVUY> (Doom gameplay demo)
  - <https://youtube.com/watch?v=6jlaBD9LCnM> + <https://youtube.com/watch?v=6JT6_dRcKAw> (blockworld demo)
  - <http://youtube.com/user/eldubro/videos> (demos)
  - <http://youtube.com/watch?v=iqXKQf2BOSE> (demo)
  - <http://videolectures.net/nipsworkshops2013_mnih_atari/> (Mnih)
  - <http://youtube.com/watch?v=xzM7eI7caRk> (Mnih)
  - <http://youtube.com/watch?v=dV80NAlEins> (de Freitas)
  - <http://youtube.com/watch?v=HUmEbUkeQHg> (de Freitas)
  - <http://youtube.com/watch?v=mrgJ53TIcQc> (Pavlov, in russian)
  - <http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html>
  - <https://github.com/khanhptnk/deep-q-tensorflow>
  - <https://github.com/nivwusquorum/tensorflow-deepq>
  - <https://github.com/devsisters/DQN-tensorflow>
  - <https://github.com/carpedm20/deep-rl-tensorflow>
  - <https://github.com/VinF/deer>
  - <https://github.com/osh/kerlym>
  - <https://github.com/Jabberwockyll/deep_rl_ale>
  - <https://github.com/DanielTakeshi/rl_algorithms/blob/master/dqn/dqn.py>


#### Wang, Schaul, Hessel, van Hasselt, Lanctot, de Freitas - ["Dueling Network Architectures for Deep Reinforcement Learning"](http://arxiv.org/abs/1511.06581)
>	"In recent years there have been many successes of using deep representations in reinforcement learning. Still, many of these applications use conventional architectures, such as convolutional networks, LSTMs, or auto-encoders. In this paper, we present a new neural network architecture for model-free reinforcement learning. Our dueling network represents two separate estimators: one for the state value function and one for the state-dependent action advantage function. The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm. Our results show that this architecture leads to better policy evaluation in the presence of many similar-valued actions. Moreover, the dueling architecture enables our RL agent to outperform the state-of-the-art on the Atari 2600 domain."

>	"The advantage of the dueling architecture lies partly in its ability to learn the state-value function efficiently. With every update of the Q values in the dueling architecture, the value stream V is updated – this contrasts with the updates in a single-stream architecture where only the value for one of the actions is updated, the values for all other actions remain untouched. This more frequent updating of the value stream in our approach allocates more resources to V, and thus allows for better approximation of the state values, which in turn need to be accurate for temporal difference-based methods like Q-learning to work. This phenomenon is reflected in the experiments, where the advantage of the dueling architecture over single-stream Q networks grows when the number of actions is large. Furthermore, the differences between Q-values for a given state are often very small relative to the magnitude of Q. For example, after training with DDQN on the game of Seaquest, the average action gap (the gap between the Q values of the best and the second best action in a given state) across visited states is roughly 0.04, whereas the average state value across those states is about 15. This difference in scales can lead to small amounts of noise in the updates which can lead to reorderings of the actions, and thus make the nearly greedy policy switch abruptly. The dueling architecture with its separate advantage stream is robust to such effects."

----
>	"In advantage learning one throws away information that is not needed for coming up with a good policy. The argument is that throwing away information allows you to focus your resources on learning what is important. As an example consider Tetris when you gain a unit reward for every time step you survive. Arguably the optimal value function takes on large values when the screen is near empty, while it takes on small values when the screen is near full. The range of differences can be enormous (from millions to zero). However, for optimal decision making how long you survive does not matter. What matters is the small differences in how the screen is filled up because this is what determines where to put the individual pieces. If you learn an action value function and your algorithm focuses on something like the mean square error, i.e., getting the magnitudes right, it is very plausible that most resources of the learning algorithm will be spent on capturing how big the values are, while little resource will be spent on capturing the value differences between the actions. This is what advantage learning can fix. The fix comes because advantage learning does not need to wait until the value magnitudes are properly captured before it can start learning the value differences. As can be seen from this example, advantage learning is expected to make a bigger difference where the span of optimal values is orders of magnitudes larger than action-value differences."

>	"Many recent developments blur the distinction between model and algorithm. This is profound - at least for someone with training in statistics. Ziyu Wang replaced the convnet of DQN and re-run exactly the same algorithm but with a different net (a slight modification of the old net with two streams which he calls the dueling architecture). That is, everything is the same, but only the representation (neural net) changed slightly to allow for computation of not only the Q function, but also the value and advantage functions. The simple modification resulted in a massive performance boost. For example, for the Seaquest game, DQN of the Nature paper scored 4,216 points, while the modified net of Ziyu leads to a score of 37,361 points. For comparison, the best human we have found scores 40,425 points. Importantly, many modifications of DQN only improve on the 4,216 score by a few hundred points, while the Ziyu's network change using the old vanilla DQN code and gradient clipping increases the score by nearly a factor of 10. I emphasize that what Ziyu did was he changed the network. He did not change the algorithm. However, the computations performed by the agent changed remarkably. Moreover, the modified net could be used by any other Q learning algorithm. RL people typically try to change equations and write new algorithms, instead here the thing that changed was the net. The equations are implicit in the network. One can either construct networks or play with equations to achieve similar goals."

  - <https://youtube.com/watch?v=TpGuQaswaHs> + <https://youtube.com/watch?v=oNLITLfrvQY> (demos)
  - <http://techtalks.tv/talks/dueling-network-architectures-for-deep-reinforcement-learning/62381/> (Wang)
  - <https://youtu.be/mrgJ53TIcQc?t=35m4s> (Pavlov, in russian)
  - <http://torch.ch/blog/2016/04/30/dueling_dqn.html>
  - <https://github.com/carpedm20/deep-rl-tensorflow>
  - <https://github.com/Kaixhin/Atari>
  - <https://github.com/tambetm/gymexperiments>


#### van Hasselt, Guez, Silver - ["Deep Reinforcement Learning with Double Q-Learning"](http://arxiv.org/abs/1509.06461)
>	"The popular Q-learning algorithm is known to overestimate action values under certain conditions. It was not previously known whether, in practice, such overestimations are common, whether this harms performance, and whether they can generally be prevented. In this paper, we answer all these questions affirmatively. In particular, we first show that the recent DQN algorithm, which combines Q-learning with a deep neural network, suffers from substantial overestimations in some games in the Atari 2600 domain. We then show that the idea behind the Double Q-learning algorithm, which was introduced in a tabular setting, can be generalized to work with large-scale function approximation. We propose a specific adaptation to the DQN algorithm and show that the resulting algorithm not only reduces the observed overestimations, as hypothesized, but that this also leads to much better performance on several games."

>	"This paper has five contributions. First, we have shown why Q-learning can be overoptimistic in large-scale problems, even if these are deterministic, due to the inherent estimation errors of learning. Second, by analyzing the value estimates on Atari games we have shown that these overestimations are more common and severe in practice than previously acknowledged. Third, we have shown that Double Q-learning can be used at scale to successfully reduce this overoptimism, resulting in more stable and reliable learning. Fourth, we have proposed a specific implementation called Double DQN, that uses the existing architecture and deep neural network of the DQN algorithm without requiring additional networks or parameters. Finally, we have shown that Double DQN finds better policies, obtaining new state-of-the-art results on the Atari 2600 domain."

  - <https://youtu.be/qLaDWKd61Ig?t=32m52s> (Silver)
  - <https://youtu.be/mrgJ53TIcQc?t=17m31s> (Pavlov, in russian)
  - <https://github.com/carpedm20/deep-rl-tensorflow>
  - <https://github.com/Kaixhin/Atari>


#### Schaul, Quan, Antonoglou, Silver - ["Prioritized Experience Replay"](http://arxiv.org/abs/1511.05952)
>	"Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In prior work, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance. In this paper we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently. We use prioritized experience replay in the Deep Q-Network algorithm, which achieved human-level performance in Atari games. DQN with prioritized experience replay achieves a new state-of-the-art, outperforming DQN with uniform replay on 42 out of 57 games."

>	"Online reinforcement learning agents incrementally update their parameters (of the policy, value function or model) while they observe a stream of experience. In their simplest form, they discard incoming data immediately, after a single update. Two issues with this are (a) strongly correlated updates that break the i.i.d. assumption of many popular stochastic gradient-based algorithms, and (b) the rapid forgetting of possibly rare experiences that would be useful later on. Experience replay a ddresses both of these issues: with experience stored in a replay memory, it becomes possible to break the temporal correlations by mixing more and less recent experience for the updates, and rare experience will be used for more than just a single update. DQN used a large sliding-window replay memory, sampled from it uniformly at random, and effectively revisited each transition eight times. In general, experience replay can reduce the amount of experience required to learn, and replace it with more computation and more memory – which are often cheaper resources than the RL agent’s interactions with its environment."

>	"In this paper, we investigate how prioritizing which transitions are replayed can make experience replay more efficient and effective than if all transitions are replayed uniformly. The key idea is that an RL agent can learn more effectively from some transitions than from others. Transitions may be more or less surprising, redundant, or task-relevant. Some transitions may not be immediately useful to the agent, but might become so when the agent competence increases (Schmidhuber, 1991). We propose to more frequently replay transitions with high expected learning progress, as measured by the magnitude of their temporal-difference error. This prioritization can lead to a loss of diversity, which we alleviate with stochastic prioritization, and introduce bias, which we correct with importance sampling."

>	"Using a replay memory leads to design choices at two levels: which experience to store and which to forget, and which experience to replay (and how to do so). This paper addresses only the latter: making the most effective use of the replay memory for learning, assuming that its contents are outside of our control."

>	"We find that adding prioritized replay to DQN leads to a substantial improvement in final score on 42 out of 57 games, with the median normalized performance score across 57 games increased from 69% to 97%. Furthermore, we find that the boost from prioritized experience replay is complementary to the one from introducing Double Q-learning into DQN: performance increases another notch, leading to the current state-of-the-art on the Atari benchmark. Compared to Double DQN, the mean performance increased from 389% to 551%, and the median performance from 110% to 128% bringing additional games such as River Raid, Seaquest and Surround to a human level for the first time, and making large jumps on others (e.g., Atlantis, Gopher, James Bond 007 or Space Invaders)."

>	"We stumbled upon another phenomenon (obvious in retrospect), namely that some fraction of the visited transitions are never replayed before they drop out of the sliding window memory, and many more are first replayed only a long time after they are encountered. Also, uniform sampling is implicitly biased toward out-of-date transitions that were generated by a policy that has typically seen hundreds of thousands of updates since. Prioritized replay with its bonus for unseen transitions directly corrects the first of these issues, and also tends to help with the second one, as more recent transitions tend to have larger error – this is because old transitions will have had more opportunities to have them corrected, and because novel data tends to be less well predicted by the value function."

>	"We hypothesize that deep neural networks interact with prioritized replay in another interesting way. When we distinguish learning the value given a representation (i.e., the top layers) from learning an improved representation (i.e., the bottom layers), then transitions for which the representation is good will quickly reduce their error and then be replayed much less, increasing the learning focus on others where the representation is poor, thus putting more resources into distinguishing aliased states – if the observations and network capacity allow for it."

>	"Feedback for Exploration: An interesting side-effect of prioritized replay is that the total number Mi that a transition will end up being replayed varies widely, and this gives a rough indication of how useful it was to the agent. This potentially valuable signal can be fed back to the exploration strategy that generates the transitions. For example, we could sample exploration hyper-parameters (such as the fraction of random actions, the Boltzmann temperature, or the amount of intrinsic reward to mix in) from a parameterized distribution at the beginning of each episode, monitor the usefulness of the experience via Mi, and update the distribution toward generating more useful experience. Or, in a parallel system like the Gorila agent, it could guide resource allocation between a collection of concurrent but heterogeneous “actors”, each with different exploration hyper-parameters."

>	"Prioritized Memories: Considerations that help determine which transitions to replay are likely to be relevant for determining which memories to store and when to erase them (i.e., when it becomes unlikely that we would ever want to replay them anymore). An explicit control over which memories to keep or erase can help reduce the required total memory size, because it reduces redundancy (frequently visited transitions will have low error, so many of them will be dropped), while automatically adjusting for what has been learned already (dropping many of the ‘easy’ transitions) and biasing the contents of the memory to where the errors remain high. This is a non-trivial aspect, because memory requirements for DQN are currently dominated by the size of the replay memory, no longer by the size of the neural network. Erasing is a more final decision than reducing the replay probability, thus an even stronger emphasis of diversity may be necessary, for example by tracking the age of each transitions and using it to modulate the priority in such a way as to preserve sufficient old experience to prevent cycles (related to ‘hall of fame’ ideas in multi-agent literature) or collapsing value functions. The priority mechanism is also flexible enough to permit integrating experience from other sources, such as from a planner or from human expert trajectories, since knowing the source can be used to modulate each transition’s priority, for example in such a way as to preserve a sufficient fraction of external experience in memory."

>	"Numerous neuroscience studies have identified mechanisms of experience replay in the hippocampus of rodents, where sequences of prior experience are replayed, either during awake resting or sleep, and in particular that this happens more for rewarded paths. Furthermore, there is a likely link between increased replay of an experience, and how much can be learned from it, or its TD-error."

  - <https://youtu.be/mrgJ53TIcQc?t=25m43s> (Pavlov, in russian)
  - <https://github.com/Kaixhin/Atari>
  - <https://github.com/carpedm20/deep-rl-tensorflow>


#### Heinrich, Silver - ["Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"](http://arxiv.org/abs/1603.01121) (Poker)
>	"Many real-world applications can be described as large-scale games of imperfect information. To deal with these challenging domains, prior work has focused on computing Nash equilibria in a handcrafted abstraction of the domain. In this paper we introduce the first scalable end-to-end approach to learning approximate Nash equilibria without any prior knowledge. Our method combines fictitious self-play with deep reinforcement learning. When applied to Leduc poker, Neural Fictitious Self-Play (NFSP) approached a Nash equilibrium, whereas common reinforcement learning methods diverged. In Limit Texas Holdem, a poker game of real-world scale, NFSP learnt a competitive strategy that approached the performance of human experts and state-of-the-art methods."

>	"We have introduced NFSP, the first end-to-end deep reinforcement learning approach to learning approximate Nash equilibria of imperfect-information games from self-play. NFSP addresses three problems. Firstly, NFSP agents learn without prior knowledge. Secondly, they do not rely on local search at runtime. Thirdly, they converge to approximate Nash equilibria in self-play. Our empirical results provide the following insights. The performance of fictitious play degrades gracefully with various approximation errors. NFSP converges reliably to approximate Nash equilibria in a small poker game, whereas DQN’s greedy and average strategies do not. NFSP learned a competitive strategy in a real-world scale imperfect-information game from scratch without using explicit prior knowledge. In this work, we focussed on imperfect-information two-player zero-sum games. Fictitious play, however, is also guaranteed to converge to Nash equilibria in cooperative, potential games. It is therefore conceivable that NFSP can be successfully applied to these games as well. Furthermore, recent developments in continuous-action reinforcement learning (Lillicrap et al., 2015) could enable NFSP to be applied to continuous-action games, which current game-theoretic methods cannot deal with directly."

  - <http://techtalks.tv/talks/deep-reinforcement-learning/62360/> (Silver, 1:05:00)



---
### interesting papers - policy-based methods

[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reinforcement-learning---algorithms)


#### Duan, Chen, Houthooft, Schulman, Abbeel - ["Benchmarking Deep Reinforcement Learning for Continuous Control"](http://arxiv.org/abs/1604.06778)
>	"Recently, researchers have made significant progress combining the advances in deep learning for learning feature representations with reinforcement learning. Some notable examples include training agents to play Atari games based on raw pixel data and to acquire advanced manipulation skills using raw sensory inputs. However, it has been difficult to quantify progress in the domain of continuous control due to the lack of a commonly adopted benchmark. In this work, we present a benchmark suite of continuous control tasks, including classic tasks like cart-pole swing-up, tasks with very high state and action dimensionality such as 3D humanoid locomotion, tasks with partial observations, and tasks with hierarchical structure. We report novel findings based on the systematic evaluation of a range of implemented reinforcement learning algorithms. Both the benchmark and reference implementations are released open-source in order to facilitate experimental reproducibility and to encourage adoption by other researchers."

>	"In this work, a benchmark of continuous control problems for reinforcement learning is presented, covering a wide variety of challenging tasks. We implemented several reinforcement learning algorithms, and presented them in the context of general policy parameterizations. Results show that among the implemented algorithms, TNPG, TRPO, and DDPG are effective methods for training deep neural network policies. Still, the poor performance on the proposed hierarchical tasks calls for new algorithms to be developed. Implementing and evaluating existing and newly proposed algorithms will be our continued effort. By providing an open-source release of the benchmark, we encourage other researchers to evaluate their algorithms on the proposed tasks."

>	"This paper benchmarks performance over a wide range of tasks of Reinforce, (Truncated) Natural Policy Gradient, RWR, REPS, TRPO, CEM, CMA-ES, DDPG."

  - <http://techtalks.tv/talks/benchmarking-deep-reinforcement-learning-for-continuous-control/62380/> (Duan)
  - <https://github.com/rllab/rllab>


#### Salimans, Ho, Chen, Sutskever - ["Evolution Strategies as a Scalable Alternative to Reinforcement Learning"](https://arxiv.org/abs/1703.03864)
>	"We explore the use of Evolution Strategies, a class of black box optimization algorithms, as an alternative to popular RL techniques such as Q-learning and Policy Gradients. Experiments on MuJoCo and Atari show that ES is a viable solution strategy that scales extremely well with the number of CPUs available: By using hundreds to thousands of parallel workers, ES can solve 3D humanoid walking in 10 minutes and obtain competitive results on most Atari games after one hour of training time. In addition, we highlight several advantages of ES as a black box optimization technique: it is invariant to action frequency and delayed rewards, tolerant of extremely long horizons, and does not need temporal discounting or value function approximation."

>	"In future work we plan to apply evolution strategies to those problems for which reinforcement learning is less well-suited: problems with long time horizons and complicated reward structure. We are particularly interested in meta-learning, or learning-to-learn. A proof of concept for meta-learning in an RL setting was given by Duan et al. (2016b): Using ES instead of RL we hope to be able to extend these results. Another application which we plan to examine is to combine ES with fast low precision neural network implementations to fully make use of its gradient-free nature."

>	"Large source of difficulty in RL stems from the lack of informative gradients of policy performance: such gradients may not exist due to non-smoothness of the environment or policy, or may only be available as high-variance estimates because the environment usually can only be accessed via sampling."

>	"In addition to being easy to parallelize, and to having an advantage in cases with long action sequences and delayed rewards, black box optimization algorithms like ES have other advantages over reinforcement learning techniques that calculate gradients."

>	"The communication overhead of implementing ES in a distributed setting is lower than for reinforcement learning methods such as policy gradients and Q-learning, as the only information that needs to be communicated across processes are the scalar return and the random seed that was used to generate the perturbations ε, rather than a full gradient. Also, it can deal with maximally sparse and delayed rewards; there is no need for the assumption that time information is part of the reward."

>	"By not requiring backpropagation, black box optimizers reduce the amount of computation per episode by about two thirds, and memory by potentially much more. In addition, not explicitly calculating an analytical gradient protects against problems with exploding gradients that are common when working with recurrent neural networks. By smoothing the cost function in parameter space, we reduce the pathological curvature that causes these problems: bounded cost functions that are smooth enough can’t have exploding gradients. At the extreme, ES allows us to incorporate non-differentiable elements into our architecture, such as modules that use hard attention."

>	"Black box optimization methods are uniquely suited to capitalize on advances in low precision hardware for deep learning. Low precision arithmetic, such as in binary neural networks, can be performed much cheaper than at high precision. When optimizing such low precision architectures, biased low precision gradient estimates can be a problem when using gradient-based methods. Similarly, specialized hardware for neural network inference, such as TPUs, can be used directly when performing optimization using ES, while their limited memory usually makes backpropagation impossible."

>	"By perturbing in parameter space instead of action space, black box optimizers are naturally invariant to the frequency at which our agent acts in the environment. For reinforcement learning, on the other hand, it is well known that frameskip is a crucial parameter to get right for the optimization to succeed. While this is usually a solvable problem for games that only require short-term planning and action, it is a problem for learning longer term strategic behavior. For these problems, RL needs hierarchy to succeed, which is not as necessary when using black box optimization."

>	"The resemblance of ES to finite differences suggests the method will scale poorly with the dimension of the parameters θ. Theoretical analysis indeed shows that for general non-smooth optimization problems, the required number of optimization steps scales linearly with the dimension (Nesterov & Spokoiny, 2011). However, it is important to note that this does not mean that larger neural networks will perform worse than smaller networks when optimized using ES: what matters is the difficulty, or intrinsic dimension, of the optimization problem. To see that the dimensionality of our model can be completely separate from the effective dimension of the optimization problem, consider a regression problem where we approximate a univariate variable y with a linear model yˆ = x · w: if we double the number of features and parameters in this model by concatenating x with itself (i.e. using features x0 = (x, x)), the problem does not become more difficult. In fact, the ES algorithm will do exactly the same thing when applied to this higher dimensional problem, as long as we divide the standard deviation of the noise by two, as well as the learning rate. In practice, we observe slightly better results when using larger networks with ES. We hypothesize that this is due to the same effect that makes standard gradient-based optimization of large neural networks easier than for small ones: large networks have fewer local minima."

----
>	"Learning bigger model is faster in case of ES while it is slower in case of policy gradient."

>	"ES is able to focus on intrinsic dimensionality of the problem by ignoring irrelevant dimensions of problem."

>	"Every trick that helps backpropagation, also helps evolution strategies: scale of random initialization, batch normalization, Residual Networks."

----
>	"Solving 3D Humanoid with ES on one 18-core machine takes about 11 hours, which is on par with RL. However, when distributed across 80 machines and 1,440 CPU cores, ES can solve 3D Humanoid in just 10 min- utes, reducing experiment turnaround time by two orders of magnitude. Figure 1 shows that, for this task, ES is able to achieve linear speedup in the number of CPU cores."

>	"All games were trained for 1 billion frames, which requires about the same amount of neural network computation as the published 1-day results for A3C which uses 320 million frames. The difference is due to the fact that ES does not perform backpropagation and does not use a value function. By parallelizing the evaluation of perturbed parameters across 720 CPUs on Amazon EC2, we can bring down the time required for the training process to about one hour per game. After training, we compared final performance against the published A3C results and found that ES performed better in 23 games tested, while it performed worse in 28."

----
>	"ES rivals the performance of standard RL techniques on moder benchmarks, while overcoming many of RL’s inconveniences. ES is simpler to implement (there is no need for backpropagation), it is easier to scale in a distributed setting, it does not suffer in settings with sparse rewards, and has fewer hyperparameters. This outcome is surprising because ES resembles simple hill-climbing in a high-dimensional space based only on finite differences along a few random directions at each step."

----
>	"Mathematically, you’ll notice that this is also equivalent to estimating the gradient of the expected reward in the parameter space using finite differences, except we only do it along 100 random directions. Yet another way to see it is that we’re still doing RL (Policy Gradients, or REINFORCE specifically), where the agent’s actions are to emit entire parameter vectors using a gaussian policy."

>	"Notice that the objective is identical to the one that RL optimizes: the expected reward. However, RL injects noise in the action space and uses backpropagation to compute the parameter updates, while ES injects noise directly in the parameter space. Another way to describe this is that RL is a “guess and check” on actions, while ES is a “guess and check” on parameters. Since we’re injecting noise in the parameters, it is possible to use deterministic policies (and we do, in our experiments). It is also possible to add noise in both actions and parameters to potentially combine the two approaches."

----
>	"Data efficiency comparison. ES is less efficient than TRPO, but no worse than about a factor of 10."

>	"Wall clock comparison. On Atari, ES trained on 720 cores in 1 hour achieves comparable performance to A3C trained on 32 cores in 1 day. We were able to solve one of the hardest MuJoCo tasks (a 3D humanoid) using 1,440 CPUs across 80 machines in only 10 minutes. As a comparison, in a typical setting 32 A3C workers on one machine would solve this task in about 10 hours. We found that naively scaling A3C in a standard cloud CPU setting is challenging due to high communication bandwidth requirements."

>	"It is also important to note that supervised learning problems (e.g. image classification, speech recognition, or most other tasks in the industry), where one can compute the exact gradient of the loss function with backpropagation, are not directly impacted by these findings. For example, in our preliminary experiments we found that using ES to estimate the gradient on the MNIST digit recognition task can be as much as 1,000 times slower than using backpropagation. It is only in RL settings, where one has to estimate the gradient of the expected reward by sampling, where ES becomes competitive."

----
>	"This gradient estimator may be slightly biased as well as high variance. The second order Taylor approximation is the part where bias may be introduced, if the real objective function has non-negligible (i.e. weird) third order gradients. The size of the bias will be in the order of σ² so as long as σ is small, the bias is probably negligible from a practical perspective. Therefore you can kind of say ES provides an approximately unbiased gradient estimate. So this is basically SGD - as SGD only requires an unbiased estimate of gradients. The unbiased estimate typically comes from minibatches, but no-one said it cannot come from a different Monte Carlo estimate. In this respect, the only difference between backprop-SGD and ES is the source of randomness in the gradient estimator. Consequently, Adam or RMS-prop or Nesterov might still make perfect sense on top of these gradients, for example."

  - <https://blog.openai.com/evolution-strategies/>
  - <https://www.technologyreview.com/s/603916/a-new-direction-for-artificial-intelligence/> (Sutskever)
  - <https://youtube.com/watch?v=Rd0UdJFYkqI> (Temirchev, in russian)
  - <http://inference.vc/evolutionary-strategies-embarrassingly-parallelizable-optimization/> (Huszar)
  - <http://inference.vc/evolution-strategies-variational-optimisation-and-natural-es-2/> (Huszar)
  - <http://davidbarber.github.io/blog/2017/04/03/variational-optimisation/> (Barber)
  - <http://argmin.net/2017/04/03/evolution/> (Recht and Frostig)
  - ["Random Gradient-Free Minimization of Convex Functions"](https://mipt.ru/dcam/students/elective/a_5gc1te/RandomGradFree.PDF) by Nesterov
  - ["Natural Evolution Strategies"](http://jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) by Wierstra, Schaul, Glasmachers, Sun, Peters, Schmidhuber
  - <https://github.com/openai/evolution-strategies-starter>
  - <https://github.com/atgambardella/pytorch-es>
  - <https://github.com/mdibaiee/flappy-es>
  - <https://gist.github.com/kashif/5748e199a3bec164a867c9b654e5ffe5>
  - <https://github.com/DanielTakeshi/rl_algorithms/blob/master/es/basic_es.py>


#### Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver, Kavukcuoglu - ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/abs/1602.01783)
>	"We propose a conceptually simple and lightweight framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. We present asynchronous variants of four standard reinforcement learning algorithms and show that parallel actor-learners have a stabilizing effect on training allowing all four methods to successfully train neural network controllers. The best performing method, an asynchronous variant of actor-critic, surpasses the current state-of-the-art on the Atari domain while training for half the time on a single multi-core CPU instead of a GPU. Furthermore, we show that asynchronous actor-critic succeeds on a wide variety of continuous motor control problems as well as on a new task of navigating random 3D mazes using a visual input."

>	"We have presented asynchronous versions of four standard reinforcement learning algorithms and showed that they are able to train neural network controllers on a variety of domains in a stable manner. Our results show that in our proposed framework stable training of neural networks through reinforcement learning is possible with both valuebased and policy-based methods, off-policy as well as onpolicy methods, and in discrete as well as continuous domains. When trained on the Atari domain using 16 CPU cores, the proposed asynchronous algorithms train faster than DQN trained on an Nvidia K40 GPU, with A3C surpassing the current state-of-the-art in half the training time. One of our main findings is that using parallel actorlearners to update a shared model had a stabilizing effect on the learning process of the three value-based methods we considered. While this shows that stable online Q-learning is possible without experience replay, which was used for this purpose in DQN, it does not mean that experience replay is not useful. Incorporating experience replay into the asynchronous reinforcement learning framework could substantially improve the data efficiency of these methods by reusing old data. This could in turn lead to much faster training times in domains like TORCS where interacting with the environment is more expensive than updating the model for the architecture we used."

  - <http://youtube.com/watch?v=0xo1Ldx3L5Q> (TORCS demo)
  - <http://youtube.com/watch?v=nMR5mjCFZCw> (3D Labyrinth demo)
  - <http://youtube.com/watch?v=Ajjc08-iPx8> (MuJoCo demo)
  - <http://youtube.com/watch?v=9sx1_u2qVhQ> (Mnih)
  - <http://techtalks.tv/talks/asynchronous-methods-for-deep-reinforcement-learning/62475/> (Mnih)
  - <http://pemami4911.github.io/paper-summaries/2016/08/02/A3C.html>
  - <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2FMnihBMGLHSK16>
  - <https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2>
  - <https://github.com/Zeta36/Asynchronous-Methods-for-Deep-Reinforcement-Learning>
  - <https://github.com/miyosuda/async_deep_reinforce>
  - <https://github.com/muupan/async-rl>
  - <https://github.com/yandexdataschool/AgentNet/blob/master/agentnet/learning/a2c_n_step.py>
  - <https://github.com/coreylynch/async-rl>
  - <https://github.com/carpedm20/deep-rl-tensorflow/blob/master/agents/async.py>
  - <https://github.com/ikostrikov/pytorch-a3c>
  - <https://github.com/danijar/mindpark/blob/master/mindpark/algorithm/a3c.py>


#### Schulman, Levine, Moritz, Jordan, Abbeel - ["Trust Region Policy Optimization"](http://arxiv.org/abs/1502.05477)
>	"In this article, we describe a method for optimizing control policies, with guaranteed monotonic improvement. By making several approximations to the theoretically-justified scheme, we develop a practical algorithm, called Trust Region Policy Optimization. This algorithm is effective for optimizing large nonlinear policies such as neural networks. Our experiments demonstrate its robust performance on a wide variety of tasks: learning simulated robotic swimming, hopping, and walking gaits; and playing Atari games using images of the screen as input. Despite its approximations that deviate from the theory, TRPO tends to give monotonic improvement, with little tuning of hyperparameters."

>	"We proposed and analyzed trust region methods for optimizing stochastic control policies. We proved monotonic improvement for an algorithm that repeatedly optimizes a local approximation to the expected cost of the policy with a KL divergence penalty, and we showed that an approximation to this method that incorporates a KL divergence constraint achieves good empirical results on a range of challenging policy learning tasks, outperforming prior methods. Our analysis also provides a perspective that unifies policy gradient and policy iteration methods, and shows them to be special limiting cases of an algorithm that optimizes a certain objective subject to a trust region constraint. In the domain of robotic locomotion, we successfully learned controllers for swimming, walking and hopping in a physics simulator, using general purpose neural networks and minimally informative costs. To our knowledge, no prior work has learned controllers from scratch for all of these tasks, using a generic policy search method and non-engineered, general-purpose policy representations. In the game-playing domain, we learned convolutional neural network policies that used raw images as inputs. This requires optimizing extremely high-dimensional policies, and only two prior methods report successful results on this task. Since the method we proposed is scalable and has strong theoretical foundations, we hope that it will serve as a jumping-off point for future work on training large, rich function approximators for a range of challenging problems. At the intersection of the two experimental domains we explored, there is the possibility of learning robotic control policies that use vision and raw sensory data as input, providing a unified scheme for training robotic controllers that perform both perception and control. The use of more sophisticated policies, including recurrent policies with hidden state, could further make it possible to roll state estimation and control into the same policy in the partially-observed setting. By combining our method with model learning, it would also be possible to substantially reduce its sample complexity, making it applicable to real-world settings where samples are expensive."

----
>	"Combines theoretical ideas from conservative policy gradient algorithm to prove that monotonic improvement can be guaranteed when one solves a series of subproblems of optimizing a bound on the policy performance. The conclusion is that one should use KL-divergence constraint."

>	"As you iteratively improve your policy, it’s important to constrain the KL divergence between the old and new policy to be less than some constant δ. This δ (in the unit of nats) is better than a fixed step size, since the meaning of the step size changes depending on what the rewards and problem structure look like at different points in training. This is called Trust Region Policy Optimization (or, in a first-order variant, Proximal Policy Optimization) and it matters more as we do more experience replay."

  - <https://youtube.com/watch?v=jeid0wIrSn4> + <https://vimeo.com/113957342> (demo)
  - <https://youtu.be/xe-z4i3l-iQ?t=30m35s> (Abbeel)
  - <http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/> (Abbeel, 0:27:10)
  - <https://youtube.com/watch?v=gb5Q2XL5c8A> (Schulman)
  - <http://kvfrans.com/what-is-the-natural-gradient-and-where-does-it-appear-in-trust-region-policy-optimization/>
  - <https://github.com/joschu/modular_rl>
  - <https://github.com/rll/deeprlhw2/blob/master/ppo.py>
  - <https://github.com/wojzaremba/trpo>
  - <https://github.com/rllab/rllab/blob/master/rllab/algos/trpo.py>
  - <https://github.com/kvfrans/parallel-trpo>


#### Schulman, Moritz, Levine, Jordan, Abbeel - ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](http://arxiv.org/abs/1506.02438)
>	"Policy gradient methods are an appealing approach in reinforcement learning because they directly optimize the cumulative reward and can straightforwardly be used with nonlinear function approximators such as neural networks. The two main challenges are the large number of samples typically required, and the difficulty of obtaining stable and steady improvement despite the nonstationarity of the incoming data. We address the first challenge by using value functions to substantially reduce the variance of policy gradient estimates at the cost of some bias, with an exponentially-weighted estimator of the advantage function that is analogous to TD(lambda). We address the second challenge by using trust region optimization procedure for both the policy and the value function, which are represented by neural networks. Our approach yields strong empirical results on highly challenging 3D locomotion tasks, learning running gaits for bipedal and quadrupedal simulated robots, and learning a policy for getting the biped to stand up from starting out lying on the ground. In contrast to a body of prior work that uses hand-crafted policy representations, our neural network policies map directly from raw kinematics to joint torques. Our algorithm is fully model-free, and the amount of simulated experience required for the learning tasks on 3D bipeds corresponds to 1-2 weeks of real time."

>	"Policy gradient methods provide a way to reduce reinforcement learning to stochastic gradient descent, by providing unbiased gradient estimates. However, so far their success at solving difficult control problems has been limited, largely due to their high sample complexity. We have argued that the key to variance reduction is to obtain good estimates of the advantage function. We have provided an intuitive but informal analysis of the problem of advantage function estimation, and justified the generalized advantage estimator, which has two parameters which adjust the bias-variance tradeoff. We described how to combine this idea with trust region policy optimization and a trust region algorithm that optimizes a value function, both represented by neural networks. Combining these techniques, we are able to learn to solve difficult control tasks that have previously been out of reach for generic reinforcement learning methods. One question that merits future investigation is the relationship between value function estimation error and policy gradient estimation error. If this relationship were known, we could choose an error metric for value function fitting that is well-matched to the quantity of interest, which is typically the accuracy of the policy gradient estimation. Some candidates for such an error metric might include the Bellman error or projected Bellman error, as described in Bhatnagar et al. (2009). Another enticing possibility is to use a shared function approximation architecture for the policy and the value function, while optimizing the policy using generalized advantage estimation. While formulating this problem in a way that is suitable for numerical optimization and provides convergence guarantees remains an open question, such an approach could allow the value function and policy representations to share useful features of the input, resulting in even faster learning. In concurrent work, researchers have been developing policy gradient methods that involve differentiation with respect to the continuous-valued action (Lillicrap et al., 2015; Heess et al., 2015). While we found empirically that the one-step return (lambda = 0) leads to excessive bias and poor performance, these papers show that such methods can work when tuned appropriately. However, note that those papers consider control problems with substantially lower-dimensional state and action spaces than the ones considered here. A comparison between both classes of approach would be useful for future work."

  - <https://youtu.be/gb5Q2XL5c8A?t=21m2s> + <https://youtube.com/watch?v=ATvp0Hp7RUI> + <https://youtube.com/watch?v=Pvw28wPEWEo> (demo)
  - <https://youtu.be/xe-z4i3l-iQ?t=30m35s> (Abbeel)
  - <https://youtu.be/rO7Dx8pSJQw?t=40m20s> (Schulman)
  - <https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/>
  - <https://github.com/joschu/modular_rl>
  - <https://github.com/rll/deeprlhw2/blob/master/ppo.py>


#### Silver, Lever, Heess, Degris, Wierstra, Riedmiller - ["Deterministic Policy Gradient Algorithms"](http://jmlr.org/proceedings/papers/v32/silver14.html)
>	"In this paper we consider deterministic policy gradient algorithms for reinforcement learning with continuous actions. The deterministic policy gradient has a particularly appealing form: it is the expected gradient of the action-value function. This simple form means that the deterministic policy gradient can be estimated much more efficiently than the usual stochastic policy gradient. To ensure adequate exploration, we introduce an off-policy actor-critic algorithm that learns a deterministic target policy from an exploratory behaviour policy. We demonstrate that deterministic policy gradient algorithms can significantly outperform their stochastic counter-parts in high-dimensional action spaces."

>	"Policy gradient algorithms are widely used in reinforcement learning problems with continuous action spaces. The basic idea is to represent the policy by a parametric probability distribution πθ(a|s) = P [a|s; θ] that stochastically selects action a in state s according to parameter vector θ. Policy gradient algorithms typically proceed by sampling this stochastic policy and adjusting the policy parameters in the direction of greater cumulative reward. In this paper we instead consider deterministic policies a=μθ(s). It is natural to wonder whether the same approach can be followed as for stochastic policies: adjusting the policy parameters in the direction of the policy gradient. It was previously believed that the deterministic policy gradient did not exist, or could only be obtained when using a model. However, we show that the deterministic policy gradient does indeed exist, and furthermore it has a simple model-free form that simply follows the gradient of the action-value function. In addition, we show that the deterministic policy gradient is the limiting case, as policy variance tends to zero, of the stochastic policy gradient."

>	"From a practical viewpoint, there is a crucial difference between the stochastic and deterministic policy gradients. In the stochastic case, the policy gradient integrates over both state and action spaces, whereas in the deterministic case it only integrates over the state space. As a result, computing the stochastic policy gradient may require more samples, especially if the action space has many dimensions. In order to explore the full state and action space, a stochastic policy is often necessary. To ensure that our deterministic policy gradient algorithms continue to explore satisfactorily, we introduce an off-policy learning algorithm. The basic idea is to choose actions according to a stochastic behaviour policy (to ensure adequate exploration), but to learn about a deterministic target policy (exploiting the efficiency of the deterministic policy gradient). We use the deterministic policy gradient to derive an off-policy actor-critic algorithm that estimates the action-value function using a differentiable function approximator, and then updates the policy parameters in the direction of the approximate action-value gradient. We also introduce a notion of compatible function approximation for deterministic policy gradients, to ensure that the approximation does not bias the policy gradient."

>	"We apply our deterministic actor-critic algorithms to several benchmark problems: a high-dimensional bandit; several standard benchmark reinforcement learning tasks with low dimensional action spaces; and a high-dimensional task for controlling an octopus arm. Our results demonstrate a significant performance advantage to using deterministic policy gradients over stochastic policy gradients, particularly in high dimensional tasks. In practice, the deterministic actor-critic significantly outperformed its stochastic counterpart by several orders of magnitude in a bandit with 50 continuous action dimensions, and solved a challenging reinforcement learning problem with 20 continuous action dimensions and 50 state dimensions. Furthermore, our algorithms require no more computation than prior methods: the computational cost of each update is linear in the action dimensionality and the number of policy parameters."

----
>	"DPG provides a continuous analogue to DQN, exploiting the differentiability of the Q-network to solve a wide variety of continuous control tasks."

  - <http://videolectures.net/rldm2015_silver_reinforcement_learning/> (Silver, 1:07:23)
  - <http://youtube.com/watch?v=qLaDWKd61Ig&t=38m58s> (Silver)
  - <http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/> (Abbeel, 1:02:04)
  - <https://youtu.be/mrgJ53TIcQc?t=1h3m2s> (Seleznev, in russian)
  - <https://youtu.be/rO7Dx8pSJQw?t=50m> (Schulman)
  - <https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html>
  - <http://pemami4911.github.io/blog_posts/2016/08/21/ddpg-rl.html>
  - <https://github.com/matthiasplappert/keras-rl/blob/master/rl/agents/ddpg.py>
  - <https://github.com/rllab/rllab/blob/master/rllab/algos/ddpg.py>
  - <https://github.com/yandexdataschool/AgentNet/blob/master/agentnet/learning/dpg_n_step.py>
  - <https://github.com/MOCR/DDPG>


#### Lillicrap, Hunt, Pritzel, Heess, Erez, Tassa, Silver, Wierstra - ["Continuous Control with Deep Reinforcement Learning"](http://arxiv.org/abs/1509.02971)
>	"We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. We further demonstrate that for many of the tasks the algorithm can learn policies “end-to-end”: directly from raw pixel inputs."

>	"The work presented here combines insights from recent advances in deep learning and reinforcement learning, resulting in an algorithm that robustly solves challenging problems across a variety of domains with continuous action spaces, even when using raw pixels for observations. As with most reinforcement learning algorithms, the use of non-linear function approximators nullifies any convergence guarantees; however, our experimental results demonstrate that stable learning without the need for any modifications between environments. Interestingly, all of our experiments used substantially fewer steps of experience than was used by DQN learning to find solutions in the Atari domain. Nearly all of the problems we looked at were solved within 2.5 million steps of experience (and usually far fewer), a factor of 20 fewer steps than DQN requires for good Atari solutions. This suggests that, given more simulation time, DDPG may solve even more difficult problems than those considered here. A few limitations to our approach remain. Most notably, as with most model-free reinforcement approaches, DDPG requires a large number training episodes to find solutions. However, we believe that a robust model-free approach may be an important component of larger systems which may attack these limitations."

>	"While DQN solves problems with high-dimensional observation spaces, it can only handle discrete and low-dimensional action spaces. Many tasks of interest, most notably physical control tasks, have continuous (real valued) and high dimensional action spaces. DQN cannot be straightforwardly applied to continuous domains since it relies on a finding the action that maximises the action-value function, which in the continuous valued case requires an iterative optimization process at every step."

>	"In this work we present a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces. Our work is based on the deterministic policy gradient algorithm. However, as we show below, a naive application of this actor-critic method with neural function approximators is unstable for challenging problems. Here we combine the actor-critic approach with insights from the recent success of Deep Q Network. Prior to DQN, it was generally believed that learning value functions using large, non-linear function approximators was difficult and unstable. DQN is able to learn value functions using such function approximators in a stable and robust way due to two innovations: 1. the network is trained off-policy with samples from a replay buffer to minimize correlations between samples; 2. the network is trained with a target Q network to give consistent targets during temporal difference backups. In this work we make use of the same ideas, along with batch normalization, a recent advance in deep learning."

>	"A key feature of the approach is its simplicity: it requires only a straightforward actor-critic architecture and learning algorithm with very few “moving parts”, making it easy to implement and scale to more difficult problems and larger networks. For the physical control problems we compare our results to a baseline computed by a planner that has full access to the underlying simulated dynamics and its derivatives. Interestingly, DDPG can sometimes find policies that exceed the performance of the planner, in some cases even when learning from pixels (the planner always plans over the true, low-dimensional state space)."

>	"Surprisingly, in some simpler tasks, learning policies from pixels is just as fast as learning using the low-dimensional state descriptor. This may be due to the action repeats making the problem simpler. It may also be that the convolutional layers provide an easily separable representation of state space, which is straightforward for the higher layers to learn on quickly."

  - <http://youtube.com/watch?v=tJBIqkC1wWM> (demo)
  - <http://youtube.com/watch?v=Tb5gASEJIRM> (demo) + <https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html>
  - <http://videolectures.net/rldm2015_silver_reinforcement_learning/> (Silver, 1:07:23)
  - <http://youtu.be/qLaDWKd61Ig?t=39m> (Silver)
  - <http://youtu.be/KHZVXao4qXs?t=52m58s> (Silver)
  - <http://youtu.be/M6nfipCxQBc?t=7m45s> (Lillicrap)
  - <http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/> (Abbeel, 1:02:04)
  - <https://youtu.be/mrgJ53TIcQc?t=1h3m2s> (Seleznev, in russian)
  - <https://youtu.be/rO7Dx8pSJQw?t=50m> (Schulman)
  - <https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html>
  - <http://pemami4911.github.io/blog_posts/2016/08/21/ddpg-rl.html>
  - <https://github.com/matthiasplappert/keras-rl/blob/master/rl/agents/ddpg.py>
  - <https://github.com/rllab/rllab/blob/master/rllab/algos/ddpg.py>
  - <https://github.com/yandexdataschool/AgentNet/blob/master/agentnet/learning/dpg_n_step.py>
  - <https://github.com/MOCR/DDPG>


#### Heess, Wayne, Silver, Lillicrap, Tassa, Erez - ["Learning Continuous Control Policies by Stochastic Value Gradients"](http://arxiv.org/abs/1510.09142)
>	"We present a unified framework for learning continuous control policies using backpropagation. It supports stochastic control by treating stochasticity in the Bellman equation as a deterministic function of exogenous noise. The product is a spectrum of general policy gradient algorithms that range from model-free methods with value functions to model-based methods without value functions. We use learned models but only require observations from the environment instead of observations from model-predicted trajectories, minimizing the impact of compounded model errors. We apply these algorithms first to a toy stochastic control problem and then to several physics-based control problems in simulation. One of these variants, SVG(1), shows the effectiveness of learning models, value functions, and policies simultaneously in continuous domains."

>	"We have shown that two potential problems with value gradient methods, their reliance on planning and restriction to deterministic models, can be exorcised, broadening their relevance to reinforcement learning. We have shown experimentally that the SVG framework can train neural network policies in a robust manner to solve interesting continuous control problems. Furthermore, we did not harness sophisticated generative models of stochastic dynamics, but one could readily do so, presenting great room for growth."

>	"Policy gradient algorithms maximize the expectation of cumulative reward by following the gradient of this expectation with respect to the policy parameters. Most existing algorithms estimate this gradient in a model-free manner by sampling returns from the real environment and rely on a likelihood ratio estimator. Such estimates tend to have high variance and require large numbers of samples or, conversely, low-dimensional policy parameterizations. A second approach to estimate a policy gradient relies on backpropagation instead of likelihood ratio methods. If a differentiable environment model is available, one can link together the policy, model, and reward function to compute an analytic policy gradient by backpropagation of reward along a trajectory. Instead of using entire trajectories, one can estimate future rewards using a learned value function (a critic) and compute policy gradients from subsequences of trajectories. It is also possible to backpropagate analytic action derivatives from a Q-function to compute the policy gradient without a model. Following Fairbank, we refer to methods that compute the policy gradient through backpropagation as value gradient methods. In this paper, we address two limitations of prior value gradient algorithms. The first is that, in contrast to likelihood ratio methods, value gradient algorithms are only suitable for training deterministic policies. Stochastic policies have several advantages: for example, they can be beneficial for partially observed problems; they permit on-policy exploration; and because stochastic policies can assign probability mass to off-policy trajectories, we can train a stochastic policy on samples from an experience database in a principled manner. When an environment model is used, value gradient algorithms have also been critically limited to operation in deterministic environments. By exploiting a mathematical tool known as “re-parameterization” that has found recent use for generative models, we extend the scope of value gradient algorithms to include the optimization of stochastic policies in stochastic environments. We thus describe our framework as Stochastic Value Gradient methods. Secondly, we show that an environment dynamics model, value function, and policy can be learned jointly with neural networks based only on environment interaction. Learned dynamics models are often inaccurate, which we mitigate by computing value gradients along real system trajectories instead of planned ones, a feature shared by model-free methods. This substantially reduces the impact of model error because we only use models to compute policy gradients, not for prediction, combining advantages of model-based and model-free methods with fewer of their drawbacks. We present several algorithms that range from model-based to model-free methods, flexibly combining models of environment dynamics with value functions to optimize policies in stochastic or deterministic environments. Experimentally, we demonstrate that SVG methods can be applied using generic neural networks with tens of thousands of parameters while making minimal assumptions about plans or environments. By examining a simple stochastic control problem, we show that SVG algorithms can optimize policies where model-based planning and likelihood ratio methods cannot. We provide evidence that value function approximation can compensate for degraded models, demonstrating the increased robustness of SVG methods over model-based planning. Finally, we use SVG algorithms to solve a variety of challenging, under-actuated, physical control problems, including swimming of snakes, reaching, tracking, and grabbing with a robot arm, fall-recovery for a monoped, and locomotion for a planar cheetah and biped."

----
>	"In policy-based and actor-critic methods, stochastic policy is usually defined as a fixed distribution over action domain with parameters whose values are adapted when training. SVG suggests a synthesis of model-based with model-free approaches that allows optimizing the distribution as a function by means of the standard gradient descent."

>	"Stochastic value gradients generalize DPG to stochastic policies in a number of ways, giving a spectrum from model-based to model-free algorithms. While SVG(0) is a direct stochastic generalization of DPG, SVG(1) combines an actor, critic and model f. The actor is trained through a combination of gradients from the critic, model and reward simultaneously."

  - <https://youtu.be/PYdL7bcn_cM> (demo)
  - <http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/> (Abbeel, 1:02:04)
  - <https://youtu.be/mrgJ53TIcQc?t=1h10m31s> (Seleznev, in russian)
  - <https://youtu.be/rO7Dx8pSJQw?t=50m> (Schulman)


#### Schulman, Heess, Weber, Abbeel - ["Gradient Estimation Using Stochastic Computation Graphs"](http://arxiv.org/abs/1506.05254)
>	"In a variety of problems originating in supervised, unsupervised, and reinforcement learning, the loss function is defined by an expectation over a collection of random variables, which might be part of a probabilistic model or the external world. Estimating the gradient of this loss function, using samples, lies at the core of gradient-based learning algorithms for these problems. We introduce the formalism of stochastic computation graphs---directed acyclic graphs that include both deterministic functions and conditional probability distributions---and describe how to easily and automatically derive an unbiased estimator of the loss function's gradient. The resulting algorithm for computing the gradient estimator is a simple modification of the standard backpropagation algorithm. The generic scheme we propose unifies estimators derived in variety of prior work, along with variance-reduction techniques therein. It could assist researchers in developing intricate models involving a combination of stochastic and deterministic operations, enabling, for example, attention, memory, and control actions."

>	"We have developed a framework for describing a computation with stochastic and deterministic operations, called a stochastic computation graph. Given a stochastic computation graph, we can automatically obtain a gradient estimator, given that the graph satisfies the appropriate conditions on differentiability of the functions at its nodes. The gradient can be computed efficiently in a backwards traversal through the graph: one approach is to apply the standard backpropagation algorithm to one of the surrogate loss functions; another approach (which is roughly equivalent) is to apply a modified backpropagation procedure. The results we have presented are sufficiently general to automatically reproduce a variety of gradient estimators that have been derived in prior work in reinforcement learning and probabilistic modeling. We hope that this work will facilitate further development of interesting and expressive models."

>	"Can mix and match likelihood ratio and path derivative. If black-box node: might need to place stochastic node in front of it and use likelihood ratio. This includes recurrent neural net policies."

  - <http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/> (Abbeel, 1:02:04)
  - <http://joschu.net/docs/thesis.pdf>


#### Weber, Heess, Eslami, Schulman, Wingate, Silver - ["Reinforced Variational Inference"](http://approximateinference.org/accepted/WeberEtAl2015.pdf)
>	"Recent years have seen an increase in the complexity and scale of probabilistic models used to understand and analyze data, with a corresponding increase in the difficulty of performing inference. An important enabling factor in this context has been the development of stochastic gradient algorithms for learning variational approximations to posterior distributions. In a separate line of work researchers have been investigating how to use probabilistic inference for the problem of optimal control. By viewing control as an inference problem, they showed that they could ‘borrow’ algorithms from the inference literature (e.g. belief propagation) and turn them into control algorithms. In this work, we do just the opposite: we formally map the problem of learning approximate posterior distributions in variational inference onto the policy optimization problem in reinforcement learning, explaining this connection at two levels. We first provide a high level connection, where draws from the approximate posterior correspond to trajectory samples, free energies to expected returns, and where the core computation involves computing gradients of expectations. We follow by a more detailed, sequential mapping where Markov Decision Processes concepts (state, action, rewards and transitions) are clearly defined in the inference context. We then illustrate how this allows us to leverage ideas from RL for inference network learning, for instance by introducing the concept of value functions in sequential variational inference. For concreteness and simplicity, in the main text we focus on inference for a particular model class and derive the general case in the appendix."



---
### interesting papers - behavioral cloning

[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reinforcement-learning---imitation) on imitation learning  
[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reinforcement-learning---transfer) on transfer  


#### Ross, Gordon, Bagnell - ["A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"](https://arxiv.org/abs/1011.0686)
>	"Sequential prediction problems such as imitation learning, where future observations depend on previous predictions (actions), violate the common i.i.d. assumptions made in statistical learning. This leads to poor performance in theory and often in practice. Some recent approaches (Daumé III et al., 2009; Ross and Bagnell, 2010) provide stronger guarantees in this setting, but remain somewhat unsatisfactory as they train either non-stationary or stochastic policies and require a large number of iterations. In this paper, we propose a new iterative algorithm, which trains a stationary deterministic policy, that can be seen as a no regret algorithm in an online learning setting. We show that any such no regret algorithm, combined with additional reduction assumptions, must find a policy with good performance under the distribution of observations it induces in such sequential settings. We demonstrate that this new approach outperforms previous approaches on two challenging imitation learning problems and a benchmark sequence labeling problem."

>	"We show that by batching over iterations of interaction with a system, no-regret methods, including the presented DAGGER approach can provide a learning reduction with strong performance guarantees in both imitation learning and structured prediction. In future work, we will consider more sophisticated strategies than simple greedy forward decoding for structured prediction, as well as using base classifiers that rely on Inverse Optimal Control (Abbeel and Ng, 2004; Ratliff et al., 2006) techniques to learn a cost function for a planner to aid prediction in imitation learning. Further we believe techniques similar to those presented, by leveraging a cost-to-go estimate, may provide an understanding of the success of online methods for reinforcement learning and suggest a similar data-aggregation method that can guarantee performance in such settings."

  - <https://youtube.com/watch?v=kl_G95uKTHw&t=1h5m38s> (Levine)


#### Levine, Koltun - ["Guided Policy Search"](http://vladlen.info/papers/guided-policy-search.pdf)
>	"Direct policy search can effectively scale to high-dimensional systems, but complex policies with hundreds of parameters often present a challenge for such methods, requiring numerous samples and often falling into poor local optima. We present a guided policy search algorithm that uses trajectory optimization to direct policy learning and avoid poor local optima. We show how differential dynamic programming can be used to generate suitable guiding samples, and describe a regularized importance sampled policy optimization that incorporates these samples into the policy search. We evaluate the method by learning neural network controllers for planar swimming, hopping, and walking, as well as simulated 3D humanoid running."

>	"In this paper, we show how trajectory optimization can guide the policy search away from poor local optima. Our guided policy search algorithm uses differential dynamic programming to generate “guiding samples”, which assist the policy search by exploring high-reward regions. An importance sampled variant of the likelihood ratio estimator is used to incorporate these guiding samples directly into the policy search. We show that DDP can be modified to sample from a distribution over high reward trajectories, making it particularly suitable for guiding policy search. Furthermore, by initializing DDP with example demonstrations, our method can perform learning from demonstration. The use of importance sampled policy search also allows us to optimize the policy with second order quasi-Newton methods for many gradient steps without requiring new on-policy samples, which can be crucial for complex, nonlinear policies. Our main contribution is a guided policy search algorithm that uses trajectory optimization to assist policy learning. We show how to obtain suitable guiding samples, and we present a regularized importance sampled policy optimization method that can utilize guiding samples and does not require a learning rate or new samples at every gradient step. We evaluate our method on planar swimming, hopping, and walking, as well as 3D humanoid running, using general-purpose neural network policies. We also show that both the proposed sampling scheme and regularizer are essential for good performance, and that the learned policies can generalize successfully to new environments."

>	"Standard likelihood ratio methods require new samples from the current policy at each gradient step, do not admit off-policy samples, and require the learning rate to be chosen carefully to ensure convergence. We discuss how importance sampling can be used to lift these constraints."

>	"Prior methods employed importance sampling to reuse samples from previous policies. However, when learning policies with hundreds of parameters, local optima make it very difficult to find a good solution. In this section, we show how differential dynamic programming can be used to supplement the sample set with off-policy guiding samples that guide the policy search to regions of high reward."

>	"We incorporate guiding samples into the policy search by building one or more initial DDP solutions and supplying the resulting samples to the importance sampled policy search algorithm. These solutions can be initialized with human demonstrations or with an offline planning algorithm. When learning from demonstrations, we can perform just one step of DDP starting from the example demonstration, thus constructing a Gaussian distribution around the example. If adaptive guiding distributions are used, they are constructed at each iteration of the policy search starting from the previous DDP solution. Although our policy search component is model-free, DDP requires a model of the system dynamics. Numerous recent methods have proposed to learn the model, and if we use initial examples, only local models are required. One might also wonder why the DDP policy is not itself a suitable controller. The issue is that this policy is time-varying and only valid around a single trajectory, while the final policy can be learned from many DDP solutions in many situations. Guided policy search can be viewed as transforming a collection of trajectories into a controller. This controller can adhere to any parameterization, reflecting constraints on computation or available sensors in partially observed domains. In our evaluation, we show that such a policy generalizes to situations where the DDP policy fail."

>	"Policy gradient methods often require on-policy samples at each gradient step, do not admit off-policy samples, and cannot use line searches or higher order optimization methods such as LBFGS, which makes them difficult to use with complex policy classes. Our approach follows prior methods that use importance sampling to address these challenges. While these methods recycle samples from previous policies, we also introduce guiding samples, which dramatically speed up learning and help avoid poor local optima. We also regularize the importance sampling estimator, which prevents the optimization from assigning low probabilities to all samples. The regularizer controls how far the policy deviates from the samples, serving a similar function to the natural gradient, which bounds the information loss at each iteration. Unlike Tang and Abbeel’s ESS constraint, our regularizer does not penalize reliance on a few samples, but does avoid policies that assign a low probability to all samples. Our evaluation shows that the regularizer can be crucial for learning effective policies."

>	"We presented a guided policy search algorithm that can learn complex policies with hundreds of parameters by incorporating guiding samples into the policy search. These samples are drawn from a distribution built around a DDP solution, which can be initialized from demonstrations. We evaluated our method using general-purpose neural networks on a range of challenging locomotion tasks, and showed that the learned policies generalize to new environments. While our policy search is model-free, it is guided by a model-based DDP algorithm. A promising avenue for future work is to build the guiding distributions with model-free methods that either build trajectory following policies or perform stochastic trajectory optimization. Our rough terrain results suggest that GPS can generalize by learning basic locomotion principles such as balance. Further investigation of generalization is an exciting avenue for future work. Generalization could be improved by training on multiple environments, or by using larger neural networks with multiple layers or recurrent connections. It would be interesting to see whether such extensions could learn more general and portable concepts, such as obstacle avoidance, perturbation recoveries, or even higher-level navigation skills."

----
>	"DAgger vs GPS:  
>	- DAgger does not require an adaptive expert  
>	  * Any expert will do, so long as states from learned policy can be labeled  
>	  * Assumes it is possible to match expert's behavior up to bounded loss (not always possible, e.g. in partially observed domains)  
>	- GPS adapts the expert behavior to learning agent  
>	  * Does not require bounded loss on initial expert (expert will change)"  

----
>	"Use (modification of) importance sampling to get policy gradient, where samples are obtained via trajectory optimization."

  - <https://graphics.stanford.edu/projects/gpspaper/index.htm>
  - <http://youtube.com/watch?v=o0Ebur3aNMo> (Levine)
  - <http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/> (Abbeel, part 2)
  - <http://youtube.com/watch?v=EtMyH_--vnU> (Levine)
  - <https://video.seas.harvard.edu/media/ME+Sergey+Levine+2015+-04-01/1_gqqp9r3o/23375211> (Levine)
  - <http://youtube.com/watch?v=xMHjkZBvnfU> (Abbeel)
  - <http://rll.berkeley.edu/gps/> + <http://rll.berkeley.edu/gps/faq.html>
  - <https://github.com/nivwusquorum/guided-policy-search/>


#### Levine, Abbeel - ["Learning Neural Network Policies with Guided Policy Search under Unknown Dynamics"](http://rll.berkeley.edu/nips2014gps/mfcgps.pdf)
>	"We present a policy search method that uses iteratively refitted local linear models to optimize trajectory distributions for large, continuous problems. These trajectory distributions can be used within the framework of guided policy search to learn policies with an arbitrary parameterization. Our method fits time-varying linear dynamics models to speed up learning, but does not rely on learning a global model, which can be difficult when the dynamics are complex and discontinuous. We show that this hybrid approach requires many fewer samples than model-free methods, and can handle complex, nonsmooth dynamics that can pose a challenge for model-based techniques. We present experiments showing that our method can be used to learn complex neural network policies that successfully execute simulated robotic manipulation tasks in partially observed environments with numerous contact discontinuities and underactuation."

  - <http://rll.berkeley.edu/nips2014gps/> (demos)
  - <http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/> (Abbeel, part 2)
  - <https://github.com/nivwusquorum/guided-policy-search/>


#### Levine, Wagener, Abbeel - ["Learning Contact-Rich Manipulation Skills with Guided Policy Search"](http://arxiv.org/abs/1501.05611)
>	"Autonomous learning of object manipulation skills can enable robots to acquire rich behavioral repertoires that scale to the variety of objects found in the real world. However, current motion skill learning methods typically restrict the behavior to a compact, low-dimensional representation, limiting its expressiveness and generality. In this paper, we extend a recently developed policy search method and use it to learn a range of dynamic manipulation behaviors with highly general policy representations, without using known models or example demonstrations. Our approach learns a set of trajectories for the desired motion skill by using iteratively refitted time-varying linear models, and then unifies these trajectories into a single control policy that can generalize to new situations. To enable this method to run on a real robot, we introduce several improvements that reduce the sample count and automate parameter selection. We show that our method can acquire fast, fluent behaviors after only minutes of interaction time, and can learn robust controllers for complex tasks, including stacking large lego blocks, putting together a plastic toy, placing wooden rings onto tight-fitting pegs, and screwing bottle caps onto bottles."

>	"The central idea behind guided policy search is to decompose the policy search problem into alternating trajectory optimization and supervised learning phases, where trajectory optimization is used to find a solution to the control problem and produce training data that is then used in the supervised learning phase to train a nonlinear, high-dimensional policy. By training a single policy from multiple trajectories, guided policy search can produce complex policies that generalize effectively to a range of initial states."

  - <http://rll.berkeley.edu/icra2015gps/>
  - <http://youtube.com/watch?t=35&v=JeVppkoloXs> + <http://youtube.com/watch?v=oQasCj1X0e8> (demo)
  - <http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/> (Abbeel, part 2)
  - <http://youtube.com/watch?v=EtMyH_--vnU> (Levine)
  - <https://video.seas.harvard.edu/media/ME+Sergey+Levine+2015+-04-01/1_gqqp9r3o/23375211> (Levine)
  - <http://youtube.com/watch?v=xMHjkZBvnfU> (Abbeel)


#### Levine, Finn, Darrell, Abbeel - ["End-to-End Training of Deep Visuomotor Policies"](http://arxiv.org/abs/1504.00702)
>	"Policy search methods based on reinforcement learning and optimal control can allow robots to automatically learn a wide range of tasks. However, practical applications of policy search tend to require the policy to be supported by hand-engineered components for perception, state estimation, and low-level control. We propose a method for learning policies that map raw, low-level observations, consisting of joint angles and camera images, directly to the torques at the robot's joints. The policies are represented as deep convolutional neural networks with 92,000 parameters. The high dimensionality of such policies poses a tremendous challenge for policy search. To address this challenge, we develop a sensorimotor guided policy search method that can handle high-dimensional policies and partially observed tasks. We use BADMM to decompose policy search into an optimal control phase and supervised learning phase, allowing CNN policies to be trained with standard supervised learning techniques. This method can learn a number of manipulation tasks that require close coordination between vision and control, including inserting a block into a shape sorting cube, screwing on a bottle cap, fitting the claw of a toy hammer under a nail with various grasps, and placing a coat hanger on a clothes rack."

  - <https://sites.google.com/site/visuomotorpolicy/home>
  - <http://youtube.com/watch?v=EtMyH_--vnU> (Levine)
  - <https://video.seas.harvard.edu/media/ME+Sergey+Levine+2015+-04-01/1_gqqp9r3o/23375211> (Levine)
  - <http://youtube.com/watch?v=xMHjkZBvnfU> (Abbeel)
  - <http://rll.berkeley.edu/gps/> (code) + <http://rll.berkeley.edu/gps/faq.html>


#### Zhang, Levine, McCarthy, Finn, Abbeel - ["Learning Deep Neural Network Policies with Continuous Memory States"](http://arxiv.org/abs/1507.01273)
>	"Policy learning for partially observed control tasks requires policies that can remember salient information from past observations. In this paper, we present a method for learning policies with internal memory for high-dimensional, continuous systems, such as robotic manipulators. Our approach consists of augmenting the state and action space of the system with continuous-valued memory states that the policy can read from and write to. Learning general-purpose policies with this type of memory representation directly is difficult, because the policy must automatically figure out the most salient information to memorize at each time step. We show that, by decomposing this policy search problem into a trajectory optimization phase and a supervised learning phase through a method called guided policy search, we can acquire policies with effective memorization and recall strategies. Intuitively, the trajectory optimization phase chooses the values of the memory states that will make it easier for the policy to produce the right action in future states, while the supervised learning phase encourages the policy to use memorization actions to produce those memory states. We evaluate our method on tasks involving continuous control in manipulation and navigation settings, and show that our method can learn complex policies that successfully complete a range of tasks that require memory."

>	"Our experimental results show that our method can be used to learn a variety of tasks involving continuous control in manipulation and navigation settings. In direct comparisons, we find that our approach outperforms a method where the neural network in guided policy search is na¨ıvely replaced with a recurrent network using backpropagation through time, as well as a purely feedforward policy with no memory."

>	"While specialized RNN representations such as LSTMs or GRUs can mitigate these issues, our experiments show that our guided policy search algorithm with memory states can produce more effective policies than backpropagation through time with LSTMs."

>	"Previous work has only applied guided policy search to training reactive feedforward policies, since the algorithm assumes that the policy is Markovian. We modify the BADMM-based guided policy search method to handle continuous memory states. The memory states are added to the state of the system, and the policy is tasked both with choosing the action and modifying the memory states. Although the resulting policy can be viewed as an RNN, we do not need to perform backpropagation through time to train the recurrent connections inside the policy. Instead, the memory states are optimized by the trajectory optimization algorithm, which intuitively seeks to set the memory states to values that will allow the policy to take the appropriate action at each time step, and the policy then attempts to mimic this behavior in the supervised learning phase."

>	"We presented a method for training policies for continuous control tasks that require memory. Our method consists of augmenting the state space with memory states, which the policy can choose to read and write as necessary. The resulting augmented control problem is solved using guided policy search, which uses simple trajectory-centric reinforcement learning algorithms to optimize trajectories from several initial states, and then uses these trajectories to generate a training set that can be used to optimize the policy with supervised learning. In the augmented state space, the policy is purely reactive, which means that policy training does not require backpropagating the gradient through time. However, when viewed together with the memory states, the policy is endowed with memory, and can be regarded as a recurrent neural network. Our experimental results show that our method can be used to learn policies for a variety of simulated robotic tasks that require maintaining internal memory to succeed. Part of the motivation for our approach came from the observation that even fully feed-forward neural network policies could often complete tricky tasks that seemed to require memory by using the physical state of the robot to “store” information, similarly to how a person might “remember” a number while counting by using their fingers. In our approach, we exploit this capability of reactive feedforward policies by providing extra state variables that do not have a physical analog, and exist only for the sake of memory."

>	"One interesting direction for follow-up work is to apply our approach for training recurrent networks for general supervised learning tasks, rather than just robotic control. In this case, the memory state comprises the entire state of the system, and the cost function is simply the supervised learning loss. Since the hidden memory state activations are optimized separately from the network weights, such an approach could in principle be more effective at training networks that perform complex reasoning over temporally extended intervals. Furthermore, since our method trains stochastic policies, it would also be able to train stochastic recurrent neural networks, where the transition dynamics are non-deterministic. These types of networks are typically quite challenging to train, and exploring this further is an exciting direction for future work."

  - <http://rll.berkeley.edu/gpsrnn/> (demo)
  - <http://thespermwhale.com/jaseweston/ram/slides/session4/ram_talk_zhang_marvin.pdf>



---
### interesting papers - inverse reinforcement learning

[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reinforcement-learning---imitation)


#### Wulfmeier, Ondruska, Posner - ["Maximum Entropy Deep Inverse Reinforcement Learning"](http://arxiv.org/abs/1507.04888)
>	"This paper presents a general framework for employing deep architectures - in particular neural networks - to solve the inverse reinforcement learning (IRL) problem. Specifically, we propose to exploit the representational capacity and favourable computational complexity of deep networks to approximate complex, nonlinear reward functions. We show that the Maximum Entropy paradigm for IRL lends itself naturally to the efficient training of deep architectures. At test time, the approach leads to a computational complexity independent of the number of demonstrations. This makes it especially well-suited for applications in life-long learning scenarios commonly encountered in robotics. We demonstrate that our approach achieves performance commensurate to the state-of-the-art on existing benchmarks already with simple, comparatively shallow network architectures while significantly outperforming the state-of-the-art on an alternative benchmark based on more complex, highly varying reward structures representing strong interactions between features. Furthermore, we extend the approach to include convolutional layers in order to eliminate the dependency on precomputed features of current algorithms and to underline the substantial gain in flexibility in framing IRL in the context of deep learning."


#### Finn, Levine, Abbeel - ["Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization"](https://arxiv.org/abs/1603.00448)
>	"Reinforcement learning can acquire complex behaviors from high-level specifications. However, defining a cost function that can be optimized effectively and encodes the correct task is challenging in practice. We explore how inverse optimal control can be used to learn behaviors from demonstrations, with applications to torque control of high-dimensional robotic systems. Our method addresses two key challenges in inverse optimal control: first, the need for informative features and effective regularization to impose structure on the cost, and second, the difficulty of learning the cost function under unknown dynamics for high-dimensional continuous systems. To address the former challenge, we present an algorithm capable of learning arbitrary nonlinear cost functions, such as neural networks, without meticulous feature engineering. To address the latter challenge, we formulate an efficient sample-based approximation for MaxEnt IOC. We evaluate our method on a series of simulated tasks and real-world robotic manipulation problems, demonstrating substantial improvement over prior methods both in terms of task complexity and sample efficiency."

----
>	"technique that lets one apply Maximum Entropy Inverse Optimal Control without the double-loop procedure and using policy gradient techniques"

  - <https://youtube.com/watch?v=hXxaepw0zAw> (demo)
  - <http://techtalks.tv/talks/guided-cost-learning-deep-inverse-optimal-control-via-policy-optimization/62472/> (Finn)


#### Ho, Gupta, Ermon - ["Model-Free Imitation Learning with Policy Optimization"](http://arxiv.org/abs/1605.08478)
>	"In imitation learning, an agent learns how to behave in an environment with an unknown cost function by mimicking expert demonstrations. Existing imitation learning algorithms typically involve solving a sequence of planning or reinforcement learning problems. Such algorithms are therefore not directly applicable to large, high-dimensional environments, and their performance can significantly degrade if the planning problems are not solved to optimality. Under the apprenticeship learning formalism, we develop alternative model-free algorithms for finding a parameterized stochastic policy that performs at least as well as an expert policy on an unknown cost function, based on sample trajectories from the expert. Our approach, based on policy gradients, scales to large continuous environments with guaranteed convergence to local minima."

>	"We showed that carefully blending state-of-the-art policy gradient algorithms for reinforcement learning with local cost function fitting lets us successfully train neural network policies for imitation in high-dimensional, continuous environments. Our method is able to identify a locally optimal solution, even in settings where optimal planning is out of reach. This is a significant advantage over competing algorithms that require repeatedly solving planning problems in an inner loop. In fact, when the inner planning problem is only approximately solved, competing algorithms do not even provide local optimality guarantees (Ermon et al., 2015). Our approach does not use expert interaction or reinforcement signal, fitting in a family of such approaches that includes apprenticeship learning and inverse reinforcement learning. When either of these additional resources is provided, alternative approaches (Kim et al., 2013; Daume III et al., 2009; Ross & Bagnell, 2010; Ross et al., 2011) may be more sample efficient, and investigating ways to combine these resources with our framework is an interesting research direction. We focused on the policy optimization component of apprenticeship learning, rather than the design of appropriate cost function classes. We believe this is an important area for future work. Nonlinear cost function classes have been successful in IRL (Ratliff et al., 2009; Levine et al., 2011) as well as in other machine learning problems reminiscent of ours, in particular that of training generative image models. In the language of generative adversarial networks (Goodfellow et al., 2014), the policy parameterizes a generative model of state-action pairs, and the cost function serves as an adversary. Apprenticeship learning with large cost function classes capable of distinguishing between arbitrary state-action visitation distributions would, enticingly, open up the possibility of exact imitation."

  - <http://techtalks.tv/talks/model-free-imitation-learning-with-policy-optimization/62471/> (Ho)


#### Ho, Ermon - ["Generative Adversarial Imitation Learning"](http://arxiv.org/abs/1606.03476)
>	"Consider learning a policy from example expert behavior, without interaction with the expert or access to reinforcement signal. One approach is to recover the expert’s cost function with inverse reinforcement learning, then extract a policy from that cost function with reinforcement learning. This approach is indirect and can be slow. We propose a new general framework for directly extracting a policy from data, as if it were obtained by reinforcement learning following inverse reinforcement learning. We show that a certain instantiation of our framework draws an analogy between imitation learning and generative adversarial networks, from which we derive a model-free imitation learning algorithm that obtains significant performance gains over existing model-free methods in imitating complex behaviors in large, high-dimensional environments."

>	"As we demonstrated, our method is generally quite sample efficient in terms of expert data. However, it is not particularly sample efficient in terms of environment interaction during training. The number of such samples required to estimate the imitation objective gradient was comparable to the number needed for TRPO to train the expert policies from reinforcement signals. We believe that we could significantly improve learning speed for our algorithm by initializing policy parameters with behavioral cloning, which requires no environment interaction at all. Fundamentally, our method is model free, so it will generally need more environment interaction than model-based methods. Guided cost learning, for instance, builds upon guided policy search and inherits its sample efficiency, but also inherits its requirement that the model is well-approximated by iteratively fitted time-varying linear dynamics. Interestingly, both our Algorithm 1 and guided cost learning alternate between policy optimization steps and cost fitting (which we called discriminator fitting), even though the two algorithms are derived completely differently. Our approach builds upon a vast line of work on IRL, and hence, just like IRL, our approach does not interact with the expert during training. Our method explores randomly to determine which actions bring a policy’s occupancy measure closer to the expert’s, whereas methods that do interact with the expert, like DAgger, can simply ask the expert for such actions. Ultimately, we believe that a method that combines well-chosen environment models with expert interaction will win in terms of sample complexity of both expert data and environment interaction."

>	"Popular imitation approaches involve a two-stage pipeline: first learning a reward function, then running RL on that reward. Such a pipeline can be slow, and because it’s indirect, it is hard to guarantee that the resulting policy works well. This work shows how one can directly extract policies from data via a connection to GANs. As a result, this approach can be used to learn policies from expert demonstrations (without rewards)."

  - <https://github.com/openai/imitation>
  - <https://github.com/DanielTakeshi/rl_algorithms/tree/master/il>


#### Li, Song, Ermon - ["Inferring The Latent Structure of Human Decision-Making from Raw Visual Inputs"](https://arxiv.org/abs/1703.08840)
>	"The goal of imitation learning is to match example expert behavior, without access to a reinforcement signal. Expert demonstrations provided by humans, however, often show signifi- cant variability due to latent factors that are not explicitly modeled. We introduce an extension to the Generative Adversarial Imitation Learning method that can infer the latent structure of human decision-making in an unsupervised way. Our method can not only imitate complex behaviors, but also learn interpretable and meaningful representations. We demonstrate that the approach is applicable to high-dimensional environments including raw visual inputs. In the highway driving domain, we show that a model learned from demonstrations is able to both produce different styles of human-like driving behaviors and accurately anticipate human actions. Our method surpasses various baselines in terms of performance and functionality."

>	"In imitation learning, example demonstrations are typically provided by human experts. These demonstrations can show significant variability. For example, they might be collected from multiple experts, each employing a different policy. External latent factors of variation that are not explicitly captured by the simulation environment can also significantly affect the observed behavior. For example, expert driving demonstrations might be collected from users with different skills and habits. The goal of this paper is to develop an imitation learning framework that is able to automatically discover and disentangle the latent factors of variation underlying human decision-making. Analogous to the goal of uncovering style, shape, and color in generative modeling of images (Chen et al., 2016), we aim to automatically learn concepts such as driver aggressiveness from human demonstrations."

>	"We propose a new method for learning a latent variable generative model of trajectories in a dynamic environment that not only accurately reproduce expert behavior, but also learns a latent space that is semantically meaningful. Our approach is an extension of GAIL, where the objective is augmented with a mutual information term between the latent variables and the observed state-action pairs. We demonstrate an application in autonomous driving, where we learn to imitate complex driving behaviors while learning semantically meaningful structure, without any supervision beyond the expert trajectories. Remarkably, our method performs directly on raw visual inputs, using raw pixels as the only source of perceptual information."


#### Hadfield-Menell, Dragan, Abbeel, Russell - ["Cooperative Inverse Reinforcement Learning"](http://arxiv.org/abs/1606.03137)
>	"For an autonomous system to be helpful to humans and to pose no unwarranted risks, it needs to align its values with those of the humans in its environment in such a way that its actions contribute to the maximization of value for the humans. We propose a formal definition of the value alignment problem as cooperative inverse reinforcement learning. A CIRL problem is a cooperative, partial information game with two agents, human and robot; both are rewarded according to the human’s reward function, but the robot does not initially know what this is. In contrast to classical IRL, where the human is assumed to act optimally in isolation, optimal CIRL solutions produce behaviors such as active teaching, active learning, and communicative actions that are more effective in achieving value alignment. We show that computing optimal joint policies in CIRL games can be reduced to solving a POMDP, prove that optimality in isolation is suboptimal in CIRL, and derive an approximate CIRL algorithm."

>	"In this work, we presented a game-theoretic model for cooperative learning, CIRL. Key to this model is that the robot knows that it is in a shared environment and is attempting to maximize the human’s reward (as opposed to estimating the human’s reward function and adopting it as its own). This leads to cooperative learning behavior and provides a framework in which to design HRI algorithms and analyze the incentives of both actors in a learning environment. We reduced the problem of computing an optimal policy pair to solving a POMDP. This is a useful theoretical tool and can be used to design new algorithms, but it is clear that optimal policy pairs are only part of the story. In particular, when it performs a centralized computation, the reduction assumes that we can effectively program both actors to follow a set coordination policy. This may not be feasible in reality, although it may nonetheless be helpful in training humans to be better teachers. An important avenue for future research will be to consider the problem of equilibrium acquisition: the process by which two independent actors arrive at an equilibrium pair of policies. Returning to Wiener’s warning, we believe that the best solution is not to put a specific purpose into the machine at all, but instead to design machines that provably converge to the right purpose as they go along."

  - <http://pemami4911.github.io/paper-summaries/2016/08/11/Coop-Inverse-RL.html>




<brylevkirill (at) gmail.com>
