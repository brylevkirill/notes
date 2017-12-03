

  * [overview](#overview)
  * [study](#study)
  * [problems](#problems)
  * [semantics](#semantics)
  * [compositionality](#compositionality)
  * [continuous space representations](#continuous-space-representations)
  * [interesting papers](#interesting-papers)
    - [cross-lingual tasks](#interesting-papers---cross-lingual-tasks)
    - [language models](#interesting-papers---language-models)
    - [word embeddings](#interesting-papers---word-embeddings)
    - [word sense disambiguation](#interesting-papers---word-sense-disambiguation)
    - [semantic composition](#interesting-papers---semantic-composition)
    - [syntactic parsing](#interesting-papers---syntactic-parsing)
    - [semantic parsing](#interesting-papers---semantic-parsing)
    - [text classification](#interesting-papers---text-classification)
    - [word sequence labelling](#interesting-papers---word-sequence-labelling)
    - [coreference resolution](#interesting-papers---coreference-resolution)
    - [text summarization](#interesting-papers---text-summarization)



---
### overview

  "The ambiguous nature of natural language might seem like a flaw, but in fact, it is exactly this ambiguity that makes natural language so powerful. Think of language as a game between a speaker and a listener. Game play proceeds as follows: the speaker thinks of a concept, she chooses an utterance to convey that concept, and the listener interprets the utterance. Both players win if the listener’s interpretation matches the speaker’s intention. To play this game well, the speaker should thus choose the simplest utterance that conveys her intended concept - anything the listener can infer can be omitted. How can a computer fill in these gaps, which depend on the breadth of human experience involving perception of the world and social interactions?"

  *(Percy Liang)*

  "The language is meant to serve for communication between a builder A and an assistant B. A is building with building-stones: there are blocks, pillars, slabs and beams. B has to pass the stones, and that in the order in which A needs them. For this purpose they use a language consisting of the words “block”, “pillar”, “slab”, “beam”. A calls them out - B brings the stone which he has learnt to bring at such-and-such a call. Conceive this as a complete primitive language."

  *(Ludwig Wittgenstein)*

  "This game is not just a game of words but a game of words causing things and of other things causing words. We can’t fully define the meaning of a word like “slab” without referring to the physical actions of A and B. In this way, linguistic meaning has to bottom out at some point in nonlinguistic facts."

  *(Jon Gauthier)*

  "No general solution to the problem of computer understanding of natural language is possible, i.e. language is understood only in contextual frameworks, that even these can be shared by people to only a limited extent, and that consequently even people are not embodiments of any such general solution."

  *(Joseph Weizenbaum, creator of ELIZA chat bot)*

  "All existing NLP is about mapping the internal statistical dependencies of language, missing the point that language is a *communication protocol*. You cannot study language without considering *agents* communicating *about something*. The only reason language even has any statistical dependencies to study is because it's imperfect. A maximally efficient communication protocol would look like random noise, out of context (besides error correction mechanisms). All culture is a form of communication, so "understanding" art requires grounding. Mimicking what humans do isn't enough. You can't understand language without considering it in context: agents communicating about something. An analogy could be trying to understand an economy by looking at statistical structure in stock prices only."

  *(Francois Chollet)*

  "We have a lot of language data [at Google]... and we don't even know how we would annotate it."

  *(Ray Kurzweil)* ([talk](https://youtube.com/watch?v=w9sz7eW6MY8#t=4m27s) `video`)

----

  ["A Paradigm for Situated and Goal-Driven Language Learning"](#a-paradigm-for-situated-and-goal-driven-language-learning-gauthier-mordatch) by Jon Gauthier and Igor Mordatch `paper`  
>	"We outlined a paradigm for grounded and goal-driven language learning in artificial agents. The paradigm is centered around a utilitarian definition of language understanding, which equates language understanding with the ability to cooperate with other language users in real-world environments. This position demotes language from its position as a separate task to be solved to one of several communicative tools agents might use to accomplish their real-world goals."

  ["On 'Solving Language'"](http://foldl.me/2016/solving-language/) by Jon Gauthier  
  ["Situated Language Learning"](http://foldl.me/2016/situated-language-learning/) by Jon Gauthier  

  ["From Models of Language Understanding to Agents of Language Use"](http://research.microsoft.com/apps/video/default.aspx?id=266643) by Felix Hill `video`

  [interesting recent papers - language grounding](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#language-grounding)



---
### study

  **classic NLP**

  [course](http://youtube.com/playlist?list=PL6397E4B26D00A269) by Chris Manning and Dan Jurafsky ([slides](https://web.stanford.edu/~jurafsky/NLPCourseraSlides.html))  
  [course](http://youtube.com/user/afigfigueira/playlists?shelf_id=5&view=50) by Michael Collins  
  [course](http://youtube.com/playlist?list=PLegWUnz91WfuPebLI97-WueAP90JO-15i) by Jordan Boyd-Graber  
  [course](https://stepik.org/course/1233) by Pavel Braslavski `in russian`  

----

  **NLP with Deep Learning**

  [course](http://youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) from Stanford ([notes](https://github.com/stanfordnlp/cs224n-winter17-notes))  
  [course](http://github.com/oxford-cs-deepnlp-2017/lectures) from Oxford and DeepMind  
  [course](http://phontron.com/class/nn4nlp2017/index.html) from CMU  

----

  ["Speech and Language Processing"](http://web.stanford.edu/~jurafsky/slp3/) book  by Dan Jurafsky and James Martin  
  ["Foundations of Statistical Natural Language Processing"](http://nlp.stanford.edu/fsnlp/) book by Chris Manning and Hinrich Schutze  
  ["Neural Network Methods for Natural Language Processing"](http://morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037) book by Yoav Goldberg  

----

  [blog](http://ruder.io/index.html#open) by Sebastian Ruder

----

  ["NLP News"](http://newsletter.ruder.io) newsletter from Sebastian Ruder  
  ["NLP Highlights"](https://soundcloud.com/nlp-highlights) podcast from AI2  



---
### problems

  - anaphora resolution
  - answer sentence selection
  - computational morphology
  - connecting language and perception
  - coreference resolution
  - dialog system
  - discourse analysis
  - document classification
  - entity extraction
  - entity linking
  - entity relation classification
  - entity salience
  - information extraction
  - language generation
  - language modeling
  - lexicon acquisition
  - machine comprehension
  - machine translation
  - morphological segmentation
  - named entity recognition
  - natural language inference
  - natural language generation
  - natural language understanding
  - next utterance ranking
  - optical character recognition
  - paraphrase detection
  - part-of-speech tagging
  - question answering
  - recognizing textual entailment
  - semantic relation classification
  - semantic role labeling
  - semantic parsing
  - semantic similarity
  - sentence breaking
  - sentiment analysis
  - sentiment attribution
  - slot-filling
  - speech recognition
  - statistical relational learning
  - syntactic parsing
  - text categorization / clustering
  - text segmentation / chunking
  - text summarization
  - textual entailment
  - topic modeling
  - word segmentation
  - word sense disambiguation



---
### semantics

  "Semantics encompasses extracting structured data from text (knowledge base extraction, logical form extraction, information extraction), linguistic approaches to extract and compose representation of meaning, inference and reasoning over meaning representation based on logic or algebra. It also includes approaches that aims at grounding language by learning relations between language and visual observations, linking language to the physical world."

  - surface form (text)
  - semantic representation (formulas, programs, vector space elements)
  - logical form (first-order or higher-order logic)
  - denotation (result of interpretation) (symbolic expression, changes in world state, changes in logical form)

----

  ["Natural Language Understanding: Foundations and State-of-the-Art"](https://youtube.com/watch?v=mhHfnhh-pB4) by Percy Liang `video`
	([write-up](http://topbots.com/4-different-approaches-natural-language-processing-understanding/))

>	"Building systems that can understand human language - being able to answer questions, follow instructions, carry on dialogues - has been a long-standing challenge since the early days of AI. Due to recent advances in machine learning, there is again renewed interest in taking on this formidable task. A major question is how one represents and learns the semantics (meaning) of natural language, to which there are only partial answers. The goal of this tutorial is (i) to describe the linguistic and statistical challenges that any system must address; and (ii) to describe the types of cutting edge approaches and the remaining open problems. Topics include distributional semantics (e.g., word vectors), frame semantics (e.g., semantic role labeling), model-theoretic semantics (e.g., semantic parsing), the role of context, grounding, neural networks, latent variables, and inference. The hope is that this unified presentation will clarify the landscape, and show that this is an exciting time for the machine learning community to engage in the problems in natural language understanding."

----

  - *formal semantics*  
	* focusing on functional elements and composition while largely ignoring lexical aspects of meaning and lacking methods to learn the proposed structures from data  
	* focusing on prepositions, articles, quantifiers, coordination, auxiliary verbs, propnouns, negation  

  - *distributional semantics*  
	* based on the idea that semantic information can be extracted from lexical co-occurrence from large-scale data corpora  
	* allows the construction of model of meaning, where the degree of the semantic association between different words can be quantified  
	* distributional interpretation of a target term is defined by a weighted vector of the contexts in which the term occurs  
	* focusing on single content words while ignoring functional elements and compositionality  
	* focusing on nouns, adjectives, verbs  

  [formal vs distributional semantics](http://unibuc.ro/prof/dinu_a_d/docs/2014/feb/24_16_00_58FormalvsDistributionalSemantics.pdf)



---
### compositionality

  "The principle of compositionality states that the meaning of a complex syntactic phrase is a function of the meanings of its parts and their mode of combination."

  "Intuitively, compositionality outlines a recursive interpretation process in which the lexical items are listed as base cases and the recursive clauses define the modes of combination. Current theories assume that the modes of combination are few and highly general, which places essentially all of the complexity in the lexicon."

  "Compositionality is central to characterizing the ways in which small changes to a syntactic structure can yield profound changes in meaning. For instance, “two minus three” and “three minus two” contain the same words but lead to different denotations. Superficially least contentful words (“every”, “no”, “not”, “might”, “but”, etc.) often drive the most dramatic effects depending on where they appear in the constituent structure, which determines what their corresponding semantic arguments are. The grammar gives a precise account of how this interpretive consequence follows from the syntax of the two utterances."

  "Compositionality is often linked to our ability to produce and interpret novel utterances. While it is too strong to say that compositionality is necessary or sufficient for this kind of creative ability, it does help characterize it: once one has acquired the syntax of the language, memorized all the lexical meanings, and mastered the few modes of composition, one can interpret novel combinations of them. This abstract capacity is at the heart of what computational models of semantics would like to learn, so that they too can efficiently interpret novel combinations of words and phrases."

  "The problem is that human language is much more fluid than old-school, Chomskyan grammars can support. Even limiting research to just the English language, there are so many variations and so many mutations that the lifetime of a rigid, formally defined grammar becomes too short to be feasible. For a language processing system to have meaning in the real world, it must be able to update and change to reflect change in the language. Logic-based systems would be much more interesting - and, certainly, such systems are necessary if we ever want to build a system that understands instead of just translates. However, the system must be able to learn the rules of the language on the fly. It seems to me that hybrid systems have the most potential."

  "It's important to make a distinction between (1) Chomskyan linguistics, (2) 90s style symbolic systems, (3) 90s/early 2000s style statistical systems and (4) 2010s style statistical systems. Chomskyan linguistics assumes that statistics and related stuff is not relevant at all, and that instead you need to find the god-given (or at least innate) Universal Grammar and then everything will be great. 90s style symbolic systems adopt a more realistic approach, relying on lots of heuristics that kind of work but aim at good performance rather than unattainable perfection. 90s style statistical models give up some of the insights in these heuristics to construct tractable statistical models. If you look at 2010s style statistical models, you'll notice that machine learning has become more powerful and you can use a greater variety of information, either using good linguistic intuitions (which help even more with better learning algorithms, but require a certain expressivity as well as some degree of matching between the way of constructing the features and the classification) or unsupervised/deep-NN learning, which constructs generalizations over features."

----

  - (probabilistic) regular grammar (~Markov chain)
  - (probabilistic) context-free grammar
  - compositional vector grammar (pcfg + compositional distributed representation)
  - combinatory categorial grammar
  - pregroup grammar
  - lexical functional grammar
  - head-driven phrase structure grammar
  - generalized phrase structure grammar
  - lexicalized tree-adjoining grammars
  - two-level grammar
  - Aspects model
  - RG model

----

  - constituency parsing  
	[overview](http://lxmls.it.pt/2016/Part_1_Constituency_Parsing_2016.pdf) by Slav Petrov `slides`  
	[overview](http://youtube.com/watch?v=sL_W_I8DpuU) by John Boyd-Graber `video`  

  - dependency parsing  
	[overview](http://lxmls.it.pt/2016/Part2_Dependency_Parsing_2016.pdf) by Slav Petrov `slides`  
	[overview](http://youtube.com/watch?v=du9VQaFEyeA) by John Boyd-Graber `video`  

  - CCG parsing  
	[overview](http://youtube.com/playlist?list=PLun-LUE1uLNvWi-qV-tRHohfHR90Y_cAk) by Yoav Artzi `video`

----

  [English grammar explained](http://chompchomp.com/terms/)



---
### continuous space representations

  [distributed representations](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#architectures---distributed-representations)

----

  “You can’t cram the meaning of a whole %&!$ing sentence into a single $&!*ing vector!”

  *(Ray Mooney)*

----

  ["Vector Space Models of Lexical Meaning"](http://www.cl.cam.ac.uk/~sc609/pubs/sem_handbook.pdf) by Stephen Clark `paper`
>	"In Formal Semantics the meanings of phrases or sentences are represented in terms of set-theoretic models. The key intuition behind Formal Semantics, very roughly, is that the world is full of objects; objects have properties; and relations hold between objects. Set-theoretic models are ideal for capturing this intuition, and have been succcessful at providing formal descriptions of key elements of natural language semantics, for example quantification. This approach has also proven attractive for Computational Semantics - the discipline concerned with representing, and reasoning with, the meanings of natural language utterances using a computer. One reason is that the formalisms used in the set-theoretic approaches, e.g. first-order predicate calculus, have well-defined inference mechanisms which can be implemented on a computer.  
>	Another approach is a different branch of mathematics from the set theory employed in most studies in Formal Semantics, namely the mathematical framework of vector spaces and linear algebra. The attraction of using vector spaces is that they provide a natural mechanism for talking about distance and similarity, concepts from geometry. Why should a geometric approach to modelling natural language semantics be appropriate? There are many aspects of semantics, particularly lexical semantics, which require a notion of distance. For example, the meaning of the word cat is closer to the meaning of the word dog than the meaning of the word car. The modelling of such distances is now commonplace in Computational Linguistics, since many examples of language technology benefit from knowing how word meanings are related geometrically; for example, a search engine could expand the range of web pages being returned for a set of query terms by considering additional terms which are close in meaning to those in the query.  
>	The meanings of words have largely been neglected in Formal Semantics, typically being represented as atomic entities such as dog, whose interpretation is to denote some object (or set of objects) in a set-theoretic model. In this framework semantic relations among lexical items are encoded in meaning postulates, which are constraints on possible models. Contrary, meanings of words can be represented using vectors, as part of a high-dimensional “semantic space”. The fine-grained structure of this space is provided by considering the contexts in which words occur in large corpora of text. Words can easily be compared for similarity in the vector space, using any of the standard similarity or distance measures available from linear algebra, for example, the cosine of the angle between two vectors."  
>	"One of the reasons for popularity of those new models is that they provide a level of robustness required for effective language technology which has notoriously not been provided by more traditional logic-based approaches. It is probably fair to say that the classical logic-based enterprise in natural language processing has failed. However, whether fundamental concepts from semantics, such as inference, can be suitably incorporated into vector space models of meaning is an open question.  
>	In addition to incorporating traditional concepts from semantics, another area which is likely to grow is the incorporation of other modalities, such as vision, into the vector representations. A semantic theory incorporating distributional representations, at the word, phrase, sentence- and perhaps even document-level; multi-modal features; and effective and robust methods of inference, is a grand vision."

----

  [overview](http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/) of word embeddings by Sebastian Ruder  
  "On Word Embeddings" by Sebastian Ruder ([part 1](http://sebastianruder.com/word-embeddings-1/index.html), [part 2](http://sebastianruder.com/word-embeddings-softmax/index.html))  

  ["Word Vector Representations: word2vec"](https://youtube.com/watch?v=ERibwqs9p38) by Richard Socher `video`  
  ["GloVe: Global Vectors for Word Representation"](https://youtube.com/watch?v=ASn7ExxLZws) by Richard Socher `video`  
  ["Vector Representations of Words and Documents"](https://youtube.com/watch?v=KEXWC-ICH_Y) by Anna Potapenko `video` `in russian`  

  ["A Theoretical Approach to Semantic Representations"](https://youtube.com/watch?v=KR46z_V0BVw) by Sanjeev Arora `video`  
  "Word Embeddings: Explaining their properties" by Sanjeev Arora
	([part 1](http://www.offconvex.org/2015/12/12/word-embeddings-1/), [part 2](http://www.offconvex.org/2016/02/14/word-embeddings-2/))  



---
### interesting papers

  - [language models](#interesting-papers---language-models)
  - [dialog systems](#interesting-papers---dialog-systems)
  - [cross-lingual tasks](#interesting-papers---cross-lingual-tasks)
  - [word embeddings](#interesting-papers---word-embeddings)
  - [word sense disambiguation](#interesting-papers---word-sense-disambiguation)
  - [semantic composition](#interesting-papers---semantic-composition)
  - [syntactic parsing](#interesting-papers---syntactic-parsing)
  - [semantic parsing](#interesting-papers---semantic-parsing)
  - [text classification](#interesting-papers---text-classification)
  - [word sequence labelling](#interesting-papers---word-sequence-labelling)
  - [coreference resolution](#interesting-papers---coreference-resolution)
  - [text summarization](#interesting-papers---text-summarization)


[interesting recent papers - natural language processing](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#natural-language-processing)  
[interesting recent papers - language grounding](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#language-grounding)  


----

#### ["A Paradigm for Situated and Goal-Driven Language Learning"](https://arxiv.org/abs/1610.03585) Gauthier, Mordatch
>	"A distinguishing property of human intelligence is the ability to flexibly use language in order to communicate complex ideas with other humans in a variety of contexts. Research in natural language dialogue should focus on designing communicative agents which can integrate themselves into these contexts and productively collaborate with humans. In this abstract, we propose a general situated language learning paradigm which is designed to bring about robust language agents able to cooperate productively with humans. This dialogue paradigm is built on a utilitarian definition of language understanding. Language is one of multiple tools which an agent may use to accomplish goals in its environment. We say an agent “understands” language only when it is able to use language productively to accomplish these goals. Under this definition, an agent’s communication success reduces to its success on tasks within its environment. This setup contrasts with many conventional natural language tasks, which maximize linguistic objectives derived from static datasets. Such applications often make the mistake of reifying language as an end in itself. The tasks prioritize an isolated measure of linguistic intelligence (often one of linguistic competence, in the sense of Chomsky), rather than measuring a model’s effectiveness in real-world scenarios. Our utilitarian definition is motivated by recent successes in reinforcement learning methods. In a reinforcement learning setting, agents maximize success metrics on real-world tasks, without requiring direct supervision of linguistic behavior."

>	"We outlined a paradigm for grounded and goal-driven language learning in artificial agents. The paradigm is centered around a utilitarian definition of language understanding, which equates language understanding with the ability to cooperate with other language users in real-world environments. This position demotes language from its position as a separate task to be solved to one of several communicative tools agents might use to accomplish their real-world goals. While this paradigm does already capture a small amount of recent work in dialogue, on the whole it has not received the focus it deserves in the research communities of natural language processing and machine learning. We hope this paper brings focus to the task of situated language learning as a way forward for research in natural language dialogue."

  - `post` ["On 'Solving Language'"](http://foldl.me/2016/solving-language/) (Gauthier)
  - `post` ["Situated Language Learning"](http://foldl.me/2016/situated-language-learning/) (Gauthier)


#### ["Emergence of Grounded Compositional Language in Multi-Agent Populations"](http://arxiv.org/abs/1703.04908) Mordatch, Abbeel
>	"By capturing statistical patterns in large corpora, machine learning has enabled significant advances in natural language processing, including in machine translation, question answering, and sentiment analysis. However, for agents to intelligently interact with humans, simply capturing the statistical patterns is insufficient. In this paper we investigate if, and how, grounded compositional language can emerge as a means to achieve goals in multi-agent populations. Towards this end, we propose a multi-agent learning environment and learning methods that bring about emergence of a basic compositional language. This language is represented as streams of abstract discrete symbols uttered by agents over time, but nonetheless has a coherent structure that possesses a defined vocabulary and syntax. We also observe emergence of non-verbal communication such as pointing and guiding when language communication is unavailable."

>	"Though the agents come up with words that we found to correspond to objects and other agents, as well as actions like 'Look at' or 'Go to', to the agents these words are abstract symbols represented by one-hot vector - we label these one-hot vectors with English words that capture their meaning for the sake of interpretability."

>	"One possible scenario is from goal oriented-dialog systems. Where one agent tries to transmit to another certain API call that it should perform (book restaurant, hotel, whatever). I think these models can make it more data efficient. At the first stage two agents have to communicate and discover their own language, then you can add regularization to make the language look more like natural language and on the final stage, you are adding a small amount of real data (dialog examples specific for your task). I bet that using additional communication loss will make the model more data efficient."

>	"The big outcome to hunt for in this space is a post-gradient descent learning algorithm. Of course you can make agents that play the symbol grounding game, but it's not a very big step from there to compression of data, and from there to compression of 'what you need to know to solve the problem you're about to encounter' - at which point you have a system which can learn by training or learn by receiving messages. It was pretty easy to get stuff like one agent learning a classifier, encoding it in a message, and transmitting it to a second agent who has to use it for zero-shot classification. But it's still single-task specific communication, so there's no benefit to the agent for receiving, say, the messages associated with the previous 100 tasks. The tricky thing is going from there to something more abstract and cumulative, so that you can actually use message generation as an iterative learning mechanism. I think a large part of that difficulty is actually designing the task ensemble, not just the network architecture."

  - `post` <https://blog.openai.com/learning-to-communicate/>
  - `video` <https://youtube.com/watch?v=liVFy7ZO4OA> (demo)
  - `video` <https://youtube.com/watch?v=f4gKhK8Q6mY&t=22m20s> (Abbeel)
  - `paper` ["A Paradigm for Situated and Goal-Driven Language Learning"](#a-paradigm-for-situated-and-goal-driven-language-learning-gauthier-mordatch) by Gauthier and Mordatch


#### ["Multi-Agent Cooperation and the Emergence of (Natural) Language"](https://arxiv.org/abs/1612.07182) Lazaridou, Peysakhovich, Baroni
>	"The current mainstream approach to train natural language systems is to expose them to large amounts of text. This passive learning is problematic if we are interested in developing interactive machines, such as conversational agents. We propose a framework for language learning that relies on multi-agent communication. We study this learning in the context of referential games. In these games, a sender and a receiver see a pair of images. The sender is told one of them is the target and is allowed to send a message from a fixed, arbitrary vocabulary to the receiver. The receiver must rely on this message to identify the target. Thus, the agents develop their own language interactively out of the need to communicate. We show that two networks with simple configurations are able to learn to coordinate in the referential game. We further explore how to make changes to the game environment to cause the "word meanings" induced in the game to better reflect intuitive semantic properties of the images. In addition, we present a simple strategy for grounding the agents' code into natural language. Both of these are necessary steps towards developing machines that are able to communicate with humans productively."

  - `video` <https://facebook.com/iclr.cc/videos/1712966538732405/> (Peysakhovich)


#### ["Learning Language Games through Interaction"](https://arxiv.org/abs/1606.02447) Wang, Liang, Manning
>	"We introduce a new language learning setting relevant to building adaptive natural language interfaces. It is inspired by Wittgenstein’s language games: a human wishes to accomplish some task (e.g., achieving a certain configuration of blocks), but can only communicate with a computer, who performs the actual actions (e.g., removing all red blocks). The computer initially knows nothing about language and therefore must learn it from scratch through interaction, while the human adapts to the computer’s capabilities. We created a game called SHRDLURN in a blocks world and collected interactions from 100 people playing it. First, we analyze the humans’ strategies, showing that using compositionality and avoiding synonyms correlates positively with task performance. Second, we compare computer strategies, showing that modeling pragmatics on a semantic parsing model accelerates learning for more strategic players."

>	"Today, natural language interfaces on computers or phones are often trained once and deployed, and users must just live with their limitations. Allowing users to demonstrate or teach the computer appears to be a central component to enable more natural and usable NLIs. Examining language acquisition research, there is considerable evidence suggesting that human children require interactions to learn language, as opposed to passively absorbing language, such as when watching TV. Research suggests that when learning a language, rather than consciously analyzing increasingly complex linguistic structures (e.g. sentence forms, word conjugations), humans advance their linguistic ability through meaningful interactions. In contrast, the standard machine learning dataset setting has no interaction. The feedback stays the same and does not depend on the state of the system or the actions taken. We think that interactivity is important, and that an interactive language learning setting will enable adaptive and customizable systems, especially for resource-poor languages and new domains where starting from close to scratch is unavoidable. We describe two attempts towards interactive language learning — an agent for manipulating blocks, and a calendar scheduler."

>	"Inspired by the human language acquisition process, we investigated a simple setting where language learning starts from scratch. We explored the idea of language games, where the computer and the human user need to collaboratively accomplish a goal even though they do not initially speak a common language. Specifically, in our pilot we created a game called SHRDLURN, in homage to the seminal work of Terry Winograd. As shown in Figure 1a, the objective is to transform a start state into a goal state, but the only action the human can take is entering an utterance. The computer parses the utterance and produces a ranked list of possible interpretations according to its current model. The human scrolls through the list and chooses the intended one, simultaneously advancing the state of the blocks and providing feedback to the computer. Both the human and the computer wish to reach the goal state (only known to the human) with as little scrolling as possible. For the computer to be successful, it has to learn the human’s language quickly over the course of the game, so that the human can accomplish the goal more efficiently. Conversely, the human can also speed up progress by accommodating to the computer, by at least partially understanding what it can and cannot currently do."

>	"We model the computer as a semantic parser, which maps natural language utterances (e.g., ‘remove red’) into logical forms (e.g., remove(with(red))). The semantic parser has no seed lexicon and no annotated logical forms, so it just generates many candidate logical forms. From the human’s feedback, it learn by adjusting the parameters corresponding to simple and generic lexical features. It is crucial that the computer learns quickly, or users are frustrated and the system is less usable. In addition to feature engineering and tuning online learning algorithms, we achieved higher learning speed by incorporating pragmatics. However, what is special here is the real-time nature of learning, in which the human also learns and adapts to the computer, thus making it easier to achieve good task performance. While the human can teach the computer any language - in our pilot, Mechanical Turk users tried English, Arabic, Polish, and a custom programming language - a good human player will choose to use utterances so that the computer is more likely to learn quickly."

>	"Looking forward, we believe that the ILLG setting is worth studying and has important implications for natural language interfaces. Today, these systems are trained once and deployed. If these systems could quickly adapt to user feedback in real-time as in this work, then we might be able to more readily create systems for resource-poor languages and new domains, that are customizable and improve through use."

  - `post` <http://nlp.stanford.edu/blog/interactive-language-learning/>
  - `video` <http://youtube.com/watch?v=PfW4_3tCiw0> (demo, calendar)
  - <http://shrdlurn.sidaw.xyz> (demo, blocks world)
  - `video` <https://youtube.com/watch?v=iuazFltYgCE> (Wang)
  - `video` <https://youtu.be/mhHfnhh-pB4?t=1h5m45s> (Liang)
  - `video` <https://youtu.be/6O5sttckalE?t=40m45s> (Liang)


#### ["Ask Me Anything: Dynamic Memory Networks for Natural Language Processing"](http://arxiv.org/abs/1506.07285) Kumar, Irsoy, Su, Bradbury, English, Pierce, Ondruska, Gulrajani, Socher
>	"Most tasks in natural language processing can be cast into question answering problems over language input. We introduce the dynamic memory network, a unified neural network framework which processes input sequences and questions, forms semantic and episodic memories, and generates relevant answers. Questions trigger an iterative attention process which allows the model to condition its attention on the result of previous iterations. These results are then reasoned over in a hierarchical recurrent sequence model to generate answers. The DMN can be trained end-to-end and obtains state of the art results on several types of tasks and datasets: question answering (Facebook’s bAbI dataset), sequence modeling for part of speech tagging (WSJ-PTB), and text classification for sentiment analysis (Stanford Sentiment Treebank). The model relies exclusively on trained word vector representations and requires no string matching or manually engineered features."

>	"Question answering is a complex natural language processing task which requires an understanding of the meaning of a text and the ability to reason over relevant facts. Most, if not all, tasks in natural language processing can be cast as a question answering problem: high level tasks like machine translation ("What is the translation into French?"); sequence modeling tasks like named entity recognition ("What are the named entity tags in this sentence?") or part of speech tagging ("What are the part of speech tags?"); classification problems like sentiment analysis ("What is the sentiment?"); even multi-sentence joint classification problems like coreference resolution ("Who does 'their' refer to?"). The dynamic memory network is a neural network based model which can be trained in an end-to-end fashion for any QA task using raw input-question-answer triplets. Generally, DMN can solve sequence tagging tasks, classification problems, sequence to sequence tasks, and question answering tasks that require transitive reasoning. The DMN first processes all input, question and answer texts into sequences of semantic vector representations. The question representation triggers an iterative attention process that searches the input and retrieves relevant facts. The DMN then reasons over retrieved facts and provides an answer sequence model with an appropriate summary."

>	"Input Module: This module processes raw inputs and maps them to a representation that is useful for asking questions about this input. The input may be, for instance, an image, video, or audio signal. We focus on NLP in this paper. Hence, the input may be a sentence, a long story, a movie review, a news article, or all of Wikipedia."

>	"Semantic Memory Module: Semantic memory stores general knowledge about concepts and facts. For example, it might contain information about what a hang glider is. Initialization strategies such as distributed word vectors are popular semantic memory components that have been shown to improve performance on many NLP tasks. More complex information can be stored in the form of knowledge bases that capture relationships in the form of triplets or gazetteers, which have been useful for tasks such as named entity recognition or question answering."

>	"Question Module: The question module computes a representation of a question such as "Where did the author first fly?". This representation triggers the episodic memory module to start an iterative attention process over facts from the input sequence."

>	"Episodic Memory Module: This is the central part of the DMN. A question draws attention to specific facts from the input sequence, which are reasoned over to update this module’s memory state. This process then iterates, with each iteration providing the module with newly relevant information about the input. In other words, the module has the ability to retrieve new facts which were thought to be irrelevant in previous iterations. After several passes the module then summarizes its knowledge and provides the answer module with a final representation to produce an answer. The episodic memory module retrieves facts from the input module conditioned on the question. It then reasons over those facts to produce a final representation that the answer module will use to generate an answer. We refer to this representation as a memory. Importantly, we allow our module to take multiple passes over the facts, focusing attention on different facts at each pass. Each pass produces an episode, and these episodes are then summarized into the memory. Endowing our module with this episodic component allows its attention mechanism to attend more selectively to specific facts on each pass, as it can attend to other important facts at a later pass. It also allows for a type of transitive inference, since the first pass may uncover the need to retrieve additional facts. For instance, we are asked "Where is the football?" In the first iteration, the model ought attend to sentence "John put down the football", as the question asks about the football. Only once the model sees that John is relevant can it reason the second iteration should retrieve where John was. In its general form, the episodic memory module is characterized by an attention mechanism, a function which returns an episode given the output of the attention mechanism and the facts from the input module, and a function that summarizes the episodes into a memory."

>	"Answer Module: Given a representation from the episodic memory module, the answer module generates the model’s predicted answer."

>	"Training is cast as a supervised classification problem to minimize cross entropy error of the answer sequence. For datasets with gate supervision, such as bAbI, we also include the cross entropy error of the gates into the overall cost. Because all modules communicate over vector representations and various types of differentiable and deep neural networks with gates, the entire DMN model can be trained via backpropagation and gradient descent."

>	"There are several deep learning models that have been applied to many different tasks in NLP. For instance, recursive neural networks have been used for parsing, sentiment analysis, paraphrase detection, question answering and logical inference, among other tasks. However, because they lack the memory and question modules, a single model cannot solve as many varied tasks, nor tasks that require transitive reasoning over multiple sentences."

>	"Memory Networks model cannot be applied to the same variety of NLP tasks (unlike Dynamic Memory Networks model) since it processes sentences independently and not via a sequence model. It requires bag of n-gram string matching features as well as a separate feature that captures whether a sentence came before another one. The DMN does worse than the MemNN on tasks with long input sequences. We suspect this is due to the recurrent input sequence model having trouble modeling very long inputs. The MemNN does not suffer from this problem as it views each sentence seperately. The power of the episodic memory module is evident in tasks which require the model to iteratively retrieve facts and store them in a representation that slowly incorporates more of the relevant information of the input sequence."

>	"We believe the DMN is a potentially general model for a variety of NLP applications. The entire model can be trained end-to-end with one, albeit complex, objective function. The model uses some ideas from neuroscience such as semantic and episodic memories known to be required for complex types of reasoning. Future work will explore additional tasks, larger multi-task models and multimodal inputs and questions."

----
>	"The paper proposes an end-to-end differentiable NN module called DMN. It consists of 4 parts, an input module, question module, episodic memory module and an answer module. Their works appear quite similar to the MemNN neural network with the main difference being the episodic memory module, which is an attention based recurrent module over the input hidden states and the question state. The authors argue that in many cases multiple passes over all the facts and question can help in better question answering."

>	"The main modeling idea is that you take a question and use it to condition a neural attention mechanism that goes over some text. The text is represented in terms of hidden states of a bidirectional sequence model. Conditioned on the question, that attention mechanism goes over inputs at each time step, and connects them to an episodic memory module. That is, it opens a gate that lets the vector at a time step be fed into the episodic memory. One important aspect for some tasks is that the model goes over the input multiple times. After each time it classifies the memory state by asking "do I know enough to answer the question?" And if not, then it goes over the input again, but conditioned on the question and also the previous memory state. That way it can reason over multiple facts. Once it classifies "yes I know enough", it gives that memory vector to an output sequence model which generates the answer."

  - <http://yerevann.com/dmn-ui/> (demo)
  - `video` <https://youtube.com/watch?v=T3octNTE7Is> (Socher)
  - `video` <http://videolectures.net/deeplearning2015_socher_nlp_applications/#t=2520> (Socher)
  - `notes` <https://theneuralperspective.com/2016/11/28/ask-me-anything-dynamic-memory-networks-for-natural-language-processing/>
  - `post` <http://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/>
  - `code` <https://github.com/swstarlab/DynamicMemoryNetworks>


#### ["Harnessing Deep Neural Networks with Logic Rules"](http://arxiv.org/abs/1603.06318) Hu, Ma, Liu, Hovy, Xing
>	"Combining deep neural networks with structured logic rules is desirable to harness flexibility and reduce unpredictability of the neural models. We propose a general framework capable of enhancing various types of neural networks (e.g., CNNs and RNNs) with declarative first-order logic rules. Specifically, we develop an iterative distillation method that transfers the structured information of logic rules into the weights of neural networks. We deploy the framework on a CNN for sentiment analysis, and an RNN for named entity recognition. With a few highly intuitive rules, we obtain substantial improvements and achieve state-of-the-art or comparable results to previous best-performing systems."

>	"We have developed a framework which combines deep neural networks with first-order logic rules to allow integrating human knowledge and intentions into the neural models. In particular, we proposed an iterative distillation procedure that transfers the structured information of logic rules into the weights of neural networks. The transferring is done via a teacher network constructed using the posterior regularization principle. Our framework is general and applicable to various types of neural architectures. With a few intuitive rules, our framework significantly improves base networks on sentiment analysis and named entity recognition, demonstrating the practical significance of our approach. The encouraging results indicate a strong potential of our approach on improving other NLP tasks and application domains. We plan to explore more applications and incorporate more structured knowledge in neural networks. We also would like to improve our framework to automatically learn the importance of different rules, and derive new rules from data."

>	"Despite the impressive advances, the widely-used DNN methods still have limitations. The high predictive accuracy has heavily relied on large amounts of labeled data; and the purely data-driven learning can lead to uninterpretable and sometimes counter-intuitive results. It is also difficult to encode human intention to guide the models to capture desired patterns, without expensive direct supervision or ad-hoc initialization. On the other hand, the cognitive process of human beings have indicated that people learn not only from concrete examples (as DNNs do) but also from different forms of general knowledge and rich experiences. Logic rules provide a flexible declarative language for communicating high-level cognition and expressing structured knowledge. It is therefore desirable to integrate logic rules into DNNs, to transfer human intention and domain knowledge to neural models, and regulate the learning process."

>	"We present a framework capable of enhancing general types of neural networks, such as convolutional networks and recurrent networks, on various tasks, with logic rule knowledge. Our framework enables a neural network to learn simultaneously from labeled instances as well as logic rules, through an iterative rule knowledge distillation procedure that transfers the structured information encoded in the logic rules into the network parameters. Since the general logic rules are complementary to the specific data labels, a natural “side-product” of the integration is the support for semi-supervised learning where unlabeled data can be used to better absorb the logical knowledge. Methodologically, our approach can be seen as a combination of the knowledge distillation and the posterior regularization method. In particular, at each iteration we adapt the posterior constraint principle from PR to construct a rule-regularized teacher, and train the student network of interest to imitate the predictions of the teacher network. We leverage soft logic to support flexible rule encoding."

>	"We apply the proposed framework on both CNN and RNN, and deploy on the task of sentiment analysis and named entity recognition, respectively. With only a few (one or two) very intuitive rules, the enhanced networks strongly improve over their basic forms (without rules), and achieve better or comparable performance to state-of-the-art models which typically have more parameters and complicated architectures. By incorporating the bi-gram transition rules, we obtain 1.56 improvement in F1 score that outperforms all previous neural based methods on named entity recognition task, including the BLSTM-CRF model which applies a conditional random field on top of a BLSTM model in order to capture the transition patterns and encourage valid sequences. In contrast, our method implements the desired constraints in a more straightforward way by using the declarative logic rule language, and at the same time does not introduce extra model parameters to learn. Further integration of the list rule provides a second boost in performance, achieving an F1 score very close to the best-performing system Joint-NER-EL which is a probabilistic graphical model based method optimizing NER and entity linking jointly and using large amount of external resources."

  - `notes` <http://www.erogol.com/harnessing-deep-neural-networks-with-logic-rules/>


#### ["Sequence to Sequence Learning with Neural Networks"](http://arxiv.org/abs/1409.3215) Sutskever, Vinyals, Le
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#sequence-to-sequence-learning-with-neural-networks-sutskever-vinyals-le>


#### ["Online Segment to Segment Neural Transduction"](https://arxiv.org/abs/1609.08194) Yu, Buys, Blunsom
>	"We introduce an online neural sequence to sequence model that learns to alternate between encoding and decoding segments of the input as it is read. By independently tracking the encoding and decoding representations our algorithm permits exact polynomial marginalization of the latent segmentation during training, and during decoding beam search is employed to find the best alignment path together with the predicted output sequence. Our model tackles the bottleneck of vanilla encoder-decoders that have to read and memorize the entire input sequence in their fixedlength hidden states before producing any output. It is different from previous attentive models in that, instead of treating the attention weights as output of a deterministic function, our model assigns attention weights to a sequential latent variable which can be marginalized out and permits online generation. Experiments on abstractive sentence summarization and morphological inflection show significant performance gains over the baseline encoder-decoders."

  - `video` <http://techtalks.tv/talks/online-segment-to-segment-neural-transduction/63323/> (Yu)


#### ["The Neural Noisy Channel"](https://arxiv.org/abs/1611.02554) Yu, Blunsom, Dyer, Grefenstette, Kocisky
>	"We formulate sequence to sequence transduction as a noisy channel decoding problem and use recurrent neural networks to parameterise the source and channel models. Unlike direct models which can suffer from explaining-away effects during training, noisy channel models must produce outputs that explain their inputs, and their component models can be trained with not only paired training samples but also unpaired samples from the marginal output distribution. Using a latent variable to control how much of the conditioning sequence the channel model needs to read in order to generate a subsequent symbol, we obtain a tractable and effective beam search decoder. Experimental results on abstractive sentence summarisation, morphological inflection, and machine translation show that noisy channel models outperform direct models, and that they significantly benefit from increased amounts of unpaired output data that direct models cannot easily use."

  - `video` <http://videolectures.net/deeplearning2017_blunsom_language_understanding/> (1:15:45) (Blunsom)


#### ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) Vaswani et al.
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#attention-is-all-you-need-vaswani-et-al>


#### ["Sequence Level Training with Recurrent Neural Networks"](http://arxiv.org/abs/1511.06732) Ranzato, Chopra, Auli, Zaremba
>	"Many natural language processing applications use language models to generate text. These models are typically trained to predict the next word in a sequence, given the previous words and some context such as an image. However, at test time the model is expected to generate the entire sequence from scratch. This discrepancy makes generation brittle, as errors may accumulate along the way. We address this issue by proposing a novel sequence level training algorithm that directly optimizes the metric used at test time, such as BLEU or ROUGE. On three different tasks, our approach outperforms several strong baselines for greedy generation. The method is also competitive when these baselines employ beam search, while being several times faster."

>	"A wide variety of applications rely on text generation, including machine translation, video/text summarization, question answering, among others. From a machine learning perspective, text generation is the problem of predicting a syntactically and semantically correct sequence of consecutive words given some context. For instance, given an image, generate an appropriate caption or given a sentence in English language, translate it into French. Popular choices for text generation models are language models based on n-grams, feed-forward neural networks and recurrent neural networks. These models when used as is to generate text suffer from two major drawbacks. First, they are trained to predict the next word given the previous ground truth words as input. However, at test time, the resulting models are used to generate an entire sequence by predicting one word at a time, and by feeding the generated word back as input at the next time step. This process is very brittle because the model was trained on a different distribution of inputs, namely, words drawn from the data distribution, as opposed to words drawn from the model distribution. As a result the errors made along the way will quickly accumulate. We refer to this discrepancy as exposure bias which occurs when a model is only exposed to the training data distribution, instead of its own predictions. Second, the loss function used to train these models is at the word level. A popular choice is the cross-entropy loss used to maximize the probability of the next correct word. However, the performance of these models is typically evaluated using discrete metrics. One such metric is called BLEU for instance, which measures the n-gram overlap between the model generation and the reference text. Training these models to directly optimize metrics like BLEU is hard because a) these are not differentiable, and b) combinatorial optimization is required to determine which sub-string maximizes them given some context."

>	"This paper proposes a novel training algorithm which results in improved text generation compared to standard models. The algorithm addresses the two issues discussed above as follows. First, while training the generative model we avoid the exposure bias by using model predictions at training time. Second, we directly optimize for our final evaluation metric. We build on the REINFORCE algorithm to achieve the above two objectives. While sampling from the model during training is quite a natural step for the REINFORCE algorithm, optimizing directly for any test metric can also be achieved by it. REINFORCE side steps the issues associated with the discrete nature of the optimization by not requiring rewards (or losses) to be differentiable. While REINFORCE appears to be well suited to tackle the text generation problem, it suffers from a significant issue. The problem setting of text generation has a very large action space which makes it extremely difficult to learn with an initial random policy. Specifically, the search space for text generation is of size O(WT), where W is the number of words in the vocabulary (typically around 10^4 or more) and T is the length of the sentence (typically around 10-30). Towards that end, we introduce Mixed Incremental Cross-Entropy Reinforce. MIXER is an easy-to-implement recipe to make REINFORCE work well for text generation applications. It is based on two key ideas: incremental learning and the use of a hybrid loss function which combines both REINFORCE and cross-entropy. Both ingredients are essential to training with large action spaces. In MIXER, the model starts from the optimal policy given by cross-entropy training (as opposed to a random one), from which it then slowly deviates, in order to make use of its own predictions, as is done at test time."

>	"Our results show that MIXER with a simple greedy search achieves much better accuracy compared to the baselines on Text Summarization, Machine Translation and Image Captioning tasks. In addition we show that MIXER with greedy search is even more accurate than the cross entropy model augmented with beam search at inference time as a post-processing step. This is particularly remarkable because MIXER with greedy search is at least 10 times faster than the cross entropy model with a beam of size 10. Lastly, we note that MIXER and beam search are complementary to each other and can be combined to further improve performance, although the extent of the improvement is task dependent."

  - <https://www.evernote.com/shard/s189/sh/ada01a82-70a9-48d4-985c-20492ab91e84/8da92be19e704996dc2b929473abed46> (Larochelle)
  - <https://github.com/facebookresearch/MIXER>


#### ["Self-critical Sequence Training for Image Captioning"](https://arxiv.org/abs/1612.00563) Rennie, Marcheret, Mroueh, Ross, Goel
>	"Recently it has been shown that policy-gradient methods for reinforcement learning can be utilized to train deep end-to-end systems directly on non-differentiable metrics for the task at hand. In this paper we consider the problem of optimizing image captioning systems using reinforcement learning, and show that by carefully optimizing our systems using the test metrics of the MSCOCO task, significant gains in performance can be realized. Our systems are built using a new optimization approach that we call self-critical sequence training. SCST is a form of the popular REINFORCE algorithm that, rather than estimating a “baseline” to normalize the rewards and reduce variance, utilizes the output of its own test-time inference algorithm to normalize the rewards it experiences. Using this approach, estimating the reward signal (as actor-critic methods must do) and estimating normalization (as REINFORCE algorithms typically do) is avoided, while at the same time harmonizing the model with respect to its test-time inference procedure. Empirically we find that directly optimizing the CIDEr metric with SCST and greedy decoding at test-time is highly effective. Our results on the MSCOCO evaluation sever establish a new state-of-the-art on the task, improving the best result in terms of CIDEr from 104.9 to 112.3."

  - `video` <https://youtube.com/watch?v=UnT5wTe13yc> (Rennie)
  - `video` <https://yadi.sk/i/-U5w4NpJ3H5TWD> (Ratnikov) `in russian`


#### ["Reward Augmented Maximum Likelihood for Neural Structured Prediction"](https://arxiv.org/abs/1609.00150) Norouzi, Bengio, Chen, Jaitly, Schuster, Wu, Schuurmans
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#reward-augmented-maximum-likelihood-for-neural-structured-prediction-norouzi-bengio-chen-jaitly-schuster-wu-schuurmans>


#### ["Style Transfer from Non-Parallel Text by Cross-Alignment"](https://arxiv.org/abs/1705.09655) Shen, Lei, Barzilay, Jaakkola
>	"This paper focuses on style transfer on the basis of non-parallel text. This is an instance of a broader family of problems including machine translation, decipherment, and sentiment modification. The key technical challenge is to separate the content from desired text characteristics such as sentiment. We leverage refined alignment of latent representations across mono-lingual text corpora with different characteristics. We deliberately modify encoded examples according to their characteristics, requiring the reproduced instances to match available examples with the altered characteristics as a population. We demonstrate the effectiveness of this cross-alignment method on three tasks: sentiment modification, decipherment of word substitution ciphers, and recovery of word order."



---
### interesting papers - cross-lingual tasks


#### ["Neural Machine Translation by Jointly Learning to Align and Translate"](http://arxiv.org/abs/1409.0473) Bahdanau, Cho, Bengio
>	"Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance. The models proposed recently for neural machine translation often belong to a family of encoder–decoders and encodes a source sentence into a fixed-length vector from which a decoder generates a translation. In this paper, we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder–decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly. With this new approach, we achieve a translation performance comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation. Furthermore, qualitative analysis reveals that the (soft-)alignments found by the model agree well with our intuition."

  - `video` <http://slideshot.epfl.ch/play/khnnunGF0elc> (Cho)
  - `video` <https://youtu.be/_XRBlhzb31U?t=42m26s> (Figurnov) `in russian`
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/BahdanauCB14>
  - `post` <http://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-with-gpus/>
  - `post` <http://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-2/>
  - `post` <http://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-3/>


#### ["Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"](https://arxiv.org/abs/1609.08144) Wu et al.
>	"Neural Machine Translation is an end-to-end learning approach for automated translation, with the potential to overcome many of the weaknesses of conventional phrase-based translation systems. Unfortunately, NMT systems are known to be computationally expensive both in training and in translation inference. Also, most NMT systems have difficulty with rare words. These issues have hindered NMT's use in practical deployments and services, where both accuracy and speed are essential. In this work, we present GNMT, Google's Neural Machine Translation system, which attempts to address many of these issues. Our model consists of a deep LSTM network with 8 encoder and 8 decoder layers using attention and residual connections. To improve parallelism and therefore decrease training time, our attention mechanism connects the bottom layer of the decoder to the top layer of the encoder. To accelerate the final translation speed, we employ low-precision arithmetic during inference computations. To improve handling of rare words, we divide words into a limited set of common sub-word units ("wordpieces") for both input and output. This method provides a good balance between the flexibility of "character"-delimited models and the efficiency of "word"-delimited models, naturally handles translation of rare words, and ultimately improves the overall accuracy of the system. Our beam search technique employs a length-normalization procedure and uses a coverage penalty, which encourages generation of an output sentence that is most likely to cover all the words in the source sentence. On the WMT'14 English-to-French and English-to-German benchmarks, GNMT achieves competitive results to state-of-the-art. Using a human side-by-side evaluation on a set of isolated simple sentences, it reduces translation errors by an average of 60% compared to Google's phrase-based production system."

  - <http://translate.google.com> (demo)
  - `notes` <http://smerity.com/articles/2016/google_nmt_arch.html>
  - `code` <https://github.com/google/seq2seq>


#### ["Bandit Structured Prediction for Neural Sequence-to-Sequence Learning"](https://arxiv.org/abs/1704.06497) Kreutzer, Sokolov, Riezler
>	"Bandit structured prediction describes a stochastic optimization framework where learning is performed from partial feedback. This feedback is received in the form of a task loss evaluation to a predicted output structure, without having access to gold standard structures. We advance this framework by lifting linear bandit learning to neural sequence-to-sequence learning problems using attention-based recurrent neural networks. Furthermore, we show how to incorporate control variates into our learning algorithms for variance reduction and improved generalization. We present an evaluation on a neural machine translation task that shows improvements of up to 5.89 BLEU points for domain adaptation from simulated bandit feedback."


#### ["Reinforcement Learning for Bandit Neural Machine Translation with Simulated Human Feedback"](https://arxiv.org/abs/1707.07402) Nguyen, Daume, Boyd-Graber
>	"Machine translation is a natural candidate problem for reinforcement learning from human feedback: users provide quick, dirty ratings on candidate translations to guide a system to improve. Yet, current neural machine translation training focuses on expensive human-generated reference translations. We describe a reinforcement learning algorithm that improves neural machine translation systems from simulated human feedback. Our algorithm combines the advantage actor-critic algorithm (Mnih et al., 2016) with the attention-based neural encoder-decoder architecture (Luong et al., 2015). This algorithm (a) is well-designed for problems with a large action space and delayed rewards, (b) effectively optimizes traditional corpus-level machine translation metrics, and (c) is robust to skewed, high-variance, granular feedback modeled after actual human behaviors."


#### ["Word Translation Without Parallel Data"](https://arxiv.org/abs/1710.04087) Conneau, Lample, Ranzato, Denoyer, Jegou
>	"State-of-the-art methods for learning cross-lingual word embeddings have relied on bilingual dictionaries or parallel corpora. Recent works showed that the need for parallel data supervision can be alleviated with character-level information. While these methods showed encouraging results, they are not on par with their supervised counterparts and are limited to pairs of languages sharing a common alphabet. In this work, we show that we can build a bilingual dictionary between two languages without using any parallel corpora, by aligning monolingual word embedding spaces in an unsupervised way. Without using any character information, our model even outperforms existing supervised methods on cross-lingual tasks for some language pairs. Our experiments demonstrate that our method works very well also for distant language pairs, like English-Russian or EnglishChinese. We finally show that our method is a first step towards fully unsupervised machine translation and describe experiments on the English-Esperanto language pair, on which there only exists a limited amount of parallel data."

>	"Our method leverages adversarial training to learn a linear mapping from a source to a target space and operates in two steps. First, in a two-player game, a discriminator is trained to distinguish between the mapped source embeddings and the target embeddings, while the mapping (which can be seen as a generator) is jointly trained to fool the discriminator. Second, we extract a synthetic dictionary from the resulting shared embedding space and fine-tune the mapping with the closed-form Procrustes solution."

>	"(A) There are two distributions of word embeddings, English words in red denoted by X and Italian words in blue denoted by Y, which we want to align/translate. Each dot represents a word in that space. The size of the dot is proportional to the frequency of the words in the training corpus of that language.  
>	(B) Using adversarial learning, we learn a rotation matrix W which roughly aligns the two distributions. The green stars are randomly selected words that are fed to the discriminator to determine whether the two word embeddings come from the same distribution.  
>	(C) The mapping W is further refined via Procrustes. This method uses frequent words aligned by the previous step as anchor points, and minimizes an energy function that corresponds to a spring system between anchor points. The refined mapping is then used to map all words in the dictionary.  
>	(D) Finally, we translate by using the mapping W and a distance metric, dubbed CSLS, that expands the space where there is high density of points (like the area around the word “cat”), so that “hubs” (like the word “cat”) become less close to other word vectors than they would otherwise (compare to the same region in panel (A))."  


#### ["Unsupervised Neural Machine Translation"](https://arxiv.org/abs/1710.11041) Artetxe, Labaka, Agirre, Cho
>	"In spite of the recent success of neural machine translation in standard benchmarks, the lack of large parallel corpora poses a major practical problem for many language pairs. There have been several proposals to alleviate this issue with, for instance, triangulation and semi-supervised learning techniques, but they still require a strong cross-lingual signal. In this work, we completely remove the need of parallel data and propose a novel method to train an NMT system in a completely unsupervised manner, relying on nothing but monolingual corpora. Our model builds upon the recent work on unsupervised embedding mappings, and consists of a slightly modified attentional encoder-decoder model that can be trained on monolingual corpora alone using a combination of denoising and backtranslation. Despite the simplicity of the approach, our system obtains 15.56 and 10.21 BLEU points in WMT 2014 French → English and German → English translation. The model can also profit from small parallel corpora, and attains 21.81 and 15.24 points when combined with 100,000 parallel sentences, respectively. Our approach is a breakthrough in unsupervised NMT, and opens exciting opportunities for future research."


#### ["Unsupervised Machine Translation Using Monolingual Corpora Only"](https://arxiv.org/abs/1711.00043) Lample, Denoyer, Ranzato
>	"Machine translation has recently achieved impressive performance thanks to recent advances in deep learning and the availability of large-scale parallel corpora. There have been numerous attempts to extend these successes to low-resource language pairs, yet requiring tens of thousands of parallel sentences. In this work, we take this research direction to the extreme and investigate whether it is possible to learn to translate even without any parallel data. We propose a model that takes sentences from monolingual corpora in two different languages and maps them into the same latent space. By learning to reconstruct in both languages from this shared feature space, the model effectively learns to translate without using any labeled data. We demonstrate our model on two widely used datasets and two language pairs, reporting BLEU scores up to 32.8, without using even a single parallel sentence at training time."


#### ["Adversarial Deep Averaging Networks for Cross-Lingual Sentiment Classification"](http://arxiv.org/abs/1606.01614) Chen, Athiwaratkun, Sun, Weinberger, Cardie
>	"In recent years deep neural networks have achieved great success in sentiment classification for English, thanks in part to the availability of copious annotated resources. Unfortunately, most other languages do not enjoy such an abundance of annotated data for sentiment analysis. To combat this problem, we propose the Adversarial Deep Averaging Network to transfer sentiment knowledge learned from labeled English data to low-resource languages where only unlabeled data exists. ADAN is a "Y-shaped" network with two discriminative branches: a sentiment classifier and an adversarial language predictor. Both branches take input from a feature extractor that aims to learn hidden representations that capture the underlying sentiment of the text and are invariant across languages. Experiments on Chinese sentiment classification demonstrate that ADAN significantly outperforms several baselines, including a strong pipeline approach that relies on Google Translate, the state-of-the-art commercial machine translation system."

>	"In this work, we propose an end-to-end neural network model that only requires labeled English data and unlabeled Chinese text as input, and explicitly transfers the knowledge learned on English sentiment analysis to Chinese. Our trained system directly operates on Chinese sentences to predict their sentiment (e.g. positive or negative). We hypothesize that an ideal model for cross-lingual sentiment analysis should learn features that both perform well on the English sentiment classification task, and are invariant with respect to the shift in language. Therefore, ADAN simultaneously optimizes two components: i) a sentiment classifier P for English; and ii) an adversarial language predictor Q that tries to predict whether a sentence x is from English or Chinese. The two classifiers take input from the jointly learned feature extractor F, which is trained to maximize accuracy on English sentiment analysis and simultaneously to minimize the language predictor’s chance of correctly predicting the language of the text. This is why the language predictor Q is called “adversarial”. The model is exposed to both English and Chinese sentences during training, but only the labeled English sentences pass through the sentiment classifier. The feature extractor and the sentiment classifier are then used for Chinese sentences at test time. In this manner, we can train the system with massive amounts of unlabeled text in Chinese. Upon convergence, the joint features (output of F) are thus encouraged to be both discriminative for sentiment analysis and invariant across languages."



---
### interesting papers - language models


#### ["A Neural Probabilistic Language Model"](http://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) Bengio, Ducharme, Vincent, Jauvin
>	"A goal of statistical language modeling is to learn the joint probability function of sequences of words in a language. This is intrinsically difficult because of the curse of dimensionality: a word sequence on which the model will be tested is likely to be different from all the word sequences seen during training. Traditional but very successful approaches based on n-grams obtain generalization by concatenating very short overlapping sequences seen in the training set. We propose to fight the curse of dimensionality by learning a distributed representation for words which allows each training sentence to inform the model about an exponential number of semantically neighboring sentences. The model learns simultaneously a distributed representation for each word along with the probability function for word sequences, expressed in terms of these representations. Generalization is obtained because a sequence of words that has never been seen before gets high probability if it is made of words that are similar (in the sense of having a nearby representation) to words forming an already seen sentence. Training such large models (with millions of parameters) within a reasonable time is itself a signiﬁcant challenge. We report on experiments using neural networks for the probability function, showing on two text corpora that the proposed approach signiﬁcantly improves on state-of-the-art n-gram models, and that the proposed approach allows to take advantage of longer contexts."


#### ["Strategies for Training Large Vocabulary Neural Language Models"](http://arxiv.org/abs/1512.04906) Chen, Grangier, Auli
>	"Training neural network language models over large vocabularies is still computationally very costly compared to count-based models such as Kneser-Ney. At the same time, neural language models are gaining popularity for many applications such as speech recognition and machine translation whose success depends on scalability. We present a systematic comparison of strategies to represent and train large vocabularies, including softmax, hierarchical softmax, target sampling, noise contrastive estimation and self normalization. We further extend self normalization to be a proper estimator of likelihood and introduce an efficient variant of softmax. We evaluate each method on three popular benchmarks, examining performance on rare words, the speed/accuracy trade-off and complementarity to Kneser-Ney."

>	"This paper presents the first comprehensive analysis of strategies to train large vocabulary neural language models. Large vocabularies are a challenge for neural networks as they need to compute the partition function over the entire vocabulary at each evaluation. We compared classical softmax to hierarchical softmax, target sampling, noise contrastive estimation and infrequent normalization, commonly referred to as self-normalization. Furthermore, we extend infrequent normalization, or self-normalization, to be a proper estimator of likelihood and we introduce differentiated softmax, a novel variant of softmax which assigns less capacity to rare words in order to reduce computation. Our results show that methods which are effective on small vocabularies are not necessarily the best on large vocabularies. In our setting, target sampling and noise contrastive estimation failed to outperform the softmax baseline. Overall, differentiated softmax and hierarchical softmax are the best strategies for large vocabularies. Compared to classical Kneser-Ney models, neural models are better at modeling frequent words, but they are less effective for rare words. A combination of the two is therefore very effective. From this paper, we conclude that there is still a lot to explore in training from a combination of normalized and unnormalized objectives. We also see parallel training and better rare word modeling as promising future directions."


#### ["Recurrent Neural Network Based Language Model"](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf) Mikolov, Karafiat, Burget, Cernocky, Khudanpur
>	"A new recurrent neural network based language model with applications to speech recognition is presented. Results indicate that it is possible to obtain around 50% reduction of perplexity by using mixture of several RNN LMs, compared to a state of the art backoff language model. Speech recognition experiments show around 18% reduction of word error rate on the Wall Street Journal task when comparing models trained on the same amount of data, and around 5% on the much harder NIST RT05 task, even when the backoff model is trained on much more data than the RNN LM. We provide ample empirical evidence to suggest that connectionist language models are superior to standard n-gram techniques, except their high computational (training) complexity."

  - <http://deeplearning.net/tutorial/rnnslu.html>


#### ["Scaling Recurrent Neural Network Language Models"](http://arxiv.org/abs/1502.00512) Williams, Prasad, Mrva, Ash, Robinson
>	"This paper investigates the scaling properties of Recurrent Neural Network Language Models. We discuss how to train very large RNNs on GPUs and address the questions of how RNNLMs scale with respect to model size, training-set size, computational costs and memory. Our analysis shows that despite being more costly to train, RNNLMs obtain much lower perplexities on standard benchmarks than n-gram models. We train the largest known RNNs and present relative word error rates gains of 18% on an ASR task. We also present the new lowest perplexities on the recently released billion word language modelling benchmark, 1 BLEU point gain on machine translation and a 17% relative hit rate gain in word prediction."

>	"We have shown that large RNNLMs can be trained efficiently on GPUs by exploiting data parallelism and minimising the number of extra parameters required during training. Such RNNs reduce the perplexity on standard benchmarks by over 40% against 5-grams, whilst using a fraction of the parameters. We believe RNNs now offer a lower perplexity than 5-grams for any amount of training data. In addition, we showed that state of the art ASR systems can be trained with Kaldi and high-end GPUs by rescoring with an RNNLM. Despite being both compute and GPU memory bound, RNNLMs are comfortably ahead of n-grams at present. We believe that future developments in compute power and memory capacity will further favour them."


#### ["LSTM Neural Networks for Language Modeling"](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.248.4448) Sundermeyer, Schluter, Ney
>	"Neural networks have become increasingly popular for the task of language modeling. Whereas feed-forward networks only exploit a fixed context length to predict the next word of a sequence, conceptually, standard recurrent neural networks can take into account all of the predecessor words. On the other hand, it is well known that recurrent networks are difficult to train and therefore are unlikely to show the full potential of recurrent models. These problems are addressed by a the Long Short-Term Memory neural network architecture. In this work, we analyze this type of network on an English and a large French language modeling task. Experiments show improvements of about 8% relative in perplexity over standard recurrent neural network LMs. In addition, we gain considerable improvements in WER on top of a state-of-the-art speech recognition system."


#### ["Character-Aware Neural Language Models"](http://arxiv.org/abs/1508.06615) Kim, Jernite, Sontag, Rush
>	"We describe a simple neural language model that relies only on character-level inputs. Predictions are still made at the word-level. Our model employs a convolutional neural network over characters, whose output is given to a long short-term memory recurrent neural network language model. On the English Penn Treebank the model is on par with the existing state-of-the-art despite having 60% fewer parameters. On languages with rich morphology (Czech, German, French, Spanish, Russian), the model consistently outperforms a Kneser-Ney baseline (by 30–35%) and a word-level LSTM baseline (by 15–25%), again with far fewer parameters. Our results suggest that on many languages, character inputs are sufficient for language modeling."

>	"Analysis of word representations obtained from the character composition part of the model further indicates that the model is able to encode semantically meaningful features that are not immediately apparent from orthography alone. Our work questions the necessity of word embeddings as inputs for neural language modeling. Insofar as language modeling mostly relies on capturing a word’s syntactic role, it would be interesting to see if the architecture introduced in this paper is viable for more semantic tasks - for example, as an encoder/decoder in neural machine translation."

  - `video` <http://research.microsoft.com/apps/video/default.aspx?id=260041> (Sontag)
  - <https://www.evernote.com/shard/s267/sh/64195d10-53b4-4312-8c5a-d10ab5138c36/22ce804ec6c8ab2b9ceb3096b8cd929e>
  - <https://github.com/yoonkim/lstm-char-cnn>
  - <https://github.com/carpedm20/lstm-char-cnn-tensorflow>


#### ["Neural Language Correction with Character-Based Attention"](http://arxiv.org/abs/1603.09727) Xie, Avati, Arivazhagan, Jurafsky, Ng
>	"Natural language correction has the potential to help language learners improve their writing skills. While approaches with separate classifiers for different error types have high precision, they do not flexibly handle errors such as redundancy or non-idiomatic phrasing. On the other hand, word and phrase-based machine translation methods are not designed to cope with orthographic errors, and have recently been outpaced by neural models. Motivated by these issues, we present a neural network-based approach to language correction. The core component of our method is an encoder-decoder recurrent neural network with an attention mechanism. By operating at the character level, the network avoids the problem of out-of-vocabulary words. We illustrate the flexibility of our approach on dataset of noisy, user-generated text collected from an English learner forum. When combined with a language model, our method achieves a state-of-the-art F0.5-score on the CoNLL 2014 Shared Task. We further demonstrate that training the network on additional data with synthesized errors can improve performance."


#### ["Contextual LSTM (CLSTM) Ghosh, Vinyals, Strope, Roy, Dean, Heck Models for Large Scale NLP Tasks"](http://arxiv.org/abs/1602.06291)
>	"Documents exhibit sequential structure at multiple levels of abstraction (e.g., sentences, paragraphs, sections). These abstractions constitute a natural hierarchy for representing the context in which to infer the meaning of words and larger fragments of text. In this paper, we present CLSTM (Contextual LSTM), an extension of the recurrent neural network LSTM (Long-Short Term Memory) model, where we incorporate contextual features (e.g., topics) into the model. We evaluate CLSTM on three specific NLP tasks: word prediction, next sentence selection, and sentence topic prediction. Results from experiments run on two corpora, English documents in Wikipedia and a subset of articles from a recent snapshot of English Google News, indicate that using both words and topics as features improves performance of the CLSTM models over baseline LSTM models for these tasks. For example on the next sentence selection task, we get relative accuracy improvements of 21% for the Wikipedia dataset and 18% for the Google News dataset. This clearly demonstrates the significant benefit of using context appropriately in natural language tasks. This has implications for a wide variety of natural language applications like question answering, sentence completion, paraphrase generation, and next utterance prediction in dialog systems."



---
### interesting papers - word embeddings


#### ["Are Distributional Representations Ready for the Real World? Evaluating Word Vectors for Grounded Perceptual Meaning"](https://arxiv.org/abs/1705.11168) Lucy, Gauthier
>	"Distributional word representation methods exploit word co-occurrences to build compact vector encodings of words. While these representations enjoy widespread use in modern natural language processing, it is unclear whether they accurately encode all necessary facets of conceptual meaning. In this paper, we evaluate how well these representations can predict perceptual and conceptual features of concrete concepts, drawing on two semantic norm datasets sourced from human participants. We find that several standard word representations fail to encode many salient perceptual features of concepts, and show that these deficits correlate with word-word similarity prediction errors. Our analyses provide motivation for grounded and embodied language learning approaches, which may help to remedy these deficits."


#### ["Efficient Estimation of Word Representations in Vector Space"](http://arxiv.org/abs/1301.3781) Mikolov, Chen, Corrado, Dean (word2vec)
>	"We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities."

  - <http://www.shortscience.org/paper?bibtexKey=mikolov2013efficient>
  - <http://alexminnaar.com/word2vec-tutorial-part-i-the-skip-gram-model.html>
  - <http://alexminnaar.com/word2vec-tutorial-part-ii-the-continuous-bag-of-words-model.html>
  - <http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf>
  - <http://arxiv.org/abs/1402.3722>
  - `video` <http://youtube.com/watch?v=fwcJpSYNsNs> (Mikolov)


#### ["word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method"](http://arxiv.org/abs/1402.3722) Goldberg, Levy
>	"The word2vec software of Tomas Mikolov and colleagues has gained a lot of traction lately, and provides state-of-the-art word embeddings. The learning models behind the software are described in two research papers. We found the description of the models in these papers to be somewhat cryptic and hard to follow. While the motivations and presentation may be obvious to the neural-networks language-modeling crowd, we had to struggle quite a bit to figure out the rationale behind the equations. This note is an attempt to explain equation (4) (negative sampling)."


#### ["word2vec Parameter Learning Explained"](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf) Rong
>	"The word2vec model and application by Mikolov et al. have attracted a great amount of attention in recent two years. The vector representations of words learned by word2vec models have been proven to be able to carry semantic meanings and are useful in various NLP tasks. As an increasing number of researchers would like to experiment with word2vec, I notice that there lacks a material that comprehensively explains the parameter learning process of word2vec in details, thus preventing many people with less neural network experience from understanding how exactly word2vec works. This note provides detailed derivations and explanations of the parameter update equations for the word2vec models, including the original continuous bag-of-word and skip-gram models, as well as advanced tricks, hierarchical soft-max and negative sampling. In the appendix a review is given on the basics of neuron network models and backpropagation."


#### ["GloVe: Global Vectors for Word Representation"](http://nlp.stanford.edu/pubs/glove.pdf) Pennington, Socher, Manning
>	"Recent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using vector arithmetic, but the origin of this regularities has remained opaque. We analyze and make explicit the model properties needed for such regularities to emerge in word vectors. The result is a new global log-bilinear regression model that combines the advantages of the two major model families in the literature: global matrix factorization and local context window methods. Our model efficiently leverages statistical information by training only on the nonzero elements in a word-word co-occurence matrix, rather than on the entire sparse matrix or on inidividual context windows in a large corpus. On a recent word analogy task our model obtains 75% accuracy, an improvement of 11% over word2vec. It also outperforms related word vector models on similarity tasks and named entity recognition."

>	"The paper starts with a simple observation that relationships between words can be discovered by ratios of co-occurrence statistics. The example they give is with the words "ice" and "steam". We can represent each word by the counts of other words it appears with (a traditional "count" model). When we want to know what relation "ice" has with "steam", we can look at the ratios of their co-occurrence statistics with other words. Both of them will appear frequently with "water", and infrequently with unrelated words, like "fashion". "Ice" will appear more with "solid", and less with "gas". If we look at the ratios of these statistics, both "water" and "fashion" will have a value close to 1 (because both "ice" and "steam" occur with those words with similar frequency), and the words that express the relationship between "ice" and "steam" will have values that are much higher or lower than 1 ("solid" and "gas", in this example). It's quite a nice insight. And further, ratios of co-occurrence statistics translate to vector differences in log space. So if your model looks at log probabilities of co-occurrence, vector addition should be sufficient to recover analogies, as done in the skip-gram papers. The authors go on to construct a "count" model based on these ideas, and show that it outperforms skip-gram."

----
>	"GloVe translates meaningful relationships between word-word cooccurrence counts into linear relations in the word vector space. GloVe shows the connection between global counting and local prediction models - appropriate scaling of counts gives global counting models the properties and performance of prediction models."

----
>	"GloVe is an approach that unlike word2vec works from the precomputed corpus co-occurrence statistics. The authors posit several constraints that should lead to preserving the “linear directions of meaning”. Based on ratios of conditional probabilities of words in context, they suggest that a natural model for learning such linear structure should minimize the following cost function for a given focus word i and context word j: ... Here, bi and bj are bias terms that are specific to each focus word and each context word, respectively. Using stochastic gradient descent, GloVe learns the model parameters for W, b, W ̃ and b ̃: it selects a pair of words observed to co-occur in the corpus, retrieves the corresponding embedding parameters, computes the loss, and back-propagates the error to update the parameters. GloVe therefore requires training time proportional to the number of observed co-occurrence pairs, allowing it to scale independently of corpus size."

>	"Although GloVe was developed independently from Skip-Gram Negative Sampling (and, as far as we know, without knowledge of Levy and Goldberg’s 2014 analysis), it is interesting how similar these two models are.
>	- Both seek to minimize the difference between the model’s estimate and the log of the co-occurrence count. GloVe has additional free “bias” parameters that, in SGNS, are pegged to the corpus frequency of the individual words. Empirically, it can be observed that the bias terms are highly correlated to the frequency of the row and column features in a trained GloVe model.
>	- Both weight the loss according to the frequency of the co-occurrence count such that frequent co-occurrences incur greater penalty than rare ones
>	Levy et al. (2015) note these algorithmic similarities. In their controlled empirical comparison of several different embedding approaches, results produced by SGNS and GloVe differ only modestly.
>	There are subtle differences, however. The negative sampling regime of SGNS ensures that the model does not place features near to one another in the embedding space whose co-occurrence isn’t observed in the corpus. This is distinctly different from GloVe, which trains only on the observed co-occurrence statistics. The GloVe model incurs no penalty for placing features near to one another whose co-occurrence has not been observed. This can result in poor estimates for uncommon features."

----
>	"In some sense step back: word2vec counts co-occurrences and does dimensionality reduction together, GloVe is two-pass algorithm."

  - <http://nlp.stanford.edu/projects/glove/>
  - `video` <http://youtube.com/watch?v=RyTpzZQrHCs> (Pennington)
  - <http://www.shortscience.org/paper?bibtexKey=conf/emnlp/PenningtonSM14#shagunsodhani>
  - <https://github.com/shashankg7/glove-theano>
  - <https://github.com/GradySimon/tensorflow-glove>


#### ["Swivel: Improving Embeddings by Noticing What’s Missing"](http://arxiv.org/abs/1602.02215) Shazeer, Doherty, Evans, Waterson
>	"We present Submatrix-wise Vector Embedding Learner (Swivel), a method for generating low-dimensional feature embeddings from a feature co-occurrence matrix. Swivel performs approximate factorization of the point-wise mutual information matrix via stochastic gradient descent. It uses a piecewise loss with special handling for unobserved co-occurrences, and thus makes use of all the information in the matrix. While this requires computation proportional to the size of the entire matrix, we make use of vectorized multiplication to process thousands of rows and columns at once to compute millions of predicted values. Furthermore, we partition the matrix into shards in order to parallelize the computation across many nodes. This approach results in more accurate embeddings than can be achieved with methods that consider only observed co-occurrences, and can scale to much larger corpora than can be handled with sampling methods."

>	"Swivel produces low-dimensional feature embeddings from a co-occurrence matrix. It optimizes an objective that is very similar to that of SGNS and GloVe: the dot product of a word embedding with a context embedding ought to approximate the observed PMI of the two words in the corpus. Unlike Skip-Gram Negative Sampling, Swivel’s computational requirements depend on the size of the co-occurrence matrix, rather than the size of the corpus. This means that it can be applied to much larger corpora. Unlike GloVe, Swivel explicitly considers all the co-occurrence information - including unobserved co-occurrences - to produce embeddings. In the case of unobserved co-occurrences, a “soft hinge” loss prevents the model from over-estimating PMI. This leads to demonstrably better embeddings for rare features without sacrificing quality for common ones. Swivel capitalizes on vectorized hardware, and uses block structure to amortize parameter transfer cost and avoid contention. This results in the ability to handle very large co-occurrence matrices in a scalable way that is easy to parallelize."

>	"Due to the fact that Skip-Gram Negative Sampling slides a sampling window through the entire training corpus, a significant drawback of the algorithm is that it requires training time proportional to the size of the corpus."

  - <https://github.com/tensorflow/models/tree/master/research/swivel>


#### ["Linguistic Regularities in Sparse and Explicit Word Representations"](https://levyomer.files.wordpress.com/2014/04/linguistic-regularities-in-sparse-and-explicit-word-representations-conll-2014.pdf) Levy, Goldberg
>	"Recent work has shown that neural-embedded word representations capture many relational similarities, which can be recovered by means of vector arithmetic in the embedded space. We show that Mikolov et al.’s method of first adding and subtracting word vectors, and then searching for a word similar to the result, is equivalent to searching for a word that maximizes a linear combination of three pairwise word similarities. Based on this observation, we suggest an improved method of recovering relational similarities, improving the state-of-the-art results on two recent word-analogy datasets. Moreover, we demonstrate that analogy recovery is not restricted to neural word embeddings, and that a similar amount of relational similarities can be recovered from traditional distributional word representations."

>	"Paper tries to give some intuition for why the skip-gram models work on analogy tasks - it seems surprising that they should, as you wouldn't a priori expect that relationships between words are consistently encoded as linear translations in the vector space learned by these models. Levy and Goldberg show that these relationships still exist in what they call the "explicit" vector space (also called "count" models), so they are not a product of the neural network embedding, just preserved by it. But it turns out that you need to slightly change the similarity equation to match the performance of skip-gram with an explicit vector space - instead of adding together the similarities of the words in the analogy, you have to multiply them."

>	"This fascinating result raises a question: to what extent are the relational semantic properties a result of the embedding process? Experiments show that the RNN-based embeddings are superior to other dense representations, but how crucial is it for a representation to be dense and low-dimensional at all?"


#### ["Neural Word Embedding as Implicit Matrix Factorization"](http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf) Levy, Goldberg
>	"We analyze skip-gram with negative-sampling, a word embedding method introduced by Mikolov et al., and show that it is implicitly factorizing a word-context matrix, whose cells are the pointwise mutual information of the respective word and context pairs (shifted by a global constant). We find that another embedding method, NCE, is implicitly factorizing a similar matrix, where each cell is the (shifted) log conditional probability of a word given its context. We show that using a sparse Shifted Positive PMI word-context matrix to represent words improves results on two word similarity tasks and one of two analogy tasks. When dense, low-dimensional vectors are preferred, exact factorization with SVD can achieve solutions that are at least as good as SGNS’s solutions for word similarity tasks. On analogy questions SGNS remains superior to SVD. We conjecture that this stems from the weighted nature of SGNS’s factorization."

>	"It looks at the objective function optimized by skip-gram, and shows that it is implicitly factoring a (shifted) PMI matrix. That's a really interesting (and non-obvious) connection. They further show that they can optimize the objective directly, either by just constructing the shifted PMI matrix, or by using a truncated (and thus sparse) version of that. And, instead of implicitly factorizing this shifted matrix, once you know what skip-gram is doing, you can just directly factor the matrix yourself using SVD. They show that for several tasks (all that they tested except for syntactic analogies), these more direct techniques outperform skip-gram."

>	"- with proper tuning, traditional distributional similarity methods can be very competitive with word2vec"
>	"- by analyzing word2vec we found a novel variation on the PMI association measure which is kind-of ugly but works surprisingly well"
>	"- we tried and tried, but couldn't get GloVe to outperform word2vec. Their w+c idea is neat and works very well, though"

>	"Skip-Gram Negative Sampling can be seen as producing two matrices, W for focus words and W ̃ for context words, such that their product WW ̃ approximates the observed PMI between respective word/context pairs. Given a specific focus word i and context word j, SGNS minimizes the magnitude of the difference between wiT*w ̃j and pmi(i; j), tempered by a monotonically increasing weighting function of the observed co-occurrence count."

  - <https://minhlab.wordpress.com/2015/06/08/a-new-proof-for-the-equivalence-of-word2vec-skip-gram-and-shifted-ppmi/>
  - <https://building-babylon.net/2016/05/12/skipgram-isnt-matrix-factorisation/>
  - <http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/>


#### ["Improving Distributional Similarity with Lessons Learned from Word Embeddings"](https://levyomer.files.wordpress.com/2015/03/improving-distributional-similarity-tacl-2015.pdf) Levy, Goldberg, Dagan (comparison of word2vec and GloVe)
>	"Recent trends suggest that neural-network-inspired word embedding models outperform traditional count-based distributional models on word similarity and analogy detection tasks. We reveal that much of the performance gains of word embeddings are due to certain system design choices and hyperparameter optimizations, rather than the embedding algorithms themselves. Furthermore, we show that these modifications can be transferred to traditional distributional models, yielding similar gains. In contrast to prior reports, we observe mostly local or insignificant performance differences between the methods, with no global advantage to any single approach over the others."

>	"Recent embedding methods introduce a plethora of design choices beyond network architecture and optimization algorithms. We reveal that these seemingly minor variations can have a large impact on the success of word representation methods. By showing how to adapt and tune these hyperparameters in traditional methods, we allow a proper comparison between representations, and challenge various claims of superiority from the word embedding literature. This study also exposes the need for more controlled-variable experiments, and extending the concept of “variable” from the obvious task, data, and method to the often ignored preprocessing steps and hyperparameter settings."

  - <http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/>
  - <http://bitbucket.org/omerlevy/hyperwords>


#### ["Do Supervised Distributional Methods Really Learn Lexical Inference Relations"](https://levyomer.files.wordpress.com/2015/03/do-supervised-distributional-models-naacl-2015.pdf) Levy, Remus, Biemann, Dagan
	"Distributional representations of words have been recently used in supervised settings for recognizing lexical inference relations between word pairs, such as hypernymy and entailment. We investigate a collection of these state-of-the-art methods, and show that they do not actually learn a relation between two words. Instead, they learn an independent property of a single word in the pair: whether that word is a “prototypical hypernym”.
>	"In this work, we showed that state-of-the-art supervised methods for recognizing lexical inference appear to be learning whether y is a prototypical hypernym, regardless of its relation with x. We tried to factor in the similarity between x and y, yet observed only marginal improvements. While more sophisticated methods might be able to extract the necessary relational information from contextual features alone, it is also possible that this information simply does not exist in those features."

  - `video` <http://techtalks.tv/talks/do-supervised-distributional-methods-really-learn-lexical-inference-relations/61506/>


#### ["RAND-WALK: A Latent Variable Model Approach to Word Embeddings"](http://arxiv.org/abs/1502.03520) Arora, Li, Liang, Ma, Risteski
>	"Semantic word embeddings represent the meaning of a word via a vector, and are created by diverse methods. Many use nonlinear operations on co-occurrence statistics, and have hand-tuned hyperparameters and reweighting methods. This paper proposes a new generative model, a dynamic version of the log-linear topic model of Mnih and Hinton (2007). The methodological novelty is to use the prior to compute closed form expressions for word statistics. This provides a theoretical justification for nonlinear models like PMI, word2vec, and GloVe, as well as some hyperparameter choices. It also helps explain why lowdimensional semantic embeddings contain linear algebraic structure that allows solution of word analogies, as shown by Mikolov et al. (2013a) and many subsequent papers. Experimental support is provided for the generative model assumptions, the most important of which is that latent word vectors are fairly uniformly dispersed in space."

  - `video` <http://youtube.com/watch?v=gaVR3WnczOQ> (Ma)
  - `video` <http://youtube.com/watch?v=KR46z_V0BVw> (Arora)
  - `video` <http://youtube.com/watch?v=BCsOrewkmH4> (Liang)
  - <http://www.offconvex.org/2016/02/14/word-embeddings-2/> + <http://www.offconvex.org/2015/12/12/word-embeddings-1/>
  - <https://akshayka.github.io/papers/html/arora2016pmi-embeddings.html>
  - <https://github.com/YingyuLiang/SemanticVector>



---
### interesting papers - word sense disambiguation


#### ["Improving Word Representations via Global Context and Multiple Word Prototypes"](http://aclweb.org/anthology/P12-1092) Huang, Socher, Manning, Ng
>	"Unsupervised word representations are very useful in NLP tasks both as inputs to learning algorithms and as extra word features in NLP systems. However, most of these models are built with only local context and one representation per word. This is problematic because words are often polysemous and global context can also provide useful information for learning word meanings. We present a new neural network architecture which 1) learns word embeddings that better capture the semantics of words by incorporating both local and global document context, and 2) accounts for homonymy and polysemy by learning multiple embeddings per word. We introduce a new dataset with human judgments on pairs of words in sentential context, and evaluate our model on it, showing that our model outperforms competitive baselines and other neural language models."


#### ["Efficient Non-parametric Estimation of Multiple Embeddings per Word in Vector Space"](https://people.cs.umass.edu/~arvind/emnlp2014.pdf) Neelakantan, Shankar, Passos, McCallum
>	"There is rising interest in vector-space word embeddings and their use in NLP, especially given recent methods for their fast estimation at very large scale. Nearly all this work, however, assumes a single vector per word type - ignoring polysemy and thus jeopardizing their usefulness for downstream tasks. We present an extension to the Skip-gram model that efficiently learns multiple embeddings per word type. It differs from recent related work by jointly performing word sense discrimination and embedding learning, by non-parametrically estimating the number of senses per word type, and by its efficiency and scalability. We present new state-of-the-art results in the word similarity in context task and demonstrate its scalability by training with one machine on a corpus of nearly 1 billion tokens in less than 6 hours."

  - `video` <http://youtube.com/watch?v=EeBj4TyW8B8>
  - <https://people.cs.umass.edu/~arvind/emnlp2014wordvectors/>


#### ["A Probabilistic Model for Learning Multi-Prototype Word Embeddings"](http://research.microsoft.com/apps/pubs/default.aspx?id=226629) Tian et al.
>	"Distributed word representations have been widely used and proven to be useful in quite a few natural language processing and text mining tasks. Most of existing word embedding models aim at generating only one embedding vector for each individual word, which, however, limits their effectiveness because huge amounts of words are polysemous (such as bank and star). To address this problem, it is necessary to build multi embedding vectors to represent different meanings of a word respectively. Some recent studies attempted to train multi-prototype word embeddings through clustering context window features of the word. However, due to a large number of parameters to train, these methods yield limited scalability and are inefficient to be trained with big data. In this paper, we introduce a much more efficient method for learning multi embedding vectors for polysemous words. In particular, we first propose to model word polysemy from a probabilistic perspective and integrate it with the highly efficient continuous Skip-Gram model. Under this framework, we design an Expectation-Maximization algorithm to learn the word’s multi embedding vectors. With much less parameters to train, our model can achieve comparable or even better results on word-similarity tasks compared with conventional methods."

>	"In this paper, we introduce a fast and probabilistic method to generate multiple embedding vectors for polysemous words, based on the continuous Skip-Gram model. On one hand, our method addresses the drawbacks of the original Word2Vec model by leveraging multi-prototype word embeddings; on the other hand, our model yields much less complexity without performance loss compared with the former clustering based multi-prototype algorithms. In addition, the probabilistic framework of our method avoids the extra efforts to perform clustering besides training word embeddings."


#### ["Breaking Sticks and Ambiguities with Adaptive Skip-gram"](http://arxiv.org/abs/1502.07257) Bartunov, Kondrashkin, Osokin, Vetrov
>	"Recently proposed Skip-gram model is a powerful method for learning high-dimensional word representations that capture rich semantic relationships between words. However, Skip-gram as well as most prior work on learning word representations does not take into account word ambiguity and maintain only single representation per word. Although a number of Skip-gram modifications were proposed to overcome this limitation and learn multi-prototype word representations, they either require a known number of word meanings or learn them using greedy heuristic approaches. In this paper we propose the Adaptive Skip-gram model which is a nonparametric Bayesian extension of Skip-gram capable to automatically learn the required number of representations for all words at desired semantic resolution. We derive efficient online variational learning algorithm for the model and empirically demonstrate its efficiency on wordsense induction task."

  - `video` <http://youtube.com/watch?v=vYbee1InliU> (Vetrov)
  - <http://postnauka.ru/video/49258> (Vetrov) `in russian`
  - `video` <http://youtu.be/uoRwjxaDgt0?t=33m58s> (Vetrov) `in russian`
  - <https://github.com/sbos/AdaGram.jl>


#### ["Infinite Dimensional Word Embeddings"](http://arxiv.org/abs/1511.05392) Nalisnick, Ravi
>	"We describe a method for learning word embeddings with stochastic dimensionality. Our Infinite Skip-Gram model specifies an energy-based joint distribution over a word vector, a context vector, and their dimensionality, which can be defined over a countably infinite domain by employing the same techniques used to make the Infinite Restricted Boltzmann Machine tractable. We find that the distribution over embedding dimensionality for a given word is highly interpretable and leads to an elegant probabilistic mechanism for word sense induction. We show qualitatively and quantitatively that the iSG produces parameter-efficient representations that are robust to language’s inherent ambiguity."

>	"We’ve proposed a novel word embedding model called Infinite Skip-Gram that defines vector dimensionality as a random variable with a countably infinite domain. Training via the generalized EM framework allows embeddings to grow as the data requires. This property is especially well suited for learning representations of homographs, which Skip-Gram notably fails at. A unique quality of the iSG is that it is highly interpretable (in comparison to other embedding methods) due to its ability to produce a distribution over the dimensionality of a given word (p(z|w)). Plots of p(z|w) concisely show how specific/vague a word is and its various senses just from the mode landscape."

>	"During training, the iSGM allows word representations to grow naturally based on how well they can predict their context. This behavior enables the vectors of specific words to use few dimensions and the vectors of vague words to elongate as needed. Manual and experimental analysis reveals this dynamic representation elegantly captures specificity, polysemy, and homonymy without explicit definition of such concepts within the model."

  - <https://www.evernote.com/shard/s189/sh/2da41f5c-7fc2-4bb1-8c00-dad613404328/e1d1c853162af5fbe537a02796a4ba4e>
  - <http://dustintran.com/blog/infinite-dimensional-word-embeddings/>
  - `video` <http://videolectures.net/deeplearning2016_cote_boltzmann_machine/> (Cote)


#### ["Sense2Vec - A Fast and Accurate Method for Word Sense Disambiguation in Neural Word Embeddings"](http://arxiv.org/abs/1511.06388) Trask, Michalak, Liu
>	"Neural word representations have proven useful in Natural Language Processing tasks due to their ability to efficiently model complex semantic and syntactic word relationships. However, most techniques model only one representation per word, despite the fact that a single word can have multiple meanings or ”senses”. Some techniques model words by using multiple vectors that are clustered based on context. However, recent neural approaches rarely focus on the application to a consuming NLP algorithm. Furthermore, the training process of recent word-sense models is expensive relative to single-sense embedding processes. This paper presents a novel approach which addresses these concerns by modeling multiple embeddings for each word based on supervised disambiguation, which provides a fast and accurate way for a consuming NLP model to select a sense-disambiguated embedding. We demonstrate that these embeddings can disambiguate both contrastive senses such as nominal and verbal senses as well as nuanced senses such as sarcasm. We further evaluate Part-of-Speech disambiguated embeddings on neural dependency parsing, yielding a greater than 8% average error reduction in unlabeled attachment scores across 6 languages."


#### ["Word Sense Disambiguation with Neural Language Models"](http://arxiv.org/abs/1603.07012) Yuan, Doherty, Richardson, Evans, Altendorf
>	"Determining the intended sense of words in text - word sense disambiguation - is a long-standing problem in natural language processing. In this paper, we present WSD algorithms which use neural network language models to achieve state-of-the-art precision. Each of these methods learns to disambiguate word senses using only a set of word senses, a few example sentences for each sense taken from a licensed lexicon, and a large unlabeled text corpus. We classify based on cosine similarity of vectors derived from the contexts in unlabeled query and labeled example sentences. We demonstrate state-of-the-art results when using the WordNet sense inventory, and significantly better than baseline performance using the New Oxford American Dictionary inventory. The best performance was achieved by combining an LSTM language model with graph label propagation."

>	"Our experiments show that using the LSTM language model achieves significantly higher precision than the CBOW language model, especially on verbs and adverbs. This suggests that sequential order information is important to discriminating senses of verbs and adverbs. The best performance was achieved by using an LSTM language model with label propagation. Our algorithm outperforms the baseline by more than 10% (0.87 vs. 0.75)."


#### ["Word Representations via Gaussian Embedding"](http://arxiv.org/abs/1412.6623) Vilnis, McCallum
>	"Current work in lexical distributed representations maps each word to a point vector in low-dimensional space. Mapping instead to a density provides many interesting advantages, including better capturing uncertainty about a representation and its relationships, expressing asymmetries more naturally than dot product or cosine similarity, and enabling more expressive parameterization of decision boundaries. This paper advocates for density-based distributed embeddings and presents a method for learning representations in the space of Gaussian distributions. We compare performance on various word embedding benchmarks, investigate the ability of these embeddings to model entailment and other asymmetric relationships, and explore novel properties of the representation."

>	"In this work we introduced a method to embed word types into the space of Gaussian distributions, and learn the embeddings directly in that space. This allows us to represent words not as low-dimensional vectors, but as densities over a latent space, directly representing notions of uncertainty and enabling a richer geometry in the embedded space. We demonstrated the effectiveness of these embeddings on a linguistic task requiring asymmetric comparisons, as well as standard word similarity benchmarks, learning of synthetic hierarchies, and several qualitative examinations. In future work, we hope to move beyond spherical or diagonal covariances and into combinations of low rank and diagonal matrices. Efficient updates and scalable learning is still possible due to the Sherman-Woodbury-Morrison formula. Additionally, going beyond diagonal covariances will enable us to keep our semantics from being axis-aligned, which will increase model capacity and expressivity. We also hope to move past stochastic gradient descent and warm starting and be able to learn the Gaussian representations robustly in one pass from scratch by using e.g. proximal or block coordinate descent methods. Improved optimization strategies will also be helpful on the highly nonconvex problem of training supervised hierarchies with KL divergence. Representing words and concepts as different types of distributions (including other elliptic distributions such as the Student’s t) is an exciting direction – Gaussians concentrate their density on a thin spherical ellipsoidal shell, which can lead to counterintuitive behavior in high dimensions. Multimodal distributions represent another clear avenue for future work. Combining ideas from kernel methods and manifold learning with deep learning and linguistic representation learning is an exciting frontier. In other domains, we want to extend the use of potential function representations to other tasks requiring embeddings, such as relational learning with the universal schema. We hope to leverage the asymmetric measures, probabilistic interpretation, and flexible training criteria of our model to tackle tasks involving similarity-in-context, comparison of sentences and paragraphs, and more general common sense reasoning."

----
	"- represent symbols not as points but as regions in space
	 - captures uncertainty of representation
	 - enhances asymmetric reasoning such as entailment/implicature
	 - more expressive decision boundaries and representational power"

  - `video` <http://youtu.be/Xm1XGjc9lDc?t=14m52s> (McCallum)
  - `video` <http://youtube.com/watch?v=PKTfALFk03M> (Vilnis)
  - <https://github.com/seomoz/word2gauss>



---
### interesting papers - semantic composition


#### ["Learning Distributed Representations of Sentences from Unlabelled Data"](http://arxiv.org/abs/1602.03483) Hill, Cho, Korhonen
>	"Unsupervised methods for learning distributed representations of words are ubiquitous in today’s NLP research, but far less is known about the best ways to learn distributed phrase or sentence representations from unlabelled data. This paper is a systematic comparison of models that learn such representations. We find that the optimal approach depends critically on the intended application. Deeper, more complex models are preferable for representations to be used in supervised systems, but shallow log-linear models work best for building representation spaces that can be decoded with simple spatial distance metrics. We also propose two new unsupervised representation-learning objectives designed to optimise the trade-off between training time, domain portability and performance."


#### ["Distributed Representations of Words and Phrases and Their Compositionality"](http://arxiv.org/abs/1310.4546) Mikolov, Sutskever, Chen, Corrado, Dean (word2vec)
>	"The recently introduced continuous Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships. In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word representations. We also describe a simple alternative to the hierarchical softmax called negative sampling. An inherent limitation of word representations is their indifference to word order and their inability to represent idiomatic phrases. For example, the meanings of "Canada" and "Air" cannot be easily combined to obtain "Air Canada". Motivated by this example, we present a simple method for finding phrases in text, and show that learning good vector representations for millions of phrases is possible."

----
>	"First, they provide a slightly modified objective function and a few other sampling heuristics that result in a more computationally efficient model. Why does this produce good word representations? The distributional hypothesis states that words in similar contexts have similar meanings. The objective above clearly tries to increase the quantity vw·vc for good word-context pairs, and decrease it for bad ones. Intuitively, this means that words that share many contexts will be similar to each other (note also that contexts sharing many words will also be similar to each other).

>	Second, they show that their model works with phrases, too, though they just do this by replacing the individual tokens in a multiword expression with a single symbol representing the phrase - pretty simple, but it works.

>	Third, they show what to me was a very surprising additional feature of the learned vector spaces: some relationships are encoded compositionally in the vector space, meaning that you can just add the vectors for two words like "Russian" and "capital" to get a vector that is very close to "Moscow". They didn't do any kind of thorough evaluation of this, but the fact the it works at all was very surprising to me. They did give a reasonable explanation, however, and I've put it into math below. The probability of two words i and j appearing in the same context in this model is proportional to exp(vi⋅vj). Now, if we have a third word, k, and its probability of appearing with both word i and word j is proportional to exp(vk⋅vi)*exp(vk⋅vj)=exp(vk⋅(vi+vj)). So what you get when you add the vectors for two words is something that is likely to show up in the contexts of both of them. Thus if you pick word i to be "Russian" and word j to be "capital", a word k that has high probability might very well be "Moscow", because it tends to show up in the context of both of those words. So we can see that this method does have some reasonable explanation for why it works."


#### ["Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank"](http://nlp.stanford.edu/sentiment/index.html) Socher, Perelygin, Wu, Chuang, Manning, Ng, Potts
>	"Semantic word spaces have been very useful but cannot express the meaning of longer phrases in a principled way. Further progress towards understanding compositionality in tasks such as sentiment detection requires richer supervised training and evaluation resources and more powerful models of composition. To remedy this, we introduce a Sentiment Treebank. It includes fine grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences and presents new challenges for sentiment compositionality. To address them, we introduce the Recursive Neural Tensor Network. When trained on the new treebank, this model outperforms all previous methods on several metrics. It pushes the state of the art in single sentence positive/negative classification from 80% up to 85.4%. The accuracy of predicting fine-grained sentiment labels for all phrases reaches 80.7%, an improvement of 9.7% over bag of features baselines. Lastly, it is the only model that can accurately capture the effects of negation and its scope at various tree levels for both positive and negative phrases."

>	"Most sentiment prediction systems work just by looking at words in isolation, giving positive points for positive words and negative points for negative words and then summing up these points. That way, the order of words is ignored and important information is lost. In constrast, our new deep learning model actually builds up a representation of whole sentences based on the sentence structure. It computes the sentiment based on how words compose the meaning of longer phrases. This way, the model is not as easily fooled as previous models. For example, our model learned that funny and witty are positive but the following sentence is still negative overall: This movie was actually neither that funny, nor super witty."

  - <http://nlp.stanford.edu/sentiment/>
  - <http://nlp.stanford.edu:8080/sentiment/rntnDemo.html>
  - `video` <http://youtube.com/watch?v=mVfPGu8rrXM> (Socher)
  - `video` <http://youtube.com/watch?v=tdLmf8t4oqM> (Socher)
  - `video` <https://youtu.be/sVXp0UwheXw?t=36m8s> (Socher)


#### ["Learning to Understand Phrases by Embedding the Dictionary"](http://arxiv.org/abs/1504.00548) Hill, Cho, Korhonen, Bengio
>	"Distributional models that learn rich semantic word representations are a success story of recent NLP research. However, developing models that learn useful representations of phrases and sentences has proved far harder. We propose using the definitions found in everyday dictionaries as a means of bridging this gap between lexical and phrasal semantics. We train a recurrent neural network to map dictionary definitions (phrases) to (lexical) representations of the words those definitions define. We present two applications of this architecture: a reverse dictionary, for returning the name of a concept given a definition or description, and a general-knowledge (crossword) question answerer. On both tasks, the RNN trained on definitions from a handful of freely-available lexical resources performs comparably or better than existing commercial systems that rely on major task-specific engineering and far greater memory footprints. This strong performance highlights the general effectiveness of both neural language models and definition-based training for training machines to understand phrases and sentences."

>	"Dictionaries exist in many of the world’s languages. We have shown how these lexical resources can be a valuable resource for training the latest neural language models to interpret and represent the meaning of phrases and sentences. While humans use the phrasal definitions in dictionaries to better understand the meaning of words, machines can use the words to better understand the phrases. We presented an recurrent neural network architecture with a long-short-term memory to explicitly exploit this idea. On the reverse dictionary task that mirrors its training setting, the RNN performs comparably to the best known commercial applications despite having access to many fewer definitions. Moreover, it generates smoother sets of candidates, uses less memory at query time and, perhaps most significantly, requires no linguistic pre-processing or task-specific engineering. We also showed how the description-to-word objective can be used to train models useful for other tasks. The architecture trained additionally on an encyclopedia performs well as a crossword question answerer, outperforming commercial systems on questions containing more than four words. While our QA experiments focused on a particular question type, the results suggest that a similar neural-language-model approach may ultimately lead to improved output from more general QA and dialog systems and information retrieval engines in general. In particular, we propose the reverse dictionary task as a comparatively general-purpose and objective way of evaluating how well models compose lexical meaning into phrase or sentence representations (whether or not they involve training on definitions directly). In the next stage of this research, we will explore ways to enhance the RNN model, especially in the question-answering context. The model is currently not trained on any question-like language, and would conceivably improve on exposure to such linguistic forms. Compared to state-of-the-art word representation learning models, it actually sees very few words during training, and may also benefit from learning from both dictionaries and unstructured text. Finally, we intend to explore ways to endow the model with richer world knowledge. This may require the integration of an external memory module."

  - `video` <http://youtube.com/watch?v=H16w6Z2CHkk> (Hill)
  - <https://github.com/fh295/DefGen2>


#### ["Skip-Thought Vectors"](http://arxiv.org/abs/1506.06726) Kiros, Zhu, Salakhutdinov, Zemel, Torralba, Urtasun, Fidler
>	"We describe an approach for unsupervised learning of a generic, distributed sentence encoder. Using the continuity of text from books, we train an encoder-decoder model that tries to reconstruct the surrounding sentences of an encoded passage. Sentences that share semantic and syntactic properties are thus mapped to similar vector representations. We next introduce a simple vocabulary expansion method to encode words that were not seen as part of training, allowing us to expand our vocabulary to a million words. After training our model, we extract and evaluate our vectors with linear models on 8 tasks: semantic relatedness, paraphrase detection, image-sentence ranking, question-type classification and 4 benchmark sentiment and subjectivity datasets. The end result is an off-the-shelf encoder that can produce highly generic sentence representations that are robust and perform well in practice."

>	"We evaluated the effectiveness of skip-thought vectors as an off-the-shelf sentence representation with linear classifiers across 8 tasks. Many of the methods we compare against were only evaluated on 1 task. The fact that skip-thought vectors perform well on all tasks considered highlight the robustness of our representations. We believe our model for learning skip-thought vectors only scratches the surface of possible objectives. Many variations have yet to be explored, including (a) deep encoders and decoders, (b) larger context windows, (c) encoding and decoding paragraphs, (d) other encoders, such as convnets. It is likely the case that more exploration of this space will result in even higher quality representations."

>	"Developing learning algorithms for distributed compositional semantics of words has been a long-standing open problem at the intersection of language understanding and machine learning. In recent years, several approaches have been developed for learning composition operators that map word vectors to sentence vectors including recursive networks, recurrent networks, convolutional networks and recursive-convolutional methods. All of these methods produce sentence representations that are passed to a supervised task and depend on a class label in order to backpropagate through the composition weights. Consequently, these methods learn high-quality sentence representations but are tuned only for their respective task. The paragraph vector is an alternative to the above models in that it can learn unsupervised sentence representations by introducing a distributed sentence indicator as part of a neural language model. The downside is at test time, inference needs to be performed to compute a new vector. In this paper we abstract away from the composition methods themselves and consider an alternative loss function that can be applied with any composition operator. We consider the following question: is there a task and a corresponding loss that will allow us to learn highly generic sentence representations? We give evidence for this by proposing a model for learning high-quality sentence vectors without a particular supervised task in mind. Using word vector learning as inspiration, we propose an objective function that abstracts the skip-gram model to the sentence level. That is, instead of using a word to predict its surrounding context, we instead encode a sentence to predict the sentences around it. Thus, any composition operator can be substituted as a sentence encoder and only the objective function becomes modified. We call our model skip-thoughts and vectors induced by our model are called skip-thought vectors. Our model depends on having a training corpus of contiguous text."

>	"One difficulty that arises with such an experimental setup is being able to construct a large enough word vocabulary to encode arbitrary sentences. For example, a sentence from a Wikipedia article might contain nouns that are highly unlikely to appear in our book vocabulary. We solve this problem by learning a mapping that transfers word representations from one model to another. Using pretrained word2vec representations learned with a continuous bag-of-words model, we learn a linear mapping from a word in word2vec space to a word in the encoder’s vocabulary space. The mapping is learned using all words that are shared between vocabularies. After training, any word that appears in word2vec can then get a vector in the encoder word embedding space."

----
>	"It turns out that skip-thought vectors have some intriguing properties that allow us to construct F in a really simple way. Suppose we have 3 vectors: an image caption x, a "caption style" vector c and a "book style" vector b. Then we define F as F(x) = x - c + b which intuitively means: keep the "thought" of the caption, but replace the image caption style with that of a story. Then, we simply feed F(x) to the decoder."

  - `video` <http://videolectures.net/deeplearning2015_salakhutdinov_deep_learning_2/#t=3776> (Salakhutdinov)
  - <https://github.com/tensorflow/models/tree/master/research/skip_thoughts>
  - <https://github.com/ryankiros/skip-thoughts>
  - <https://github.com/ryankiros/neural-storyteller> + <https://medium.com/@samim/generating-stories-about-images-d163ba41e4ed> (demo)


#### ["Deep Unordered Composition Rivals Syntactic Methods for Text Classification"](http://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf) Iyyer, Manjunatha, Boyd-Graber, Daume
>	"Many existing deep learning models for natural language processing tasks focus on learning the compositionality of their inputs, which requires many expensive computations. We present a simple deep neural network that competes with and, in some cases, outperforms such models on sentiment analysis and factoid question answering tasks while taking only a fraction of the training time. While our model is syntactically-ignorant, we show significant improvements over previous bag-of-words models by deepening our network and applying a novel variant of dropout. Moreover, our model performs better than syntactic models on datasets with high syntactic variance. We show that our model makes similar errors to syntactically-aware models, indicating that for the tasks we consider, nonlinearly transforming the input is more important than tailoring a network to incorporate word order and syntax."

>	"In this paper, we introduce the deep averaging network, which feeds an unweighted average of word vectors through multiple hidden layers before classification. The DAN performs competitively with more complicated neural networks that explicitly model semantic and syntactic compositionality. It is further strengthened by word dropout, a regularizer that reduces input redundancy. DANs obtain close to state-of-the-art accuracy on both sentence and document-level sentiment analysis and factoid question-answering tasks with much less training time than competing methods; in fact, all experiments were performed in a matter of minutes on a single laptop core. We find that both DANs and syntactic functions make similar errors given syntactically-complex input, which motivates research into more powerful models of compositionality."

>	"Theoretically, word dropout can also be applied to other neural network-based approaches. However, we observe no significant performance differences in preliminary experiments when applying word dropout to leaf nodes in RecNNs for sentiment analysis (dropped leaf representations are set to zero vectors), and it slightly hurts performance on the question answering task."

>	"We compare DANs to both the shallow NBOW model as well as more complicated syntactic models on sentence and document-level sentiment analysis and factoid question answering tasks. Our results show that DANs outperform other bag-ofwords models and many syntactic models with very little training time. On the question-answering task, DANs effectively train on out-of-domain data, while RecNNs struggle to reconcile the syntactic differences between the training and test data."

  - `video` <http://youtube.com/watch?v=y1_0i1RF74c> (Iyyer)
  - <https://cs.umd.edu/~miyyer/data/acldan_slides.pdf>
  - <http://github.com/miyyer/dan>


#### ["Towards Universal Paraphrastic Sentence Embeddings"](https://arxiv.org/abs/1511.08198) Wieting, Bansal, Gimpel, Livescu
>	"We consider the problem of learning general-purpose, paraphrastic sentence embeddings based on supervision from the Paraphrase Database (Ganitkevitch et al., 2013). We compare six compositional architectures, evaluating them on annotated textual similarity datasets drawn both from the same distribution as the training data and from a wide range of other domains. We find that the most complex architectures, such as long short-term memory (LSTM) recurrent neural networks, perform best on the in-domain data. However, in out-of-domain scenarios, simple architectures such as word averaging vastly outperform LSTMs. Our simplest averaging model is even competitive with systems tuned for the particular tasks while also being extremely efficient and easy to use. In order to better understand how these architectures compare, we conduct further experiments on three supervised NLP tasks: sentence similarity, entailment, and sentiment classification. We again find that the word averaging models perform well for sentence similarity and entailment, outperforming LSTMs. However, on sentiment classification, we find that the LSTM performs very strongly-even recording new state-of-the-art performance on the Stanford Sentiment Treebank. We then demonstrate how to combine our pretrained sentence embeddings with these supervised tasks, using them both as a prior and as a black box feature extractor. This leads to performance rivaling the state of the art on the SICK similarity and entailment tasks. We release all of our resources to the research community with the hope that they can serve as the new baseline for further work on universal sentence embeddings."

  - `video` <http://youtube.com/watch?v=fC0j6mEFdZE> (Gimpel)
  - `video` <http://videolectures.net/iclr2016_wieting_universal_paraphrastic/> (Wieting)
  - <https://github.com/jwieting/iclr2016>


#### ["A Simple but Tough-to-Beat Baseline for Sentence Embeddings"](https://openreview.net/pdf?id=SyK00v5xx) Arora, Liang, Ma
>	"The success of neural network methods for computing word embeddings has motivated methods for generating semantic embeddings of longer pieces of text, such as sentences and paragraphs. Surprisingly, Wieting et al (ICLR’16) showed that such complicated methods are outperformed, especially in out-of-domain (transfer learning) settings, by simpler methods involving mild retraining of word embeddings and basic linear regression. The method of Wieting et al. requires retraining with a substantial labeled dataset such as Paraphrase Database (Ganitkevitch et al., 2013). The current paper goes further, showing that the following completely unsupervised sentence embedding is a formidable baseline: Use word embeddings computed using one of the popular methods on unlabeled corpus like Wikipedia, represent the sentence by a weighted average of the word vectors, and then modify them a bit using PCA/SVD. This weighting improves performance by about 10% to 30% in textual similarity tasks, and beats sophisticated supervised methods including RNN’s and LSTM’s. It even improves Wieting et al.’s embeddings. This simple method should be used as the baseline to beat in future, especially when labeled training data is scarce or nonexistent. The paper also gives a theoretical explanation of the success of the above unsupervised method using a latent variable generative model for sentences, which is a simple extension of the model in Arora et al. (TACL’16) with new “smoothing” terms that allow for words occurring out of context, as well as high probabilities for words like and, not in all contexts."

  - `video` <https://youtube.com/watch?v=BCsOrewkmH4> (Liang)
  - `video` <https://youtu.be/KR46z_V0BVw?t=49m10s> (Arora)
  - <https://akshayka.github.io/papers/html/arora2017sentence-embeddings.html>
  - <https://github.com/PrincetonML/SIF>
  - <https://github.com/YingyuLiang/SIF>


#### ["Distributed Representations of Sentences and Documents"](http://arxiv.org/abs/1405.4053) Le, Mikolov (Paragraph vector)
>	"Many machine learning algorithms require the input to be represented as a fixed-length feature vector. When it comes to texts, one of the most common fixed-length features is bag-of-words. Despite their popularity, bag-of-words features have two major weaknesses: they lose the ordering of the words and they also ignore semantics of the words. For example, “powerful,” “strong” and “Paris” are equally distant. In this paper, we propose Paragraph Vector, an unsupervised algorithm that learns fixed-length feature representations from variable-length pieces of texts, such as sentences, paragraphs, and documents. Our algorithm represents each document by a dense vector which is trained to predict words in the document. Its construction gives our algorithm the potential to overcome the weaknesses of bag-of-words models. Empirical results show that Paragraph Vectors outperform bag-of-words models as well as other techniques for text representations. Finally, we achieve new state-of-the-art results on several text classification and sentiment analysis tasks."

>	"Paragraph vectors can be computed for things that are not paragraphs, in particular, documents, users, products, videos, audios."

----
>	"Again, we can point to the general trend of AI toward simpler models. RNNs are a way of combining semantic vectors with probabilistic context-free grammers; Paragraph Vector combines semantic vectors with a markov model. Markov models are simpler and less powerful; therefore, by the contrarian logic of the field, we expect them to do better. And, they do."

  - <http://building-babylon.net/2015/06/03/document-embedding-with-paragraph-vectors/>
  - <https://github.com/inejc/paragraph-vectors>


#### ["Document Embedding with Paragraph Vectors"](http://arxiv.org/abs/1507.07998) Dai, Olah, Le
>	"Paragraph Vectors has been recently proposed as an unsupervised method for learning distributed representations for pieces of texts. In their work, the authors showed that the method can learn an embedding of movie review texts which can be leveraged for sentiment analysis. That proof of concept, while encouraging, was rather narrow. Here we consider tasks other than sentiment analysis, provide a more thorough comparison of Paragraph Vectors to other document modelling algorithms such as Latent Dirichlet Allocation, and evaluate performance of the method as we vary the dimensionality of the learned representation. We benchmarked the models on two document similarity data sets, one from Wikipedia, one from arXiv. We observe that the Paragraph Vector method performs significantly better than other methods, and propose a simple improvement to enhance embedding quality. Somewhat surprisingly, we also show that much like word embeddings, vector operations on Paragraph Vectors can perform useful semantic results."

>	"We described a new set of results on Paragraph Vectors showing they can effectively be used for measuring semantic similarity between long pieces of texts. Our experiments show that Paragraph Vectors are superior to LDA for measuring semantic similarity on Wikipedia articles across all sizes of Paragraph Vectors. Paragraph Vectors also perform on par with LDA’s best performing number of topics on arXiv papers and perform consistently relative to the embedding size. Also surprisingly, vector operations can be performed on them similarly to word vectors. This can provide interesting new techniques for a wide range of applications: local and nonlocal corpus navigation, dataset exploration, book recommendation and reviewer allocation."

>	"We can perform vector operations on paragraph vectors for local and non-local browsing of Wikipedia. The first experiment is to find related articles to “Lady Gaga.” The second experiment is to find the Japanese equivalence of “Lady Gaga.” This can be achieved by vector operations: pv(“Lady Gaga”) - wv(“American”) + wv(“Japanese”) where pv is paragraph vectors and wv is word vectors. Both sets of results show that Paragraph Vectors can achieve the same kind of analogies like Word Vectors."

>	"It can be seen that paragraph vectors perform better than LDA on Wikipedia article similarity task. Both paragraph vectors and averaging word embeddings perform better than the LDA model. For LDA, we found that TF-IDF weighting of words and their inferred topic allocations did not affect the performance. From these results, we can also see that joint training of word vectors improves the final quality of the paragraph vectors."


#### ["Learning Deep Structured Semantic Models for Web Search using Clickthrough Data"](http://research.microsoft.com/apps/pubs/default.aspx?id=198202) Huang, He, Gao, Deng, Acero, Heck (DSSM and sent2vec models)
>	"Latent semantic models, such as LSA, intend to map a query to its relevant documents at the semantic level where keyword-based matching often fails. In this study we strive to develop a series of new latent semantic models with a deep structure that project queries and documents into a common low-dimensional space where the relevance of a document given a query is readily computed as the distance between them. The proposed deep structured semantic models are discriminatively trained by maximizing the conditional likelihood of the clicked documents given a query using the clickthrough data. To make our models applicable to large-scale Web search applications, we also use a technique called word hashing, which is shown to effectively scale up our semantic models to handle large vocabularies which are common in such tasks. The new models are evaluated on a Web document ranking task using a real-world data set. Results show that our best model significantly outperforms other latent semantic models, which were considered state-of-the-art in the performance prior to the work presented in this paper."

>	"We present and evaluate a series of new latent semantic models, notably those with deep architectures which we call the DSSM. The main contribution lies in our significant extension of the previous latent semantic models (e.g., LSA) in three key aspects. First, we make use of the clickthrough data to optimize the parameters of all versions of the models by directly targeting the goal of document ranking. Second, inspired by the deep learning framework recently shown to be highly successful in speech recognition, we extend the linear semantic models to their nonlinear counterparts using multiple hidden-representation layers. The deep architectures adopted have further enhanced the modeling capacity so that more sophisticated semantic structures in queries and documents can be captured and represented. Third, we use a letter n-gram based word hashing technique that proves instrumental in scaling up the training of the deep models so that very large vocabularies can be used in realistic web search. In our experiments, we show that the new techniques pertaining to each of the above three aspects lead to significant performance improvement on the document ranking task. A combination of all three sets of new techniques has led to a new state-of-the-art semantic model that beats all the previously developed competing models with a significant margin."

>	"DSSM stands for Deep Structured Semantic Model, or more general, Deep Semantic Similarity Model. DSSM is a deep neural network modeling technique for representing text strings (sentences, queries, predicates, entity mentions, etc.) in a continuous semantic space and modeling semantic similarity between two text strings (e.g., Sent2Vec). DSSM has wide applications including information retrieval and web search ranking (Huang et al. 2013; Shen et al. 2014a,2014b), ad selection/relevance, contextual entity search and interestingness tasks (Gao et al. 2014a), question answering (Yih et al., 2014), knowledge inference (Yang et al., 2014), image captioning (Fang et al., 2014), and machine translation (Gao et al., 2014b) etc. DSSM can be used to develop latent semantic models that project entities of different types (e.g., queries and documents) into a common low-dimensional semantic space for a variety of machine learning tasks such as ranking and classification. For example, in web search ranking, the relevance of a document given a query can be readily computed as the distance between them in that space. With the latest GPUs from Nvidia, we are able to train our models on billions of words."

>	"Sent2vec maps a pair of short text strings (e.g., sentences or query-answer pairs) to a pair of feature vectors in a continuous, low-dimensional space where the semantic similarity between the text strings is computed as the cosine similarity between their vectors in that space. sent2vec performs the mapping using the Deep Structured Semantic Model (DSSM) or the DSSM with convolutional-pooling structure (CDSSM)."

  - <http://research.microsoft.com/en-us/projects/dssm/>
  - <http://research.microsoft.com/pubs/232372/CIKM14_tutorial_HeGaoDeng.pdf>
  - <http://research.microsoft.com/en-us/downloads/731572aa-98e4-4c50-b99d-ae3f0c9562b9/default.aspx> (sent2vec code)
  - `video` <https://youtu.be/x7B6RudUQLI?t=1h5m5s> (Gulin) `in russian`
  - <https://habrahabr.ru/company/yandex/blog/314222/> `in russian`


#### ["Order-Embeddings of Images and Language"](http://arxiv.org/abs/1511.06361) Vendrov, Kiros, Fidler, Urtasun
>	"Hypernymy, textual entailment, and image captioning can be seen as special cases of a single visual-semantic hierarchy over words, sentences, and images. In this paper we advocate for explicitly modeling the partial order structure of this hierarchy. Towards this goal, we introduce a general method for learning ordered representations, and show how it can be applied to a variety of tasks involving images and language. We show that the resulting representations improve performance over current approaches for hypernym prediction and image-caption retrieval."

  - `video` <http://videolectures.net/iclr2016_vendrov_order_embeddings/> (Vendrov)
  - `code` <https://github.com/ivendrov/order-embedding>
  - `code` <https://github.com/ivendrov/order-embeddings-wordnet>
  - `code` <https://github.com/LeavesBreathe/tensorflow_with_latest_papers/blob/master/partial_ordering_embedding.py>



---
### interesting papers - syntactic parsing


#### ["Grammar As a Foreign Language"](http://arxiv.org/abs/1412.7449) Vinyals, Kaiser, Koo, Petrov, Sutskever, Hinton
>	"Syntactic parsing is a fundamental problem in computational linguistics and natural language processing. Traditional approaches to parsing are highly complex and problem specific. Recently, Sutskever et al. (2014) presented a domain-independent method for learning to map input sequences to output sequences that achieved strong results on a large scale machine translation problem. In this work, we show that precisely the same sequence-to-sequence method achieves results that are close to state-of-the-art on syntactic constituency parsing, whilst making almost no assumptions about the structure of the problem."

>	"An attention mechanism inspired by Bahdanau et al.'s model made our parser much more statistically efficient. In addition, the model learned a stack decoder solely from data. Our model (an ensemble) matches the BerkeleyParser when trained on the labeled 40K training sentences. When training our model on a large number of automatically-generated high-confidence parses, we achieve the best published results on constituency parsing."


#### ["Parsing With Compositional Vector Grammars"](http://socher.org/index.php/Main/ParsingWithCompositionalVectorGrammars) Socher, Bauer, Manning, Ng
>	"Natural language parsing has typically been done with small sets of discrete categories such as NP and VP, but this representation does not capture the full syntactic nor semantic richness of linguistic phrases, and attempts to improve on this by lexicalizing phrases or splitting categories only partly address the problem at the cost of huge feature spaces and sparseness. Instead, we introduce a Compositional Vector Grammar, which combines PCFGs with a syntactically untied recursive neural network that learns syntactico-semantic, compositional vector representations. The CVG improves the PCFG of the Stanford Parser by 3.8% to obtain an F1 score of 90.4%. It is fast to train and implemented approximately as an efficient reranker it is about 20% faster than the current Stanford factored parser. The CVG learns a soft notion of head words and improves performance on the types of ambiguities that require semantic information such as PP attachments."

>	"Model jointly learns how to parse and how to represent phrases as both discrete categories and continuous vectors. CVGs combine the advantages of standard probabilistic context free grammars with those of recursive neural networks. The former can capture the discrete categorization of phrases into NP or PP while the latter can capture fine-grained syntactic and compositional-semantic information on phrases and words. This information can help in cases where syntactic ambiguity can only be resolved with semantic information, such as in the PP attachment of the two sentences: They ate udon with forks. vs. They ate udon with chicken."

  - `video` <https://youtu.be/DJHvaGU9SW8?t=23m30s> (Socher)
  - <https://github.com/ofirnachum/tree_rnn>


#### ["A Fast and Accurate Dependency Parser using Neural Networks"](http://cs.stanford.edu/~danqi/papers/emnlp2014.pdf) Chen, Manning
>	"Almost all current dependency parsers classify based on millions of sparse indicator features. Not only do these features generalize poorly, but the cost of feature computation restricts parsing speed significantly. In this work, we propose a novel way of learning a neural network classifier for use in a greedy, transition-based dependency parser. Because this classifier learns and uses just a small number of dense features, it can work very fast, while achieving an about 2% improvement in unlabeled and labeled attachment scores on both English and Chinese datasets. Concretely, our parser is able to parse more than 1000 sentences per second at 92.2% unlabeled attachment score on the English Penn Treebank."

>	"We have presented a novel dependency parser using neural networks. Experimental evaluations show that our parser outperforms other greedy parsers using sparse indicator features in both accuracy and speed. This is achieved by representing all words, POS tags and arc labels as dense vectors, and modeling their interactions through a novel cube activation function. Our model only relies on dense features, and is able to automatically learn the most useful feature conjunctions for making predictions. An interesting line of future work is to combine our neural network based classifier with search-based models to further improve accuracy. Also, there is still room for improvement in our architecture, such as better capturing word conjunctions, or adding richer features (e.g., distance, valency)."

  - `video` <http://youtube.com/watch?v=MLAcBv5dLEs> (Chen)


#### ["Transition-Based Dependency Parsing with Stack Long Short-Term Memory"](http://arxiv.org/abs/1505.08075) Dyer, Ballesteros, Ling, Matthews, Smith
>	"We propose a technique for learning representations of parser states in transition-based dependency parsers. Our primary innovation is a new control structure for sequence-to-sequence neural networks - the stack LSTM. Like the conventional stack data structures used in transition-based parsing, elements can be pushed to or popped from the top of the stack in constant time, but, in addition, an LSTM maintains a continuous space embedding of the stack contents. This lets us formulate an efficient parsing model that captures three facets of a parser’s state: (i) unbounded look-ahead into the buffer of incoming words, (ii) the complete history of actions taken by the parser, and (iii) the complete contents of the stack of partially built tree fragments, including their internal structures. Standard backpropagation techniques are used for training and yield state-of-the-art parsing performance."

>	"We presented stack LSTMs, recurrent neural networks for sequences, with push and pop operations, and used them to implement a state-of-the-art transition-based dependency parser. We conclude by remarking that stack memory offers intriguing possibilities for learning to solve general information processing problems. Here, we learned from observable stack manipulation operations (i.e., supervision from a treebank), and the computed embeddings of final parser states were not used for any further prediction. However, this could be reversed, giving a device that learns to construct context-free programs (e.g., expression trees) given only observed outputs; one application would be unsupervised parsing. Such an extension of the work would make it an alternative to architectures that have an explicit external memory such as neural Turing machines and memory networks. However, as with those models, without supervision of the stack operations, formidable computational challenges must be solved (e.g., marginalizing over all latent stack operations), but sampling techniques and techniques from reinforcement learning have promise here, making this an intriguing avenue for future work."

>	"Transition-based dependency parsing formalizes the parsing problem as a series of decisions that read words sequentially from a buffer and combine them incrementally into syntactic structures. This formalization is attractive since the number of operations required to build any projective parse tree is linear in the length of the sentence, making transition-based parsing computationally efficient relative to graph- and grammar-based formalisms. The challenge in transition-based parsing is modeling which action should be taken in each of the unboundedly many states encountered as the parser progresses. This challenge has been addressed by development of alternative transition sets that simplify the modeling problem by making better attachment decisions, through feature engineering and more recently using neural networks. We extend this last line of work by learning representations of the parser state that are sensitive to the complete contents of the parser’s state: that is, the complete input buffer, the complete history of parser actions, and the complete contents of the stack of partially constructed syntactic structures. This “global” sensitivity to the state contrasts with previous work in transition-based dependency parsing that uses only a narrow view of the parsing state when constructing representations (e.g., just the next few incoming words, the head words of the top few positions in the stack, etc.). Although our parser integrates large amounts of information, the representation used for prediction at each time step is constructed incrementally, and therefore parsing and training time remain linear in the length of the input sentence. The technical innovation that lets us do this is a variation of recurrent neural networks with long short-term memory units which we call stack LSTMs, and which support both reading (pushing) and “forgetting” (popping) inputs. Our parsing model uses three stack LSTMs: one representing the input, one representing the stack of partial syntactic trees, and one representing the history of parse actions to encode parser states. Since the stack of partial syntactic trees may contain both individual tokens and partial syntactic structures, representations of individual tree fragments are computed compositionally with recursive neural networks. The parameters are learned with back-propagation, and we obtain state-of-the-art results on Chinese and English dependency parsing tasks."

  - `video` <http://youtube.com/watch?v=KNH5A_7-KVM> (Smith)
  - `video` <http://youtube.com/watch?v=cp88pPknvDY> (Ballesteros)
  - `video` <http://techtalks.tv/talks/transition-based-dependency-parsing-with-stack-long-short-term-memory/61731/> (Ballesteros)
  - `audio` <https://soundcloud.com/nlp-highlights/05-transition-based-dependency-parsing-with-stack-long-short-term-memory> (Gardner, Ammar)


#### ["Structured Training for Neural Network Transition-Based Parsing"](http://arxiv.org/abs/1506.06158) Weiss, Alberti, Collins, Petrov
>	"We present structured perceptron training for neural network transition-based dependency parsing. We learn the neural network representation using a gold corpus augmented by a large number of automatically parsed sentences. Given this fixed network representation, we learn a final layer using the structured perceptron with beam-search decoding. On the Penn Treebank, our parser reaches 94.26% unlabeled and 92.41% labeled attachment accuracy, which to our knowledge is the best accuracy on Stanford Dependencies to date. We also provide in-depth ablative analysis to determine which aspects of our model provide the largest gains in accuracy."

>	"We presented a new state of the art in dependency parsing: a transition-based neural network parser trained with the structured perceptron and ASGD. We then combined this approach with unlabeled data and tri-training to further push state-of-the-art in semi-supervised dependency parsing. Nonetheless, our ablative analysis suggests that further gains are possible simply by scaling up our system to even larger representations. In future work, we will apply our method to other languages, explore end-to-end training of the system using structured learning, and scale up the method to larger datasets and network structures."

  - `video` <http://techtalks.tv/talks/structured-training-for-neural-network-transition-based-parsing/61730/>


#### ["Dependency Parsing as Head Selection"](http://arxiv.org/abs/1606.01280) Zhang, Cheng, Lapata
>	"Conventional dependency parsers rely on a statistical model and a transition system or graph algorithm to enforce tree-structured outputs during training and inference. In this work we formalize dependency parsing as the problem of selecting the head (a.k.a. parent) of each word in a sentence. Our model which we call DENSE (as shorthand for Dependency Neural Selection) employs bidirectional recurrent neural networks for the head selection task. Without enforcing any structural constraints during training, DENSE generates (at inference time) trees for the overwhelming majority of sentences (95% on an English dataset), while remaining non-tree outputs can be adjusted with a maximum spanning tree algorithm. We evaluate DENSE on four languages (English, Chinese, Czech, and German) with varying degrees of non-projectivity. Despite the simplicity of our approach, experiments show that the resulting parsers are on par with or outperform the state of the art."

>	"Compared to previous work, we formalize dependency parsing as the task of finding for each word in a sentence its most probable head. Both head selection and the features it is based on are learned using neural networks. The model locally optimizes a set of head-dependent decisions without attempting to enforce any global consistency during training. As a result, DENSE predicts dependency arcs greedily following a simple training procedure without predicting a parse tree, i.e., without performing a sequence of transition actions or employing a graph algorithm during training. Nevertheless, it can be seamlessly integrated with a graph-based decoder to ensure tree-structured output. In common with recent neural network-based dependency parsers, we aim to alleviate the need for hand-crafting feature combinations. Beyond feature learning, we further show that it is possible to simplify inference and training with bi-directional recurrent neural networks."

>	"Experimental results show that DENSE achieves competitive performance across four different languages and can seamlessly transfer from a projective to a non-projective parser simply by changing the post-processing MST algorithm during inference."



---
### interesting papers - semantic parsing


#### ["Learning Dependency-Based Compositional Semantics"](http://arxiv.org/abs/1109.6841) Liang, Jordan, Klein
>	"Suppose we want to build a system that answers a natural language question by representing its semantics as a logical form and computing the answer given a structured database of facts. The core part of such a system is the semantic parser that maps questions to logical forms. Semantic parsers are typically trained from examples of questions annotated with their target logical forms, but this type of annotation is expensive. Our goal is to learn a semantic parser from question-answer pairs instead, where the logical form is modeled as a latent variable. Motivated by this challenging learning problem, we develop a new semantic formalism, dependency-based compositional semantics, which has favorable linguistic, statistical, and computational properties. We define a log-linear distribution over DCS logical forms and estimate the parameters using a simple procedure that alternates between beam search and numerical optimization. On two standard semantic parsing benchmarks, our system outperforms all existing state-of-the-art systems, despite using no annotated logical forms."

  - `video` <http://youtube.com/watch?v=z4XCjlCeGkQ>


#### ["Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing"](http://jmlr.org/proceedings/papers/v22/bordes12/bordes12.pdf) Bordes, Glorot, Weston, Bengio
>	"Open-text semantic parsers are designed to interpret any statement in natural language by inferring a corresponding meaning representation (MR - a formal representation of its sense). Unfortunately, large scale systems cannot be easily machine-learned due to a lack of directly supervised data. We propose a method that learns to assign MRs to a wide range of text (using a dictionary of more than 70,000 words mapped to more than 40,000 entities) thanks to a training scheme that combines learning from knowledge bases (e.g. WordNet) with learning from raw text. The model jointly learns representations of words, entities and MRs via a multi-task training process operating on these diverse sources of data. Hence, the system ends up providing methods for knowledge acquisition and word-sense disambiguation within the context of semantic parsing in a single elegant framework. Experiments on these various tasks indicate the promise of the approach."

>	"This paper presented a large-scale system for semantic parsing mapping raw text to disambiguated MRs. The key contributions are: (i) an energy-based model that scores triplets of relations between ambiguous lemmas and unambiguous entities (synsets); and (ii) multi-tasking the learning of such a model over several resources so that we can effectively learn to build disambiguated meaning representations from raw text with relatively limited supervision."

>	"For a given sentence, we infer a MR in two stages: (1) a semantic role labeling step predicts the semantic structure; (2) a disambiguation step assigns a corresponding entity to each relevant word, so as to minimize a learnt energy function. For step (1) we use an existing approach. Our key contribution is a novel inference model for performing step (2). This consists of an energy-based model that is trained with data from multiple sources in order to combat the lack of strong supervision. Our energy-based model is trained to jointly capture semantic information between words, entities and combinations of those. This is encoded in a distributed representation in which a low dimensional embedding vector (or simply embedding in the following) is learnt for each symbol. Our semantic matching energy function is designed to blend such embeddings in order to assign low energy values to plausible combinations."

>	"Resources like WordNet and ConceptNet encode common-sense knowledge in the form of relations between entities (e.g. has part(car, wheel)) but do not link this knowledge to raw text (sentences). On the other hand, text resources like Wikipedia are not grounded with entities. Our training procedure is based on multi-task learning across different data sets including the three mentioned above. In this way MRs induced from text and relations between entities are embedded (and integrated) in the same space. This allows us to learn to perform disambiguation on raw text by using large amounts of indirect supervision and little direct supervision. The model learns to use common-sense knowledge (such as WordNet relations between entities) to help disambiguate, i.e. to choose the correct WordNet sense of a word."

>	"We also demonstrate the possibility that our system can perform knowledge extraction, i.e. learn new common-sense relations that do not exist in WordNet by multi-tasking with raw text."


#### ["A Deep Architecture for Semantic Parsing"](http://arxiv.org/abs/1404.7296) Grefenstette, Blunsom, Freitas, Hermann
>	"Many successful approaches to semantic parsing build on top of the syntactic analysis of text, and make use of distributional representations or statistical models to match parses to ontology-specific queries. This paper presents a novel deep learning architecture which provides a semantic parsing system through the union of two neural models of language semantics. It allows for the generation of ontology-specific queries from natural language statements and questions without the need for parsing, which makes it especially suitable to grammatically malformed or syntactically atypical text, such as tweets, as well as permitting the development of semantic parsers for resource-poor languages."

>	"The ubiquity of always-online computers in the form of smartphones, tablets, and notebooks has boosted the demand for effective question answering systems. This is exemplified by the growing popularity of products like Apple’s Siri or Google’s Google Now services. In turn, this creates the need for increasingly sophisticated methods for semantic parsing. Recent work has answered this call by progressively moving away from strictly rule-based semantic parsing, towards the use of distributed representations in conjunction with traditional grammatically-motivated re-write rules. This paper seeks to extend this line of thinking to its logical conclusion, by providing the first (to our knowledge) entirely distributed neural semantic generative parsing model. It does so by adapting deep learning methods from related work in sentiment analysis, document classification, frame-semantic parsing, and machine translation, combining two empirically successful deep learning models to form a new architecture for semantic parsing."

>	"We begin by introducing two deep learning models with different aims, namely the joint learning of embeddings in parallel corpora, and the generation of strings of a language conditioned on a latent variable, respectively. We then discuss how both models can be combined and jointly trained to form a deep learning model supporting the generation of knowledgebase queries from natural language questions."

>	"With the provision that the model can generate freebase queries correctly, further work will seek to determine whether this architecture can generate other structured formal language expressions, such as lambda expressions for use in textual entailment task."


#### ["Language to Logical Form with Neural Attention"](http://arxiv.org/abs/1601.01280) Dong, Lapata
>	"Semantic parsing aims at mapping natural language to machine interpretable meaning representations. Traditional approaches rely on high-quality lexicons, manually-built templates, and linguistic features which are either domainor representation-specific. In this paper we present a general method based on an attention-enhanced encoder-decoder model. We encode input utterances into vector representations, and generate their logical forms by conditioning the output sequences or trees on the encoding vectors. Experimental results on four datasets show that our approach performs competitively without using hand-engineered features and is easy to adapt across domains and meaning representations."

>	"In this paper we presented an encoder-decoder neural network model for mapping natural language descriptions to their meaning representations. We encode natural language utterances into vectors and generate their corresponding logical forms as sequences or trees using recurrent neural networks with long short-term memory units. Experimental results show that enhancing the model with a hierarchical tree decoder and an attention mechanism improves per formance across the board. Extensive comparisons with previous methods show that our approach performs competitively, without recourse to domain- or representation-specific features. Directions for future work are many and varied. For example, it would be interesting to learn a model from question-answer pairs without access to target logical forms. Beyond semantic parsing, we would also like to apply our Seq2Tree model to related structured prediction tasks such as constituency parsing."


#### ["Semantic Parsing using Distributional Semantics and Probabilistic Logic"](http://yoavartzi.com/sp14/pub/bem-sp14-2014.pdf) Beltagy, Erk, Mooney
>	"We propose a new approach to semantic parsing that is not constrained by a fixed formal ontology and purely logical inference. Instead, we use distributional semantics to generate only the relevant part of an on-the-fly ontology. Sentences and the on-the-fly ontology are represented in probabilistic logic. For inference, we use probabilistic logic frameworks like Markov Logic Networks and Probabilistic Soft Logic. This semantic parsing approach is evaluated on two tasks, Recognizing Textual Entailment and Semantic Textual Similarity, both accomplished using inference in probabilistic logic. Experiments show the potential of the approach."

>	"Traditional logical approaches to semantics and newer distributional or vector space approaches have complementary strengths and weaknesses. We have developed methods that integrate logical and distributional models by using a CCG-based parser to produce a detailed logical form for each sentence, and combining the result with soft inference rules derived from distributional semantics that connect the meanings of their component words and phrases. For recognizing textual entailment we use Markov Logic Networks to combine these representations, and for semantic textual similarity we use Probabilistic Soft Logic. We present experimental results on standard benchmark datasets for these problems and emphasize the advantages of combining logical structure of sentences with statistical knowledge mined from large corpora."

  - `video` <http://youtube.com/watch?v=GhBHRhIsQIE> (Mooney)
  - `video` <http://youtube.com/watch?v=yNrIXkwAPfQ> (Mooney)


#### ["Unsupervised Induction of Semantic Roles within a Reconstruction-Error Minimization Framework"](http://arxiv.org/abs/1502.03044) Titov, Khoddam
>	"We introduce a new approach to unsupervised estimation of feature-rich semantic role labeling models. Our model consists of two components: (1) an encoding component: a semantic role labeling model which predicts roles given a rich set of syntactic and lexical features; (2) a reconstruction component: a tensor factorization model which relies on roles to predict argument fillers. When the components are estimated jointly to minimize errors in argument reconstruction, the induced roles largely correspond to roles defined in annotated resources. Our method performs on par with most accurate role induction methods on English and German, even though, unlike these previous approaches, we do not incorporate any prior linguistic knowledge about the languages."

  - `video` <https://events.yandex.ru/lib/talks/2728/> (Titov)
  - `video` <http://techtalks.tv/talks/unsupervised-induction-of-semantic-roles-within-a-reconstruction-error-minimization-framework/61463/> (Titov)



---
### interesting papers - text classification


#### ["A Convolutional Neural Network for Modelling Sentences"](http://arxiv.org/abs/1404.2188) Kalchbrenner, Grefenstette, Blunsom
>	"The ability to accurately represent sentences is central to language understanding. We describe a convolutional architecture dubbed the Dynamic Convolutional Neural Network that we adopt for the semantic modelling of sentences. The network uses Dynamic k-Max Pooling, a global pooling operation over linear sequences. The network handles input sentences of varying length and induces a feature graph over the sentence that is capable of explicitly capturing short and long-range relations. The network does not rely on a parse tree and is easily applicable to any language. We test the DCNN in four experiments: small scale binary and multi-class sentiment prediction, six-way question classification and Twitter sentiment prediction by distant supervision. The network achieves excellent performance in the first three tasks and a greater than 25% error reduction in the last task with respect to the strongest baseline."

  - <https://github.com/FredericGodin/DynamicCNN>


#### ["Deep Multi-Instance Transfer Learning"](http://arxiv.org/abs/1411.3128) Kotzias, Denil, Blunsom, Freitas (sentence-level and entity-level sentiment classification learned from text-level classification)
>	"We present a new approach for transferring knowledge from groups to individuals that comprise them. We evaluate our method in text, by inferring the ratings of individual sentences using full-review ratings. This approach combines ideas from transfer learning, deep learning and multi-instance learning, and reduces the need for laborious human labelling of fine-grained data when abundant labels are available at the group level."

>	"In this work, we present a novel objective function, for instance learning in an a multi-instance learning setting. A similarity measure between instances is required in order to optimise the objective function. Deep Neural Networks have been very successful in creating representations of data, that capture their underlying characteristics. This work capitalises on their success by using embeddings of data and their similarity, as produced by a deep network, as instances for experiments. In this paper we show that this idea can be used to infer ratings of sentences (individuals) from ratings of reviews (groups of sentences). This enables us to extract the most positive and negative sentences in a review. In applications where reviews are overwhelmingly positive, detecting negative comments is a key step toward improving costumer service."


#### ["Convolutional Neural Networks for Sentence Classification"](http://arxiv.org/abs/1408.5882) Kim
>	"We report on a series of experiments with convolutional neural networks trained on top of pre-trained word vectors for sentence-level classification tasks. We first show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static word vectors. The CNN models discussed herein improve upon the state-of-the-art on 4 out of 7 tasks, which include sentiment analysis and question classification."

>	"In the present work, we train a simple CNN with one layer of convolution on top of word vectors obtained from an unsupervised neural language model. These vectors were trained by word2vec. We initially keep the word vectors static and learn only the other parameters of the model. Despite little tuning of hyperparameters, this simple model achieves excellent results on multiple benchmarks, suggesting that the pre-trained vectors are ‘universal’ feature extractors that can be utilized for various classification tasks. Learning task-specific vectors through fine-tuning results in further improvements. We finally describe a simple modification to the architecture to allow for the use of both pre-trained and task-specific vectors by having multiple channels."

>	"Despite little tuning of hyperparameters, a simple CNN with one layer of convolution performs remarkably well. Our results add to the well-established evidence that unsupervised pre-training of word vectors is an important ingredient in deep learning for NLP."

  - <http://www.shortscience.org/paper?bibtexKey=journals/corr/Kim14f>
  - <https://github.com/harvardnlp/sent-conv-torch>
  - <https://github.com/yoonkim/CNN_sentence>
  - <https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras>


#### ["#TAGSPACE: Semantic Embeddings from Hashtags"](https://research.facebook.com/publications/279494668926031/-tagspace-semantic-embeddings-from-hashtags/) Weston, Chopra, Adams
>	"We describe a convolutional neural network that learns feature representations for short textual posts using hashtags as a supervised signal. The proposed approach is trained on up to 5.5 billion words predicting 100,000 possible hashtags. As well as strong performance on the hashtag prediction task itself, we show that its learned representation of text (ignoring the hashtag labels) is useful for other tasks as well. To that end, we present results on a document recommendation task, where it also outperforms a number of baselines."

  - `video` <http://youtube.com/watch?v=K5C9TPaxuWM> (Weston)


#### ["Semi-supervised Sequence Learning"](http://arxiv.org/abs/1511.01432) Dai, Le
>	"We present two approaches that use unlabeled data to improve sequence learning with recurrent networks. The first approach is to predict what comes next in a sequence, which is a conventional language model in natural language processing. The second approach is to use a sequence autoencoder, which reads the input sequence into a vector and predicts the input sequence again. These two algorithms can be used as a “pretraining” step for a later supervised sequence learning algorithm. In other words, the parameters obtained from the unsupervised step can be used as a starting point for other supervised training models. In our experiments, we find that long short term memory recurrent networks after being pretrained with the two approaches are more stable and generalize better. With pretraining, we are able to train long short term memory recurrent networks up to a few hundred timesteps, thereby achieving strong performance in many text classification tasks, such as IMDB, DBpedia and 20 Newsgroups."

>	"Recurrent neural networks are powerful tools for modeling sequential data, yet training them by back-propagation through time can be difficult. For that reason, RNNs have rarely been used for natural language processing tasks such as text classification despite their powers in representing sequential structures. In our experiments on document classification with 20 Newsgroups and DBpedia, and sentiment analysis with IMDB and Rotten Tomatoes, LSTMs pretrained by recurrent language models or sequence autoencoders are usually better than LSTMs initialized randomly. This pretraining helps LSTMs reach or surpass previous baselines on these datasets without additional data. Another important result from our experiments is that using more unlabeled data from related tasks in the pretraining can improve the generalization of a subsequent supervised model. For example, using unlabeled data from Amazon reviews to pretrain the sequence autoencoders can improve classification accuracy on Rotten Tomatoes from 79.7% to 83.3%, an equivalence of adding substantially more labeled data. This evidence supports the thesis that it is possible to use unsupervised learning with more unlabeled data to improve supervised learning. With sequence autoencoders, and outside unlabeled data, LSTMs are able to match or surpass previously reported results."

>	"We believe our semi-supervised approach has some advantages over other unsupervised sequence learning methods, e.g., Paragraph Vectors, because it can allow for easy fine-tuning. Our semi-supervised learning approach is related to Skip-Thought vectors, with two differences. The first difference is that Skip-Thought is a harder objective, because it predicts adjacent sentences. The second is that Skip-Thought is a pure unsupervised learning algorithm, without fine-tuning."

  - `video` <http://youtu.be/KmOdBS4BXZ0?t=1h2m16s> (Le)


#### ["Character-level Convolutional Networks for Text Classification"](http://arxiv.org/abs/1509.01626) Zhang, Zhao, LeCun
>	"This article offers an empirical exploration on the use of character-level convolutional networks (ConvNets) for text classification. We constructed several large-scale datasets to show that character-level convolutional networks could achieve state-of-the-art or competitive results. Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF variants, and deep learning models such as word-based ConvNets and recurrent neural networks."

>	"In most neural net NLP systems, the first layer maps words to vectors through a (learned) lookup table. Our system just looks at individual characters. This makes it robust to misspelling and sensitive to morphology, i.e. an known word with an unknown prefix or suffix, or an unknown word with a similar morphology as a known one, or a word formed by concatenating several known words, will be handled properly. This allows the system to handle essentially unlimited vocabularies."

  - <https://github.com/mhjabreel/CharCNN>


#### ["A Multiplicative Model for Learning Distributed Text-Based Attribute Representations"](http://arxiv.org/abs/1406.2710) Kiros, Zemel, Salakhutdinov
>	"In this paper we propose a general framework for learning distributed representations of attributes: characteristics of text whose representations can be jointly learned with word embeddings. Attributes can correspond to document indicators (to learn sentence vectors), language indicators (to learn distributed language representations), meta-data and side information (such as the age, gender and industry of a blogger) or representations of authors. We describe a third-order model where word context and attribute vectors interact multiplicatively to predict the next word in a sequence. This leads to the notion of conditional word similarity: how meanings of words change when conditioned on different attributes. We perform several experimental tasks including sentiment classification, cross-lingual document classification, and blog authorship attribution. We also qualitatively evaluate conditional word neighbours and attribute-conditioned text generation."



---
### interesting papers - word sequence labelling


#### ["Recurrent Neural Networks for Language Understanding"](http://research.microsoft.com/apps/pubs/?id=200236) Yao, Zweig, Hwang, Shi, Yu
>	"Recurrent Neural Network Language Models have recently shown exceptional performance across a variety of applications. In this paper, we modify the architecture to perform Language Understanding, and advance the state-of-the-art for the widely used ATIS dataset. The core of our approach is to take words as input as in a standard RNN-LM, and then to predict slot labels rather than words on the output side. We present several variations that differ in the amount of word context that is used on the input side, and in the use of non-lexical features. Remarkably, our simplest model produces state-of-the-art results, and we advance state-of-the-art through the use of bag-of-words, word embedding, named-entity, syntactic, and word-class features. Analysis indicates that the superior performance is attributable to the task-specific word representations learned by the RNN."


#### ["Using Recurrent Neural Networks for Slot Filling in Spoken Language Understanding"](http://www.iro.umontreal.ca/~lisa/pointeurs/taslp_RNNSLU.R1.pdf) Mesnil et al.
>	"Semantic slot filling is one of the most challenging problems in spoken language understanding. In this study, we propose to use recurrent neural networks for this task, and present several novel architectures designed to efficiently model past and future temporal dependencies. Specifically, we implemented and compared several important RNN architectures, including Elman, Jordan and hybrid variants. To facilitate reproducibility, we implemented these networks with the publicly available Theano neural network toolkit and completed experiments on the well-known airline travel information system benchmark. In addition, we compared the approaches on two custom SLU data sets from the entertainment and movies domains. Our results show that the RNN-based models outperform the conditional random field baseline by 2% in absolute error reduction on the ATIS benchmark. We improve the state-of-the-art by 0.5% in the Entertainment domain, and 6.7% for the movies domain."

>	"We carried out comprehensive investigations of RNNs for the task of slot filling in SLU. We implemented and compared several RNN architectures, including the Elman-type and Jordan-type networks with their variants. We also studied the effectiveness of word embeddings for slot filling. To make the results easy to reproduce and to compare, we implemented all networks on the common Theano neural network toolkit, and evaluated them on the ATIS benchmark. Our results show that both Elman and Jordan-type networks outperform the CRF baseline substantially, both giving similar performance. A bidirectional version of the Jordan-RNN gave the best performance, outperforming the CRF-based baseline by 14% in relative error reduction. Future work will explore more efficient training of RNNs and the choice of more comprehensive features [28] and using a different RNN training toolkit [14] incorporating more advanced features."

  - <http://deeplearning.net/tutorial/rnnslu.html> + <https://github.com/mesnilgr/is13>


#### ["Recurrent Conditional Random Field for Language Understanding"](http://research.microsoft.com/apps/pubs/default.aspx?id=210167) Yao, Peng, Zweig, Yu, Li, Gao
>	"Recurrent neural networks have recently produced record setting performance in language modeling and word-labeling tasks. In the word-labeling task, the RNN is used analogously to the more traditional conditional random field to assign a label to each word in an input sequence, and has been shown to significantly outperform CRFs. In contrast to CRFs, RNNs operate in an online fashion to assign labels as soon as a word is seen, rather than after seeing the whole word sequence. In this paper, we show that the performance of an RNN tagger can be significantly improved by incorporating elements of the CRF model; specifically, the explicit modeling of output-label dependencies with transition features, its global sequence-level objective function, and offline decoding. We term the resulting model a “recurrent conditional random field” and demonstrate its effectiveness on the ATIS travel domain dataset and a variety of web-search language understanding datasets."


#### ["Bidirectional LSTM-CRF Models for Sequence Tagging"](http://arxiv.org/abs/1508.01991) Huang, Xu, Yu
>	"In this paper, we propose a variety of Long Short-Term Memory based models for sequence tagging. These models include LSTM networks, bidirectional LSTM networks, LSTM with a Conditional Random Field layer and bidirectional LSTM with a CRF layer. Our work is the first to apply a bidirectional LSTM CRF model to NLP benchmark sequence tagging data sets. We show that the BILSTM-CRF model can efficiently use both past and future input features thanks to a bidirectional LSTM component. It can also use sentence level tag information thanks to a CRF layer. The BI-LSTMCRF model can produce state of the art (or close to) accuracy on POS, chunking and NER data sets. In addition, it is robust and has less dependence on word embedding as compared to previous observations."

>	"Our work is close to the work of (Collobert et al., 2011) as both of them utilized deep neural networks for sequence tagging. While their work used convolutional neural networks, ours used bidirectional LSTM networks."

>	"In this paper, we systematically compared the performance of LSTM networks based models for sequence tagging. We presented the first work of applying a BI-LSTM-CRF model to NLP benchmark sequence tagging data. Our model can produce state of the art (or close to) accuracy on POS, chunking and NER data sets. In addition, our model is robust and it has less dependence on word embedding as compared to the observation in (Collobert et al., 2011). It can achieve accurate tagging accuracy without resorting to word embedding."


#### ["Segmental Recurrent Neural Networks"](http://arxiv.org/abs/1511.06018) Kong, Dyer, Smith
>	"We introduce segmental recurrent neural networks which define, given an input sequence, a joint probability distribution over segmentations of the input and labelings of the segments. Representations of the input segments (i.e., contiguous subsequences of the input) are computed by encoding their constituent tokens using bidirectional recurrent neural nets, and these “segment embeddings” are used to define compatibility scores with output labels. These local compatibility scores are integrated using a global semi-Markov conditional random field. Both fully supervised training - in which segment boundaries and labels are observed - as well as partially supervised training - in which segment boundaries are latent - are straightforward. Experiments on handwriting recognition and joint Chinese word segmentation/POS tagging show that, compared to models that do not explicitly represent segments such as BIO tagging schemes and connectionist temporal classification, SRNNs obtain substantially higher accuracies."

>	"We have proposed a new model for segment labeling problems that learns representations of segments of an input sequence and then labels these. We outperform existing alternatives both when segmental information should be recovered and when it is only latent. We have not trained the segmental representations to be of any use beyond making good labeling (or segmentation) decisions, but an intriguing avenue for future work would be to construct representations that are useful for other tasks."

>	"Segmental labeling problems have been widely studied. A widely used approach to a segmental labeling problems with neural networks is the connectionist temporal classification objective and decoding rule of Graves et al. (2006) CTC reduces the “segmental” sequence label problem to a classical sequence labeling problem in which every position in an input sequence x is explicitly labeled by interpreting repetitions of input labels - or input labels followed by a special “blank” output symbol - as being a single label with a longer duration. During training, the marginal likelihood of the set of labelings compatible (according to the CTC interpretation rules) with the reference label y is maximized. Although CTC has been used successfully and its reuse of conventional sequence labeling architectures is appealing, it has several potentially serious limitations. First, it is not possible to model interlabel dependencies explicitly - these must instead be captured indirectly by the underlying RNNs. Second, CTC has no explicit segmentation model. Although this is most serious in applications where segmentation is a necessary/desired output (e.g., information extraction, protein secondary structure prediction), we argue that explicit segmentation is potentially valuable even when the segmentation is not required. To illustrate the value of explicit segments, consider the problem of phone recognition. For this task, segmental duration is strongly correlated with label identity (e.g., while an [o] phone token might last 300ms, it is unlikely that a [t] would) and thus modeling it explicitly may be useful. Finally, making an explicit labeling decision for every position (and introducing a special blank symbol) in an input sequence is conceptually unappealing."


#### ["End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF"](http://arxiv.org/abs/1603.01354) Ma, Hovy
>	"State-of-the-art sequence labeling systems traditionally require large amounts of task-specific knowledge in the form of handcrafted features and data pre-processing. In this paper, we introduce a novel neural network architecture that benefits from both word- and character-level representations automatically, by using combination of bidirectional LSTM, CNN and CRF. Our system is truly end-to-end, requiring no feature engineering or data preprocessing, thus making it applicable to a wide range of sequence labeling tasks on different languages. We evaluate our system on two data sets for two sequence labeling tasks — Penn Treebank WSJ corpus for part-of-speech tagging and CoNLL 2003 corpus for named entity recognition. We obtain state-of-the-art performance on both datasets - 97.55% accuracy for POS tagging and 91.21% F1 for NER."



---
### interesting papers - coreference resolution


#### ["A Joint Model for Entity Analysis: Coreference, Typing, and Linking"](http://www.eecs.berkeley.edu/~gdurrett/papers/durrett-klein-tacl2014.pdf) Durrett, Klein
>	"We present a joint model of three core tasks in the entity analysis stack: coreference resolution (within-document clustering), named entity recognition (coarse semantic typing), and entity linking (matching to Wikipedia entities). Our model is formally a structured conditional random field. Unary factors encode local features from strong baselines for each task. We then add binary and ternary factors to capture cross-task interactions, such as the constraint that coreferent mentions have the same semantic type. On the ACE 2005 and OntoNotes datasets, we achieve state-of-the-art results for all three tasks. Moreover, joint modeling improves performance on each task over strong independent baselines."

  - <http://nlp.cs.berkeley.edu/projects/entity.shtml>
  - `video` <http://techtalks.tv/talks/a-joint-model-for-entity-analysis-coreference-typing-and-linking/61534/>


#### ["Learning Anaphoricity and Antecedent Ranking Features for Coreference Resolution"](http://people.seas.harvard.edu/~srush/acl15.pdf) Wiseman, Rush, Shieber, Weston
>	"We introduce a simple, non-linear mention-ranking model for coreference resolution that attempts to learn distinct feature representations for anaphoricity detection and antecedent ranking, which we encourage by pre-training on a pair of corresponding subtasks. Although we use only simple, unconjoined features, the model is able to learn useful representations, and we report the best overall score on the CoNLL 2012 English test set to date."

>	"In this work, we propose a data-driven model for coreference that does not require prespecifying any feature relationships. Inspired by recent work in learning representations for natural language tasks (Collobert et al., 2011), we explore neural network models which take only raw, unconjoined features as input, and attempt to learn intermediate representations automatically. In particular, the model we describe attempts to create independent feature representations useful for both detecting the anaphoricity of a mention (that is, whether or not a mention is anaphoric) and ranking the potential antecedents of an anaphoric mention. Adequately capturing anaphoricity information has long been thought to be an important aspect of the coreference task, since a strong non-anaphoric signal might, for instance, discourage the erroneous prediction of an antecedent for a non-anaphoric mention even in the presence of a misleading head match. We furthermore attempt to encourage the learning of the desired feature representations by pretraining the model’s weights on two corresponding subtasks, namely, anaphoricity detection and antecedent ranking of known anaphoric mentions."

>	"We have presented a simple, local model capable of learning feature representations useful for coreference-related subtasks, and of thereby achieving state-of-the-art performance. Because our approach automatically learns intermediate representations given raw features, directions for further research might alternately explore including additional (perhaps semantic) raw features, as well as developing loss functions that further discourage learning representations that allow for common errors (such as those involving pleonastic pronouns)."

  - `code` <https://github.com/swiseman/nn_coref>


#### ["End-to-end Neural Coreference Resolution"](https://arxiv.org/abs/1707.07045) Lee, He, Lewis, Zettlemoyer
>	"We introduce the first end-to-end coreference resolution model and show that it significantly outperforms all previous work without using a syntactic parser or hand-engineered mention detector. The key idea is to directly consider all spans in a document as potential mentions and learn distributions over possible antecedents for each. The model computes span embeddings that combine context-dependent boundary representations with a head-finding attention mechanism. It is trained to maximize the marginal likelihood of gold antecedent spans from coreference clusters and is factored to enable aggressive pruning of potential mentions. Experiments demonstrate state-of-the-art performance, with a gain of 1.5 F1 on the OntoNotes benchmark and by 3.1 F1 using a 5-model ensemble, despite the fact that this is the first approach to be successfully trained with no external resources."

  - <https://kentonl.github.io/e2e-coref/> (demo)
  - `video` <https://ku.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=0954a17c-2702-4d8e-9412-12ae958a2790&start=5660> (Lee)
  - `code` <https://github.com/kentonl/e2e-coref>



---
### interesting papers - text summarization


#### ["Modelling, Visualising and Summarising Documents with a Single Convolutional Neural Network"](http://arxiv.org/abs/1406.3830) Denil, Demiraj, Kalchbrenner, Blunsom, Freitas
>	"Capturing the compositional process which maps the meaning of words to that of documents is a central challenge for researchers in Natural Language Processing and Information Retrieval. We introduce a model that is able to represent the meaning of documents by embedding them in a low dimensional vector space, while preserving distinctions of word and sentence order crucial for capturing nuanced semantics. Our model is based on an extended Dynamic Convolution Neural Network, which learns convolution filters at both the sentence and document level, hierarchically learning to capture and compose low level lexical features into high level semantic concepts. We demonstrate the effectiveness of this model on a range of document modelling tasks, achieving strong results with no feature engineering and with a more compact model. Inspired by recent advances in visualising deep convolution networks for computer vision, we present a novel visualisation technique for our document networks which not only provides insight into their learning process, but also can be interpreted to produce a compelling automatic summarisation system for texts."

  - `code` <https://github.com/mdenil/txtnets>


#### ["Extraction of Salient Sentences from Labelled Documents"](http://arxiv.org/abs/1412.6815) Denil, Demiraj, Freitas
>	"We present a hierarchical convolutional document model with an architecture designed to support introspection of the document structure. Using this model, we show how to use visualisation techniques from the computer vision literature to identify and extract topic-relevant sentences. We also introduce a new scalable evaluation technique for automatic sentence extraction systems that avoids the need for time consuming human annotation of validation data."

  - `code` <https://github.com/mdenil/txtnets>


#### ["Sentence Compression by Deletion with LSTMs"](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43852.pdf) Filippova, Alfonseca, Colmenares, Kaiser, Vinyals
>	"We present an LSTM approach to deletion-based sentence compression where the task is to translate a sentence into a sequence of zeros and ones, corresponding to token deletion decisions. We demonstrate that even the most basic version of the system, which is given no syntactic information (no PoS or NE tags, or dependencies) or desired compression length, performs surprisingly well: around 30% of the compressions from a large test set could be regenerated. We compare the LSTM system with a competitive baseline which is trained on the same amount of data but is additionally provided with all kinds of linguistic features. In an experiment with human raters the LSTM-based model outperforms the baseline achieving 4.5 in readability and 3.8 in informativeness."

>	"We presented, to our knowledge, a first attempt at building a competitive compression system which is given no linguistic features from the input. The two important components of the system are (1) word embeddings, which can be obtained by anyone either pre-trained, or by running word2vec on a large corpus, and (2) an LSTM model which draws on the very recent advances in research on RNNs. The training data of about two million sentence-compression pairs was collected automatically from the Internet. Our results clearly indicate that a compression model which is not given syntactic information explicitly in the form of features may still achieve competitive performance. The high readability and informativeness scores assigned by human raters support this claim. In the future, we are planning to experiment with more “interesting” paraphrasing models which translate the input not into a zero-one sequence but into words."


#### ["A Neural Attention Model for Abstractive Sentence Summarization"](http://arxiv.org/abs/1509.00685) Rush, Chopra, Weston
>	"Summarization based on text extraction is inherently limited, but generation-style abstractive methods have proven challenging to build. In this work, we propose a fully data-driven approach to abstractive sentence summarization. Our method utilizes a local attention-based model that generates each word of the summary conditioned on the input sentence. While the model is structurally simple, it can easily be trained end-to-end and scales to a large amount of training data. The model shows significant performance gains on the DUC-2004 shared task compared with several strong baselines."

>	"We have presented a neural attention-based model for abstractive summarization, based on recent developments in neural machine translation. We combine this probabilistic model with a generation algorithm which produces accurate abstractive summaries. As a next step we would like to further improve the grammaticality of the summaries in a data-driven way, as well as scale this system to generate paragraph-level summaries. Both pose additional challenges in terms of efficient alignment and consistency in generation."

  - `code` <https://github.com/jaseweston/NAMAS>
  - `code` <https://github.com/carpedm20/neural-summary-tensorflow>


#### ["Sequence-to-Sequence RNNs for Text Summarization"](http://arxiv.org/abs/1602.06023) Nallapati, Xiang, Zhou
>	"In this work, we cast text summarization as a sequence-to-sequence problem and apply the attentional encoder-decoder RNN that has been shown to be successful for Machine Translation (Bahdanau et al. (2014)). Our experiments show that the proposed architecture significantly outperforms the state-of-the art model of Rush et al. (2015) on the Gigaword dataset without any additional tuning. We also propose additional extensions to the standard architecture, which we show contribute to further improvement in performance."


#### ["Incorporating Copying Mechanism in Sequence-to-Sequence Learning"](http://arxiv.org/abs/1603.06393) Gu, Lu, Li, Li
>	"We address an important problem in sequence-to-sequence (Seq2Seq) learning referred to as copying, in which certain segments in the input sequence are selectively replicated in the output sequence. A similar phenomenon is observable in human language communication. For example, humans tend to repeat entity names or even long phrases in conversation. The challenge with regard to copying in Seq2Seq is that new machinery is needed to decide when to perform the operation. In this paper, we incorporate copying into neural network-based Seq2Seq learning and propose a new model called COPYNET with encoder-decoder structure. COPYNET can nicely integrate the regular way of word generation in the decoder with the new copying mechanism which can choose subsequences in the input sequence and put them at proper places in the output sequence. Our empirical study on both synthetic data sets and real world data sets demonstrates the efficacy of COPYNET. For example, COPYNET can outperform regular RNN-based model with remarkable margins on text summarization tasks."

>	"Both the canonical encoder-decoder and its variants with attention mechanism rely heavily on the representation of “meaning”, which might not be sufficiently accurate in cases in which the system needs to refer to sub-sequences of input like entity names or dates. In contrast, the copying mechanism is closer to the rote memorization in language processing of human being, deserving a different modeling strategy in neural network-based models. We argue that it will benefit many Seq2Seq tasks to have an elegant unified model that can accommodate both understanding and rote memorization. Towards this goal, we propose COPYNET, which is not only capable of the regular generation of words but also the operation of copying appropriate segments of the input sequence. Despite the seemingly “hard” operation of copying, COPYNET can be trained in an end-to-end fashion."

>	"Our work is partially inspired by the recent work of Pointer Networks, in which a pointer mechanism (quite similar with the proposed copying mechanism) is used to predict the output sequence directly from the input. In addition to the difference with ours in application, Pointer Networks cannot predict outside of the set of input sequence, while COPYNET can naturally combine generating and copying. COPYNET is also related to the effort to solve the OOV problem in neural machine translation. Luong et al. (2015) introduced a heuristics to postprocess the translated sentence using annotations on the source sentence. In contrast COPYNET addresses the OOV problem in a more systemic way with an end-to-end model. However, as COPYNET copies the exact source words as the output, it cannot be directly applied to machine translation. For future work, we will extend this idea to the task where the source and target are in different languages, for example, machine translation."

>	"The copying mechanism can also be viewed as carrying information over to the next stage without any nonlinear transformation. Similar ideas are proposed for training very deep neural networks in (Srivastava et al., 2015; He et al., 2015) for classification tasks, where shortcuts are built between layers for the direct carrying of information."


#### ["Neural Network-Based Abstract Generation for Opinions and Arguments"](http://www.ccs.neu.edu/home/luwang/papers/NAACL2016.pdf) Wang, Ling
>	"We study the problem of generating abstractive summaries for opinionated text. We propose an attention-based neural network model that is able to absorb information from multiple text units to construct informative, concise, and fluent summaries. An importance-based sampling method is designed to allow the encoder to integrate information from an important subset of input. Automatic evaluation indicates that our system outperforms state-of-the-art abstractive and extractive summarization systems on two newly collected datasets of movie reviews and arguments. Our system summaries are also rated as more informative and grammatical in human evaluation."
