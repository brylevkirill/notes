  Knowledge is in our minds and language is just orienting us within our shared experiences.  
  Language is an index pointing to shared experiences of people on which meaning is grounded.  


  * [**overview**](#overview)
  * [**study**](#study)
  * [**problems**](#problems)
  * [**grounding**](#grounding)
  * [**semantics**](#semantics)
  * [**compositionality**](#compositionality)
  * [**embeddings**](#embeddings)
  * [**interesting papers**](#interesting-papers)
    - [**language grounding**](#interesting-papers---language-grounding)
    - [**language modeling**](#interesting-papers---language-modeling)
    - [**sequence transduction**](#interesting-papers---sequence-transduction)
    - [**machine translation**](#interesting-papers---machine-translation)
    - [**word embeddings**](#interesting-papers---word-embeddings)
    - [**word sense disambiguation**](#interesting-papers---word-sense-disambiguation)
    - [**text embeddings**](#interesting-papers---text-embeddings)
    - [**syntactic parsing**](#interesting-papers---syntactic-parsing)
    - [**text classification**](#interesting-papers---text-classification)
    - [**topic modeling**](#interesting-papers---topic-modeling)
    - [**text summarization**](#interesting-papers---text-summarization)
    - [**coreference resolution**](#interesting-papers---coreference-resolution)



---
### overview

  "All existing NLP is about mapping the internal statistical dependencies of language, missing the point that language is a communication protocol. You cannot study language without considering agents communicating about something. The only reason language even has any statistical dependencies to study is because it's imperfect. A maximally efficient communication protocol would look like random noise, out of context (besides error correction mechanisms). All culture is a form of communication, so "understanding" art requires grounding. Mimicking what humans do isn't enough. You can't understand language without considering it in context: agents communicating about something. An analogy could be trying to understand an economy by looking at statistical structure in stock prices only."

  *(Francois Chollet)*

  "No general solution to the problem of computer understanding of natural language is possible, i.e. language is understood only in contextual frameworks, that even these can be shared by people to only a limited extent, and that consequently even people are not embodiments of any such general solution."

  *(Joseph Weizenbaum)*

  "The ambiguous nature of natural language might seem like a flaw, but in fact, it is exactly this ambiguity that makes natural language so powerful. Think of language as a game between a speaker and a listener. Game play proceeds as follows: the speaker thinks of a concept, she chooses an utterance to convey that concept, and the listener interprets the utterance. Both players win if the listener’s interpretation matches the speaker’s intention. To play this game well, the speaker should thus choose the simplest utterance that conveys her intended concept - anything the listener can infer can be omitted. How can a computer fill in these gaps, which depend on the breadth of human experience involving perception of the world and social interactions?"

  *(Percy Liang)*

  "The language is meant to serve for communication between a builder A and an assistant B. A is building with building-stones: there are blocks, pillars, slabs and beams. B has to pass the stones, and that in the order in which A needs them. For this purpose they use a language consisting of the words “block”, “pillar”, “slab”, “beam”. A calls them out - B brings the stone which he has learnt to bring at such-and-such a call. Conceive this as a complete primitive language."

  *(Ludwig Wittgenstein)*

  "This game is not just a game of words but a game of words causing things and of other things causing words. We can’t fully define the meaning of a word like “slab” without referring to the physical actions of the builder A and the assistant B. In this way, linguistic meaning has to bottom out at some point in nonlinguistic facts."

  *(Jon Gauthier)*

  "We have a lot of language [data at Google] but it's not annotated with what it means, and we don't even know how we would annotate it. That's a big challenge now."

  *(Ray Kurzweil)* ([talk](https://youtube.com/watch?v=w9sz7eW6MY8#t=4m27s) `video`)



---
### study

  **classic NLP**

  [course](http://youtube.com/playlist?list=PL6397E4B26D00A269) by Chris Manning and Dan Jurafsky `video` ([slides](https://web.stanford.edu/~jurafsky/NLPCourseraSlides.html))  
  [course](http://youtube.com/user/afigfigueira/playlists?shelf_id=5&view=50) by Michael Collins `video`  
  [course](http://youtube.com/playlist?list=PLegWUnz91WfuPebLI97-WueAP90JO-15i) by Jordan Boyd-Graber `video`  
  [course](https://stepik.org/course/1233) by Pavel Braslavski `video` `in russian`  

----

  **NLP with Deep Learning**

  [course](http://youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z) from Stanford `video` ([notes](https://github.com/stanfordnlp/cs224n-winter17-notes))  
  [course](http://github.com/oxford-cs-deepnlp-2017/lectures) from Oxford and DeepMind `video`  
  [course](http://phontron.com/class/nn4nlp2019/schedule.html) from CMU ([videos](https://youtube.com/user/neubig/playlists))  
  [course](http://youtube.com/playlist?list=PLt1IfGj6-_-f55ULcae3v7YuG8p_eUjnk) from iPavlov and Yandex `video` `in russian`  

----

  ["Statistical NLP"](https://uclmr.github.io/stat-nlp-book) interactive lecture notes, slides and exercises from UCL Machine Reading group  
  ["Natural Language Processing"](https://github.com/jacobeisenstein/gt-nlp-class/) notes and slides from Georgia Tech  

----

  ["Speech and Language Processing"](http://web.stanford.edu/~jurafsky/slp3/) by Dan Jurafsky and James Martin `book`  
  ["Foundations of Statistical Natural Language Processing"](http://nlp.stanford.edu/fsnlp/) by Chris Manning and Hinrich Schutze `book`  
  ["Neural Network Methods for Natural Language Processing"](http://morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037) by Yoav Goldberg `book`  

----

  [blog](http://ruder.io/index.html#open) by Sebastian Ruder

----

  ["NLP News"](http://newsletter.ruder.io) newsletter from Sebastian Ruder  
  ["NLP Highlights"](https://soundcloud.com/nlp-highlights) podcast from AI2  



---
### problems

  [state-of-the-art methods](https://github.com/sebastianruder/NLP-progress)

  ["NLP’s generalization problem, and how researchers are tackling it"](https://thegradient.pub/frontiers-of-generalization-in-natural-language-processing) by Ana Marasovic

----

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
### grounding

  ["A Paradigm for Situated and Goal-Driven Language Learning"](#a-paradigm-for-situated-and-goal-driven-language-learning-gauthier-mordatch) by Jon Gauthier and Igor Mordatch `paper` `summary`  
>	"We outlined a paradigm for grounded and goal-driven language learning in artificial agents. The paradigm is centered around a utilitarian definition of language understanding, which equates language understanding with the ability to cooperate with other language users in real-world environments. This position demotes language from its position as a separate task to be solved to one of several communicative tools agents might use to accomplish their real-world goals."

  ["On 'Solving Language'"](http://foldl.me/2016/solving-language/) by Jon Gauthier  
  ["Situated Language Learning"](http://foldl.me/2016/situated-language-learning/) by Jon Gauthier  

  ["From Models of Language Understanding to Agents of Language Use"](http://research.microsoft.com/apps/video/default.aspx?id=266643) by Felix Hill `video`

  [**interesting papers - language grounding**](#interesting-papers---language-grounding)



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

  "Compositionality is central to characterizing the ways in which small changes to a syntactic structure can yield profound changes in meaning. For instance, “two minus three” and “three minus two” contain the same words but lead to different denotations. Superficially least contentful words (“every”, “no”, “not”, “might”, “but”, etc.) often drive the most dramatic effects depending on where they appear in the constituent structure, which determines what their corresponding semantic arguments are. The grammar gives a precise account of how this interpretive consequence follows from the syntax of the two utterances."

  "Compositionality is often linked to our ability to produce and interpret novel utterances. While it is too strong to say that compositionality is necessary or sufficient for this kind of creative ability, it does help characterize it: once one has acquired the syntax of the language, memorized all the lexical meanings, and mastered the few modes of composition, one can interpret novel combinations of them. This abstract capacity is at the heart of what computational models of semantics would like to learn, so that they too can efficiently interpret novel combinations of words and phrases."

  "The problem is that human language is much more fluid than old-school, Chomskyan grammars can support. Even limiting research to just the English language, there are so many variations and so many mutations that the lifetime of a rigid, formally defined grammar becomes too short to be feasible. For a language processing system to have meaning in the real world, it must be able to update and change to reflect change in the language. Logic-based systems would be much more interesting - and, certainly, such systems are necessary if we ever want to build a system that understands instead of just translates. However, the system must be able to learn the rules of the language on the fly."

  "It's important to make a distinction between (1) Chomskyan linguistics, (2) 90s style symbolic systems, (3) 90s/early 2000s style statistical systems and (4) 2010s style statistical systems. Chomskyan linguistics assumes that statistics and related stuff is not relevant at all, and that instead you need to find the god-given (or at least innate) Universal Grammar and then everything will be great. 90s style symbolic systems adopt a more realistic approach, relying on lots of heuristics that kind of work but aim at good performance rather than unattainable perfection. 90s style statistical models give up some of the insights in these heuristics to construct tractable statistical models. If you look at 2010s style statistical models, you'll notice that machine learning has become more powerful and you can use a greater variety of information, either using good linguistic intuitions or unsupervised/deep-NN learning, which constructs generalizations over features."

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

  - *constituency parsing*  
	[overview](http://lxmls.it.pt/2016/Part_1_Constituency_Parsing_2016.pdf) by Slav Petrov `slides`  
	[overview](http://youtube.com/watch?v=sL_W_I8DpuU) by John Boyd-Graber `video`  

  - *dependency parsing*  
	[overview](http://lxmls.it.pt/2016/Part2_Dependency_Parsing_2016.pdf) by Slav Petrov `slides`  
	[overview](http://youtube.com/watch?v=du9VQaFEyeA) by John Boyd-Graber `video`  

	[overview](http://youtube.com/watch?v=zgd_ZuaWvfs) of transition-based parsing by Graham Neubig `video`  
	[overview](http://youtube.com/watch?v=wUjIOxeCnqM) of graph-based parsing by Graham Neubig `video`  

  - *CCG parsing*  
	[overview](http://youtube.com/playlist?list=PLun-LUE1uLNvWi-qV-tRHohfHR90Y_cAk) by Yoav Artzi `video`

----

  [English grammar explained](http://chompchomp.com/terms/)



---
### embeddings

  [overview](http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/) of word embeddings by Sebastian Ruder  
  "On Word Embeddings" by Sebastian Ruder ([part 1](http://sebastianruder.com/word-embeddings-1/index.html), [part 2](http://sebastianruder.com/word-embeddings-softmax/index.html))  

  ["Word Vector Representations: word2vec"](https://youtube.com/watch?v=ERibwqs9p38) by Richard Socher `video`  
  ["GloVe: Global Vectors for Word Representation"](https://youtube.com/watch?v=ASn7ExxLZws) by Richard Socher `video`  
  ["Vector Representations of Words and Documents"](https://youtube.com/watch?v=KEXWC-ICH_Y) by Anna Potapenko `video` `in russian`  

  ["A Theoretical Approach to Semantic Representations"](https://youtube.com/watch?v=KR46z_V0BVw) by Sanjeev Arora `video`  
  "Word Embeddings: Explaining their properties" by Sanjeev Arora
	([part 1](http://www.offconvex.org/2015/12/12/word-embeddings-1/), [part 2](http://www.offconvex.org/2016/02/14/word-embeddings-2/))  

  ["Are Distributional Representations Ready for the Real World? Evaluating Word Vectors for Grounded Perceptual Meaning"](#are-distributional-representations-ready-for-the-real-world-evaluating-word-vectors-for-grounded-perceptual-meaning-lucy-gauthier) by Lucy and Gauthier `paper` `summary`

----

  ["Deep Learning for Semantic Composition"](https://vimeo.com/234950059) tutorial by Xiaodan Zhu and Edward Grefenstette `video`

----

  [**distributed representation**](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#architectures---distributed-representation)



---
### interesting papers

  - [**language grounding**](#interesting-papers---language-grounding)
  - [**language modeling**](#interesting-papers---language-modeling)
  - [**sequence transduction**](#interesting-papers---sequence-transduction)
  - [**machine translation**](#interesting-papers---machine-translation)
  - [**word embeddings**](#interesting-papers---word-embeddings)
  - [**word sense disambiguation**](#interesting-papers---word-sense-disambiguation)
  - [**text embeddings**](#interesting-papers---text-embeddings)
  - [**syntactic parsing**](#interesting-papers---syntactic-parsing)
  - [**text classification**](#interesting-papers---text-classification)
  - [**topic modeling**](#interesting-papers---topic-modeling)
  - [**text summarization**](#interesting-papers---text-summarization)
  - [**coreference resolution**](#interesting-papers---coreference-resolution)


[**interesting recent papers**](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#natural-language-processing)  



---
### interesting papers - language grounding

[**interesting recent papers**](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#language-grounding)


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
  - `paper` ["A Paradigm for Situated and Goal-Driven Language Learning"](#a-paradigm-for-situated-and-goal-driven-language-learning-gauthier-mordatch) by Gauthier and Mordatch `summary`


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



---
### interesting papers - language modeling

#### ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) Radford, Wu, Child, Luan, Amodei, Sutskever
  `language modeling` `GPT-2`
>	"Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets. We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText. When conditioned on a document plus questions, the answers generated by the language model reach 55 F1 on the CoQA dataset - matching or exceeding the performance of 3 out of 4 baseline systems without using the 127,000+ training examples. The capacity of the language model is essential to the success of zero-shot task transfer and increasing it improves performance in a log-linear fashion across tasks. Our largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the art results on 7 out of 8 tested language modeling datasets in a zero-shot setting but still underfits WebText."  
>	"Generative Pre-Training"  
  - `post` <https://openai.com/blog/better-language-models>
  - `video` <https://youtu.be/w3ues-NayAs?t=23m15s> (Sutskever)
  - `video` <https://youtu.be/X-B3nAN7YRM?t=14m25s> (Sutskever)
  - `video` <https://youtu.be/T0I88NhR_9M?t=6m11s> (Sutskever)
  - `video` <https://youtu.be/oaQSQsjhcwQ> + <https://youtu.be/9dSQmXDH--I> + <https://youtu.be/Rbs55YaK7EU> (Tatman)
  - `video` <https://youtube.com/watch?v=u1_qMdb0kYU> (Kilcher)
  - `video` <https://youtube.com/watch?v=n_UlVuFAzbU> (Amjadian)
  - `video` <https://youtu.be/qIEfJ6OBGj8?t=14m25s> (Hinton)
  - `post` <https://jalammar.github.io/illustrated-gpt2>

#### ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805) Devlin, Chang, Lee, Toutanova
  `language modeling` `BERT`
>	"Unlike recent language representation models (Peters et al., 2018; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications."  
  - `post` <https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html>
  - `video` <https://youtu.be/AC4l_MY2Dhc?t=1h51m52s> (Srinivas)
  - `video` <https://youtu.be/BaPM47hO8p8> + <https://youtu.be/yqwnehmB9uQ> + <https://youtu.be/IF6xV0RI91s> + <https://youtu.be/4qDJwq42q-k> (Tatman)
  - `video` <https://youtube.com/watch?v=9-Om2YcHhIA> (Li)
  - `video` <https://youtube.com/watch?v=BhlOGGzC0Q0> + <https://youtube.com/watch?v=rMQMHA-uv_E> (Gharbieh)
  - `video` <https://youtube.com/watch?v=BhlOGGzC0Q0> (Luo)
  - `video` <https://youtu.be/99CymZY-4Tc?t=5m36s> (Lyalin) `in russian`
  - `post` <http://jalammar.github.io/illustrated-bert>
  - `post` <https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3> + <https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73> + <https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f>
  - `paper` ["Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference"](https://arxiv.org/abs/1902.01007) by McCoy et al.
  - `paper` ["Probing Neural Network Comprehension of Natural Language Arguments"](https://arxiv.org/abs/1907.07355) by Niven et al.
  - `paper` ["BERT Rediscovers the Classical NLP Pipeline"](https://arxiv.org/abs/1905.05950) by Tenney et al.
  - `paper` ["XLNet: Generalized Autoregressive Pretraining for Language Understanding"](https://arxiv.org/abs/1906.08237) by Yang et al.



---
### interesting papers - sequence transduction


#### ["Sequence to Sequence Learning with Neural Networks"](http://arxiv.org/abs/1409.3215) Sutskever, Vinyals, Le
  `seq2seq`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#sequence-to-sequence-learning-with-neural-networks-sutskever-vinyals-le>


#### ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
  `Transformer`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#attention-is-all-you-need-vaswani-shazeer-parmar-uszkoreit-jones-gomez-kaiser-polosukhin>


#### ["Sequence Level Training with Recurrent Neural Networks"](http://arxiv.org/abs/1511.06732) Ranzato, Chopra, Auli, Zaremba
  `MIXER`
>	"Many natural language processing applications use language models to generate text. These models are typically trained to predict the next word in a sequence, given the previous words and some context such as an image. However, at test time the model is expected to generate the entire sequence from scratch. This discrepancy makes generation brittle, as errors may accumulate along the way. We address this issue by proposing a novel sequence level training algorithm that directly optimizes the metric used at test time, such as BLEU or ROUGE. On three different tasks, our approach outperforms several strong baselines for greedy generation. The method is also competitive when these baselines employ beam search, while being several times faster."

>	"A wide variety of applications rely on text generation, including machine translation, video/text summarization, question answering, among others. From a machine learning perspective, text generation is the problem of predicting a syntactically and semantically correct sequence of consecutive words given some context. For instance, given an image, generate an appropriate caption or given a sentence in English language, translate it into French. Popular choices for text generation models are language models based on n-grams, feed-forward neural networks and recurrent neural networks. These models when used as is to generate text suffer from two major drawbacks. First, they are trained to predict the next word given the previous ground truth words as input. However, at test time, the resulting models are used to generate an entire sequence by predicting one word at a time, and by feeding the generated word back as input at the next time step. This process is very brittle because the model was trained on a different distribution of inputs, namely, words drawn from the data distribution, as opposed to words drawn from the model distribution. As a result the errors made along the way will quickly accumulate. We refer to this discrepancy as exposure bias which occurs when a model is only exposed to the training data distribution, instead of its own predictions. Second, the loss function used to train these models is at the word level. A popular choice is the cross-entropy loss used to maximize the probability of the next correct word. However, the performance of these models is typically evaluated using discrete metrics. One such metric is called BLEU for instance, which measures the n-gram overlap between the model generation and the reference text. Training these models to directly optimize metrics like BLEU is hard because a) these are not differentiable, and b) combinatorial optimization is required to determine which sub-string maximizes them given some context."

>	"This paper proposes a novel training algorithm which results in improved text generation compared to standard models. The algorithm addresses the two issues discussed above as follows. First, while training the generative model we avoid the exposure bias by using model predictions at training time. Second, we directly optimize for our final evaluation metric. We build on the REINFORCE algorithm to achieve the above two objectives. While sampling from the model during training is quite a natural step for the REINFORCE algorithm, optimizing directly for any test metric can also be achieved by it. REINFORCE side steps the issues associated with the discrete nature of the optimization by not requiring rewards (or losses) to be differentiable. While REINFORCE appears to be well suited to tackle the text generation problem, it suffers from a significant issue. The problem setting of text generation has a very large action space which makes it extremely difficult to learn with an initial random policy. Specifically, the search space for text generation is of size O(WT), where W is the number of words in the vocabulary (typically around 10^4 or more) and T is the length of the sentence (typically around 10-30). Towards that end, we introduce Mixed Incremental Cross-Entropy Reinforce. MIXER is an easy-to-implement recipe to make REINFORCE work well for text generation applications. It is based on two key ideas: incremental learning and the use of a hybrid loss function which combines both REINFORCE and cross-entropy. Both ingredients are essential to training with large action spaces. In MIXER, the model starts from the optimal policy given by cross-entropy training (as opposed to a random one), from which it then slowly deviates, in order to make use of its own predictions, as is done at test time."

>	"Our results show that MIXER with a simple greedy search achieves much better accuracy compared to the baselines on Text Summarization, Machine Translation and Image Captioning tasks. In addition we show that MIXER with greedy search is even more accurate than the cross entropy model augmented with beam search at inference time as a post-processing step. This is particularly remarkable because MIXER with greedy search is at least 10 times faster than the cross entropy model with a beam of size 10. Lastly, we note that MIXER and beam search are complementary to each other and can be combined to further improve performance, although the extent of the improvement is task dependent."

  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/RanzatoCAZ15>
  - `code` <https://github.com/facebookresearch/MIXER>


#### ["Reward Augmented Maximum Likelihood for Neural Structured Prediction"](https://arxiv.org/abs/1609.00150) Norouzi, Bengio, Chen, Jaitly, Schuster, Wu, Schuurmans
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#reward-augmented-maximum-likelihood-for-neural-structured-prediction-norouzi-bengio-chen-jaitly-schuster-wu-schuurmans>


#### ["Self-critical Sequence Training for Image Captioning"](https://arxiv.org/abs/1612.00563) Rennie, Marcheret, Mroueh, Ross, Goel
>	"Recently it has been shown that policy-gradient methods for reinforcement learning can be utilized to train deep end-to-end systems directly on non-differentiable metrics for the task at hand. In this paper we consider the problem of optimizing image captioning systems using reinforcement learning, and show that by carefully optimizing our systems using the test metrics of the MSCOCO task, significant gains in performance can be realized. Our systems are built using a new optimization approach that we call self-critical sequence training. SCST is a form of the popular REINFORCE algorithm that, rather than estimating a “baseline” to normalize the rewards and reduce variance, utilizes the output of its own test-time inference algorithm to normalize the rewards it experiences. Using this approach, estimating the reward signal (as actor-critic methods must do) and estimating normalization (as REINFORCE algorithms typically do) is avoided, while at the same time harmonizing the model with respect to its test-time inference procedure. Empirically we find that directly optimizing the CIDEr metric with SCST and greedy decoding at test-time is highly effective. Our results on the MSCOCO evaluation sever establish a new state-of-the-art on the task, improving the best result in terms of CIDEr from 104.9 to 112.3."

>	"REINFORCE with reward normalization but without baseline estimation"  

  - `video` <https://youtube.com/watch?v=UnT5wTe13yc> (Rennie)
  - `video` <https://yadi.sk/i/-U5w4NpJ3H5TWD> (Ratnikov) `in russian`
  - `post` <http://artem.sobolev.name/posts/2017-11-12-stochastic-computation-graphs-fixing-reinforce.html>


#### ["Segmental Recurrent Neural Networks"](http://arxiv.org/abs/1511.06018) Kong, Dyer, Smith
>	"We introduce segmental recurrent neural networks which define, given an input sequence, a joint probability distribution over segmentations of the input and labelings of the segments. Representations of the input segments (i.e., contiguous subsequences of the input) are computed by encoding their constituent tokens using bidirectional recurrent neural nets, and these “segment embeddings” are used to define compatibility scores with output labels. These local compatibility scores are integrated using a global semi-Markov conditional random field. Both fully supervised training - in which segment boundaries and labels are observed - as well as partially supervised training - in which segment boundaries are latent - are straightforward. Experiments on handwriting recognition and joint Chinese word segmentation/POS tagging show that, compared to models that do not explicitly represent segments such as BIO tagging schemes and connectionist temporal classification, SRNNs obtain substantially higher accuracies."

>	"We have proposed a new model for segment labeling problems that learns representations of segments of an input sequence and then labels these. We outperform existing alternatives both when segmental information should be recovered and when it is only latent. We have not trained the segmental representations to be of any use beyond making good labeling (or segmentation) decisions, but an intriguing avenue for future work would be to construct representations that are useful for other tasks."

>	"Segmental labeling problems have been widely studied. A widely used approach to a segmental labeling problems with neural networks is the connectionist temporal classification objective and decoding rule of Graves et al. (2006) CTC reduces the “segmental” sequence label problem to a classical sequence labeling problem in which every position in an input sequence x is explicitly labeled by interpreting repetitions of input labels - or input labels followed by a special “blank” output symbol - as being a single label with a longer duration. During training, the marginal likelihood of the set of labelings compatible (according to the CTC interpretation rules) with the reference label y is maximized. Although CTC has been used successfully and its reuse of conventional sequence labeling architectures is appealing, it has several potentially serious limitations. First, it is not possible to model interlabel dependencies explicitly - these must instead be captured indirectly by the underlying RNNs. Second, CTC has no explicit segmentation model. Although this is most serious in applications where segmentation is a necessary/desired output (e.g., information extraction, protein secondary structure prediction), we argue that explicit segmentation is potentially valuable even when the segmentation is not required. To illustrate the value of explicit segments, consider the problem of phone recognition. For this task, segmental duration is strongly correlated with label identity (e.g., while an [o] phone token might last 300ms, it is unlikely that a [t] would) and thus modeling it explicitly may be useful. Finally, making an explicit labeling decision for every position (and introducing a special blank symbol) in an input sequence is conceptually unappealing."


#### ["Online Segment to Segment Neural Transduction"](https://arxiv.org/abs/1609.08194) Yu, Buys, Blunsom
>	"We introduce an online neural sequence to sequence model that learns to alternate between encoding and decoding segments of the input as it is read. By independently tracking the encoding and decoding representations our algorithm permits exact polynomial marginalization of the latent segmentation during training, and during decoding beam search is employed to find the best alignment path together with the predicted output sequence. Our model tackles the bottleneck of vanilla encoder-decoders that have to read and memorize the entire input sequence in their fixedlength hidden states before producing any output. It is different from previous attentive models in that, instead of treating the attention weights as output of a deterministic function, our model assigns attention weights to a sequential latent variable which can be marginalized out and permits online generation. Experiments on abstractive sentence summarization and morphological inflection show significant performance gains over the baseline encoder-decoders."

  - `video` <http://techtalks.tv/talks/online-segment-to-segment-neural-transduction/63323/> (Yu)


#### ["The Neural Noisy Channel"](https://arxiv.org/abs/1611.02554) Yu, Blunsom, Dyer, Grefenstette, Kocisky
>	"We formulate sequence to sequence transduction as a noisy channel decoding problem and use recurrent neural networks to parameterise the source and channel models. Unlike direct models which can suffer from explaining-away effects during training, noisy channel models must produce outputs that explain their inputs, and their component models can be trained with not only paired training samples but also unpaired samples from the marginal output distribution. Using a latent variable to control how much of the conditioning sequence the channel model needs to read in order to generate a subsequent symbol, we obtain a tractable and effective beam search decoder. Experimental results on abstractive sentence summarisation, morphological inflection, and machine translation show that noisy channel models outperform direct models, and that they significantly benefit from increased amounts of unpaired output data that direct models cannot easily use."

  - `video` <http://videolectures.net/deeplearning2017_blunsom_language_understanding/#t=4545> (Blunsom)



---
### interesting papers - machine translation


#### ["Sequence to Sequence Learning with Neural Networks"](http://arxiv.org/abs/1409.3215) Sutskever, Vinyals, Le
  `seq2seq`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#sequence-to-sequence-learning-with-neural-networks-sutskever-vinyals-le>


#### ["Neural Machine Translation by Jointly Learning to Align and Translate"](http://arxiv.org/abs/1409.0473) Bahdanau, Cho, Bengio
>	"Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance. The models proposed recently for neural machine translation often belong to a family of encoder–decoders and encodes a source sentence into a fixed-length vector from which a decoder generates a translation. In this paper, we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder–decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly. With this new approach, we achieve a translation performance comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation. Furthermore, qualitative analysis reveals that the (soft-)alignments found by the model agree well with our intuition."

  - `video` <http://slideshot.epfl.ch/play/khnnunGF0elc> (Cho)
  - `video` <https://youtube.com/watch?v=QuvRWevJMZ4> (Chevalier)
  - `video` <https://youtube.com/watch?v=wyQBfi6uOHk> (Chen)
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


#### ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
  `Transformer`
  - <https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md#attention-is-all-you-need-vaswani-shazeer-parmar-uszkoreit-jones-gomez-kaiser-polosukhin>


#### ["Word Translation Without Parallel Data"](https://arxiv.org/abs/1710.04087) Conneau, Lample, Ranzato, Denoyer, Jegou
>	"State-of-the-art methods for learning cross-lingual word embeddings have relied on bilingual dictionaries or parallel corpora. Recent works showed that the need for parallel data supervision can be alleviated with character-level information. While these methods showed encouraging results, they are not on par with their supervised counterparts and are limited to pairs of languages sharing a common alphabet. In this work, we show that we can build a bilingual dictionary between two languages without using any parallel corpora, by aligning monolingual word embedding spaces in an unsupervised way. Without using any character information, our model even outperforms existing supervised methods on cross-lingual tasks for some language pairs. Our experiments demonstrate that our method works very well also for distant language pairs, like English-Russian or EnglishChinese. We finally show that our method is a first step towards fully unsupervised machine translation and describe experiments on the English-Esperanto language pair, on which there only exists a limited amount of parallel data."

>	"Our method leverages adversarial training to learn a linear mapping from a source to a target space and operates in two steps. First, in a two-player game, a discriminator is trained to distinguish between the mapped source embeddings and the target embeddings, while the mapping (which can be seen as a generator) is jointly trained to fool the discriminator. Second, we extract a synthetic dictionary from the resulting shared embedding space and fine-tune the mapping with the closed-form Procrustes solution."

>	"(A) There are two distributions of word embeddings, English words in red denoted by X and Italian words in blue denoted by Y, which we want to align/translate. Each dot represents a word in that space. The size of the dot is proportional to the frequency of the words in the training corpus of that language.  
>	(B) Using adversarial learning, we learn a rotation matrix W which roughly aligns the two distributions. The green stars are randomly selected words that are fed to the discriminator to determine whether the two word embeddings come from the same distribution.  
>	(C) The mapping W is further refined via Procrustes. This method uses frequent words aligned by the previous step as anchor points, and minimizes an energy function that corresponds to a spring system between anchor points. The refined mapping is then used to map all words in the dictionary.  
>	(D) Finally, we translate by using the mapping W and a distance metric, dubbed CSLS, that expands the space where there is high density of points (like the area around the word “cat”), so that “hubs” (like the word “cat”) become less close to other word vectors than they would otherwise (compare to the same region in panel (A))."  

  - `video` <https://youtu.be/Chj8yTm5eGk?t=2m58s> (Lyalin) `in russian`
  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/1710.04087>
  - `code` <https://github.com/facebookresearch/MUSE>


#### ["Unsupervised Neural Machine Translation"](https://arxiv.org/abs/1710.11041) Artetxe, Labaka, Agirre, Cho
>	"In spite of the recent success of neural machine translation in standard benchmarks, the lack of large parallel corpora poses a major practical problem for many language pairs. There have been several proposals to alleviate this issue with, for instance, triangulation and semi-supervised learning techniques, but they still require a strong cross-lingual signal. In this work, we completely remove the need of parallel data and propose a novel method to train an NMT system in a completely unsupervised manner, relying on nothing but monolingual corpora. Our model builds upon the recent work on unsupervised embedding mappings, and consists of a slightly modified attentional encoder-decoder model that can be trained on monolingual corpora alone using a combination of denoising and backtranslation. Despite the simplicity of the approach, our system obtains 15.56 and 10.21 BLEU points in WMT 2014 French → English and German → English translation. The model can also profit from small parallel corpora, and attains 21.81 and 15.24 points when combined with 100,000 parallel sentences, respectively. Our approach is a breakthrough in unsupervised NMT, and opens exciting opportunities for future research."

  - `video` <https://youtu.be/_tQVfq2IOHA?t=25m30s> (Cho)
  - `notes` <http://ankitg.me/blog/2017/11/05/unsupervised-machine-translation.html>
  - `code` <https://github.com/artetxem/undreamt>


#### ["Unsupervised Machine Translation Using Monolingual Corpora Only"](https://arxiv.org/abs/1711.00043) Lample, Denoyer, Ranzato
>	"Machine translation has recently achieved impressive performance thanks to recent advances in deep learning and the availability of large-scale parallel corpora. There have been numerous attempts to extend these successes to low-resource language pairs, yet requiring tens of thousands of parallel sentences. In this work, we take this research direction to the extreme and investigate whether it is possible to learn to translate even without any parallel data. We propose a model that takes sentences from monolingual corpora in two different languages and maps them into the same latent space. By learning to reconstruct in both languages from this shared feature space, the model effectively learns to translate without using any labeled data. We demonstrate our model on two widely used datasets and two language pairs, reporting BLEU scores up to 32.8, without using even a single parallel sentence at training time."

  - `video` <https://youtube.com/watch?v=yRlINjCQnc8> (Lample)
  - `video` <https://youtu.be/Chj8yTm5eGk?t=15m17s> (Lyalin) `in russian`
  - `video` <https://youtu.be/nIw-hNAjTwE?t=10m57s> (Moiseev) `in russian`
  - `notes` <http://ankitg.me/blog/2017/11/05/unsupervised-machine-translation.html>


#### ["Adversarial Deep Averaging Networks for Cross-Lingual Sentiment Classification"](http://arxiv.org/abs/1606.01614) Chen, Athiwaratkun, Sun, Weinberger, Cardie
>	"In recent years deep neural networks have achieved great success in sentiment classification for English, thanks in part to the availability of copious annotated resources. Unfortunately, most other languages do not enjoy such an abundance of annotated data for sentiment analysis. To combat this problem, we propose the Adversarial Deep Averaging Network to transfer sentiment knowledge learned from labeled English data to low-resource languages where only unlabeled data exists. ADAN is a "Y-shaped" network with two discriminative branches: a sentiment classifier and an adversarial language predictor. Both branches take input from a feature extractor that aims to learn hidden representations that capture the underlying sentiment of the text and are invariant across languages. Experiments on Chinese sentiment classification demonstrate that ADAN significantly outperforms several baselines, including a strong pipeline approach that relies on Google Translate, the state-of-the-art commercial machine translation system."

>	"In this work, we propose an end-to-end neural network model that only requires labeled English data and unlabeled Chinese text as input, and explicitly transfers the knowledge learned on English sentiment analysis to Chinese. Our trained system directly operates on Chinese sentences to predict their sentiment (e.g. positive or negative). We hypothesize that an ideal model for cross-lingual sentiment analysis should learn features that both perform well on the English sentiment classification task, and are invariant with respect to the shift in language. Therefore, ADAN simultaneously optimizes two components: i) a sentiment classifier P for English; and ii) an adversarial language predictor Q that tries to predict whether a sentence x is from English or Chinese. The two classifiers take input from the jointly learned feature extractor F, which is trained to maximize accuracy on English sentiment analysis and simultaneously to minimize the language predictor’s chance of correctly predicting the language of the text. This is why the language predictor Q is called “adversarial”. The model is exposed to both English and Chinese sentences during training, but only the labeled English sentences pass through the sentiment classifier. The feature extractor and the sentiment classifier are then used for Chinese sentences at test time. In this manner, we can train the system with massive amounts of unlabeled text in Chinese. Upon convergence, the joint features (output of F) are thus encouraged to be both discriminative for sentiment analysis and invariant across languages."


#### ["Style Transfer from Non-Parallel Text by Cross-Alignment"](https://arxiv.org/abs/1705.09655) Shen, Lei, Barzilay, Jaakkola
>	"This paper focuses on style transfer on the basis of non-parallel text. This is an instance of a broader family of problems including machine translation, decipherment, and sentiment modification. The key technical challenge is to separate the content from desired text characteristics such as sentiment. We leverage refined alignment of latent representations across mono-lingual text corpora with different characteristics. We deliberately modify encoded examples according to their characteristics, requiring the reproduced instances to match available examples with the altered characteristics as a population. We demonstrate the effectiveness of this cross-alignment method on three tasks: sentiment modification, decipherment of word substitution ciphers, and recovery of word order."

  - `video` <https://facebook.com/nipsfoundation/videos/1554741734617060?t=4850> (Shen)



#### ["Bandit Structured Prediction for Neural Sequence-to-Sequence Learning"](https://arxiv.org/abs/1704.06497) Kreutzer, Sokolov, Riezler
>	"Bandit structured prediction describes a stochastic optimization framework where learning is performed from partial feedback. This feedback is received in the form of a task loss evaluation to a predicted output structure, without having access to gold standard structures. We advance this framework by lifting linear bandit learning to neural sequence-to-sequence learning problems using attention-based recurrent neural networks. Furthermore, we show how to incorporate control variates into our learning algorithms for variance reduction and improved generalization. We present an evaluation on a neural machine translation task that shows improvements of up to 5.89 BLEU points for domain adaptation from simulated bandit feedback."


#### ["Reinforcement Learning for Bandit Neural Machine Translation with Simulated Human Feedback"](https://arxiv.org/abs/1707.07402) Nguyen, Daume, Boyd-Graber
>	"Machine translation is a natural candidate problem for reinforcement learning from human feedback: users provide quick, dirty ratings on candidate translations to guide a system to improve. Yet, current neural machine translation training focuses on expensive human-generated reference translations. We describe a reinforcement learning algorithm that improves neural machine translation systems from simulated human feedback. Our algorithm combines the advantage actor-critic algorithm (Mnih et al., 2016) with the attention-based neural encoder-decoder architecture (Luong et al., 2015). This algorithm (a) is well-designed for problems with a large action space and delayed rewards, (b) effectively optimizes traditional corpus-level machine translation metrics, and (c) is robust to skewed, high-variance, granular feedback modeled after actual human behaviors."



---
### interesting papers - word embeddings


#### ["Efficient Estimation of Word Representations in Vector Space"](http://arxiv.org/abs/1301.3781) Mikolov, Chen, Corrado, Dean
  `word2vec` `Skip-gram`
>	"We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities."

  - `video` <https://youtube.com/watch?v=fwcJpSYNsNs> (Mikolov)
  - `video` <https://youtube.com/watch?v=c3yRH0XZN2g> (Boyd-Graber)
  - `video` <https://youtube.com/watch?v=Ab5cgv9w7ac> (Scarfe)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=mikolov2013efficient>
  - `post` <http://jalammar.github.io/illustrated-word2vec>
  - `post` <http://alexminnaar.com/word2vec-tutorial-part-i-the-skip-gram-model.html>
  - `post` <http://alexminnaar.com/word2vec-tutorial-part-ii-the-continuous-bag-of-words-model.html>
  - `paper` ["word2vec Parameter Learning Explained"](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf) by Rong


#### ["Distributed Representations of Words and Phrases and Their Compositionality"](http://arxiv.org/abs/1310.4546) Mikolov, Sutskever, Chen, Corrado, Dean
  `word2vec` `Skip-gram` `negative sampling`
>	"The recently introduced continuous Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships. In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word representations. We also describe a simple alternative to the hierarchical softmax called negative sampling. An inherent limitation of word representations is their indifference to word order and their inability to represent idiomatic phrases. For example, the meanings of "Canada" and "Air" cannot be easily combined to obtain "Air Canada". Motivated by this example, we present a simple method for finding phrases in text, and show that learning good vector representations for millions of phrases is possible."

----
>	"First, they provide a slightly modified objective function and a few other sampling heuristics that result in a more computationally efficient model. Why does this produce good word representations? The distributional hypothesis states that words in similar contexts have similar meanings. The objective above clearly tries to increase the quantity vw·vc for good word-context pairs, and decrease it for bad ones. Intuitively, this means that words that share many contexts will be similar to each other (note also that contexts sharing many words will also be similar to each other).  
>	Second, they show that their model works with phrases, too, though they just do this by replacing the individual tokens in a multiword expression with a single symbol representing the phrase - pretty simple, but it works.  
>	Third, they show what to me was a very surprising additional feature of the learned vector spaces: some relationships are encoded compositionally in the vector space, meaning that you can just add the vectors for two words like "Russian" and "capital" to get a vector that is very close to "Moscow". They didn't do any kind of thorough evaluation of this, but the fact the it works at all was very surprising to me. They did give a reasonable explanation, however, and I've put it into math below. The probability of two words i and j appearing in the same context in this model is proportional to exp(vi⋅vj). Now, if we have a third word, k, and its probability of appearing with both word i and word j is proportional to exp(vk⋅vi)*exp(vk⋅vj)=exp(vk⋅(vi+vj)). So what you get when you add the vectors for two words is something that is likely to show up in the contexts of both of them. Thus if you pick word i to be "Russian" and word j to be "capital", a word k that has high probability might very well be "Moscow", because it tends to show up in the context of both of those words. So we can see that this method does have some reasonable explanation for why it works."

  - `post` <http://jalammar.github.io/illustrated-word2vec>
  - `paper` ["word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method"](http://arxiv.org/abs/1402.3722) by Goldberg and Levy


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
  - `video` <https://youtu.be/Mk8xPTxZ1kQ?t=6m29s> (Ghosh)
  - `notes` <http://www.shortscience.org/paper?bibtexKey=conf/emnlp/PenningtonSM14#shagunsodhani>
  - `code` <https://github.com/GradySimon/tensorflow-glove>


#### ["Swivel: Improving Embeddings by Noticing What’s Missing"](http://arxiv.org/abs/1602.02215) Shazeer, Doherty, Evans, Waterson
>	"We present Submatrix-wise Vector Embedding Learner (Swivel), a method for generating low-dimensional feature embeddings from a feature co-occurrence matrix. Swivel performs approximate factorization of the point-wise mutual information matrix via stochastic gradient descent. It uses a piecewise loss with special handling for unobserved co-occurrences, and thus makes use of all the information in the matrix. While this requires computation proportional to the size of the entire matrix, we make use of vectorized multiplication to process thousands of rows and columns at once to compute millions of predicted values. Furthermore, we partition the matrix into shards in order to parallelize the computation across many nodes. This approach results in more accurate embeddings than can be achieved with methods that consider only observed co-occurrences, and can scale to much larger corpora than can be handled with sampling methods."

>	"Swivel produces low-dimensional feature embeddings from a co-occurrence matrix. It optimizes an objective that is very similar to that of SGNS and GloVe: the dot product of a word embedding with a context embedding ought to approximate the observed PMI of the two words in the corpus. Unlike Skip-Gram Negative Sampling, Swivel’s computational requirements depend on the size of the co-occurrence matrix, rather than the size of the corpus. This means that it can be applied to much larger corpora. Unlike GloVe, Swivel explicitly considers all the co-occurrence information - including unobserved co-occurrences - to produce embeddings. In the case of unobserved co-occurrences, a “soft hinge” loss prevents the model from over-estimating PMI. This leads to demonstrably better embeddings for rare features without sacrificing quality for common ones. Swivel capitalizes on vectorized hardware, and uses block structure to amortize parameter transfer cost and avoid contention. This results in the ability to handle very large co-occurrence matrices in a scalable way that is easy to parallelize."

>	"Due to the fact that Skip-Gram Negative Sampling slides a sampling window through the entire training corpus, a significant drawback of the algorithm is that it requires training time proportional to the size of the corpus."

  - `code` <https://github.com/tensorflow/models/tree/master/research/swivel>


#### ["Bag of Tricks for Efficient Text Classification"](http://arxiv.org/abs/1607.01759) Joulin, Grave, Bojanowski, Mikolov
  `fastText`
>	"This paper explores a simple and efficient baseline for text classification. Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation. We can train fastText on more than one billion words in less than ten minutes using a standard multicore CPU, and classify half a million sentences among 312K classes in less than a minute."

----
>	"At par with deep learning models in terms of accuracy though an order of magnitude faster in performance."

  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1607.01759>
  - `notes` <https://medium.com/paper-club/bag-of-tricks-for-efficient-text-classification-818bc47e90f>
  - `code` <https://fasttext.cc>
  - `code` <https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py>


#### ["Learning to Compute Word Embeddings On the Fly"](https://arxiv.org/abs/1706.00286) Bahdanau, Bosc, Jastrzebski, Grefenstette, Vincent, Bengio
>	"Words in natural language follow a Zipfian distribution whereby some words are frequent but most are rare. Learning representations for words in the "long tail" of this distribution requires enormous amounts of data. Representations of rare words trained directly on end-tasks are usually poor, requiring us to pre-train embeddings on external data, or treat all rare words as out-of-vocabulary words with a unique representation. We provide a method for predicting embeddings of rare words on the fly from small amounts of auxiliary data with a network trained against the end task. We show that this improves results against baselines where embeddings are trained on the end task in a reading comprehension task, a recognizing textual entailment task, and in language modelling."

  - `notes` <https://theneuralperspective.com/2017/06/05/more-on-embeddings-spring-2017/>


#### ["Neural Word Embedding as Implicit Matrix Factorization"](http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf) Levy, Goldberg
>	"We analyze skip-gram with negative-sampling, a word embedding method introduced by Mikolov et al., and show that it is implicitly factorizing a word-context matrix, whose cells are the pointwise mutual information of the respective word and context pairs (shifted by a global constant). We find that another embedding method, NCE, is implicitly factorizing a similar matrix, where each cell is the (shifted) log conditional probability of a word given its context. We show that using a sparse Shifted Positive PMI word-context matrix to represent words improves results on two word similarity tasks and one of two analogy tasks. When dense, low-dimensional vectors are preferred, exact factorization with SVD can achieve solutions that are at least as good as SGNS’s solutions for word similarity tasks. On analogy questions SGNS remains superior to SVD. We conjecture that this stems from the weighted nature of SGNS’s factorization."

>	"It looks at the objective function optimized by skip-gram, and shows that it is implicitly factoring a (shifted) PMI matrix. That's a really interesting (and non-obvious) connection. They further show that they can optimize the objective directly, either by just constructing the shifted PMI matrix, or by using a truncated (and thus sparse) version of that. And, instead of implicitly factorizing this shifted matrix, once you know what skip-gram is doing, you can just directly factor the matrix yourself using SVD. They show that for several tasks (all that they tested except for syntactic analogies), these more direct techniques outperform skip-gram."

>	"- with proper tuning, traditional distributional similarity methods can be very competitive with word2vec"
>	"- by analyzing word2vec we found a novel variation on the PMI association measure which is kind-of ugly but works surprisingly well"
>	"- we tried and tried, but couldn't get GloVe to outperform word2vec. Their w+c idea is neat and works very well, though"

>	"Skip-Gram Negative Sampling can be seen as producing two matrices, W for focus words and W ̃ for context words, such that their product WW ̃ approximates the observed PMI between respective word/context pairs. Given a specific focus word i and context word j, SGNS minimizes the magnitude of the difference between wiT*w ̃j and pmi(i; j), tempered by a monotonically increasing weighting function of the observed co-occurrence count."

  - `video` <https://youtu.be/KEXWC-ICH_Y?t=28m51s> (Potapenko) `in russian`
  - `post` <https://minhlab.wordpress.com/2015/06/08/a-new-proof-for-the-equivalence-of-word2vec-skip-gram-and-shifted-ppmi/>
  - `post` <https://building-babylon.net/2016/05/12/skipgram-isnt-matrix-factorisation/>
  - `post` <http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/>


#### ["RAND-WALK: A Latent Variable Model Approach to Word Embeddings"](http://arxiv.org/abs/1502.03520) Arora, Li, Liang, Ma, Risteski
  `RAND-WALK`
>	"Semantic word embeddings represent the meaning of a word via a vector, and are created by diverse methods. Many use nonlinear operations on co-occurrence statistics, and have hand-tuned hyperparameters and reweighting methods. This paper proposes a new generative model, a dynamic version of the log-linear topic model of Mnih and Hinton (2007). The methodological novelty is to use the prior to compute closed form expressions for word statistics. This provides a theoretical justification for nonlinear models like PMI, word2vec, and GloVe, as well as some hyperparameter choices. It also helps explain why low-dimensional semantic embeddings contain linear algebraic structure that allows solution of word analogies, as shown by Mikolov et al. (2013a) and many subsequent papers. Experimental support is provided for the generative model assumptions, the most important of which is that latent word vectors are fairly uniformly dispersed in space."

  - `video` <http://youtube.com/watch?v=gaVR3WnczOQ> (Ma)
  - `video` <http://youtube.com/watch?v=KR46z_V0BVw> (Arora)
  - `video` <http://youtube.com/watch?v=BCsOrewkmH4> (Ma)
  - `post` <http://www.offconvex.org/2015/12/12/word-embeddings-1/> + <http://www.offconvex.org/2016/02/14/word-embeddings-2/>
  - `notes` <https://akshayka.github.io/papers/html/arora2016pmi-embeddings.html>
  - `code` <https://github.com/YingyuLiang/SemanticVector>


#### ["Skip-Gram – Zipf + Uniform = Vector Additivity"](http://aclweb.org/anthology/P17-1007) Gittens, Achlioptas, Mahoney
>	"In recent years word-embedding models have gained great popularity due to their remarkable performance on several tasks, including word analogy questions and caption generation. An unexpected “sideeffect” of such models is that their vectors often exhibit compositionality, i.e., adding two word-vectors results in a vector that is only a small angle away from the vector of a word representing the semantic composite of the original words, e.g., “man” + “royal” = “king”."

>	"This work provides a theoretical justification for the presence of additive compositionality in word vectors learned using the Skip-Gram model. In particular, it shows that additive compositionality holds in an even stricter sense (small distance rather than small angle) under certain assumptions on the process generating the corpus. As a corollary, it explains the success of vector calculus in solving word analogies. When these assumptions do not hold, this work describes the correct non-linear composition operator."

>	"Finally, this work establishes a connection between the Skip-Gram model and the Sufficient Dimensionality Reduction framework of Globerson and Tishby: the parameters of SDR models can be obtained from those of Skip-Gram models simply by adding information on symbol frequencies. This shows that SkipGram embeddings are optimal in the sense of Globerson and Tishby and, further, implies that the heuristics commonly used to approximately fit Skip-Gram models can be used to fit SDR models."

  - `video` <https://vimeo.com/234954195> (Gittens)


#### ["What the Vec? Towards Probabilistically Grounded Embeddings"](https://arxiv.org/abs/1805.12164) Allen, Balazavic, Hospedales
>	"We take two main results, the well known relationship to PMI learned by W2V embeddings and a recent connection drawn between PMI and analogies, to show that word embedding models that linearly project PMI vectors capture the semantic properties of relatedness, similarity, paraphrase and analogy. The loss functions of W2V and Glove can be seen to approximate such a projection, non-linearly in the case of W2V and not fully capturing PMI in Glove, thus explaining why their embeddings exhibit semantic properties useful in downstream tasks."  
>	"From linear projection, we derive a relationship between embedding matrices W and C that enables common word embedding interactions to be semantically interpreted. This suggests that the familiar cosine similarity may serve as a blend of more semantically meaningful interactions. Our theoretical results explain several empirical observations, e.g. why W and C are not found to be equal, despite representing the same words, their symmetric treatment in the loss function and a symmetric PMI matrix; why mean embeddings (A) are often found to outperform those from W; and why relatedness corresponds to interactions between W and C, and similarity to interactions between W and W."  
>	"We reveal an interesting hierarchical structure between these semantic properties in which relatedness serves as a fundamental pairwise comparison from which a relationship can be defined to all words. This forms a basis for the semantic relationships of similarity, paraphrasing and analogies. Error terms arise in the latter higher order relationships due to statistical relationships within the word sets. These errors can be interpreted geometrically with respect to the surface S on which all PMI vectors lie and, in principle, can be evaluated from higher order statistics (e.g trigrams co-occurrences)."  
>	"There remain a few further aspects of W2V and Glove to address, e.g. that PMI is weighted over the context window, unigram distributions are often raised to the power 3/4, and the probability weighting of Glove."  


#### ["Analogies Explained: Towards Understanding Word Embeddings"](https://arxiv.org/abs/1901.09813) Allen, Hospedales
>	"Word embeddings generated by neural network methods such as word2vec (W2V) are well known to exhibit seemingly linear behaviour, e.g. the embeddings of analogy "woman is to queen as man is to king" approximately describe a parallelogram. This property is particularly intriguing since the embeddings are not trained to achieve it. Several explanations have been proposed, but each introduces assumptions that do not hold in practice. We derive a probabilistically grounded definition of paraphrasing that we re-interpret as word transformation, a mathematical description of "wx is to wy". From these concepts we prove existence of linear relationships between W2V-type embeddings that underlie the analogical phenomenon, identifying explicit error terms."

  - `video` <https://slideslive.com/38917410/applications-natural-language-processing>


#### ["Fair is Better than Sensational: Man is to Doctor as Woman is to Doctor"](https://arxiv.org/abs/1905.09866) Nissim, Noord, Goot
>	"Analogies such as man is to king as woman is to X are often used to illustrate the amazing power of word embeddings. Concurrently, they have also exposed how strongly human biases are encoded in vector spaces built on natural language. While finding that queen is the answer to man is to king as woman is to X leaves us in awe, papers have also reported finding analogies deeply infused with human biases, like man is to computer programmer as woman is to homemaker, which instead leave us with worry and rage. In this work we show that,often unknowingly, embedding spaces have not been treated fairly. Through a series of simple experiments, we highlight practical and theoretical problems in previous works, and demonstrate that some of the most widely used biased analogies are in fact not supported by the data. We claim that rather than striving to find sensational biases, we should aim at observing the data "as is", which is biased enough. This should serve as a fair starting point to properly address the evident, serious, and compelling problem of human bias in word embeddings."

>	"Do word embeddings really say that man is to doctor as woman is to nurse? Apparently not. The original analogy codes, both w2v and gensim, implement a constraint whereby no input vector can be returned. In any query A:B :: C:D, one could never get an answer D such that D==A or D==B or D==C simply because the code does not allow it. Therefore, for the query "man is to doctor as woman is to X", doctor cannot be returned as answer! We modified the code to make it unrestricted and tested several classic (biased) examples as well as the original analogy set. Specific outcomes are shown and discussed in the paper. We saw that: (i) if you let it, B is almost always returned. (ii) then, the analogy task doesn't work that well (no "queen" for "man is to king as woman is to X", but "king") and (iii) analogy-based biases have often been overemphasized in the literature."


#### ["Order-Embeddings of Images and Language"](http://arxiv.org/abs/1511.06361) Vendrov, Kiros, Fidler, Urtasun
>	"Hypernymy, textual entailment, and image captioning can be seen as special cases of a single visual-semantic hierarchy over words, sentences, and images. In this paper we advocate for explicitly modeling the partial order structure of this hierarchy. Towards this goal, we introduce a general method for learning ordered representations, and show how it can be applied to a variety of tasks involving images and language. We show that the resulting representations improve performance over current approaches for hypernym prediction and image-caption retrieval."

  - `video` <http://videolectures.net/iclr2016_vendrov_order_embeddings/> (Vendrov)
  - `code` <https://github.com/ivendrov/order-embedding>
  - `code` <https://github.com/ivendrov/order-embeddings-wordnet>
  - `code` <https://github.com/LeavesBreathe/tensorflow_with_latest_papers/blob/master/partial_ordering_embedding.py>


#### ["Poincare Embeddings for Learning Hierarchical Representations"](https://arxiv.org/abs/1705.08039) Nickel, Kiela
>	"Representation learning has become an invaluable approach for learning from symbolic data such as text and graphs. However, while complex symbolic datasets often exhibit a latent hierarchical structure, state-of-the-art methods typically learn embeddings in Euclidean vector spaces, which do not account for this property. For this purpose, we introduce a new approach for learning hierarchical representations of symbolic data by embedding them into hyperbolic space -- or more precisely into an n-dimensional Poincar\'e ball. Due to the underlying hyperbolic geometry, this allows us to learn parsimonious representations of symbolic data by simultaneously capturing hierarchy and similarity. We introduce an efficient algorithm to learn the embeddings based on Riemannian optimization and show experimentally that Poincar\'e embeddings outperform Euclidean embeddings significantly on data with latent hierarchies, both in terms of representation capacity and in terms of generalization ability."

  - `video` <https://cds.cern.ch/record/2306315> (Nickel)
  - `video` <https://facebook.com/nipsfoundation/videos/1553634558061111?t=3781> (Nickel)
  - `notes` <https://medium.com/towards-data-science/facebook-research-just-published-an-awesome-paper-on-learning-hierarchical-representations-34e3d829ede7>
  - `notes` <https://medium.com/@hol_io/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795>
  - `post` <https://rare-technologies.com/implementing-poincare-embeddings/>
  - `code` <https://github.com/facebookresearch/poincare-embeddings>
  - `code` <https://github.com/TatsuyaShirakawa/poincare-embedding>
  - `code` <https://github.com/lateral/poincare-embeddings>



---
### interesting papers - word sense disambiguation


#### ["Efficient Non-parametric Estimation of Multiple Embeddings per Word in Vector Space"](https://people.cs.umass.edu/~arvind/emnlp2014.pdf) Neelakantan, Shankar, Passos, McCallum
>	"There is rising interest in vector-space word embeddings and their use in NLP, especially given recent methods for their fast estimation at very large scale. Nearly all this work, however, assumes a single vector per word type - ignoring polysemy and thus jeopardizing their usefulness for downstream tasks. We present an extension to the Skip-gram model that efficiently learns multiple embeddings per word type. It differs from recent related work by jointly performing word sense discrimination and embedding learning, by non-parametrically estimating the number of senses per word type, and by its efficiency and scalability. We present new state-of-the-art results in the word similarity in context task and demonstrate its scalability by training with one machine on a corpus of nearly 1 billion tokens in less than 6 hours."

  - <https://people.cs.umass.edu/~arvind/emnlp2014wordvectors/>
  - `video` <http://youtube.com/watch?v=EeBj4TyW8B8>


#### ["Word Representations via Gaussian Embedding"](http://arxiv.org/abs/1412.6623) Vilnis, McCallum
>	"Current work in lexical distributed representations maps each word to a point vector in low-dimensional space. Mapping instead to a density provides many interesting advantages, including better capturing uncertainty about a representation and its relationships, expressing asymmetries more naturally than dot product or cosine similarity, and enabling more expressive parameterization of decision boundaries. This paper advocates for density-based distributed embeddings and presents a method for learning representations in the space of Gaussian distributions. We compare performance on various word embedding benchmarks, investigate the ability of these embeddings to model entailment and other asymmetric relationships, and explore novel properties of the representation."

>	"In this work we introduced a method to embed word types into the space of Gaussian distributions, and learn the embeddings directly in that space. This allows us to represent words not as low-dimensional vectors, but as densities over a latent space, directly representing notions of uncertainty and enabling a richer geometry in the embedded space. We demonstrated the effectiveness of these embeddings on a linguistic task requiring asymmetric comparisons, as well as standard word similarity benchmarks, learning of synthetic hierarchies, and several qualitative examinations. In future work, we hope to move beyond spherical or diagonal covariances and into combinations of low rank and diagonal matrices. Efficient updates and scalable learning is still possible due to the Sherman-Woodbury-Morrison formula. Additionally, going beyond diagonal covariances will enable us to keep our semantics from being axis-aligned, which will increase model capacity and expressivity. We also hope to move past stochastic gradient descent and warm starting and be able to learn the Gaussian representations robustly in one pass from scratch by using e.g. proximal or block coordinate descent methods. Improved optimization strategies will also be helpful on the highly nonconvex problem of training supervised hierarchies with KL divergence. Representing words and concepts as different types of distributions (including other elliptic distributions such as the Student’s t) is an exciting direction – Gaussians concentrate their density on a thin spherical ellipsoidal shell, which can lead to counterintuitive behavior in high dimensions. Multimodal distributions represent another clear avenue for future work. Combining ideas from kernel methods and manifold learning with deep learning and linguistic representation learning is an exciting frontier. In other domains, we want to extend the use of potential function representations to other tasks requiring embeddings, such as relational learning with the universal schema. We hope to leverage the asymmetric measures, probabilistic interpretation, and flexible training criteria of our model to tackle tasks involving similarity-in-context, comparison of sentences and paragraphs, and more general common sense reasoning."

----
>	"- represent symbols not as points but as regions in space
>	- captures uncertainty of representation
>	- enhances asymmetric reasoning such as entailment/implicature
>	- more expressive decision boundaries and representational power"

  - `video` <http://youtu.be/Xm1XGjc9lDc?t=14m52s> (McCallum)
  - `video` <http://youtube.com/watch?v=PKTfALFk03M> (Vilnis)
  - `code` <https://github.com/seomoz/word2gauss>


#### ["A Probabilistic Model for Learning Multi-Prototype Word Embeddings"](http://research.microsoft.com/apps/pubs/default.aspx?id=226629) Tian et al.
>	"Distributed word representations have been widely used and proven to be useful in quite a few natural language processing and text mining tasks. Most of existing word embedding models aim at generating only one embedding vector for each individual word, which, however, limits their effectiveness because huge amounts of words are polysemous (such as bank and star). To address this problem, it is necessary to build multi embedding vectors to represent different meanings of a word respectively. Some recent studies attempted to train multi-prototype word embeddings through clustering context window features of the word. However, due to a large number of parameters to train, these methods yield limited scalability and are inefficient to be trained with big data. In this paper, we introduce a much more efficient method for learning multi embedding vectors for polysemous words. In particular, we first propose to model word polysemy from a probabilistic perspective and integrate it with the highly efficient continuous Skip-Gram model. Under this framework, we design an Expectation-Maximization algorithm to learn the word’s multi embedding vectors. With much less parameters to train, our model can achieve comparable or even better results on word-similarity tasks compared with conventional methods."

>	"In this paper, we introduce a fast and probabilistic method to generate multiple embedding vectors for polysemous words, based on the continuous Skip-Gram model. On one hand, our method addresses the drawbacks of the original Word2Vec model by leveraging multi-prototype word embeddings; on the other hand, our model yields much less complexity without performance loss compared with the former clustering based multi-prototype algorithms. In addition, the probabilistic framework of our method avoids the extra efforts to perform clustering besides training word embeddings."


#### ["Breaking Sticks and Ambiguities with Adaptive Skip-gram"](http://arxiv.org/abs/1502.07257) Bartunov, Kondrashkin, Osokin, Vetrov
>	"Recently proposed Skip-gram model is a powerful method for learning high-dimensional word representations that capture rich semantic relationships between words. However, Skip-gram as well as most prior work on learning word representations does not take into account word ambiguity and maintain only single representation per word. Although a number of Skip-gram modifications were proposed to overcome this limitation and learn multi-prototype word representations, they either require a known number of word meanings or learn them using greedy heuristic approaches. In this paper we propose the Adaptive Skip-gram model which is a nonparametric Bayesian extension of Skip-gram capable to automatically learn the required number of representations for all words at desired semantic resolution. We derive efficient online variational learning algorithm for the model and empirically demonstrate its efficiency on wordsense induction task."

  - `video` <http://youtube.com/watch?v=vYbee1InliU> (Vetrov)
  - `video` <http://postnauka.ru/video/49258> (Vetrov) `in russian`
  - `video` <http://youtu.be/uoRwjxaDgt0?t=33m58s> (Vetrov) `in russian`
  - `code` <https://github.com/sbos/AdaGram.jl>


#### ["Infinite Dimensional Word Embeddings"](http://arxiv.org/abs/1511.05392) Nalisnick, Ravi
>	"We describe a method for learning word embeddings with stochastic dimensionality. Our Infinite Skip-Gram model specifies an energy-based joint distribution over a word vector, a context vector, and their dimensionality, which can be defined over a countably infinite domain by employing the same techniques used to make the Infinite Restricted Boltzmann Machine tractable. We find that the distribution over embedding dimensionality for a given word is highly interpretable and leads to an elegant probabilistic mechanism for word sense induction. We show qualitatively and quantitatively that the iSG produces parameter-efficient representations that are robust to language’s inherent ambiguity."

>	"We’ve proposed a novel word embedding model called Infinite Skip-Gram that defines vector dimensionality as a random variable with a countably infinite domain. Training via the generalized EM framework allows embeddings to grow as the data requires. This property is especially well suited for learning representations of homographs, which Skip-Gram notably fails at. A unique quality of the iSG is that it is highly interpretable (in comparison to other embedding methods) due to its ability to produce a distribution over the dimensionality of a given word (p(z|w)). Plots of p(z|w) concisely show how specific/vague a word is and its various senses just from the mode landscape."

>	"During training, the iSGM allows word representations to grow naturally based on how well they can predict their context. This behavior enables the vectors of specific words to use few dimensions and the vectors of vague words to elongate as needed. Manual and experimental analysis reveals this dynamic representation elegantly captures specificity, polysemy, and homonymy without explicit definition of such concepts within the model."

  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals/corr/NalisnickR15>
  - `post` <http://dustintran.com/blog/infinite-dimensional-word-embeddings/>
  - `video` <http://videolectures.net/deeplearning2016_cote_boltzmann_machine/> (Cote)


#### ["Sense2Vec - A Fast and Accurate Method for Word Sense Disambiguation in Neural Word Embeddings"](http://arxiv.org/abs/1511.06388) Trask, Michalak, Liu
>	"Neural word representations have proven useful in Natural Language Processing tasks due to their ability to efficiently model complex semantic and syntactic word relationships. However, most techniques model only one representation per word, despite the fact that a single word can have multiple meanings or ”senses”. Some techniques model words by using multiple vectors that are clustered based on context. However, recent neural approaches rarely focus on the application to a consuming NLP algorithm. Furthermore, the training process of recent word-sense models is expensive relative to single-sense embedding processes. This paper presents a novel approach which addresses these concerns by modeling multiple embeddings for each word based on supervised disambiguation, which provides a fast and accurate way for a consuming NLP model to select a sense-disambiguated embedding. We demonstrate that these embeddings can disambiguate both contrastive senses such as nominal and verbal senses as well as nuanced senses such as sarcasm. We further evaluate Part-of-Speech disambiguated embeddings on neural dependency parsing, yielding a greater than 8% average error reduction in unlabeled attachment scores across 6 languages."


#### ["Word Sense Disambiguation with Neural Language Models"](http://arxiv.org/abs/1603.07012) Yuan, Doherty, Richardson, Evans, Altendorf
>	"Determining the intended sense of words in text - word sense disambiguation - is a long-standing problem in natural language processing. In this paper, we present WSD algorithms which use neural network language models to achieve state-of-the-art precision. Each of these methods learns to disambiguate word senses using only a set of word senses, a few example sentences for each sense taken from a licensed lexicon, and a large unlabeled text corpus. We classify based on cosine similarity of vectors derived from the contexts in unlabeled query and labeled example sentences. We demonstrate state-of-the-art results when using the WordNet sense inventory, and significantly better than baseline performance using the New Oxford American Dictionary inventory. The best performance was achieved by combining an LSTM language model with graph label propagation."

>	"Our experiments show that using the LSTM language model achieves significantly higher precision than the CBOW language model, especially on verbs and adverbs. This suggests that sequential order information is important to discriminating senses of verbs and adverbs. The best performance was achieved by using an LSTM language model with label propagation. Our algorithm outperforms the baseline by more than 10% (0.87 vs. 0.75)."



---
### interesting papers - text embeddings

#### ["Learning Deep Structured Semantic Models for Web Search using Clickthrough Data"](http://research.microsoft.com/apps/pubs/default.aspx?id=198202) Huang, He, Gao, Deng, Acero, Heck
  `DSSM`
>	"Latent semantic models, such as LSA, intend to map a query to its relevant documents at the semantic level where keyword-based matching often fails. In this study we strive to develop a series of new latent semantic models with a deep structure that project queries and documents into a common low-dimensional space where the relevance of a document given a query is readily computed as the distance between them. The proposed deep structured semantic models are discriminatively trained by maximizing the conditional likelihood of the clicked documents given a query using the clickthrough data. To make our models applicable to large-scale Web search applications, we also use a technique called word hashing, which is shown to effectively scale up our semantic models to handle large vocabularies which are common in such tasks. The new models are evaluated on a Web document ranking task using a real-world data set. Results show that our best model significantly outperforms other latent semantic models, which were considered state-of-the-art in the performance prior to the work presented in this paper."

>	"We present and evaluate a series of new latent semantic models, notably those with deep architectures which we call the DSSM. The main contribution lies in our significant extension of the previous latent semantic models (e.g., LSA) in three key aspects. First, we make use of the clickthrough data to optimize the parameters of all versions of the models by directly targeting the goal of document ranking. Second, inspired by the deep learning framework recently shown to be highly successful in speech recognition, we extend the linear semantic models to their nonlinear counterparts using multiple hidden-representation layers. The deep architectures adopted have further enhanced the modeling capacity so that more sophisticated semantic structures in queries and documents can be captured and represented. Third, we use a letter n-gram based word hashing technique that proves instrumental in scaling up the training of the deep models so that very large vocabularies can be used in realistic web search. In our experiments, we show that the new techniques pertaining to each of the above three aspects lead to significant performance improvement on the document ranking task. A combination of all three sets of new techniques has led to a new state-of-the-art semantic model that beats all the previously developed competing models with a significant margin."

>	"DSSM stands for Deep Structured Semantic Model, or more general, Deep Semantic Similarity Model. DSSM is a deep neural network modeling technique for representing text strings (sentences, queries, predicates, entity mentions, etc.) in a continuous semantic space and modeling semantic similarity between two text strings (e.g., Sent2Vec). DSSM has wide applications including information retrieval and web search ranking (Huang et al. 2013; Shen et al. 2014a,2014b), ad selection/relevance, contextual entity search and interestingness tasks (Gao et al. 2014a), question answering (Yih et al., 2014), knowledge inference (Yang et al., 2014), image captioning (Fang et al., 2014), and machine translation (Gao et al., 2014b) etc. DSSM can be used to develop latent semantic models that project entities of different types (e.g., queries and documents) into a common low-dimensional semantic space for a variety of machine learning tasks such as ranking and classification. For example, in web search ranking, the relevance of a document given a query can be readily computed as the distance between them in that space. With the latest GPUs from Nvidia, we are able to train our models on billions of words."

  - <http://research.microsoft.com/en-us/projects/dssm/>
  - `video` <https://youtu.be/g1Pgo5yTIKg?t=17m10s> (Mitra)
  - `video` <https://youtu.be/x7B6RudUQLI?t=1h5m5s> (Gulin) `in russian`
  - `slides` <http://research.microsoft.com/pubs/232372/CIKM14_tutorial_HeGaoDeng.pdf>
  - `post` <https://habrahabr.ru/company/yandex/blog/314222/> `in russian`
  - `code` <https://github.com/faneshion/MatchZoo>
  - `code` <https://github.com/airalcorn2/Deep-Semantic-Similarity-Model>


#### ["Skip-Thought Vectors"](http://arxiv.org/abs/1506.06726) Kiros, Zhu, Salakhutdinov, Zemel, Torralba, Urtasun, Fidler
>	"We describe an approach for unsupervised learning of a generic, distributed sentence encoder. Using the continuity of text from books, we train an encoder-decoder model that tries to reconstruct the surrounding sentences of an encoded passage. Sentences that share semantic and syntactic properties are thus mapped to similar vector representations. We next introduce a simple vocabulary expansion method to encode words that were not seen as part of training, allowing us to expand our vocabulary to a million words. After training our model, we extract and evaluate our vectors with linear models on 8 tasks: semantic relatedness, paraphrase detection, image-sentence ranking, question-type classification and 4 benchmark sentiment and subjectivity datasets. The end result is an off-the-shelf encoder that can produce highly generic sentence representations that are robust and perform well in practice."

>	"We evaluated the effectiveness of skip-thought vectors as an off-the-shelf sentence representation with linear classifiers across 8 tasks. Many of the methods we compare against were only evaluated on 1 task. The fact that skip-thought vectors perform well on all tasks considered highlight the robustness of our representations. We believe our model for learning skip-thought vectors only scratches the surface of possible objectives. Many variations have yet to be explored, including (a) deep encoders and decoders, (b) larger context windows, (c) encoding and decoding paragraphs, (d) other encoders, such as convnets. It is likely the case that more exploration of this space will result in even higher quality representations."

>	"Developing learning algorithms for distributed compositional semantics of words has been a long-standing open problem at the intersection of language understanding and machine learning. In recent years, several approaches have been developed for learning composition operators that map word vectors to sentence vectors including recursive networks, recurrent networks, convolutional networks and recursive-convolutional methods. All of these methods produce sentence representations that are passed to a supervised task and depend on a class label in order to backpropagate through the composition weights. Consequently, these methods learn high-quality sentence representations but are tuned only for their respective task. The paragraph vector is an alternative to the above models in that it can learn unsupervised sentence representations by introducing a distributed sentence indicator as part of a neural language model. The downside is at test time, inference needs to be performed to compute a new vector. In this paper we abstract away from the composition methods themselves and consider an alternative loss function that can be applied with any composition operator. We consider the following question: is there a task and a corresponding loss that will allow us to learn highly generic sentence representations? We give evidence for this by proposing a model for learning high-quality sentence vectors without a particular supervised task in mind. Using word vector learning as inspiration, we propose an objective function that abstracts the skip-gram model to the sentence level. That is, instead of using a word to predict its surrounding context, we instead encode a sentence to predict the sentences around it. Thus, any composition operator can be substituted as a sentence encoder and only the objective function becomes modified. We call our model skip-thoughts and vectors induced by our model are called skip-thought vectors. Our model depends on having a training corpus of contiguous text."

>	"One difficulty that arises with such an experimental setup is being able to construct a large enough word vocabulary to encode arbitrary sentences. For example, a sentence from a Wikipedia article might contain nouns that are highly unlikely to appear in our book vocabulary. We solve this problem by learning a mapping that transfers word representations from one model to another. Using pretrained word2vec representations learned with a continuous bag-of-words model, we learn a linear mapping from a word in word2vec space to a word in the encoder’s vocabulary space. The mapping is learned using all words that are shared between vocabularies. After training, any word that appears in word2vec can then get a vector in the encoder word embedding space."

----
>	"It turns out that skip-thought vectors have some intriguing properties that allow us to construct F in a really simple way. Suppose we have 3 vectors: an image caption x, a "caption style" vector c and a "book style" vector b. Then we define F as F(x) = x - c + b which intuitively means: keep the "thought" of the caption, but replace the image caption style with that of a story. Then, we simply feed F(x) to the decoder."

  - `video` <http://videolectures.net/deeplearning2015_salakhutdinov_deep_learning_2/#t=3776> (Salakhutdinov)
  - `post` <https://medium.com/@samim/generating-stories-about-images-d163ba41e4ed> (demo)
  - `post` <http://sanyam5.github.io/my-thoughts-on-skip-thoughts/>
  - `code` <https://github.com/tensorflow/models/tree/master/research/skip_thoughts>
  - `code` <https://github.com/sanyam5/skip-thoughts>


#### ["Towards Universal Paraphrastic Sentence Embeddings"](https://arxiv.org/abs/1511.08198) Wieting, Bansal, Gimpel, Livescu
>	"We consider the problem of learning general-purpose, paraphrastic sentence embeddings based on supervision from the Paraphrase Database (Ganitkevitch et al., 2013). We compare six compositional architectures, evaluating them on annotated textual similarity datasets drawn both from the same distribution as the training data and from a wide range of other domains. We find that the most complex architectures, such as long short-term memory (LSTM) recurrent neural networks, perform best on the in-domain data. However, in out-of-domain scenarios, simple architectures such as word averaging vastly outperform LSTMs. Our simplest averaging model is even competitive with systems tuned for the particular tasks while also being extremely efficient and easy to use. In order to better understand how these architectures compare, we conduct further experiments on three supervised NLP tasks: sentence similarity, entailment, and sentiment classification. We again find that the word averaging models perform well for sentence similarity and entailment, outperforming LSTMs. However, on sentiment classification, we find that the LSTM performs very strongly-even recording new state-of-the-art performance on the Stanford Sentiment Treebank. We then demonstrate how to combine our pretrained sentence embeddings with these supervised tasks, using them both as a prior and as a black box feature extractor. This leads to performance rivaling the state of the art on the SICK similarity and entailment tasks. We release all of our resources to the research community with the hope that they can serve as the new baseline for further work on universal sentence embeddings."

  - `video` <http://youtube.com/watch?v=fC0j6mEFdZE> (Gimpel)
  - `video` <http://videolectures.net/iclr2016_wieting_universal_paraphrastic/> (Wieting)
  - `code` <https://github.com/jwieting/iclr2016>


#### ["A Simple but Tough-to-Beat Baseline for Sentence Embeddings"](https://openreview.net/pdf?id=SyK00v5xx) Arora, Liang, Ma
>	"The success of neural network methods for computing word embeddings has motivated methods for generating semantic embeddings of longer pieces of text, such as sentences and paragraphs. Surprisingly, Wieting et al (ICLR’16) showed that such complicated methods are outperformed, especially in out-of-domain (transfer learning) settings, by simpler methods involving mild retraining of word embeddings and basic linear regression. The method of Wieting et al. requires retraining with a substantial labeled dataset such as Paraphrase Database (Ganitkevitch et al., 2013). The current paper goes further, showing that the following completely unsupervised sentence embedding is a formidable baseline: Use word embeddings computed using one of the popular methods on unlabeled corpus like Wikipedia, represent the sentence by a weighted average of the word vectors, and then modify them a bit using PCA/SVD. This weighting improves performance by about 10% to 30% in textual similarity tasks, and beats sophisticated supervised methods including RNN’s and LSTM’s. It even improves Wieting et al.’s embeddings. This simple method should be used as the baseline to beat in future, especially when labeled training data is scarce or nonexistent. The paper also gives a theoretical explanation of the success of the above unsupervised method using a latent variable generative model for sentences, which is a simple extension of the model in Arora et al. (TACL’16) with new “smoothing” terms that allow for words occurring out of context, as well as high probabilities for words like and, not in all contexts."

  - `post` <http://www.offconvex.org/2018/06/17/textembeddings>
  - `video` <https://youtube.com/watch?v=BCsOrewkmH4> (Ma)
  - `video` <https://youtu.be/KR46z_V0BVw?t=49m10s> (Arora)
  - `code` <https://github.com/PrincetonML/SIF>
  - `code` <https://github.com/YingyuLiang/SIF>


#### ["No Training Required: Exploring Random Encoders for Sentence Classification"](https://arxiv.org/abs/1901.10444) Wieting, Kiela
>	"We explore various methods for computing sentence representations from pre-trained word embeddings without any training, i.e., using nothing but random parameterizations. Our aim is to put sentence embeddings on more solid footing by 1) looking at how much modern sentence embeddings gain over random methods---as it turns out, surprisingly little; and by 2) providing the field with more appropriate baselines going forward---which are, as it turns out, quite strong. We also make important observations about proper experimental protocol for sentence classification evaluation, together with recommendations for future research."

>	"In this work we have sought to put sentence embeddings on more solid footing by examining how much trained sentence encoders improve over random sentence encoders. As it turns out, differences exist, but are smaller than we would have hoped: in comparison to sentence encoders such as SkipThought (which was trained for a very long time) and InferSent (which requires large amounts of annotated data), performance improvements are less than 2 points on average over the 10 SentEval tasks. Therefore one may wonder to what extent sentence encoders are worth the attention they’re receiving. Hope remains, however, if we as a community start focusing on more sophisticated tasks that require more sophisticated learned representations that cannot merely rely on having good pre-trained word embeddings."



---
### interesting papers - syntactic parsing


#### ["Grammar As a Foreign Language"](http://arxiv.org/abs/1412.7449) Vinyals, Kaiser, Koo, Petrov, Sutskever, Hinton
>	"Syntactic parsing is a fundamental problem in computational linguistics and natural language processing. Traditional approaches to parsing are highly complex and problem specific. Recently, Sutskever et al. (2014) presented a domain-independent method for learning to map input sequences to output sequences that achieved strong results on a large scale machine translation problem. In this work, we show that precisely the same sequence-to-sequence method achieves results that are close to state-of-the-art on syntactic constituency parsing, whilst making almost no assumptions about the structure of the problem."

>	"An attention mechanism inspired by Bahdanau et al.'s model made our parser much more statistically efficient. In addition, the model learned a stack decoder solely from data. Our model (an ensemble) matches the BerkeleyParser when trained on the labeled 40K training sentences. When training our model on a large number of automatically-generated high-confidence parses, we achieve the best published results on constituency parsing."


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


#### ["Globally Normalized Transition-Based Neural Networks"](http://arxiv.org/abs/1603.06042) Andor, Alberti, Weiss, Severyn, Presta, Ganchev, Petrov, Collins
  `SyntaxNet` `Parsey McParseface`
>	"We introduce a globally normalized transition-based neural network model that achieves state-of-the-art part-of-speech tagging, dependency parsing and sentence compression results. Our model is a simple feed-forward neural network that operates on a task-specific transition system, yet achieves comparable or better accuracies than recurrent models. We discuss the importance of global as opposed to local normalization: a key insight is that the label bias problem implies that globally normalized models can be strictly more expressive than locally normalized models."

>	"The parser uses a feed forward NN, which is much faster than the RNN usually used for parsing. Also the paper is using a global method to solve the label bias problem. This method can be used for many tasks and indeed in the paper it is used also to shorten sentences by throwing unnecessary words. The label bias problem arises when predicting each label in a sequence using a softmax over all possible label values in each step. This is a local approach but what we are really interested in is a global approach in which the sequence of all labels that appeared in a training example are normalized by all possible sequences. This is intractable so instead a beam search is performed to generate alternative sequences to the training sequence. The search is stopped when the training sequence drops from the beam or ends. The different beams with the training sequence are then used to compute the global loss."

  - `notes` <http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1603.06042>
  - `code` <https://github.com/tensorflow/models/tree/master/research/syntaxnet>



---
### interesting papers - text classification


#### ["#TAGSPACE: Semantic Embeddings from Hashtags"](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/) Weston, Chopra, Adams
>	"We describe a convolutional neural network that learns feature representations for short textual posts using hashtags as a supervised signal. The proposed approach is trained on up to 5.5 billion words predicting 100,000 possible hashtags. As well as strong performance on the hashtag prediction task itself, we show that its learned representation of text (ignoring the hashtag labels) is useful for other tasks as well. To that end, we present results on a document recommendation task, where it also outperforms a number of baselines."

  - `video` <http://youtube.com/watch?v=K5C9TPaxuWM> (Weston)


#### ["Character-level Convolutional Networks for Text Classification"](http://arxiv.org/abs/1509.01626) Zhang, Zhao, LeCun
>	"This article offers an empirical exploration on the use of character-level convolutional networks (ConvNets) for text classification. We constructed several large-scale datasets to show that character-level convolutional networks could achieve state-of-the-art or competitive results. Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF variants, and deep learning models such as word-based ConvNets and recurrent neural networks."

>	"In most neural net NLP systems, the first layer maps words to vectors through a (learned) lookup table. Our system just looks at individual characters. This makes it robust to misspelling and sensitive to morphology, i.e. an known word with an unknown prefix or suffix, or an unknown word with a similar morphology as a known one, or a word formed by concatenating several known words, will be handled properly. This allows the system to handle essentially unlimited vocabularies."

  - `code` <https://github.com/mhjabreel/CharCNN>


#### ["Bag of Tricks for Efficient Text Classification"](http://arxiv.org/abs/1607.01759) Joulin, Grave, Bojanowski, Mikolov
  `fastText`
  - <https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#bag-of-tricks-for-efficient-text-classification-joulin-grave-bojanowski-mikolov>


#### ["Deep Multi-Instance Transfer Learning"](http://arxiv.org/abs/1411.3128) Kotzias, Denil, Blunsom, Freitas
>	"We present a new approach for transferring knowledge from groups to individuals that comprise them. We evaluate our method in text, by inferring the ratings of individual sentences using full-review ratings. This approach combines ideas from transfer learning, deep learning and multi-instance learning, and reduces the need for laborious human labelling of fine-grained data when abundant labels are available at the group level."

>	"In this work, we present a novel objective function, for instance learning in an a multi-instance learning setting. A similarity measure between instances is required in order to optimise the objective function. Deep Neural Networks have been very successful in creating representations of data, that capture their underlying characteristics. This work capitalises on their success by using embeddings of data and their similarity, as produced by a deep network, as instances for experiments. In this paper we show that this idea can be used to infer ratings of sentences (individuals) from ratings of reviews (groups of sentences). This enables us to extract the most positive and negative sentences in a review. In applications where reviews are overwhelmingly positive, detecting negative comments is a key step toward improving costumer service."

----
>	"sentence-level and entity-level sentiment classification learned from text-level classification"



---
### interesting papers - topic modeling


#### ["Discovering Discrete Latent Topics with Neural Variational Inference"](https://arxiv.org/abs/1706.00359) Miao, Grefenstette, Blunsom
>	"Topic models have been widely explored as probabilistic generative models of documents. Traditional inference methods have sought closed-form derivations for updating the models, however as the expressiveness of these models grows, so does the difficulty of performing fast and accurate inference over their parameters. This paper presents alternative neural approaches to topic modelling by providing parameterisable distributions over topics which permit training by backpropagation in the framework of neural variational inference. In addition, with the help of a stick-breaking construction, we propose a recurrent network that is able to discover a notionally unbounded number of topics, analogous to Bayesian non-parametric topic models. Experimental results on the MXM Song Lyrics, 20NewsGroups and Reuters News datasets demonstrate the effectiveness and efficiency of these neural topic models."

  - `video` <https://vimeo.com/238222598> (Miao)


#### ["Additive Regularization of Topic Models"](https://link.springer.com/article/10.1007/s10994-014-5476-6) Vorontsov, Potapenko
  `ARTM`
>	"Probabilistic topic modeling of text collections has been recently developed mainly within the framework of graphical models and Bayesian inference. In this paper we introduce an alternative semi-probabilistic approach, which we call additive regularization of topic models (ARTM). Instead of building a purely probabilistic generative model of text we regularize an ill-posed problem of stochastic matrix factorization by maximizing a weighted sum of the log-likelihood and additional criteria. This approach enables us to combine probabilistic assumptions with linguistic and problem-specific requirements in a single multi-objective topic model. In the theoretical part of the work we derive the regularized EM-algorithm and provide a pool of regularizers, which can be applied together in any combination. We show that many models previously developed within Bayesian framework can be inferred easier within ARTM and in some cases generalized. In the experimental part we show that a combination of sparsing, smoothing, and decorrelation improves several quality measures at once with almost no loss of the likelihood."

  - `video` <https://youtube.com/watch?v=00qF2yMuRkQ> (Vorontsov)
  - `video` <https://youtu.be/3Lxb-DqPtv4?t=1h45m9s> (Vorontsov) `in russian`
  - `video` <https://youtube.com/watch?v=eJzNAhsbQNI> (Vorontsov) `in russian`
  - `video` <https://youtube.com/watch?v=Y7lGYjJ7TR8> (Ianina) `in russian`
  - `paper` ["Fast and Modular Regularized Topic Modelling"](https://fruct.org/publications/fruct21/files/Koc.pdf) by Kochedykov et al.
  - `paper` <http://www.machinelearning.ru/wiki/images/d/d5/Voron17survey-artm.pdf> by Vorontsov `in russian`
  - `code` <https://github.com/bigartm/bigartm>



---
### interesting papers - text summarization


#### ["Modelling, Visualising and Summarising Documents with a Single Convolutional Neural Network"](http://arxiv.org/abs/1406.3830) Denil, Demiraj, Kalchbrenner, Blunsom, Freitas
>	"Capturing the compositional process which maps the meaning of words to that of documents is a central challenge for researchers in Natural Language Processing and Information Retrieval. We introduce a model that is able to represent the meaning of documents by embedding them in a low dimensional vector space, while preserving distinctions of word and sentence order crucial for capturing nuanced semantics. Our model is based on an extended Dynamic Convolution Neural Network, which learns convolution filters at both the sentence and document level, hierarchically learning to capture and compose low level lexical features into high level semantic concepts. We demonstrate the effectiveness of this model on a range of document modelling tasks, achieving strong results with no feature engineering and with a more compact model. Inspired by recent advances in visualising deep convolution networks for computer vision, we present a novel visualisation technique for our document networks which not only provides insight into their learning process, but also can be interpreted to produce a compelling automatic summarisation system for texts."

  - `code` <https://github.com/mdenil/txtnets>


#### ["Extraction of Salient Sentences from Labelled Documents"](http://arxiv.org/abs/1412.6815) Denil, Demiraj, de Freitas
>	"We present a hierarchical convolutional document model with an architecture designed to support introspection of the document structure. Using this model, we show how to use visualisation techniques from the computer vision literature to identify and extract topic-relevant sentences. We also introduce a new scalable evaluation technique for automatic sentence extraction systems that avoids the need for time consuming human annotation of validation data."

  - `code` <https://github.com/mdenil/txtnets>


#### ["Sentence Compression by Deletion with LSTMs"](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43852.pdf) Filippova, Alfonseca, Colmenares, Kaiser, Vinyals
>	"We present an LSTM approach to deletion-based sentence compression where the task is to translate a sentence into a sequence of zeros and ones, corresponding to token deletion decisions. We demonstrate that even the most basic version of the system, which is given no syntactic information (no PoS or NE tags, or dependencies) or desired compression length, performs surprisingly well: around 30% of the compressions from a large test set could be regenerated. We compare the LSTM system with a competitive baseline which is trained on the same amount of data but is additionally provided with all kinds of linguistic features. In an experiment with human raters the LSTM-based model outperforms the baseline achieving 4.5 in readability and 3.8 in informativeness."

>	"We presented, to our knowledge, a first attempt at building a competitive compression system which is given no linguistic features from the input. The two important components of the system are (1) word embeddings, which can be obtained by anyone either pre-trained, or by running word2vec on a large corpus, and (2) an LSTM model which draws on the very recent advances in research on RNNs. The training data of about two million sentence-compression pairs was collected automatically from the Internet. Our results clearly indicate that a compression model which is not given syntactic information explicitly in the form of features may still achieve competitive performance. The high readability and informativeness scores assigned by human raters support this claim. In the future, we are planning to experiment with more “interesting” paraphrasing models which translate the input not into a zero-one sequence but into words."

  - `video` <https://vimeo.com/159981337> (Alfonseca)


#### ["A Neural Attention Model for Abstractive Sentence Summarization"](http://arxiv.org/abs/1509.00685) Rush, Chopra, Weston
>	"Summarization based on text extraction is inherently limited, but generation-style abstractive methods have proven challenging to build. In this work, we propose a fully data-driven approach to abstractive sentence summarization. Our method utilizes a local attention-based model that generates each word of the summary conditioned on the input sentence. While the model is structurally simple, it can easily be trained end-to-end and scales to a large amount of training data. The model shows significant performance gains on the DUC-2004 shared task compared with several strong baselines."

>	"We have presented a neural attention-based model for abstractive summarization, based on recent developments in neural machine translation. We combine this probabilistic model with a generation algorithm which produces accurate abstractive summaries. As a next step we would like to further improve the grammaticality of the summaries in a data-driven way, as well as scale this system to generate paragraph-level summaries. Both pose additional challenges in terms of efficient alignment and consistency in generation."

  - `video` <https://vimeo.com/159993537> (Rush)
  - `code` <https://github.com/jaseweston/NAMAS>
  - `code` <https://github.com/carpedm20/neural-summary-tensorflow>


#### ["Sequence-to-Sequence RNNs for Text Summarization"](http://arxiv.org/abs/1602.06023) Nallapati, Xiang, Zhou
>	"In this work, we cast text summarization as a sequence-to-sequence problem and apply the attentional encoder-decoder RNN that has been shown to be successful for Machine Translation (Bahdanau et al. (2014)). Our experiments show that the proposed architecture significantly outperforms the state-of-the art model of Rush et al. (2015) on the Gigaword dataset without any additional tuning. We also propose additional extensions to the standard architecture, which we show contribute to further improvement in performance."



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
