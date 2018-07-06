

  * [**overview**](#overview)
  * [**interesting papers**](#interesting-papers)
    - [**ranking**](#interesting-papers---ranking)
    - [**document models**](#interesting-papers---document-models)
    - [**entity-centric search**](#interesting-papers---entity-centric-search)



---
### overview

  ["Foundations of Information Retrieval"](https://drive.google.com/file/d/0B-GJrccmbImkZ3pjNl9sczQxd3M) by Maarten de Rijke (SIGIR 2017) `slides`

  ["What Every Software Engineer Should Know about Search"](https://medium.com/startup-grind/what-every-software-engineer-should-know-about-search-27d1df99f80d) by Max Grigorev

  ["An Introduction to Information Retrieval"](https://nlp.stanford.edu/IR-book/) book by Manning, Raghavan, Schutze  
  ["Search Engines. Information Retrieval in Practice"](http://ciir.cs.umass.edu/irbook/) book by Croft, Metzler, Strohman  

----

  ["Neural Networks for Information Retrieval"](http://nn4ir.com) tutorials (ECIR 2018, WSDM 2018, SIGIR 2017) `slides`  
  ["Neural Text Embeddings for Information Retrieval"](https://microsoft.com/en-us/research/event/wsdm-2017-tutorial-neural-text-embeddings-information-retrieval/)
	tutorial by Bhaskar Mitra and Nick Craswell (WSDM 2017)
	([slides](https://slideshare.net/BhaskarMitra3/neural-text-embeddings-for-information-retrieval-wsdm-2017), [paper](https://arxiv.org/abs/1705.01509))  

  ["Neural Models for Information Retrieval"](https://youtube.com/watch?v=g1Pgo5yTIKg) by Bhaskar Mitra `video`

----

  [course](http://youtube.com/watch?v=5L1qemKyUKA&index=75&list=PL6397E4B26D00A269) by Chris Manning `video`  
  [course](http://homepages.inf.ed.ac.uk/vlavrenk/tts.html) by Victor Lavrenko `video`  

  [course](https://compscicenter.ru/courses/information-retrieval/2016-autumn/) from Yandex `video` `in russian`  
  [course](https://compsciclub.ru/courses/informationretrieval) from Yandex `video` `in russian`  

  course from Mail.ru
	([first part](https://youtube.com/playlist?list=PLrCZzMib1e9o_BlrSB5bFkLq8h2i4pQjz),
	[second part](https://youtube.com/playlist?list=PLrCZzMib1e9o7YIhOfJtD1EaneGOGkN-_)) `video` `in russian`  
  [course](http://habrahabr.ru/company/mailru/blog/257119/) from Mail.ru `video` `in russian`  

  [course](http://nzhiltsov.github.io/IR-course/) by Nikita Zhiltsov `in russian`

  [overview](https://youtube.com/watch?v=3R6vBd_Y8O4) of ranking by Sergey Nikolenko `video` `in russian`  
  overview of ranking by Nikita Volkov
	([first part](https://youtube.com/watch?v=GctrEpJinhI),
	[second part](https://youtube.com/watch?v=GZmXKBzIfkA)) `video` `in russian`  

---

  challenges:
  - full text document retrieval, passage retrieval, question answering
  - web search, searching social media, distributed information retrieval, entity ranking
  - learning to rank combined with neural network based representation learning
  - user and task modelling, personalized search, diversity
  - query formulation assistance, query recommendation, conversational search
  - multimedia retrieval
  - learning dense representations for long documents
  - dealing with rare queries and rare words
  - modelling text at different granularities (character, word, passage, document)
  - compositionality of vector representations
  - jointly modelling queries, documents, entities and other structured data



---
### interesting papers

  - [**ranking**](#interesting-papers---ranking)  
  - [**document models**](#interesting-papers---document-models)  
  - [**entity-centric search**](#interesting-papers---entity-centric-search)  

----

  - [**question answering over texts**](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-texts)  
  - [**question answering over knowledge bases**](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-knowledge-bases)  
  - [**information extraction and integration**](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---information-extraction-and-integration)  

----

[**interesting recent papers**](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reasoning)


----

#### ["Neural Information Retrieval: at the End of the Early Years"](https://link.springer.com/content/pdf/10.1007%2Fs10791-017-9321-y.pdf) Onal et al.
>	"In this paper, we survey the current landscape of Neural IR research, paying special attention to the use of learned distributed representations of textual units. We highlight the successes of neural IR thus far, catalog obstacles to its wider adoption, and suggest potentially promising directions for future research."


#### ["Neural Models for Information Retrieval"](https://arxiv.org/abs/1705.01509) Mitra, Craswell
>	"Neural ranking models for information retrieval use shallow or deep neural networks to rank search results in response to a query. Traditional learning to rank models employ machine learning techniques over hand-crafted IR features. By contrast, neural models learn representations of language from raw text that can bridge the gap between query and document vocabulary. Unlike classical IR models, these new machine learning based approaches are data-hungry, requiring large scale training data before they can be deployed. This tutorial introduces basic concepts and intuitions behind neural IR models, and places them in the context of traditional retrieval models. We begin by introducing fundamental concepts of IR and different neural and non-neural approaches to learning vector representations of text. We then review shallow neural IR methods that employ pre-trained neural term embeddings without learning the IR task end-to-end. We introduce deep neural networks next, discussing popular deep architectures. Finally, we review the current DNN models for information retrieval. We conclude with a discussion on potential future directions for neural IR."

  - `video` <https://youtube.com/watch?v=g1Pgo5yTIKg> (Mitra)
  - `slides` <https://slideshare.net/BhaskarMitra3/neural-text-embeddings-for-information-retrieval-wsdm-2017>


#### ["Online Evaluation for Information Retrieval"](https://microsoft.com/en-us/research/publication/online-evaluation-information-retrieval) Hofmann, Li, Radlinski
>	"Online evaluation is one of the most common approaches to measure the effectiveness of an information retrieval system. It involves fielding the information retrieval system to real users, and observing these users’ interactions in-situ while they engage with the system. This allows actual users with real world information needs to play an important part in assessing retrieval quality. As such, online evaluation complements the common alternative offline evaluation approaches which may provide more easily interpretable outcomes, yet are often less realistic when measuring of quality and actual user experience.  
>	In this survey, we provide an overview of online evaluation techniques for information retrieval. We show how online evaluation is used for controlled experiments, segmenting them into experiment designs that allow absolute or relative quality assessments. Our presentation of different metrics further partitions online evaluation based on different sized experimental units commonly of interest: documents, lists and sessions. Additionally, we include an extensive discussion of recent work on data re-use, and experiment estimation based on historical data.  
>	A substantial part of this work focuses on practical issues: How to run evaluations in practice, how to select experimental parameters, how to take into account ethical considerations inherent in online evaluations, and limitations. While most published work on online experimentation today is at large scale in systems with millions of users, we also emphasize that the same techniques can be applied at small scale. To this end, we emphasize recent work that makes it easier to use at smaller scales and encourage studying real-world information seeking in a wide range of scenarios. Finally, we present a summary of the most recent work in the area, and describe open problems, as well as postulating future directions."



---
### interesting papers - ranking


#### ["Learning Rank Functionals: An Empirical Study"](https://arxiv.org/abs/1407.6089) Tran et al.
>	"Ranking is a key aspect of many applications, such as information retrieval, question answering, ad placement and recommender systems. Learning to rank has the goal of estimating a ranking model automatically from training data. In practical settings, the task often reduces to estimating a rank functional of an object with respect to a query. In this paper, we investigate key issues in designing an effective learning to rank algorithm. These include data representation, the choice of rank functionals, the design of the loss function so that it is correlated with the rank metrics used in evaluation. For the loss function, we study three techniques: approximating the rank metric by a smooth function, decomposition of the loss into a weighted sum of element-wise losses and into a weighted sum of pairwise losses. We then present derivations of piecewise losses using the theory of high-order Markov chains and Markov random fields. In experiments, we evaluate these design aspects on two tasks: answer ranking in a Social Question Answering site, and Web Information Retrieval."


#### ["Learning to Rank using Gradient Descent"](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) Burges et al.
  `learning to rank using relevance labels` `RankNet`
>	"We investigate using gradient descent methods for learning ranking functions; we propose a simple probabilistic cost function, and we introduce RankNet, an implementation of these ideas using a neural network to model the underlying ranking function. We present test results on toy data and on data from a commercial internet search engine."

>	"We have proposed a probabilistic cost for training systems to learn ranking functions using pairs of training examples. The approach can be used for any differentiable function; we explored using a neural network formulation, RankNet. RankNet is simple to train and gives excellent performance on a real world ranking problem with large amounts of data. Comparing the linear RankNet with other linear systems clearly demonstrates the benefit of using our pair-based cost function together with gradient descent; the two layer net gives further improvement. For future work it will be interesting to investigate extending the approach to using other machine learning methods for the ranking function; however evaluation speed and simplicity is a critical constraint for such systems."

  - `video` <http://videolectures.net/icml2015_burges_learning_to_rank/> (Burges)
  - `video` <https://youtu.be/3R6vBd_Y8O4> (Nikolenko) `in russian`
  - `code` <https://github.com/shiba24/learning2rank>


#### ["Learning to Rank with Nonsmooth Cost Functions"](https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions) Burges et al.
  `learning to rank using relevance labels` `LambdaRank`
>	"The quality measures used in information retrieval are particularly difficult to optimize directly, since they depend on the model scores only through the sorted order of the documents returned for a given query. Thus, the derivatives of the cost with respect to the model parameters are either zero, or are undefined. In this paper, we propose a class of simple, flexible algorithms, called LambdaRank, which avoids these difficulties by working with implicit cost functions. We describe LambdaRank using neural network models, although the idea applies to any differentiable function class. We give necessary and sufficient conditions for the resulting implicit cost function to be convex, and we show that the general method has a simple mechanical interpretation. We demonstrate significantly improved accuracy, over a state-of-the-art ranking algorithm, on several datasets. We also show that LambdaRank provides a method for significantly speeding up the training phase of that ranking algorithm. Although this paper is directed towards ranking, the proposed method can be extended to any non-smooth and multivariate cost functions."

----
>	"LambdaRank is a method for learning arbitrary information retrieval measures; it can be applied to any algorithm that learns through gradient descent. LambdaRank is a listwise method, in that the cost depends on the sorted order of the documents. The key LambdaRank insight is to define the gradient of the cost with respect to the score that the model assigns to a given xi after all of the xi have been sorted by their scores si; thus the gradients take into account the rank order of the documents, as defined by the current model. LambdaRank is an empirical algorithm, in that the form that the gradients take was chosen empirically: the λ’s are those gradients, and the contribution to a given feature vector xi’s λi from a pair (xi, xj), y(xi) != y(xj), is just the gradient of the logistic regression loss (viewed as a function of si - sj) multiplied by the change in Z caused by swapping the rank positions of the two documents while keeping all other documents fixed, where Z is the information retrieval measure being learned. λi is then the sum of contributions for all such pairs. Remarkably, it has been shown that a LambdaRank model trained on Z, for Z equal to Normalized Cumulative Discounted Gain (NDCG), Mean Reciprocal Rank, or Mean Average Precision (three commonly used IR measures), given sufficient training data, consistently finds a local optimum of that IR measure (in the space of the measure viewed as a function of the model parameters)."

  - `video` <https://youtu.be/3R6vBd_Y8O4?t=32m8s> (Nikolenko) `in russian`
  - `paper` ["Learning to Rank Using an Ensemble of Lambda-Gradient Models"](http://proceedings.mlr.press/v14/burges11a/burges11a.pdf) by Burges et al. (optimizing Expected Reciprocal Rank)


#### ["From RankNet to LambdaRank to LambdaMART: An Overview"](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) Burges
  `learning to rank using relevance labels` `LambdaMART`
>	"LambdaMART is the boosted tree version of LambdaRank, which is based on RankNet. RankNet, LambdaRank, and LambdaMART have proven to be very successful algorithms for solving real world ranking problems: for example an ensemble of LambdaMART rankers won Track 1 of the 2010 Yahoo! Learning To Rank Challenge. The details of these algorithms are spread across several papers and reports, and so here we give a self-contained, detailed and complete description of them."

----
>	"While LambdaRank was originally instantiated using neural nets, it was found that a boosted tree multiclass classifier (McRank) gave improved performance. Combining these ideas led to LambdaMART, which instantiates the LambdaRank idea using gradient boosted decision trees. This work showed that McRank’s improved performance over LambdaRank (instantiated in a neural net) is due to the difference in the expressiveness of the underlying models (boosted decision trees versus neural nets) rather than being due to an inherent limitation of the lambda-gradient idea."

>	"LambdaMART combines LambdaRank and MART (Multiple Additive Regression Trees). While MART uses gradient boosted decision trees for prediction tasks, LambdaMART uses gradient boosted decision trees using a cost function derived from LambdaRank for solving a ranking task. On experimental datasets, LambdaMART has shown better results than LambdaRank and the original RankNet."

>	"Cascade of trees, in which each new tree contributes to a gradient step in the direction that minimises the loss function. The ensemble of these trees is the final model. LambdaMART uses this ensemble but it replaces that gradient with the lambda (gradient computed given the candidate pairs) presented in LambdaRank."

  - `video` <https://youtu.be/3R6vBd_Y8O4?t=48m9s> (Nikolenko) `in russian`
  - `post` <https://wellecks.wordpress.com/2015/02/21/peering-into-the-black-box-visualizing-lambdamart/>
  - `paper` ["Learning to Rank Using an Ensemble of Lambda-Gradient Models"](http://proceedings.mlr.press/v14/burges11a/burges11a.pdf) by Burges et al.


#### ["An Efficient Boosting Algorithm for Combining Preferences"](http://jmlr.org/papers/volume4/freund03a/freund03a.pdf) Freund, Iyer, Schapire, Singer
  `learning to rank using relevance labels` `RankBoost`
>	"We study the problem of learning to accurately rank a set of objects by combining a given collection of ranking or preference functions. This problem of combining preferences arises in several applications, such as that of combining the results of different search engines, or the "collaborative-filtering" problem of ranking movies for a user based on the movie rankings provided by other users. In this work, we begin by presenting a formal framework for this general problem. We then describe and analyze an efficient algorithm called RankBoost for combining preferences based on the boosting approach to machine learning. We give theoretical results describing the algorithm's behavior both on the training data, and on new test data not seen during training. We also describe an efficient implementation of the algorithm for a particular restricted but common case. We next discuss two experiments we carried out to assess the performance of RankBoost. In the first experiment, we used the algorithm to combine different web search strategies, each of which is a query expansion for a given domain. The second experiment is a collaborative-filtering task for making movie recommendations."

  - `video` <https://youtu.be/3R6vBd_Y8O4?t=42m13s> (Nikolenko) `in russian`


#### ["Neural Ranking Models with Weak Supervision"](https://arxiv.org/abs/1704.08803) Dehghani, Zamani, Severyn, Kamps, Croft
  `unsupervised learning to rank`
>	"Despite the impressive improvements achieved by unsupervised deep neural networks in computer vision and NLP tasks, such improvements have not yet been observed in ranking for information retrieval. The reason may be the complexity of the ranking problem, as it is not obvious how to learn from queries and documents when no supervised signal is available. Hence, in this paper, we propose to train a neural ranking model using weak supervision, where labels are obtained automatically without human annotators or any external resources (e.g., click data). To this aim, we use the output of an unsupervised ranking model, such as BM25, as a weak supervision signal. We further train a set of simple yet effective ranking models based on feed-forward neural networks. We study their effectiveness under various learning scenarios (point-wise and pair-wise models) and using different input representations (i.e., from encoding query-document pairs into dense/sparse vectors to using word embedding representation). We train our networks using tens of millions of training instances and evaluate it on two standard collections: a homogeneous news collection (Robust) and a heterogeneous large-scale web collection (ClueWeb). Our experiments indicate that employing proper objective functions and letting the networks to learn the input representation based on weakly supervised data leads to impressive performance, with over 13% and 35% MAP improvements over the BM25 model on the Robust and the ClueWeb collections. Our findings also suggest that supervised neural ranking models can greatly benefit from pre-training on large amounts of weakly labeled data that can be easily obtained from unsupervised IR models."

  - `post` <https://mostafadehghani.com/2017/04/23/beating-the-teacher-neural-ranking-models-with-weak-supervision> (Dehghani)
  - `slides` <http://mostafadehghani.com/wp-content/uploads/2016/07/SIGIR2017_Presentation.pdf>


#### ["Gathering Additional Feedback on Search Results by Multi-Armed Bandits with Respect to Production Ranking"](http://www.www2015.it/documents/proceedings/proceedings/p1177.pdf) Vorobev, Lefortier, Gusev, Serdyukov
  `online learning to rank using click data` `BBRA`
>	"Given a repeatedly issued query and a document with a not-yet-confirmed potential to satisfy the users’ needs, a search system should place this document on a high position in order to gather user feedback and obtain a more confident estimate of the document utility. On the other hand, the main objective of the search system is to maximize expected user satisfaction over a rather long period, what requires showing more relevant documents on average. The state-of-the-art approaches to solving this exploration-exploitation dilemma rely on strongly simplified settings making these approaches infeasible in practice. We improve the most flexible and pragmatic of them to handle some actual practical issues. The first one is utilizing prior information about queries and documents, the second is combining bandit-based learning approaches with a default production ranking algorithm. We show experimentally that our framework enables to significantly improve the ranking of a leading commercial search engine."


#### ["Online Learning to Rank in Stochastic Click Models"](https://arxiv.org/abs/1703.02527) Zoghi, Tunys, Ghavamzadeh, Kveton, Szepesvari, Wen
  `online learning to rank using click data` `BatchRank`
>	"Online learning to rank is a core problem in information retrieval and machine learning. Many provably efficient algorithms have been recently proposed for this problem in specific click models. The click model is a model of how the user interacts with a list of documents. Though these results are significant, their impact on practice is limited, because all proposed algorithms are designed for specific click models and lack convergence guarantees in other models. In this work, we propose BatchRank, the first online learning to rank algorithm for a broad class of click models. The class encompasses two most fundamental click models, the cascade and position-based models. We derive a gap-dependent upper bound on the T-step regret of BatchRank and evaluate it on a range of web search queries. We observe that BatchRank outperforms ranked bandits and is more robust than CascadeKL-UCB, an existing algorithm for the cascade model."

  - `video` <https://youtu.be/__En7H2awqM?t=24m21s> (Szepesvari)


#### ["A Neural Click Model for Web Search"](http://www2016.net/proceedings/proceedings/p531.pdf) Borisov, Markov, Rijke, Serdyukov
  `click prediction`
>	"Understanding user browsing behavior in web search is key to improving web search effectiveness. Many click models have been proposed to explain or predict user clicks on search engine results. They are based on the probabilistic graphical model (PGM) framework, in which user behavior is represented as a sequence of observable and hidden events. The PGM framework provides a mathematically solid way to reason about a set of events given some information about other events. But the structure of the dependencies between the events has to be set manually. Different click models use different hand-crafted sets of dependencies. We propose an alternative based on the idea of distributed representations: to represent the user’s information need and the information available to the user with a vector state. The components of the vector state are learned to represent concepts that are useful for modeling user behavior. And user behavior is modeled as a sequence of vector states associated with a query session: the vector state is initialized with a query, and then iteratively updated based on information about interactions with the search engine results. This approach allows us to directly understand user browsing behavior from click-through data, i.e., without the need for a predefined set of rules as is customary for PGM-based click models. We illustrate our approach using a set of neural click models. Our experimental results show that the neural click model that uses the same training data as traditional PGM-based click models, has better performance on the click prediction task (i.e., predicting user click on search engine results) and the relevance prediction task (i.e., ranking documents by their relevance to a query). An analysis of the best performing neural click model shows that it learns similar concepts to those used in traditional click models, and that it also learns other concepts that cannot be designed manually."


#### ["A Click Sequence Model for Web Search"](https://arxiv.org/abs/1805.03411) Borisov, Wardenaar, Markov, Rijke
  `click prediction`
>	"Getting a better understanding of user behavior is important for advancing information retrieval systems. Existing work focuses on modeling and predicting single interaction events, such as clicks. In this paper, we for the first time focus on modeling and predicting sequences of interaction events. And in particular, sequences of clicks. We formulate the problem of click sequence prediction and propose a click sequence model (CSM) that aims to predict the order in which a user will interact with search engine results. CSM is based on a neural network that follows the encoder-decoder architecture. The encoder computes contextual embeddings of the results. The decoder predicts the sequence of positions of the clicked results. It uses an attention mechanism to extract necessary information about the results at each timestep. We optimize the parameters of CSM by maximizing the likelihood of observed click sequences. We test the effectiveness of CSM on three new tasks: (i) predicting click sequences, (ii) predicting the number of clicks, and (iii) predicting whether or not a user will interact with the results in the order these results are presented on a search engine result page (SERP). Also, we show that CSM achieves state-of-the-art results on a standard click prediction task, where the goal is to predict an unordered set of results a user will click on."



---
### interesting papers - document models

[**interesting papers - question answering over texts**](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-texts)


#### ["Learning Deep Structured Semantic Models for Web Search using Clickthrough Data"](http://research.microsoft.com/apps/pubs/default.aspx?id=198202) Huang, He, Gao, Deng, Acero, Heck
  `DSSM`
  - <https://github.com/brylevkirill/notes/blob/master/Natural%20Language%20Processing.md#learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data-huang-he-gao-deng-acero-heck>


#### ["A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval"](http://www.msr-waypoint.com/pubs/226585/cikm2014_cdssm_final.pdf) Shen, He, Gao, Deng, Mesnil
  `CLSM`
>	"In this paper, we propose a new latent semantic model that incorporates a convolutional-pooling structure over word sequences to learn low-dimensional, semantic vector representations for search queries and Web documents. In order to capture the rich contextual structures in a query or a document, we start with each word within a temporal context window in a word sequence to directly capture contextual features at the word n-gram level. Next, the salient word n-gram features in the word sequence are discovered by the model and are then aggregated to form a sentence-level feature vector. Finally, a non-linear transformation is applied to extract high-level semantic information to generate a continuous vector representation for the full text string. The proposed convolutional latent semantic model is trained on clickthrough data and is evaluated on a Web document ranking task using a large-scale, real-world data set. Results show that the proposed model effectively captures salient semantic information in queries and documents for the task while significantly outperforming previous state-of-the-art semantic models."

>	"In this paper, we have reported a novel deep learning architecture called the CLSM, motivated by the convolutional structure of the CNN, to extract both local contextual features at the word-n-gram level (via the convolutional layer) and global contextual features at the sentence-level (via the max-pooling layer) from text. The higher layer(s) in the overall deep architecture makes effective use of the extracted context-sensitive features to generate latent semantic vector representations which facilitates semantic matching between documents and queries for Web search applications. We have carried out extensive experimental studies of the proposed model whereby several state-of-the-art semantic models are compared and significant performance improvement on a large-scale real-world Web search data set is observed. Extended from our previous work on DSSM and C-DSSM models, the CLSM and its variations have also been demonstrated giving superior performance on a range of natural language processing tasks beyond information retrieval, including semantic parsing and question answering, entity search and online recommendation."

  - `video` <https://youtu.be/x7B6RudUQLI?t=1h33m39s> (Gulin) `in russian`
  - `code` <https://github.com/airalcorn2/Deep-Semantic-Similarity-Model>


#### ["Modeling Interestingness with Deep Neural Networks"](http://research.microsoft.com/apps/pubs/default.aspx?id=226584) Gao, Pantel, Gamon, He, Deng
>	"This paper presents a deep semantic similarity model, a special type of deep neural networks designed for text analysis, for recommending target documents to be of interest to a user based on a source document that she is reading. We observe, identify, and detect naturally occurring signals of interestingness in click transitions on the Web between source and target documents, which we collect from commercial Web browser logs. The DSSM is trained on millions of Web transitions, and maps source-target document pairs to feature vectors in a latent space in such a way that the distance between source documents and their corresponding interesting targets in that space is minimized. The effectiveness of the DSSM is demonstrated using two interestingness tasks: automatic highlighting and contextual entity search. The results on large-scale, real-world datasets show that the semantics of documents are important for modeling interestingness and that the DSSM leads to significant quality improvement on both tasks, outperforming not only the classic document models that do not use semantics but also state-of-the-art topic models."

  - `video` <https://youtube.com/watch?v=YXi66Zgd0D0> (Yih)


#### ["Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks"](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf) Severyn, Moschitti
>	"Learning a similarity function between pairs of objects is at the core of learning to rank approaches. In information retrieval tasks we typically deal with query-document pairs, in question answering - question-answer pairs. However, before learning can take place, such pairs needs to be mapped from the original space of symbolic words into some feature space encoding various aspects of their relatedness, e.g. lexical, syntactic and semantic. Feature engineering is often a laborious task and may require external knowledge sources that are not always available or difficult to obtain. Recently, deep learning approaches have gained a lot of attention from the research community and industry for their ability to automatically learn optimal feature representation for a given task, while claiming state-of-the-art performance in many tasks in computer vision, speech recognition and natural language processing. In this paper, we present a convolutional neural network architecture for reranking pairs of short texts, where we learn the optimal representation of text pairs and a similarity function to relate them in a supervised way from the available training data. Our network takes only words in the input, thus requiring minimal preprocessing. In particular, we consider the task of reranking short text pairs where elements of the pair are sentences. We test our deep learning system on two popular retrieval tasks from TREC: Question Answering and Microblog Retrieval. Our model demonstrates strong performance on the first task beating previous state-of-the-art systems by about 3% absolute points in both MAP and MRR and shows comparable results on tweet reranking, while enjoying the benefits of no manual feature engineering and no additional syntactic parsers."

  - `code` <https://github.com/aseveryn/deep-qa>
  - `code` <https://github.com/shashankg7/Keras-CNN-QA>


#### ["A Dual Embedding Space Model for Document Ranking"](https://arxiv.org/abs/1602.01137) Mitra, Nalisnick, Craswell, Caruana
  `DESM`
>	"A fundamental goal of search engines is to identify, given a query, documents that have relevant text. This is intrinsically difficult because the query and the document may use different vocabulary, or the document may contain query words without being relevant. We investigate neural word embeddings as a source of evidence in document ranking. We train a word2vec embedding model on a large unlabelled query corpus, but in contrast to how the model is commonly used, we retain both the input and the output projections, allowing us to leverage both the embedding spaces to derive richer distributional relationships. During ranking we map the query words into the input space and the document words into the output space, and compute a query-document relevance score by aggregating the cosine similarities across all the query-document word pairs."

>	"We postulate that the proposed Dual Embedding Space Model (DESM) captures evidence on whether a document is about a query term in addition to what is modelled by traditional term-frequency based approaches. Our experiments show that the DESM can re-rank top documents returned by a commercial Web search engine, like Bing, better than a term-matching based signal like TF-IDF. However, when ranking a larger set of candidate documents, we find the embeddings-based approach is prone to false positives, retrieving documents that are only loosely related to the query. We demonstrate that this problem can be solved effectively by ranking based on a linear mixture of the DESM and the word counting features."

  - `video` <https://youtu.be/g1Pgo5yTIKg?t=30m1s> (Mitra)
  - `code` <https://github.com/bmitra-msft/Demos/blob/master/notebooks/DESM.ipynb>


#### ["Query Expansion with Locally-Trained Word Embeddings"](https://arxiv.org/abs/1605.07891) Diaz, Mitra, Craswell
>	"Continuous space word embeddings have received a great deal of attention in the natural language processing and machine learning communities for their ability to model term similarity and other relationships. We study the use of term relatedness in the context of query expansion for ad hoc information retrieval. We demonstrate that word embeddings such as word2vec and GloVe, when trained globally, underperform corpus and query specific embeddings for retrieval tasks. These results suggest that other tasks benefiting from global embeddings may also benefit from local embeddings."

>	"The success of local embeddings on this task should alarm natural language processing researchers using global embeddings as a representational tool. For one, the approach of learning from vast amounts of data is only effective if the data is appropriate for the task at hand. And, when provided, much smaller high-quality data can provide much better performance. Beyond this, our results suggest that the approach of estimating global representations, while computationally convenient, may overlook insights possible at query time, or evaluation time in general. A similar local embedding approach can be adopted for any natural language processing task where topical locality is expected and can be estimated. Although we used a query to re-weight the corpus in our experiments, we could just as easily use alternative contextual information (e.g. a sentence, paragraph, or document) in other tasks."

>	"Although local embeddings provide effectiveness gains, they can be quite inefficient compared to global embeddings. We believe that there is opportunity to improve the efficiency by considering offline computation of local embeddings at a coarser level than queries but more specialized than the corpus. If the retrieval algorithm is able to select the appropriate embedding at query time, we can avoid training the local embedding."

  - `video` <https://youtu.be/g1Pgo5yTIKg?t=38m58s> (Mitra)


#### ["Learning to Match Using Local and Distributed Representations of Text for Web Search"](https://arxiv.org/abs/1610.08136) Mitra, Diaz, Craswell
  `Duet`
>	"Models such as latent semantic analysis and those based on neural embeddings learn distributed representations of text, and match the query against the document in the latent semantic space. In traditional information retrieval models, on the other hand, terms have discrete or local representations, and the relevance of a document is determined by the exact matches of query terms in the body text. We hypothesize that matching with distributed representations complements matching with traditional local representations, and that a combination of the two is favorable. We propose a novel document ranking model composed of two separate deep neural networks, one that matches the query and the document using a local representation, and another that matches the query and the document using learned distributed representations. The two networks are jointly trained as part of a single neural network. We show that this combination or ‘duet’ performs significantly better than either neural network individually on a Web page ranking task, and also significantly outperforms traditional baselines and other recently proposed models based on neural networks."

  - `video` <https://youtu.be/g1Pgo5yTIKg?t=46m31s> (Mitra)
  - `code` <https://github.com/faneshion/MatchZoo>
  - `code` <https://github.com/bmitra-msft/NDRM/blob/master/notebooks/Duet.ipynb>



---
### interesting papers - entity-centric search

[**interesting papers - question answering over knowledge bases**](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-knowledge-bases)  
[**interesting papers - information extraction and integration**](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---information-extraction-and-integration)  


#### ["Fast and Space-Efficient Entity Linking in Queries"](http://labs.yahoo.com/publication/fast-and-space-efficient-entity-linking-in-queries/) Blanco, Ottaviano, Meij
>	"Entity linking deals with identifying entities from a knowledge base in a given piece of text and has become a fundamental building block for web search engines, enabling numerous downstream improvements from better document ranking to enhanced search results pages. A key problem in the context of web search queries is that this process needs to run under severe time constraints as it has to be performed before any actual retrieval takes place, typically within milliseconds. In this paper we propose a probabilistic model that leverages user-generated information on the web to link queries to entities in a knowledge base. There are three key ingredients that make the algorithm fast and space-efficient. First, the linking process ignores any dependencies between the different entity candidates, which allows for a O(k^2) implementation in the number of query terms. Second, we leverage hashing and compression techniques to reduce the memory footprint. Finally, to equip the algorithm with contextual knowledge without sacrificing speed, we factor the distance between distributional semantics of the query words and entities into the model. We show that our solution significantly outperforms several state-of-the-art baselines by more than 14% while being able to process queries in sub-millisecond times—at least two orders of magnitude faster than existing systems."


#### ["Jigs and Lures: Associating Web Queries with Structured Entities"](http://www.aclweb.org/anthology/P11-1009) Pantel, Fuxman
>	"We propose methods for estimating the probability that an entity from an entity database is associated with a web search query. Association is modeled using a query entity click graph, blending general query click logs with vertical query click logs. Smoothing techniques are proposed to address the inherent data sparsity in such graphs, including interpolation using a query synonymy model. A large-scale empirical analysis of the smoothing techniques, over a 2-year click graph collected from a commercial search engine, shows significant reductions in modeling error. The association models are then applied to the task of recommending products to web queries, by annotating queries with products from a large catalog and then mining query-product associations through web search session analysis. Experimental analysis shows that our smoothing techniques improve coverage while keeping precision stable, and overall, that our top-performing model affects 9% of general web queries with 94% precision."


#### ["Active Objects: Actions for Entity-Centric Search"](http://research.microsoft.com/apps/pubs/default.aspx?id=161389) Lin, Pantel, Gamon, Kannan, Fuxman
>	"We introduce an entity-centric search experience, called Active Objects, in which entity-bearing queries are paired with actions that can be performed on the entities. For example, given a query for a specific flashlight, we aim to present actions such as reading reviews, watching demo videos, and finding the best price online. In an annotation study conducted over a random sample of user query sessions, we found that a large proportion of queries in query logs involve actions on entities, calling for an automatic approach to identifying relevant actions for entity-bearing queries. In this paper, we pose the problem of finding actions that can be performed on entities as the problem of probabilistic inference in a graphical model that captures how an entity bearing query is generated. We design models of increasing complexity that capture latent factors such as entity type and intended actions that determine how a user writes a query in a search box, and the URL that they click on. Given a large collection of real-world queries and clicks from a commercial search engine, the models are learned efficiently through maximum likelihood estimation using an EM algorithm. Given a new query, probabilistic inference enables recommendation of a set of pertinent actions and hosts. We propose an evaluation methodology for measuring the relevance of our recommended actions, and show empirical evidence of the quality and the diversity of the discovered actions."

>	"Search as an action broker: A promising future search scenario involves modeling the user intents (or “verbs”) underlying the queries and brokering the webpages that accomplish the intended actions. In this vision, the broker is aware of all entities and actions of interest to its users, understands the intent of the user, ranks all providers of actions, and provides direct actionable results through APIs with the providers."
