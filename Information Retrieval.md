

  * [overview](#overview)
  * [interesting papers](#interesting-papers)
    - [ranking](#interesting-papers---ranking)
    - [document models](#interesting-papers---document-models)
    - [entity-centric search](#interesting-papers---entity-centric-search)



---
### overview

  [course](http://youtube.com/watch?v=5L1qemKyUKA&index=75&list=PL6397E4B26D00A269) by Chris Manning `video`  
  [course](http://homepages.inf.ed.ac.uk/vlavrenk/tts.html) by Victor Lavrenko
	([videos](http://youtube.com/user/victorlavrenko/playlists?view=1&sort=dd))  

  ["Foundations of Information Retrieval"](https://drive.google.com/file/d/0B-GJrccmbImkZ3pjNl9sczQxd3M) by Maarten de Rijke (SIGIR 2017) `slides`  
  ["Neural Text Embeddings for Information Retrieval"](https://microsoft.com/en-us/research/event/wsdm-2017-tutorial-neural-text-embeddings-information-retrieval/)
	tutorial by Bhaskar Mitra and Nick Craswell (WSDM 2017)
	([slides](https://slideshare.net/BhaskarMitra3/neural-text-embeddings-for-information-retrieval-wsdm-2017), [paper](https://arxiv.org/abs/1705.01509))  

  ["An Introduction to Information Retrieval"](http://nlp.stanford.edu/IR-book/pdf/irbookprint.pdf) book by Manning, Raghavan, Schutze  
  ["Search Engines. Information Retrieval in Practice"](http://ciir.cs.umass.edu/irbook/) book by Croft, Metzler, Strohman  

----

  [course](https://compscicenter.ru/courses/information-retrieval/2016-autumn/) by Yandex `video` `in russian`  
  [course](https://compsciclub.ru/courses/informationretrieval) by Yandex `video` `in russian`  

  course by Mail.ru
	([first part](https://youtube.com/playlist?list=PLrCZzMib1e9o_BlrSB5bFkLq8h2i4pQjz),
	[second part](https://youtube.com/playlist?list=PLrCZzMib1e9o7YIhOfJtD1EaneGOGkN-_)) `video` `in russian`  
  [course](http://habrahabr.ru/company/mailru/blog/257119/) by Mail.ru `video` `in russian`  

  [course](http://nzhiltsov.github.io/IR-course/) by Nikita Zhiltsov `in russian`

  introduction to ranking by Nikita Volkov
	([first part](https://youtube.com/watch?v=GctrEpJinhI),
	[second part](https://youtube.com/watch?v=GZmXKBzIfkA)) `video` `in russian`

---

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

  - [ranking](#interesting-papers---ranking)  
  - [document models](#interesting-papers---document-models)  
  - [entity-centric search](#interesting-papers---entity-centric-search)  

----

  - [question answering over texts](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-texts)  
  - [question answering over knowledge bases](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-knowledge-bases)  
  - [information extraction and integration](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---information-extraction-and-integration)  

----

[interesting recent papers](https://github.com/brylevkirill/notes/blob/master/interesting%20recent%20papers.md#reasoning)


----

#### ["Neural Models for Information Retrieval"](https://arxiv.org/abs/1705.01509) Mitra, Craswell
>	"Neural ranking models for information retrieval use shallow or deep neural networks to rank search results in response to a query. Traditional learning to rank models employ machine learning techniques over hand-crafted IR features. By contrast, neural models learn representations of language from raw text that can bridge the gap between query and document vocabulary. Unlike classical IR models, these new machine learning based approaches are data-hungry, requiring large scale training data before they can be deployed. This tutorial introduces basic concepts and intuitions behind neural IR models, and places them in the context of traditional retrieval models. We begin by introducing fundamental concepts of IR and different neural and non-neural approaches to learning vector representations of text. We then review shallow neural IR methods that employ pre-trained neural term embeddings without learning the IR task end-to-end. We introduce deep neural networks next, discussing popular deep architectures. Finally, we review the current DNN models for information retrieval. We conclude with a discussion on potential future directions for neural IR."

  - `slides` <https://slideshare.net/BhaskarMitra3/neural-text-embeddings-for-information-retrieval-wsdm-2017>


#### ["Neural Information Retrieval: A Literature Review"](https://arxiv.org/abs/1611.06792) Zhang et al.
>	"A recent "third wave" of Neural Network approaches now delivers state-of-the-art performance in many machine learning tasks, spanning speech recognition, computer vision, and natural language processing. Because these modern NNs often comprise multiple interconnected layers, this new NN research is often referred to as deep learning. Stemming from this tide of NN work, a number of researchers have recently begun to investigate NN approaches to Information Retrieval. While deep NNs have yet to achieve the same level of success in IR as seen in other areas, the recent surge of interest and work in NNs for IR suggest that this state of affairs may be quickly changing. In this work, we survey the current landscape of Neural IR research, paying special attention to the use of learned representations of queries and documents (i.e., neural embeddings). We highlight the successes of neural IR thus far, catalog obstacles to its wider adoption, and suggest potentially promising directions for future research."


#### ["End-to-End Goal-Driven Web Navigation"](http://arxiv.org/abs/1602.02261) Nogueira, Cho
>	"We propose a goal-driven web navigation as a benchmark task for evaluating an agent with abilities to understand natural language and plan on partially observed environments. In this challenging task, an agent navigates through a website, which is represented as a graph consisting of web pages as nodes and hyperlinks as directed edges, to find a web page in which a query appears. The agent is required to have sophisticated high-level reasoning based on natural languages and efficient sequential decision-making capability to succeed. We release a software tool, called WebNav, that automatically transforms a website into this goal-driven web navigation task, and as an example, we make WikiNav, a dataset constructed from the English Wikipedia. We extensively evaluate different variants of neural net based artificial agents on WikiNav and observe that the proposed goal-driven web navigation well reflects the advances in models, making it a suitable benchmark for evaluating future progress. Furthermore, we extend the WikiNav with questionanswer pairs from Jeopardy! and test the proposed agent based on recurrent neural networks against strong inverted index based search engines. The artificial agents trained on WikiNav outperforms the engined based approaches, demonstrating the capability of the proposed goal-driven navigation as a good proxy for measuring the progress in real-world tasks such as focused crawling and question-answering."

>	"In this work, we describe a large-scale goal-driven web navigation task and argue that it serves as a useful test bed for evaluating the capabilities of artificial agents on natural language understanding and planning. We release a software tool, called WebNav, that compiles a given website into a goal-driven web navigation task. As an example, we construct WikiNav from Wikipedia using WebNav. We extend WikiNav with Jeopardy! questions, thus creating WikiNav-Jeopardy. We evaluate various neural net based agents on WikiNav and WikiNav-Jeopardy. Our results show that more sophisticated agents have better performance, thus supporting our claim that this task is well suited to evaluate future progress in natural language understanding and planning. Furthermore, we show that our agent pretrained on WikiNav outperforms two strong inverted-index based search engines on the WikiNav-Jeopardy. These empirical results support our claim on the usefulness of the proposed task and agents in challenging applications such as focused crawling and question-answering."

  - `video` <https://youtu.be/tXBHfbHHlKc?t=31m20s> (Tamar)
  - `paper` ["Value Iteration Networks"](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#value-iteration-networks-tamar-wu-thomas-levine-abbeel) by Tamar et al.


#### ["Ask the Right Questions: Active Question Reformulation with Reinforcement Learning"](https://arxiv.org/abs/1705.07830) Buck et al.
>	"We frame Question Answering as a Reinforcement Learning task, an approach that we call Active Question Answering. We propose an agent that sits between the user and a black box question-answering system and which learns to reformulate questions to elicit the best possible answers. The agent probes the system with, potentially many, natural language reformulations of an initial question and aggregates the returned evidence to yield the best answer. The reformulation system is trained end-to-end to maximize answer quality using policy gradient. We evaluate on SearchQA, a dataset of complex questions extracted from Jeopardy!. Our agent improves F1 by 11% over a state-of-the-art base model that uses the original question/answer pairs."

  - `video` <https://youtu.be/soZXAH3leeQ?t=34m43s> (Cho)


#### ["Task-Oriented Query Reformulation with Reinforcement Learning"](https://arxiv.org/abs/1704.04572) Nogueira, Cho
>	"Search engines play an important role in our everyday lives by assisting us in finding the information we need. When we input a complex query, however, results are often far from satisfactory. In this work, we introduce a query reformulation system based on a neural network that rewrites a query to maximize the number of relevant documents returned. We train this neural network with reinforcement learning. The actions correspond to selecting terms to build a reformulated query, and the reward is the document recall. We evaluate our approach on three datasets against strong baselines and show a relative improvement of 5-20% in terms of recall. Furthermore, we present a simple method to estimate a conservative upper-bound performance of a model in a particular environment and verify that there is still large room for improvements."

  - `video` <https://youtu.be/soZXAH3leeQ?t=34m16s> (Cho)
  - `video` <https://ku.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=c933f3da-392f-4aeb-bd09-e766a8ba83aa> (5:03:10) (Nogueira)
  - `code` <https://github.com/nyu-dl/QueryReformulator>



---
### interesting papers - ranking


#### ["From RankNet to LambdaRank to LambdaMART: An Overview"](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) Burges
  `overview of ERR/pFound`
>	"LambdaMART is the boosted tree version of LambdaRank, which is based on RankNet. RankNet, LambdaRank, and LambdaMART have proven to be very successful algorithms for solving real world ranking problems: for example an ensemble of LambdaMART rankers won Track 1 of the 2010 Yahoo! Learning To Rank Challenge. The details of these algorithms are spread across several papers and reports, and so here we give a self-contained, detailed and complete description of them."

>	"Although here we will concentrate on ranking, it is straightforward to modify MART in general, and LambdaMART in particular, to solve a wide range of supervised learning problems (including maximizing information retrieval functions, like NDCG, which are not smooth functions of the model scores).


#### ["Learning to Rank using Gradient Descent"](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) Burges et al.
>	"We investigate using gradient descent methods for learning ranking functions; we propose a simple probabilistic cost function, and we introduce RankNet, an implementation of these ideas using a neural network to model the underlying ranking function. We present test results on toy data and on data from a commercial internet search engine."

>	"We have proposed a probabilistic cost for training systems to learn ranking functions using pairs of training examples. The approach can be used for any differentiable function; we explored using a neural network formulation, RankNet. RankNet is simple to train and gives excellent performance on a real world ranking problem with large amounts of data. Comparing the linear RankNet with other linear systems clearly demonstrates the benefit of using our pair-based cost function together with gradient descent; the two layer net gives further improvement. For future work it will be interesting to investigate extending the approach to using other machine learning methods for the ranking function; however evaluation speed and simplicity is a critical constraint for such systems."

  - `video` <http://videolectures.net/icml2015_burges_learning_to_rank/> (Burges)


#### ["Learning to Rank with Nonsmooth Cost Functions"](https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions) Burges et al.
>	"The quality measures used in information retrieval are particularly difficult to optimize directly, since they depend on the model scores only through the sorted order of the documents returned for a given query. Thus, the derivatives of the cost with respect to the model parameters are either zero, or are undefined. In this paper, we propose a class of simple, flexible algorithms, called LambdaRank, which avoids these difficulties by working with implicit cost functions. We describe LambdaRank using neural network models, although the idea applies to any differentiable function class. We give necessary and sufficient conditions for the resulting implicit cost function to be convex, and we show that the general method has a simple mechanical interpretation. We demonstrate significantly improved accuracy, over a state-of-the-art ranking algorithm, on several datasets. We also show that LambdaRank provides a method for significantly speeding up the training phase of that ranking algorithm. Although this paper is directed towards ranking, the proposed method can be extended to any non-smooth and multivariate cost functions."


#### ["Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks"](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf) Severyn, Moschitti
>	"Learning a similarity function between pairs of objects is at the core of learning to rank approaches. In information retrieval tasks we typically deal with query-document pairs, in question answering - question-answer pairs. However, before learning can take place, such pairs needs to be mapped from the original space of symbolic words into some feature space encoding various aspects of their relatedness, e.g. lexical, syntactic and semantic. Feature engineering is often a laborious task and may require external knowledge sources that are not always available or difficult to obtain. Recently, deep learning approaches have gained a lot of attention from the research community and industry for their ability to automatically learn optimal feature representation for a given task, while claiming state-of-the-art performance in many tasks in computer vision, speech recognition and natural language processing. In this paper, we present a convolutional neural network architecture for reranking pairs of short texts, where we learn the optimal representation of text pairs and a similarity function to relate them in a supervised way from the available training data. Our network takes only words in the input, thus requiring minimal preprocessing. In particular, we consider the task of reranking short text pairs where elements of the pair are sentences. We test our deep learning system on two popular retrieval tasks from TREC: Question Answering and Microblog Retrieval. Our model demonstrates strong performance on the first task beating previous state-of-the-art systems by about 3% absolute points in both MAP and MRR and shows comparable results on tweet reranking, while enjoying the benefits of no manual feature engineering and no additional syntactic parsers."

>	"In this paper, we propose a novel deep learning architecture for reranking short texts. It has the benefits of requiring no manual feature engineering or external resources, which may be expensive or not available. The model with the same architecture can be successfully applied to other domains and tasks. Our experimental findings show that our deep learning model: (i) greatly improves on the previous state-of-the-art systems and a recent deep learning approach in on answer sentence selection task showing a 3% absolute improvement in MAP and MRR; (ii) our system is able to improve even the best system runs from TREC Microblog 2012 challenge; (iii) is comparable to the syntactic reranker, while our system requires no external parsers or resources."

  - `code` <https://github.com/aseveryn/deep-qa>
  - `code` <https://github.com/shashankg7/Keras-CNN-QA>


#### ["Neural Ranking Models with Weak Supervision"](https://arxiv.org/abs/1704.08803) Dehghani, Zamani, Severyn, Kamps, Croft
>	"Despite the impressive improvements achieved by unsupervised deep neural networks in computer vision and NLP tasks, such improvements have not yet been observed in ranking for information retrieval. The reason may be the complexity of the ranking problem, as it is not obvious how to learn from queries and documents when no supervised signal is available. Hence, in this paper, we propose to train a neural ranking model using weak supervision, where labels are obtained automatically without human annotators or any external resources (e.g., click data). To this aim, we use the output of an unsupervised ranking model, such as BM25, as a weak supervision signal. We further train a set of simple yet effective ranking models based on feed-forward neural networks. We study their effectiveness under various learning scenarios (point-wise and pair-wise models) and using different input representations (i.e., from encoding query-document pairs into dense/sparse vectors to using word embedding representation). We train our networks using tens of millions of training instances and evaluate it on two standard collections: a homogeneous news collection(Robust) and a heterogeneous large-scale web collection (ClueWeb). Our experiments indicate that employing proper objective functions and letting the networks to learn the input representation based on weakly supervised data leads to impressive performance, with over 13% and 35% MAP improvements over the BM25 model on the Robust and the ClueWeb collections. Our findings also suggest that supervised neural ranking models can greatly benefit from pre-training on large amounts of weakly labeled data that can be easily obtained from unsupervised IR models."


#### ["A Neural Click Model for Web Search"](http://www2016.net/proceedings/proceedings/p531.pdf) Borisov, Markov, Rijke, Serdyukov
>	"Understanding user browsing behavior in web search is key to improving web search effectiveness. Many click models have been proposed to explain or predict user clicks on search engine results. They are based on the probabilistic graphical model (PGM) framework, in which user behavior is represented as a sequence of observable and hidden events. The PGM framework provides a mathematically solid way to reason about a set of events given some information about other events. But the structure of the dependencies between the events has to be set manually. Different click models use different hand-crafted sets of dependencies. We propose an alternative based on the idea of distributed representations: to represent the user’s information need and the information available to the user with a vector state. The components of the vector state are learned to represent concepts that are useful for modeling user behavior. And user behavior is modeled as a sequence of vector states associated with a query session: the vector state is initialized with a query, and then iteratively updated based on information about interactions with the search engine results. This approach allows us to directly understand user browsing behavior from click-through data, i.e., without the need for a predefined set of rules as is customary for PGM-based click models. We illustrate our approach using a set of neural click models. Our experimental results show that the neural click model that uses the same training data as traditional PGM-based click models, has better performance on the click prediction task (i.e., predicting user click on search engine results) and the relevance prediction task (i.e., ranking documents by their relevance to a query). An analysis of the best performing neural click model shows that it learns similar concepts to those used in traditional click models, and that it also learns other concepts that cannot be designed manually."


#### ["Gathering Additional Feedback on Search Results by Multi-Armed Bandits with Respect to Production Ranking"](http://www.www2015.it/documents/proceedings/proceedings/p1177.pdf) Vorobev, Lefortier, Gusev, Serdyukov
>	"Given a repeatedly issued query and a document with a not-yet-confirmed potential to satisfy the users’ needs, a search system should place this document on a high position in order to gather user feedback and obtain a more confident estimate of the document utility. On the other hand, the main objective of the search system is to maximize expected user satisfaction over a rather long period, what requires showing more relevant documents on average. The state-of-the-art approaches to solving this exploration-exploitation dilemma rely on strongly simplified settings making these approaches infeasible in practice. We improve the most flexible and pragmatic of them to handle some actual practical issues. The first one is utilizing prior information about queries and documents, the second is combining bandit-based learning approaches with a default production ranking algorithm. We show experimentally that our framework enables to significantly improve the ranking of a leading commercial search engine."


#### ["Online Learning to Rank in Stochastic Click Models"](https://arxiv.org/abs/1703.02527) Zoghi, Tunys, Ghavamzadeh, Kveton, Szepesvari, Wen
>	"Online learning to rank is a core problem in information retrieval and machine learning. Many provably efficient algorithms have been recently proposed for this problem in specific click models. The click model is a model of how the user interacts with a list of documents. Though these results are significant, their impact on practice is limited, because all proposed algorithms are designed for specific click models and lack convergence guarantees in other models. In this work, we propose BatchRank, the first online learning to rank algorithm for a broad class of click models. The class encompasses two most fundamental click models, the cascade and position-based models. We derive a gap-dependent upper bound on the T-step regret of BatchRank and evaluate it on a range of web search queries. We observe that BatchRank outperforms ranked bandits and is more robust than CascadeKL-UCB, an existing algorithm for the cascade model."



---
### interesting papers - document models

[question answering over texts](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-texts)


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


#### ["A Dual Embedding Space Model for Document Ranking"](https://arxiv.org/abs/1602.01137) Mitra, Nalisnick, Craswell, Caruana
  `DESM`
>	"A fundamental goal of search engines is to identify, given a query, documents that have relevant text. This is intrinsically difficult because the query and the document may use different vocabulary, or the document may contain query words without being relevant. We investigate neural word embeddings as a source of evidence in document ranking. We train a word2vec embedding model on a large unlabelled query corpus, but in contrast to how the model is commonly used, we retain both the input and the output projections, allowing us to leverage both the embedding spaces to derive richer distributional relationships. During ranking we map the query words into the input space and the document words into the output space, and compute a query-document relevance score by aggregating the cosine similarities across all the query-document word pairs."

>	"We postulate that the proposed Dual Embedding Space Model (DESM) captures evidence on whether a document is about a query term in addition to what is modelled by traditional term-frequency based approaches. Our experiments show that the DESM can re-rank top documents returned by a commercial Web search engine, like Bing, better than a term-matching based signal like TF-IDF. However, when ranking a larger set of candidate documents, we find the embeddings-based approach is prone to false positives, retrieving documents that are only loosely related to the query. We demonstrate that this problem can be solved effectively by ranking based on a linear mixture of the DESM and the word counting features."

  - `code` <https://github.com/bmitra-msft/Demos/blob/master/notebooks/DESM.ipynb>


#### ["Learning to Match Using Local and Distributed Representations of Text for Web Search"](https://arxiv.org/abs/1610.08136) Mitra, Diaz, Craswell
  `Duet`
>	"Models such as latent semantic analysis and those based on neural embeddings learn distributed representations of text, and match the query against the document in the latent semantic space. In traditional information retrieval models, on the other hand, terms have discrete or local representations, and the relevance of a document is determined by the exact matches of query terms in the body text. We hypothesize that matching with distributed representations complements matching with traditional local representations, and that a combination of the two is favorable. We propose a novel document ranking model composed of two separate deep neural networks, one that matches the query and the document using a local representation, and another that matches the query and the document using learned distributed representations. The two networks are jointly trained as part of a single neural network. We show that this combination or ‘duet’ performs significantly better than either neural network individually on a Web page ranking task, and also significantly outperforms traditional baselines and other recently proposed models based on neural networks."

  - `code` <https://github.com/faneshion/MatchZoo>
  - `code` <https://github.com/bmitra-msft/NDRM/blob/master/notebooks/Duet.ipynb>



---
### interesting papers - entity-centric search

[question answering over knowledge bases](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---question-answering-over-knowledge-bases)  
[information extraction and integration](https://github.com/brylevkirill/notes/blob/master/Knowledge%20Representation%20and%20Reasoning.md#interesting-papers---information-extraction-and-integration)  


#### ["Fast and Space-Efficient Entity Linking in Queries"](http://labs.yahoo.com/publication/fast-and-space-efficient-entity-linking-in-queries/) Blanco, Ottaviano, Meij
>	"Entity linking deals with identifying entities from a knowledge base in a given piece of text and has become a fundamental building block for web search engines, enabling numerous downstream improvements from better document ranking to enhanced search results pages. A key problem in the context of web search queries is that this process needs to run under severe time constraints as it has to be performed before any actual retrieval takes place, typically within milliseconds. In this paper we propose a probabilistic model that leverages user-generated information on the web to link queries to entities in a knowledge base. There are three key ingredients that make the algorithm fast and space-efficient. First, the linking process ignores any dependencies between the different entity candidates, which allows for a O(k^2) implementation in the number of query terms. Second, we leverage hashing and compression techniques to reduce the memory footprint. Finally, to equip the algorithm with contextual knowledge without sacrificing speed, we factor the distance between distributional semantics of the query words and entities into the model. We show that our solution significantly outperforms several state-of-the-art baselines by more than 14% while being able to process queries in sub-millisecond times—at least two orders of magnitude faster than existing systems."


#### ["Jigs and Lures: Associating Web Queries with Structured Entities"](http://www.aclweb.org/anthology/P11-1009) Pantel, Fuxman
>	"We propose methods for estimating the probability that an entity from an entity database is associated with a web search query. Association is modeled using a query entity click graph, blending general query click logs with vertical query click logs. Smoothing techniques are proposed to address the inherent data sparsity in such graphs, including interpolation using a query synonymy model. A large-scale empirical analysis of the smoothing techniques, over a 2-year click graph collected from a commercial search engine, shows significant reductions in modeling error. The association models are then applied to the task of recommending products to web queries, by annotating queries with products from a large catalog and then mining query-product associations through web search session analysis. Experimental analysis shows that our smoothing techniques improve coverage while keeping precision stable, and overall, that our top-performing model affects 9% of general web queries with 94% precision."


#### ["Active Objects: Actions for Entity-Centric Search"](http://research.microsoft.com/apps/pubs/default.aspx?id=161389) Lin, Pantel, Gamon, Kannan, Fuxman
>	"We introduce an entity-centric search experience, called Active Objects, in which entity-bearing queries are paired with actions that can be performed on the entities. For example, given a query for a specific flashlight, we aim to present actions such as reading reviews, watching demo videos, and finding the best price online. In an annotation study conducted over a random sample of user query sessions, we found that a large proportion of queries in query logs involve actions on entities, calling for an automatic approach to identifying relevant actions for entity-bearing queries. In this paper, we pose the problem of finding actions that can be performed on entities as the problem of probabilistic inference in a graphical model that captures how an entity bearing query is generated. We design models of increasing complexity that capture latent factors such as entity type and intended actions that determine how a user writes a query in a search box, and the URL that they click on. Given a large collection of real-world queries and clicks from a commercial search engine, the models are learned efficiently through maximum likelihood estimation using an EM algorithm. Given a new query, probabilistic inference enables recommendation of a set of pertinent actions and hosts. We propose an evaluation methodology for measuring the relevance of our recommended actions, and show empirical evidence of the quality and the diversity of the discovered actions."

>	"Search as an action broker: A promising future search scenario involves modeling the user intents (or “verbs”) underlying the queries and brokering the webpages that accomplish the intended actions. In this vision, the broker is aware of all entities and actions of interest to its users, understands the intent of the user, ranks all providers of actions, and provides direct actionable results through APIs with the providers."
