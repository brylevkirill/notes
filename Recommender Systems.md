

  * [**overview**](#overview)
  * [**interesting papers**](#interesting-papers)
    - [**item similarity**](#interesting-papers---item-similarity)
    - [**user preferences**](#interesting-papers---user-preferences)
    - [**deep learning**](#interesting-papers---deep-learning)
    - [**interactive learning**](#interesting-papers---interactive-learning)



---
### overview

  ["Model-Based Machine Learning: Making Recommendations"](http://mbmlbook.com/Recommender.html) by John Winn, Christopher Bishop and Thomas Diethe

  ["Recommender Systems: The Textbook"](http://charuaggarwal.net/Recommender-Systems.htm) by Charu Aggarwal `book`  
  ["Recommender Systems Handbook"](http://www.cs.ubbcluj.ro/~gabis/DocDiplome/SistemeDeRecomandare/Recommender_systems_handbook.pdf) by Ricci, Rokach, Shapira, Kantor `book`  

----

  [overview](https://youtube.com/watch?v=gCaOa3W9kM0) by Alex Smola `video`  
  [overview](https://youtube.com/watch?v=xMr7I-OypVY) by Alex Smola `video`  

  [tutorial](http://technocalifornia.blogspot.ru/2014/08/introduction-to-recommender-systems-4.html) by Xavier Amatriain `video`  
  ["The Recommender Problem Revisited"](http://videolectures.net/kdd2014_amatriain_mobasher_recommender_problem) by Xavier Amatriain `video`  
  ["Lessons Learned from Building Real-life Recommender Systems"](https://youtube.com/watch?v=VJOtr47V0eo) by Xavier Amatriain and Deepak Agarwal `video`  

  ["Deep Learning for Recommender Systems"](https://slideshare.net/kerveros99/deep-learning-for-recommender-systems-recsys2017-tutorial) by Alexandros Karatzoglou and Balazs Hidasi `slides`  
  ["Deep Learning for Recommender Systems"](https://youtube.com/watch?v=KZ7bcfYGuxw) by Alexandros Karatzoglou `video`  
  ["Deep Learning for Personalized Search and Recommender Systems"](https://youtube.com/watch?v=0DYQzZp68ok) by Zhang, Le, Fawaz, Venkataraman `video`  

  ["Recent Trends in Personalization: A Netflix Perspective"](https://slideslive.com/38917692/recent-trends-in-personalization-a-netflix-perspective) by Justin Basilico `video`

  ["Billion-scale Recommender System"](https://youtube.com/watch?v=-lNF8QRtEL4) by Ivan Lobov `video`

  [ACM RecSys](https://youtube.com/channel/UC2nEn-yNA1BtdDNWziphPGA) conference `video`

----

  course by Sergey Nikolenko ([part 1](https://youtube.com/watch?v=mr8u54jsveA), [part 2](https://youtube.com/watch?v=cD47Ssp_Flk), [part 3](https://youtube.com/watch?v=OFyb8ukrRDo)) `video` `in russian`  
  [course](https://lektorium.tv/node/33563) by Evgeny Sokolov and Andrey Zimovnov `video` `in russian`  

  [overview](https://youtube.com/watch?v=Us4KJkJiYrM) by Michael Rozner `video` `in russian`  
  [overview](https://youtube.com/watch?v=kfhqzkcfMqI) by Konstantin Vorontsov `video` `in russian`  
  [overview](https://youtube.com/watch?v=Te_6TqEhyTI) by Victor Kantor `video` `in russian`  
  [overview](https://youtube.com/watch?v=5ir_fCgzfLM) by Vladimir Gulin `video` `in russian`  
  [overview](https://youtube.com/watch?v=MLljnzsz9Dk) by Alexey Dral `video` `in russian`  

  [overview](https://youtube.com/watch?v=N0NUwz3xWX4) of deep learning for recommender systems by Dmitry Ushanov `video` `in russian`

  [overview](https://youtube.com/watch?v=iGAMPnv-0VY) of applications in Yandex by Igor Lifar and Dmitry Ushanov `video` `in russian`  
  [overview](https://youtube.com/watch?v=OJ0nJb3LfNo) of applications in Yandex by Andrey Zimovnov `video` `in russian`  
  [overview](https://youtube.com/watch?v=JKTneRi2vn8) of applications in Yandex by Michael Rozner `video` `in russian`  

----

  challenges:
  - diversity vs accuracy
  - personalization vs popularity
  - novelty vs relevance
  - contextual dimensions (time)
  - presentation bias
  - explaining vs selecting items
  - influencing user vs predicting future



---
### interesting papers

  - [**item similarity**](#interesting-papers---item-similarity)
  - [**user preferences**](#interesting-papers---user-preferences)
  - [**deep learning**](#interesting-papers---deep-learning)
  - [**active learning**](#interesting-papers---active-learning)


[**selected papers**](https://yadi.sk/d/RtAsSjLG3PhrT2)



----
#### ["On the Difficulty of Evaluating Baselines: A Study on Recommender Systems"](https://arxiv.org/abs/1905.01395) Rendle, Zhang, Koren
>	"Numerical evaluations with comparisons to baselines play a central role when judging research in recommender systems. In this paper, we show that running baselines properly is difficult. We demonstrate this issue on two extensively studied datasets. First, we show that results for baselines that have been used in numerous publications over the past five years for the Movielens 10M benchmark are suboptimal. With a careful setup of a vanilla matrix factorization baseline, we are not only able to improve upon the reported results for this baseline but even outperform the reported results of any newly proposed method. Secondly, we recap the tremendous effort that was required by the community to obtain high quality results for simple methods on the Netflix Prize. Our results indicate that empirical findings in research papers are questionable unless they were obtained on standardized benchmarks where baselines have been tuned extensively by the research community."



---
### interesting papers - item similarity


#### ["Two Decades of Recommender Systems at Amazon.com"](https://www.computer.org/csdl/mags/ic/2017/03/mic2017030012.html) Smith, Linden
  `Amazon`
>	learning item-to-item similarity on offline data (e.g. item2 often bought with item1)

>	"problem with algorithms based on computing correlations between users and items: did you watch a movie because you liked it or because we showed it to you or both? requires computing causal interventions instead of correlations: p(Y|X) -> p(Y|X,do(R))"  


#### ["Real-time Personalization using Embeddings for Search Ranking at Airbnb"](https://kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb) Grbovic, Cheng
  `Airbnb` `KDD 2018`
>	"Search Ranking and Recommendations are fundamental problems of crucial interest to major Internet companies, including web search engines, content publishing websites and marketplaces. However, despite sharing some common characteristics a one-size-fits-all solution does not exist in this space. Given a large difference in content that needs to be ranked, personalized and recommended, each marketplace has a somewhat unique challenge. Correspondingly, at Airbnb, a short-term rental marketplace, search and recommendation problems are quite unique, being a two-sided marketplace in which one needs to optimize for host and guest preferences, in a world where a user rarely consumes the same item twice and one listing can accept only one guest for a certain set of dates. In this paper we describe Listing and User Embedding techniques we developed and deployed for purposes of Real-time Personalization in Search Ranking and Similar Listing Recommendations, two channels that drive 99% of conversions. The embedding models were specifically tailored for Airbnb marketplace, and are able to capture guest’s short-term and long-term interests, delivering effective home listing recommendations. We conducted rigorous offline testing of the embedding models, followed by successful online tests before fully deploying them into production."

  - `post` <https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e>
  - `video` <https://youtube.com/watch?v=aWjsUEX7B1I>
  - `video` <http://videolectures.net/kdd2018_grbovic_search_ranking_at_airbnb> (Grbovic)
  - `video` <https://infoq.com/presentations/nlp-word-embedding> (31:33) (Alammar)
  - `slides` <https://astro.temple.edu/~tua95067/Mihajlo_KDD2018_slides.pptx> (Grbovic)
  - `post` <https://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising>


#### ["Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba"](https://arxiv.org/abs/1803.02349) Wang et al.
  `Alibaba` `KDD 2018`
>	"Recommender systems have been the most important technology for increasing the business in Taobao, the largest online consumer-to-consumer platform in China. There are three major challenges facing RS in Taobao: scalability, sparsity and cold start. In this paper, we present our technical solutions to address these three challenges. The methods are based on a well- known graph embedding framework. We first construct an item graph from users’ behavior history, and learn the embeddings of all items in the graph. The item embeddings are employed to compute pairwise similarities between all items, which are then used in the recommendation process. To alleviate the sparsity and cold start problems, side information is incorporated into the graph embedding framework. We propose two aggregation methods to integrate the embeddings of items and the corresponding side information. Experimental results from offline experiments show that methods incorporating side information are superior to those that do not. Further, we describe the platform upon which the embedding methods are deployed and the workflow to process the billion-scale data in Taobao. Using A/B test, we show that the online Click-Through-Rates are improved comparing to the previous collaborative filtering based methods widely used in Taobao, further demonstrating the effectiveness and feasibility of our proposed methods in Taobao’s live production environment."

  - `video` <https://youtube.com/watch?v=TLD_bSiHZdE>
  - `video` <https://infoq.com/presentations/nlp-word-embedding> (36:56) (Alammar)


#### ["Exponential Family Embeddings"](https://arxiv.org/abs/1608.00778) Rudolph, Ruiz, Mandt, Blei
>	"Word embeddings are a powerful approach for capturing semantic similarity among terms in a vocabulary. In this paper, we develop exponential family embeddings, a class of methods that extends the idea of word embeddings to other types of high-dimensional data. As examples, we studied neural data with real-valued observations, count data from a market basket analysis, and ratings data from a movie recommendation system. The main idea is to model each observation conditioned on a set of other observations. This set is called the context, and the way the context is defined is a modeling choice that depends on the problem. In language the context is the surrounding words; in neuroscience the context is close-by neurons; in market basket data the context is other items in the shopping cart. Each type of embedding model defines the context, the exponential family of conditional distributions, and how the latent embedding vectors are shared across data. We infer the embeddings with a scalable algorithm based on stochastic gradient descent. On all three applications - neural activity of zebrafish, users' shopping behavior, and movie ratings - we found exponential family embedding models to be more effective than other types of dimension reduction. They better reconstruct held-out data and find interesting qualitative structure."

>	identifying substitutes and co-purchases in high-scale consumer data (basket analysis)

  - `video` <https://youtu.be/zwcjJQoK8_Q?t=15m14s> (Blei)
  - `code` <https://github.com/mariru/exponential_family_embeddings>
  - `code` <https://github.com/franrruiz/p-emb>


#### ["E-commerce in Your Inbox: Product Recommendations at Scale"](https://arxiv.org/abs/1606.07154) Grbovic et al.
  `prod2vec` `Yahoo`
>	"In recent years online advertising has become increasingly ubiquitous and effective. Advertisements shown to visitors fund sites and apps that publish digital content, manage social networks, and operate e-mail services. Given such large variety of internet resources, determining an appropriate type of advertising for a given platform has become critical to financial success. Native advertisements, namely ads that are similar in look and feel to content, have had great success in news and social feeds. However, to date there has not been a winning formula for ads in e-mail clients. In this paper we describe a system that leverages user purchase history determined from e-mail receipts to deliver highly personalized product ads to Yahoo Mail users. We propose to use a novel neural language-based algorithm specifically tailored for delivering effective product recommendations, which was evaluated against baselines that included showing popular products and products predicted based on co-occurrence. We conducted rigorous offline testing using a large-scale product purchase data set, covering purchases of more than 29 million users from 172 e-commerce websites. Ads in the form of product recommendations were successfully tested on online traffic, where we observed a steady 9% lift in click-through rates over other ad formats in mail, as well as comparable lift in conversion rates. Following successful tests, the system was launched into production during the holiday season of 2014."

  - `video` <https://youtube.com/watch?v=W56fZewflRw> (Djuric)



---
### interesting papers - user preferences


#### ["Recurrent Recommender Networks"](https://dl.acm.org/citation.cfm?id=3018689) Wu, Ahmed, Beutel, Smola, Jing
>	"Recommender systems traditionally assume that user profiles and movie attributes are static. Temporal dynamics are purely reactive, that is, they are inferred after they are observed, e.g. after a user’s taste has changed or based on hand-engineered temporal bias corrections for movies. We propose Recurrent Recommender Networks that are able to predict future behavioral trajectories. This is achieved by endowing both users and movies with a Long Short-Term Memory autoregressive model that captures dynamics, in addition to a more traditional low-rank factorization. On multiple real-world datasets, our model offers excellent prediction accuracy and it is very compact, since we need not learn latent state but rather just the state transition function."

>	"A common approach to practical recommender systems is to study problems of the form introduced in the Netflix contest. That is, given a set of tuples consisting of users, movies, timestamps and ratings, the goal is to find ratings for alternative combinations of the first three attributes (user, movie, time). Performance is then measured by the deviation of the prediction from the actual rating. This formulation is easy to understand and it has led to numerous highly successful approaches, such as Probabilistic Matrix Factorization, nearest neighbor based approaches, and clustering. Moreover, it is easy to define appropriate performance measures (deviation between rating estimates and true ratings over the matrix), simply by selecting a random subset of the tuples for training and the rest for testing purposes. Unfortunately, these approaches are lacking when it comes to temporal and causal aspects inherent in the data. The following examples illustrate this in some more detail:  
>	Change in Movie Perception. Plan 9 from Outer Space has achieved cult movie status by being arguably one of the world’s worst movies. As a result of the social notion of being a movie that is so bad that it is great to watch, the perception changed over time from a truly awful movie to a popular one. To capture this appropriately, the movie attribute parameters would have to change over time to track such a trend. While maybe not quite as pronounced, similar effects hold for movie awards such as the Oscars. After all, it is much more challenging to hold a contrarian view about a critically acclaimed movie than about, say, Star Wars 1, The Phantom Menace.  
>	Seasonal Changes. While not quite so extreme, the relative appreciation of romantic comedies, Christmas movies and summer blockbusters is seasonal. Beyond the appreciation, users are unlikely to watch movies about overweight bearded old men wearing red robes in summer.  
>	User Interest. User’s preferences change over time. This is well established in online communities and it arguably also applies to online consumption. A user might take a liking to a particular actor, might discover the intricacies of a specific genre, or her interest in a particular show might wane, due to maturity or a change in lifestyle. Any such aspects render existing profiles moot, yet it is difficult to model all such changes explicitly."

>	"Beyond the mere need of modeling temporal evolution, evaluating ratings with the benefit of hindsight also violates basic requirements of causality. For instance, knowing that a user will have developed a liking for Pedro Almod ́ovar in one month in the future makes it much easier to estimate what his opinion about La Mala Educacion might be. In other words, we violate causality in our statistical analysis when we use future ratings for the benefit of estimating current reviews. It also makes it impossible to translate reported accuracies on benchmarks into meaningful assessments as to whether such a system would work well in practice. While the Netflix prize generated a flurry of research, evaluating different models’ success on future predictions is hindered by the mixed distribution of training and testing data. Rather, by having an explicit model of profile dynamics, we can predict future behavior based on current trends. A model capable of capturing the actual data distribution inherent in recommender systems needs to be able to model both the temporal dynamics within each user and movie, in addition to capturing the rating interaction between both sets. This suggests the use of latent variable models to infer the unobserved state governing their behavior."

>	"Nonlinear nonparametric recommender systems have proven to be somewhat elusive. In particular, nonlinear substitutes of the inner product formulation showed only limited promise in our experiments. To the best of our knowledge this is the first paper addressing movie recommendation in a fully causal and integrated fashion. That is, we believe that this is the first model which attempts to capture the dynamics of both users and movies. Moreover, our model is nonparametric. This allows us to model the data rather than having to assume a specific form of a state space."

>	"Recurrent Recommender Networks are very concise since we only learn the dynamics rather than the state. This is one of the key differences to typical latent variable models where considerable effort is spent on estimating the latent state."

>	"Experiments show that our model outperforms all others in terms of forward prediction, i.e. in the realistic scenario where we attempt to estimate future ratings given data that occurred strictly prior to the to-be-predicted ratings. We show that our model is able to capture exogenous dynamics (e.g. an Oscar award) and endogenous dynamics (e.g. Christmas movies) quite accurately. Moreover, we demonstrate that the model is able to predict changes in future user preferences accurately."


#### ["Latent LSTM Allocation: Joint Clustering and Non-Linear Dynamic Modeling of Sequential Data"](http://proceedings.mlr.press/v70/zaheer17a/zaheer17a.pdf) Zaheer, Ahmed, Smola
  `Google`
>	"Recurrent neural networks, such as LSTM networks, are powerful tools for modeling sequential data like user browsing history or natural language text. However, to generalize across different user types, LSTMs require a large number of parameters, notwithstanding the simplicity of the underlying dynamics, rendering it uninterpretable, which is highly undesirable in user modeling. The increase in complexity and parameters arises due to a large action space in which many of the actions have similar intent or topic. In this paper, we introduce Latent LSTM Allocation for user modeling combining hierarchical Bayesian models with LSTMs. In LLA, each user is modeled as a sequence of actions, and the model jointly groups actions into topics and learns the temporal dynamics over the topic sequence, instead of action space directly. This leads to a model that is highly interpretable, concise, and can capture intricate dynamics. We present an efficient Stochastic EM inference algorithm for our model that scales to millions of users/documents. Our experimental evaluations show that the proposed model compares favorably with several state-of-the-art baselines."

  - `video` <https://vimeo.com/240608072> (Zaheer)
  - `video` <https://youtube.com/watch?v=ofaPq5aRKZ0> (Smola)
  - `paper` ["State Space LSTM Models with Particle MCMC Inference"](https://arxiv.org/abs/1711.11179) by Zheng et al.


#### ["State Space LSTM Models with Particle MCMC Inference"](https://arxiv.org/abs/1711.11179) Zheng, Zaheer, Ahmed, Wang, Xing, Smola
>	"LSTM is one of the most powerful sequence models. Despite the strong performance, however, it lacks the nice interpretability as in state space models. In this paper, we present a way to combine the best of both worlds by introducing State Space LSTM models that generalizes the earlier work of combining topic models with LSTM. However we do not make any factorization assumptions in our inference algorithm. We present an efficient sampler based on sequential Monte Carlo method that draws from the joint posterior directly. Experimental results confirms the superiority and stability of this SMC inference algorithm on a variety of domains."


#### ["Session-based Recommendations with Recurrent Neural Networks"](http://arxiv.org/abs/1511.06939) Hidasi, Karatzoglou, Baltrunas, Tikk
>	"We apply recurrent neural networks on a new domain, namely recommendation system. Real-life recommender systems often face the problem of having to base recommendations only on short session-based data (e.g. a small sportsware website) instead of long user histories (as in the case of Netflix). In this situation the frequently praised matrix factorization approaches are not accurate. This problem is usually overcome in practice by resorting to item-to-item recommendations, i.e. recommending similar items. We argue that by modeling the whole session, more accurate recommendations can be provided. We therefore propose an RNN-based approach for session-based recommendations. Our approach also considers practical aspects of the task and introduces several modifications to classic RNNs such as a ranking loss function that make it more viable for this specific problem. Experimental results on two data-sets show marked improvements over widely used approaches."

>	"In this paper we applied a kind of modern recurrent neural network to new application domain: recommender systems. We chose task of session based recommendations, because it is a practically important area, but not well researched. We modified the basic GRU in order to fit the task better by introducing session-parallel mini-batches, mini-batch based output sampling and ranking loss function. We showed that our method can significantly outperform popular baselines that used for this task. We think that our work can be the basis of both deep learning applications in recommender systems and session based recommendations in general. We plan to train the network on automatically extracted item representation that is built on content of the item itself (e.g. thumbnail, video, text) instead of the current input."

  - `video` <https://youtube.com/watch?v=M7FqgXySKYk> (Karatzoglou)
  - `video` <https://youtube.com/watch?v=Mw2AV12WH4s> (Hidasi)
  - `post` <http://blog.deepsystems.io/session-based-recommendations-rnn> `in russian`
  - `code` <https://github.com/yhs-968/pyGRU4REC>


#### ["DropoutNet: Addressing Cold Start in Recommender Systems"](https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems) Volkovs, Yu, Poutanen
  `Layer6`
>	"Latent models have become the default choice for recommender systems due to their performance and scalability. However, research in this area has primarily focused on modeling user-item interactions, and few latent models have been developed for cold start. Deep learning has recently achieved remarkable success showing excellent results for diverse input types. Inspired by these results we propose a neural network based latent model called DropoutNet to address the cold start problem in recommender systems. Unlike existing approaches that incorporate additional content-based objective terms, we instead focus on the optimization and show that neural network models can be explicitly trained for cold start through dropout."

>	"Our approach is based on the observation that cold start is equivalent to the missing data problem where preference information is missing. Hence, instead of adding additional objective terms to model content, we modify the learning procedure to explicitly condition the model for the missing input. The key idea behind our approach is that by applying dropout to input mini-batches, we can train DNNs to generalize to missing input. By selecting an appropriate amount of dropout we show that it is possible to learn a DNN-based latent model that performs comparably to state-of-the-art on warm start while significantly outperforming it on cold start. The resulting model is simpler than most hybrid approaches and uses a single objective function, jointly optimizing all components to maximize recommendation accuracy."

>	"Training with dropout has a two-fold effect: pairs with dropout encourage the model to only use content information, while pairs without dropout encourage it to ignore content and simply reproduce preference input. The net effect is balanced between these two extremes. The model learns to reproduce the accuracy of the input latent model when preference data is available while also generalizing to cold start."

>	"An additional advantage of our approach is that it can be applied on top of any existing latent model to provide/enhance its cold start capability. This requires virtually no modification to the original model thus minimizing the implementation barrier for any production environment that’s already running latent models."

  - `video` <https://youtu.be/N0NUwz3xWX4?t=10m44s> (Ushanov) `in russian`
  - `code` <https://github.com/HongleiXie/DropoutNet>


#### ["Variational Autoencoders for Collaborative Filtering"](https://arxiv.org/abs/1802.05814) Liang, Krishnan, Hoffman, Jebara
  `VAE-CF` `Netflix`
>	"We extend variational autoencoders to collaborative filtering for implicit feedback. This non-linear probabilistic model enables us to go beyond the limited modeling capacity of linear factor models which still largely dominate collaborative filtering research. We introduce a generative model with multinomial likelihood and use Bayesian inference for parameter estimation. Despite widespread use in language modeling and economics, the multinomial likelihood receives less attention in the recommender systems literature. We introduce a different regularization parameter for the learning objective, which proves to be crucial for achieving competitive performance. Remarkably, there is an efficient way to tune the parameter using annealing. The resulting model and learning algorithm has information-theoretic connections to maximum entropy discrimination and the information bottleneck principle. Empirically, we show that the proposed approach significantly outperforms several state-of-the-art baselines, including two recently-proposed neural network approaches, on several real-world datasets. We also provide extended experiments comparing the multinomial likelihood with other commonly used likelihood functions in the latent factor collaborative filtering literature and show favorable results. Finally, we identify the pros and cons of employing a principled Bayesian inference approach and characterize settings where it provides the most significant improvements."

>	"Recommender systems is more of a "small data" than a "big data" problem."  
>	"VAE generalizes linear latent factor model and recovers Gaussian matrix factorization as a special linear case. No iterative procedure required to rank all the items given a user's watch history - only need to evaluate inference and generative functions."  
>	"We introduce a regularization parameter for the learning objective to trade-off the generative power for better predictive recommendation performance. For recommender systems, we don't necessarily need all the statistical property of a generative model. We trade off the ability of performing ancestral sampling for better fitting the data."  

  - `video` <https://youtube.com/watch?v=gRvxr47Gj3k> (Liang)
  - `code` <https://github.com/dawenl/vae_cf>
  - `code` <https://github.com/belepi93/vae-cf-pytorch>


#### ["Content-based Recommendations with Poisson Factorization"](http://www.cs.toronto.edu/~lcharlin/papers/GopalanCharlinBlei_nips14.pdf) Gopalan, Charlin, Blei
  `CTPF`
>	"We develop collaborative topic Poisson factorization (CTPF), a generative model of articles and reader preferences. CTPF can be used to build recommender systems by learning from reader histories and content to recommend personalized articles of interest. In detail, CTPF models both reader behavior and article texts with Poisson distributions, connecting the latent topics that represent the texts with the latent preferences that represent the readers. This provides better recommendations than competing methods and gives an interpretable latent space for understanding patterns of readership. Further, we exploit stochastic variational inference to model massive real-world datasets. For example, we can fit CTPF to the full arXiv usage dataset, which contains over 43 million ratings and 42 million word counts, within a day. We demonstrate empirically that our model outperforms several baselines, including the previous state-of-the art approach."

>	collaborative topic models:  
>	- blending factorization-based and content-based recommendation  
>	- describing user preferences with interpretable topics  

  - `video` <http://www.fields.utoronto.ca/video-archive/2017/03/2267-16706> (26:36) (Blei)
  - `code` <https://github.com/premgopalan/collabtm>


#### ["Scalable Recommendation with Hierarchical Poisson Factorization"](http://auai.org/uai2015/proceedings/papers/208.pdf) Gopalan, Hofman, Blei
  `HPF`
>	"We develop hierarchical Poisson matrix factorization, a novel method for providing users with high quality recommendations based on implicit feedback, such as views, clicks, or purchases. In contrast to existing recommendation models, HPF has a number of desirable properties. First, we show that HPF more accurately captures the long-tailed user activity found in most consumption data by explicitly considering the fact that users have finite attention budgets. This leads to better estimates of users’ latent preferences, and therefore superior recommendations, compared to competing methods. Second, HPF learns these latent factors by only explicitly considering positive examples, eliminating the often costly step of generating artificial negative examples when fitting to implicit data. Third, HPF is more than just one method- it is the simplest in a class of probabilistic models with these properties, and can easily be extended to include more complex structure and assumptions. We develop a variational algorithm for approximate posterior inference for HPF that scales up to large data sets, and we demonstrate its performance on a wide variety of real-world recommendation problems - users rating movies, listening to songs, reading scientific papers, and reading news articles."

>	discovering correlated preferences (devising new utility models and other factors such as time of day, date, in stock, customer demographic information)

  - `video` <https://youtu.be/zwcjJQoK8_Q?t=41m49s> (Blei)
  - `code` <https://github.com/premgopalan/hgaprec>


#### ["E-commerce in Your Inbox: Product Recommendations at Scale"](https://arxiv.org/abs/1606.07154) Grbovic et al.
  `user2vec` `Yahoo`
>	"In recent years online advertising has become increasingly ubiquitous and effective. Advertisements shown to visitors fund sites and apps that publish digital content, manage social networks, and operate e-mail services. Given such large variety of internet resources, determining an appropriate type of advertising for a given platform has become critical to financial success. Native advertisements, namely ads that are similar in look and feel to content, have had great success in news and social feeds. However, to date there has not been a winning formula for ads in e-mail clients. In this paper we describe a system that leverages user purchase history determined from e-mail receipts to deliver highly personalized product ads to Yahoo Mail users. We propose to use a novel neural language-based algorithm specifically tailored for delivering effective product recommendations, which was evaluated against baselines that included showing popular products and products predicted based on co-occurrence. We conducted rigorous offline testing using a large-scale product purchase data set, covering purchases of more than 29 million users from 172 e-commerce websites. Ads in the form of product recommendations were successfully tested on online traffic, where we observed a steady 9% lift in click-through rates over other ad formats in mail, as well as comparable lift in conversion rates. Following successful tests, the system was launched into production during the holiday season of 2014."

  - `video` <https://youtube.com/watch?v=W56fZewflRw> (Djuric)


#### ["Metadata Embeddings for User and Item Cold-start Recommendations"](https://arxiv.org/abs/1507.08439) Kula
  `LightFM`
>	"I present a hybrid matrix factorisation model representing users and items as linear combinations of their content features' latent factors. The model outperforms both collaborative and content-based models in cold-start or sparse interaction data scenarios (using both user and item metadata), and performs at least as well as a pure collaborative matrix factorisation model where interaction data is abundant. Additionally, feature embeddings produced by the model encode semantic information in a way reminiscent of word embedding approaches, making them useful for a range of related tasks such as tag recommendations."

  - `video` <https://youtube.com/watch?v=EgE0DUrYmo8> (Kula)
  - `code` <https://github.com/lyst/lightfm>


#### ["Causal Inference for Recommendation"](http://people.hss.caltech.edu/~fde/UAI2016WS/papers/Liang.pdf) Liang, Charlin, Blei
>	"We develop a causal inference approach to recommender systems. Observational recommendation data contains two sources of information: which items each user decided to look at and which of those items each user liked. We assume these two types of information come from different models - the exposure data comes from a model by which users discover items to consider; the click data comes from a model by which users decide which items they like. Traditionally, recommender systems use the click data alone (or ratings data) to infer the user preferences. But this inference is biased by the exposure data, i.e., that users do not consider each item independently at random. We use causal inference to correct for this bias. On real-world data, we demonstrate that causal inference for recommender systems leads to improved generalization to new data."

  - `slides` <http://people.hss.caltech.edu/~fde/UAI2016WS/talks/Dawen.pdf> (Liang)
  - `slides` <http://www.homepages.ucl.ac.uk/~ucgtrbd/whatif/David.pdf> (Blei)



---
### interesting papers - deep learning


#### ["Deep Learning based Recommender System: A Survey and New Perspectives"](https://arxiv.org/abs/1707.07435) Zhang et al.
>	"With the ever-growing volume of online information, recommender systems have been an effective strategy to overcome such information overload. The utility of recommender systems cannot be overstated, given its widespread adoption in many web applications, along with its potential impact to ameliorate many problems related to over-choice. In recent years, deep learning has garnered considerable interest in many research fields such as computer vision and natural language processing, owing not only to stellar performance but also the attractive property of learning feature representations from scratch. The influence of deep learning is also pervasive, recently demonstrating its effectiveness when applied to information retrieval and recommender systems research. Evidently, the field of deep learning in recommender system is flourishing. This article aims to provide a comprehensive review of recent research efforts on deep learning based recommender systems. More concretely, we provide and devise a taxonomy of deep learning based recommendation models, along with providing a comprehensive summary of the state-of-the-art. Finally, we expand on current trends and provide new perspectives pertaining to this new exciting development of the field."


#### ["Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches"](https://arxiv.org/abs/1907.06902) Dacrema, Cremonesi, Jannach
>	"Deep learning techniques have become the method of choice for researchers working on algorithmic aspects of recommender systems. With the strongly increased interest in machine learning in general, it has, as a result, become difficult to keep track of what represents the state-of-the-art at the moment, e.g., for top-n recommendation tasks. At the same time, several recent publications point out problems in today's research practice in applied machine learning, e.g., in terms of the reproducibility of the results or the choice of the baselines when proposing new models. In this work, we report the results of a systematic analysis of algorithmic proposals for top-n recommendation tasks. Specifically, we considered 18 algorithms that were presented at top-level research conferences in the last years. Only 7 of them could be reproduced with reasonable effort. For these methods, it however turned out that 6 of them can often be outperformed with comparably simple heuristic methods, e.g., based on nearest-neighbor or graph-based techniques. The remaining one clearly outperformed the baselines but did not consistently outperform a well-tuned non-neural linear ranking method. Overall, our work sheds light on a number of potential problems in today's machine learning scholarship and calls for improved scientific practices in this area."


#### ["Deep Learning Recommendation Model for Personalization and Recommendation Systems"](https://arxiv.org/abs/1906.00091) Naumov et al.
  `DLRM` `Facebook`
>	"With the advent of deep learning, neural network-based recommendation models have emerged as an important tool for tackling personalization and recommendation tasks. These networks differ significantly from other deep learning networks due to their need to handle categorical features and are not well studied or understood. In this paper, we develop a state-of-the-art deep learning recommendation model and provide its implementation in both PyTorch and Caffe2 frameworks. In addition, we design a specialized parallelization scheme utilizing model parallelism on the embedding tables to mitigate memory constraints while exploiting data parallelism to scale-out compute from the fully-connected layers. We compare DLRM against existing recommendation models and characterize its performance on the Big Basin AI platform, demonstrating its usefulness as a benchmark for future algorithmic experimentation and system co-design."

  - `post` <https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model>
  - `paper` ["The Architectural Implications of Facebook's DNN-based Personalized Recommendation"](https://arxiv.org/abs/1906.03109) by Gupta et al.


#### ["Wide & Deep Learning"](https://arxiv.org/abs/1606.07792) Cheng et al.
  `Google`
>	"Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning---jointly trained wide linear models and deep neural networks---to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow."

  - `post` <https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html>
  - `video` <https://youtube.com/watch?v=NV1tkZ9Lq48> (Cheng)
  - `post` <https://www.tensorflow.org/tutorials/wide_and_deep>
  - `code` <https://github.com/tensorflow/models/blob/master/official/wide_deep/wide_deep.py>


#### ["Deep Neural Networks for YouTube Recommendations"](http://research.google.com/pubs/pub45530.html) Covington, Adams, Sargin
  `YouTube`
>	"YouTube represents one of the largest scale and most sophisticated industrial recommendation systems in existence. In this paper, we describe the system at a high level and focus on the dramatic performance improvements brought by deep learning. The paper is split according to the classic two-stage information retrieval dichotomy: first, we detail a deep candidate generation model and then describe a separate deep ranking model. We also provide practical lessons and insights derived from designing, iterating and maintaining a massive recommendation system with enormous userfacing impact."

>	"We have described our deep neural network architecture for recommending YouTube videos, split into two distinct problems: candidate generation and ranking. Our deep collaborative filtering model is able to effectively assimilate many signals and model their interaction with layers of depth, outperforming previous matrix factorization approaches used at YouTube. There is more art than science in selecting the surrogate problem for recommendations and we found classifying a future watch to perform well on live metrics by capturing asymmetric co-watch behavior and preventing leakage of future information. Withholding discrimative signals from the classifier was also essential to achieving good results - otherwise the model would overfit the surrogate problem and not transfer well to the homepage. We demonstrated that using the age of the training example as an input feature removes an inherent bias towards the past and allows the model to represent the time-dependent behavior of popular of videos. This improved offline holdout precision results and increased the watch time dramatically on recently uploaded videos in A/B testing. Ranking is a more classical machine learning problem yet our deep learning approach outperformed previous linear and tree-based methods for watch time prediction. Recommendation systems in particular benefit from specialized features describing past user behavior with items. Deep neural networks require special representations of categorical and continuous features which we transform with embeddings and quantile normalization, respectively. Layers of depth were shown to effectively model non-linear interactions between hundreds of features. Logistic regression was modified by weighting training examples with watch time for positive examples and unity for negative examples, allowing us to learn odds that closely model expected watch time. This approach performed much better on watch-time weighted ranking evaluation metrics compared to predicting click-through rate directly."

  - `video` <https://youtube.com/watch?v=WK_Nr4tUtl8> (Covington)
  - `video` <https://youtube.com/watch?v=ZrgNalPlvSs> (Nada)
  - `notes` <https://blog.acolyer.org/2016/09/19/deep-neural-networks-for-youtube-recommendations>
  - `notes` <https://ekababisong.org/deep-neural-networks-youtube>
  - `code` <https://github.com/ogerhsou/Youtube-Recommendation-Tensorflow/blob/master/youtube_recommendation.py>


#### ["Latent Cross: Making Use of Context in Recurrent Recommender Systems"](https://dl.acm.org/citation.cfm?id=3159727) Beutel et al.
  `YouTube`
>	"The success of recommender systems often depends on their ability to understand and make use of the context of the recommendation request. Significant research has focused on how time, location, interfaces, and a plethora of other contextual features affect recommendations. However, in using deep neural networks for recommender systems, researchers often ignore these contexts or incorporate them as ordinary features in the model. In this paper, we study how to effectively treat contextual data in neural recommender systems. We begin with an empirical analysis of the conventional approach to context as features in feed-forward recommenders and demonstrate that this approach is inefficient in capturing common feature crosses. We apply this insight to design a state-of-the-art RNN recommender system. We first describe our RNN-based recommender system in use at YouTube. Next, we offer “Latent Cross,” an easy-to-use technique to incorporate contextual data in the RNN by embedding the context feature first and then performing an element-wise product of the context embedding with model’s hidden states. We demonstrate the improvement in performance by using this Latent Cross technique in multiple experimental settings."



---
### interesting papers - interactive learning


#### ["Making Contextual Decisions with Low Technical Debt"](http://arxiv.org/abs/1606.03966) Agarwal et al.
  `Microsoft Custom Decision Service`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#making-contextual-decisions-with-low-technical-debt-agarwal-et-al>


#### ["Top-K Off-Policy Correction for a REINFORCE Recommender System"](https://arxiv.org/abs/1812.02353) Chen, Beutel, Covington, Jain, Belletti, Chi
  `YouTube`
  - <https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#top-k-off-policy-correction-for-a-reinforce-recommender-system-chen-beutel-covington-jain-belletti-chi>


#### ["Q&R: A Two-Stage Approach toward Interactive Recommendation"](http://alexbeutel.com/papers/q-and-r-kdd2018.pdf) Christakopoulou, Beutel, Li, Jain, Chi
  `YouTube`
>	"Recommendation systems, prevalent in many applications, aim to surface to users the right content at the right time. Recently, researchers have aspired to develop conversational systems that offer seamless interactions with users, more effectively eliciting user preferences and offering better recommendations. Taking a step towards this goal, this paper explores the two stages of a single round of conversation with a user: which question to ask the user, and how to use their feedback to respond with a more accurate recommendation. Following these two stages, first, we detail an RNN-based model for generating topics a user might be interested in, and then extend a state-of-the-art RNN-based video recommender to incorporate the user’s selected topic. We describe our proposed system Q&R, i.e., Question & Recommendation, and the surrogate tasks we utilize to bootstrap data for training our models. We evaluate different components of Q&R on live traffic in various applications within YouTube: User Onboarding, Home-page Recommendation, and Notifications. Our results demonstrate that our approach improves upon state-of-the-art recommendation models, including RNNs, and makes these applications more useful, such as a >1% increase in video notifications opened. Further, our design choices can be useful to practitioners wanting to transition to more conversational recommendation systems."

>	"To the best of our knowledge, this is the first work on learned interactive recommendation (i.e., asking questions and giving recommendations) demonstrated in a large-scale industrial setting. In building Q&R, we set out to improve the user experience of casual users in YouTube. Users become 18% more likely to complete the User Onboarding experience, and when they do, the numbers of topics they select goes up by 77.7%."

>	"We provide a novel neural-based recommendation approach, which factorizes video recommendation to a two-fold problem: user history-to-topic, and topic& user history-to-video."

>	"Having shed light on a single round of conversation, the area of research in industrial conversational recommendation systems seems to be wide-open for exploration, with incorporating multi-turn conversations and multiple types of data sources, as well as developing models for deciding when to trigger a conversational experience, being exciting topics to be explored in the future."

  - `video` <https://youtube.com/watch?v=-02mJfFoLQo>
  - `video` <http://videolectures.net/kdd2018_beutel_interactive_recommendation> (Beutel)


#### ["Towards Conversational Recommender Systems"](https://chara.cs.illinois.edu/sites/fa16-cs591txt/pdf/Christakopoulou-2016-KDD.pdf) Christakopoulou, Radlinski, Hofmann
>	"People often ask others for restaurant recommendations as a way to discover new dining experiences. This makes restaurant recommendation an exciting scenario for recommender systems and has led to substantial research in this area. However, most such systems behave very di↵erently from a human when asked for a recommendation. The goal of this paper is to begin to reduce this gap. In particular, humans can quickly establish preferences when asked to make a recommendation for someone they do not know. We address this cold-start recommendation problem in an online learning setting. We develop a preference elicitation framework to identify which questions to ask a new user to quickly learn their preferences. Taking advantage of latent structure in the recommendation space using a probabilistic latent factor model, our experiments with both synthetic and real world data compare di↵erent types of feedback and question selection strategies. We find that our framework can make very e↵ective use of online user feedback, improving personalized recommendations over a static model by 25% after asking only 2 questions. Our results demonstrate dramatic benefits of starting from offline embeddings, and highlight the benefit of bandit-based explore-exploit strategies in this setting."

  - `video` <https://youtube.com/watch?v=Wz4vogHwHzY>
  - `video` <https://youtube.com/watch?v=udrkPBIb8D4> (Christakopoulou)
  - `video` <https://youtube.com/watch?v=nLUfAJqXFUI> (Christakopoulou)
