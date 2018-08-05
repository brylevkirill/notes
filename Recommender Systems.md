

  * [**overview**](#overview)
  * [**interesting papers**](#interesting-papers)
    - [**item similarity**](#interesting-papers---item-similarity)
    - [**user preferences**](#interesting-papers---user-preferences)
    - [**deep learning**](#interesting-papers---deep-learning)
    - [**active learning**](#interesting-papers---active-learning)



---
### overview

  ["Model-Based Machine Learning: Making Recommendations"](http://mbmlbook.com/Recommender.html) by Chris Bishop

  ["Recommender Systems: The Textbook"](http://charuaggarwal.net/Recommender-Systems.htm) by Charu Aggarwal ([book](https://yadi.sk/i/eMDtp31h3P8AxL))  
  ["Recommender Systems Handbook"](http://www.cs.ubbcluj.ro/~gabis/DocDiplome/SistemeDeRecomandare/Recommender_systems_handbook.pdf) by Ricci, Rokach, Shapira, Kantor  

----

  [overview](https://youtube.com/watch?v=gCaOa3W9kM0) by Alex Smola `video`  
  [overview](https://youtube.com/watch?v=xMr7I-OypVY) by Alex Smola `video`  

  [tutorial](http://technocalifornia.blogspot.ru/2014/08/introduction-to-recommender-systems-4.html) by Xavier Amatriain `video`  
  ["The Recommender Problem Revisited"](http://videolectures.net/kdd2014_amatriain_mobasher_recommender_problem) by Xavier Amatriain `video`  
  ["Lessons Learned from Building Real-life Recommender Systems"](https://youtube.com/watch?v=VJOtr47V0eo) by Xavier Amatriain and Deepak Agarwal `video`  

  ["Deep Learning for Recommender Systems"](https://slideshare.net/kerveros99/deep-learning-for-recommender-systems-recsys2017-tutorial) by Alexandros Karatzoglou and Balazs Hidasi `slides`  
  ["Deep Learning for Recommender Systems"](https://youtube.com/watch?v=KZ7bcfYGuxw) by Alexandros Karatzoglou `video`  
  ["Deep Learning for Personalized Search and Recommender Systems"](https://youtube.com/watch?v=0DYQzZp68ok) by Zhang, Le, Fawaz, Venkataraman `video`  

  [ACM RecSys](https://youtube.com/channel/UC2nEn-yNA1BtdDNWziphPGA) conference `video`

----

  [overview](https://youtube.com/watch?v=Us4KJkJiYrM) by Michael Rozner `video` `in russian`  
  overview by Sergey Nikolenko ([part 1](https://youtube.com/watch?v=mr8u54jsveA), [part 2](https://youtube.com/watch?v=cD47Ssp_Flk), [part 3](https://youtube.com/watch?v=OFyb8ukrRDo)) `video` `in russian`  
  [overview](https://youtube.com/watch?v=kfhqzkcfMqI) by Konstantin Vorontsov `video` `in russian`  
  [overview](https://youtube.com/watch?v=Te_6TqEhyTI) by Victor Kantor `video` `in russian`  
  [overview](https://youtube.com/watch?v=5ir_fCgzfLM) by Vladimir Gulin `video` `in russian`  
  [overview](https://youtube.com/watch?v=MLljnzsz9Dk) by Alexey Dral `video` `in russian`  

  [overview](https://youtube.com/watch?v=N0NUwz3xWX4) of deep learning for recommender systems by Dmitry Ushanov `video` `in russian`

  [overview](https://youtube.com/watch?v=iGAMPnv-0VY) of Yandex.Zen by Igor Lifar and Dmitry Ushanov `video` `in russian`  
  [overview](https://youtube.com/watch?v=JKTneRi2vn8) of Yandex.Disco by Michael Rozner `video` `in russian`  

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



---
### interesting papers - item similarity


#### ["Two Decades of Recommender Systems at Amazon.com"](https://www.computer.org/csdl/mags/ic/2017/03/mic2017030012.html) Smith, Linden
  `Amazon`
>	learning item-to-item similarity on offline data (e.g. item2 often bought with item1)


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
  `Netflix`
>	"We extend variational autoencoders to collaborative filtering for implicit feedback. This non-linear probabilistic model enables us to go beyond the limited modeling capacity of linear factor models which still largely dominate collaborative filtering research. We introduce a generative model with multinomial likelihood and use Bayesian inference for parameter estimation. Despite widespread use in language modeling and economics, the multinomial likelihood receives less attention in the recommender systems literature. We introduce a different regularization parameter for the learning objective, which proves to be crucial for achieving competitive performance. Remarkably, there is an efficient way to tune the parameter using annealing. The resulting model and learning algorithm has information-theoretic connections to maximum entropy discrimination and the information bottleneck principle. Empirically, we show that the proposed approach significantly outperforms several state-of-the-art baselines, including two recently-proposed neural network approaches, on several real-world datasets. We also provide extended experiments comparing the multinomial likelihood with other commonly used likelihood functions in the latent factor collaborative filtering literature and show favorable results. Finally, we identify the pros and cons of employing a principled Bayesian inference approach and characterize settings where it provides the most significant improvements."

>	"Recommender systems is more of a "small data" than a "big data" problem."  
>	"VAE generalizes linear latent factor model and recovers Gaussian matrix factorization as a special linear case. No iterative procedure required to rank all the items given a user's watch history - only need to evaluate inference and generative functions."  
>	"We introduce a regularization parameter for the learning objective to trade-off the generative power for better predictive recommendation performance. For recommender systems, we don't necessarily need all the statistical property of a generative model. We trade off the ability of performing ancestral sampling for better fitting the data."  

  - `video` <https://youtube.com/watch?v=gRvxr47Gj3k> (Liang)
  - `code` <https://github.com/dawenl/vae_cf>


#### ["Content-based Recommendations with Poisson Factorization"](http://www.cs.toronto.edu/~lcharlin/papers/GopalanCharlinBlei_nips14.pdf) Gopalan, Charlin, Blei
>	"We develop collaborative topic Poisson factorization (CTPF), a generative model of articles and reader preferences. CTPF can be used to build recommender systems by learning from reader histories and content to recommend personalized articles of interest. In detail, CTPF models both reader behavior and article texts with Poisson distributions, connecting the latent topics that represent the texts with the latent preferences that represent the readers. This provides better recommendations than competing methods and gives an interpretable latent space for understanding patterns of readership. Further, we exploit stochastic variational inference to model massive real-world datasets. For example, we can fit CPTF to the full arXiv usage dataset, which contains over 43 million ratings and 42 million word counts, within a day. We demonstrate empirically that our model outperforms several baselines, including the previous state-of-the art approach."

>	collaborative topic models:  
>	- blending factorization-based and content-based recommendation  
>	- describing user preferences with interpretable topics  

  - `video` <http://www.fields.utoronto.ca/video-archive/2017/03/2267-16706> (26:36) (Blei)
  - `code` <https://github.com/premgopalan/collabtm>


#### ["Scalable Recommendation with Hierarchical Poisson Factorization"](http://auai.org/uai2015/proceedings/papers/208.pdf) Gopalan, Hofman, Blei
>	"We develop hierarchical Poisson matrix factorization (HPF), a novel method for providing users with high quality recommendations based on implicit feedback, such as views, clicks, or purchases. In contrast to existing recommendation models, HPF has a number of desirable properties. First, we show that HPF more accurately captures the long-tailed user activity found in most consumption data by explicitly considering the fact that users have finite attention budgets. This leads to better estimates of users’ latent preferences, and therefore superior recommendations, compared to competing methods. Second, HPF learns these latent factors by only explicitly considering positive examples, eliminating the often costly step of generating artificial negative examples when fitting to implicit data. Third, HPF is more than just one method- it is the simplest in a class of probabilistic models with these properties, and can easily be extended to include more complex structure and assumptions. We develop a variational algorithm for approximate posterior inference for HPF that scales up to large data sets, and we demonstrate its performance on a wide variety of real-world recommendation problems - users rating movies, listening to songs, reading scientific papers, and reading news articles."

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


#### ["Wide & Deep Learning"](https://arxiv.org/abs/1606.07792) Cheng et al.
  `Google`
>	"Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning---jointly trained wide linear models and deep neural networks---to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow."

  - `post` <https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html>
  - `video` <https://youtube.com/watch?v=NV1tkZ9Lq48> (Cheng)
  - `post` <https://www.tensorflow.org/tutorials/wide_and_deep>
  - `code` <https://github.com/tensorflow/models/blob/master/official/wide_deep/wide_deep.py>


#### ["Deep Neural Networks for YouTube Recommendations"](http://research.google.com/pubs/pub45530.html) Covington, Adams, Sargin
  `Google`
>	"YouTube represents one of the largest scale and most sophisticated industrial recommendation systems in existence. In this paper, we describe the system at a high level and focus on the dramatic performance improvements brought by deep learning. The paper is split according to the classic two-stage information retrieval dichotomy: first, we detail a deep candidate generation model and then describe a separate deep ranking model. We also provide practical lessons and insights derived from designing, iterating and maintaining a massive recommendation system with enormous userfacing impact."

>	"We have described our deep neural network architecture for recommending YouTube videos, split into two distinct problems: candidate generation and ranking. Our deep collaborative filtering model is able to effectively assimilate many signals and model their interaction with layers of depth, outperforming previous matrix factorization approaches used at YouTube. There is more art than science in selecting the surrogate problem for recommendations and we found classifying a future watch to perform well on live metrics by capturing asymmetric co-watch behavior and preventing leakage of future information. Withholding discrimative signals from the classifier was also essential to achieving good results - otherwise the model would overfit the surrogate problem and not transfer well to the homepage. We demonstrated that using the age of the training example as an input feature removes an inherent bias towards the past and allows the model to represent the time-dependent behavior of popular of videos. This improved offline holdout precision results and increased the watch time dramatically on recently uploaded videos in A/B testing. Ranking is a more classical machine learning problem yet our deep learning approach outperformed previous linear and tree-based methods for watch time prediction. Recommendation systems in particular benefit from specialized features describing past user behavior with items. Deep neural networks require special representations of categorical and continuous features which we transform with embeddings and quantile normalization, respectively. Layers of depth were shown to effectively model non-linear interactions between hundreds of features. Logistic regression was modified by weighting training examples with watch time for positive examples and unity for negative examples, allowing us to learn odds that closely model expected watch time. This approach performed much better on watch-time weighted ranking evaluation metrics compared to predicting click-through rate directly."

  - `video` <https://youtube.com/watch?v=WK_Nr4tUtl8> (Covington)
  - `notes` <https://blog.acolyer.org/2016/09/19/deep-neural-networks-for-youtube-recommendations>
  - `notes` <https://ekababisong.org/deep-neural-networks-youtube>
  - `code` <https://github.com/ogerhsou/Youtube-Recommendation-Tensorflow/blob/master/youtube_recommendation.py>


#### [Spotlight](https://maciejkula.github.io/spotlight) Kula
>	"Spotlight uses PyTorch to build both deep and shallow recommender models. By providing both a slew of building blocks for loss functions (various pointwise and pairwise ranking losses), representations (shallow factorization representations, deep sequence models), and utilities for fetching (or generating) recommendation datasets, it aims to be a tool for rapid exploration and prototyping of new recommender models."

>	"Recommendations can be seen as a sequence prediction task: given the items a user has interacted with in the past, what will be the next item they will interact with? Spotlight provides a range of models and utilities for fitting next item recommendation models, including:  
>	- pooling models, as in YouTube recommendations  
>	- LSTM models, as in Session-based recommendations  
>	- causal convolution models, as in WaveNet."  

  - `video` <https://youtube.com/watch?v=ZkBQ6YA9E40> (Kula)
  - `code` <https://github.com/maciejkula/spotlight>



---
### interesting papers - active learning


#### ["Making Contextual Decisions with Low Technical Debt"](http://arxiv.org/abs/1606.03966) Agarwal et al.
  `Microsoft Custom Decision Service`

  - ["Making Contextual Decisions with Low Technical Debt"](https://github.com/brylevkirill/notes/blob/master/Reinforcement%20Learning.md#making-contextual-decisions-with-low-technical-debt-agarwal-et-al) by Agarwal et al. `summary`


#### ["Towards Conversational Recommender Systems"](https://chara.cs.illinois.edu/sites/fa16-cs591txt/pdf/Christakopoulou-2016-KDD.pdf) Christakopoulou, Radlinski, Hofmann
>	"People often ask others for restaurant recommendations as a way to discover new dining experiences. This makes restaurant recommendation an exciting scenario for recommender systems and has led to substantial research in this area. However, most such systems behave very di↵erently from a human when asked for a recommendation. The goal of this paper is to begin to reduce this gap. In particular, humans can quickly establish preferences when asked to make a recommendation for someone they do not know. We address this cold-start recommendation problem in an online learning setting. We develop a preference elicitation framework to identify which questions to ask a new user to quickly learn their preferences. Taking advantage of latent structure in the recommendation space using a probabilistic latent factor model, our experiments with both synthetic and real world data compare di↵erent types of feedback and question selection strategies. We find that our framework can make very e↵ective use of online user feedback, improving personalized recommendations over a static model by 25% after asking only 2 questions. Our results demonstrate dramatic benefits of starting from offline embeddings, and highlight the benefit of bandit-based explore-exploit strategies in this setting."

  - `video` <https://youtube.com/watch?v=udrkPBIb8D4> (Christakopoulou)
  - `video` <https://youtube.com/watch?v=nLUfAJqXFUI> (Christakopoulou)
