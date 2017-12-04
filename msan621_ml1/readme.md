# Machine Learning
### Unofficial Lecture Note Website

Welcome to the MSAN 2017 Machine Learning 1 unofficial lecture notes website. I've collected all my inclass notes from our 8-week course in the fall. Each of the 2-hour lecture notes are provided below.


### Lecture 1
- Introductions and class basics

--- 
### [Lecture 2 Notes](ML-Lecture2.ipynb)

- Python basics
- Git, Symlink, AWS
- Python notebook basics
- Crash course on pandas
- FastAI introduction
- `add_datepart`
- `train_cats`
- Feather Format
- Run your first Random Forest

---

### [Lecture 3 Notes](ML-Lecture3.1.ipynb)
- R^2 accuracy
- How to make validation sets
- Test vs. Validation Set
- Diving into RandomForests
- Examination of One tree
- What is 'bagging'
- What is OOB Out-of-Box score
- RF Hyperparameter 1: Trees
- RF Hyperparameter 2: max Samples per leaf
- RF Hyperparameter 3: max features

---

### [Lecture 4 Notes](ML-Lecture3.2-public.ipynb)
- Forecasting: Grocery Kaggle discussion, Parallel to Rossman stores
- Random Forests: Confidence based tree variance
- Random Forests: Feature Importance Intro
- Random Forests: Decoupled Shuffling

---


### [Lecture 5 Notes](ML-Lecture4.1-public.ipynb)
- Summary of Random Forests
	- Data needs to be numeric
	- Categories go to numbers
- Subsampling in different trees
	- Tree size
	- Records per node
	- Information Gain (improvement)
	- Repeat process for different subsetes
	- Each tree should be better
	- Trees should not be correlated
- Min Leaf Samples
- Max Features
- n_jobs
- oob
- interpretting OOB vs. Training vs. Test score
- Feature Importance Deep dive
- One hot encoding
- Redundant features
- Partial Dependence

---


### [Lecture 6 Notes](ML-Lecture4.2-public.ipynb)
- What makes a good validation set?
- What makes a good test set?
- Random Forest from scratch : setup framework

---


### [Lecture 7 Notes](ML-Lecture5.1-public.ipynb)
- Motivations for data science
- Thinking about the business implications
- Tell the story
- Review of Confidence in Tree Prediction Variance, Feature importance, Partial Dependence

---


### [Lecture 8 Notes](ML-Lecture5.2-public.ipynb)
- Building a Decision Tree from scratch
- Optimizing and comparing to SKlearn
- How to do 2 levels of decision trees
- Fleshing out the RF `predict` function
- Assembling our own decision tree
- Cython

---


### [Lecture 9 Notes](ML-Lecture6.1-public.ipynb)
- Deep Learning
- Using pytorch and a 1-level NN
- Walkthrough of MNIST number sets
- Binary Loss func
- Making a LogReg equivalent NN pytorch

---

### [Lecture 10 Notes](ML-Lecture7.1-public.ipynb)
- Rewriting the 1-layer NN from scratch
- Rewrite LinearLayer
- Rewrite Softmax
- Understanding numpy and torch matrix operations
- Understanding Broadcasting rules
- Rewriting matrix mult from scratch
- Start looking at the `fit` function

---

### [Lecture 11](ML-Lecture7.2-public.ipynb)
- Rewriting `fit` from scratch
- Digression of Momentum
- Rewriting `gradient` and `step` within `fit` function
- NLP
- Bag of words / CountVectorizer
- LogisticRegression w. Sentiment

---

### [Lecture 12](ML-Lecture7.3.ipynb)
- NLP : trigrams
- Naive Bayes Classifier
- Binarized version of NB
- NBSVM - combination of probs
- Storage efficiency of 1-hot
- RossMan store examination
- Introduction to embeddings



