from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string, re, pickle, os, operator
import numpy
import constants
from collections import Counter
import nltk

GENERATED_RESOURCES = constants.GENERATED_RESOURCE_LOCATION

class LDA:
	def __init__(self,K, alpha, beta, docs, V, smartinit=True):
		## LDA Algorithm Parameters
		self.K = K
		self.alpha = alpha # parameter of topics prior
		self.beta = beta   # parameter of words prior
		self.docs = docs
		self.V = V 			## Vocab size .. updated once Vocab is built
		## LDA Algorithm initializations
		self.z_m_n = [] # topics of words of documents
		self.n_m_z = numpy.zeros((len(self.docs), K)) + alpha     # word count of each document and topic
		self.n_z_t = numpy.zeros((K, V)) + beta # word count of each topic and vocabulary
		self.n_z = numpy.zeros(K) + V * beta    # word count of each topic

		self.N = 0
		for m, doc in enumerate(docs):
			self.N += len(doc)
			z_n = []
			for t in doc:
				if smartinit:
					p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
					z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
				else:
					z = numpy.random.randint(0, K)
				z_n.append(z)
				self.n_m_z[m, z] += 1
				self.n_z_t[z, t] += 1
				self.n_z[z] += 1
			self.z_m_n.append(numpy.array(z_n))

	
	##"""learning once iteration"""
	def inference(self):
		for m, doc in enumerate(self.docs):
			z_n = self.z_m_n[m]
			n_m_z = self.n_m_z[m]
			for n, t in enumerate(doc):
				# discount for n-th word t with topic z
				z = z_n[n]
				n_m_z[z] -= 1
				self.n_z_t[z, t] -= 1
				self.n_z[z] -= 1
				# sampling topic new_z for t
				p_z = self.n_z_t[:, t] * n_m_z / self.n_z
				new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
				# set z the new topic and increment counters
				z_n[n] = new_z
				n_m_z[new_z] += 1
				self.n_z_t[new_z, t] += 1
				self.n_z[new_z] += 1


	def worddist(self):
		"""get topic-word distribution"""
		return self.n_z_t / self.n_z[:, numpy.newaxis]


	def perplexity(self, docs=None):
		if docs == None: docs = self.docs
		phi = self.worddist()
		log_per = 0
		N = 0
		Kalpha = self.K * self.alpha
		for m, doc in enumerate(docs):
			theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
			for w in doc:
				log_per -= numpy.log(numpy.inner(phi[:,w], theta))
			N += len(doc)
		return numpy.exp(log_per / N)






class Vocabulary:
	def __init__(self):
		## Vocabulary creating
		self.vocas = []        # id to word
		self.vocas_id = dict() # word to id
		self.docfreq = []      # id to document frequency

	def term_to_id(self, term):
		if not re.match(r'[a-z]+$',term):
			return None
		if term not in self.vocas_id:
			voca_id = len(self.vocas)
			self.vocas_id[term] = voca_id
			self.vocas.append(term)
			self.docfreq.append(0)
		else:
			voca_id = self.vocas_id[term]
		return voca_id

	def doc_to_ids(self,doc):
		doc_id_lst = []
		words = dict()
		for term in doc:
			id = self.term_to_id(term)
			if id != None:
				doc_id_lst.append(id)
				if not id in words:
					words[id] = 1
					self.docfreq[id] += 1
		if "close" in dir(doc): doc.close()
		return doc_id_lst

	def size(self):
		return len(self.vocas)

	def __getitem__(self, v):
		return self.vocas[v]

	def save_vocab(self,vocab, vocab_id, vocab_freq):
	    saveDict = []
	    for i in range(0, len(vocab_freq)):
	        tup = tuple
	        tup = i , vocab[i], vocab_freq[i]
	        saveDict.append(tup)
	    pickle.dump(saveDict, open(os.path.join(GENERATED_RESOURCES,"training_vocab"),'wb'))




def lda_learning(lda, iteration, voca):
    pre_perp = lda.perplexity()
    #print ("initial perplexity=%f" % pre_perp)
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        #print ("-%d p=%f" % (i + 1, perp))
        if pre_perp:
            if pre_perp < perp:
                output_word_topic_dist(lda, voca)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(lda, voca)




def output_word_topic_dist(lda, voca):
    zcount = numpy.zeros(lda.K, dtype=int)
    wordcount = [dict() for k in range(lda.K)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist, zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1
    ## phi => the probability of a word belonging to the Topic
    phi = lda.worddist()
    modelWriter = {}
    with open(os.path.join(GENERATED_RESOURCES,'topic_word_distribution.dat'), 'w') as f:
        for k in range(lda.K):
            # print ("\n-- topic: %d (%d words)" % (k, zcount[k]))
            line = "\n-- topic: %d (%d words)" % (k, zcount[k])
            f.write('%r\n\n' % line)
            f.write('')
            key = k
            val = tuple
            ListAllTuples = []     

            ## write the latent topics and most dominant words in the topic to file 
            for w in numpy.argsort(-phi[k])[:20]:                
                #print ("%s: %f (%d) (%d)" % (voca[w], phi[k,w], wordcount[k].get(w,0), zcount[k]))
                line = "%s: %f (%d) (%d)" % (voca[w], phi[k,w], wordcount[k].get(w,0), zcount[k])
                f.write('%r\n' % line)
            f.write('**************************************\n\n')
            ## store the Model for later use
            for w in numpy.argsort(-phi[k]):
            	val = voca[w], phi[k,w], wordcount[k].get(w, 0), zcount[k]
            	ListAllTuples.append(val)
            modelWriter[key] = ListAllTuples
        ##	save the Model to disc
        pickle.dump(modelWriter, open(os.path.join(GENERATED_RESOURCES,"model"),"wb"))





def normalize_data(review_list):
	stop = set(stopwords.words('english'))
	exclude = set(string.punctuation)
	lemma = WordNetLemmatizer()
	def clean(doc):
	    # stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
	    #punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	    #normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
	    tokens = nltk.word_tokenize(doc)
	    noun_lst = []
	    tagged_words = nltk.pos_tag(tokens)
	    for word in tagged_words:
	    	if word[1] in ["NN", "NNS"]:
	    		noun_lst.append(word[0])
	    #	nouns = " ".join(noun_lst)
	    lemmatized = " ".join(lemma.lemmatize(word) for word in noun_lst)
	    stop_free = " ".join([i for i in lemmatized.split() if i not in stop])
	    #print(normalized)
	    return stop_free
	doc_clean = [clean(doc).split() for doc in review_list]
	#print(doc_clean)
	return doc_clean





def tag_reviews(review_list):
	clean_reviews = normalize_data(review_list)
	model = pickle.load(open(os.path.join(GENERATED_RESOURCES,'model'),'rb'))
	training_vocab = pickle.load(open(os.path.join(GENERATED_RESOURCES,'training_vocab'),'rb'))
	# print (model)
	# print (training_vocab)
	##### voca = Vocabulary()
	##### docs = [voca.doc_to_ids(doc) for doc in clean_reviews]
	#print (clean_reviews)
	doc_word_freq = []		## words and their frequencies in a specific review
	for doc in clean_reviews:
		word_counts = Counter(doc)
		word_freq = []
		for key in word_counts:
			word_freq.append((key, word_counts[key]))
		doc_word_freq.append(word_freq)
	#print(len(doc_word_freq[0]))
	## Get a list of words in each review that are present in the training Vocab
	doc_word_freq_known = []	## words present in training vocab and their frequencies in a specific review
	for review in doc_word_freq:
		single_review_freq_lst = []
		for word_freq in review:
			word, freq = word_freq
			for item in training_vocab:
				if (word == item[1]):
					single_review_freq_lst.append(word_freq)
		doc_word_freq_known.append(single_review_freq_lst)
	#print(len(doc_word_freq_known[0]))
	## Calculate the topic scores for the reviews
	doc_topic_score = []	## holds the scores of all the topics for each review
	for doc in doc_word_freq_known:	## taking a single review
		scores = {}		## holds the topic scores for a single review
		for topic in model:	## taking a single topic from model
			scores[topic] = 0	##	initialize each of the currect topic score for the current review to 0
			topic_word_prop_lst = model[topic]
			for word_freq in doc:	## taking each of the word_freq tuple from a single review
				for topic_word_prop in topic_word_prop_lst: ## taking each of the word_props from a single topic
					#print (topic_word_prop[0], word_freq[0], topic)
					if (topic_word_prop[0] == word_freq[0]):
						word_freq_review = word_freq[1]
						word_topic_score_model = topic_word_prop[1]
						scores[topic] = scores[topic] + (word_freq_review*word_topic_score_model)
						break
		doc_topic_score.append(scores)
	## tagging the reviews to a certain topic (two top topics)
	tagged_review_list = []
	for review, topic_scores in zip(review_list, doc_topic_score):
		#print (review, topic_scores)
		## Pull Prediction
		#review_topic = max(topic_scores.items(), key=operator.itemgetter(1))[0]
		#review_topics = list(dict(sorted(topic_scores.items(), key=operator.itemgetter(1), reverse=True)[:2]).keys())
		#tagged_review = {'text':review, 'topics':review_topics}
		#print(tagged_review)
		## Pull whole score
		scored_review = {'text':review, 'topics':topic_scores}
		tagged_review_list.append(scored_review)

	#print(tagged_review_list)
	return tagged_review_list






def run_lda(K, alpha, beta, review_list, iterations=50):
	clean_reviews = normalize_data(review_list)
	#print (clean_reviews)
	voca = Vocabulary()
	docs = [voca.doc_to_ids(doc) for doc in clean_reviews]
	## save the vocab to be used later for Tagging unseen Reviews
	voca.save_vocab(voca.vocas, voca.vocas_id, voca.docfreq)
	#print(docs)
	lda = LDA (K, alpha, beta, docs, voca.size())
	print ("corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(docs), len(voca.vocas), K, alpha, beta))
	## get the latent topics
	lda_learning(lda, iterations, voca)
	print ('LDA Model Generated !!')
	tagged_review_list = tag_reviews(review_list)
	return tagged_review_list


^^^()()^^^