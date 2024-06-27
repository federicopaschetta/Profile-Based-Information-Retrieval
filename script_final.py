#!/usr/bin/env python
# coding: utf-8

# # Profile-Based Retrieval

# #### We first install all required packages:

# In[1]:


get_ipython().system('pip3 install pandas numpy nltk matplotlib')


# In[3]:


get_ipython().system('jupyter nbconvert --to script script_final.ipynb')


# In[3]:


import pandas as pd
import numpy as np
import string as st
from nltk import WordNetLemmatizer
import matplotlib.pyplot as plt


# #### We now import the NLKT library and download all the supplementary data

# In[4]:


import nltk
nltk.download('all')


# #### We read our dataset from .csv file

# In[5]:


df = pd.read_csv('bbc-text.csv', header=0)
df.head(10)


# In[6]:


df.shape


# # Text cleaning and processing steps
# * Remove punctuations
# * Convert text to tokens
# * Remove tokens of length less than or equal to 3
# * Remove stopwords using NLTK corpus stopwords list to match
# * Apply stemming
# * Apply lemmatization
# * Convert words to feature vectors

# In[7]:


# Remove all punctuations from the text

def remove_punct(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))


# In[8]:


df['removed_punc'] = df['text'].apply(lambda x: remove_punct(x))
df.head()


# In[9]:


from nltk.tokenize import word_tokenize
df['tokens'] = df['removed_punc'].apply(lambda msg : word_tokenize(msg.lower()))
df.head()


# In[10]:


# Remove tokens of length less than 3",
def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]


# In[11]:


df['larger_tokens'] = df['tokens'].apply(lambda x : remove_small_words(x))
df.head()


# In[12]:


''' Remove stopwords. Here, NLTK corpus list is used for a match. However, a customized user-defined
    "    list could be created and used to limit the matches in input text.
    "'''
def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]


# In[13]:


df['clean_tokens'] = df['larger_tokens'].apply(lambda x : remove_stopwords(x))
df.head()


# Lemmatization converts word to it's dictionary base form. This process takes language grammar and vocabulary into consideration while conversion. Hence, it is different from Stemming in that it does not merely truncate the suffixes to get the root word.

# In[14]:


# Apply lemmatization on tokens
def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]


# In[15]:


df['lemma_words'] = df['clean_tokens'].apply(lambda x : lemmatize(x))
df.head()


# Let us now annotate each token in a document with its Part-Of-Speech tag (note that tokenized FULL sentences are required!)

# In[16]:


# Annotate each word with its part-of-speech tag

def get_pos_tag(tokenized_sentence):
    return nltk.pos_tag(tokenized_sentence)


# In[17]:


df['pos_tag'] = df['tokens'].apply(lambda x : get_pos_tag(x))
df.head()


# In[18]:


# Create sentences to get clean text as input for vectors",
def return_sentences(tokens):
    return " ".join([word for word in tokens])


# In[19]:


df['clean_text'] = df['lemma_words'].apply(lambda x : return_sentences(x))
df.head()


# In[20]:


df = df[['category', 'text', 'clean_text']]


# In[21]:


df.head()


# ### TF-IDF : Term Frequency - Inverse Document Frequency
# The term frequency is the number of times a term occurs in a document. Inverse document frequency is an inverse function of the number of documents in which a given word occurs.
# The product of these two terms gives tf-idf weight for a word in the corpus. The higher the frequency of occurrence of a word, lower is it's weight and vice-versa. This gives more weightage to rare terms in the corpus and penalizes more commonly occuring terms.

# #### Other widely used vectorizer is Count vectorizer which only considers the frequency of occurrence of a word across the corpus.

# In[22]:


# Convert lemmatized words to Tf-Idf feature vectors\n",
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_vect = tfidf.fit_transform(df['clean_text'])
tfidf_vect.shape


# In[23]:


# Get feature names in the vector
tfidf.get_feature_names_out()


# In[24]:


df['category'].unique()


# In[25]:


# Calculate average TF-IDF values for each word
average_tfidf = tfidf_vect.mean(axis=0)

# Find words with lowest TF-IDF values
lowest_tfidf_indices = average_tfidf.argsort()[:, :50]  # Change 50 to the desired number of lowest values
feature_names = tfidf.get_feature_names_out()
lowest_tfidf_words = [feature_names[idx] for idx in lowest_tfidf_indices.tolist()[0]]

# Remove words with lowest TF-IDF values from texts and save them in a new column
def remove_lowest_tfidf_words(text):
    words = text.split()
    filtered_words = [word for word in words if word not in lowest_tfidf_words]
    return ' '.join(filtered_words)

df['text_with_lowest_tfidf_removed'] = df['clean_text'].apply(remove_lowest_tfidf_words)


# In[26]:


words_to_remove = lowest_tfidf_words + ["said"]

# Remove words with lowest TF-IDF values and "said" from texts and save them in a new column
def remove_lowest_tfidf_words(text):
    words = text.split()
    filtered_words = [word for word in words if word not in words_to_remove]
    return ' '.join(filtered_words)

df['text_with_lowest_tfidf_removed'] = df['clean_text'].apply(remove_lowest_tfidf_words)


# In[27]:


print("Words with lowest TF-IDF values:", lowest_tfidf_words)


# ## Apply LDA to find the main topics in the documents
# 

# In[28]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['text_with_lowest_tfidf_removed'])

# LDA
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Extract top words for each topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx + 1}:")
    print(", ".join([feature_names[i] for i in topic.argsort()[:-51:-1]]))
    print()


# Once we found the main words for 5 topics, label them 

# In[29]:


topics = ['entertainment', 'sport', 'business', 'politics', 'tech']


# In[30]:


doc_topic_probs = lda.transform(X)


# ### Extract the most relevant documents for each topic

# In[31]:


top_documents_per_topic = {}
for topic_idx, topic in enumerate(topics):
    # Get the indices of documents sorted by their probabilities for the current topic
    top_documents_indices = doc_topic_probs[:, topic_idx].argsort()[::-1]
    
    # Get the IDs of the top documents for the current topic
    top_documents_ids = top_documents_indices[:50]
    
    # Store the IDs of the top documents for the current topic
    top_documents_per_topic[topic] = top_documents_ids

# Print the most relevant documents for each topic
for topic, documents_indices in top_documents_per_topic.items():
    print(f"Topic: {topic}")
    for i, doc_id in enumerate(documents_indices, 1):
        print(f"Document {i} - ID: {doc_id}")
        print(df['text_with_lowest_tfidf_removed'].iloc[doc_id])
        print()
    print()


# In[32]:


top_documents_per_topic


# ## Example users
# Define the users and their respective topics of interest

# In[33]:


users = {
    'user1': 'entertainment',
    'user2': 'tech business',
    'user3': 'politics',
    'user4': 'business politics',
    'user5': 'sport'
}


# Create a dictionary with the top 50 documents for each user

# In[34]:


top_documents_per_user = {}

for user, topics_of_interest in users.items():
    top_documents_per_user[user] = []
    topics_of_interest = topics_of_interest.split()
    if len(topics_of_interest) > 1:
        for topic in topics_of_interest:
            top_documents_per_user[user].extend(top_documents_per_topic[topic][:25])
        
        top_documents_per_user[user] = np.array(top_documents_per_user[user])

    else: 
        top_documents_per_user[user] = top_documents_per_topic[topics_of_interest[0]]
    


# Print the most relevant documents for each user

# In[35]:


for user, documents_indices in top_documents_per_user.items():
    topic = users[user]
    print(f"User: {user}")
    print(f"Topics of interest: {topic}")
    for i, doc_id in enumerate(documents_indices, 1):
        print(f"Document {i} - ID: {doc_id}")
        print(df['text_with_lowest_tfidf_removed'].iloc[doc_id])
        print()
    print()


# ## Evaluation

# In[46]:


def gen_eval_dataset(nq=0, nd=0):
    if nq==0:
        nq=np.random.randint(2,4)
    Q=[]
    R=[]
    if nd==0:
        nd=np.random.randint(3,7)
    def generate(nd):
        keep=True
        while keep:
            Q_=1+np.random.permutation(nd)
            val=np.random.rand(1)*.2+.4
            R_=np.sign(np.random.rand(nd)-val)
            keep= np.abs(np.sum(R_))==nd
        return Q_, R_
    for q in range(nq):
        Q_, R_=generate(nd)
        R.append(R_) 
        Q.append(Q_)
    return Q, R


# In[47]:


def generate_ranking(nq, nd):
    Q,R=random_evaluations(nq,nd)
    nq=len(Q)
    nd=len(Q[0])
    print('Your IR system was issued %d queries from users'%nq)
    print('Here are the top %d results your IR system retrieved for each query, '%nd+\
          'together with the relevance judgements')
    header=['q%d\tRel\t\t'%(q+1) for q in range(nq)]
    print('d\t'+''.join(header))
    for d in range(nd):
        row=['%d\t%d\t\t'%(Q[q][d]+1,1 if R[q][d]==1 else 0) for q in range(nq)]
        print('#%d\t'%(d+1)+''.join(row))
    return Q, R


# In[48]:


def evaluate(ex,Q,R):
    nq=len(Q)
    nd=len(Q[0])
    R_=np.array(R)
    R_=.5*(R_+1)
    Prec_tot=[]
    Rec_tot=[]    
    def compute_PR(print_screen=True):
        Prec_tot=[]
        Rec_tot=[]        
        if print_screen:
            print('Precision and Recall at k for k=1,...,%d'%nd)      
        for q in range(nq):
            q1=q+1
            r=R_[q,:]
            if print_screen:
                print('\tQuery %d'%q1)
            Prec_q=[]
            Rec_q=[]
            for k in range(nd):
                k1=k+1
                Prec=np.sum(r[:k1])/k1
                Rec=np.sum(r[:k1])/np.sum(r)                
                if print_screen:                    
                    print('\t\tP(%d)=%d/%d=%.2f,\tR(%d)=%d/%d=%.2f'\
                     %(k1, np.sum(r[:k1]), k1, Prec, k1, np.sum(r[:k1]),np.sum(r),Rec))
                Prec_q.append(Prec)
                Rec_q.append(Rec)
            Prec_tot.append(Prec_q)
            Rec_tot.append(Rec_q)
        Prec_tot=np.array(Prec_tot)
        Rec_tot=np.array(Rec_tot)
        return Prec_tot, Rec_tot
    def compute_TPFP(TP_rate=None):
        TP_tot=[]        
        FP_tot=[]        
        print('TP_rate and FP_rate at k for k=1,...,%d'%nd)      
        for q in range(nq):
            q1=q+1
            r=R_[q,:]
            nr=1-r
            print('\tQuery %d'%q1)
            TP_q=[]
            FP_q=[]
            for k in range(nd):
                k1=k+1
                TP=np.sum(r[:k1])/np.sum(r)                
                FP=np.sum(nr[:k1])/np.sum(nr)
                
                print('\t\tTP_rate(%d)=R(%d)=%d/%d=%.2f\t FP_rate(%d)=%d/%d=%.2f\t'\
                     %(k1, k1, np.sum(r[:k1]),np.sum(r),TP, k1,np.sum(nr[:k1]),np.sum(nr),FP))
                TP_q.append(TP)
                FP_q.append(FP)
            TP_tot.append(TP_q)
            FP_tot.append(FP_q)
        TP_tot=np.array(TP_tot)
        FP_tot=np.array(FP_tot)
        return TP_tot, FP_tot        
    if ex=='prec_rec' or ex=='all':        
        Prec_tot, Rec_tot=compute_PR()
        print('\n Draw the Precision-Recall curve for each query')  
        for q in range(nq):
            q1=q+1
            print('\tQuery %d'%q1)            
            plt.figure()
            Rec_q=Rec_tot[q,:]
            Prec_q=Prec_tot[q,:]
            plt.scatter(np.array(Rec_q), np.array(Prec_q))
            plt.plot(np.array(Rec_q), np.array(Prec_q),label='Precision-Recall curve')            
            plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
            plt.xlabel('Recall'); plt.ylabel('Precision')
            R_int=np.hstack([0,Rec_q,1])
            P_int=np.zeros(R_int.size)
            for i_r in range(R_int.size-1):
                r=R_int[i_r]
                if i_r!=0 and R_int[i_r+1]==r:
                    P_int[i_r]=np.max(Prec_q[i_r-1:])    
                else:
                    P_int[i_r]=np.max(Prec_q[i_r:])            
            plt.plot(R_int,P_int,color='r',label='Interpolated PR curve')
            plt.legend(loc='lower left')
            plt.show()
    if ex=='r-prec' or ex=='all':        
        if len(Prec_tot) == 0:
            Prec_tot, Rec_tot=compute_PR()
        print('\n Determine R-precision for each query') 
        for q in range(nq):            
            Rec_q=Rec_tot[q,:]
            Prec_q=Prec_tot[q,:]
            r=int(np.sum(R_[q]))
            q1=q+1
            print('\tQuery %d'%q1)
            print('\t\tNumber of relevant documents: %d --> P(%d)=%.2f'%(r,r,Prec_q[r-1]))
    if ex=='map' or ex=='all':        
        if len(Prec_tot)== 0:
            Prec_tot, Rec_tot=compute_PR()
        print('\n Calculate the Mean Average Precision')
        APs=[]
        for q in range(nq):            
            Prec_q=Prec_tot[q,:]            
            r=int(np.sum(R_[q]))
            q1=q+1
            print('\tQuery %d'%q1)
            str_formula='1/%d '%r
            rs=np.where(R_[q]==1)[0]+1
            str_formula+='{' + ' + '.join(['P(%d)'%rs_ for rs_ in rs]) + '}'
            AP=np.mean(Prec_q[np.where(R_[q]==1)])            
            print('\t\tAP=%s=%.2f'%(str_formula, AP))
            APs.append(AP)
        APstring='1/%d {'%nq
        APstring+= ' + '.join(['AP_%d'%(q+1) for q in range(nq)]) 
        APstring+='}=1/%d {'%nq
        APstring+= ' + '.join(['%.2f'%(AP) for AP in APs]) 
        APstring+='}'        
        print('\tMAP=%s=%.2f'%(APstring, np.mean(np.array(APs))))
    if ex=='roc' or ex=='all' or ex=='auc':
        TP_tot, FP_tot=compute_TPFP()    
        print('\n Draw the ROC curve for each query')  
        for q in range(nq):
            q1=q+1
            print('\tQuery %d'%q1)            
            plt.figure()
            TP_q=TP_tot[q,:]
            FP_q=FP_tot[q,:]
            plt.scatter(np.array(FP_q), np.array(TP_q))
            TP_q_=np.hstack([0,TP_q,1])
            FP_q_=np.hstack([0,FP_q,1])
            plt.plot(np.array(FP_q_), np.array(TP_q_),label='ROC curve')            
            plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
            plt.xlabel('FP rate'); plt.ylabel('TP rate')
            plt.show()
            if ex=='auc' or ex=='all':
                AUC=[]
                for i_x in range(TP_q_.size-1):
                    delta_x=FP_q_[i_x+1]-FP_q_[i_x]
                    base=TP_q_[i_x+1]+TP_q_[i_x]
                    AUC.append(base*delta_x/2)
                AUC=np.array(AUC)
                AUC=AUC[AUC>0]
                string_AUC=' + '.join(['%.2f'%auc for auc in AUC])
                if string_AUC!='':
                    string_AUC+=' = '    
                print('\tAUC = %s %.2f\n\n'%(string_AUC, np.sum(AUC)))            
    if ex=='clear':
        return
    else:
        return Prec_tot, Rec_tot 


# ### Calculate the predicted categories using the results of the topic modelling

# In[37]:


categories = list(df['category'].unique())


# In[38]:


real_categories = []
predicted_categories = []


# In[39]:


for item in top_documents_per_user.items():
    for doc in item[1]:
        real_categories.append(df['category'].iloc[doc])
        predicted_categories.append(users[item[0]])


# ### Compare the queries with the results, considering the queries the preferencies of each user and the results our predicted categories based on the topic modelling.
# #### The prediction will be considered correct if the documents returned for a specific interest have been classified correctly, comparing the initial labels

# In[43]:


Q = np.ones((1, len(real_categories)), dtype=int)

R = []
for i in range(len(predicted_categories)):
    pred = predicted_categories[i].split()
    real = real_categories[i].split()
    if any(word in pred for word in real):
        R.append(1)
    else: 
        R.append(0)

R = np.array([R])


# Calculate precision and recall using the eval function

# In[49]:


precision, recall = evaluate('all', Q, R)


print("Precision:", precision)
print("Recall:", recall)


# # Second Version - Use selected words to identify documents 

# ## We asked ChatGPT to give us a list of the 50 most appropriate words to each of the given topics and saved them in words.txt file

# In[66]:


words_topic = {}


# ### Reading the words and for each topic creating the list of words

# In[67]:


with open('words.txt', 'r') as gpt_words:
    for line in gpt_words:
        words = line.strip().split(',')
        words_topic[words[0]] = words[1:]


# In[68]:


tfidf_vect.toarray().shape
tfidf_arr = tfidf_vect.toarray()


# ### Creating the TF-IDF vectors matrix

# In[69]:


tokens = []
for i, feature in enumerate(tfidf.get_feature_names_out()):
    tokens.append(feature)
tfidf_matrix = pd.DataFrame(tfidf_arr, columns = tokens)
tfidf_matrix.iloc[:,8000:]


# In[70]:


doc_vect = tfidf.fit_transform(df["clean_text"])


# ### Creating a list of keyword lists for each user

# In[71]:


users_keywords = []
for index, i in enumerate(users.keys()):
    split = users[i].strip().split()
    if len(split) == 1:
        users_keywords.append(words_topic[split[0]])
    else:
        users_keywords.append(words_topic[split[0]]+words_topic[split[1]])


# In[72]:


print(users_keywords)


# ### Creating a list of users vectors

# In[73]:


users_vect = []
for i in users_keywords:
    users_vect.append(tfidf.transform([" ".join(i)]))


# In[74]:


users_vect
doc_vect


# ## Getting most appropriate documents for each user

# ### Creating a DataFrame to get similarity of each document with users keywords

# In[75]:


sim_df = pd.DataFrame(columns=['articleId', 'clean_text','user1', 'user2', 'user3', 'user4', 'user5', 'category'])
sim_df['articleId'] = df.index
sim_df['clean_text'] = df['clean_text']
sim_df['category'] = df['category']


# In[76]:


sim_df


# ### Filling the DataFrame just created with similarity values

# In[78]:


from sklearn.metrics.pairwise import cosine_similarity
df_index = 2
for i in range(len(users_vect)):
    sim_list = []
    for j in range(len(list(doc_vect))):
        sim = cosine_similarity(users_vect[i], doc_vect[j])
        sim_list.append(sim[0][0])
    sim_df[sim_df.columns[df_index]] = sim_list
    df_index+=1



# In[79]:


sim_df.head()


# ### Top Entertainment documents (related to User 1)

# In[80]:


entertainment = sim_df.sort_values('user1', ascending=False)
entertainment[:10]['clean_text']


# ## Getting most appropriate users for each document

# In[99]:


threshold = 0.025
doc_users_dict = {}


# ### Filling dictionary just created with users interested in document for each document index

# In[100]:


# doc_index is index of document chosen
# show param determines if need to print users selection or not
def get_interested(doc_index, show=False):
    list_sim = []
    perfiles = []
    topics_usr = []

    for j in range(len(users_vect)):
        similarities = cosine_similarity(doc_vect[doc_index], users_vect[j]) # Computes the distance of cosine between document and user vectors
        perfiles.append(j+1)
        topics_usr.append(users[list(users.keys())[j]])
        list_sim.append(similarities[0][0])

    # Creates a DataFrame with information retrieved and ranks users by score

    ranking = pd.DataFrame()
    ranking["users"] = perfiles
    ranking['topics'] = topics_usr
    ranking["score"] = list_sim
    ranking = ranking.sort_values('score', ascending=False)
    
    index_max = ranking['score'].idxmax()
    row_max = ranking.loc[index_max]
    
    # Prints information about previous process
    if show:
        print("Document",df.iloc[doc_index], ":", df.iloc[doc_index]["text"])
        print('Actual Topic: ', df['category'][doc_index], '-- Most Appropriate User Topics: ', row_max['topics'])
        print()
        print('Ranking')
        print(ranking)
        for index, row in ranking.iterrows():
            if row.iloc[2] > threshold:
                print('Most Interested User: ', row.iloc[0])
    
    # Creates a list with most suitable users for document
    # User with more than 0.010 of similarity are selected as suitable, if none, highest score is selected
    interested_users = []
    for index, row in ranking.iterrows():
        if row.iloc[2] > threshold:
            interested_users.append(row.iloc[0])
        if len(interested_users) == 0:
            interested_users.append(row_max.iloc[0])
    doc_users_dict[doc_index] = interested_users
    


# ### Selecting random document from DataFrame

# In[101]:


import random
random_index = random.randint(0, len(df["clean_text"])-1)
get_interested(random_index, True)


# ### Filling dictionary with most appropriate users for each document

# In[102]:


for index, row in df.iterrows():
    get_interested(index)


# ## Evaluation

# ### Evaluating general accuracy of the method, comparing users selected topics with category of document

# #### loose version assures that user with the highest score is interested in the document topic, regardless of the threshold
# #### strict version (with loose=False) checks that all the users selected are interested in the document topic

# In[105]:


def evaluate_all(loose=True):
    users_topics_list = list(users.keys())
    correct = 0
    for i in range(len(df)):
        all_chosen = True
        user_chosen = doc_users_dict[i]
        stop = len(user_chosen)
        if loose:
            stop = 1
        
        for j in range(stop):
            if df.iloc[i]['category'] not in users[users_topics_list[user_chosen[j]-1]]:
                all_chosen = False
        if all_chosen:
            correct = correct + 1
    return correct/len(df)


# In[106]:


print("Loose Accuracy: ", evaluate_all())
print("Strict Accuracy: ", evaluate_all(False))


# In[107]:


q=[]
r=[]

for i in range(len(users.keys())):
    qi = sim_df.sort_values(list(users.keys())[i], ascending=False)
    ri = qi['category'].str.contains(list(users.values())[i]) 

    q.append(qi)
    r.append(ri)


# In[108]:


precision, recall = evaluate('all', q, r)

print("Precision:", precision)
print("Recall:", recall)

