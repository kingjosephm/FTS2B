# -*- coding: utf-8 -*-
"""
Created on Mon Oct  28 13:02:49 2019

    Uses latent Dirichlet allocation on fault descriptions to cluster fault
    types. Using 10% panel of serial numbers to iterate through range of various numbers
    of topics to find the one with the smallest AIC/BIC & best coherence. Uses this number of topics to
    apply LDA model to full sample. Creates new column of dominant topics, i.e. for each
    fault what topic number has highest contribution score.

@author: jking
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, gensim, json, warnings, pickle, json
import statsmodels.api as sm
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel, TfidfModel
pd.options.display.max_rows = 200
pd.options.display.max_columns = 75
pd.options.mode.chained_assignment = None # turns of SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess(text):
    '''
    Function takes Pandas series, removes punctuation, numbers, and other non-characters,
        returns list of 3-15 char words per observation (a list of lists).
    :text: Pandas series
    '''
    doc = pd.Series(text)
    ### Tokenize ###
    doc = doc.str.replace('T/R', 'tail rotor').str.replace('LH', 'left')\
        .str.replace('RH', 'right').str.replace('PMI', 'inspection').str.replace('RMVD', 'remove')\
        .str.replace('PMD|PMS', 'maint')
    doc.fillna(value='', inplace=True) # replace missing fault fields with empty string
    doc = doc.apply(lambda x: ' '.join(s for s in x.split() if not any(c.isdigit() for c in s))) # remove whole words if any char number
    string = '\n+|\d+|^\W+|\(|\)|\.+|\-|:|,+|\"|#+|@|\\\+|&|%|\'s|!|$' # regex non-chars to catch
    doc = doc.str.replace(string, '').str.lstrip().str.replace('\s+|\/', ' ').str.lower() # remove punctuation, extra whitespace, all to lower case
    doc = doc.str.findall('\w{3,15}').str.join(' ') # keep words 3-15 char in length
    return doc.str.split().tolist() # split string into individual words, return as list

def lemmatize_stemming(token):
    '''
    Function stems nouns to base form (e.g. removing plural) and lemmatizes verbs 
        (i.e. change to present tense) for a token (an element) in a list of word tokens.
        Takes a string, outputs a modified string.
    :token: str element in a list
    '''
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v'))

def bigram_trigram(text):
    '''
    Creates bigram and trigrams (2- and 3-word combinations, respectively) for
        words that commonly appear together. Accepts list of lists, outputs
        a new list of lists with bigrams and trigrams.
    '''
    # Make bigrams and trigrams (words that appear together), see: https://radimrehurek.com/gensim/models/phrases.html
    t = 40 # threshold for scoring parameter, higher means fewer phrases
    bigram = gensim.models.Phrases(text, min_count=5, threshold=t)
    trigram = gensim.models.Phrases(bigram[text], threshold=t)
    bigram = gensim.models.phrases.Phraser(bigram) # apparently this speeds things up
    trigram = gensim.models.phrases.Phraser(trigram)
    return [trigram[bigram[doc]] for doc in text]

def process(text):
    '''
    Function removes stopwords (prepositions and pronouns) and applies preprocessing
        and lemmatizer/stemmer functions. Applied to Pandas series, outputs list of 
        unique tokens by default and this same list converted to pandas series.
    :text: Pandas series
    '''
    p = preprocess(text)
    stop_words = stopwords.words('english')
    stop_words.extend(['day', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                        'aug', 'sept', 'oct', 'nov', 'dec', 'due', 'hours', 
                        'hour', 'red', 'hand', 'could', 'right', 'left', 'fwd',
                        'rear', 'rpm', 'aft', 'upr'])
    result = []
    for nr in range(0, len(p)): # for each list of lists
        words = []
        try:
            for element in p[nr]: # elements in that list
                if element not in stop_words:
                    words.append(lemmatize_stemming(element))
                words = list(set(words)) # remove duplicate words within list
        except TypeError: # if no text for fault, NaN
            words = ['N/A']
        result.append(words)
    # Get bigram and trigrams
    return bigram_trigram(result)
       
def bld_corpus(text, view=False, tfidf_thresh=0.20):
    '''
    Builds corpus (int mappings between each unique word ID across all documents
        and frequency of that word in a particular document) and bag of words 
        dictionary of actual words, returning both objects. Also returns series
        of bag of words to append to original df for comparison of orig fault
        to cleaned fault.
    :text: list of lists, containing all parsed and cleaned word for each text field.
    :view: bool, view first 5 obs of corpus
    :tfidf_thresh: tf-idf ratio minimum cut-point. Keep words above this ratio in corpus.
    :returns: pd.Series of processed words, corpus, and bag of words
    '''
    # Keep words (tokens) appearing at least 20 times
    frequency = defaultdict(int) # Get freq of each word across all documents
    for indiv_doc in text:
        for token in indiv_doc:
            frequency[token] += 1
    text = [[token for token in indiv_doc if frequency[token] > 100] for indiv_doc in text]
    #id2word.filter_extremes(no_below=30, no_above=len(text)*0.05) # words must be present 30+ times across faults, but present in <5% of all faults

    id2word = corpora.Dictionary(text) # dictionary (aka bag of words) of words for mapping
    corpus = [id2word.doc2bow(token) for token in text] # mapping of unique words to frequencies
    model = TfidfModel(corpus, normalize=True) # fit tf-idf model, normalize to account for various document lengths (shouldn't matter much with these data)
    vector = model[corpus] # get individual tf-idf values for all words in corpus

    # Iterate through each document getting tf-idf scores on each word updating with each doc
    # This yields final tf-idf
    scores = {}
    for indiv_doc in vector:
        for id, value in indiv_doc:
            token = id2word.get(id)
            scores[token] = value

    #pd.DataFrame.from_dict(scores, orient='index').reset_index()[0].describe()  # moments of final tf-idf values

    # Remove words with final tf-idf scores below threshold
    thresh_words = []
    trash_words = []
    for key, value in scores.items():
        if value > tfidf_thresh:
            thresh_words.append(key)
        else:
            trash_words.append(key)
    text = [[token for token in indiv_doc if token in thresh_words] for indiv_doc in text]
    id2word = corpora.Dictionary(text) # respecify bag of words and corpus
    corpus = [id2word.doc2bow(token) for token in text]

    if view:
        print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:4]])

    # Convert bag of words dictionary to series
    s = pd.Series()
    for nr in range(0, len(text)): # for each list of lists
        row = pd.Series(', '.join(str(x) for x in text[nr] if x in thresh_words)) # elements in sublist, keep if in bag of words dictionary
        s = s.append(row, ignore_index=True)
    s.reset_index(drop=True, inplace=True)
    return s, corpus, id2word

def fault_duration(df):
    '''
    Creates fault start and end columns in datetime format. Outputs Pandas 
        dataframe with these additional columns.
    :df: Pandas dataframe
    '''
    d = df.copy()
    d.loc[d['FTIME'].isnull(), 'FTIME'] = '08:00' # assign missing FTIME to presumed start time of workday
    d.loc[d['FTIME']=='00:00', 'FTIME'] = '08:00'
    d.loc[d['CTIME']=='00:00', 'CTIME'] = '17:00'
    
    d['fault_bgn'] = d['FDATE'].astype(str) + ' ' + d['FTIME'].astype(str)
    d['fault_end'] = d['CDATE'].astype(str) + ' ' + d['CTIME'].astype(str)
    for x in ['fault_bgn', 'fault_end']:
        d[x] = pd.to_datetime(d[x], format ='%Y-%m-%d %H:%M', errors='coerce')

    # Fault duration in hours
    d['fault_dur'] = (d['fault_end'] - d['fault_bgn']).dt.total_seconds().div(60).div(60).round(2) # rounded to 2 decimal places

   # for faults that apparently ended same day before fault opened, reverse the two
    temp = d[['fault_bgn', 'fault_end']]
    d.loc[d.fault_dur<0, 'fault_end'] = temp['fault_bgn']
    d.loc[d.fault_dur<0, 'fault_bgn'] = temp['fault_end']
    d['fault_dur'] = (d['fault_end'] - d['fault_bgn']).dt.total_seconds().div(60).div(60).round(2) # rounded to 2 decimal places

    # Rearrange columns, drop original columns since no longer necessary
    cols = ['SERNO', 'fault_bgn', 'fault_end', 'fault_dur', 'PHASE', 'FLT_HRS', 'CWUC', 'FAULT', 'phase_fault', 'phase1', 'phase2']
    d = d[cols]
    return d

def run_lda_model(corpus, id2word, texts, num_topics=None):
    '''
    Function estimates Latent Dirichlet Allocation model using desired number of latent topics.
    Returns model results as wrapper, pandas df of top keywords per topic, a wrapper of model coherence,
    and a wrapper of model perplexity.
    :corpus: list of lists of int mappings between each unique word ID across all documents
        and frequency of that word in a particular document.
    :id2word: gensim dictionary wrapper containing all unique words across all documents
    :texts: list of lists, containing all parsed and cleaned word for each text field.
    :num_topics: int, number of topics specified
    '''

    print("\rEstimating model with {} topics".format(num_topics))
    model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100,
                     update_every=1, chunksize=100, passes=100, alpha='auto',
                     per_word_topics=True, minimum_probability=0.05)

    # Top keywords for each topic
    keywords = pd.DataFrame(data={'topic_nr': [], 'keyword': [], 'weight': []})
    for top_nr in range(0, model.get_topics().shape[0]):
        for item in sorted(model.show_topic(top_nr), key=lambda x: (x[1]), reverse=True)[:4]:
            keywords = keywords.append({'topic_nr': int(top_nr), 'keyword': item[0], 'weight': item[1]},
                                       ignore_index=True)

    # Coherece at this particular number of topics, higher the better
    try:
        coherence = CoherenceModel(model=model, texts=texts, dictionary=id2word,
                               coherence='c_v').get_coherence().round(4)
    except (BrokenPipeError, OSError) as error: # ipython seems to break, command line works
        coherence = None

    # Perplexity - lower the better
    perplexity = model.log_perplexity(corpus).round(4)

    return model, keywords, coherence, perplexity

def dominant_topic(model, corpus, texts):
    '''
    Function outputs dictionary of x number of dataframes where x is the number of
        various LDA models each estimated with a different number of topics.
    :model_dict: dictionary of LDA model estimates
    :corpus: list of lists of int mappings between each unique word ID across all documents
        and frequency of that word in a particular document.
    :text: list of lists, containing all parsed and cleaned word for each text field.
    '''

    # Initialize
    dom = pd.DataFrame(data={'dom_topic': [], 'contrib': []})  # categories of individual LDA model

    for z in range(0, len(texts)):
        try:
            top = sorted(model[corpus[z]][0], key=lambda x: (x[1]), reverse=True)[0]  # Get top category for each obs
            dom = dom.append({'dom_topic': top[0], 'contrib': top[1]}, ignore_index=True)
        except IndexError:
            dom = dom.append({'dom_topic': np.NaN, 'contrib': np.NaN}, ignore_index=True) # no dominant topic, did not surpass minimum_probability threshold
    return dom

def diagnostics(df, dom_topics, view_results=False):
    '''
    Iterating through models with differing numbers of topics, estimates a series of OLS
        regressions to obtain fit statistics (AIC, BIC, MSE). Does the same for functional
        group categories and an empty model with just the intercept.
    :df: Pandas dataframe
    :dom_topics: dictionary of dataframes of length df.shape[0] with a column for dominant category
        of each fault and the contribution of that dominant category.
    '''
    mse_dict = {}
    aic_dict = {}
    bic_dict = {}
    funct_group = {}
    mean_dict = {}
    
    # Iterate through each model with diff number of topics
    for z in dom_topics.keys():
        d = df.copy()
        d = pd.concat([d, dom_topics[z]], axis=1) # concat column to original df
        X = pd.DataFrame(OneHotEncoder().fit_transform(d[['dom_topic']]).toarray()[:,1:]) # omit first categ as ref
        X = sm.add_constant(X) # add constant
        y = np.array(d['fault_dur'])
        results = sm.OLS(y, X, missing='drop').fit()
        if view_results:
            print("\nResults using {} topics:\n\n".format(z))
            print(results.summary())
        mse_dict[z] = np.log(results.mse_model)
        aic_dict[z] = results.aic
        bic_dict[z] = results.bic
    
    # Compare to functional group codes
    d = df[['fault_dur', 'CWUC']].copy().loc[df['CWUC'].notnull()]
    y = np.array(d['fault_dur'])
    X = pd.DataFrame(OneHotEncoder().fit_transform(d[['CWUC']].astype(int)).toarray())
    results = sm.OLS(y, X.iloc[:,:-1], missing='drop').fit()
    if view_results:
        print("\nResults using functional categories:\n\n")
        print(results.summary())
    funct_group['mse'] = np.log(results.mse_model)
    funct_group['aic'] = results.aic
    funct_group['bic'] = results.bic
    
    # Mean of outcome variable, BIC/AIC score
    d = df[['fault_dur']].copy()
    y = np.array(d['fault_dur'])
    X = np.ones_like(y)
    results = sm.OLS(y,X, hasconst=False).fit()
    results.summary()
    mean_dict['aic'] = results.aic
    mean_dict['bic'] = results.bic
    
    return mse_dict, aic_dict, bic_dict, funct_group, mean_dict

def uniq_categ(dictionary, cat_col):
    '''
    Function iterates over a particular column (cat_col) in a dictionary of
        Pandas dataframes, obtaining number of unique categories in that
        column indexed by its key. Used to get number of unique "best" topics 
        as a function of number of topics for a particular LDA model.
    :dictionary: dictionary of Pandas dataframes
    :cat_col: str column name of topic categories
    '''
    new_dict = {} # initialize
    
    for key in dictionary.keys(): # iterate over keys
        new_dict[key] = len(dictionary[key][cat_col].unique())
    return new_dict

def main(df, sample=False, num_topics=None):
    '''
    Main function to either iteratively fit LDA models with varying numbers of topics
        or to run just one LDA model. Function also builds corpus, bag of words dictionary,
        and creates new columns in original dataframe. Returns modified dataframe of original
        (d), list of lists of cleaned text documents (doc), corpus (corpus), bag of words
        dictionary (id2word), dictionary of model results (models) or single model results 
        (model), a dictionary of coherence scores per model (coherences), a dictionary
        of model perplexities (perplexities), a pandas df(s) of each fault's dominant topic
        (an integer) within a dictionary, and Pandas df(s) within a dictionary of topic words
        corresponding to the dominant topic numbers.
    :df: Pandas dataframe
    :sample: bool, default False. Specifies whether to iteratively fit many LDA models
    :num_topics: int, for full sample specify "best" number of topics according to sample
        diagnostics
    '''
    d = df.copy()

    # Fault duration
    d = fault_duration(d)
    
    # Clean fault fields, get list of lists, one per fault field
    doc = process(d['FAULT'])
    
    # Build corpus & bag of words dictionary
    d['fault_clean'], corpus, id2word = bld_corpus(doc)

    models = {}
    topic_words = {}
    coherences = {}
    perplexities = {}
    dom_topics = {}

    if sample: # iterations across models with different numbers of topics if (small) sample True

        for nt in range(5, 105, 5):
            # Get dictionaries of models, coherence and perplexity scores by number of topics
            models[nt], topic_words[nt], coherences[nt], perplexities[nt] = run_lda_model(corpus=corpus, id2word=id2word, texts=doc, num_topics=nt)
            dom_topics[nt] = dominant_topic(model=models[nt], corpus=corpus, texts=doc)   # Get dominant topics

    else: # full sample, just one model
        
        assert(num_topics is not None)
        models[num_topics], topic_words[num_topics], coherences[num_topics], perplexities[num_topics] = run_lda_model(corpus=corpus, id2word=id2word, texts=doc, num_topics=num_topics)
        dom_topics[num_topics] = dominant_topic(model=models[num_topics], corpus=corpus, texts=doc)   # Get dominant topics
        
    return d, doc, corpus, id2word, models, topic_words, coherences, perplexities, dom_topics
    
def train_valid_sets(df, train_frac=0.10, valid_frac=0.05):
    '''
    Generates mutually-exclusive train, test and cross-validation panel subsample 
        of helicopter serial numbers of desired size.
    :df: Pandas dataframe
    :train_frac: float, fraction of original sample desired for training set
    :test_frac: float, fraction of original sample desired for test set
    :valid_frac: float, fraction of original sample desired for validation set
    '''
    assert(train_frac+valid_frac <= 1), "\rTrain and validation fractions cannot sum >1!"
    trsamp = df['SERNO'].drop_duplicates().sample(frac=train_frac, random_state=123).tolist()
    train = df.loc[df.SERNO.isin(trsamp)].reset_index(drop=True)
    vsamp = df['SERNO'].loc[~df.SERNO.isin(trsamp)].drop_duplicates().sample(frac=valid_frac, random_state=123).tolist()
    valid = df.loc[df.SERNO.isin(vsamp)].reset_index(drop=True)
    return train, valid

def read_faults():
    '''
    Reads fault data, returns this as Pandas df, as well as string directory path
    '''

    try:
        directory = r'C:\Users\jking\Documents\FTS2B\data'
    except:
        directory = r'X:\Pechacek\ARNG Aircraft Readiness\Data\Processed\Master fault data intermediate\NGB Blackhawk'
    df = pd.read_csv(os.path.join(directory, 'NMC_faults.csv'))
    df.drop(columns=['ACTION'], inplace=True)
    return df, directory

def iterate_sample():
    '''
    Iteratively estimates LDA model with various numbers of topics to find which one is the 'best' number of topics.
    :return: Outputs graphics of numbers of topics and a json of diagnostics
    '''

    # Panel of serial numbers
    train, valid = train_valid_sets(f)

    # Get all the things
    tr, doc, corpus, id2word, models, topic_words, coherences, perplexities, dom_topics = main(train, sample=True)

    # Save model to disk
    with open(os.path.join(directory, 'sample_models.pickle'), 'wb') as p:
        pickle.dump(models, p)

    # Plot coherence values by number of topics - higher better
    plt.plot(list(coherences.keys()), list(coherences.values()))
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.savefig(os.path.join(directory + '\\summary_stats', 'coherence_numtopics.png'), dpi=250)
    plt.close()

    # Plot perplexity by number of topics - lower better
    plt.plot(list(perplexities.keys()), list(perplexities.values()))
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity")
    plt.savefig(os.path.join(directory + '\\summary_stats', 'perplexity_numtopics.png'), dpi=250)
    plt.close()

    # Predictive power of num topics vs functional groups
    print("\nEstimating diagnostics...")
    mse, aic, bic, fg, mean_dict = diagnostics(tr, dom_topics=dom_topics)

    fg_mse = []
    fg_aic = []
    fg_bic = []
    mean_aic = []
    mean_bic = []

    for x in mse.keys():
        fg_mse.append(fg['mse'])
        fg_aic.append(fg['aic'])
        fg_bic.append(fg['bic'])
        mean_aic.append(mean_dict['aic'])
        mean_bic.append(mean_dict['bic'])

    # Mean squared log error - lower better
    plt.plot(list(mse.keys()), list(mse.values()), label='LDA')
    plt.plot(list(mse.keys()), fg_mse, label='fun cat')
    plt.legend(loc='best')
    plt.xlabel("Num Topics")
    plt.ylabel("Mean Sq Log Error")
    plt.savefig(os.path.join(directory + '\\summary_stats', 'mse_fun_numtopics.png'), dpi=250)
    plt.close()
    # Without functional categories
    plt.plot(list(mse.keys()), list(mse.values()))
    plt.xlabel("Num Topics")
    plt.ylabel("Mean Sq Log Error")
    plt.savefig(os.path.join(directory + '\\summary_stats', 'mse_numtopics.png'), dpi=250)
    plt.close()

    # AIC - lower better
    plt.plot(list(aic.keys()), list(aic.values()), label='LDA')
    plt.plot(list(aic.keys()), fg_aic, label='fun cat')
    plt.legend(loc='best')
    plt.xlabel("Num Topics")
    plt.savefig(os.path.join(directory + '\\summary_stats', 'aic_fun_numtopics.png'), dpi=250)
    plt.close()
    # Without functional categories, including mean aic
    plt.plot(list(aic.keys()), list(aic.values()), label='LDA')
    plt.plot(list(aic.keys()), mean_aic, label='mean')
    plt.legend(loc='best')
    plt.xlabel("Num Topics")
    plt.savefig(os.path.join(directory + '\\summary_stats', 'aic_numtopics.png'), dpi=250)
    plt.close()

    # BIC - lower better
    plt.plot(list(bic.keys()), list(bic.values()), label='LDA')
    plt.plot(list(bic.keys()), fg_bic, label='fun cat')
    plt.legend(loc='best')
    plt.xlabel("Num Topics")
    plt.savefig(os.path.join(directory + '\\summary_stats', 'bic_fun_numtopics.png'), dpi=250)
    plt.close()
    # Without functional categories
    plt.plot(list(bic.keys()), list(bic.values()), label='LDA')
    plt.plot(list(bic.keys()), mean_bic, label='mean')
    plt.legend(loc='best')
    plt.xlabel("Num Topics")
    plt.savefig(os.path.join(directory + '\\summary_stats', 'bic_numtopics.png'), dpi=250)
    plt.close()

    del fg_mse, fg_aic, fg_bic, mean_bic, mean_aic  # clear clutter

    # Save sample diagnostics
    diag = {}
    diag['coherences'] = coherences
    diag['perplexities'] = perplexities
    diag['mse'] = mse
    diag['bic'] = bic
    diag['aic'] = aic
    diag['funct_group'] = fg
    diag['num_uniq_topics'] = uniq_categ(dictionary=dom_topics,
                                         cat_col='dom_topic')  # Unique number of "best/dominant" topics across models
    diag['best_aic'] = min(aic, key=aic.get)
    diag['best_bic'] = min(bic, key=bic.get)

    with open(os.path.join(directory + '\\summary_stats', 'model_diagnostics.json'), 'w') as j:
        json.dump(diag, j)

    # Plot number of unique best topics as function total topics
    plt.plot(list(diag['num_uniq_topics'].keys()), list(diag['num_uniq_topics'].values()))
    plt.xlabel("Num Topics")
    plt.ylabel("Dominant (best) topics")
    plt.savefig(os.path.join(directory + '\\summary_stats', 'domtopics_numtopics.png'), dpi=250)
    plt.close()

    print("\nDone iterating through sample, see output graphics for diagnostics")

def population(num_topics):
    '''

    :param num_topics: int, number of topics to apply to population dataset
    :return: Outputs CSV file with each fault's dominant topic and keywords from bag of words what that dominant
        topic numbeer corresponds to
    '''

    # Get all the things
    f_, doc_, corpus_, id2word_, models_, topic_words_, coherences_, perplexities_, dom_topics_ = main(f, sample=False,
                                                                                                       num_topics=num_topics)

    # Save model to disk
    with open(os.path.join(directory, 'final_model.pickle'), 'wb') as p:
        pickle.dump(models_, p)

    # Concat dominant category to each fault
    f_ = pd.concat([f_, dom_topics_[num_topics]['dom_topic']], axis=1)

    # Attach topic keywords
    kw = topic_words_[num_topics][['topic_nr', 'keyword']].set_index('topic_nr')
    kw['keywords'] = kw.groupby("topic_nr")['keyword'].transform(lambda x: ', '.join(x));
    del kw['keyword']
    kw = kw.reset_index(drop=False).drop_duplicates()
    f_ = f_.merge(kw, left_on='dom_topic', right_on='topic_nr', how='left');
    del f_['topic_nr']

    # Output
    f_.to_csv(os.path.join(directory, 'NMC_faults_lda.csv'), index=False)

    print("\nDone with population dataset")


if __name__ == '__main__':

    # Get faults
    f, directory = read_faults()

    # Calls externally-created json to specify configurations of this script
    with open('lda_config.json') as j:
        config = json.load(j)

    # Run iteratively on sample or once on population
    if config['iterate_sample'] == True:
        iterate_sample()
    else:
        population(config['num_topics'])