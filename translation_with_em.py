# -*- coding: utf-8 -*-
"""
@author: ryan.shea
"""

# add NULLs to english sentences and get translation probs based on word alignments 
def init_vars():
    with open('training_data.txt', 'r', encoding='utf8') as inp:
        prob_ef={}
        count_ef={}
        total_f={}
        
        chi_sents=[]
        eng_sents=[]
        
        for line in inp.readlines():
            chinese, english=line.split('\t')
            
            chinese=chinese.split()
            english=english.split()
            english.insert(0,'NULL')
            
            chi_sents.append(chinese)
            eng_sents.append(english)
            
            for c in chinese:
                for e in english:
                    prob_ef[(c,e)]=.5
                    count_ef[(c,e)]=0
                    total_f[c]=0
    return prob_ef, count_ef, total_f, chi_sents, eng_sents

'EM Algorithm'

# implement the EM algorithm for translation
def em_algorithm(epochs):
    prob_ef, count_ef, total_f, chi_sents, eng_sents = init_vars()
    
    s_total={}
    for epoch in range(epochs):

        count_ef=count_ef.fromkeys(count_ef, 0)
        total_f=total_f.fromkeys(total_f, 0)
        'E step'
        for eng,chi in zip(eng_sents,chi_sents):
            #get count denominator (sum of all probabilities for e given different c)
            #den=\sum_{i=0}^{len(c)} t(e|c_i) <-- LaTeX format
            for e in eng:
                s_total[e]=0
                for c in chi:
                    s_total[e]+= prob_ef[(c,e)]
            #get counts
            #count_ef will be the numerator for the next step
            #total_f is the denominator for the next step (sum of of all new probabilities for e given different c)
            #total_f=\sum_{i=0}^{len(e)} t(e|c_i) <-- LaTeX format
            for e in eng:
                for c in chi:
                    count_ef[(c,e)]+= prob_ef[(c,e)]/s_total[e]
                    total_f[c]+=prob_ef[(c,e)]/s_total[e]
        'M step'
        for eng,chi in zip(eng_sents,chi_sents):
            for c in chi:
                for e in eng:
                    prob_ef[(c,e)]=count_ef[(c,e)]/total_f[c]

    return prob_ef

trans_probs=em_algorithm(4)

# return the highest probability translation
def get_best_eng_translation(word, probs):
    sub_dict={key:value for key,value in probs.items() if key[1]==word}
    top_key=max(sub_dict, key=sub_dict.get)
    return [(word, top_key[0]), probs[top_key]]

# last two are incorrect, may be b/c of EM vs SGD
words=['jedi','droid', 'force', 'midi-chlorians', 'yousa']

translations=[get_best_eng_translation(word, trans_probs) for word in words]




