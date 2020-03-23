import numpy as np
import scipy.stats as sp
import statsmodels.api as smapi
import pandas as pd
import matplotlib.pyplot as plt

def metro_hast(f_pdf, g_pdf, g_gen,ndraws = 100,init_value = None, *args,**kwargs):
    ### function to run the general metropolis-hastings algorithm
    ### inputs
    ### f_pdf: a function that returns the pdf f(x)
    ### g_pdf: a function that returns the candidate value g(x_1|x_0)
    ### g_gen: a function that returns a random variable drawn from g(x_1|x_0)
    ### n_samples: the number of samples we want to generate from
    
    # following the text, initialize the initial samples
    # the first one is our initial
    sim_samples = g_gen(ndraws+1,*args,**kwargs)
    
    # allow for an initialization value 
    if init_value is not None:
        sim_samples[0]=init_value
        
    # we pull all the uniform variables we need upfront
    unif_rv = sp.uniform.rvs(size=ndraws)
    acceptance = np.zeros(ndraws)
    accep_prob = lambda x1,x0: f_pdf(x1,x0,*args,**kwargs)*g_pdf(x0,x1,*args,**kwargs)/(f_pdf(x0,x1,*args,**kwargs)*g_pdf(x1,x0,*args,**kwargs))
    
    for i_sim in range(1,len(sim_samples+1)):
        y = sim_samples[i_sim]
        x = sim_samples[i_sim-1]
        p_acc = min(1,accep_prob(y,x))
        sim_samples[i_sim] = x+(y-x)*(unif_rv[i_sim-1]<=p_acc)
        if unif_rv[i_sim-1]<=p_acc:
            acceptance[i_sim-1]=1
                
    results = {'samples':sim_samples,'accept_rate':acceptance.sum()/ndraws}
    return results

def accept_reject(f_pdf,g_pdf,g_gen,M=1,ndraws=100,*args,**kwargs):
    # function to do the accept reject method
    # f_pdf is desired pdf to sample from
    # g_pdf is the candidate pdf function
    # g_geg is the function to generate a candidate value from distribution g_gen - we may have to use exotic densities
    # M is the scalar such that M>=sup(f(x)/g(x)) for all x
    
    sim_values = np.zeros(ndraws,)
    sim_success = 0 # python index starts at 0
    sim_idx = -1
    n_count=0
    
    while sim_success < ndraws:
        # draw a random value from g_gen
        cand_rv = g_gen(nobs=1,*args,**kwargs)
        # draw uniform random value bounded between 0 and M*cand_rand
        unif_rv = sp.uniform.rvs(loc=0,scale=g_pdf(cand_rv,*args,**kwargs)*M,size=1,)
        
        if unif_rv < f_pdf(cand_rv,):
            sim_success +=1
            sim_idx +=1
            sim_values[sim_idx]=cand_rv
        
        results={'samples':sim_values,'accept_rate':sim_success/ndraws} 
    return results   

