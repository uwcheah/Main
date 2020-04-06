import scipy as sp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime


# test data
# stupid quandl blowing up everything
def get_sample_data(*args,**kwargs):
    f_ = pd.ExcelFile('c:/users/nowuc/Code/data/VX1.xlsx')
    raw_data = f_.parse()
    dts  = pd.DatetimeIndex(np.flip(raw_data['Trade Date'].values))
    close_data = raw_data['Close']
    return pd.Series(index=dts,data=np.flip(close_data.values))
    
class EWMA(object):
    def __init__(self,*args,**kwargs):
        pass
    
    @classmethod
    def level_changes(cls,levs,tdelta=[1,2,3],asrate=False,*args,**kwargs):
        # code to compute a full dataframe of level changes/returns
        # tdelta should be an array of changes with default on just single period change
        
        level_changes = {}
        for td_ in tdelta:
            lev_chg_ = {}
            for s_ in range(td_):
                # this get the appropriate frequency data
                data = levs[s_::td_].fillna(method='pad')
                # get the raw change
                chg = data.subtract(data.shift(1))
                if asrate:
                    chg = chg.divide(data.shift(1))
                else:
                    pass
                # putting the changes together
                lev_chg_['m{0}'.format(s_)]=chg
            level_changes[td_]=pd.DataFrame(lev_chg_)
        
        results = pd.concat(level_changes,sort=True,names=('lag','dates'))
        results = results.stack().unstack(level='lag')
        results.index.names = ('dates','model')
        return results
    
    @classmethod
    def ewa_changes_by_col(cls,data_chg,halflife=5,min_periods=0,*args,**kwargs):
        
        ewm_chg = pd.DataFrame({col_: data_chg[col_].dropna(how='all').ewm(halflife=halflife,min_periods=min_periods,ignore_na=True).mean()
                                for col_ in data_chg})
        
        return ewm_chg
    
    @classmethod
    def predict(cls,data,halflife=5,min_period=0,tdelta=[1,],asrate=False,*args,**kwargs):
        pass
    
        
        
        
        
    
                
                
                