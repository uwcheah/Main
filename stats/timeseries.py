import scipy as sp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import quandl

sns.set_style('darkgrid')

# test data
# stupid quandl blowing up everything
def get_sample_data(*args,**kwargs):
    f_ = pd.ExcelFile('c:/users/nowuc/Code/data/VX1.xlsx')
    raw_data = f_.parse()
    dts  = pd.DatetimeIndex(np.flip(raw_data['Trade Date'].values))
    close_data = raw_data['Close']
    return pd.Series(index=dts,data=np.flip(close_data.values))

class STATS(object):
    @staticmethod
    def estimate_residuals(y,yhat,*args,**kwargs):
        resid = y.subtract(yhat).dropna()
        return resid
    
    @staticmethod
    def estimate_r2(y,yhat,*args,**kwargs):
        resid = STATS.estimate_residuals(y,yhat)
        y_all = pd.DataFrame({'y':y,'yhat':yhat,'resid':resid}).dropna(how='any',axis=0)
        y_var = y_all.var()
        r2 = y_var.divide(y_var['y'])
        return r2
    
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
                data = levs[s_::td_]
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
    def ewa_by_col(cls,data_chg,halflife=5,min_periods=0,*args,**kwargs):
        
        ewm_chg = pd.DataFrame({col_: data_chg[col_].dropna(how='all').ewm(halflife=halflife,min_periods=min_periods,ignore_na=True).mean()
                                for col_ in data_chg})
        
        return ewm_chg
    
    @classmethod
    def predicted_chgs(cls,levels,halflife=5,min_periods=0,tdelta=[1,],asrate=False,*args,**kwargs):
        # getting tdeltas and changes
        lev_chg = cls.level_changes(levels,tdelta=tdelta,asrate=asrate,*args,**kwargs)
        # cycling through lags and changes to caculate ewm
        chgs_ew = {}
        for td_ in lev_chg.columns.tolist():
            chgs = lev_chg[td_].unstack(level='model')
            chgs_ew_ = cls.ewa_by_col(chgs,halflife=halflife,min_periods=min_periods,*args,**kwargs)
            chgs_ew[td_]=chgs_ew_
        results = pd.concat(chgs_ew,sort=True).stack()
        results.index.names = ('lag','dates','model')
        results.swaplevel('dates','lag').swaplevel('model','lag')
        results = results.unstack(level='lag').unstack(level='model').dropna(how='all',axis=1)
        return results
    
    @classmethod
    def prediction_dts(cls,dtindex,p_interval=1):
        add_dts = dtindex[-1:]+pd.offsets.BDay(p_interval)
        dts = dtindex.append(add_dts)
        dts = dts.unique()
        return dts    
        
    @classmethod
    def predictions(cls,lev,chg,):
        # we add the values
        fitted_values = chg.add(lev,axis=0)
        # looping through and getting the dates and shifting
        def shift_pred(x_series,tshift):
            x_clean = x_series.dropna()
            x_pdts = cls.prediction_dts(x_clean.index,p_interval=tshift)
            x_pred = x_clean.reindex(index=x_pdts)
            # shift one back
            x_prediction = x_pred.shift(1)
            return x_prediction
        
        pred =pd.DataFrame({y:shift_pred(fitted_values[y].dropna(),y[0]) for y in fitted_values.columns.tolist()})
        preds = pred.stack().stack() 
        preds.index.names =('dates','model','interval')
        return preds
    
    @classmethod
    def plot_predictions(cls,predict_series,start_date,cval=None, *args,**kwargs):
        # get the prediction 
        pred_data = predict_series.truncate(before=start_date)
        pred_data.index.names = ('dates','model','interval')
        # getting the prediction d
        pred_dts = sorted(pred_data.index.unique(level=0))
        # getting the index of dates
        days_out = pd.Series(index=pred_data.index,data=np.int_(np.array([pred_dts.index(x[0]) for x in pred_data.index])))
        plot_frame = pred_data.index.to_frame()
        plot_frame['prediction']=predict_series
        plot_frame['days_out'] = days_out
        ax_ = sns.scatterplot(x='days_out',y='prediction',data=plot_frame,hue='interval')
        if cval is not None:
            ax_.hlines(cval,xmin=ax_.get_xlim()[0],xmax=ax_.get_xlim()[-1],linestyle='--',colors='gray')
        return ax_,plot_frame
    
        
    @classmethod
    def get_residuals(cls,raw_series,predictions,*args,**kwargs):
        """Getting residuals

        
        Arguments:
            raw_series {pandas series, dataframe} -- dafuq
            predictions {pandas series} -- dafuq
        """       
        a = {a:12,b:23,
             }
        pass
    
