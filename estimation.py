import lmfit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.stats.distributions import t as t_distribution
import utils
import splines

def load_params_from_excel_file(excelfile, return_bounds=False):
    params_df = pd.read_excel(excelfile)
    params = {}
    for (model, variable), subdf in params_df.groupby(['model', 'variable']):
        if model not in params:
            params[model] = {}
        if subdf.shape[0] == 1:
            params[model][variable] = subdf.iloc[0]['value'] #scalar
        elif subdf['y'].notnull().all():
            pvt = subdf.pivot_table(index='x', columns='y', values='value')
            params[model][variable] = {
                'x': pvt.index.tolist(),
                'y': pvt.columns.tolist(),
                'z': pvt.values.tolist(),
                'type': subdf['type'].iloc[0],               
            }
        else:
            params[model][variable] = {
                'x': subdf['x'].tolist(),
                'y': subdf['value'].tolist(),
                'type': subdf['type'].iloc[0],
            }
            
    if not return_bounds:
        return params
            
    fit_bound = {}
    for (model, variable), subdf in params_df.groupby(['model', 'variable']):
        if model not in fit_bound:
            fit_bound[model] = {}
        if subdf.shape[0] == 1:
            fit_bound[model][variable] = {
                'value': subdf.iloc[0]['value'],
                'freeze': subdf.iloc[0].fillna(0)['freeze'],
                'lowerbound': subdf.iloc[0].fillna(-np.inf)['lowerbound'],
                'upperbound': subdf.iloc[0].fillna(np.inf)['upperbound'],
            }
        else:
            fit_bound[model][variable] = {
                'x': subdf['x'].tolist(),
                'y': subdf['value'].tolist(),
                'freeze': subdf['freeze'].fillna(0).tolist(),
                'lowerbound': subdf['lowerbound'].fillna(-np.inf).tolist(),
                'upperbound': subdf['upperbound'].fillna(np.inf).tolist(),
            }

    return params, fit_bound

def callback(params, niter, resid):        
    print ("%03d:" % niter + ' | '.join(['%.4f' % p.value for p in params.values()]) + '| Resid:%.4f' % np.dot(resid, resid))

class Estimator:
    """Nonlinear model calibration tool."""

    def __init__(self, model, params, data, actual=None, weight=None, fit_bounds=None, precalc_burnout=True):
        """@model: model instance
        @params: complete parameter set (dict)
        @data: pd.DataFrame
        @actual: str, or pd.Series of the same length
        @weight: str, or pd.Series of the same length
        """

        self.model = model
        self.params = params
        self.data = data

        #calculate burnout and used it in the future without recalc each iteration
        self.precalc_burnout = precalc_burnout
        if self.precalc_burnout:
            self.model.reset_data(dataframe=self.data, calc_burnout=True) 

        if isinstance(actual, str) and actual in data:
            self.actual = data[actual]
        else:
            self.actual = actual

        if isinstance(weight, str) and weight in data:
            self.weight = data[weight]
        else:
            self.weight = weight

        self.fit_params = lmfit.Parameters()
        self.fit_config = OrderedDict()
        self.fit_bounds = fit_bounds

    def residual(self, fit_params=None):
        fit_params = fit_params or self.fit_params        
        self.update_model_params(fit_params)
        
        if self.precalc_burnout:
            #don't pass in data again, which will reset model data
            resid = self.model.project(calc_from_orig=False) - self.actual 
        else:
            resid = self.model.project(self.data) - self.actual
        if self.weight is not None:
            resid *= np.sqrt(self.weight)
        return resid.fillna(0)

    def add_fit_params(self, fit_params_path, spline_dim=1, **kwargs):
        """Add fitting parameters path (`fit_params_path` in self.params)
        e.g., fit_params_path = (refi, incentive)
        """
        param_val = self.params
        for p in fit_params_path:
            param_val = param_val[p]
        
        param_bounds = self.fit_bounds
        for p in fit_params_path:
            param_bounds = param_bounds[p]

        if spline_dim == 0:
            name = '_'.join(fit_params_path)
            self.fit_params.add(name=name, value=param_val, 
                                vary=not bool(param_bounds['freeze']), 
                                min=param_bounds['lowerbound'] if pd.notnull(param_bounds['lowerbound']) else None, 
                                max=param_bounds['upperbound'] if pd.notnull(param_bounds['upperbound']) else None
            )
            config = self.fit_config[name] = OrderedDict()
            config['spline_dim'] = 0

        elif spline_dim == 1:
            name = '_'.join(fit_params_path)
            config = self.fit_config[name] = OrderedDict()
            config['spline_dim'] = 1
            config['spline_shape'] = len(param_val['x'])
            for i in range(len(param_val['x'])):
                self.fit_params.add(name=name + '_{:02d}'.format(i), value=param_val['y'][i],
                                   vary=not bool(param_bounds['freeze'][i]), 
                                   min=param_bounds['lowerbound'][i] if pd.notnull(param_bounds['lowerbound'][i]) else None, 
                                   max=param_bounds['upperbound'][i] if pd.notnull(param_bounds['upperbound'][i]) else None                
                )

        else:
            raise NotImplementedError("spline_dim > 1 not implemented yet")

    def update_model_params(self, fit_params=None):
        fit_params = fit_params or self.fit_params

        for name, config in self.fit_config.items():

            fit_params_path = name.split('_')
            if config['spline_dim'] == 0:
                target = self.params
                for p in fit_params_path[:-1]:
                    target = target[p]
                target[fit_params_path[-1]] = fit_params[name].value

            elif config['spline_dim'] == 1:
                target = self.params
                for p in fit_params_path:
                    target = target[p]
                target['y'] = [ fit_params[name + '_{:02d}'.format(i)].value for i in range(config['spline_shape'])]

        self.model.update_params(self.params)

    def fit(self, method='leastsq', params=None, iter_cb=callback):
        """method : str, optional
            Name of the fitting method to use.
            One of:
            'leastsq'                -    Levenberg-Marquardt (default)
            'nelder'                 -    Nelder-Mead
            'lbfgsb'                 -    L-BFGS-B
            'powell'                 -    Powell
            'cg'                     -    Conjugate-Gradient
            'newton'                 -    Newton-CG
            'cobyla'                 -    Cobyla
            'tnc'                    -    Truncate Newton
            'trust-ncg'              -    Trust Newton-CGn
            'dogleg'                 -    Dogleg
            'slsqp'                  -    Sequential Linear Squares Programming
            'differential_evolution' -    differential evolution

            params : Parameters, optional
            parameters to use as starting values"""

        self.minimizer = lmfit.Minimizer(self.residual, self.fit_params, iter_cb=iter_cb)
        self.minimizer.minimize(method=method, params=params)
        self.result = self.minimizer.result
        if hasattr(self.result, 'lmdif_message'):
            print(self.result.lmdif_message)
            print('=' * 50)
        lmfit.report_fit(self.result)
        self.fit_params = self.result.params
        self.update_model_params()
        #if self.result.errorbars:
        #    self._calc_ci(alpha=kwargs.get('alpha', 0.95))

    def mva(self, data=None):
        mvadata = data or self.data
        if self.precalc_burnout:
            proj = self.model.project(data, get_total_only=False, calc_from_orig=False)
        else:
            proj = self.model.project(mvadata, get_total_only=False)
        proj.index = mvadata.index
        return mvadata.join(proj, rsuffix='_model')

    def _calc_ci(self, alpha=0.95):
        result = self.result
        nobs = result.residual.size - np.isnan(result.residual).sum()
        if nobs < result.ndata:
            print("Missing values were generated as a result of operations.")
        dof = max(0, nobs - result.nvarys)
        tval = t_distribution.ppf(0.5 + alpha / 2, dof)
        cov_x = result.covar
        if self.minimizer.scale_covar:
            cov_x = result.covar / result.redchi

        ssq = (result.residual**2).sum()
        stderr = result.stderr = np.sqrt(ssq * cov_x.diagonal() / dof)
        x = result.x = np.asarray(
            [p.value for p in result.params.values() if p.vary])
        ci = result.ci = np.hstack([x - tval * stderr,
                                    x + tval * stderr]).reshape(-1, 2, order='F')
        result.fit_stats = pd.DataFrame(np.hstack([np.vstack([x, stderr]).T, ci]),
                                        columns=[
                                            'Estimate', 'StdErr',
                                            '{:.0%} CI Lo'.format(alpha),
                                            '{:.0%} CI Hi'.format(alpha)
                                        ])

    def report_fit(self, ci_alpha=0.95):
        self._calc_ci(alpha=ci_alpha)
        return self.result.fit_stats


def plot_by_buckets(mva, bucket_variable='fico', bucket_size=20, wavgvars=['smm', 'smm_model'], groupby=None, wgt='schbal', thresh=1e9, **kwargs):
    
    mva['bkt'] = (mva[bucket_variable] / bucket_size).round() * bucket_size
    agg = utils.aggregate(mva, groupby=([groupby] if groupby else []) + ['bkt'], wgt=wgt, wavgvars=wavgvars, sumvars=[wgt], as_index=False)
    agg.set_index('bkt', inplace=True)
    
    if 'smm' in agg:
        agg['CPR'] = utils.smm2cpr(agg['smm'])
    if 'smm_model' in agg:
        agg['CPR_Model'] = utils.smm2cpr(agg['smm_model'])
    if 'curt_smm' in agg:
        agg['CPR_Curtail'] =  utils.smm2cpr(agg['curt_smm'])
    cprcols = [c for c in agg if 'CPR' in c]
    agg = agg[agg[wgt] > thresh]
    
    if groupby:
        grpd = agg.groupby(groupby)
        fig, axes = plt.subplots(1, grpd.ngroups, figsize=(6*grpd.ngroups, 4))
        n = 0
        for g, subdf in grpd:
            ax = subdf[cprcols].plot(ax=axes[n])
            subdf[wgt].plot(kind='area', secondary_y=True, alpha=0.3, lw=0, ax=ax, title=groupby+"="+str(g))
            ax.set_xlabel(bucket_variable)
            n += 1
    else:
        ax = agg[cprcols].plot()
        agg[wgt].plot(kind='area', secondary_y=True, alpha=0.3, lw=0, ax=ax)

def plot_spline(p, **kwargs):
    if p['type'] == 'linear':
        sp = splines.linear_spline(p['x'], p['y'])
    elif p['type'] == 'cubic':
        sp = splines.monotone_cubic_spline(p['x'], p['y'])
    elif p['type'] == 'bilinear':
        sp = splines.bilinear_spline(p['x'], p['y'], p['z'])
    elif p['type'] == 'bicubic':
        sp = splines.bicubic_spline(p['x'], p['y'], p['z'])
    else:
        raise ValueError("type {} not recognized".format(p['type']))    
    sp.plot(**kwargs)
