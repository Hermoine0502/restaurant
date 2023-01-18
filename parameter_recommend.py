"""
@ Created on 2022-01-05

@ author: Charlie.Wei

@ purpose: The parameter recommendation in the MCF includes a single value and interval.
    
@ structure: 
    # libraries
    # user-defined class
        ## Event
        ## ValueRecommend
        ## RangeRecommend
"""


#region = libraries
import numpy as np
import pandas as pd
import pyswarms as ps
from scipy import stats
import random
#endregion

random.seed(66)
np.random.seed(66)

#region = user-defined-class
class Event:
    """
    Parameter encapsulation and foolproof mechanism.
    """
    def __init__(self,
                 list_control,
                 list_X_bound,
                 all_parameter,
                 PSO_options,
                 PSO_n_particles,
                 PSO_iters):
        """
        Initialize the Event class.
        
        Parameters
        -----
        list_control : list[str]
            All controllable parameters.
        list_X_bound : list[list[float], list[float]]
            A list of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound.
        all_parameter : bool
            whether to display all parameters.
        PSO_options : dict with keys {'c1', 'c2', 'w'}
            A dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        PSO_n_particles : int
            Number of particles in the swarm.
        PSO_iters : int
            Number of iterations.
        """
        # Parameter setting
        if not isinstance(list_control, list):
            raise TypeError("list_control must be a list.")
        else:
            self._list_control = list_control
        
        if not isinstance(list_X_bound, list):
            raise TypeError("list_X_bound must be a list.")
        if len(list_X_bound) != 2:
            raise ValueError("list_X_bound must contain only lower and upper.")
        if len(list_X_bound[0]) != len(list_X_bound[1]):
            raise ValueError("len() of bounds of list_X_bound must be the same.")
        else:
            self._list_X_bound = list_X_bound
            self._dim_X = len(self.list_X_bound[0])
        if not type(all_parameter) == bool:
            raise ValueError("type of all_parameter must be the bool.")
        else:
            self._all_parameter = all_parameter
        if not isinstance(PSO_options, dict):
            raise TypeError("PSO_options must be a dictionary.")
        else:
            self._PSO_options = PSO_options
        
        if not isinstance(PSO_n_particles, int):
            raise TypeError("PSO_n_particles must be an integer.")
        else:
            self._PSO_n_particles = PSO_n_particles
        
        if not isinstance(PSO_iters, int):
            raise TypeError("PSO_iters must be an integer.")
        else:
            self._PSO_iters = PSO_iters
    
    @property
    def list_control(self):
        return self._list_control
    
    @property
    def PSO_options(self):
        return self._PSO_options
    
    @property
    def dim_X(self):
        return self._dim_X

    @property
    def all_parameter(self):
        return self._all_parameter
    
    @property
    def list_X_bound(self):
        return self._list_X_bound
    
    @property
    def PSO_n_particles(self):
        return self._PSO_n_particles
        
    @property
    def PSO_iters(self):
        return self._PSO_iters


class ValueRecommend(Event):
    """
    Single value recommendation for parameters.
    """
    _eps = np.finfo(np.float64).eps
    def __init__(self,
                 list_control,
                 list_X_bound,
                 all_parameter=False,
                 PSO_options={'c1': 2, 'c2': 2, 'w':0.5},
                 PSO_n_particles=100,
                 PSO_iters=100):
        """
        Initialize the Event class.
        
        Parameters
        -----
        list_control : list[str]
            All controllable parameters.
        list_X_bound : list[list[float], list[float]]
            A list of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound.
        all_parameter : bool, default: False
            whether to display all parameters.
        PSO_options : dict with keys {'c1', 'c2', 'w'}, default: {'c1': 2, 'c2': 2, 'w':0.5}
            A dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        PSO_n_particles : int, default: 1000
            Number of particles in the swarm.
        PSO_iters : int, default: 100
            Number of iterations.
        """
        super().__init__(
            list_control=list_control,
            list_X_bound=list_X_bound,
            all_parameter=all_parameter,
            PSO_options=PSO_options,
            PSO_n_particles=PSO_n_particles,
            PSO_iters=PSO_iters)
    
    @property
    def data_X(self):
        return self._data_X
    
    @property
    def list_y_target(self):
        return self._list_y_target
    
    @property
    def model(self):
        return self._model
    
    @property
    def scaler(self):
        return self._scaler
    
    @property
    def PSO_position(self):
        return self._pos
    
    @property
    def PSO_cost(self):
        return self._cost

    @property
    def PSO_result(self):
        mask_control = self._data_X.columns.isin(self.list_control)
        return pd.DataFrame(self._pos[mask_control], 
                            index=self._data_X.columns[mask_control],
                            columns=['Recommendation'])
    
    @property
    def PSO_result_all(self):
        result = pd.DataFrame(
            self._pos,
            index=self._data_X.columns,
            columns=['Recommendation']
        )
        result['self_controled'] = result.index.isin(self.list_control)
        return result
    
    def _loss_target(self, x):
        """ 
        The part of loss function of PSO. 
        Calculate the distance to the target.
        """
        # x scaling
        x = self._scaler.transform(x)
        loss = (self._model.predict(x)-self.list_y_target) / (self.list_y_target+self._eps)
        return np.sum(abs(loss), axis=1)/len(self.list_y_target)
    
    def _loss_history(self, x):
        """
        The part of loss function of PSO.
        Calculate the distance to the history.
        """
        x_out_control = x[:, self._mask_out_control]
        loss = (x_out_control-np.array(self._data_X_history)) / (np.array(self._data_X_history)+self._eps)
        loss = np.sum(np.abs(loss), axis=1)/len(self._data_X_history)
        return loss
    
    def _pso_loss(self, x):
        """
        The loss function of PSO.
        
        Parameters
        -----
        x : array
            The n*m array where n is PSO_n_particles and m is dim(data_X).
                
        Returns
        -----
        loss: array
            The loss of each particle.
        """
        return 0.8*self._loss_target(x) + 0.2*self._loss_history(x)

    def optimize(self, data_X, list_y_target, model, scaler):
        """
        Use PSO to perform parameter optimization.
        
        Parameters
        -----
        data_X : pd.DataFrame
            The raw data.
        list_y_target : list[float]
            The targets of each y.
        model : object
            Trained model.
        scaler : object
            Trained scaler model.
            
        Returns
        -----
        PSO_result: pd.DataFrame
            The single value recommendation of each parameter in list_control.
        """
        # data_X setting
        if not isinstance(data_X, pd.DataFrame):
            raise TypeError("data_X must be a pd.DataFrame.")
        elif data_X.shape[1] != self.dim_X:
            raise ValueError("data_X must have the same dimension as X_bound.")
        else:
            self._data_X = data_X
        
        # list_y_target setting
        if not isinstance(list_y_target, list):
            raise TypeError("list_y_target must be a list.")
        else:
            self._list_y_target = np.array(list_y_target)
        
        # model setting
        if not isinstance(model, object):
            raise TypeError("model must be a object.")
        else:
            self._model = model
            
        # scaler setting
        if not isinstance(scaler, object):
            raise TypeError("scaler must be a object.")
        else:
            self._scaler = scaler
        
        # out of control setting
        self._list_out_control = list(set(self.data_X) - set(self.list_control))
        self._mask_out_control = self.data_X.columns.isin(self._list_out_control)
        
        # data_X scaling
        self._data_X_history = self._data_X.iloc[:, self._mask_out_control].mean(axis=0)
        
        # PSO run
        self._PSO_optimizer = ps.single.GlobalBestPSO(n_particles=self.PSO_n_particles,
                                                      dimensions=self.dim_X,
                                                      options=self.PSO_options,
                                                      bounds=self.list_X_bound)
        self._cost, self._pos = self._PSO_optimizer.optimize(self._pso_loss,
                                                             iters=self.PSO_iters,
                                                             verbose=False)
        if self.all_parameter:
            return self.PSO_result_all
        else:
            return self.PSO_result


class RangeRecommend(Event):
    """
    Interval recommendation for parameters.
    """
    def __init__(self,
                 list_control,
                 list_X_bound,
                 all_parameter=False,
                 PSO_options={'c1': 2, 'c2': 2, 'w':0.5},
                 PSO_n_particles=100,
                 PSO_iters=100):
        """
        Initialize the Event class.
        
        Parameters
        -----
        list_control : list[str]
            All controllable parameters.
        list_X_bound : list[list[float], list[float]]
            A list of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound.
        all_parameter : bool, default: False
            whether to display all parameters.
        PSO_options : dict with keys {'c1', 'c2', 'w'}, default: {'c1': 2, 'c2': 2, 'w':0.5}
            A dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        PSO_n_particles : int, default: 1000
            Number of particles in the swarm.
        PSO_iters : int, default: 100
            Number of iterations.
        """
        super().__init__(
            list_control=list_control,
            list_X_bound=list_X_bound,
            all_parameter=all_parameter,
            PSO_options=PSO_options,
            PSO_n_particles=PSO_n_particles,
            PSO_iters=PSO_iters)
    
    @property
    def data_X(self):
        return self._data_X
    
    @property
    def list_y_target(self):
        return self._list_y_target
    
    @property
    def model(self):
        return self._model
    
    @property
    def scaler(self):
        return self._scaler
    
    @property
    def iter_optimize(self):
        return self._iter_optimize
    
    @property
    def best_position(self):
        return self._best_position
    
    @property
    def best_cost(self):
        return self._best_cost
    
    @property
    def result(self):
        return self._df_resulf
    
    def _data_encapsulation(self, data_X, list_y_target, model, scaler):
        """
        Data encapsulation and foolproof mechanism.
        """
        # data_X setting
        if not isinstance(data_X, pd.DataFrame):
            raise TypeError("data_X must be a pd.DataFrame.")
        elif data_X.shape[1] != self.dim_X:
            raise ValueError("data_X must have the same dimension as X_bound.")
        else:
            self._data_X = data_X
        
        # list_y_target setting
        if not isinstance(list_y_target, list):
            raise TypeError("list_y_target must be a list.")
        else:
            self._list_y_target = list_y_target
        
        # model setting
        if not isinstance(model, object):
            raise TypeError("model must be a object.")
        else:
            self._model = model
            
        # scaler setting
        if not isinstance(scaler, object):
            raise TypeError("scaler must be a object.")
        else:
            self._scaler = scaler
        
    
    def optimize(self, data_X, list_y_target, model, scaler, iter_optimize=10):
        """
        Use PSO multiple times to perform parameter optimization 
        and calculate its 95% confidence interval as interval recommendation.
        
        Parameters
        -----
        data_X : pd.DataFrame
            The raw data.
        list_y_target : list[float]
            The targets of each y.
        model : object
            Trained model.
        scaler : object
            Trained scaler model.
        iter_optimize : int, default: 100
            number of optimizations.
            
        Returns
        -----
        df_resulf: pd.DataFrame
            The interval recommendation of each parameter in list_control.
        """
        # Parameter setting
        if not isinstance(iter_optimize, int):
            raise TypeError("iter_optimize must be a int.")
        else:
            self._iter_optimize = iter_optimize
        
        # Data encapsulation
        self._data_encapsulation(data_X, list_y_target, model, scaler)
        
        # Value Recommend
        self._best_cost = np.inf
        if self.all_parameter:
            n_col = self._data_X.shape[1]
        else:
            n_col = len(self.list_control)
        optimize_result = np.zeros((self._iter_optimize, n_col))
        for iter_optimize in range(self._iter_optimize):
            VR = ValueRecommend(list_control=self.list_control,
                                list_X_bound=self.list_X_bound,
                                all_parameter=self.all_parameter)
            result = VR.optimize(data_X=self.data_X,
                                 list_y_target=self.list_y_target,
                                 model=self.model,
                                 scaler=self.scaler)
            if VR.PSO_cost < self._best_cost:
                self._best_cost = VR.PSO_cost
                self._best_position = result
            optimize_result[iter_optimize] = result['Recommendation']
        
        # 95% confidence interval
        result_mean = optimize_result.mean(axis=0)
        result_std = optimize_result.std(axis=0)
        self._lower_result, self._upper_result = stats.norm.interval(0.95, result_mean, result_std)
        
        # out of bound
        mask_control = self._data_X.columns.isin(result.index)
        np_bound = np.array(self.list_X_bound)[:,mask_control]
        self._lower_result = np.maximum(self._lower_result, np_bound[0])
        self._upper_result = np.minimum(self._upper_result, np_bound[1])
        
        # Range recommend result
        self._df_resulf = pd.DataFrame(np.vstack((self._upper_result, self._lower_result)).T,
                                       index=self._data_X.columns[mask_control],
                                       columns=['Upper_bound', 'Lower_bound'])
        
        # for all_parameter
        if self.all_parameter:
            self._df_resulf['self_controled'] = result['self_controled']

        return self._df_resulf
#endregion
