from copy import deepcopy
import logging
import numpy as np
import scipy.stats as ss
from turbo import TurboM
from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace
logger = logging.getLogger(__name__)


def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


class TurboOptimizer_M(AbstractOptimizer):
    primary_import = "Turbo"

    def __init__(self, api_config, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        self.space_x = JointSpace(api_config)
        self.bounds = self.space_x.get_bounds()
        self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]
        self.dim = len(self.bounds)
        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.batch_size = None
        self.history = []

        self.turbo = TurboM(
            f=None,
            lb=self.bounds[:, 0],
            ub=self.bounds[:, 1],
            n_init=self.dim  , # origin 2 * self.dim + 1
            max_evals=self.max_evals,
            n_trust_regions=2 , # original 1
            batch_size=1,  # We need to update this later
            verbose=False,
            n_training_steps=70 # origin 50
        )
        

        
    def print_all_data(self):
        print("="*10)
        print("X",self.turbo.X)
        print("_X",self.turbo._X)
        print("fX",self.turbo.fX)
        print("_fX",self.turbo._fX)
        print("_idx",self.turbo._idx)

    def restart(self):
        self.turbo._restart()
        self.turbo._X = np.zeros((0, self.turbo.dim))
        self.turbo._fX = np.zeros((0, 1))
        
        self.X_init = np.zeros((0, self.dim))
        self.turbo._idx = np.zeros((0,1), dtype=int)
        self.init_idx = np.zeros((0,1), dtype=int)
        
        for i in range(self.turbo.n_trust_regions):
            X_init = latin_hypercube(self.turbo.n_init, self.dim)
            #fX_init =  np.zeros((0, 1))
            self.X_init = np.vstack((self.X_init,from_unit_cube(X_init, self.lb, self.ub)))
            self.init_idx =  np.vstack((self.init_idx, i * np.ones((self.turbo.n_init, 1), dtype=int)))
        
    def suggest(self, n_suggestions=1):
        if self.batch_size is None:  # Remember the batch size on the first call to suggest
            self.batch_size = n_suggestions
            self.turbo.batch_size = n_suggestions
            self.turbo.failtol = np.ceil(np.max([4.0 / self.batch_size, self.dim / self.batch_size]))
            self.turbo.n_init = max([self.turbo.n_init, self.batch_size])
            self.restart()

        X_next = np.zeros((n_suggestions, self.dim))
        X_next_idx = np.zeros((n_suggestions, 1))
        
        # Pick from the initial points
        # Need to fix blow for flexble parameters
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :]) 
            X_next_idx[:n_init] = deepcopy(self.init_idx[:n_init, :])
            self.X_init = self.X_init[n_init:, :]  # Remove these pending points
            self.init_idx = self.init_idx[n_init:, :]
            
        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        ### 이 부분을 TR 별로 할 수 있어야.
        ### self.length, self.hypers 모두 list
        ### X_cand = (trust region, n_suggestion, dim) 으로 모두 모은 다음에 select_candidate 로 넘기기
        
        X_cand = np.zeros((self.turbo.n_trust_regions, self.turbo.n_cand, self.dim))
        y_cand = np.inf * np.ones((self.turbo.n_trust_regions,\
                                  self.turbo.n_cand, self.batch_size))
        
        if n_adapt > 0 :
            if len(self.turbo._X) > 0 :
                for i in range(self.turbo.n_trust_regions):
                    idx = np.where(self.turbo._idx == i)[0]
                    #print("Self.turbo._idx",self.turbo._idx)
                    X = deepcopy(self.turbo._X[idx,:])
                    X = to_unit_cube(X, self.lb, self.ub)
                    fX = copula_standardize(deepcopy(self.turbo._fX[idx,:]).ravel())
                    #print("="*20)
                    #print("idx : {} X : {} fX : {} hyper :{}".format(idx.shape,X.shape,fX.shape,self.turbo.hypers[i]))
                    #print("x_cand : {} y_cand : {}".format(X_cand.shape,y_cand.shape))

                    n_training_steps = 0 if self.turbo.hypers[i] else self.turbo.n_training_steps
                    X_cand[i, :, :], y_cand[i, :, :] , self.turbo.hypers[i] = self.turbo._create_candidates(
                    X, fX, length=self.turbo.length[i], n_training_steps=n_training_steps, hypers=self.turbo.hypers[i]) 
                
                # Select next candidates
                X_next_cand, idx_next_cand = self.turbo._select_candidates(X_cand, y_cand)
                assert X_next_cand.min() >= 0.0 and X_next_cand.max() <= 1.0
                
                # Undo the warping
                X_next[-n_adapt:] = from_unit_cube(X_next_cand[-n_adapt:], self.lb, self.ub)
                X_next_idx[-n_adapt:,0] = idx_next_cand[-n_adapt:,0]
                


        # Unwarp the suggestions
        #print("Lower bound : {} Upper bound : {}".format(self.lb, self.ub))
        #print("X_next",X_next)
        suggestions = self.space_x.unwarp(X_next)
        self.turbo.suggestion_idx = X_next_idx
        return suggestions

    def observe(self, X, y):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        assert len(X) == len(y)
        
        XX, yy = self.space_x.warp(X), np.array(y)[:, None]
        
        # Update trust regions
        idx_next = self.turbo.suggestion_idx
        for i in range(self.turbo.n_trust_regions):
            idx_i = self.turbo._idx[:, 0] == i
            if not np.sum(self.init_idx == i) and sum(idx_i) != 0: # Restart 된 게 아니다

                idx_i = np.where(idx_next == i)[0]
                if len(idx_i) > 0 : # Next에 포함되어 있다
                    self.turbo.hypers[i] = {}
                    fX_i = yy[idx_i]
                    self.turbo._adjust_length(fX_i, i) 
        
        
        self.turbo.n_evals += self.batch_size
        self.turbo._X = np.vstack((self.turbo._X, deepcopy(XX)))
        self.turbo._fX = np.vstack((self.turbo._fX, deepcopy(yy)))
        self.turbo.X = np.vstack((self.turbo.X, deepcopy(XX)))
        self.turbo.fX = np.vstack((self.turbo.fX, deepcopy(yy)))
        self.turbo._idx = np.vstack((self.turbo._idx, deepcopy(idx_next)))
#         print("Suggestion index", self.turbo._idx)
#         print("turbo.x", self.turbo._X)
#         print("turbo.fx", self.turbo._fX)

        # Restart
        for i in range(self.turbo.n_trust_regions):
            if self.turbo.length[i] < self.turbo.length_min:
                idx_i = self.turbo._idx[:, 0] == i
                
                # Reset Length and counters, remove old data from TR
                self.turbo.length[i] = self.turbo.length_init
                self.turbo.succcount[i] = 0
                self.turbo.failcount[i] = 0
                self.turbo._idx[idx_i, 0] =  -1 
                self.turbo.hypers[i] = {}
                
                X_init = latin_hypercube(self.turbo.n_init, self.dim)
                
                self.X_init = np.vstack((self.X_init,from_unit_cube(X_init, self.lb, self.ub)))
                self.init_idx =  np.vstack((self.init_idx, i * np.ones((self.turbo.n_init, 1), dtype=int)))
            


if __name__ == "__main__":
    experiment_main(TurboOptimizer_M)
