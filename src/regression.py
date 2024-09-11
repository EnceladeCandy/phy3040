import numpy as np
from typing import Union, Sequence

class FrequentistRegression:
    """
    Class to perform frequentist linear regression. 
    """
    def __init__(self, degree: int, sigma_noise: float):
        self.degree = degree
        self.sigma_noise = sigma_noise
        self.design_matrix = np.nan
        self.coeff_fitted = np.nan
        self.cov = np.nan
        self.std_fitted = np.nan        
        self.chi2 = np.nan


    def create_design_matrix(self, x:np.array, degree:int = 1, fit_intercept:bool = True) -> np.array:
        """
        Creates the design matrix of the linear regression.

        Args:
            x (np.array): Input variable (to the forward model)
            degree (int, optional): Degree of the polynomial. Defaults to 1.
            fit_intercept (bool, optional): . Defaults to True.
        """
        X = np.column_stack([x ** i for i in range(1, degree+1)])
        if fit_intercept: 
            X = np.column_stack([np.ones_like(x), X])
        return X
        
    def fit(self, obs: np.array, x: np.array, ridge_reg:float = 1e-8, fit_intercept: bool = True) -> Sequence:
        if np.isnan(self.design_matrix).any(): 
            self.design_matrix = self.create_design_matrix(x, 
                                      degree = self.degree, 
                                      fit_intercept = fit_intercept
                                      )
        X = self.design_matrix
        print(X.shape)
        A = X.T @ X  
        print(A.shape)
        inv_mat = np.linalg.inv(A + ridge_reg) 
        self.coeff_fitted = (inv_mat @ X.T @ obs)
        
        # Computing covariance 
        self.cov = (self.sigma_noise) ** 2 * inv_mat
        self.std = np.sqrt(np.diag(self.cov)) 
        return (self.coeff_fitted, self.std)  

    def forward_model(self) -> np.array:
        return self.design_matrix @ self.coeff_fitted 
    
    def compute_chi2(self, obs: np.array) -> float: 
        """
        Computes the reduced chi squared using the observation and the model prediction

        Args:
            obs (np.array): Noisy observation
            x (np.array): Input variable 
        """
        obs_pred = self.forward_model()
        residuals = (obs - obs_pred) / self.sigma_noise
        self.chi2 = np.sum(residuals ** 2) / len(obs)
        return self.chi2

    def general_routine(self, obs: np.array, x: np.array, ridge_reg:float = 1e-8, fit_intercept:bool = True):
        """
        Performs the linear regression and computes the chi squared (assuming Gaussian noise)
        """
        self.fit(
            obs = obs, 
            x = x, 
            ridge_reg = ridge_reg, 
            fit_intercept = fit_intercept
        )
        self.compute_chi2(self, obs, x)


class BayesianRegression:
    """
    Class to perform Bayesian linear regression. 
    """
    def __init__(self, degree, sigma_noise):
        self.degree = degree
        self.sigma_noise = sigma_noise
        self.design_matrix = np.nan
        self.coeff_fitted = np.nan
        self.cov = np.nan
        self.std_fitted = np.nan        
        self.chi2 = np.nan


    def create_design_matrix(self, x, degree = 1, fit_intercept = True):
        """
        Creates the design matrix of the linear regression.

        Args:
            x (np.array): Input variable (to the forward model)
            degree (int, optional): Degree of the polynomial. Defaults to 1.
            fit_intercept (bool, optional): Defaults to True.
        """
        X = np.column_stack([x ** i for i in range(1, degree+1)])
        if fit_intercept: 
            X = np.column_stack([np.ones_like(x), X])
        return X
        
    def fit(self, obs, x, ridge_reg = 1e-8, fit_intercept = True):
        self.design_matrix = self.create_design_matrix(x, 
                                      degree = self.degree, 
                                      fit_intercept = fit_intercept
                                      )
        X = self.design_matrix
        A = X.T @ X  
        inv_mat = np.linalg.inv(A + ridge_reg) 
        self.coeff_fitted = (inv_mat @ X.T @ obs)

        # Computing covariance 
        self.cov = (self.sigma_noise) ** 2 * inv_mat
        self.std = np.sqrt(np.diag(self.cov)) 
        return (self.coeff_fitted, self.std)  

    def forward_model(self, x):
        return self.design_matrix @ self.coeff_fitted 
    
    def compute_chi2(self, obs: np.array, x:np.array) -> float: 
        """
        Computes the reduced chi squared using the observation and the model prediction

        Args:
            obs (np.array): Noisy observation
            x (np.array): Input variable 
        """
        obs_pred = self.forward_model(x)
        residuals = (obs - obs_pred) / self.sigma_noise
        self.chi2 = np.sum(residuals ** 2) / len(obs)
        return self.chi2

    def general_routine(self, obs, x, ridge_reg = 1e-8, fit_intercept = True):
        """
        Fit the parameters and compute the chi squared as well.

        Args:
            obs (_type_): _description_
            x (_type_): _description_
            ridge_reg (_type_, optional): _description_. Defaults to 1e-8.
            fit_intercept (bool, optional): _description_. Defaults to True.
        """
        self.fit(
            obs = obs, 
            x = x, 
            ridge_reg = ridge_reg, 
            fit_intercept = fit_intercept
        )
        self.compute_chi2(self, obs, x)

        