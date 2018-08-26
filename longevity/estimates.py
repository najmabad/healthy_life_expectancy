from typing import Dict, Tuple, Any

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LongevityEstimator:
    """
    Longevity Estimator class containing all the
    calculations to estimate the longevity of a
    subset of the share data-set obtained in the
    longevity.DataManager
    """

    def __init__(self, df: pd.DataFrame, panel_df: pd.DataFrame, gender: str, income_dcl: int):
        """
        Initialize a LongevityEstimator object
        :param df: a dataframe resulting from a call to longevity.DataManager.prepare_dataset
        :param panel_df: a dataframe resulting from a call to longevity.DataManager.create_panel_dataset
        :param gender: the gender of the individual, can be "male" or "female"
        :param income_dcl:
        """
        self.df = df
        self.panel_df = panel_df
        self.dis_prev_by_ctry_and_age = LongevityEstimator.compute_disability_prevalence_by_age(df)
        self.start_age, self.end_age = int(self.df.age.min()), int(self.df.age.max()) + 1
        self.diff = int(self.end_age - self.start_age)
        self.gender = gender
        self.income_dcl = income_dcl
        self.income_buckets = len(self.df.income_dcl.unique())
        self.clf = Pipeline([
            ('scaler', StandardScaler(with_std=False, with_mean=False)),
            ('lr', LogisticRegression())
        ])

    @staticmethod
    def compute_disability_prevalence_by_age(df: pd.DataFrame) -> Dict[Any, Any]:
        """
        Computes the prevalence of disability matrix by age
        :param df: a dataframe resulting from a call to longevity.DataManager.prepare_dataset
        :return: the disability prevalence matrix
        """
        return (pd.pivot_table(
            df[df['disabled'] == 1],
            columns=['gender'],
            index=['income_dcl', 'is_aged'],
            values=['mergeid'],
            aggfunc='count'
        ).replace(np.nan, 0) / pd.pivot_table(
            df,
            columns=['gender'],
            index=['income_dcl', 'is_aged'],
            values=['mergeid'],
            aggfunc='count'
        )).to_dict()

    def smooth_disability_curve(self, y: np.array) -> np.array:
        """
        Smooths out the disability curve in order to reduce noise
        1) substitutes 0 values with the last available prevalence
        2) fits an exponential function through the data
        :param y: disability curve
        :return: smoothed disability curve
        """
        # for idx, i in enumerate(y):
        #    if i == 0:
        #        y[idx] = y[idx - 1] if idx > 0 else y[idx + 1]
        x = range(self.start_age, self.end_age)
        f = lambda t, a, b: a * (np.e ** (t * b))
        a, b = sp.optimize.curve_fit(f, x, y, p0=(4, 0.1))[0]
        return f(x, a, b)

    def prevalence(self, age: int) -> float:
        """
        Compute the disability prevalence by age
        :param age: age for which to compute the disability
        :return: a float representing the disability prevalence
                 for the specified age class
        """
        ps = [self.dis_prev_by_ctry_and_age[('mergeid', self.gender)].get((self.income_dcl, age), 0) for age in
              range(self.start_age, self.end_age)]
        #for idx, i in enumerate(ps):
        #    if np.isnan(i):
        #        ps[idx] = 0 if idx == 0 else ps[idx - 1]
        #ps = self.smooth_disability_curve(ps)
        return ps[int(age - self.start_age)]

    def generate_prevalence_matrix(self) -> np.array:
        """
        Generate the healthy prevalence matrix
        :return: a healthy prevalence matrix
        """
        m = []
        for age in range(self.start_age, self.end_age):
            m.append(self.prevalence(age))
        m.append(0)
        return 1 - np.vstack([[m] * (len(m) - 1), [i / 2 for i in m]])

    def compute_UP(self) -> Tuple[np.array, np.array]:
        """
        Computes the U and P matrices as defined in Caswell, Zarulli (2018)
        :return: U and P matrices
        """
        self.clf.fit(self.panel_df[['age', 'income_dcl', 'gender_num']], self.panel_df['y'])

        lr = self.clf.named_steps['lr']
        scaler = self.clf.named_steps['scaler']
        print(lr.coef_)

        P = np.hstack(
            [np.zeros((self.diff + 1, self.diff)), np.hstack([np.zeros(self.diff), [1]]).reshape(self.diff + 1, 1)])
        for age in range(self.start_age, self.end_age):
            row = [age, self.income_dcl, 1 if self.gender == 'female' else 0]
            row = scaler.transform([row])
            if age != 90:
                p_alive, p_dead = lr.predict_proba(row)[0]
                P[age - (self.start_age - 1), age - self.start_age] = p_alive
                P[self.diff, age - self.start_age] = p_dead
            else:
                P[self.diff - 1, self.diff - 1] = lr.predict_proba(row)[0][0]
                P[self.diff, self.diff - 1] = p_dead
        U = P[:self.diff, :self.diff]
        return U, P

    def compute_moments(self, U: np.array, P: np.array, prevalence_matrix:
    np.array, healthy_life_only: bool) -> Tuple[np.array, np.array, np.array]:
        """
        Compute vector of rewards moments as defined in Caswell, Zarulli (2018)
        :param U: the U matrix
        :param P: the P matrix
        :param prevalence_matrix: the prevalence matrix
        :param healthy_life_only: whether to compute moments for healthy or total life 
        :return: the first three moments of the vector of rewards
        """
        Z = np.hstack([np.eye(self.diff), np.zeros((self.diff, 1))])
        N = np.linalg.inv(np.eye(self.diff) - U)

        if healthy_life_only:
            R_1 = R_2 = R_3 = prevalence_matrix
        else:
            R_1 = R_2 = R_3 = np.hstack(
                [np.vstack([np.ones((self.diff, self.diff)), np.ones(self.diff) / 2]), np.zeros((self.diff + 1, 1))])

        ones = np.matrix(np.ones(self.diff + 1)).T

        # 1st moment
        rho_1 = (N.T @ Z @ (np.multiply(P, R_1)).T @ ones)

        # 2nd moment
        R_1_tilde = Z @ R_1 @ Z.T
        rho_2 = N.T @ (Z @ np.multiply(P, R_2).T @ ones + 2 * (np.multiply(U, R_1_tilde)).T @ rho_1)

        # 3rd moment
        R_2_tilde = Z @ R_2 @ Z.T
        rho_3 = N.T @ (Z @ np.multiply(P, R_3).T @ ones + 3 * (np.multiply(U, R_2_tilde)).T @ rho_1 + 3 * (
            np.multiply(U, R_1_tilde)).T @ rho_2)

        return rho_1, rho_2, rho_3

    def compute_lifetime_estimates(self, healthy_life_only: bool) -> Tuple[float, float]:
        """
        Compute the lifetime estimates
        :param healthy_life_only: whether to compute lifetime for healthy or total life
        :return:
        """
        prevalence_matrix = self.generate_prevalence_matrix()
        U, P = self.compute_UP()
        mu, m2, _ = self.compute_moments(U, P, prevalence_matrix, healthy_life_only)
        std = np.sqrt(np.array(m2).squeeze() - np.array(mu).squeeze() * np.array(mu).squeeze())
        mu = np.array(mu).squeeze()
        return mu, std

    @staticmethod
    def gini(y: np.array) -> float:
        """
        Calculate the Gini coefficient of a numpy array.
        :param y: the array for which to compute the coefficient
        :return: the Gini coefficient
        """
        y = y.flatten()  # all values are treated equally, arrays must be 1d
        if np.amin(y) < 0:
            y -= np.amin(y)  # values cannot be negative
        y += 0.0000001  # values cannot be 0
        y = np.sort(y)  # values must be sorted
        index = np.arange(1, y.shape[0] + 1)  # index per array element
        n = y.shape[0]  # number of array elements
        return np.sum((2 * index - n - 1) * y) / (n * np.sum(y))  # Gini coefficient

    @staticmethod
    def theil(y: np.array) -> float:
        """
        Calculate the Theil coefficient of a numpy array.
        :param y: the array for which to compute the coefficient
        :return: the Theil coefficient
        """
        n = len(y)
        y = y + 1e-08 * (y == 0)
        yt = y.sum(axis=0)
        s = y / (yt * 1.0)
        lns = np.log(n * s)
        slns = s * lns
        t = sum(slns)
        return t
