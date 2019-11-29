# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp
from sklearn_lvq import GlvqModel, GmlvqModel, LgmlvqModel


class LvqCounterfactualBase(ABC):
    def __init__(self, model):
        self.model = model
        self.prototypes = model.w_
        self.labels = model.c_w_
        self.dim = model.w_[0].shape[0]

        super(LvqCounterfactualBase, self).__init__()
            
    @abstractmethod
    def generate_counterfactual(self, x_orig, y_target, features_whitelist=None, mad=None):
        raise NotImplementedError()


class MatrixLvqCounterfactual(LvqCounterfactualBase):
    def __init__(self, model):
        if not isinstance(model, GmlvqModel):
            raise TypeError(f"model has to be an instance of 'sklearn_lvq.GmlvqModel' but not of {type(model)}")

        super(MatrixLvqCounterfactual, self).__init__(model)
    
    def _solve(self, prob):
        prob.solve(max_iters=100000, solver=cp.SCS, verbose=False)

    def _build_omega(self):
        return np.dot(self.model.omega_.T, self.model.omega_)

    def _compute_counterfactual_target_prototype(self, x_orig, target_prototype, other_prototypes, y_target, features_whitelist=None, mad=None):
        # Variables
        x = cp.Variable(self.dim)
        beta = cp.Variable(self.dim)
        
        # Constants
        c = np.ones(self.dim)
        z = np.zeros(self.dim)
        I = np.eye(self.dim)

        # Construct constraints
        p_i = target_prototype

        Omega = self._build_omega()

        G = []
        b = np.zeros(len(other_prototypes))
        k = 0
        for k in range(len(other_prototypes)):
            p_j = other_prototypes[k]
            G.append(0.5 * np.dot(Omega, p_j - p_i))
            b[k] = -0.5 * (np.dot(p_i, np.dot(Omega, p_i)) - np.dot(p_j, np.dot(Omega, p_j)))
        G = np.array(G)

        # If requested, fix the values of some features/dimensions
        A = None
        a = None
        if features_whitelist is not None:
            A = []
            a = []

            for j in range(self.dim):
                if j not in features_whitelist:
                    t = np.zeros(self.dim)
                    t[j] = 1.
                    A.append(t)
                    a.append(x_orig[j])
            A = np.array(A)
            a = np.array(a)

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1. / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        constraints = None
        if mad is not None:
            f = cp.Minimize(c.T @ beta)    # Minimize (weighted) Manhattan distance
            constraints = [G @ x <= b, Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, I @ beta >= z]
        else:
            f = cp.Minimize((1/2)*cp.quad_form(x, I) - x_orig.T@x)  # Minimize L2 distance
            constraints = [G @ x <= b]
        
        if A is not None and a is not None:
            constraints += [A @ x == a]
        
        prob = cp.Problem(f, constraints)
        
        # Solve it!
        self._solve(prob)
        
        return x.value

    def generate_counterfactual(self, x_orig, y_target, features_whitelist=None, mad=None):
        xcf = None
        xcf_dist = float("inf")

        dist = lambda x: np.linalg.norm(x - x_orig, 2)
        if mad is not None:
            dist = lambda x: np.dot(mad, np.abs(x - x_orig))
        
        # Search for suitable prototypes
        target_prototypes = []
        other_prototypes = []
        for p, l in zip(self.prototypes, self.labels):
            if l == y_target:
                target_prototypes.append(p)
            else:
                other_prototypes.append(p)
        
        # Compute a counterfactual for each prototype
        for i in range(len(target_prototypes)):
            try:
                xcf_ = self._compute_counterfactual_target_prototype(x_orig, target_prototypes[i], other_prototypes, y_target, features_whitelist, mad)
                ycf_ = self.model.predict([xcf_])[0]

                if ycf_ == y_target:
                    if dist(xcf_) < xcf_dist:
                        xcf = xcf_
                        xcf_dist = dist(xcf_)
            except:
                pass
    
        if xcf is None:
            # It might happen that the solver (for a specific set of parameter values) does not find a counterfactual, although the feasible region is always non-empty
            j = np.argmin([dist(proto) for proto in target_prototypes]) # Select the nearest prototype!
            xcf = target_prototypes[j]
        
        return xcf, self.model.predict([xcf])[0], x_orig - xcf


class LvqCounterfactual(MatrixLvqCounterfactual):
    def __init__(self, model):
        if not isinstance(model, GlvqModel):
            raise TypeError(f"model has to be an instance of 'sklearn_lvq.GlvqModel' but not of {type(model)}")
        
        self.dim = model.w_[0].shape[0]

        LvqCounterfactualBase.__init__(self, model) # Note: We can not call the constructor of the parent class because it expects a GmlvqModel
    
    def _build_omega(self):
        return np.eye(self.dim)


class LocalizedMatrixLvqCounterfactual(LvqCounterfactualBase):
    def __init__(self, model):
        if not isinstance(model, LgmlvqModel):
            raise TypeError(f"model has to be an instance of 'sklearn_lvq.LgmlvqModel' but not of {type(model)}")

        self.omegas = [np.dot(o.T, o) for o in model.omegas_]

        super(LocalizedMatrixLvqCounterfactual, self).__init__(model)
    
    def _solve(self, prob):
        prob.solve(max_iters=100000, solver=cp.SCS, verbose=False)

    def solve_aux(self, xcf, tao, x_orig, y_target, target_prototype, target_omega, other_prototypes, other_omegas, mad=None, features_whitelist=None):
        try:
            # Variables
            x = cp.Variable(self.dim)
            beta = cp.Variable(self.dim)
            s = cp.Variable(len(other_prototypes))

            # Constants
            s_z = np.zeros(len(other_prototypes))
            s_c = np.ones(len(other_prototypes))
            c = np.ones(self.dim)
            z = np.zeros(self.dim)
            I = np.eye(self.dim)

            # Construct constraints
            p_i = target_prototype
            omega_i = target_omega
            r_i = 0.5 * np.dot(p_i, np.dot(omega_i, p_i))
            q_i = 0.5 * np.dot(omega_i, p_i)

            constraints = []
            for p_j, o_j in zip(other_prototypes, other_omegas):
                q =  0.5 * np.dot(o_j, p_j) - q_i
                r = r_i - 0.5 * np.dot(p_j, np.dot(o_j, p_j))
                xcf_oj = np.dot(o_j, xcf)
                g_approx = 0.5 * np.dot(xcf, xcf_oj) - np.dot(xcf, xcf_oj) + xcf_oj.T @ x

                constraints.append(cp.quad_form(x, omega_i) + q.T @ x + r - g_approx <= 0)

            # If requested, fix the values of some features/dimensions
            A = None
            a = None
            if features_whitelist is not None:
                A = []
                a = []

                for j in range(dim):
                    if j not in features_whitelist:
                        t = np.zeros(dim)
                        t[j] = 1.
                        A.append(t)
                        a.append(x_orig[j])
                A = np.array(A)
                a = np.array(a)
            
            # If necessary, construct the weight matrix for the weighted Manhattan distance
            Upsilon = None
            if mad is not None:
                alpha = 1. / mad
                Upsilon = np.diag(alpha)

            # Build the final program
            f = None
            if mad is not None:
                f = cp.Minimize(c.T @ beta + s.T @ (tao*s_c))   # Minimize (weighted) Manhattan distance
                constraints += [s >= s_z, Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, I @ beta >= z]
            else:
                f = cp.Minimize((1/2)*cp.quad_form(x, I) - x_orig.T @ x + s.T @ (tao*s_c))  # Minimize L2 distance
                constraints += [s >= s_z]
            
            if A is not None and a is not None:
                constraints += [A @ x == a]

            
            prob = cp.Problem(f, constraints)
        
            # Solve it!
            self._solve(prob)

            if x.value is None:
                raise Exception("No solution found!")
            else:
                return x.value
        except:
            return target_prototype 

    def _compute_counterfactual_target_prototype(self, x_orig, target_prototype, target_omega, other_prototypes, other_omegas, y_target, features_whitelist, mad, tao, tao_max, mu):
        ####################################
        # Penalty convex-concave procedure #
        ####################################

        # Use the target prototype as an inital feasible solution
        xcf = target_prototype

        # Hyperparameters
        cur_tao = tao

        # Solve a bunch of CCPs
        n_iter = 0
        while cur_tao < tao_max:
            xcf_ = self.solve_aux(xcf, cur_tao, x_orig, y_target, target_prototype, target_omega, other_prototypes, other_omegas, mad, features_whitelist)

            if y_target == self.model.predict([xcf_])[0]:
                xcf = xcf_

            # Increase penalty parameter
            cur_tao *= mu
            n_iter += 1
        
        return xcf

    def generate_counterfactual(self, x_orig, y_target, features_whitelist=None, mad=None, tao=1.2, tao_max=100, mu=1.5):
        xcf = None
        xcf_dist = float("inf")

        dist = lambda x: np.linalg.norm(x - x_orig, 2)
        if mad is not None:
            dist = lambda x: np.dot(mad, np.abs(x - x_orig))
        
        # Search for suitable prototypes
        target_prototypes = []
        target_omegas = []
        other_prototypes = []
        other_omegas = []
        for p, l, o in zip(self.prototypes, self.labels, self.omegas):
            if l == y_target:
                target_prototypes.append(p)
                target_omegas.append(o)
            else:
                other_prototypes.append(p)
                other_omegas.append(o)
        
        # Compute a counterfactual for each prototype
        for i in range(len(target_prototypes)):
            try:
                xcf_ = self._compute_counterfactual_target_prototype(x_orig, target_prototypes[i], target_omegas[i], other_prototypes, other_omegas, y_target, features_whitelist, mad, tao, tao_max, mu)
                ycf_ = self.model.predict([xcf_])[0]

                if ycf_ == y_target:
                    if dist(xcf_) < xcf_dist:
                        xcf = xcf_
                        xcf_dist = dist(xcf_)
            except:
                pass
    
        if xcf is None:
            # It might happen that the solver (for a specific set of parameter values) does not find a counterfactual, although the feasible region is always non-empty
            j = np.argmin([dist(proto) for proto in target_prototypes]) # Select the nearest prototype!
            xcf = target_prototypes[j]
        
        return xcf, self.model.predict([xcf])[0], x_orig - xcf
