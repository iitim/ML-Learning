from math import dist, log, exp
import logging

import numpy as np


class UKMeans:
    def __init__(self, e, data_points):
        self.t = 0  # The number of iteration
        self.e = e  # Threshold for the centroids movement
        self.n = len(data_points)  # The number of data points
        self.c = len(data_points)  # The number of centroids
        self.c_cum = [len(data_points)]
        self.remain_cluster = range(self.c)
        self.alpha = [1/self.n] * self.n  # The probability of one data point belonged to the kth class
        self.a = data_points.copy()  # The position of kth centroids
        self.beta = 1  # Learning rate β
        self.gamma = 1  # Learning rate γ

        self.x = data_points  # data points
        self.z = []  # 0 if data point xi belongs to kth cluster

    def compute_z(self):
        z = []

        for xi in self.x:
            zi = [0] * self.c
            min_entropy = float('inf')
            belonging_cluster = []

            for k in range(self.c):
                entropy = (dist(xi, self.a[k])**2) - (self.gamma * log(self.alpha[k]))
                if min_entropy > entropy:
                    min_entropy = entropy
                    belonging_cluster = [k]
                elif min_entropy == entropy:
                    belonging_cluster.append(k)

            for k in belonging_cluster:
                zi[k] = 1

            z.append(zi)

        return z

    def compute_gamma(self, param_decrease_rate=250):
        return exp(-self.c/param_decrease_rate)

    def compute_alpha(self):
        new_alpha = []
        common_equation = sum(alpha_s*log(alpha_s) for alpha_s in self.alpha)
        z_transpose = np.array(self.z).T.tolist()

        for k, alpha_k in enumerate(self.alpha):
            new_alpha.append(
                (sum(z_transpose[k])/self.n) + ((self.beta*alpha_k/self.gamma) * (log(alpha_k)-common_equation))
            )

        return new_alpha

    def compute_beta(self, current_alpha, new_alpha):
        z_transpose = np.array(self.z).T.tolist()

        eta = 1  # To be updated after understand the formula
        time_based_beta = 0
        beta_restriction_max_numerator = float('-inf')
        beta_restriction_max_denominator = float('-inf')

        for k in range(self.c):
            time_based_beta += exp(-1 * eta * self.n * abs(new_alpha[k] - current_alpha[k]))

            kth_numerator = sum(z_transpose[k])/self.n
            if beta_restriction_max_numerator < kth_numerator:
                beta_restriction_max_numerator = kth_numerator

            kth_denominator = current_alpha[k] * sum(log(alpha_k) for alpha_k in current_alpha)
            if beta_restriction_max_denominator < kth_denominator:
                beta_restriction_max_denominator = kth_denominator

        time_based_beta /= self.c
        beta_restriction = (1-beta_restriction_max_numerator) / (-beta_restriction_max_denominator) if beta_restriction_max_denominator != 0 else float('inf')

        return min(time_based_beta, beta_restriction)

    def update_cluster(self, new_alpha):
        z_transpose = np.array(self.z).T.tolist()
        remain_alpha = []
        remain_z = []
        self.remain_cluster = [k for k, alpha_k in enumerate(new_alpha) if alpha_k > 1/self.n]
        for k in self.remain_cluster:
            remain_alpha.append(new_alpha[k]),
            remain_z.append(z_transpose[k])

        sum_alpha = sum(remain_alpha)

        self.c = len(self.remain_cluster)
        self.c_cum.append(self.c)
        self.alpha.clear()
        self.z.clear()
        z_transpose = []
        for k in range(self.c):
            self.alpha.append(remain_alpha[k] / sum_alpha)
            zk = []
            sum_zk = sum(remain_z[k])
            for i in range(self.n):
                zk.append(remain_z[k][i] / sum_zk)
            z_transpose.append(zk)
        self.z = np.array(z_transpose).T.tolist()

        if self.t >= 60 and self.c_cum[self.t-60]-self.c == 0:
            self.beta = 0

    def compute_a(self):
        z_transpose_np = np.array(self.z).T
        x_np = np.array(self.x)
        new_a = []
        for k in self.remain_cluster:
            new_a.append(sum(np.stack((z_transpose_np[k],z_transpose_np[k]), axis=-1))*x_np/sum(z_transpose_np[k]).tolist())

        return new_a

    def stop_loop(self, current_a, new_a):
        current_a = np.array(current_a)
        new_a = np.array(new_a)

        return np.amax(np.linalg.norm(new_a - current_a)) < self.e

    def run_iteration(self):
        self.z = self.compute_z()
        self.gamma = self.compute_gamma()
        new_alpha = self.compute_alpha()
        self.beta = self.compute_beta(self.alpha, new_alpha)
        self.update_cluster(new_alpha)
        new_a = self.compute_a()
        loop_justification = not self.stop_loop(self.a, new_a)
        self.a = new_a
        self.t += 1

        return loop_justification

    def main(self):
        while self.run_iteration():
            logging.info(f"Iteration: {self.t}, Remain K: {self.c}")
        logging.info(f"Done")
