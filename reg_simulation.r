# -----------------------------
# Setup
# -----------------------------
rm(list = ls())
library(truncnorm)
library(lpSolve)
library(lintools)
library(Rcpp)

# Source custom C++ functions
sourceCpp("hit_and_run.cpp")
sourceCpp("tum_check.cpp")

# -----------------------------
# Parameters and Simulation Setup
# -----------------------------
set.seed(123)

n <- 1000  # number of observations
p <- 5     # number of predictors
d <- 2    # number of outcomes
m <- 1     # number of constraints

X <- matrix(rnorm(n * p), n, p)

# Generate constraint matrix A (Totally Unimodular or incidence matrix)
if (d * m <= 50 | m < 2) {
  A <- round(matrix(runif(m * d, -1, 1), m, d))
  while (!is_totally_unimodular(A) || any(rowSums(abs(A)) == 0)) {
    A <- round(matrix(runif(m * d, -1, 1), m, d))
  }
  b <- rep(1, m)
} else {
  A <- matrix(0, m, d)
  for (i in 1:m) {
    idx <- sample(1:d, 2, replace = FALSE)
    A[i, idx[1]] <- 1
    A[i, idx[2]] <- -1
  }
  b <- sample(0:1, m, replace = TRUE)
}

cat("Constraint matrix A:\n"); print(A)
cat("Right-hand side vector b:\n"); print(b)

# Check if A is TUM (Not efficient for high dimension)
# is_totally_unimodular(A)  # From lintools
# checkTotallyUnimodularArma(A)  # From tum_check.cpp


# ILP Mapping Function
ilp_map <- function(c) {
  lp(direction = "max", 
     objective.in = c, 
     const.mat = A, 
     const.dir = rep("<=", length(b)), 
     const.rhs = b,
     all.bin = TRUE)$solution
}

# Generate true coefficients
beta_true <- matrix(rnorm(p * d), p, d)
cat("True beta:\n"); print(beta_true)

# Generate latent response and binary observations
zeta_true <- X %*% beta_true + matrix(rnorm(n * d), n, d)
y <- t(apply(zeta_true, 1, ilp_map))

# Feasible support indicator and initial U
U_free <- 1 * (t(A %*% t(y) == b))
U <- matrix(rexp(n * m), n, m) * U_free

# -----------------------------
# Prior and Precomputations
# -----------------------------
b0 <- rep(0, p)
B0 <- 10
V <- solve(diag(1/B0, p) + t(X) %*% X)
L <- t(chol(V))

# -----------------------------
# Gibbs Sampler Setup
# -----------------------------
n_iter <- 50000
beta_samples <- array(NA, dim = c(n_iter, p, d))
beta <- matrix(0, p, d)
zeta <- matrix(NA, n, d)
UA <- U %*% A

# Initialize zeta
for (j in 1:d) {
  mu_j <- X %*% beta[, j]
  zeta[, j] <- rtruncnorm(n, 
                          a = ifelse(y[, j] == 1, UA[, j], -Inf), 
                          b = ifelse(y[, j] == 1, Inf, UA[, j]),
                          mean = mu_j, sd = 1)
}

# -----------------------------
# Gibbs Sampler Loop
# -----------------------------
running_time <- system.time({
  for (iter in 1:n_iter) {
    if (iter %% 100 == 0) {
      cat("MCMC iteration:", iter, "; Current AR:", mean(accept_or_not), "\n")
    }
    
    greaterequal <- 1 - y
    U <- loop_hit_and_run(t(A), zeta, greaterequal, U, U_free, n_iter = 100)
    UA <- U %*% A
    
    # Resample zeta
    zeta_new <- zeta
    sampled_indices <- 1:d
    for (j in sampled_indices) {
      mu_j <- X %*% beta[, j]
      zeta_new[, j] <- rtruncnorm(n, 
                                  a = ifelse(y[, j] == 1, UA[, j], -Inf), 
                                  b = ifelse(y[, j] == 1, Inf, UA[, j]),
                                  mean = mu_j, sd = 1)
    }
    
    zeta_tilde <- ifelse(y > 0.5, pmax(zeta, zeta_new), pmin(zeta, zeta_new))
    U_star <- loop_hit_and_run(t(A), zeta_tilde, greaterequal, U, U_free, n_iter = 100)
    accept_or_not <- check_feasible(t(A), zeta, greaterequal, U_star)
    zeta <- zeta_new * accept_or_not + zeta * (1 - accept_or_not)
    
    # Update beta
    mean_normal <- V %*% ((1/B0) * matrix(b0, p, d) + t(X) %*% zeta)
    beta <- mean_normal + L %*% matrix(rnorm(p * d), p, d)
    
    # Save draw
    beta_samples[iter, , ] <- beta
  }
})
cat(n_iter, "iterations of MCMC completed.\n")
print(running_time)

# -----------------------------
# Post-processing
# -----------------------------
# Save results (optional)
# saveRDS(beta_samples, file = sprintf("../Results/Simulation/iter%d_n%d_d%d_p%d_m%d_beta_samples.rds", n_iter, n, d, p, m))
# saveRDS(beta_true, file = sprintf("../Results/Simulation/iter%d_n%d_d%d_p%d_m%d_beta_true.rds", n_iter, n, d, p, m))

# Compute posterior means
burn_in <- n_iter / 2
beta_post_mean <- apply(beta_samples[(burn_in + 1):n_iter, , ], c(2, 3), mean)
cat("Posterior mean of beta:\n"); print(beta_post_mean)
cat("RMSE:\n"); print(sqrt(mean((beta_post_mean - beta_true)^2)))

# -----------------------------
# Trace Plots
# -----------------------------
d_plot <- min(d, 5)
par(mfrow = c(p, d_plot))
for (k in 1:p) {
  for (j in 1:d_plot) {
    plot(beta_samples[(burn_in + 1):n_iter, k, j], type = 'l',
         main = sprintf("Trace for beta[%d,%d]", k, j),
         xlab = "Iteration", ylab = sprintf("beta[%d,%d]", k, j))
    abline(h = beta_true[k, j], col = 'red')
  }
}

# -----------------------------
# ACF Plots
# -----------------------------
thin <- 10
thinned_indices <- seq(from = burn_in + 1, to = n_iter, by = thin)
par(mfrow = c(p, d_plot))
for (j in 1:d_plot) {
  for (k in 1:p) {
    acf(beta_samples[thinned_indices, k, j],
        main = sprintf("ACF for beta[%d,%d]", k, j))
  }
}
