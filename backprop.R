library(tidyverse)

## Backprop

# Number of features (including bias)
nx = 4

# Training set size
m = 100

X = cbind(rep(1, m), replicate(nx - 1, rnorm(m)) )
gen_params <- rnorm(nx)
mu = X %*% gen_params
p_real <- 1/(1 + exp(-mu))
y = rbinom(size = 1, n = length(p_real), prob = p_real)

# Number of hidden nodes (including bias)
q = 3

theta = replicate(q, 0.01*rnorm(nx))
beta = 0.01*rnorm(q)

sigma_1 <- function(x){
  return(1/(1 + exp(-x)))
}

sigma_2 <- function(x){
  return(1/(1 + exp(-x)))
}

compute_p_hat <- function(theta, beta, X){
  A = sigma_1(X %*% theta)
  p_hat = sigma_2(A %*% beta)
  return(p_hat)
}

deviance <- function(p_hat, y){
  dev = - mean(y*log(p_hat) + (1-y)*log(1-p_hat))
  return(dev)
}


alpha = 0.01
i = 0
n_iter <- 200

deviance_df <- data_frame(dev = rep(0.0, n_iter))

while(i < n_iter){
  i = i + 1
  cat("Iter:", i, "\n")
  p_hat = compute_p_hat(theta, beta, X)
  deviance_df$dev[i] <- deviance(p_hat, y)
  
  p_minus_y = p_hat - y
  A = sigma_1(zeta)
  dL_dbeta = mean(t(A) %*% p_minus_y)
  zeta = X %*% theta
  
  dL_dtheta <- matrix(rep(NA, nx*q), ncol = q)
  for(l in 1:length(beta)){
    temp <- (beta[l] * p_minus_y * A[,l]) * (1 - A[,l])
    
    dl_dtheta_nl <- matrix(rep(temp, nx), ncol = nx, byrow = F) * X
    dL_dtheta[,l] <- colMeans(dl_dtheta_nl)
  }
  
  # # first beta
  # bbb <- cbind(  (beta[1] * p_minus_y * A[,1]) * (1 - A[,1]) * X[,1],
  #         (beta[1] * p_minus_y * A[,1]) * (1 - A[,1]) * X[,2],
  #         (beta[1] * p_minus_y * A[,1]) * (1 - A[,1]) * X[,3],
  #         (beta[1] * p_minus_y * A[,1]) * (1 - A[,1]) * X[,4])
  # 
  # # second beta
  # (beta[2] * p_minus_y * A[,2]) * (1 - A[,2]) * X[,1]
  # (beta[2] * p_minus_y * A[,2]) * (1 - A[,2]) * X[,2]
  # (beta[2] * p_minus_y * A[,2]) * (1 - A[,2]) * X[,3]
  # (beta[2] * p_minus_y * A[,2]) * (1 - A[,2]) * X[,4]
  # # third beta
  # (beta[3] * p_minus_y * A[,3]) * (1 - A[,3]) * X[,1]
  # (beta[3] * p_minus_y * A[,3]) * (1 - A[,3]) * X[,2]
  # (beta[3] * p_minus_y * A[,3]) * (1 - A[,3]) * X[,3]
  # (beta[3] * p_minus_y * A[,3]) * (1 - A[,3]) * X[,4]

  beta <- beta - alpha*dL_dbeta
  theta <- theta - alpha*dL_dtheta
}

deviance_df %>% 
  mutate(ix = 1:nrow(.)) %>%  
  ggplot(aes(ix, dev)) + 
  geom_point(size = 0.5) +
  geom_line()



