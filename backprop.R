library(tidyverse)

## Backprop


# # Number of hidden nodes (including bias)
# q = 3
# 
# theta = replicate(q, 0.01*rnorm(nx))
# beta = 0.01*rnorm(q)

sigma_1 <- function(x){
  return(1/(1 + exp(-x)))
}

sigma_2 <- function(x){
  return(1/(1 + exp(-x)))
}

# compute_p_hat <- function(theta, beta, X){
#   A = sigma_1(X %*% theta)
#   p_hat = sigma_2(A %*% beta)
#   return(p_hat)
# }

loss_function <- function(p_hat, y){
  lossf = - 2 * mean(y*log(p_hat) + (1-y)*log(1-p_hat))
  return(lossf)
}

predict <- function(ann, X){
  m = nrow(X)
  X = cbind(rep(1, m), X)
  zeta = X %*% ann$theta
  A = sigma_1(zeta)
  A_aug = cbind(rep(1, m), A)
  p_hat = sigma_2(A_aug %*% ann$beta)
  return(p_hat)
}

logistic_func <- function(x){
  1/(1 + exp(-x))
}



# 
# alpha = 0.01
# i = 0
# n_iter <- 200
# 
# convergence_monitor_df <- data_frame(lossf = rep(0.0, n_iter))
# 
# while(i < n_iter){
#   i = i + 1
#   cat("Iter:", i, "\n")
#   p_hat = compute_p_hat(theta, beta, X)
#   convergence_monitor_df$lossf[i] <- loss_function(p_hat, y)
#   
#   p_minus_y = p_hat - y
#   A = sigma_1(zeta)
#   dL_dbeta = mean(t(A) %*% p_minus_y)
#   zeta = X %*% theta
#   
#   dL_dtheta <- matrix(rep(NA, nx*q), ncol = q)
#   for(l in 1:length(beta)){
#     temp <- (beta[l] * p_minus_y * A[,l]) * (1 - A[,l])
#     
#     dl_dtheta_nl <- matrix(rep(temp, nx), ncol = nx, byrow = F) * X
#     dL_dtheta[,l] <- colMeans(dl_dtheta_nl)
#   }
#   
#   # # first beta
#   # bbb <- cbind(  (beta[1] * p_minus_y * A[,1]) * (1 - A[,1]) * X[,1],
#   #         (beta[1] * p_minus_y * A[,1]) * (1 - A[,1]) * X[,2],
#   #         (beta[1] * p_minus_y * A[,1]) * (1 - A[,1]) * X[,3],
#   #         (beta[1] * p_minus_y * A[,1]) * (1 - A[,1]) * X[,4])
#   # 
#   # # second beta
#   # (beta[2] * p_minus_y * A[,2]) * (1 - A[,2]) * X[,1]
#   # (beta[2] * p_minus_y * A[,2]) * (1 - A[,2]) * X[,2]
#   # (beta[2] * p_minus_y * A[,2]) * (1 - A[,2]) * X[,3]
#   # (beta[2] * p_minus_y * A[,2]) * (1 - A[,2]) * X[,4]
#   # # third beta
#   # (beta[3] * p_minus_y * A[,3]) * (1 - A[,3]) * X[,1]
#   # (beta[3] * p_minus_y * A[,3]) * (1 - A[,3]) * X[,2]
#   # (beta[3] * p_minus_y * A[,3]) * (1 - A[,3]) * X[,3]
#   # (beta[3] * p_minus_y * A[,3]) * (1 - A[,3]) * X[,4]
# 
#   beta <- beta - alpha*dL_dbeta
#   theta <- theta - alpha*dL_dtheta
# }
# 
# convergence_monitor_df %>% 
#   mutate(ix = 1:nrow(.)) %>%  
#   ggplot(aes(ix, lossf)) + 
#   geom_point(size = 0.5) +
#   geom_line()

ann <- function(X, y, q = 3, alpha = 0.01, n_iter = 200, init_beta = "random", init_theta = "random", seed = "random"){
  # X: data matrix
  # y: response vector
  # q: number of hidden nodes (including bias)
  # alpha: learning rate
  # n_iter: number of iterations
  
  m = nrow(X)
  X = cbind(rep(1, m), X)
  nx = ncol(X)
  
  
  i = 0
  
  if(init_beta == "random") {
    if(seed != "random") set.seed(seed)
    beta = 0.1*runif(q + 1, -0.5, 0.5)
    init_beta = beta
  } else{
    beta = init_beta
  }
  if(init_theta == "random"){
    if(seed != "random") set.seed(seed)
    theta = replicate(q, 0.1*runif(nx, -0.5, 0.5))
    init_theta = theta
  } else {
    theta = init_theta
  }
  
  convergence_monitor_df <- data_frame(
    iter = 1:n_iter,
    lossf = as.numeric(rep(NA, n_iter)),
    norm_diff_params = as.numeric(rep(NA, n_iter)),
    norm_gradient = as.numeric(rep(NA, n_iter))
    )
  
  while(i < n_iter){
    i = i + 1
    cat("Iter:", i, "\n")
    zeta = X %*% theta
    A = sigma_1(zeta)
    A_aug = cbind(rep(1, m), A)
    p_hat = sigma_2(A_aug %*% beta)
    #p_hat = compute_p_hat(theta, beta, X)
    
    p_minus_y = p_hat - y
    dL_dbeta = colMeans(matrix(rep(p_minus_y, q + 1), ncol = q + 1, byrow = F) * A_aug)
    #dL_dbeta = mean(t(A) %*% p_minus_y)
    
    dL_dtheta <- matrix(rep(NA, nx*q), ncol = q)
    for(l in 1:ncol(dL_dtheta)){
      temp <- (beta[l] * p_minus_y * A[,l]) * (1 - A[,l])
      
      dl_dtheta_nl <- matrix(rep(temp, nx), ncol = nx, byrow = F) * X
      dL_dtheta[,l] <- colMeans(dl_dtheta_nl)
    }
    
    theta_old <- theta
    beta_old <- beta
    
    beta <- beta - alpha*dL_dbeta
    theta <- theta - alpha*dL_dtheta
    
    norm_diff_beta_sq <- sum((beta - beta_old)^2)
    norm_diff_theta_sq <- sum((theta - theta_old)^2)
    
    convergence_monitor_df$lossf[i] <- loss_function(p_hat, y)
    convergence_monitor_df$norm_diff_params[i] <- sqrt(norm_diff_beta_sq + norm_diff_theta_sq)
    convergence_monitor_df$norm_gradient[i] <- sqrt(sum(dL_dbeta^2) + sum(dL_dtheta^2))

  }
  
  # gg_deviance_iter <- convergence_monitor_df %>% 
  #   ggplot(aes(iter, lossf)) + 
  #   geom_point(size = 0.5) +
  #   geom_line() +
  #   theme_bw()
  
  out <- list(
    beta = beta,
    theta = theta, 
    init_theta = init_theta,
    init_beta = init_beta,
    convergence_monitor_df = convergence_monitor_df
    #gg_deviance_iter = gg_deviance_iter
  )
  
  return(out)
}

### Example 1

# # Number of features (including bias)
# nx = 4
# 
# # Training set size
# m = 100
# 
# X = cbind(rep(1, m), replicate(nx - 1, rnorm(m)) )
# gen_params <- rnorm(nx)
# mu = X %*% gen_params
# p_real <- 1/(1 + exp(-mu))
# y = rbinom(size = 1, n = length(p_real), prob = p_real)
# 
# ann_1 <- ann(X, y, 3)


plot_convergence <- function(ann){
  
  gg_lossf <- ann$convergence_monitor_df %>% 
    ggplot(aes(iter, lossf)) + 
    geom_point(size = 0.5) +
    geom_line() +
    theme_bw()
  
  gg_norm_diff_params <- ann$convergence_monitor_df %>% 
    ggplot(aes(iter, norm_diff_params)) + 
    geom_point(size = 0.5) +
    geom_line() +
    theme_bw()
  
  gg_norm_gradient <- ann$convergence_monitor_df %>% 
    ggplot(aes(iter, norm_gradient)) + 
    geom_point(size = 0.5) +
    geom_line() +
    theme_bw()
  
  return(list(
    gg_lossf = gg_lossf,
    gg_norm_diff_params = gg_norm_diff_params,
    gg_norm_gradient = gg_norm_gradient
  ))
}


### Example 2

set.seed(2805721)
dat_2 <- data.frame(x_1 = runif(300, -2, 2)) %>% 
  mutate(y = rbinom(300, 1, logistic_func(2 - 3 * x_1^2)))

dat_p <- data.frame(x_2 = seq(-2, 2, 0.01)) %>% 
  mutate(p = logistic_func(2 - 3 * x_2^2))

dat_p %>% 
  ggplot() +
  geom_line(aes(x_2, p)) +
  geom_point(data = dat_2, aes(x = x_1, y = y), colour = 'red')

ann_2 <- ann(as.matrix(dat_2$x_1), dat_2$y, q = 4, alpha = 0.3, n_iter = 15000, seed = 2018)

ann_2

plot_convergence(ann_2)


#ann_2$gg_deviance_iter

## hacer feed forward con beta encontrados

predictions_2 <- predict(ann_2, as.matrix(dat_2$x_1))

dat_2 %>% 
  mutate(p_2 = predictions_2) %>% 
  ggplot(aes(x = x_1, y = p_2)) + 
  geom_line() +
  geom_line(data = dat_p, aes(x = x_2, y = p), col='red') + 
  ylim(c(0,1)) +
  geom_point(data = dat_2, aes(x = x_1, y = y)) +
  theme_bw()

data.frame(
  p = predict(ann_2, as.matrix(dat_2$x_1)),
  x = dat_2$x_1) %>% 
  ggplot() +
  geom_point(aes(x, p))



ann_2_1 <- ann(as.matrix(dat_2$x_1), dat_2$y, q = 4, alpha = 0.5, n_iter = 3000)
ann_2_1

ann_2_2 <- ann(as.matrix(dat_2$x_1), dat_2$y, q = 4, alpha = 0.5, n_iter = 3000, 
               init_beta = ann_2_1$beta, init_theta = ann_2_1$theta)
ann_2_2


predictions_2_1 <- predict(ann_2_1, as.matrix(dat_2$x_1))
predictions_2_2 <- predict(ann_2_2, as.matrix(dat_2$x_1))


dat_2 %>% 
  mutate(p_2 = predictions_2_1) %>% 
  ggplot(aes(x = x_1, y = p_2)) + 
  geom_line() +
  geom_line(data = dat_p, aes(x = x_2, y = p), col='red') + 
  ylim(c(0,1)) +
  geom_point(data = dat_2, aes(x = x_1, y = y)) +
  theme_bw()


dat_2 %>% 
  mutate(p_2 = predictions_2_2) %>% 
  ggplot(aes(x = x_1, y = p_2)) + 
  geom_line() +
  geom_line(data = dat_p, aes(x = x_2, y = p), col='red') + 
  ylim(c(0,1)) +
  geom_point(data = dat_2, aes(x = x_1, y = y)) +
  theme_bw()




### Example 3

dat_p_3 <- data.frame(x = seq(-2,2,0.05)) %>% 
  mutate(p = logistic_func(3 + x- 3*x^2 + 3*cos(4*x)))

set.seed(280572)

dat_3 <- data.frame(x = runif(300, -2, 2)) %>% 
  mutate(y = rbinom(300, 1, logistic_func(3 + x- 3*x^2 + 3*cos(4*x))))

dat_p_3 %>% 
  ggplot() +
  geom_line(aes(x, p), col = 'red') +
  geom_jitter(data = dat_3, aes(x = x, y = y), col ='black',
              position = position_jitter(height=0.05), alpha = 0.4) +
  theme_bw()



# ann_3_1 <- nnet::nnet(y ~ x, data = dat_3, size = 3, decay = 0.001, 
#                       entropy = T, maxit = 500)
# 
# dat_3 %>% 
#   mutate(pred = ann_3_1$fitted.values) %>% 
#   ggplot(aes(x = x, y = pred)) + 
#   geom_line() +
#   geom_line(data = dat_p_3, aes(x = x, y = p), col='red') + 
#   geom_jitter(data = dat_3, aes(x = x, y = y), col ='black',
#               position = position_jitter(height = 0.05), alpha = 0.4) +
#   theme_bw()


ann_3 <- ann(as.matrix(dat_3$x), dat_3$y, q = 4, alpha = 0.1, n_iter = 50000, seed = 2018)

ann_3

plot_convergence(ann_3)

#ann_2$gg_deviance_iter

## hacer feed forward con beta encontrados

predictions_3 <- predict(ann_3, as.matrix(dat_3$x))

dat_3 %>% 
  mutate(pred = predictions_3) %>% 
  ggplot(aes(x = x, y = pred)) + 
  geom_jitter(data = dat_3, aes(x = x, y = y), col ='black',
              position = position_jitter(height=0.05), alpha = 0.4) +
  geom_line(color = 'blue') +
  geom_line(data = dat_p_3, aes(x = x, y = p), col='red') + 
  theme_bw()

### Example 4

dat_p_4 <- data.frame(x = seq(-2,2,0.05)) %>% 
  mutate(p = logistic_func(3 + x- 3*x^2 + 3*cos(4*x)))

set.seed(1234)

dat_4 <- data.frame(x = runif(5000, -2, 2)) %>% 
  mutate(y = rbinom(5000, 1, logistic_func(3 + x- 3*x^2 + 3*cos(4*x))))

dat_p_4 %>% 
  ggplot() +
  geom_line(aes(x, p), col = 'red') +
  geom_jitter(data = dat_4, aes(x = x, y = y), col ='black',
              position = position_jitter(height=0.05), alpha = 0.4) +
  theme_bw()



# ann_4_1 <- nnet::nnet(y ~ x, data = dat_4, size = 3, decay = 0.001, 
#                       entropy = T, maxit = 500)
# 
# dat_4 %>% 
#   mutate(pred = ann_4_1$fitted.values) %>% 
#   ggplot(aes(x = x, y = pred)) + 
#   geom_line() +
#   geom_line(data = dat_p_4, aes(x = x, y = p), col='red') + 
#   geom_jitter(data = dat_4, aes(x = x, y = y), col ='black',
#               position = position_jitter(height = 0.05), alpha = 0.4) +
#   theme_bw()


ann_4 <- ann(as.matrix(dat_4$x), dat_4$y, q = 6, alpha = 0.1, n_iter = 50000, seed = 2018)

ann_4

plot_convergence(ann_4)

#ann_2$gg_deviance_iter

## hacer feed forward con beta encontrados

predictions_4 <- predict(ann_4, as.matrix(dat_4$x))

dat_4 %>% 
  mutate(pred = predictions_4) %>% 
  ggplot(aes(x = x, y = pred)) + 
  geom_jitter(data = dat_4, aes(x = x, y = y), col ='black',
              position = position_jitter(height=0.05), alpha = 0.4) +
  geom_line(color = 'blue') +
  geom_line(data = dat_p_4, aes(x = x, y = p), col='red') + 
  theme_bw()


# Example 5

p_5 <- function(x1, x2){
  logistic_func(-5 + 10*x1 + 10*x2 - 30*x1*x2)
}

expand.grid(x1 = seq(0, 1, 0.05), x2 = seq(0, 1, 0.05)) %>% 
  mutate(p = p_5(x1, x2)) %>% 
  ggplot(aes(x = x1, y = x2)) + 
  geom_tile(aes(fill = p))

set.seed(2018)

dat_5 <- data_frame(x1 = runif(5000, 0, 1), x2 = runif(5000, 0, 1)) %>%
  mutate(p = p_5(x1, x2)) %>%
  mutate(y = rbinom(5000, 1, p))

dat_5 %>% 
  ggplot(aes(x = x1, y = x2)) + 
  geom_point(aes(color = y)) +
  theme_bw()

ann_5 <- ann(as.matrix(dat_5[, c("x1", "x2")]), 
             dat_5$y, 
             q = 5, 
             alpha = 0.5, 
             n_iter = 15000, 
             seed = 2018)

plot_convergence(ann_5)


expand.grid(x1 = seq(0, 1, 0.05), x2 = seq(0, 1, 0.05)) %>% 
  mutate(p = p_5(x1, x2)) %>% 
  #rowwise %>% 
  mutate(pred = predict(ann_5, as.matrix(.[, c("x1", "x2")]))) %>% 
  ggplot(aes(x = x1, y = x2)) + 
  geom_tile(aes(fill = pred))

predictions_5 <- predict(ann_5, as.matrix(dat_5[, c("x1", "x2")]))

data.frame(pred = predictions_5, p = p_5(dat_5$x1, dat_5$x2)) %>% 
  ggplot() + 
  geom_point(aes(p, pred)) +
  geom_abline(slope = 1) +
  xlim(0, 1) +
  ylim(0, 1)



ann_5_2 <- ann(as.matrix(dat_5[, c("x1", "x2")]), 
               dat_5$y, 
               q = 5, 
               alpha = 0.5, 
               n_iter = 100000, 
               seed = 2018)

plot_convergence(ann_5_2)

predictions_5_2 <- predict(ann_5_2, as.matrix(dat_5[, c("x1", "x2")]))


expand.grid(x1 = seq(0, 1, 0.05), x2 = seq(0, 1, 0.05)) %>% 
  mutate(p = p_5(x1, x2)) %>% 
  #rowwise %>% 
  mutate(pred = predict(ann_5_2, as.matrix(.[, c("x1", "x2")]))) %>% 
  ggplot(aes(x = x1, y = x2)) + 
  geom_tile(aes(fill = pred))

