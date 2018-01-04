 #include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]

List compute_gradient(NumericMatrix X, 
                      NumericVector p_hat, 
                      NumericVector y, 
                      NumericVector beta,
                      NumericMatrix A) {
  int q = beta.size() - 1, nx = X.ncol(), m = X.nrow();
  
  NumericVector dL_dbeta(q+1);
  NumericMatrix dL_dtheta(nx, q);
  
  // NumericVector dbeta0_vec(m);
  // NumericVector dbeta_vec(m);
  // NumericVector dtheta_vec(m);
  
  double dbeta;
  double pk_minus_yk;
  double sum_beta0;
  double sum_theta;
  double sum_beta;
  for(int l = 0; l < q; l++){
    for(int n = 0; n < nx; n++){
      sum_theta = 0;
      sum_beta = 0;
      sum_beta0 = 0;
      
      // for(int k = 0; k < m; k++){
      //   pk_minus_yk = p_hat(k) - y(k);
      //   dbeta = pk_minus_yk*A(k,l);
      //   dbeta0_vec(k) = pk_minus_yk;
      //   dbeta_vec(k) = dbeta;
      //   dtheta_vec(k) = dbeta*beta(l+1)*(1-A(k,l))*X(k,n);
      // } // end for k to m
      
      for(int k = 0; k < m; k++){
        pk_minus_yk = p_hat(k) - y(k);
        dbeta = pk_minus_yk*A(k,l);
        sum_theta = sum_theta + dbeta*beta(l+1)*(1-A(k,l))*X(k,n);
        sum_beta = sum_beta + dbeta;
        sum_beta0 = sum_beta0 + pk_minus_yk;
      } // end for k to m
      
      // Rcout << "n: " << n << std::endl;
      // Rcout << "l: " << l << std::endl;
      // Rcout << "Sum theta: " << sum_theta << std::endl;
      
      dL_dtheta(n,l) = sum_theta/m;
      // dL_dtheta(n,l) = sum(dtheta_vec)/m;
    } // end for n to nx
    dL_dbeta(l+1) = sum_beta/m;
  } // end for l to q
  dL_dbeta(0) = sum_beta0/m;
  
  //Rcout << "dtheta: " << dL_dtheta << std::endl;
  
  return List::create(
    dL_dbeta,
    dL_dtheta
  );
}