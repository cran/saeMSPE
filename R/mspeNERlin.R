# The arguments of this function are
# (i)   a numeric vector. It represents the sample number for every small area
# (ii)  a numeric matrix. Stands for the available auxiliary values.
# (iii) a numeric vector. It represents the response value for Nested error regression model.
# (iV)  a numeric matrix. Stands for the population mean of auxiliary values.
# The function below gives the Prasad & Rao (1990) MSPE estimator
#' @rdname mspeNERlin
#' @export
mspeNERPR = function(ni, X, Y, X.mean, var.method = "default"){
  if(var.method != "MOM" & var.method != "default" ) stop("var.method is not available")
  phat = varner(ni, X, Y, 1)
  sige2 = phat$sigehat2
  sigv2 = phat$sigvhat2
  bhat = phat$bhat
  #calculate the mean of x and y within group
  xy.bar = aggregate(cbind(X,Y),by = list(rep(1:m,ni)),FUN = mean)[,-1]
  x.bar = as.matrix(xy.bar[,1:p])
  y.bar = as.matrix(xy.bar[,p+1])
  n = sum(ni); m = length(ni); p = ncol(X)
  gama = ni * sigv2/(ni * sigv2 + sige2)
  Vi.inv = Zi = list()
  g1 = g2 = g3 = c()
  xmat = 0
  for(j in 1:m){
    xmat = xmat + ni[j]^2 * x.bar[j, ] %*% t(x.bar[j, ])
  }
  ntemp = solve(t(X) %*% X)
  nstar = n - sum(diag(ntemp %*% xmat))
  for(t in 1:m){
    vtemp = 1/sige2 * (diag(ni[t]) - (gama[t]/ni[t]) * rep(1,ni[t]) %*% t(rep(1,ni[t])))
    Vi.inv[[t]] = vtemp
    Zi[[t]] = rep(1,ni[t])
  }
  V.inv = as.matrix(bdiag(Vi.inv))
  Zmat = as.matrix(bdiag(Zi))
  M = diag(n) - X %*% ntemp %*% t(X)
  temp = M %*% Zmat %*% t(Zmat)
  nstar2 = sum(temp * t(temp))
  vare = (2/(n - m - ncol(X) + 1)) * sige2^2
  varv = (2/nstar^2) * ((1/(n - m - ncol(X) + 1)) * (m - 1) * (n - ncol(X)) * sige2^2 +
                          2 * nstar * sige2 * sigv2 + nstar2 * sigv2^2)
  covev = -(m - 1) * (1/nstar) * vare
  msePR = numeric(m)
  for(i in 1:m){
    g1[i] = (1 - gama[i]) * sigv2
    g2[i] = t(X.mean[i, ] - gama[i] * x.bar[i, ])%*%solve(t(X) %*% V.inv %*% X)%*%(X.mean[i, ] - gama[i] * x.bar[i, ])
    g3[i] = (1/(ni[i]^2 * (sigv2 + sige2 /ni[i])^3)) *
      (sige2^2 * varv + sigv2^2 *
         vare - 2 * sige2 * sigv2 * covev)
  }
  mspe = g1 + g2 + 2 * g3
  return(list(MSPE = mspe, bhat = bhat, sigvhat2 = sigv2, sigehat2 = sige2))
}

# The arguments of this function are
# (i)   a numeric vector. It represents the sample number for every small area
# (ii)  a numeric matrix. Stands for the available auxiliary values.
# (iii) a numeric vector. It represents the response value for Nested error regression model.
# (iV)  a numeric matrix. Stands for the population mean of auxiliary values.
# The function below gives the Datta & Lahiri (2000) MSPE estimator
#' @rdname mspeNERlin
#' @export
mspeNERDL = function(ni, X, Y, X.mean, var.method = "default"){
  if(var.method == "REML" | var.method == "default"){
    phat = varner(ni, X, Y, 2)
    sige2 = c(phat$sigehat2)
    sigv2 = c(phat$sigvhat2)
    bhat = phat$bhat
  } else{
    if(var.method == "ML"){
      phat = varner(ni, X, Y, 3)
      sige2 = phat$sigehat2
      sigv2 = phat$sigvhat2
      bhat = phat$bhat
    } else stop( "var.method is not available")
  }
  
  #calculate the mean of x and y within group
  xy.bar = aggregate(cbind(X,Y),by = list(rep(1:m,ni)),FUN = mean)[,-1]
  x.bar = as.matrix(xy.bar[,1:p])
  y.bar = as.matrix(xy.bar[,p+1])
  n = sum(ni); m = length(ni); p = ncol(X)
  Vi.inv = list()
  for (i in 1:m) {
    vtemp = 1/sige2 * diag(ni[i]) - sigv2/((ni[i]*sigv2+sige2)*sige2)*rep(1,ni[i])%*%t(rep(1,ni[i]))
    Vi.inv[[i]]<-vtemp
  }
  V.inv = as.matrix(bdiag(Vi.inv))
  gama = ni * sigv2 / (ni*sigv2 + sige2)
  g1temp = (1 - gama) * sigv2
  w = sige2 + ni * sigv2
  a = sum(ni^2 * w^(-2)) * sum((ni -1)* sige2^(-2) + w^(-2))-(sum(ni * w^(-2)))^2
  Ivv = 2 * a^(-1) * sum((ni-1) * sige2^(-2) + w^(-2))
  Iee = 2 * a^(-1) * sum(ni^2 * w^(-2))
  Ive = -2 * a^(-1) * sum(ni * w^(-2))
  g1 = g2 = g3 = c();
  for (i in 1:m) {
    g1[i] = g1temp[i]
    g2[i] = t(X.mean[i, ] - gama[i] * x.bar[i, ])%*%solve(t(X) %*% V.inv %*% X)%*%(X.mean[i, ] - gama[i] * x.bar[i, ])
    g3[i] = (1/(ni[i]^2 * (sigv2 + sige2 /ni[i])^3)) * (sige2^2 * Ivv+sigv2^2 * Iee-2 * sige2 * sigv2 * Ive)
  }
  mspe = g1+g2+2*g3
  return(list(MSPE = mspe, bhat = bhat, sigvhat2 = sigv2, sigehat2 = sige2))
}

# A function to compute MSPE estimator for FH model
# The arguments of this function are
# (i)   a numeric vector. It represents the sample number for every small area
# (ii)  a numeric matrix. Stands for the available auxiliary values.
# (iii) a numeric vector. It represents the response value for Nested error regression model.
# (Vi)  a numeric matrix. Stands for the population mean of auxiliary values.
# (V)   The MSPE estimation method to be used
# (Vi)  The variance component estimation method to be used
#' @export
mspeNERlin = function(ni, X, Y, X.mean, method = "PR", var.method = "default"){
  ni = as.vector(ni)
  Y = as.vector(Y)
  X = as.matrix(X)
  X.mean = as.matrix(X.mean)
  n = sum(ni); m = length(ni); p = ncol(X)
  if(m != nrow(X.mean)){stop( "rowlength of population group mean of response doesnot match small area number" )}
  else{if(p != ncol(X.mean)){stop( "collength of population group mean of response doesnot match the dimension of response" )}
  else{if(n != length(Y) | n != nrow(X)){stop( "sample number (X or Y) doesnot match sum of small area sample numbers" )}else{
        if(method == "PR"){
          # PR mspe approximation method (Prasad and Rao 1990)
          mspe = mspeNERPR(ni, X, Y, X.mean, var.method)
          return(mspe)
        }
        if(method == "DL"){
          # DL mspe approximation method (Datta and Lahiri 1990)
          mspe = mspeNERDL(ni, X, Y, X.mean, var.method)
          return(mspe)
        }
        # Available Linearization method includes "PR","DL".
        else stop( "method is not available" )
      }
    }
  }
}



