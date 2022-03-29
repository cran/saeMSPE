mspeNERlnr = function(ni, X, Y, X.mean, method = "PR", var.method = "default"){
  # this function obtain the mspe approximation through Linearization method
  # basic function
  mspeNERPR = function(ni, X, Y, X.mean, sigv2, sige2){
    #calculate the mean of x and y within group
    xy.bar = aggregate(cbind(X,Y),by = list(rep(1:m,ni)),FUN = mean)[,-1]
    x.bar = as.matrix(xy.bar[,1:p])
    y.bar = as.matrix(xy.bar[,p+1])
    n = sum(ni)
    m = length(ni)
    p = ncol(X)
    gama = ni * sigv2/(ni * sigv2 + sige2)
    Vi.inv = list()
    Zi = list()
    g1 = c()
    g2 = c()
    g3 = c()
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
    nstar2 = (sum(diag(M %*% Zmat %*% t(Zmat))))^2
    vare = (2/(n - m - 2 + 1)) * sige2^2
    varv = (2/nstar^2) * ((1/(n - m - 2 + 1)) * (m - 1) * (n - p) * sige2^2 +
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
    return(mspe)
  }
  
  mspeNERDL = function(ni, X, Y, X.mean, sigv2, sige2){
    #calculate the mean of x and y within group
    xy.bar = aggregate(cbind(X,Y),by = list(rep(1:m,ni)),FUN = mean)[,-1]
    x.bar = as.matrix(xy.bar[,1:p])
    y.bar = as.matrix(xy.bar[,p+1])
    n = sum(ni)
    m = length(ni)
    p = ncol(X)
    Vi.inv = list()
    for (i in 1:m) {
      vtemp = 1/sige2 * diag(ni[i]) - sigv2/((ni[i]*sigv2+sige2)*sige2)*rep(1,ni[i])%*%t(rep(1,ni[i]))
      Vi.inv[[i]]<-vtemp
    }
    V.inv = as.matrix(bdiag(Vi.inv))
    gama = ni * sigv2 / (ni*sigv2 + sige2)
    g1temp = (1 - gama) * sigv2
    
    w = sige2 + ni * sigv2
    a = sum(ni^2 * w^(-2)) * sum((ni -1)* sige2^(-2) + w^(-2))-(sum(ni^2 * w^(-2)))^2
    Ivv = 2 * a^(-1) * sum((ni-1) * sige2^(-2) + w^(-2))
    Iee = 2 * a^(-1) * sum(ni^2 * w^(-2))
    Ive = -2 * a^(-1) * sum(ni * w^(-2))
    g1 = c();g2 = c();g3 = c();
    
    for (i in 1:m) {
      g1[i] = g1temp[i]
      g2[i] = t(X.mean[i, ] - gama[i] * x.bar[i, ])%*%solve(t(X) %*% V.inv %*% X)%*%(X.mean[i, ] - gama[i] * x.bar[i, ])
      g3[i] = (1/(ni[i]^2 * (sigv2 + sige2 /ni[i])^3)) * (sige2^2 * Ivv+sigv2^2 * Iee-2 * sige2 * sigv2 * Ive)
    }
    return(g1+g2+2*g3)
  }
  n = sum(ni)
  m = length(ni)
  p = ncol(X)
  if(method == "PR"){
    if(var.method != "MOM" & var.method != "default" ) stop("var.method is not available")
    # PR mspe approximation method (Prasad and Rao 1990)
    phat = varner(ni, X, Y, 1)
    sige2 = phat$sigehat2
    sigv2 = phat$sigvhat2
    mspe = mspeNERPR(ni, X, Y, X.mean, sigv2, sige2)
    return(mspe)
  }
  if(method == "DL"){
    if(var.method == "REML" | var.method == "default"){
      phat = varner(ni, X, Y, 2)
      sige2 = c(phat$sigehat2)
      sigv2 = c(phat$sigvhat2)
    }
    if(var.method == "ML"){
      phat = varner(ni, X, Y, 3)
      sige2 = phat$sigehat2
      sigv2 = phat$sigvhat2
    }
    mspe = mspeNERDL(ni, X, Y, X.mean, sigv2, sige2)
    return(mspe)
  }
  # Available Linearization method includes "PR","DL".
  else stop( "method is not available" )
}



