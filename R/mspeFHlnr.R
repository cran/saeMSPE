varOBP= function(Y, X, D){
  qfun = function(A){
    m = nrow(X)
    gamma2 = diag((D/(A+D))^2)
    betahat = solve(t(X) %*% gamma2 %*% X) %*% (t(X) %*% gamma2 %*% Y)
    res0 = Y - X %*% betahat
    sum0 = sum(D/(A+D))
    temp = t(res0) %*% gamma2 %*% res0
    return(temp + 2 * A *sum0)
  }
  Ahat = nlminb(sd(Y), qfun, lower = 0)$par
  return(Ahat)
}

mspeFHlnr = function(Y, X, D, method = "PR", var.method = "default"){
  # this function obtain the mspe approximation through Linearization method
  # basic function
  mspeFHPR = function(Y, X, D, A){
    m = length(Y)
    p = ncol(X)
    g1iA = c()
    g2iA = c()
    g3iA = c()
    for(i in 1:m){
      g1iA[i] = A * D[i] / ( A+D[i] )
      g2iA[i] = ( ( D[i] / ( A + D[i] ) ) ^ 2) * t( X [i,]) %*% solve(t( X ) %*% diag((A + D)^(-1)) %*%  X  ) %*%  X [i,]
      varA    = 2 * m^(-1)*(A^2 + 2 * A * sum(D) / m + sum( D^2 ) / m )
      g3iA[i] = ( D[i]^2 ) / ( ( A + D[i] )^3 ) * varA
    }
    mspe = g1iA + g2iA + g3iA
    return(mspe)
  }
  
  mspeFHDL=function(Y, X, D, A){
    m = length(Y)
    p = ncol(X)
    g1iA = c()
    g2iA = c()
    g3iA = c()
    for(i in 1:m){
      g1iA[i] = A * D[i]/( A + D[i] )
      temp = matrix(0,p,p)
      for (u in 1:m) {
        temp = temp +  X [u,] %*% t( X [u,])/(A + D[u])
      }
      g2iA[i] = (( D[i] / (A + D[i] ))^2) * t( X [i,] ) %*% solve( temp ) %*%X [i,]
      g3iA[i] = ( D[i]^2)/((A + D[i])^3)/(sum((A + D)^-2))
    }
    mspe = g1iA + g2iA + 2 * g3iA
    return(mspe)
  }
  
  mspeFHDRS = function(Y, X, D, A){
    m = length(Y)
    p = ncol(X)
    trsum1 = sum((A + D )^(-1))
    trsum2 = sum((A + D )^(-2))
    B =  D /( D + A)
    b = 2*(m * trsum2 - trsum1^2)/(trsum1^3)
    B2b = B^2*b
    g1iA = c()
    g2iA = c()
    g3iA = c()
    for(i in 1:m){
      g1iA[i] = A * D [i]/(A + D [i])
      temp = matrix(0, p, p)
      for (u in 1:m) {
        temp = temp +  X [u,] %*% t( X [u,]) / (A +  D [u])
      }
      g2iA[i] = (( D [i]/(A +  D [i]))^2) * t( X [i,]) %*% solve(temp) %*%  X [i,]
      g3iA[i] = 2 * m * ( D [i]^2) * (A +  D [i])^(-3) / (trsum1^2)
    }
    mspe = g1iA + g2iA + 2 * g3iA - B2b
    return(mspe)
  }
  
  mspeFHMPR = function(Y, X, D, A){
    m = length(Y)
    p = ncol(X)
    V.inv = diag(1/(A+D))
    betA = solve(t(X)%*%V.inv%*%X)%*%t(X)%*%V.inv%*%Y
    AD = A + D
    AD2 = AD^2
    hatr = D / AD; hatr2 = hatr^2
    S0m = sum(hatr2); S1m = sum(hatr2 / AD); S2m = sum(hatr2 / AD2)
    res   = numeric(m)
    for(jj in 1:m){res[jj] = Y[jj] - sum(X[jj, ] * betA)}
    temp = hatr^3 * D
    U0m = p * sum(temp); U1m = p * sum(temp / AD)
    hatr4 = hatr^4
    temp = hatr4 * (res^4 / AD2 - 1)
    V0m = sum(temp)
    V1m = sum(temp / AD)
    tm = sum(1 / AD2)
    Tm = sum(res^4 / AD2) - 3 * m
    hata = Tm / (tm * S1m) * hatr4 / AD2
    hatc = A * hatr
    hatbd = hatr2 * (2 * U1m / (S0m * S1m) + 3 * V1m / S1m^2 -
                       3 * S2m * V0m / S1m^3 +
                       2 * V0m / (AD * S1m^2))
    return(-2 * hata + hatbd + hatc)
  }
  ########################
  #some exception handling
  ########################
  if(method == "PR"){
    if(var.method != "MOM" & var.method != "default" ) stop("var.method is not available")
    # PR mspe approximation method (Prasad and Rao 1990)
    A = smallarea::prasadraoest(Y, X, D)$estimate
    mspe = mspeFHPR(Y, X, D, A)
    return(mspe)
  }
  if(method == "DL"){
    if(var.method == "REML" | var.method == "default" ){
      A = smallarea::resimaxilikelihood( Y, X , D , 1000)$estimate
    }
    else if(var.method == "ML"){
      A = smallarea::maximlikelihood( Y, X , D )$estimate
    } else stop("var.method is not available")
    # DL mspe approximation method (Datta and Lahiri 2000)
    mspe = mspeFHDL(Y, X, D, A)
    return(mspe)
  }
  if(method == "DRS"){
    # DRS mspe approximation method (Datta ,Rao ,Smith 2005)
    if(var.method != "EB" & var.method != "default" ) stop("var.method is not available")
    A = smallarea::fayherriot( Y , X , D )$estimate
    mspe = mspeFHDRS(Y, X, D, A)
    return(mspe)
  }
  if(method == "MPR"){
    # MPR mspe approximation method (Liu 2020)
    if(var.method != "OBP" & var.method != "default" ) stop("var.method is not available")
    A = varOBP(Y, X, D)
    mspe = mspeFHMPR(Y, X, D, A)
    return(mspe)
  }
  # Available Linearization method includes "PR","DL","DRS","MPR".
  else stop( "method is not available" )
}


