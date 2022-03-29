\name{varner}
\alias{varner}
\title{
Estimates of the variance component using several methods for Nested error regression model.
}
\description{
This function returns the estimate of variance component with several existing method for Nested error regression model. This function does not accept missing values.
}
\usage{
varner(ni, X, Y, method)
}

\arguments{
  \item{ni}{
a numeric vector. It represents the sample number for every small area.
}
  \item{X}{
a numeric matrix. Stands for the available auxiliary values.
}
  \item{Y}{
a numeric vector. It represents the response value for Nested error regression model.
}
  \item{method}{
The variance component estimation method to be used. See "Details".
}
}
\value{
This function returns a list of the variance component estimates.
}
\details{
Default value for \code{method} is 1, It represents the moment estimator, Also called ANOVA estimator, The available variance component estimation method are list as follows:
\code{method = 1} represents the MOM estimator;
\code{method = 2} represents the restricted maximum likelihood(REML) estimator; \code{method = 3} represents the maximum likelihood(ML) estimator;  
}

\references{
J. Jiang. Linear and Generalized Linear Mixed Models and Their Applications. 2007.
}
\author{
Peiwen Xiao, Xiaohui Liu, Yuzi Liu, Jiming Jiang, and Shaochu Liu
}


\examples{
### parameter setting 
Ni = 1000; sigmaX = 1.5; m = 10
beta = c(0.5, 1)
sigma_v2 = 0.8; sigma_e2 = 1
ni = sample(seq(1,10), m,replace = TRUE); n = sum(ni)
p = length(beta)
### population function
pop.model = function(Ni, sigmaX, beta, sigma_v2, sigma_e2, m){
  x = rnorm(m * Ni, 1, sqrt(sigmaX)); v = rnorm(m, 0, sqrt(sigma_v2)); y = numeric(m * Ni)
  theta = numeric(m); kk = 1
  for(i in 1 : m){
    sumx = 0
    for(j in 1:Ni){
      sumx = sumx + x[kk]
      y[kk] = beta[1] + beta[2] * x[kk] + v[i] + rnorm(1, 0, sqrt(sigma_e2))
      kk = kk + 1
    }
    meanx = sumx/Ni
    theta[i] = beta[1] + beta[2] * meanx + v[i]
  }
  group = rep(seq(m), each = Ni)
  x = cbind(rep(1, m*Ni), x)
  data = cbind(x, y, group)
  return(list(data = data, theta = theta))
} 
### sample function
sampleXY = function(Ni, ni, m, Population){
  Indx = c()
  for(i in 1:m){
    Indx = c(Indx, sample(c(((i - 1) * Ni + 1) : (i * Ni)), ni[i]))
  }
  Sample = Population[Indx, ]; Nonsample = Population[-Indx, ]
  return(list(Sample, Nonsample))
} 
### data generation process
Population = pop.model(Ni, sigmaX, beta, sigma_v2, sigma_e2, m)$data
XY = sampleXY(Ni, ni, m, Population)[[1]]
X = XY[, 1:p]
Y = XY[, p+1]
### variance component estimate
result = varner(ni,X,Y,1)
}

