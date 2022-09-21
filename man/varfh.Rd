\name{varfh}
\alias{varfh}
\alias{varOBP}
\title{
Estimates of the variance component using several methods for Fay Herriot model.
}
\description{
This function returns the estimate of variance component with several existing method for Fay Herriot model. This function does not accept missing values.
}
\usage{
varfh(Y, X, D, method)

varOBP(Y, X, D)
}

\arguments{
  \item{Y}{
(vector). It represents the response value for Fay Herriot model.
}
  \item{X}{
(matrix). It stands for the available auxiliary values.
}
  \item{D}{
(vector). It represents the knowing sampling variance for Fay Herriot model.
}
  \item{method}{
Variance component estimation method. See "Details".
}
}
\value{
This function returns a list with components:
\item{bhat}{(vector) Estimates of the unknown regression coefficients.}
\item{Ahat}{(numeric) Estimates of the variance component.}
}
\details{
Default value for \code{method} is 1, It represents the moment estimator, Also called ANOVA estimator, The available variance component estimation method are list as follows:

\code{method = 1} represents the moment (MOM) estimator, ; 

\code{method = 2} represents the restricted maximum likelihood (REML) estimator; 

\code{method = 3} represents the maximum likelihood (ML) estimator;  

\code{method = 4} represents the empirical bayesian (EB) estimator;  

}

\references{
J. Jiang. Linear and Generalized Linear Mixed Models and Their Applications. 2007.
}
\author{
Peiwen Xiao, Xiaohui Liu, Yuzi Liu, Jiming Jiang, and Shaochu Liu
}


\examples{
X = matrix(runif(10 * 3), 10, 3)
X[,1] = rep(1, 10) 
D = (1:10) / 10 + 0.5
Y = X \%*\% c(0.5,1,1.5) + rnorm(10, 0, sqrt(2)) + rnorm(10, 0, sqrt(D))
varOBP(Y, X, D)
varfh(Y, X, D, 1)
}

