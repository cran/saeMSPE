\name{varOBP}
\alias{varOBP}
\title{
Estimate of the variance component using best predictive estimation method (BPE, also called OBP method) for Fay Herriot model.
}
\description{
This function returns the estimate of variance component with OBP method for Fay Herriot model. This function does not accept missing values.
}
\usage{
varOBP(Y, X, D)
}

\arguments{
  \item{Y}{
a numeric vector. It represents the response value for Fay Herriot model.
}
  \item{X}{
a numeric matrix. Stands for the available auxiliary values.
}
  \item{D}{
a numeric vector consisting of the known sampling variances of each of the small area levels.
}
}
\value{
This function returns a list of the variance component estimates based on OBP method.
}
\references{
X. Liu, H. Ma, and J. Jiang. That prasad-rao is robust: Estimation of mean squared prediction error of observed best predictor under potential model misspecification. \emph{Statistica Sinica}, 2020. 
}
\author{
Peiwen Xiao, Xiaohui Liu, Yuzi Liu, Jiming Jiang, and Shaochu Liu
}


\examples{
X = matrix(runif(10 * 3), 10, 3)
X[,1] = rep(1, 10) 
D = (1:10) / 10 + 0.5
Y = X \%*\% c(0.5,1,1.5) + rnorm(10, 0, sqrt(2)) + rnorm(10, 0, sqrt(D))
Ahat = varOBP(Y,X,D)
}

