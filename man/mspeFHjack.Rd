\name{mspeFHjack}
\alias{mspeFHjack}
\title{
Compute MSPE through Jackknife method for Fay Herriot model
}
\description{
This function returns MSPE estimator with jackknife method for Fay Herriot model.
}
\usage{
mspeFHjack(Y, X, D, method = 2)
}

\arguments{
  \item{Y}{
a numeric vector. It represents the response value for Fay Herriot model.
}
  \item{X}{
a numeric matrix. Stands for the available auxiliary values.
}
  \item{D}{
a numeric vector. It represents the knowing sampling variance for Fay Herriot model.
}
  \item{method}{
It represents the variance component estimation method. See "Details"
}

}

\details{
This method was proposed by J. Jiang and L. S. M. Wan, jackknife method is used to obtain the bias and variation of estimators.

Default value for \code{method} is 2, \code{method = 2} represents the REML method and \code{method = 1} represents MOM method.

}

\value{
This function returns a vector of the MSPE estimates based on jackknife method.
}

\references{
M. H. Quenouille. Approximate tests of correlation in time series. \emph{Journal of the Royal Statistical Society. Series B (Methodological)}, 11(1):68-84, 1949.

J. W. Tukey. Bias and confidence in not quite large samples. \emph{Annals of Mathematical Statistics}, 29(2):614, 1958.

J. Jiang and L. S. M. Wan. A unified jackknife theory for empirical best prediction with m estimation. \emph{Annals of Statistics}, 30(6):1782-1810, 2002.

}



\author{
Peiwen Xiao, Xiaohui Liu, Yuzi Liu, Jiming Jiang, and Shaochu Liu
}


\examples{
X = matrix(runif(10 * 3), 10, 3)
X[,1] = rep(1, 10) 
D = (1:10) / 10 + 0.5
Y = X \%*\% c(0.5,1,1.5) + rnorm(10, 0, sqrt(2)) + rnorm(10, 0, sqrt(D))
result = mspeFHjack(Y,X,D,method = 2)
}

