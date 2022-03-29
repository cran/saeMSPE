#include <RcppArmadillo.h>
#include <iostream>
#include <armadillo>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace std;



// estimation function for MOM

List varnerMOM(arma::vec ni, arma::mat X, arma::vec Y){
  int m = ni.size(); int n = sum(ni); int p = X.n_cols;
  arma::mat xbar(m, p); arma::vec ybar(m); arma::vec ystar(n); arma::mat xstar(n, p);
  // calculate xbar and ybar through ni, X, Y
  for(int i = 0; i < m; i++){
    int nstart = sum(ni(arma::span(0,i))) - ni(i);
    if(ni(i) == 1){
      xbar.row(i) = X.row(nstart);
      ybar(i) = Y(nstart);
    }
    else {
      xbar.row(i) = mean(X.rows(nstart, nstart + ni(i) - 1), 0);
      ybar(i) = mean(Y(arma::span(nstart, nstart + ni(i) - 1)));
    }
    ystar(arma::span(nstart, nstart + ni(i) - 1)) = Y(arma::span(nstart, nstart + ni(i) - 1)) - ybar(i);
    xstar.rows(nstart, nstart + ni(i) - 1) = X.rows(nstart, nstart + ni(i) - 1) - arma::ones(ni(i), 1) * xbar.row(i);
  }
  arma::mat xmat = arma::zeros(p, p);
  for(int j = 0; j < m; j++){
    xmat = xmat + pow(ni(j),2) * xbar.row(j).t() * xbar.row(j);
  }
  arma::mat xstar1 = xstar.cols(1,p-1);
  arma::vec estar = ystar - xstar1 * inv(xstar1.t() * xstar1) * xstar1.t() * ystar;
  arma::vec u = Y - X * inv(X.t() * X) * X.t() * Y;
  arma::mat ntemp = inv(X.t() * X);
  double nstar = n - sum(diagvec(ntemp * xmat));
  arma::vec temp = pow(estar,2);
  double sige2 = sum(temp) / (n - m - p + 1); double sigv2;
  arma::vec temp1 = pow(u, 2);
  double temp2 = sum(temp1) - (n - 2) * sige2;
  if(temp2 < 0){sigv2 = 0;}
  else{sigv2 = temp2/nstar;}
  return List::create(Named("sigehat2") = sige2, Named("sigvhat2") = sigv2);
}

// estiarma::mation function for REML
double Ofun(double gamma, int m, int p, int n, arma::vec ni, arma::mat X, arma::mat Y, arma::mat xdot, arma::mat ydot,
            arma::mat tXX_inv, arma::mat temp1, arma::mat temp2){
  arma::mat tempd1 = arma::zeros(m, m);
  arma::mat tempd2 = arma::zeros(m, m);

  for(int g=0; g<m; g++){
    tempd1(g,g) = ni(g) + 1.0/gamma;
    tempd2(g,g) = 1.0 + ni(g) * gamma;
  }
  arma::mat temp3 = inv(tempd1 - xdot * tXX_inv * xdot.t());
  arma::vec temp4vec = (temp1 - temp2 * temp3 * temp2.t())/(n-p);
  double temp4 = temp4vec(0);
  double temp5 = det(tempd2 - gamma * xdot * tXX_inv * xdot.t());
  double val = (n-p) * log(temp4) + log(temp5);
  return val;
}

List varnerREML(arma::vec ni, arma::mat X, arma::vec Y){
  int m = ni.size(); int n = sum(ni); int p = X.n_cols;
  arma::mat xdot(m,p); arma::vec ydot(m); arma::mat xbar(m, p); arma::vec ybar(m);
  // calculate xbar and ybar through ni, X, Y
  for(int i = 0; i < m; i++){
    int nstart = sum(ni(arma::span(0,i))) - ni(i);
    if(ni(i) == 1){
      xbar.row(i) = X.row(nstart);
      ybar(i) = Y(nstart);
    }
    else {
      xbar.row(i) = mean(X.rows(nstart, nstart + ni(i) - 1), 0);
      ybar(i) = mean(Y(arma::span(nstart, nstart + ni(i) - 1)));
    }
  }
  for(int i = 0; i < m; i++){
    xdot.row(i) = ni(i) * xbar.row(i);
    ydot(i) = ni(i) * ybar(i);
  }
  arma::mat tXX_inv = inv(X.t() * X);
  arma::mat pmat = arma::eye(n, n) - X * tXX_inv * X.t();
  arma::mat temp1 = Y.t() * pmat * Y;
  arma::mat temp2 = ydot.t() - Y.t() * X * tXX_inv * xdot.t();
  arma::vec aa(1000);
  for(int i = 0; i < 1000; i++){
    aa(i) = 2.0 * (i+1)/1000;
  }
  arma::vec bb(1000);
  for(int j = 0; j < 1000; j++){
    bb(j) = Ofun(aa(j), m, p, n, ni, X, Y, xdot, ydot, tXX_inv, temp1, temp2);
  }
  int minIndx = bb.index_min();
  double ghat = aa(minIndx);
  arma::mat tempg = arma::zeros(m, m);;
  for(int g=0; g<m; g++){
    tempg(g,g) = ni(g) + 1.0/ghat;
  }
  arma::mat temp6 = inv(tempg - xdot * tXX_inv * xdot.t());
  arma::vec tempe = (temp1 - temp2 * temp6 * temp2.t())/(n-p);
  double sige2 = tempe(0);
  double sigv2 = ghat * sige2;
  arma::mat mata = arma::zeros(m, m);
  for(int a=0; a<m; a++){
    mata(a,a) = ghat/(1.0 + ni(a) * ghat);
  }
  arma::mat temp7 = X.t() * X - xdot.t() * mata * xdot;
  arma::mat temp8 = X.t() * Y - xdot.t() * mata * ydot;
  arma::vec bhat = inv(temp7) * temp8;
  return List::create(Named("bhat") = bhat, Named("sigehat2") = sige2, Named("sigvhat2") = sigv2);
}

// estimation function for ML
double gamfun(arma::vec ni, arma::mat X, arma::vec Y, arma::mat xbar, arma::vec ybar, arma::mat xdot, arma::vec ydot, double gamma){
  int m = ni.size(); int n = sum(ni);
  arma::mat amata = arma::zeros(m, m);
  for(int i = 0; i < m; i++){
    amata(i,i) = gamma/(1.0 + ni(i) * gamma);
  }
  arma::mat temp1 = X.t() * X - xdot.t() * amata * xdot;
  arma::mat temp2 = X.t() * Y - xdot.t() * amata * ydot;
  arma::vec btil = inv(temp1) * temp2;
  arma::vec temp3 = pow(Y - X * btil, 2);
  arma::vec temp4 = pow(ydot - xdot * btil, 2);
  arma::vec a = gamma / (1 + ni * gamma);
  double sige2 = (sum(temp3) - sum(temp4 % a)) / n;
  double val = n * log(abs(sige2)) + m * log(1 + gamma * n / m);
  return val;
}

List varnerML(arma::vec ni, arma::mat X, arma::vec Y){
  int m = ni.size(); int n = sum(ni); int p = X.n_cols;
  arma::mat xbar(m, p); arma::vec ybar(m); arma::vec ydot(m); arma::mat xdot(m, p);
  // calculate xbar and ybar through ni, X, Y
  for(int i = 0; i < m; i++){
    int nstart = sum(ni(arma::span(0,i))) - ni(i);
    if(ni(i) == 1){
      xbar.row(i) = X.row(nstart);
      xdot.row(i) = X.row(nstart);
      ybar(i) = Y(nstart);
      ydot(i) = Y(nstart);
    }
    else {
      xbar.row(i) = mean(X.rows(nstart, nstart + ni(i) - 1), 0);
      xdot.row(i) = xbar.row(i) * ni(i);
      ybar(i) = mean(Y(arma::span(nstart, nstart + ni(i) - 1)));
      ydot(i) = ybar(i) * ni(i);
    }
  }
  arma::vec aa(1000); arma::vec bb(1000);
  for(int i = 0; i < 1000; i++){
    aa(i) = (i+1) * 0.002;
    bb(i) = gamfun(ni, X, Y, xbar, ybar, xdot, ydot, aa(i));
  }
  double ghat = aa(bb.index_min());
  arma::vec a = ghat / (1 + ni * ghat);
  arma::mat amata = diagmat(a);
  arma::mat temp1 = X.t() * X - xdot.t() * amata * xdot;
  arma::mat temp2 = X.t() * Y - xdot.t() * amata * ydot;
  arma::vec bhat = inv(temp1) * temp2;
  arma::vec temp3 = pow(Y - X * bhat, 2);
  arma::vec temp4 = pow(ydot - xdot * bhat, 2);
  double sige2 = (sum(temp3) - sum(temp4 % a)) / n;
  double sigv2 = ghat * sige2;
  return List::create(Named("bhat") = bhat, Named("sigehat2") = sige2, Named("sigvhat2") = sigv2);
}

List varnerEB(arma::vec ni, arma::mat X, arma::vec Y){
  int m = ni.size(); int p = X.n_cols; int n = sum(ni);
  arma::mat xbar(m, p); arma::vec ybar(m);
  for(int i = 0; i < m; i++){
    int nstart = sum(ni(arma::span(0,i))) - ni(i);
    if(ni(i) == 1){
      xbar.row(i) = X.row(nstart);
      ybar(i) = Y(nstart);
    }
    else {
      xbar.row(i) = mean(X.rows(nstart, nstart + ni(i) - 1), 0);
      ybar(i) = mean(Y(arma::span(nstart, nstart + ni(i) - 1)));
    }
  }
  arma::vec Yseq(n);
  arma::mat Xseq = arma::ones(n, p);
  int ii = 0;
  for(int i = 0; i<m; i++){
    for(int j=0; j<ni(i); j++){
      Yseq(ii) = Y(ii) - ybar(i);
      Xseq.row(ii) = X.row(ii) - xbar.row(i);
      ii = ii + 1;
    }
  }
  Xseq.col(0) = arma::ones(n, 1);
  arma::vec betaols = inv(Xseq.t() * Xseq) * Xseq .t() * Yseq;
  arma::vec ehat = Yseq - Xseq * betaols;
  double sige2 = as_scalar(ehat.t() * ehat /(n - m -2));
  arma::vec b(m); arma::vec u_delta(m); arma::vec d(m);
  arma::mat invtXX = inv(X.t() *X);
  arma::mat ni2mat = diagmat(pow(ni,2));
  for(int i=0; i<m; i++){
    u_delta(i) = as_scalar(ybar(i) - xbar.row(i) * invtXX * X.t() * Y);
    arma::vec temp2 = (1 - 2 * ni(i) * xbar.row(i) * invtXX * xbar.row(i).t() + xbar.row(i) *
      invtXX  * (xbar.t() * ni2mat * xbar) * invtXX * xbar.row(i).t());
    b(i) = temp2(0);
    d(i) = as_scalar((1 - ni(i) * xbar.row(i) * invtXX * xbar.row(i).t())/ni(i));

  }
  double temphat1 = 0.0; double temphat2 = 0.0; double temphat3 = 0.0;
  for(int i=0; i<m; i++){
    temphat1 = temphat1 + ni(i) * pow(u_delta(i),2);
    temphat2 = temphat2 + ni(i) * b(i);
    temphat3 = temphat3 + ni(i) * d(i);
  }
  double mhat = temphat1/temphat2;
  double c = temphat3/ temphat2;
  double  sigv2 = mhat - c * sige2;
  if(sigv2 < 0){sigv2 = 0;}
  return List::create(Named("sigvhat2") = sigv2, Named("sigehat2") = sige2);
}

// main function
// [[Rcpp::export]]
List varner(arma::vec ni, arma::mat X, arma::vec Y, int method){
  List varEST;
  if (method == 1){
    varEST = varnerMOM(ni, X, Y);
  }
  if (method == 2){
    varEST = varnerREML(ni, X, Y);
  }
  if (method == 3){
    varEST = varnerML(ni, X, Y);
  }
  if (method == 4){
    varEST = varnerEB(ni, X, Y);
  }
  return varEST;
}


// double bootstrap functions for NER model
List dbootstrap(arma::vec ni, arma::mat X, arma::vec Y, arma::mat xbar, arma::mat Xmean, int method){
  int m = ni.size(); arma::vec ybar(m);
  for(int i = 0; i < m; i++){
    int nstart = sum(ni(arma::span(0,i))) - ni(i);
    if(ni(i) == 1){
      ybar(i) = Y(nstart);
    } else {
      ybar(i) = mean(Y(arma::span(nstart, nstart + ni(i) - 1)));
    }
  }
  List temp = varner(ni, X, Y, method);
  arma::vec bhat = temp["bhat"]; double sige2 = temp["sigehat2"]; double sigv2 = temp["sigvhat2"];
  arma::vec theta(m);
  for(int i = 0; i < m; i++){
    theta(i) = as_scalar(Xmean.row(i) * bhat) + (ybar(i) - as_scalar(xbar.row(i) * bhat)) * ni(i) * sigv2/(sige2+sigv2*ni(i));
  }
  return List::create(Named("theta") = theta, Named("bhat") = bhat, Named("sigvhat2") = sigv2, Named("sigehat2") = sige2);
}

// [[Rcpp::export]]
arma::vec mspeNERdb(arma::vec ni, arma::mat X, arma::vec Y, arma::mat Xmean, int K = 50, int C = 50, int method = 2){
  int m = ni.size(); int n = sum(ni); int p = X.n_cols;
  arma::mat xbar(m, p); arma::vec ybar(m);
  List phat = varner(ni, X, Y, method);
  arma::vec bhat = phat["bhat"]; double sige2 = phat["sigehat2"]; double sigv2 = phat["sigvhat2"];
  for(int i = 0; i < m; i++){
    int nstart = sum(ni(arma::span(0,i))) - ni(i);
    if(ni(i) == 1){
      xbar.row(i) = X.row(nstart);
      ybar(i) = Y(nstart);
    } else {
      xbar.row(i) = mean(X.rows(nstart, nstart + ni(i) - 1), 0);
      ybar(i) = mean(Y(arma::span(nstart, nstart + ni(i) - 1)));
    }
  }
  arma::mat umat = arma::zeros(m, K);
  arma::mat vmat = arma::zeros(m, K);
  arma::mat tempmat = arma::zeros(m, C);
  for(int k=0; k<K; k++){
    arma::vec vi = pow(sigv2, 0.5) * arma::randn(m,1);
    arma::vec vistar(n);
    int vv = 0;
    for(int v=0; v<m; v++){
      for(int ii=0; ii<ni(v); ii++){
        vistar(vv) = vi(v);
        vv = vv+1;
      }
    }
    arma::vec thetak = Xmean * bhat + vi;
    arma::vec Yk = X * bhat + vistar + pow(sige2, 0.5) * arma::randn(n,1);
    List temp = dbootstrap(ni, X, Yk, xbar, Xmean, method);
    arma::vec temptheta = temp["theta"];
    arma::vec bhat = temp["bhat"];
    double sige2hat = temp["sigehat2"];
    double sigv2hat = temp["sigvhat2"];
    umat.col(k) = pow(temptheta - thetak, 2);
    arma::vec thetac(m);
    for(int c=0; c<C; c++){
      arma::mat vik = pow(sigv2hat, 0.5) * arma::randn(m,1);
      arma::vec vikstar(n);
      int vv = 0;
      for(int v=0; v<m; v++){
        for(int ii=0; ii<ni(v); ii++){
          vikstar(vv) = vik(v);
          vv = vv+1;
        }
      }
      thetac = Xmean * bhat + vik;
      arma::vec Ykc = X * bhat + vikstar + pow(sige2hat, 0.5) * arma::randn(n,1);
      List temp = dbootstrap(ni, X, Ykc, xbar, Xmean, method);
      arma::vec temptheta1 = temp["theta"];
      arma::vec bhat = temp["bhat"];
      tempmat.col(c) = pow(temptheta1 - thetac, 2);
    }
    vmat.col(k) = mean(thetac,1);
  }
  arma::vec u = mean(umat,1);
  arma::vec v = mean(vmat,1);
  arma::vec mspe_DB(m);
  for(int ii=0; ii<m; ii++){
    if(u(ii) >= v(ii)){
      mspe_DB(ii) = u(ii) + atan(m * (u(ii) - v(ii)))/m;
    }else{
      mspe_DB(ii) = pow(u(ii),2)/(u(ii) + atan(m * (v(ii) - u(ii)))/m);
    }
  }
  return mspe_DB;
}

// jackknife functions for NER model
// [[Rcpp::export]]
arma::vec mspeNERjack(arma::vec ni, arma::mat X, arma::vec Y, arma::mat Xmean, int method = 2){
  int m = ni.size(); int p = X.n_cols;
  arma::mat xbar(m, p); arma::vec ybar(m); arma::vec thetahat(m); arma::vec msetheta(m);
  // calculate xbar and ybar through ni, X, Y
  for(int i = 0; i < m; i++){
    int nstart = sum(ni(arma::span(0,i))) - ni(i);
    if(ni(i) == 1){
      xbar.row(i) = X.row(nstart);
      ybar(i) = Y(nstart);
    } else {
      xbar.row(i) = mean(X.rows(nstart, nstart + ni(i) - 1), 0);
      ybar(i) = mean(Y(arma::span(nstart, nstart + ni(i) - 1)));
    }
  }
  List phat;
  phat = varner(ni, X, Y, method);
  arma::vec bhat = phat["bhat"];
  double sige2 = phat["sigehat2"];
  double sigv2 = phat["sigvhat2"];
  arma::vec Bi = sige2/(sige2 + ni * sigv2); arma::vec bi = sigv2 * Bi;
  arma::vec temp = Xmean * bhat; arma::vec temp1 = xbar * bhat;
  thetahat = temp + (1 - Bi) % (ybar - temp1);
  for(int i = 0; i < m; i++){
    arma::vec biu(m); arma::vec thetahatu(m);
    for(int j = 0; j < m; j++){
      int nstart = sum(ni(arma::span(0,j))) - ni(j);
      arma::mat Xu = X; Xu.shed_rows(nstart, nstart + ni(j) - 1);
      arma::vec Yu = Y; Yu.shed_rows(nstart, nstart + ni(j) - 1);
      arma::vec niu = ni; niu.shed_rows(j, j);
      List phatu;
      phatu = varner(niu, Xu, Yu, method);
      arma::vec bhatu = phatu["bhat"]; double sige2u = phatu["sigehat2"]; double sigv2u = phatu["sigvhat2"];
      double Biu = sige2u/(sige2u + ni(i) * sigv2u); biu(j) = sigv2u* Biu;
      thetahatu(j) = as_scalar(Xmean.row(i) * bhatu + (1 - Biu)*(ybar(i) - xbar.row(i) * bhatu));
    }
    msetheta(i) = bi(i) - (m - 1) * sum(biu - bi(i))/m + (m - 1) * sum(pow(thetahatu - thetahat(i),2))/m;
  }
  return msetheta;
}

// function for mcjack method
List Jackfun(arma::vec ni, arma::mat X, arma::vec Y, arma::mat xbar, arma::vec ybar){
  int m = ni.size(); int p = X.n_cols;
  arma::vec sigvjack(m); arma::vec sigejack(m);
  arma::mat bjack = arma::zeros(m,p);
  for(int j = 0;j < m;j++){
    int nstart = sum(ni(arma::span(0,j))) - ni(j);
    arma::mat Xu = X; Xu.shed_rows(nstart, nstart + ni(j) - 1);
    arma::vec Yu = Y; Yu.shed_rows(nstart, nstart + ni(j) - 1);
    arma::vec niu = ni; niu.shed_rows(j, j);
    List phatj = varner(niu, Xu, Yu, 3);
    sigvjack(j) = phatj["sigvhat2"];
    sigejack(j) = phatj["sigehat2"];
    arma::vec bjackj = phatj["bhat"];
    bjack.row(j) = bjackj.t();
  }
  return List::create(Named("sigvj") = sigvjack, Named("sigej") = sigejack, Named("bj") = bjack);
}

double thetafunner(arma::vec beta, double sigv2, arma::mat Xxbar, double xi){
  double theta = dot(Xxbar,beta) + sqrt(sigv2) * xi;
  return theta;
}

double thetahatfunner(arma::vec beta, double sigv2, double sige2, double nii, arma::mat Xxbar, double Yybar){
  double temp = dot(Xxbar,beta);
  double thetahat = temp + (Yybar - temp) * nii * sigv2/(sige2 + nii * sigv2);
  return thetahat;
}

//[[Rcpp::export]]
arma::vec mspeNERmcjack(arma::vec ni, arma::mat X, arma::vec Y, arma::mat Xmean, int K = 50, int method = 2){
  int m = ni.size(); int p = X.n_cols;
  arma::mat xbar(m, p); arma::vec ybar(m); List phat; arma::vec bpsi(m);
  phat = varner(ni, X, Y, method);
  arma::vec bhat = phat["bhat"]; double sige2 = phat["sigehat2"]; double sigv2 = phat["sigvhat2"];
  for(int i = 0; i < m; i++){
    int nstart = sum(ni(arma::span(0,i))) - ni(i);
    if(ni(i) == 1){
      xbar.row(i) = X.row(nstart);
      ybar(i) = Y(nstart);
    } else {
      xbar.row(i) = mean(X.rows(nstart, nstart + ni(i) - 1), 0);
      ybar(i) = mean(Y(arma::span(nstart, nstart + ni(i) - 1)));
    }
  }
  List temp = Jackfun(ni, X, Y, xbar, ybar);
  arma::vec sigv2j = temp["sigvj"]; arma::vec sige2j = temp["sigej"]; arma::mat bhatj = temp["bj"];
  for (int i = 0;i < m;i++){
    arma::vec dif(K); arma::mat difmat = arma::zeros(m,K);
    for(int k = 0;k < K;k++){
      arma::vec xi = arma::randn(ni(i),1);
      double mxi = sum(xi)/ni(i);
      arma::vec ei = arma::randn(ni(i),1) * sqrt(sige2);
      double thetaik = thetafunner(bhat , sigv2, Xmean.row(i) , mxi);
      arma::vec Yik = X.rows(arma::span(sum(ni(arma::span(0,i))) - ni(i), sum(ni(arma::span(0,i))) - 1)) * bhat + sqrt(sigv2) * xi + ei;
      arma::vec Yk = Y; Yk(arma::span(sum(ni(arma::span(0,i))) - ni(i), sum(ni(arma::span(0,i))) - 1)) = Yik;
      arma::vec ybark = ybar;
      ybark(i) = mean(Yik);
      List phatk = varner(ni, X, Yk, method);
      double sigv2k = phatk["sigvhat2"]; double sige2k = phatk["sigehat2"]; arma::vec bhatk = phatk["bhat"];
      double thetahatik = thetahatfunner(bhatk, sigv2k, sige2k, ni(i), Xmean.row(i), ybark(i));
      dif(k) = pow(thetahatik - thetaik,2);
      List tempk = Jackfun(ni, X, Yk, xbar, ybark);
      arma::vec sigv2jk = tempk["sigvj"]; arma::vec sige2jk = tempk["sigej"]; arma::mat bhatjk = tempk["bj"];
      for(int ii = 0; ii < m;ii++){
        double thetakj = thetafunner(bhatj.row(ii).t(), sigv2j(ii), Xmean.row(i),mxi);
        double thetahatkj = thetahatfunner(bhatj.row(ii).t(), sigv2jk(ii), sige2jk(ii), ni(i), Xmean.row(i),ybark(i));
        difmat(ii,k) = pow(thetahatkj - thetakj,2);
      }
    }
    arma::vec meandifmat = mean(difmat,1);
    arma::vec logvec = log(meandifmat);
    bpsi(i) = m * log(sum(dif)/K) - (m - 1) * sum(logvec)/ m ;
  }
  arma::vec bpsi_inv = exp(bpsi);
  return bpsi_inv;
}


// functions for bootstrap method
List NERpbootstrap(arma::vec ni, arma::mat X, arma::vec Y, arma::mat xbar, arma::mat Xmean, arma::vec ybar){
  int m = ni.size(); int n = sum(ni);
  List phat = varner(ni, X, Y, 2);
  arma::vec bhat = phat["bhat"]; double sige2 = phat["sigehat2"]; double sigv2 = phat["sigvhat2"];
  arma::mat V = arma::zeros(n, n); arma::mat Vtemp(n,n);
  for(int i = 0; i < m; i++){
    arma::mat Vtemp = arma::eye(ni(i),ni(i))/sige2 - sigv2/((ni(i)*sigv2+sige2)*sige2) * arma::ones(ni(i),ni(i));
    int nstart = sum(ni(arma::span(0,i))) - ni(i);
    V.submat(arma::span(nstart, nstart + ni(i) - 1), arma::span(nstart, nstart + ni(i) - 1)) = Vtemp;
  }
  arma::vec gama = ni * sigv2 / (ni * sigv2 + sige2);
  arma::vec g1 = (1 - gama) * sigv2;
  arma::vec uhat = ybar - xbar * bhat;
  arma::vec ghat = sigv2/(sigv2 + sige2/ni);
  arma::vec g2(m); arma::vec theta(m);
  for(int i = 0; i < m; i++){
    g2(i) = as_scalar((Xmean.row(i) - gama(i) * xbar.row(i)) * inv(X.t() * V * X) * (Xmean.row(i) - gama(i) * xbar.row(i)).t()) ;
    theta(i) = as_scalar(Xmean.row(i) * bhat + uhat(i) * ghat(i));
  }
  return List::create(Named("g1") = g1, Named("g2") = g2, Named("theta") = theta, Named("bhat")=bhat, Named("sigv2") = sigv2, Named("sige2") = sige2);
}
// [[Rcpp::export]]
arma::vec mspeNERpb(arma::vec ni, arma::mat X, arma::vec Y, arma::mat Xmean, int K = 50){
  int m = ni.size(); int n = sum(ni); int p = X.n_cols;
  arma::mat xbar(m, p); arma::vec ybar(m); arma::vec mspe(m);
  // calculate xbar and ybar through ni, X, Y
  for(int i = 0; i < m; i++){
    int nstart = sum(ni(arma::span(0,i))) - ni(i);
    if(ni(i) == 1){
      xbar.row(i) = X.row(nstart);
      ybar(i) = Y(nstart);
    } else {
      xbar.row(i) = mean(X.rows(nstart, nstart + ni(i) - 1), 0);
      ybar(i) = mean(Y(arma::span(nstart, nstart + ni(i) - 1)));
    }
  }

  List phat = NERpbootstrap(ni, X, Y, xbar, Xmean, ybar);
  double sige2 = phat["sige2"]; double sigv2 = phat["sigv2"];
  arma::vec g1 = phat["g1"]; arma::vec g2 = phat["g2"];
  arma::vec theta = phat["theta"]; arma::vec bhat = phat["bhat"];
  arma::mat g1star(m, K); arma::mat g2star(m, K); arma::mat thetastar(m, K);
  for(int k=0; k<K; k++){
    arma::mat vi = pow(sigv2, 0.5) * arma::randn(m,1);
    arma::vec vistar(n);
    int vv = 0;
    for(int v=0; v<m; v++){
      for(int ii=0; ii<ni(v); ii++){
        vistar(vv) = vi(v);
        vv = vv+1;
      }
    }
    arma::mat Yk = X * bhat + vistar + pow(sige2, 0.5) * arma::randn(n,1);
    List temp = NERpbootstrap(ni, X, Yk, xbar, Xmean, ybar);
    arma::vec tempg1 = temp["g1"]; g1star.col(k) = tempg1;
    arma::vec tempg2 = temp["g2"]; g2star.col(k) = tempg2;
    arma::vec temptheta = temp["theta"]; thetastar.col(k) = temptheta;
  }
  mspe = 2*(g1 + g2) - mean(g1star, 1) - mean(g2star, 1) + mean(thetastar, 1) - theta;
  return mspe;
}

// Sumca method for NER model
// [[Rcpp::export]]
arma::vec mspeNERsumca(arma::vec ni, arma::mat X, arma::vec Y, arma::mat Xmean, int K = 50, int method = 2){
  int m = ni.size(); int n = sum(ni); int p = X.n_cols;
  arma::mat xbar(m, p); arma::vec ybar(m);
  List phat = varner(ni, X, Y, method);
  arma::vec bhat = phat["bhat"]; double sige2 = phat["sigehat2"]; double sigv2 = phat["sigvhat2"];
  arma::vec ay = (sige2 * sigv2)/(ni * sigv2 + sige2);
  arma::mat ayk = arma::zeros(m, K);
  for(int k=0;k<K;k++){
    arma::vec Yk(n); int count=0; int kk=0;
    arma::vec v = arma::randn(m,1)*sqrt(sigv2);
    for(int i=0;i<m;i++){
      count++; double nii = ni(i);
      for(int j=0;j< nii;j++){
        Yk(kk)= arma::as_scalar(X.row(kk) * bhat) + v(i) + arma::as_scalar(arma::randn(1,1))*sqrt(sige2);
        kk++;
      }
    }
    List phatk = varner(ni, X, Yk, method);
    arma::vec bhatk = phatk("bhat");
    double sige2k = phatk("sigehat2");
    double sigv2k =phatk("sigvhat2");
    ayk.col(k) = (sige2k * sigv2k)/(ni * sigv2k + sige2k);
  }
  arma::vec mseNER = 2*ay - mean(ayk,1);
  return(mseNER);
}

// **********************
// functions for FH model
// **********************
double varfhMOM(arma::vec Y, arma::mat X, arma::vec D){
  int m = Y.size(); int p = X.n_cols;
  arma::vec coef = inv(X.t() * X) * X.t() * Y;
  arma::vec resid = Y - X * coef;
  double sum1 = 0.0;
  double sum2 = 0.0;
  double sumY = 0.0;
  arma::mat tXX_inv = inv(X.t() * X);
  for(int i=0; i<m; i++){
    sum1 = sum1 + pow(resid(i),2);
    arma::vec temp2 = D(i)*(1 - X.row(i)* tXX_inv * X.row(i).t());
    sum2 = sum2 + temp2(0);
    sumY = sumY + pow(Y(i) - mean(Y),2);
  }
  double Ahat = 0.0001;
  if(p>1){
    double tempA = (sum1 - sum2)/(m - p);
    if(tempA > 0.0001){
      Ahat = tempA;
    }
  }else{
    double tempA = sumY/(m-1) - sum(D)/m;
    if(tempA > 0.0001){
      Ahat = tempA;
    }
  }
  return Ahat;
}

double varfhREML(arma::vec Y, arma::mat X, arma::vec D){
  int m = Y.size();
  double Ahat = varfhMOM(Y, X, D);
  for(int it=0; it < 1000; it++){
    if(Ahat < 0.0001){
      Ahat = 0.0001;
    }else{
      double trace1 = 0.0;
      double trace2 = 0.0;
      arma::mat temp = diagmat(1/(Ahat + D));
      arma::mat pmatpre = temp - (temp * X * inv(X.t() * temp * X) * X.t() * temp);
      arma::mat pmatpre2 = pmatpre * pmatpre;
      for(int t=0; t<m; t++){
        trace1 = trace1 + pmatpre(t,t);
        trace2 = trace2 + pmatpre2(t,t);
      }
      arma::vec tempbias = (Y.t() * pmatpre2 * Y - trace1)/trace2;
      double bias = tempbias(0);
      double psi_temp = Ahat + bias;
      Ahat = psi_temp;
      if(bias < sqrt(m)){
        if(Ahat < 0){Ahat = 0.0001;}
        break;}
      if(Ahat < 0){Ahat = 0.0001;}
    }
  }
  return Ahat;
}
double varfhEB(arma::vec Y, arma::mat X, arma::vec D){
  int m = Y.size(); int p = X.n_cols;
  arma::mat tempX = arma::zeros(p,p);
  arma::mat tempXY = arma::zeros(p, 1);
  for(int i=0; i<m; i++){
    tempX = tempX + X.row(i).t() * X.row(i);
    tempXY = tempXY + Y(i) * X.row(i).t();
  }
  arma::mat tempX_inv = inv(tempX);
  arma::vec beta_ols = tempX_inv * tempXY;
  double sum1 = 0.0;
  double sum2 = 0.0;
  for(int ii=0; ii<m; ii++){
    arma::vec tempsum1 = pow(Y(ii) - X.row(ii) * beta_ols, 2);
    sum1 = sum1 + tempsum1(0);
    arma::vec tempsum2 = (1 - (X.row(ii) * tempX_inv * X.row(ii).t())) * D(ii);
    sum2 = sum2 + tempsum2(0);
  }
  double Ahat = (sum1 - sum2)/(m-p);
  if(Ahat < 0){
    Ahat = 0;
  }
  return Ahat;
}

double varfh(arma::vec Y, arma::mat X, arma::vec D, int method){
  double varEST = 0.0;
  if (method == 1){
    varEST = varfhMOM(Y, X, D);
  }
  if (method == 2){
    varEST = varfhREML(Y, X, D);
  }
  // if (method == 3){
  //   varEST = varfhML(Y, X, D);
  // }
  if (method == 4){
    varEST = varfhEB(Y, X, D);
  }
  return varEST;
}

// functions for double bootstrap method with FH model
List FHdbootstrap(int m, int p, arma::mat X, arma::vec Y, arma::vec D, int method){
  double Ahat = varfh(Y, X, D, method);
  arma::vec bhat = inv(X.t() * diagmat(1/(Ahat+D)) * X) * X.t() * diagmat(1/(Ahat+D)) * Y;
  arma::vec theta(m);
  for(int j=0; j<m; j++){
    arma::vec temp_theta = X.row(j) * bhat + (Y(j) - X.row(j) * bhat) * (Ahat/(Ahat + D(j)));
    theta(j) = temp_theta(0);
  }
  return List::create(Named("theta") = theta, Named("bhat") = bhat, Named("Ahat") = Ahat);
}

//[[Rcpp::export]]
arma::vec mspeFHdb(arma::vec Y, arma::mat X, arma::vec D,int K = 50, int C = 50, int method = 2){
  int m = Y.size();
  int p = X.n_cols;
  double Ahat = varfh(Y, X, D, method);
  arma::vec b = inv(X.t() * diagmat(1/(Ahat+D)) * X) * X.t() * diagmat(1/(Ahat+D)) * Y;
  arma::mat umat = arma::zeros(m, K);
  arma::mat vmat = arma::zeros(m, K);
  arma::mat temp_mat = arma::zeros(m, K);
  for(int k=0; k<K; k++){
    arma::vec thetak = X * b +  pow(Ahat, 0.5) * arma::randn(m,1);
    arma::vec eistar1(m);
    for(int d=0; d<m; d++){
      eistar1(d) = arma::as_scalar(pow(D(d), 0.5)* arma::randn(1,1));
    }
    arma::vec Y_star1 = thetak + eistar1;

    List temp1 = FHdbootstrap(m, p, X, Y_star1, D, method);
    arma::vec bhatk = temp1["bhat"];
    double Ahatk = temp1["Ahat"];
    arma::vec thetahatk = temp1["theta"];

    umat.col(k) = pow(thetahatk - thetak, 2);
    for(int c=0; c<C; c++){
      arma::vec thetac = X * bhatk +  pow(Ahatk, 0.5) * arma::randn(m,1);
      arma::vec eistar2(m);
      for(int d=0; d<m; d++){
        eistar2(d) = arma::as_scalar(pow(D(d), 0.5)* arma::randn(1,1));
      }
      arma::vec Y_star2 = thetac + eistar2;
      List temp2 = FHdbootstrap(m, p, X, Y_star2, D, method);
      arma::vec thetahatkc = temp2["theta"];
      temp_mat.col(c) = pow(thetahatkc - thetac, 2);
    }
    vmat.col(k) = mean(temp_mat,1);
  }
  arma::vec u = mean(umat, 1);
  arma::vec v = mean(vmat, 1);
  arma::vec mspe_DB(m);
  for(int ii=0; ii<m; ii++){
    if(u(ii) >= v(ii)){
      mspe_DB(ii) = u(ii) + atan(m * (u(ii) - v(ii)))/m;
    }else{
      mspe_DB(ii) = pow(u(ii),2)/(u(ii) + atan(m * (v(ii) - u(ii)))/m);
    }
  }
  return mspe_DB;
}

// functions for jackknife method with FH model
List mseu(arma::vec D,int index, int m, arma::mat X, arma::mat Y, int method){
  arma::vec biu(m); arma::vec thetaHatu(m);
  for(int u = 0;u < m;u++){
    arma::vec Du = D; Du.shed_rows(u, u);
    arma::mat dataX = X; dataX.shed_rows(u, u);
    arma::mat dataY = Y; dataY.shed_rows(u, u);
    double Ahatu = varfh(dataY,dataX,Du, method);
    arma::mat v_inv = diagmat(1/(Ahatu + Du));
    arma::vec bhatu = inv(dataX.t() * v_inv * dataX) * dataX.t() * v_inv * dataY;
    double Bu = D(index) / (Ahatu+D(index));
    biu(u) = Ahatu * Bu;
    arma::mat temp = X.row(index) * bhatu + (1 - Bu) * (Y(index) - X.row(index) * bhatu);
    thetaHatu(u) = temp(0,0);
  }
  return List::create(Named("bhatu") = biu, Named("thetahatu") = thetaHatu);
}

// [[Rcpp::export]]
arma::vec mspeFHjack(arma::vec Y, arma::mat X, arma::vec D, int method = 2){
  int m = Y.size();
  double Ahat = varfh(Y, X, D, method);
  arma::vec bhat = inv(X.t() * diagmat(1/(Ahat+D)) * X) * X.t() * diagmat(1/(Ahat+D)) * Y;
  arma::vec mseFH(m);
  arma::vec B = D / (D+Ahat);
  arma::vec b = Ahat * B;
  arma::vec thetahat = X * bhat + (1 - B) % (Y - X * bhat);
  for(int i = 0;i < m;i++){
    List uhat = mseu(D,i,m,X,Y,method);
    arma::vec bhatu =uhat["bhatu"];
    arma::vec thetahatu = uhat["thetahatu"];
    mseFH(i) = b(i) - (m - 1) * sum(bhatu - b(i))/m + (m - 1) * sum(pow(thetahatu - thetahat(i),2))/m;
  }
  return mseFH;
}


// functions for monte-carlo jackknife method with FH model
double thetafunfh(arma::vec beta, double A, arma::mat XX, double xi){
  double theta = dot(beta,XX) + pow(A,0.5) *xi;
  return theta;
}

double thetahatfunfh(arma::vec beta, double A, double D, arma::mat XX, double YY){
  double temp = dot(beta,XX);
  double thetahat = temp + A * (YY - temp)/(A+D);
  return thetahat;
}

List Jackfun(arma::mat X, arma::vec Y, arma::vec D, int m, int method){
  arma::vec Ajack(m);
  int p = X.n_cols;
  arma::mat bjack = arma::zeros(m,p);
  for(int j = 0;j < m;j++){
    arma::mat Xj = X; Xj.shed_rows(j, j);
    arma::vec Yj = Y; Yj.shed_rows(j, j);
    arma::vec Dj = D; Dj.shed_rows(j, j);
    Ajack(j) = varfh(Yj, Xj, Dj, method);
    arma::mat Vjinv = diagmat(1/(Ajack(j) + Dj));
    arma::mat tempvec = inv(Xj.t() * Vjinv * Xj) * Xj.t() * Vjinv * Yj;
    bjack.row(j) = tempvec.t();
  }
  return List::create(Named("Aj") = Ajack, Named("bj") = bjack);
}

//[[Rcpp::export]]
arma::vec mspeFHmcjack(arma::vec Y, arma::mat X, arma::vec D, int K = 50, int method = 2){
  int m = Y.size();
  double Ahat = varfh(Y, X, D, method);
  arma::vec bhat = inv(X.t() * diagmat(1/(Ahat+D)) * X) * X.t() * diagmat(1/(Ahat+D)) * Y;
  List temp = Jackfun(X,Y,D,m, method);
  arma::vec Aj = temp["Aj"]; arma::mat bhatj = temp["bj"];
  arma::vec bpsi(m);
  for(int t = 0;t < m;t++){
    arma::vec dif(K); arma::mat difmat(m,K);
    for(int k = 0;k < K;k++){
      double xi = arma::randn();
      double thetaik = thetafunfh(bhat,Ahat,X.row(t),xi);
      double Yik = thetaik + sqrt(D(t)) * arma::randn();
      arma::vec Yk = Y; Yk(t) = Yik;
      double Ahatk = varfh(Yk,X,D, method);
      arma::mat Vinvk = diagmat(1/(Ahatk + D));
      arma::vec bhatk = inv(X.t() * Vinvk * X) *  X.t() * Vinvk * Yk;
      double thetahatik = thetahatfunfh(bhatk ,Ahatk ,D(t) ,X.row(t) ,Yik);
      dif(k) = pow(thetahatik - thetaik,2);
      List tempk = Jackfun(X, Yk, D,m, method);
      arma::vec Ajk = tempk["Aj"];
      arma::mat bhatjk = tempk["bj"];
      for(int ii = 0;ii < m;ii++){
        double thetakj = thetafunfh(bhatj.row(ii).t() ,Aj(ii) ,X.row(t) ,xi);
        double thetahatkj = thetahatfunfh(bhatjk.row(ii).t() ,Ajk(ii) ,D(t) ,X.row(t), Yik);
        difmat(ii,k) = pow(thetahatkj - thetakj, 2);
      }
    }
    arma::vec meandifmat = mean(difmat,1);
    arma::vec logvec(m);
    for(int i=0; i<m; i++){
      logvec(i) = log(meandifmat(i));
    }
    bpsi(t) = m * log(sum(dif)/K) - (m - 1) * sum(logvec)/ m ;
  }
  arma::vec bpsiinv = exp(bpsi);
  return bpsiinv;
}

// functions for parameter bootstrap method with FH model
List FHpbootstrap(int m, int p, arma::mat X, arma::vec Y_star, arma::vec D, arma::mat Y){
  double psi_FH = varfhEB(Y_star, X, D);
  arma::vec bhat = inv(X.t() * diagmat(1/(psi_FH+D)) * X) * X.t() * diagmat(1/(psi_FH+D)) * Y;
  arma::vec g1 = arma::vec(m); arma::vec g2 = arma::vec(m); arma::vec g3 = arma::vec(m);
  arma::vec theta = arma::vec(m); arma::mat inv_V = diagmat(1/(psi_FH + D));
  for(int i=0; i<m; i++){
    g1(i) = psi_FH * D(i) / (psi_FH + D(i));
    arma::vec tempg2 = ((pow(D(i), 2) / pow(psi_FH + D(i), 2)) * X.row(i) *
      inv(X.t() * inv_V * X) * X.row(i).t());
    g2(i) = tempg2(0);
    arma::vec temptheta = X.row(i) * bhat + (Y(i) - X.row(i) * bhat) * psi_FH / (psi_FH + D(i));
    theta(i) = temptheta(0);
  }
  return List::create(Named("g1") = g1, Named("g2") = g2, Named("theta") = theta,Named("bhat") = bhat, Named("psi_FH") = psi_FH);
}

//[[Rcpp::export]]
arma::vec mspeFHpb(arma::vec Y, arma::mat X, arma::vec D, int K = 50){
  int m = Y.size(); int p = X.n_cols;
  List temp2 = FHpbootstrap(m, p, X, Y, D, Y);
  arma::vec g1 = temp2["g1"]; arma::vec g2 = temp2["g2"]; arma::vec theta = temp2["theta"];
  arma::vec bhat = temp2["bhat"]; double psi_FH = temp2["psi_FH"];
  arma::mat g1_star = arma::zeros(m, K); arma::mat g2_star = arma::zeros(m, K); arma::mat theta_star = arma::zeros(m, K);
  for(int k=0; k<K; k++){
    arma::vec eistar(m);
    for(int e=0; e<m; e++){
      eistar(e) = arma::as_scalar(pow(D(e), 0.5)* arma::randn(1,1));
    }
    arma::vec Y_star = X*bhat + pow(psi_FH, 0.5) * arma::randn(m,1) + eistar;
    List temp1 = FHpbootstrap(m, p, X, Y_star, D, Y);
    arma::vec g1_stark = temp1["g1"]; g1_star.col(k) = g1_stark;
    arma::vec g2_stark = temp1["g2"]; g2_star.col(k) = g2_stark;
    arma::vec thetak_star = temp1["theta"]; theta_star.col(k) = thetak_star;
  }
  arma::mat summat = arma::zeros(m,K);
  for(int i=0; i<K; i++){
    for(int j=0; j<m; j++){
      summat(j,i) = pow(theta_star(j, i) - theta(j), 2);
    }
  }
  arma::mat tempmat = g1_star + g2_star;
  arma::vec mspe_theta = 2.0 * (g1 + g2) - mean(tempmat, 1) + mean(summat, 1);
  return mspe_theta;
}

// functions for sumca method with FH model
arma::vec AhatK(int m ,arma::mat X, arma::vec bhat, double A_REML, arma::vec D, int K,int p, int method){
  arma::vec A_REML_K(K);
  for(int i = 0;i < K;i++){
    arma::vec Y_K = X * bhat + sqrt(A_REML) * arma::randn(m,1) + sqrt(D) % arma::randn(m,1);
    A_REML_K(i) = varfh(Y_K,X,D,method);
  }
  return A_REML_K;
}

// [[Rcpp::export]]
arma::vec mspeFHsumca(arma::vec Y, arma::mat X, arma::vec D, int K = 50, int method = 2){
  int m = Y.size(); int p = X.n_cols;
  double Ahat = varfh(Y, X, D, method);
  arma::vec bhat = inv(X.t() * diagmat(1/(Ahat+D)) * X) * X.t() * diagmat(1/(Ahat+D)) * Y;
  arma::vec a_yAhat = Ahat * D/(Ahat + D);
  arma::vec a_ykAhat = a_yAhat;
  arma::vec Ahat_K = AhatK(m, X ,bhat ,Ahat, D, K, p ,method);
  arma::mat a_ykAhatk = arma::mat(m, K);
  for(int j = 0;j < K;j++){
    a_ykAhatk.col(j) = Ahat_K(j) * D/(Ahat_K(j)+D);
  }
  arma::vec mseFH = 2 * a_yAhat - mean(a_ykAhatk,1);
  return mseFH;
}
