#!/usr/bin/awk -f
BEGIN{
  FS=","; OFS=",";
  # row-level accumulators (with portfolio)
  n1=0; sx1=sy1=sx2_1=sy2_1=sxy1=0; mae1=0; mse1=0;
  # row-level accumulators (no-portfolio)
  n2=0; sx2=sy2=sx2_2=sy2_2=sxy2=0; mae2=0; mse2=0;
}
NR==1{next}
{
  # columns:
  # 8=residual_bps, 10=pred_residual_bps, 11=pred_residual_bps_nopf
  y=$8; x1=$10; x2=$11;
  # sanitize
  if (y=="" || x1=="" || x2==""){}
  # row-level with pf
  if (y != "" && x1 != ""){
    yv=y+0; x1v=x1+0;
    n1++; sx1+=x1v; sy1+=yv; sx2_1+=x1v*x1v; sy2_1+=yv*yv; sxy1+=x1v*yv; d=x1v-yv; if(d<0){ad=-d}else{ad=d}; mae1+=ad; mse1+=d*d;
  }
  # row-level no-pf
  if (y != "" && x2 != ""){
    yv2=y+0; x2v=x2+0;
    n2++; sx2+=x2v; sy2+=yv2; sx2_2+=x2v*x2v; sy2_2+=yv2*yv2; sxy2+=x2v*yv2; d2=x2v-yv2; if(d2<0){ad2=-d2}else{ad2=d2}; mae2+=ad2; mse2+=d2*d2;
  }
  # basket key: portfolio_id + trade_dt
  key=$1"|"$2;
  if (y!=""){
    cnt[key]++; sumy[key]+=(y+0)
  }
  if (x1!=""){ sumpf[key]+=(x1+0) }
  if (x2!=""){ sumnopf[key]+=(x2+0) }
}
END{
  # row-level stats
  if(n1>1){
    r1=(n1*sxy1 - sx1*sy1)/sqrt((n1*sx2_1 - sx1*sx1)*(n1*sy2_1 - sy1*sy1));
    mae1/=n1; rmse1=sqrt(mse1/n1);
  } else { r1=0; mae1=0; rmse1=0 }
  if(n2>1){
    r2=(n2*sxy2 - sx2*sy2)/sqrt((n2*sx2_2 - sx2*sx2)*(n2*sy2_2 - sy2*sy2));
    mae2/=n2; rmse2=sqrt(mse2/n2);
  } else { r2=0; mae2=0; rmse2=0 }
  # basket-level: compute means per key then aggregate
  nb1=0; sbx1=sby1=sbx2_1=sby2_1=sbxy1=0; bmae1=0; bmse1=0;
  nb2=0; sbx2=sby2b=sbx2_2b=sby2_2b=sbxy2=0; bmae2=0; bmse2=0;
  for(k in cnt){
    c=cnt[k]; ybar=sumy[k]/c;
    if (k in sumpf){ xbar1=sumpf[k]/c; nb1++; sbx1+=xbar1; sby1+=ybar; sbx2_1+=xbar1*xbar1; sby2_1+=ybar*ybar; sbxy1+=xbar1*ybar; d=xbar1-ybar; if(d<0){ad=-d}else{ad=d}; bmae1+=ad; bmse1+=d*d }
    if (k in sumnopf){ xbar2=sumnopf[k]/c; nb2++; sbx2+=xbar2; sby2b+=ybar; sbx2_2b+=xbar2*xbar2; sby2_2b+=ybar*ybar; sbxy2+=xbar2*ybar; d2=xbar2-ybar; if(d2<0){ad2=-d2}else{ad2=d2}; bmae2+=ad2; bmse2+=d2*d2 }
  }
  if(nb1>1){ br1=(nb1*sbxy1 - sbx1*sby1)/sqrt((nb1*sbx2_1 - sbx1*sbx1)*(nb1*sby2_1 - sby1*sby1)); bmae1/=nb1; brmse1=sqrt(bmse1/nb1) } else { br1=0; bmae1=0; brmse1=0 }
  if(nb2>1){ br2=(nb2*sbxy2 - sbx2*sby2b)/sqrt((nb2*sbx2_2b - sbx2*sbx2)*(nb2*sby2_2b - sby2b*sby2b)); bmae2/=nb2; brmse2=sqrt(bmse2/nb2) } else { br2=0; bmae2=0; brmse2=0 }
  # print summary
  printf("Row-level WITH portfolio: n=%d, r=%.4f, MAE=%.3f, RMSE=%.3f\n", n1, r1, mae1, rmse1);
  printf("Row-level NO-portfolio:  n=%d, r=%.4f, MAE=%.3f, RMSE=%.3f\n", n2, r2, mae2, rmse2);
  printf("Basket-level WITH portfolio: n=%d, r=%.4f, MAE=%.3f, RMSE=%.3f\n", nb1, br1, bmae1, brmse1);
  printf("Basket-level NO-portfolio:  n=%d, r=%.4f, MAE=%.3f, RMSE=%.3f\n", nb2, br2, bmae2, brmse2);
}