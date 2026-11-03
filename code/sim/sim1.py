import numpy as np
rng=np.random.default_rng(1)
# HPP-like parameters (from report): within-person SD 0.84h (M), 0.87h (D); single-night ICC 0.33 (M), 0.26 (D)
sw_M,sw_D=0.84,0.87
sb_M=np.sqrt(0.33/0.67)*sw_M; sb_D=np.sqrt(0.26/0.74)*sw_D
rho_b=-0.72; rho_w=-0.6  # between-person and within-night correlation of M,D
N=2011; sdY=0.113
def simulate(bM,bD,k_probs=(0.7,0.3)):
    Sb=np.array([[sb_M**2,rho_b*sb_M*sb_D],[rho_b*sb_M*sb_D,sb_D**2]])
    Sw=np.array([[sw_M**2,rho_w*sw_M*sw_D],[rho_w*sw_M*sw_D,sw_D**2]])
    X=rng.multivariate_normal([20.8,3.0],Sb,N)
    k=rng.choice([2,3],N,p=k_probs)
    nights=[X[i]+rng.multivariate_normal([0,0],Sw,k[i]) for i in range(N)]
    W=np.array([n.mean(0) for n in nights])
    signal=X@np.array([bM,bD])
    Y=signal+rng.normal(0,np.sqrt(sdY**2-signal.var()),N)
    return X,W,Y,nights,k
def ols(Xm,Y):
    A=np.column_stack([np.ones(len(Y)),Xm]); return np.linalg.lstsq(A,Y,rcond=None)[0][1:]
def rc(W,Y,nights,k):
    # multivariate regression calibration: Sigma_U from within-person deviations
    dev=np.vstack([n-n.mean(0) for n in nights]); df=sum(len(n)-1 for n in nights)
    Su=dev.T@dev/df
    mu=W.mean(0); SW=np.cov(W.T)
    SX=SW-Su*np.mean(1/k)
    Xhat=np.array([mu+SX@np.linalg.inv(SX+Su/k[i])@(W[i]-mu) for i in range(len(W))])
    return ols(Xhat,Y)
for name,(bM,bD) in {"truth: gap only (bD=-0.005)":(0,-0.005),"truth: clock only (bM=+0.005)":(0.005,0),"truth: null":(0,0)}.items():
    res={"oracle":[],"naive":[],"RC":[]}
    for r in range(300):
        X,W,Y,n,k=simulate(bM,bD)
        res["oracle"].append(ols(X,Y)); res["naive"].append(ols(W,Y)); res["RC"].append(rc(W,Y,n,k))
    print(name)
    for m,v in res.items():
        v=np.array(v); print(f"  {m:6s} bM={v[:,0].mean():+.4f} (sd {v[:,0].std():.4f})  bD={v[:,1].mean():+.4f} (sd {v[:,1].std():.4f})")
