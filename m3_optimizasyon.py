"""
ENM412 – MAN Türkiye A.Ş.
Modül 3 – Grid Search + Optuna (r,Q) Optimizasyonu
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN
"""
import numpy as np, pandas as pd, optuna
from scipy.stats import norm
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

def lt_parametreleri(mu, sigma, LT_ay):
    mu_L    = mu * LT_ay
    sigma_L = np.sqrt(sigma**2*LT_ay + (mu*LT_ay*0.20)**2)
    return float(mu_L), float(max(sigma_L,1e-6))

def analitik_maliyet(Q, r, mu, mu_L, sigma_L, h, p, S):
    Q = max(float(Q),1.)
    z    = (r-mu_L)/sigma_L if sigma_L>0 else 0
    E_so = max(sigma_L*(norm.pdf(z)-z*(1-norm.cdf(z))),0.)
    SS   = max(0., r-mu_L)
    et,si,sk = h*(Q/2.+SS), S*(mu/Q), p*(mu/Q)*E_so
    return {"elde_tutma":et,"siparis":si,"stoksuz":sk,"toplam":et+si+sk}

def grid_search_rQ(mu,sigma,LT_ay,h,p,S,Q_ref,grid_adim=15):
    mu_L,sigma_L = lt_parametreleri(mu,sigma,LT_ay)
    Q_ref = max(Q_ref,1.)
    Q_grid = np.unique(np.linspace(max(1.,Q_ref*.3),Q_ref*4,grid_adim).astype(int))
    r_grid = np.unique(np.linspace(max(0.,mu_L-sigma_L),mu_L+3*sigma_L,grid_adim).astype(int))
    best_tc,best_Q,best_r = np.inf,Q_ref,mu_L+1.65*sigma_L
    for Qv in Q_grid:
        for rv in r_grid:
            tc = analitik_maliyet(Qv,rv,mu,mu_L,sigma_L,h,p,S)["toplam"]
            if tc<best_tc: best_tc,best_Q,best_r = tc,float(Qv),float(rv)
    return best_Q,best_r,mu_L,sigma_L

def optuna_rQ(mu,sigma,LT_ay,h,p,S,best_Qg,best_rg,n_trials=50,MOQ=1):
    mu_L,sigma_L = lt_parametreleri(mu,sigma,LT_ay)
    Q_lb = max(float(MOQ),best_Qg*.4); Q_ub = max(best_Qg*2.5,Q_lb+1.)
    r_lb = max(0.,best_rg-sigma_L);    r_ub = best_rg+sigma_L
    def obj(t):
        Q = t.suggest_float("Q",Q_lb,Q_ub); r = t.suggest_float("r",r_lb,r_ub)
        return analitik_maliyet(Q,r,mu,mu_L,sigma_L,h,p,S)["toplam"]
    s = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj,n_trials=n_trials,show_progress_bar=False)
    opt_Q  = max(int(round(s.best_params["Q"])),MOQ)
    opt_r  = max(0,int(round(s.best_params["r"])))
    opt_SS = max(0,int(round(1.65*sigma_L)))
    hizmet = float(norm.cdf((opt_r-mu_L)/sigma_L)) if sigma_L>0 else 1.
    komp   = analitik_maliyet(opt_Q,opt_r,mu,mu_L,sigma_L,h,p,S)
    return {"optimal_Q":opt_Q,"optimal_r":opt_r,"optimal_SS":opt_SS,
            "hizmet_duzeyi":hizmet,"min_maliyet":komp["toplam"],
            "elde_tutma":komp["elde_tutma"],"siparis_maliyeti":komp["siparis"],
            "stoksuz_maliyet":komp["stoksuz"]}

def parca_optimize(parca_kodu,opt_df,abc_df,tahmin_listesi,grid_adim=12,n_trials=50):
    opt_row = opt_df[opt_df["Parça_Kodu"]==parca_kodu]
    abc_row = abc_df[abc_df["Parça_Kodu"]==parca_kodu]
    if opt_row.empty: raise ValueError(f"{parca_kodu} bulunamadı")
    o  = opt_row.iloc[0]
    lt = float(o.get("LT_ay", o.get("LT_gun",20)/30))
    h,p,S = float(o["h"]),float(o["p"]),float(o["Siparis_Maliyeti"])
    if not abc_row.empty:
        mu  = float(abc_row.iloc[0]["Ort_Aylik_Talep"])
        sig = float(abc_row.iloc[0]["Std_Sapma"])
    else:
        mu  = float(np.mean(tahmin_listesi)) if tahmin_listesi else 1.
        sig = float(np.std(tahmin_listesi))  if tahmin_listesi else 1.
    if tahmin_listesi:
        mu_t = np.mean([t for t in tahmin_listesi if t>0])
        if mu_t>0: mu = .6*mu + .4*mu_t
    mu_L,sigma_L = lt_parametreleri(mu,sig,lt)
    Q_ref = max(np.sqrt(2*S*max(mu,1)/max(h,1e-9)),1.)
    r0 = float(o.get("Baslangic_Stok", mu_L+1.65*sigma_L))
    mev = analitik_maliyet(max(Q_ref,1),r0,mu,mu_L,sigma_L,h,p,S)
    best_Qg,best_rg,_,_ = grid_search_rQ(mu,sig,lt,h,p,S,Q_ref,grid_adim)
    opt = optuna_rQ(mu,sig,lt,h,p,S,best_Qg,best_rg,n_trials=n_trials)
    Q_eoq  = max(int(round(Q_ref)),1)
    r_eoq  = int(round(mu_L+1.65*sigma_L))
    SS_eoq = int(round(1.65*sigma_L))
    eoq    = analitik_maliyet(Q_eoq,r_eoq,mu,mu_L,sigma_L,h,p,S)
    tasarruf_tl   = mev["toplam"]-opt["min_maliyet"]
    tasarruf_oran = (tasarruf_tl/mev["toplam"]*100) if mev["toplam"]>0 else 0.
    return {"optimal_Q":opt["optimal_Q"],"optimal_r":opt["optimal_r"],
            "optimal_SS":opt["optimal_SS"],"hizmet_duzeyi":opt["hizmet_duzeyi"],
            "yeni_maliyet":opt["min_maliyet"],"mevcut_maliyet":mev["toplam"],
            "tasarruf_tl":tasarruf_tl,"tasarruf_oran":tasarruf_oran,
            "elde_tutma_yeni":opt["elde_tutma"],"siparis_maliyet_yeni":opt["siparis_maliyeti"],
            "stoksuz_maliyet_yeni":opt["stoksuz_maliyet"],
            "elde_tutma_mev":mev["elde_tutma"],"siparis_maliyet_mev":mev["siparis"],
            "stoksuz_maliyet_mev":mev["stoksuz"],
            "Q_eoq":Q_eoq,"r_eoq":r_eoq,"SS_eoq":SS_eoq,
            "et_eoq":eoq["elde_tutma"],"si_eoq":eoq["siparis"],
            "sk_eoq":eoq["stoksuz"],"tc_eoq":eoq["toplam"],
            "mu":mu,"sigma_L":sigma_L,"mu_L":mu_L}

def aksiyon_uyarisi(opt_res,tahminler):
    Q,r,SS,hiz = opt_res["optimal_Q"],opt_res["optimal_r"],opt_res["optimal_SS"],opt_res["hizmet_duzeyi"]
    trend = ((tahminler[-1]-tahminler[0])/max(tahminler[0],1)*100 if len(tahminler)>=2 else 0.)
    if hiz<.85:   return {"renk":"kirmizi","trend":trend,"mesaj":f"🔴 KRİTİK: Hizmet düzeyi düşük (%{hiz*100:.0f}). Acil sipariş önerilir. Q={Q:,} adet verin."}
    elif trend>20: return {"renk":"sari","trend":trend,  "mesaj":f"🟡 UYARI: Talep %{trend:.0f} artış bekleniyor. Q={Q:,} adet sipariş verin, r={r:,} eşiğini izleyin."}
    elif trend<-20:return {"renk":"mavi","trend":trend,  "mesaj":f"🔵 BİLGİ: Talep %{abs(trend):.0f} azalış bekleniyor. Sipariş öncesi stoku kontrol edin."}
    else:          return {"renk":"yesil","trend":trend, "mesaj":f"🟢 NORMAL: Q={Q:,} adet sipariş verin. r={r:,} eşiğinde yeni sipariş tetikleyin. SS={SS:,} bulundurun."}
