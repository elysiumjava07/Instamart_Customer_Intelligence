"""Part B tasks 4-8: lapse definition, leak-free preprocessing, tuning, balancing, interpretation.
Run after part_b_segments.py:  python src/part_b_lapse.py"""
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF, END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
seg = pd.read_csv("outputs/customer_segments.csv")

# ---- task 4: cohort, label, features (only orders BEFORE the cutoff feed the features) ----
orders = orders.merge(prod[["product_id", "category"]], on="product_id")
pre = orders[orders.order_ts < CUTOFF]
done_pre = pre[pre.status == "completed"]
g = done_pre.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().sort_index().idxmax()),
})
feat["cancel_rate"] = (pre[pre.status == "cancelled"].groupby("customer_id").size()
                       .reindex(feat.index, fill_value=0) / pre.groupby("customer_id").size().reindex(feat.index))
feat = feat.join(cust.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
feat["tenure_days"] = (CUTOFF - feat.pop("signup_date")).dt.days
sept = orders[(orders.order_ts >= CUTOFF) & (orders.order_ts < END) & (orders.status == "completed")]
y = (~feat.index.isin(sept.customer_id)).astype(int)
y = pd.Series(y, index=feat.index, name="lapsed")
print(f"cohort={len(feat)} lapsed={y.sum()} ({y.mean()*100:.2f}%)")

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X = feat[NUM + CAT]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train/test:", len(X_tr), len(X_te))
base = np.zeros(len(y_te), int)
print("baseline accuracy %.3f F1 %.3f" % (accuracy_score(y_te, base), f1_score(y_te, base, zero_division=0)))

# ---- task 5: preprocessing inside a Pipeline (fitted on training rows only) ----
def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
assert X.isna().sum().sum() == 0, "missing values in features"
print("no missing values in any feature column: True")
chk = prep().fit(X_tr)
print("scaler mean(frequency) train=%.3f | full cohort=%.3f" % (chk.named_transformers_["num"].mean_[NUM.index("frequency")], X.frequency.mean()))

# ---- task 6: tuning ----
def metrics(m, Xe, ye):
    p = m.predict(Xe)
    return dict(precision=precision_score(ye, p, zero_division=0), recall=recall_score(ye, p, zero_division=0),
                f1=f1_score(ye, p, zero_division=0)), confusion_matrix(ye, p)
lr = GridSearchCV(Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000))]),
                  {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
rf = GridSearchCV(Pipeline([("prep", prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                  {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(X_tr, y_tr)
res = {}
for name, gs in [("logreg", lr), ("forest", rf)]:
    m, cm = metrics(gs.best_estimator_, X_te, y_te); res[name] = m
    print(f"{name}: best={gs.best_params_} cv_f1={gs.best_score_:.3f} | test P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(" confusion [[TN FP],[FN TP]]:", cm.tolist())

# selection uses cross-validated F1 on training data (not the test set)
sel_name, sel = ("forest", rf) if rf.best_score_ >= lr.best_score_ else ("logreg", lr)
model = sel.best_estimator_
print("selected model:", sel_name, "| test F1 =", round(res[sel_name]["f1"], 3), "(must be >= 0.55)")

# ---- task 7: class-balance experiment (training rows only) ----
lapsed_idx = X_tr[y_tr == 1]
n_add = int((y_tr == 0).sum() - (y_tr == 1).sum())
extra = lapsed_idx.sample(n=n_add, replace=True, random_state=42)
X_bal = pd.concat([X_tr, extra]); y_bal = pd.concat([y_tr, pd.Series(1, index=extra.index)])
print("balanced train class counts:", y_bal.value_counts().to_dict())
lr_bal = clone(lr.best_estimator_).fit(X_bal, y_bal)
m_bal, cm_bal = metrics(lr_bal, X_te, y_te)
m_lr = res["logreg"]
print("logreg unbalanced: P=%.3f R=%.3f F1=%.3f" % (m_lr["precision"], m_lr["recall"], m_lr["f1"]))
print("logreg balanced  : P=%.3f R=%.3f F1=%.3f" % (m_bal["precision"], m_bal["recall"], m_bal["f1"]), cm_bal.tolist())

# ---- task 8: interpret, persist, use ----
names = model.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.best_estimator_.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 5 forest importances:\n", imp.head(5).round(3).to_string())
coef = pd.Series(lr.best_estimator_.named_steps["clf"].coef_[0], index=lr.best_estimator_.named_steps["prep"].get_feature_names_out())
print("logreg coefficients:\n", coef[["num__recency", "num__frequency", "num__tenure_days"]].round(3).to_string())

joblib.dump(model, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
assert (re.predict(X_te) == model.predict(X_te)).all()
print("reloaded model reproduces test predictions exactly: True")

prem = set(seg.loc[seg.is_premium, "customer_id"])
proba = pd.Series(model.predict_proba(X_te)[:, 1], index=X_te.index)
sub = proba[proba.index.isin(prem)].sort_values(ascending=False).head(10)
ret = pd.DataFrame({"customer_id": sub.index, "lapse_probability": sub.values.round(4),
                    "actually_lapsed": y_te.loc[sub.index].values})
ret.to_csv("outputs/retention_list.csv", index=False)
print(f"retention list: {len(ret)} premium test customers, {int(ret.actually_lapsed.sum())} truly lapsed")
print("premium customers in test set:", int(X_te.index.isin(prem).sum()))
