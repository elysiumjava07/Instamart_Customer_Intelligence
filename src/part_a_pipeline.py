"""Part A: audit raw exports, clean them, store CSV + SQLite. Run: python src/part_a_pipeline.py"""
import os, sqlite3
import numpy as np, pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True)

def load(name):
    # only truly blank cells are missing; 'N/A' and 'error' stay as text so we can count them
    return pd.read_csv(os.path.join(RAW, name), na_values=[""], keep_default_na=False)

customers, products, orders = load("customers.csv"), load("products.csv"), load("orders.csv")

# ---------------- audit ----------------
print("=== AUDIT ===")
for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print(f"\n{name}: shape={df.shape}")
    summary = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                            "missing_pct": (df.isna().mean() * 100).round(2)})
    print(summary.to_string())

amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(customers["age"], errors="coerce")
audit = {
    "raw order rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": orders["order_id"].nunique(),
    "non-numeric order_amount": int((amount_num.isna() & orders["order_amount"].notna()).sum()),
    "negative delivery_minutes": int((deliv < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(customers["age"].isna().sum()),
    "age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "distinct raw city spellings": customers["city"].nunique(),
    "distinct non-blank raw gender spellings": customers["gender"].nunique(),
    "blank gender": int(customers["gender"].isna().sum()),
}
print("\n=== AUDIT FIGURES ==="); [print(f"{k}: {v}") for k, v in audit.items()]

# ---------------- clean customers ----------------
c = customers.copy()
c["city"] = c["city"].str.strip().str.title()                                   # (a)
g = c["gender"].fillna("").str.strip().str.lower().str[:1]                      # (b)
c["gender"] = g.map({"m": "male", "f": "female"}).fillna("unknown")
c["age"] = pd.to_numeric(c["age"], errors="coerce")                             # (c)
n_bad_age = int(((c["age"] < 10) | (c["age"] > 100)).sum())
c.loc[(c["age"] < 10) | (c["age"] > 100), "age"] = np.nan
n_age_filled = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["signup_date"] = pd.to_datetime(c["signup_date"])                             # (d)
print("\n=== CLEAN CUSTOMERS ===")
print(c["city"].value_counts().to_string()); print(c["gender"].value_counts().to_string())
print("missing ages:", c["age"].isna().sum(), "| age range:", c["age"].min(), c["age"].max())
print(f"rule counts: city cells changed={int((customers['city'] != c['city']).sum())}, "
      f"impossible ages blanked={n_bad_age}, ages filled (incl. blanks)={n_age_filled}")

# ---------------- clean orders ----------------
o = orders.copy()
o["order_ts"] = pd.to_datetime(o["order_ts"])                                    # 1
n0 = len(o)
o = o.drop_duplicates()                                                          # 2
n_exact = n0 - len(o)
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")   # 3
n_latest = n1 - len(o)
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")            # 4
o = o.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
miss_amt = o["order_amount"].isna()
n_amt = int(miss_amt.sum())
o.loc[miss_amt, "order_amount"] = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)[miss_amt]
o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")    # 5
neg = o["delivery_minutes"] < 0
n_neg = int(neg.sum())
o.loc[neg, "delivery_minutes"] = np.nan
done = o["status"] == "completed"
med = o[done].groupby("category")["delivery_minutes"].transform("median")      # completed rows only
fill = done & o["delivery_minutes"].isna()
n_deliv = int(fill.sum())
o.loc[fill, "delivery_minutes"] = med[fill]
o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
print("\n=== CLEAN ORDERS ===")
print(f"rule counts: exact duplicates removed={n_exact}, superseded records removed={n_latest}, "
      f"amounts imputed={n_amt}, negative delivery set missing={n_neg}, completed delivery imputed={n_deliv}")
print("rows:", len(o), "| status:", o["status"].value_counts().to_dict())
print("missing amount:", o["order_amount"].isna().sum(),
      "| missing delivery completed:", o.loc[o.status == 'completed', 'delivery_minutes'].isna().sum(),
      "| missing delivery cancelled:", o.loc[o.status == 'cancelled', 'delivery_minutes'].isna().sum())
print("completed revenue INR:", round(o.loc[o.status == 'completed', 'order_amount'].sum(), 2))

# ---------------- store ----------------
o_out = o.copy(); o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out = c.copy(); c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
products.to_csv(f"{CLEAN}/products.csv", index=False)
o_out.to_csv(f"{CLEAN}/orders.csv", index=False)
if os.path.exists(DB): os.remove(DB)
with sqlite3.connect(DB) as con:
    c_out.to_sql("customers", con, index=False)
    products.to_sql("products", con, index=False)
    o_out.to_sql("orders", con, index=False)
print("\nstored: data/clean/*.csv and", DB)
