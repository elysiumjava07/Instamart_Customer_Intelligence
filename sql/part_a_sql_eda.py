"""Part A: run the seven SQL queries, pandas cross-check and EDA. Run: python src/part_a_sql_eda.py"""
import re, sqlite3
import numpy as np, pandas as pd

con = sqlite3.connect("data/instamart.db")
sql = open("sql/queries.sql").read()
parts = re.split(r"^-- Q(\d):.*$", sql, flags=re.M)           # ['', '1', body, '2', body, ...]
Q = {int(parts[i]): "\n".join(l for l in parts[i + 1].splitlines() if not l.startswith("--")).strip()
     for i in range(1, len(parts), 2)}
R = {n: pd.read_sql_query(q, con) for n, q in Q.items()}
pd.set_option("display.width", 160)

print("Q1 customers:", len(R[1]))
q2 = R[2]
print("Q2 rows:", len(q2), "| sum:", round(q2.revenue.sum(), 2)); print(q2.head(1).to_string(index=False))
print("Q3 products above average:", len(R[3]))
print("Q4 customers without completed order:", len(R[4]))
q5 = R[5]; print("Q5 customers above average:", len(q5), "| average:", q5.avg_total.iloc[0])
q6 = R[6]
print("Q6 top customer per city:"); print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows with RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
q7 = R[7]; print("Q7 daily rows:", len(q7), "| last cumulative:", q7.cumulative_revenue.iloc[-1])

# ---------- pandas cross-check ----------
o = pd.read_sql_query("SELECT * FROM orders", con); c = pd.read_sql_query("SELECT * FROM customers", con)
p = pd.read_sql_query("SELECT * FROM products", con)
done = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(p[["product_id", "category"]], on="product_id")
pv = pd.pivot_table(done, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("\npivot grand total:", round(pv.loc["All", "All"], 2), "| equals SQL total:", np.isclose(pv.loc["All", "All"], q2.revenue.sum()))
city_stats = done.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                     orders=("order_amount", "count")).round(2)
print(city_stats.to_string())

# ---------- EDA on completed order_amount ----------
a = done["order_amount"]
q1_, q3_ = a.quantile(.25), a.quantile(.75); iqr = q3_ - q1_
lo, hi = q1_ - 1.5 * iqr, q3_ + 1.5 * iqr
print(f"\nmean={a.mean():.2f} median={a.median():.2f} skew={a.skew():.2f} Q1={q1_:.2f} Q3={q3_:.2f}")
print(f"fences: lower={lo:.2f} upper={hi:.2f} | below={int((a < lo).sum())} above={int((a > hi).sum())}")
corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
print("largest |corr| pair:", pairs.abs().idxmax(), round(pairs[pairs.abs().idxmax()], 3))
