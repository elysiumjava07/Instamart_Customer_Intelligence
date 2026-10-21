"""Validate Part A reference checkpoints. Run: python src/validate_part_a.py"""
import sqlite3
import pandas as pd

c = pd.read_csv("data/clean/customers.csv"); o = pd.read_csv("data/clean/orders.csv")
con = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
done = o[o.status == "completed"]
checks = {
    "orders has 12,309 rows": len(o) == 12309,
    "order_id unique": o.order_id.is_unique,
    "11,110 completed": (o.status == "completed").sum() == 11110,
    "1,199 cancelled": (o.status == "cancelled").sum() == 1199,
    "no pending rows": (o.status == "pending").sum() == 0,
    "no missing order_amount": o.order_amount.isna().sum() == 0,
    "no missing delivery (completed)": done.delivery_minutes.isna().sum() == 0,
    "1,199 missing delivery (cancelled)": o[o.status == "cancelled"].delivery_minutes.isna().sum() == 1199,
    "completed revenue 2,998,668.80": round(done.order_amount.sum(), 2) == 2998668.80,
    "six cities": c.city.nunique() == 6 and c.city.value_counts().to_dict() ==
        {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "genders 610/560/30": c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "ages complete and 18-60": c.age.isna().sum() == 0 and c.age.between(18, 60).all(),
    "db has 3 tables": {"customers", "products", "orders"} <= tables,
    "db orders count matches": con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309,
}
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
assert all(checks.values()), "some checkpoints failed"
print("ALL CHECKPOINTS PASSED")
