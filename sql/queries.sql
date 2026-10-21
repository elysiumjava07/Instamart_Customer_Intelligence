-- Part A SQL analysis. Run against data/instamart.db. Queries are separated by "-- Qn" markers.
-- Only completed orders count as revenue/spend.

-- Q1: WHERE filters rows before grouping; HAVING filters groups after aggregation
SELECT customer_id, COUNT(*) AS n_orders, ROUND(SUM(order_amount), 2) AS total_amount
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_amount DESC;

-- Q2: three-table join, revenue by city and category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: scalar subquery, products priced above the catalogue average
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN, customers with no completed order
-- (the status test sits in the ON clause so customers with only cancelled orders are kept)
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: two CTEs, customers whose spend is above the average customer spend
WITH cust_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders WHERE status = 'completed' GROUP BY customer_id
), avg_spend AS (
    SELECT AVG(total_spend) AS avg_total FROM cust_spend
)
SELECT s.customer_id, ROUND(s.total_spend, 2) AS total_spend, ROUND(a.avg_total, 2) AS avg_total
FROM cust_spend s CROSS JOIN avg_spend a
WHERE s.total_spend > a.avg_total
ORDER BY s.total_spend DESC;

-- Q6: ranking window functions within each city
WITH spend AS (
    SELECT o.customer_id, c.city, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, row_num;

-- Q7: daily revenue with running total
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, ROUND(SUM(order_amount), 2) AS revenue
    FROM orders WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
