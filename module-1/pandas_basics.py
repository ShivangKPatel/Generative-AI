import pandas as pd

def basic_dataframe_ops():
    # Create DataFrame
    data = {
        "city": ["NY", "NY", "SF", "SF", "LA"],
        "category": ["A", "B", "A", "B", "A"],
        "sales": [100, 150, 200, 120, 90],
        "discount": [5, 10, 0, 15, 0],
    }
    df = pd.DataFrame(data)

    # Basic inspection
    print("=== Head ===")
    print(df.head(), "\n")

    print("=== Info ===")
    print(df.info(), "\n")

    # Selecting columns
    print("=== Select columns ===")
    print(df[["city", "sales"]], "\n")

    # Filtering rows
    print("=== Filter rows (sales > 120) ===")
    print(df[df["sales"] > 120], "\n")

    # New columns
    df["net_sales"] = df["sales"] - df["discount"]
    print("=== With net_sales column ===")
    print(df, "\n")

    return df


def groupby_ops(df: pd.DataFrame):
    # Group by a single column
    print("=== Group by city: sum sales & net_sales ===")
    city_agg = (
        df.groupby("city")[["sales", "net_sales"]]
        .sum()
        .reset_index()
    )
    print(city_agg, "\n")

    # Group by multiple columns with multiple aggregations
    print("=== Group by city, category with multiple aggs ===")
    grouped = (
        df.groupby(["city", "category"])
        .agg(
            total_sales=("sales", "sum"),
            avg_sales=("sales", "mean"),
            order_count=("sales", "size"),
        )
        .reset_index()
    )
    print(grouped, "\n")

    return city_agg, grouped


def join_ops():
    # Left table: customers
    customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "city": ["NY", "SF", "LA"],
        }
    )

    # Right table: orders
    orders = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104],
            "customer_id": [1, 2, 2, 4],
            "amount": [250, 100, 150, 300],
        }
    )

    print("=== Customers ===")
    print(customers, "\n")

    print("=== Orders ===")
    print(orders, "\n")

    # INNER JOIN on customer_id
    inner = pd.merge(
        customers,
        orders,
        on="customer_id",
        how="inner",
    )
    print("=== Inner join on customer_id ===")
    print(inner, "\n")

    # LEFT JOIN to keep all customers
    left = pd.merge(
        customers,
        orders,
        on="customer_id",
        how="left",
    )
    print("=== Left join (all customers) ===")
    print(left, "\n")

    # FULL OUTER JOIN to keep all customers and orders
    outer = pd.merge(
        customers,
        orders,
        on="customer_id",
        how="outer",
        indicator=True,  # show source
    )
    print("=== Outer join (all rows) ===")
    print(outer, "\n")

    return inner, left, outer


def main():
    df = basic_dataframe_ops()
    groupby_ops(df)
    join_ops()


if __name__ == "__main__":
    main()