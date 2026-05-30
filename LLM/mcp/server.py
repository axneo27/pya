from contextlib import closing

from fastmcp import FastMCP

from db import get_connection

mcp = FastMCP("Products MCP Server")


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


@mcp.tool()
def get_categories():
    with closing(get_connection()) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT Id, Name
            FROM dbo.Categories
            ORDER BY Name
            """
        )
        rows = cursor.fetchall()

    return [{"Id": row[0], "Name": row[1]} for row in rows]


@mcp.tool()
def get_products_by_category(category_id: int):
    _validate_positive_int(category_id, "category_id")

    with closing(get_connection()) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                p.Id,
                p.Title AS Name,
                p.Price,
                c.Name AS Category
            FROM dbo.Products AS p
            INNER JOIN dbo.Categories AS c ON p.CategoryId = c.Id
            WHERE c.Id = %s
            ORDER BY p.Price DESC, p.Id ASC
            """,
            (category_id,),
        )
        rows = cursor.fetchall()

    return [
        {
            "Id": row[0],
            "Name": row[1],
            "Price": float(row[2]),
            "Category": row[3],
        }
        for row in rows
    ]


@mcp.tool()
def get_top_products(category_id: int, top_n: int = 3):
    _validate_positive_int(category_id, "category_id")
    _validate_positive_int(top_n, "top_n")

    with closing(get_connection()) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            WITH ranked_products AS (
                SELECT
                    p.Id,
                    p.Title AS Name,
                    p.Price,
                    c.Name AS Category,
                    ROW_NUMBER() OVER (ORDER BY p.Price DESC, p.Id ASC) AS rn
                FROM dbo.Products AS p
                INNER JOIN dbo.Categories AS c ON p.CategoryId = c.Id
                WHERE c.Id = %s
            )
            SELECT Id, Name, Price, Category
            FROM ranked_products
            WHERE rn <= %s
            ORDER BY Price DESC, Id ASC
            """,
            (category_id, top_n),
        )
        rows = cursor.fetchall()

    return [
        {
            "Id": row[0],
            "Name": row[1],
            "Price": float(row[2]),
            "Category": row[3],
        }
        for row in rows
    ]


if __name__ == "__main__":
    mcp.run()