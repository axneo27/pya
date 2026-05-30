# AI Product Database Assistant

This repository uses an MCP server built with FastMCP
for working with product data stored in MS SQL.

The AI assistant must use MCP tools whenever
working with products or categories.

---

# MCP Server

Framework:
- FastMCP

Database:
- Microsoft SQL Server

---

# Available MCP Tools

## get_categories()

Description:
Returns all product categories.

Returns:
- Id
- Name

Example:
get_categories()

---

## get_products_by_category(category_id)

Description:
Returns products for a category.

Parameters:
- category_id: int

Returns:
- Id
- Name
- Price
- Category

Example:
get_products_by_category(1)

---

## get_top_products(category_id, top_n)

Description:
Returns TOP expensive products
sorted by price descending.

Parameters:
- category_id: int
- top_n: int

Returns:
- Id
- Name
- Price
- Category

Example:
get_top_products(1, 3)

---

# AI Responsibilities

When user asks about products:

1. Use MCP tools first
2. Avoid generating fake product data
3. Analyze returned database data
4. Format responses clearly
5. Explain product statistics briefly

---

# Development Rules

- Use parameterized SQL queries
- Never hardcode credentials
- Use environment variables
- Prefer async operations where possible
- Validate all tool parameters
- Use concise code comments

---

# Product Analysis Rules

When displaying products:
- sort by price descending when relevant;
- clearly display category names;
- format prices consistently;
- explain expensive products briefly.

---

# Example Workflow

User Request:
"Show top phones"

Expected AI behavior:

1. Call:
   get_categories()

2. Detect:
   Phones category id

3. Call:
   get_top_products(category_id, 3)

4. Generate concise response

---

# Example Output

TOP 3 Phones:

1. iPhone 15 — 52000
2. Samsung S24 — 47000
3. Xiaomi 14 — 31000

iPhone 15 is the most expensive phone
currently available.