import os
import pandas as pd

orders = pd.read_csv('../assets/orders.csv')
orders['OrderDate'] = pd.to_datetime(orders['OrderDate'])

orders['TotalAmount'] = orders['Quantity'] * orders['Price']

print('--- Повна таблиця ---')
print(orders.to_string(index=False), '\n')

total_income = orders['TotalAmount'].sum()
print('a. Сумарний дохід магазину:', total_income)

mean_total = orders['TotalAmount'].mean()
print('b. Середнє TotalAmount:', mean_total)

orders_by_customer = orders.groupby('Customer').size().rename('OrderCount')
print('c. Кількість замовлень по кожному клієнту:')
print(orders_by_customer, '\n')

big_orders = orders[orders['TotalAmount'] > 500]
print('4. Замовлення з TotalAmount > 500:')
print(big_orders.to_string(index=False), '\n')

sorted_desc = orders.sort_values('OrderDate', ascending=False)
print('5. Сортування за OrderDate у зворотному порядку:')
print(sorted_desc.to_string(index=False), '\n')

mask = (orders['OrderDate'] >= '2023-06-05') & (orders['OrderDate'] <= '2023-06-10')
selected_period = orders.loc[mask]
print('6. Замовлення з 5 по 10 червня включно:')
print(selected_period.to_string(index=False), '\n')

category_summary = orders.groupby('Category').agg(
    ItemsCount=('Quantity', 'sum'),
    TotalSales=('TotalAmount', 'sum')
).reset_index()
print('7. Групування за Category:')
print(category_summary.to_string(index=False), '\n')

customer_total = orders.groupby('Customer', as_index=False).agg(TotalAmount=('TotalAmount', 'sum'))
customer_top3 = customer_total.sort_values('TotalAmount', ascending=False).head(3)
print('8. ТОП-3 клієнтів за TotalAmount:')
print(customer_top3.to_string(index=False), '\n')

orders.to_csv('../assets/orders_processed.csv', index=False)
customer_top3.to_csv('../assets/top3_customers.csv', index=False)
print('Files written: ../assets/orders_processed.csv , ../assets/top3_customers.csv')
