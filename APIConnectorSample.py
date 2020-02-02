import pyodbc
import datetime
import time
import requests
import random
from tqdm import tqdm


def send_product(prod_id, prod_name, prod_cat_1, prod_cat_2, prod_cat_3,
    meta_prod_id, net_purch_price, net_purch_currency, net_sales_price,
    net_sales_currency, std_sales_volume, sales_unit, active):

    if prod_id in meta_product_list:
        send_meta_products(prod_id, prod_name, True)
    else:
        product = {
            "prod_id": f"{prod_id}",
            "name": f"{prod_name}",
            "prod_cat1": f"{prod_cat_1}",
            "prod_cat2": f"{prod_cat_2}",
            "prod_cat3": f"{prod_cat_3}",
            "meta_prod_id": meta_prod_id,
            "net_purch_price": f"{net_purch_price}",
            "net_purch_currency": f"{net_purch_currency}",
            "net_sales_price": f"{net_sales_price}",
            "net_sales_currency": f"{net_sales_currency}",
            "std_sales_volume": f"{std_sales_volume}",
            "sales_unit": f"{sales_unit}",
            "active": f"{active}"
        }

        r = requests.post(
            'https://api.sdasp.com/products/',
            json=product,
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Token XXXXXXXXXX',
                'user-agent': 'Client_Name-Connector'
            }
        )
        if r.status_code != 200:
            capture_message(product)


def send_customer(cust_id, cust_name, sales_rep, cust_group, country, postal_code, street, house_number, last_visit, last_phone_call, active):
    customer = {
        "cust_id": f"{cust_id}",
        "name": f"{cust_name}",
        "sales_rep": f"{sales_rep}",
        "customer_group": f"{cust_group}",
        "country": f"{country}",
        "postal_code": f"{postal_code}",
        "street": f"{street}",
        "house_number": f"{house_number}",
        "last_visit": last_visit,
        "last_phone_call": last_phone_call,
        "active": f"{active}"
    }

    r = requests.post(
        'https://api.sdasp.com/customers/',
        json=customer,
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Token XXXXXXXXXX',
            'user-agent': 'Client_Name-Connector'
        }
    )
    if r.status_code != 200:
        capture_message(customer)


def send_transaction(transaction_id, net_worth, currency, amount, unit, date, prod_id, cust_id):
    transaction = {
        "transaction_id": f"{transaction_id}",
        "net_worth": f"{net_worth}",
        "currency": f"{currency}",
        "amount": f"{amount}",
        "unit": f"{unit}",
        "date": f"{date}",
        "prod_id": f"{prod_id}",
        "cust_id": f"{cust_id}"
    }

    r = requests.post(
        'https://api.sdasp.com/transactions/',
        json=transaction,
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Token XXXXXXXXXX',
            'user-agent': 'Client_Name-Connector'
        }
    )
    if r.status_code != 200:
        capture_message(transaction)


def send_meta_products(meta_prod_id, name, active):
    meta_product = {
        "meta_prod_id": f"{meta_prod_id}",
        "name": f"{name}",
        "active": f"{active}"
    }

    r = requests.post(
        'https://api.sdasp.com/metaproducts/',
        json=meta_product,
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Token XXXXXXXXXX',
            'user-agent': 'Client_Name-Connector'
        }
    )
    if r.status_code != 200:
        capture_message(meta_product)


if __name__ == "__main__":
    
    # Establish server connection to data source
    # SQL_DB_DRIVER_NAME -> DB-specific, must be identifed before connection
    # SERVER_NAME -> server name, on which database is running
    # DB_NAME -> database, which contains data of interest

    conn = pyodbc.connect('Driver={SQL_DB_DRIVER_NAME};'
                        'Server=WS-SERVER_NAME;'
                        'Database=DB_NAME;'
                        'Trusted_Connection=yes;')


    # meta product data
    cursor = conn.cursor()
    cursor.execute(
        """
    SELECT meta_prod_id, meta_prod_name from METAPRODUCTS;

        """
    )

    for meta_prod_id, meta_prod_name in tdqm(cursor):

        send_meta_products(
            meta_prod_id=meta_prod_id,
            meta_prod_id=meta_prod_id,
            name=meta_prod_name
        )

    # product master data
    cursor = conn.cursor()
    cursor.execute(
        """
    SELECT prod_id, prod_name, prod_cat1, prod_cat2, prod_cat3, prod_cat4, 
           meta_prod_id, net_sales_price, net_sales_currency
           FROM PRODUCTS;
    """)

    for prod_id, prod_name, prod_cat_1, prod_cat_2, prod_cat_3, prod_cat_4, meta_prod_id, net_sales_price in tqdm(cursor):
        net_sales_currency = '€'


        send_product(prod_id, prod_name, prod_cat_1, prod_cat_2, prod_cat_3,
                     meta_prod_id, net_purch_price, net_purch_currency, net_sales_price,
                     net_sales_currency, std_sales_volume, sales_unit)


    # customer master data
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT cust_id, cust_name, sales_rep, country, postal_code_ city, street, house_number, cust_group
        FROM CUSTOMERS; 
        """
    )

    for cust_id, cust_name, sales_rep, country, postal_code, city, \
      street, house_number, cust_group in tqdm(cursor):
      send_customer(
          cust_id=cust_id,
          cust_name=cust_name,
          sales_rep=sales_rep,
          cust_group=cust_group,
          country=country,
          postal_code=postal_code,
          street=street,
          house_number=house_number
      )


    # transaction data
    cursor = conn.cursor()
    cursor.execute(
        """
    SELECT transaction_id, date, worth, currency, amount, unit, prod_id, cust_id FROM transactions
        """)

    for transaction_id, date, worth, currency, amount, unit, prod_id, cust_id in tqdm(cursor):

        send_transaction(
            transaction_id=transaction_id,
            net_worth=worth,
            currency=currency,
            amount=amount,
            unit=unit,
            date=date,
            prod_id=prod_id,
            cust_id=cust_id
        )
