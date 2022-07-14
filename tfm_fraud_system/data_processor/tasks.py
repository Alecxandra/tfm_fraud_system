from config import celery_app
import os
import pandas as pd
from .models import Customer, Transaction
from .import constants



def save_customers():
    # ssn | cc_num | first | last | gender | street | city | state | zip | lat | long | city_pop | job | dob | acct_num | profile

    df = pd.read_csv('./data-test/customers.csv', sep='|')
    row_iter = df.iterrows()

    for index, row in row_iter:
        customer_data = {
            'identifier': row['ssn'],
            'cc_number': row['cc_num'],
            'first_name': row['first'],
            'last_name': row['last'],
            'gender': row['gender'],
            'street': row['street'],
            'city': row['city'],
            'state': row['state'],
            'lat': row['lat'],
            'lon': row['long'],
            'job': row['job'],
            'account_number': row['acct_num'],
            'profile': row['profile'],
            'row_number': index
        }

        current_customer = Customer.objects.create(**customer_data)

        # Get current customer transactions
        range =  f"{index}-{index + 1}" if index % 2 == 0 else f"{index - 1}-{index}"
        file_url = f"{row['profile'].split('.')[0]}_{range}.csv"

        df_transactions = pd.read_csv(f"./data-test/{file_url}", sep='|')

        transactions_row_iter = df_transactions.iterrows()

        print("[data_processor][generate_test_data] Generate customer transactions")

        for t_index, t_row in transactions_row_iter:
            # save transactions
            # trans_num|trans_date|trans_time|unix_time|category|amt|is_fraud|merchant|merch_lat|merch_long

            transaction_data = {
                'transaction_number' : t_row['trans_num'],
                'transaction_date': t_row['trans_date'],
                'unix_time': t_row['unix_time'],
                'category': t_row['category'],
                'amt': t_row['amt'],
                'is_fraud': t_row['is_fraud'],
                'merchant': t_row['merchant'],
                'merch_lat': t_row['merch_lat'],
                'merch_long': t_row['merch_long'],
                'customer': current_customer,
                'environment': constants.Transaction.Environment.TESTING
            }

            Transaction.objects.create(**transaction_data)


@celery_app.task(soft_time_limit=7200, time_limit=7200)
def generate_test_data(packet):
    # Llamar el comando para generar los archivos con datos

    script_url = os.path.join(os.path.dirname(__file__), 'sparkvo_data_generation/datagen.py')

    customers = packet.get('customers', 10)
    output = packet.get('output', 'data-test')
    start_date = packet.get('start_date')
    end_date=packet.get('end_date')

    os.system('{} {}'.format('python', f'{script_url} -n {customers} -o {output} {start_date} {end_date}'))

    print("[data_processor][generate_test_data] Se guarda la información de los clientes generados")

    save_customers()
