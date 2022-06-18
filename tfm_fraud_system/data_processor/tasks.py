from config import celery_app
import os
import subprocess

@celery_app.task()
def generate_test_data(packet):
    # Llamar el comando para generar los archivos con datos

    script_url = os.path.join(os.path.dirname(__file__), 'sparkvo_data_generation/datagen.py')

    customers = packet.get('customers', 10)
    output = packet.get('output', 'data-test')
    start_date = packet.get('start_date')
    end_date=packet.get('end_date')

    os.system('{} {}'.format('python', f'{script_url} -n {customers} -o {output} {start_date} {end_date}'))
