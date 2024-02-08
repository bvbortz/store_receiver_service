import json
import logging
import requests
import pika
from api.services.handler import image_handler
import threading

def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
        logging.info("finished running func")
    logging.info("started running func")
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t

class rabbitMQServer():
    """
    Producer component that will publish message and handle
    connection and channel interactions with RabbitMQ.
    """

    def __init__(self, queue, host, routing_key, username, password, exchange=''):
        self._queue = queue
        self._host = host
        self._routing_key = routing_key
        self._exchange = exchange
        self._username = username
        self._password = password
        self.start_server()
        set_interval(self._connection.process_data_events, 40)

    def start_server(self):
        self.create_channel()
        self.create_exchange()
        self.create_bind()
        logging.info("Channel created...")

    def create_channel(self):
        credentials = pika.PlainCredentials(username=self._username, password=self._password)
        parameters = pika.ConnectionParameters(self._host, credentials=credentials)
        self._connection = pika.BlockingConnection(parameters)
        self._channel = self._connection.channel()

    def create_exchange(self):
        self._channel.exchange_declare(
            exchange=self._exchange,
            exchange_type='direct',
            passive=False,
            durable=True,
            auto_delete=False
        )
        self._channel.queue_declare(queue=self._queue, durable=False)

    def create_bind(self):
        self._channel.queue_bind(
            queue=self._queue,
            exchange=self._exchange,
            routing_key=self._routing_key
        )
        self._channel.basic_qos(prefetch_count=1)

    @staticmethod
    def callback(channel, method, properties, body):
        body_obj = json.loads(json.loads(body))
        logging.info(f'Consumed message {body_obj} from queue!')
        image_handler(body_obj['prompt'], body_obj['name'])
        
        try:
            r1 = requests.post(url="http://backend:5000/images/saved", data={'name': body_obj['name']})
            # r2 = requests.post(url="http://localhost/api/images/saved", data={'name': body_obj['name']})
        except Exception as e:
            logging.info("The error is: ",e)
        logging.info(f'saved image in minIO')

    def get_messages(self):
        try:
            logging.info("Starting the server...")
            self._channel.basic_consume(
                queue=self._queue,
                on_message_callback=rabbitMQServer.callback,
                auto_ack=True
            )
            self._channel.start_consuming()
        except Exception as e:
            logging.debug(f'Exception: {e}')
