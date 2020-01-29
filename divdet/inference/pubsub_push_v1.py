"""
Push messages to a Google PubSub queue
"""

import csv
from pathlib import Path

from absl import app, flags, logging
from google.cloud import pubsub

FLAGS = flags.FLAGS

flags.DEFINE_string('input_fpath', None, 'Filepath to list of images.')
flags.DEFINE_string('gcp_project', None, 'Project ID of GCP project.')
flags.DEFINE_string('pubsub_topic_name', None, 'PubSub topic name to create.')
flags.DEFINE_string('pubsub_subscription_name', None,
                    'PubSub subscription name to create for topic.')
flags.DEFINE_integer('message_batch_size', 100,
                     'Number of messages per batch to send to PubSub')
flags.DEFINE_bool('publish_to_existing_topic', True,
                  'Whether or not it\'s okay to publish messages to an existing topic.')
flags.DEFINE_bool('publish_to_existing_subscription', True,
                  'Whether or not it\'s okay to publish messages to an existing subscription.')


def message_generator_csv(input_fpath):
    """Convert CSV file into dicts"""

    # Open csv file and create a generator to return each row
    with open(input_fpath) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            yield row


def create_topic_and_subscription(gcp_project, topic_name, subscription_name,
                                  publish_to_existing_topic,
                                  publish_to_existing_subscription):
    """Generate the GCP PubSub topic and subscription"""

    # Create publisher client
    publisher = pubsub.PublisherClient()
    topic_path = publisher.topic_path(gcp_project, topic_name)
    project_path = publisher.project_path(gcp_project)

    # Ensure the topic we want to send messages to exists
    existing_topics = [t.name for t in publisher.list_topics(project_path)]
    if not topic_path in existing_topics:
        if publish_to_existing_topic:
            publisher.create_topic(topic_path)
            logging.info(f'Created topic {topic_path}.')
        else:
            raise RuntimeError(f'Could not find topic {topic_name}.')

    # Ensure the specified subscripton exists
    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(gcp_project,
                                                     subscription_name)

    # Ensure the subscription for out topic exists
    existing_subs = [s.name for s in subscriber.list_subscriptions(project_path)]
    if not subscription_path in existing_subs:
        if publish_to_existing_subscription:
            subscription = subscriber.create_subscription(subscription_path,
                                                          topic_path)
            logging.info(f'Created subscription {subscription_path} for topic {topic_path}.')
        else:
            raise RuntimeError(f'Could not find subscription {subscription_path}.')


def push_messages(message_generator, gcp_project, topic_name, batch_size):
    """Push messages to a pubsub queue from a generator"""

    # Create publisher client
    publisher = pubsub.PublisherClient(
        batch_settings=pubsub.types.BatchSettings(max_messages=batch_size))

    # Identify topic name
    topic_path = publisher.topic_path(gcp_project, topic_name)

    # Send each message from generator
    for mi, message in enumerate(message_generator):
        try:
            data = f'{mi}'.encode('utf-8')
            future = publisher.publish(topic_path, data, **message)
            message_id = future.result()
        except Exception as e:
            logging.error(f'Message publish failed with error {e}')


def main(_):
    input_fpath_purepath = Path(FLAGS.input_fpath)

    # Construct a generator for messages
    # TODO: Add other input file options
    if input_fpath_purepath.suffix == '.csv':
        messages = message_generator_csv(FLAGS.input_fpath)
    else:
        raise NotImplementedError('')

    # Ensure the topic exists and there is a subscription for that topic
    create_topic_and_subscription(FLAGS.gcp_project, FLAGS.pubsub_topic_name,
                                  FLAGS.pubsub_subscription_name,
                                  FLAGS.publish_to_existing_topic,
                                  FLAGS.publish_to_existing_subscription)

    # Push messages to the topic
    push_messages(messages, FLAGS.gcp_project, FLAGS.pubsub_topic_name,
                  FLAGS.message_batch_size)


if __name__ == '__main__':
    app.run(main)