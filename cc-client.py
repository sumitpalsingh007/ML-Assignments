import json
import uuid

import boto3

bucket_name = '2017ht13042-bucket1'

inputValueStr = input("Enter comma separated list of numbers: ")
print("input received is {}".format(inputValueStr))

processString = input("Enter the process to be performed wrapped in '' (sum, mean, min, max): ")
print("processString received is {}".format(processString))

# generate the unique key/pointer for the s3 object
message_identifier = str(uuid.uuid4())

# location of the object on s3 bucket
object_location = 'location1'

#to be used to send message details
input_dict = {}
input_dict['object_location'] = object_location
input_dict['process'] = processString
input_dict['message_identifier'] = message_identifier

# populate the value in s3 bucket
#s3 = boto3.resource('s3')
s3 = boto3.resource('s3', region_name='ap-south-1',
                    aws_access_key_id='AKIAJYWVLYIJDAZVLRQQ',
                    aws_secret_access_key='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')

# Print out bucket names, just for logging purpose
for bucket in s3.buckets.all():
    print(bucket.name)

# create a uniqie key based on the message identifier
object = s3.Object(bucket_name, object_location)

#convert the dict to json
print('converting:- {} to json'.format(input_dict))
final_message_str = json.dumps(input_dict)
print('converted json:- {}'.format(final_message_str))

# populate the body of the s3 object with the comma separated values string
object.put(Body=inputValueStr)
    
# Get the sqs service
#sqs = boto3.resource('sqs')
sqs = boto3.resource('sqs', region_name='ap-south-1',
                     aws_access_key_id='AKIAJYWVLYIJDAZVLRQQ',
                     aws_secret_access_key='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')

# Get the inbox queue. This returns an SQS.Queue instance
inbox_queue = sqs.get_queue_by_name(QueueName='2017ht13042-queue1')

# You can now access identifiers and attributes
print(inbox_queue.url)
print(inbox_queue.attributes.get('DelaySeconds'))

# Create a new message
# pass the unique identifier for the s3 object
response = inbox_queue.send_message(MessageBody=final_message_str, MessageAttributes={
    'Author': {
        'StringValue': 'Sumit',
        'DataType': 'String'
    }
})

##Logic to get the value back from the out queue

# get the sqs client
#sqs_client = boto3.client('sqs')
sqs_client = boto3.client('sqs', region_name='ap-south-1',
                          aws_access_key_id='AKIAJYWVLYIJDAZVLRQQ',
                          aws_secret_access_key='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')

# Get the outbox queue. This returns an SQS.Queue instance
outbox_queue = sqs.get_queue_by_name(QueueName='2017ht13042-queue3')

while True:
    # Enable long polling on an existing SQS queue
    output_queue_response = sqs_client.receive_message(
        QueueUrl=outbox_queue.url,
        AttributeNames=[
            'SentTimestamp'
        ],
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        WaitTimeSeconds=10
    )
    
    if(output_queue_response.get('Messages')):
        outbox_body_json = json.loads(output_queue_response['Messages'][0]['Body'])
        print(outbox_body_json)
        
        output_object_location = outbox_body_json['object_location']
        process = outbox_body_json['process']
        output_message_identifier = outbox_body_json['message_identifier']
        
        if(message_identifier == output_message_identifier):
            break
    
output_object = s3.Object(bucket_name, output_object_location)
valueFromObj = output_object.get()['Body'].read().decode('utf-8')

sqs_client.delete_message(QueueUrl=outbox_queue.url,ReceiptHandle=output_queue_response['Messages'][0]['ReceiptHandle'])