import boto3
import json
import decimal

bucket_name = 'arn:aws:s3:::2017ht13042-bucket1'

# Let's use Amazon S3
s3 = boto3.resource('s3', region_name='ap-south-1',
                          aws_access_key_id='AKIAI775LIZJSFSLN4EA',
                          aws_secret_access_key='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')

# Get the service resource
sqs = boto3.resource('sqs', region_name='ap-south-1',
                            aws_access_key_id='AKIAI775LIZJSFSLN4EA',
                            aws_secret_access_key='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')

# get the sqs client
sqs_client = boto3.client('sqs', region_name='ap-south-1')

# Get the queue. This returns an SQS.Queue instance
inbox_queue = sqs.get_queue_by_name(QueueName='2017ht13042-queue1')

# Get the outbox queue. This returns an SQS.Queue instance
outbox_queue = sqs.get_queue_by_name(QueueName='2017ht13042-queue3')

while True:
    # Enable long polling on an existing SQS queue
    input_queue_response = sqs_client.receive_message(
        QueueUrl=inbox_queue.url,
        AttributeNames=[
            'SentTimestamp'
        ],
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        WaitTimeSeconds=10
    )
    
    if(input_queue_response.get('Messages')):
        inbox_body_json = json.loads(input_queue_response['Messages'][0]['Body'])
        print(inbox_body_json)
        
        input_object_location = inbox_body_json['object_location']
        process = inbox_body_json['process']
        message_identifier = inbox_body_json['message_identifier']
        
        object = s3.Object(bucket_name, input_object_location)
        valueFromObj = object.get()['Body'].read().decode('utf-8')
        valueList = valueFromObj.split(',')
        # currently only supporting sum
        if(process.lower() == 'sum'):
            final_value = sum(decimal.Decimal(i) for i in valueList)
            print(final_value)
        
        # location of the output object on s3 bucket
        output_object_location = 'arn:aws:s3:::2017ht13042-bucket2'
        output_object = s3.Object(bucket_name, output_object_location)
        
        # populate the body of the s3 object with the comma separated values string
        output_object.put(Body=str(final_value))
        
        output_dict = {}
        output_dict['object_location'] = output_object_location
        output_dict['process'] = process
        output_dict['message_identifier'] = message_identifier
        
        #convert the dict to json
        final_message_str = json.dumps(output_dict)
        
        response = outbox_queue.send_message(MessageBody=str(final_message_str), MessageAttributes={
            'Author': {
                'StringValue': 'Sumit',
                'DataType': 'String'
                }
            })
        
        sqs_client.delete_message(QueueUrl=inbox_queue.url,ReceiptHandle=input_queue_response['Messages'][0]['ReceiptHandle'])
