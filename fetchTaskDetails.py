import boto3
import argparse
from boto3.dynamodb.conditions import Key

def query_task(task_id, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb')

    table = dynamodb.Table('murfUser')
    response = table.query(
        KeyConditionExpression=Key('uid').eq('TrainTask_'+task_id)
    )
    return response['Items'][0]


if __name__ == '__main__':
  # Instantiate the parser
    parser = argparse.ArgumentParser(description='Pass task id to save parameters')
    # Optional argument
    parser.add_argument('--task_id', type=str, help='Task id to train on')
    parser.add_argument('--path', type=str, help='Properties file path to dump task details')
    args = parser.parse_args()
    print("Argument values:")
    print(args.task_id)
    task = query_task(task_id)
    f = open(args.path, "a")
    f.write(','.join(task.speakerIds))
    f.close()
    
    
