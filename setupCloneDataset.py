import boto3
import argparse
from boto3.dynamodb.conditions import Key
import tarfile
import os
from shutil import copyfile
import shutil

def query_task(task_id, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-2')

    table = dynamodb.Table('murfUser')
    response = table.query(
        KeyConditionExpression=Key('uid').eq('TrainTask_'+task_id)
    )
    return response['Items'][0]


if __name__ == '__main__':
  # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Pass task id to save parameters')
    # Optional argument
    parser.add_argument('--task_id', type=str, help='Task id to train on')
    parser.add_argument('--libri_path', type=str,
                        help='Folder to dump speaker folders')
    parser.add_argument('--dataset_path', type=str,
                        help='Path to core speaker folders')
    parser.add_argument('--for_vocoder', type=str,
                        help='setup for Vocoder')

    args = parser.parse_args()
    print("Argument values:")
    print(args.task_id)
    print(args.libri_path)
    print(args.dataset_path)
    print(args.for_vocoder)
    task = query_task(args.task_id)
    s3 = boto3.resource('s3')
    
    if os.path.exists(args.libri_path) and os.path.isdir(args.libri_path):
        shutil.rmtree(args.libri_path)
    os.mkdir(args.libri_path)

    for i in task['speakerIds']:
        tarPath = args.libri_path + "/" + str(i) + ".tar"
        s3Key = "dataset/speakers/" + str(i) + ".tar"
        print("Downloading " + s3Key)
        s3.Bucket('murf-models-dev').download_file(s3Key, tarPath)
        tar = tarfile.open(tarPath)
        tar.extractall(path=args.libri_path)
        tar.close()
        os.remove(tarPath)
        print("Extracted"+str(i))

    if args.for_vocoder == 'true':
        for i in os.listdir(args.dataset_path):
            if i.isnumeric():
                speakerDestFolderPath = os.path.join(args.libri_path, i)
                if os.path.exists(speakerDestFolderPath) and os.path.isdir(speakerDestFolderPath):
                    shutil.rmtree(speakerDestFolderPath)
                os.mkdir(speakerDestFolderPath)
                speakerSrcFolderPath = os.path.join(args.dataset_path, i)
                count=0
                for fileName in os.listdir(speakerSrcFolderPath):
                    if ".wav" in fileName:
                        wavsrc = os.path.join(speakerSrcFolderPath, fileName)
                        txtsrc = os.path.join(speakerSrcFolderPath, fileName.replace(".wav", ".txt"))
                        if os.path.isfile(txtsrc):
                            copyfile(wavsrc, os.path.join(speakerDestFolderPath, fileName))
                            copyfile(txtsrc, os.path.join(speakerDestFolderPath, fileName.replace(".wav", ".txt")))
                            count += 1
                            if count > 5:
                                break

