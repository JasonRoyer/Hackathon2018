#!/usr/bin/python3

import boto3, configparser, os

class simpleS3(object):
	def __init__(self, bucket):
		self.bucket_name = str(bucket)
		config = configparser.ConfigParser()
		dirname, filename = os.path.split(os.path.abspath(__file__))
		config.read(os.path.join(dirname, 'conf.ini'))
		self.client = boto3.resource(
			's3',
			aws_access_key_id = config.get('AWS', 'ACCESS'),
			aws_secret_access_key = config.get('AWS', 'SECRET')
		)
		self.bucket = self.client.Bucket(self.bucket_name)

	def dumpFileList(self, p=False):
		junk = self.bucket.objects.all()
		if p:
			for obj in junk:
				print(obj.key)
		return [obj.key for obj in junk]

	def upload(self, file, path=""):
		self.client.Object(self.bucket_name, os.path.join(path, os.path.basename(file))).upload_file(file)

	def download(self, filename, path):
		if os.path.exists(path):
			self.client.Object(self.bucket_name, filename).download_file(os.path.join(path, filename))
		else:
			raise IOError('Path not found! Try again dumbass.')

	def testOutput(self):
		print("AYY LMAO")

if __name__ == "__main__":
	tst = simpleS3('itwasntme')
	tst.upload('test2.txt')
	tst.dumpFileList(True)
	tst.download('test2.txt', 'testes/')

