import base64, hashlib, struct, math

def imageToHash(filepath):
	with open(filepath, 'rb') as image:
		encoded = base64.b64encode(image.read())
	out = hashlib.sha512(encoded).hexdigest()
	bry = bytearray(out, 'ascii')
	arr = []
	for i in range(0,32):
		j = i*4
		num = struct.unpack('f', bry[j+0:j+4])[0]
		pow10 = math.pow(10, math.floor(math.log10(float(num))))
		arr.append(num/pow10)
	rise = 2.0
	run = float(max(arr)-min(arr))
	for i in arr:
		print((rise/run)*i - 1)


if __name__ == '__main__':
	print(imageToHash('./papa.jpg'))
