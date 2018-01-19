from pytube import YouTube
from pprint import pprint
from os import listdir
from os.path import isfile, join
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydub import AudioSegment
import nlib

#Key needed for auth
DEVELOPER_KEY = "AIzaSyCmiH0Q_kSvtillNLqdrtuYCWIiuZ6-A3k"
#Google api service name
API_SERVICE_NAME = 'youtube'
#API version #
API_VERSION = 'v3'
#Location where the videos will be downloaded
VID_FOLDER = "/home/ubuntu/Scraper/Youtube"


def search_by_keyword(service, **kwargs):
	return service.search().list(**kwargs).execute()
	
#Find the next youtube URLS to download that are not already in VID_FOLDER
#@service: The api connection
#@searchTerm: The search term used by the API to constrict the seach(like a normal youtube search)
def find_URLS(service, searchTerm, max):
	urlList = []
	skip = 0;
	reachMax = max
	while(len(urlList) < max):
		results = search_by_keyword(service,
		part='snippet',
		safeSearch="moderate",
		maxResults=max+skip,
		q=searchTerm,
		type='video')["items"]
	
		results = results[skip:]
		ownedVidIds = [f[f.rfind("+")+1:f.rfind(".")] for f in listdir(VID_FOLDER) if isfile(join(VID_FOLDER, f))]
		
		for item in results:
			if( (item["id"]["videoId"] in ownedVidIds) or (item["id"]["videoId"] in (i[0] for i in urlList))):
				skip+=1
			else:
				if(reachMax>0):
					reachMax-=1
					urlList.append(("https://www.youtube.com/watch?v=" + item["id"]["videoId"], searchTerm.replace(" ","_") + "+" + item["id"]["videoId"]))
	

	return urlList
if __name__ == '__main__':

	#Build the API service connection
	service = build(API_SERVICE_NAME, API_VERSION, developerKey=DEVELOPER_KEY)
	cata = ["vevo country music","vevo rock music"]
	
	s3 = nlib.simpleS3.simpleS3('itwasntme')
	
	for i in cata:
	
		scrapeURLs = find_URLS(service,i, 1)
		
		for scrapeURL in scrapeURLs:
			yt = YouTube(scrapeURL[0])
			print("Downloading " + scrapeURL[1])
			yt.streams.filter(subtype='mp4',only_audio=True).first().download(VID_FOLDER,scrapeURL[1])
			print("Complete")
		

		newVidFileNames = [f[:f.rfind(".")] for f in listdir(VID_FOLDER) if isfile(join(VID_FOLDER, f))]
		
		for f in newVidFileNames:
			AudioSegment.from_file(VID_FOLDER+"/"+f +".mp4","mp4").set_frame_rate(44100).set_sample_width(1).set_channels(2).export(f+".wav", bitrate='8', format="wav")
			s3.upload(f+".wav")
		