from flask import Flask, render_template, Response, request, send_from_directory
app = Flask(__name__)
 
@app.route("/")
def root():
	return render_template('index.html')

@app.route("/Capture")
def capture():
	return render_template('webcap.html')

@app.route("/dist/<path:filename>")
def getAudio(filename):	
	return send_from_directory("/home/ubuntu/server/static/jpeg_camera",filename)

@app.errorhandler(404)
def page_not_found(e):
    return "Page was not found D:"
 
if __name__ == "__main__":
    app.run()