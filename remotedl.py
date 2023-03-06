import requests
import shutil 
from flask import Flask, request
import re

app = Flask(__name__)

@app.route('/remote')
def remote_dl():
    url = request.args.get('url')
    if url is None:
        return "missing URL Parameter"
    
    fileNameFromURIPattern: str = "/^.*\/(.*)\.(.*)\?.*$/"
    foundFile = re.search(fileNameFromURIPattern, url)
    file_name = foundFile.group()
    if file_name is None:
        return "Filename missing in URL"
    
    res = requests.get(url, stream = True)

    if res.status_code == 200:
        with open(file_name,'wb') as f:
            shutil.copyfileobj(res.raw, f)
        print('Image sucessfully Downloaded: ',file_name)
    else:
        print('Image Couldn\'t be retrieved')
    

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)