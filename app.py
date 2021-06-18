from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import analysis
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
Bootstrap(app)

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['avi', 'mp4', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello():
    uploaded = False
    return render_template('index.html', uploaded = uploaded)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    analysed = False
    loading = True
    uploadDirectory = "uploads/"
    if request.method == 'POST':
        f = request.files['file']
        f.save("uploads/uploaded_video.mp4")
        analysed = analysis.newConvertToFrames('uploads/uploaded_video.mp4')
        if(analysed):
            loading = False
            return render_template('index.html', analysis_done = analysed, loading = loading)
        #   f.save(uploadDirectory+secure_filename(f.filename))
        return render_template('index.html', analysis_done = analysed, loading = loading)
    analysed = analysis.convertToFrames('uploads/uploaded_video.mp4')
    if(analysed):
        loading = False
        return render_template('index.html', analysis_done = analysed, loading = loading)


@app.route('/search', methods = ['POST'])
def search_df():
      query = request.form['search']
      processed_text = query.upper()
      print(query)
      images = analysis.searchDataFrame(query=query)
      return render_template('index.html', uploaded = True, loading = False, images = images, analysis_done = True, showImages = True)


if __name__ == '__main__':
    app.run()
