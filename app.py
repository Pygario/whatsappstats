import os
import logging
from flask import Flask, redirect, request, flash, render_template
from werkzeug.utils import secure_filename
import wastats

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = ['txt']
LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.update(SECRET_KEY=os.urandom(24))
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
app.logger.addHandler(stream_handler)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', category='error')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', category='error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            plot = wastats.process(f'{app.config["UPLOAD_FOLDER"]}/{filename}')
            os.remove(f'{app.config["UPLOAD_FOLDER"]}/{filename}')
            return render_template('results.html', name=filename.split('.')[0], source=plot)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()