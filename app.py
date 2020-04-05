import os

from flask import Flask, redirect, request, flash, render_template
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

import whatsappstats

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = ['txt']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.update(SECRET_KEY=os.urandom(24))
bootstrap = Bootstrap(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            plot = whatsappstats.process(f'./uploads/{filename}')
            os.remove(f'./uploads/{filename}')
            return render_template('results.html', name=filename.split('.')[0], source=plot)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
