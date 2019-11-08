from flask import Flask, request, render_template, Response, make_response
from wtforms import Form, TextAreaField, validators, IntegerField
import os
import numpy as np
import io
from utility_funcs import load_model, process_image, predict, model_lime, get_lime_model,resize_transform,preprocess_transform,lime_model_predict
from torchvision import models
from lime import lime_image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from PIL import Image
import glob

import datetime
import random

app= Flask(__name__)

gender_list=['Male','Female']
os.environ['TORCH_HOME'] ='./'


class ReviewForm(Form):
    Age = IntegerField('(i.e., 30)', validators= [validators.InputRequired()])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('index.html', form =form, gender_list=gender_list)

@app.route('/test' , methods=['GET', 'POST'])
def test():
    form = ReviewForm(request.form)
    if request.method == 'POST':
        file = request.files['file']

        image = file.read()
        age_var=form.Age.data
        gender=request.form.get('Gender')

        if gender == 'Male':
            gender_var=1
        else:
            gender_var=0

        data=[file, age_var, gender_var]
        return render_template('test.html',
                            data=data)

@app.route('/result', methods=['GET','POST'])
def classifer():
    form = ReviewForm(request.form)
    if request.method =='GET':
        return render_template('index.html')
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        # clear the instance directory just in case
        filelist=os.listdir('img_storage')
        for f in filelist:
            os.remove(os.path.join('img_storage', f))

        file = request.files['file']
        file.save(os.path.join('img_storage', file.filename)) # need to save before file.read()
        fname=os.path.join('img_storage', file.filename)

        age_var=form.Age.data
        gender=request.form.get('Gender')

        if gender == 'Male':
            gender_var=1
        else:
            gender_var=0

        model=load_model()
        top_p, top_class=predict(fname, age_var, gender_var, model, topk=1)
        acronym_dict={'AK': 'Actinic keratosis',
        'BCC': 'Basal cell carcinoma',
        'BKL': 'Benign keratosis',
        'DF': 'Dermatofibroma',
        'MEL': 'Melanoma',
        'NV': 'Nevus(mole)',
        'SCC': 'Squamous cell carcinoma',
        'VASC': 'Vascular lesion',
        'UNK': 'unknown'}
        lesion_name=acronym_dict[top_class[0]]

        return render_template('result.html', lesion=lesion_name, top_p=top_p)


@app.route("/lime.png")
def create_figure():
    fig = Figure(figsize=[10, 4.8])

    axis1 = fig.add_subplot(121)
    axis2=fig.add_subplot(122)

    img_dir= 'img_storage'
    list_of_files = glob.glob(img_dir+"/*.jpg")
    fname = max(list_of_files, key=os.path.getctime)
    image=Image.open(fname)

    img_transf = resize_transform()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(img_transf(image)),
                                         lime_model_predict, # classification function
                                         top_labels=1,
                                         hide_color=0,
                                         num_samples=100)
    out_img, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5,
                                            hide_rest=False)
    axis1.imshow(out_img)
    axis2.imshow(img_transf(image))
    axis1.title.set_text("LIME output showing features for (in green) and against (in red) \n the prediced class")
    axis2.title.set_text("Original image")

    canvas = FigureCanvas(fig)
    png_output = io.BytesIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    os.remove(fname)
    return response


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
