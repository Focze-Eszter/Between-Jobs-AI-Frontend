# Author: Focze Eszther =======================Frontend
# Author: Bitingo Josaphat JB =================Backend


# Importing All The Necessary Libraries All in One.
# ================================================
from flask import Flask, request, render_template, session, redirect, url_for, send_file, send_from_directory
from flask_session import Session
from werkzeug.utils import secure_filename
from weasyprint import HTML, CSS
from uuid import uuid4
import subprocess
import cv2
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
import math
import sys
import requests
import os
from pdf2docx import Converter
import utils.Filters as filter
import utils.Operations as op
import whisper
from config import BG_API_KEY, OpenAI_API

app = Flask(__name__)


# =================================================================================================================
# SENTITIVE DATA TO BE WRAPPED IN THE ENVIRONMENT VARIABLES
# Configuring For the Image Editor Feature
app.secret_key = "A0Zr98j"
SESSION_TYPE = "filesystem"
app.config.from_object(__name__)
Session(app)


# Configuration for the Documents to Pdf Feature
UPLOAD_FOLDER = 'static/Docx2PDF/uploads'
CONVERTED_FOLDER = 'static/Docx2PDF/converted_files'
ALLOWED_EXTENSIONS = {'docx', 'doc', 'pdf', 'html'}
app.config['DOC_UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOC_CONVERTED_FOLDER'] = CONVERTED_FOLDER

# Set the upload folder and allowed extensions
app.config['ENHANCER_UPLOAD_FOLDER'] = 'static/enhancerstatic/uploads'
app.config['ENHANCER_ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Create the uploads and ouputs folder if it does not exist
os.makedirs(app.config['ENHANCER_UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/enhancerstatic/outputs', exist_ok=True)


# =======================================================================================================
# Route to display the Website Landing page
# ------------------------------------------
@app.route('/between-jobs-ai')
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


# =======================================================================================================
# Routing the BackgrondRemover Feature
# -------------------------------------
@app.route('/background-remover-feature')
def BackgroundFeature():
    return render_template('BGindex.html')

# Route to handle the file upload For the Background Remover Feature
@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']
    # Check if the file is empty
    if file.filename == '':
        return 'No file selected'

    # Send the file to Remove.bg API for processing
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': file},
        data={'size': 'auto'},
        headers={'X-Api-Key': BG_API_KEY}
    )

    # Check if the response was successful
    if response.status_code == requests.codes.ok:
        # Save the processed image to disk
        with open('static/BackgroundRemoverImage/processed.png', 'wb') as f:
            f.write(response.content)

        # Render the result page with the processed image
        return render_template('BGresult.html')
    else:
        return f'Error: {response.status_code} {response.text}'


# Basic Image Editing Features
@app.route("/basic-image-editor-feature")
def ImageEditorFeature():
    if "image" in session:
        show_image = "data:image/png;base64,"+session["image"]
        if "output" in session:
            output = session["output"]
            op_name = session["op_name"]
            return render_template("ImageEditorIndex.html", orginal=show_image, output=output, op_name=op_name)
        return render_template("ImageEditorIndex.html", orginal=show_image)
    return render_template("ImageEditorIndex.html", orginal="no Image")

# creating the setImage function
@app.route("/set-image", methods=['POST', 'GET'])
def setImage():
    if request.method == "GET":
        return "<h1> it is a post method</h1>"
    image_mat = request.form["image"]
    session["image"] = image_mat
    if "output" in session:
        session.pop("output")
    if "op_name" in session:
        session.pop("op_name")

    return redirect(url_for("ImageEditorFeature"))

@app.route("/save", methods=['GET'])
def saveChanges():
    if "output" in session:
        session["image"] = session["output"].replace(
            "data:image/png;base64,", "")
        session.pop("output")
    if "op_name" in session:
        session.pop("op_name")

    return redirect(url_for("ImageEditorFeature"))


# defining the grayish Filter
@app.route("/grayish-art-filter", methods=['GET'])
def grayish():
    session["output"] = "data:image/png;base64," + \
        filter.grayish(session["image"])
    session["op_name"] = "GRAYISH"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/pixel-art-filter", methods=['GET'])
def pixelArt():
    session["output"] = "data:image/png;base64," + \
        filter.pixelArt(session["image"])
    session["op_name"] = "PIXXEL"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/pop-art-filter", methods=['GET'])
def popArt():
    session["output"] = "data:image/png;base64," + \
        filter.popArt(session["image"])
    session["op_name"] = "BONCUK"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/inverse-art-filter", methods=['GET'])
def inverse():
    session["output"] = "data:image/png;base64," + \
        op.inverse(session["image"])
    session["op_name"] = "INVERSE"
    return redirect(url_for("ImageEditorFeature"))



# =============================================================================

@app.route("/emboss-art-filter", methods=['GET'])
def emboss():
    session["output"] = "data:image/png;base64," + \
        filter.emboss(session["image"])
    session["op_name"] = "EMBOSS"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/mirror-art-filter", methods=['GET'])
def mirror():
    session["output"] = "data:image/png;base64," + \
        op.mirror(session["image"])
    session["op_name"] = "MIRROR"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/rotate-art-filter", methods=['GET'])
def rotate():
    angle = int(request.headers["angle"])
    session["output"] = "data:image/png;base64," + \
        op.rotate(session["image"], angle)

    session["op_name"] = "ROTATE"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/lumos-art-filter", methods=['GET'])
def lumos():
    lumen = int(request.headers["lumen"])
    session["output"] = "data:image/png;base64," + \
        op.lumos(session["image"], lumen)

    session["op_name"] = "BRIGHT"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/contrast-art-filter", methods=['GET'])
def contrast():
    contrast = int(request.headers["contrast"])
    session["output"] = "data:image/png;base64," + \
        op.contrast(session["image"], contrast)

    session["op_name"] = "CONTRAST"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/crop-select-art-filter", methods=['GET'])
def crop_select():

    session["op_name"] = "CROP_SELECT"
    session["output"] = "select for crop"

    return redirect(url_for("ImageEditorFeature"))


@app.route("/crop-art-filter", methods=['GET'])
def crop():

    p = request.headers["points"]
    w = int(request.headers["width"])
    p = p.split(',')
    points = []
    i = 0

    while i <= len(p)-2:
        points.append([math.floor(float(p[i])), math.floor(float(p[i+1]))])
        i += 2

    session["output"] = "data:image/png;base64," + \
        op.crop(session["image"], points, w)
    session["op_name"] = "CROP"

    return redirect(url_for("ImageEditorFeature"))


@app.route("/flip-art-filter", methods=['GET'])
def flip():
    try:
        _hor = request.headers["hor"]
        if(_hor == "false"):
            hor = False
        else:
            hor = True
        _ver = request.headers["ver"]

        if(_ver == "false"):
            ver = False
        else:
            ver = True

    except:
        hor = False
        ver = False

    session["output"] = "data:image/png;base64," + \
        op.flip(session["image"], hor, ver)

    session["op_name"] = "FLIPPER"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/oldtv-art-filter", methods=['GET'])
def oldtv():
    session["output"] = "data:image/png;base64," + \
        filter.oldtv(session["image"])
    session["op_name"] = "90's TV"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/sketch-art-filter", methods=['GET'])
def sketch():
    session["output"] = "data:image/png;base64," + \
        filter.sketch(session["image"])
    session["op_name"] = "SKETCH"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/splash-art-filter", methods=['GET'])
def splash():
    session["output"] = "data:image/png;base64," + \
        filter.splash(session["image"])
    session["op_name"] = "SPLASH"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/sepya-art-filter", methods=['GET'])
def sepya():
    session["output"] = "data:image/png;base64," + \
        filter.sepya(session["image"])
    session["op_name"] = "SEPIA"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/cartoon-art-filter", methods=['GET'])
def cartoon():
    session["output"] = "data:image/png;base64," + \
        filter.cartoon(session["image"])
    session["op_name"] = "CARTOON"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/oily-art-filter", methods=['GET'])
def oily():
    session["output"] = "data:image/png;base64," + \
        filter.oily(session["image"])
    session["op_name"] = "OILY"
    return redirect(url_for("ImageEditorFeature"))

# ===============================================================
@app.route("/autocon-art-filter", methods=['GET'])
def autocon():
    session["output"] = "data:image/png;base64," + \
        op.histogramEqualizer(session["image"])
    session["op_name"] = "EQUALIZED"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/abstractify-art-filter", methods=['GET'])
def abstractify():
    session["output"] = "data:image/png;base64," + \
        filter.abstractify(session["image"])
    session["op_name"] = "NOTIONAL"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/balmy-art-filter", methods=['GET'])
def balmy():
    session["output"] = "data:image/png;base64," + \
        filter.warm(session["image"])
    session["op_name"] = "BALMY"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/cold-art-filter", methods=['GET'])
def cold():
    session["output"] = "data:image/png;base64," + \
        filter.cold(session["image"])
    session["op_name"] = "FROSTBITE"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/lines-art-filter", methods=['GET'])
def lines():
    session["output"] = "data:image/png;base64," + \
        filter.lines(session["image"])
    session["op_name"] = "LINES"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/blush-art-filter", methods=['GET'])
def blush():
    session["output"] = "data:image/png;base64," + \
        filter.blush(session["image"])
    session["op_name"] = "BLUSH"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/glass-art-filter", methods=['GET'])
def glass():
    session["output"] = "data:image/png;base64," + \
        filter.glass(session["image"])
    session["op_name"] = "GLASS"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/xpro-art-filter", methods=['GET'])
def xpro():
    session["output"] = "data:image/png;base64," + \
        filter.xpro(session["image"])
    session["op_name"] = "XPRO"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/daylight-art-filter", methods=['GET'])
def daylight():
    session["output"] = "data:image/png;base64," + \
        filter.daylight(session["image"])
    session["op_name"] = "DAYLIGHT"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/moon-art-filter", methods=['GET'])
def moon():
    session["output"] = "data:image/png;base64," + \
        filter.moon(session["image"])
    session["op_name"] = "MOON"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/blueish-art-filter", methods=['GET'])
def blueish():
    session["output"] = "data:image/png;base64," + \
        filter.blueish(session["image"])
    session["op_name"] = "BLUEISH"
    return redirect(url_for("ImageEditorFeature"))


@app.route("/clear")
def clear():
    [session.pop(key) for key in list(session.keys())]
    return redirect(url_for("ImageEditorFeature"))

# ================================================================================================================
# PDF 2 Docx Converter Feature

@app.route('/pdf-to-document-feature')
def pdfToWordFeature():
    return render_template('PdftoWordIndex.html')

@app.route('/pdf-conversion', methods=['POST'])
def pdfConvert():
    if 'pdf_file' not in request.files:
        return 'No PDF file uploaded'

    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return 'No selected file'

    pdf_filename = pdf_file.filename
    docx_filename = pdf_filename.replace('.pdf', '.docx')

    # Save uploaded PDF file temporarily
    pdf_file_path = os.path.join('tmp', pdf_filename)
    pdf_file.save(pdf_file_path)

    # Convert PDF to DOCX
    docx_file_path = os.path.join('tmp', docx_filename)
    cv = Converter(pdf_file_path)
    cv.convert(docx_file_path, start=0, end=None)
    cv.close()

    # Provide download link for converted DOCX file
    docx_file_name = docx_filename
    docx_file_url = url_for('pdfDownload', filename=docx_filename)

    return render_template('PdfToWordSuccess.html', docx_file_name=docx_file_name, docx_file_url=docx_file_url)

@app.route('/pdf-download/<filename>')
def pdfDownload(filename):
    return send_file(os.path.join('tmp', filename), as_attachment=True)

# ================================================================================================================
# Implementing the Ms Word docx To PDF Converter

def docx_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# creating a function to handle html to pdf conversion
def convert_html_to_pdf(file_path, file_name):
    try:
        random_string = str(uuid4())[:10]
        HTML(file_path).write_pdf(os.path.join('static/Docx2PDF/converted_files',
                                               f'{file_name}_{random_string}.pdf'),
                                  stylesheets=[CSS(string='body { font-family: serif !important;margin-left:-80px; }')])
    except Exception as e:
        print("Exception occurred during HTML to PDF conversion:", e)

#  Creating a function to handle doc files to pdf conversion
def convert_doc_to_pdf(file_path):
    try:
        subprocess.call(['soffice', '--headless', '--convert-to', 'pdf',
                         '--outdir', 'static/Docx2PDF/converted_files', file_path])
    except Exception as e:
        print("Exception occurred during DOC to PDF conversion:", e)


@app.route('/word-to-pdf-conversion', methods=['GET'])
def docx2pdfindex():
    try:
        return render_template('Docx2PdfIndex.html')
    except Exception as e:
        print("Exception occurred during rendering template:", e)
        return "Something Went Wrong !!"


@app.route('/docs-download/<filename>', methods=['GET'])
def doc_download_file(filename):
    return send_from_directory('static/Docx2PDF/converted_files', filename, as_attachment=True)


@app.route('/docs-conversion-success/<filename>', methods=['GET'])
def docx_conversion_success(filename):
    return render_template('Docx2PdfSuccess.html', filename=filename)


@app.route('/docx-upload', methods=['POST'])
def docx_file_converter():
    if request.method == "POST":
        try:
            files = request.files.getlist('file')
            if len(files) > 0:
                for data in files:
                    if docx_allowed_file(data.filename):
                        filename = secure_filename(data.filename)
                        file_path = os.path.join(app.config['DOC_UPLOAD_FOLDER'], filename)
                        converted_filename = filename.rsplit('.', 1)[0]
                        converted_file_path = os.path.join(app.config['DOC_CONVERTED_FOLDER'], converted_filename)

                        data.save(file_path)

                        extension = filename.rsplit('.', 1)[1].lower()
                        if extension == 'pdf':
                            pdf_file_path = os.path.join(app.config['DOC_CONVERTED_FOLDER'], filename)
                            data.save(pdf_file_path)
                        elif extension == 'html':
                            convert_html_to_pdf(file_path, converted_filename)
                        elif extension in {'docx', 'doc'}:
                            convert_doc_to_pdf(file_path)

                        return redirect(url_for('docx_conversion_success', filename=converted_filename))
                    else:
                        return "Format Not Allowed !!"
            else:
                return "Failed"
        except Exception as e:
            print("Exception occurred during file conversion:", e)
            return "Something Went Wrong !!"
    else:
        return "Method Not Allowed !"




# ==========================================================================================================================
# Creating The AI Image Enhancer Feature

# Download weights if not available
model_weights = {
    'realesr-general-x4v3.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
    'GFPGANv1.2.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth',
    'GFPGANv1.3.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    'GFPGANv1.4.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    'RestoreFormer.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth',
    'CodeFormer.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth'
}

for weight_file, weight_url in model_weights.items():
    if not os.path.exists(weight_file):
        os.system(f"wget {weight_url} -P .")

# Create the background enhancer with RealESRGAN
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)



def enhancer_allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ENHANCER_ALLOWED_EXTENSIONS']


def enhance_image(image_path, version, scale):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    try:
        if scale > 4:
            scale = 4  # avoid too large scale value

        extension = os.path.splitext(os.path.basename(image_path))[1]
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:  # for gray inputs
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h > 3500 or w > 3500:
            print('Too large size')
            return None, None

        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        if version == 'v1.2':
            face_enhancer = GFPGANer(
                model_path='GFPGANv1.2.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'v1.3':
            face_enhancer = GFPGANer(
                model_path='GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'v1.4':
            face_enhancer = GFPGANer(
                model_path='GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'RestoreFormer':
            face_enhancer = GFPGANer(
                model_path='RestoreFormer.pth', upscale=2, arch='RestoreFormer', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'CodeFormer':
            face_enhancer = GFPGANer(
                model_path='CodeFormer.pth', upscale=2, arch='CodeFormer', channel_multiplier=2, bg_upsampler=upsampler)

        try:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        except RuntimeError as error:
            print('Error:', error)

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print('Wrong scale input.', error)

        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        else:
            extension = 'jpg'
        output_path = f'static/enhancerstatic/outputs/{filename}.{extension}'
        cv2.imwrite(output_path, output)

        return output_path, image_path

    except Exception as error:
        print('Global exception:', error)
        return None, None



@app.route('/ai-powered-image-enhancer', methods=['GET', 'POST'])
def enhancerIndex():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('enhancerIndexFile.html', error='No file part')

        file = request.files['file']
        version = request.form['version']
        scale = int(request.form['scale'])

        # Validate file
        if file.filename == '':
            return render_template('enhancerIndexFile.html', error='No file selected')
        if not enhancer_allowed_file(file.filename):
            return render_template('enhancerIndexFile.html', error='Invalid file type')

        # Save the file
        file_path = os.path.join(app.config['ENHANCER_UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Perform image enhancement
        output_path, input_path = enhance_image(file_path, version, scale)

        # Render the result page with image paths
        return render_template('enhancerResultFile.html', before_image_path=input_path, after_image_path=output_path)

    return render_template('enhancerIndexFile.html')


# ADDING THE AI AUDIO LYRICS GENERATOR FEATURE

@app.route('/ai-audio-lyrics-generator')
def speechToTextGenerator():
    return render_template('Audio-lyrics-generator-index.html')

@app.route('/audio-content-recognizer', methods=['POST'])
def audioContentRecognize():
    # Check if audio file was uploaded
    if 'file' not in request.files:
        return render_template('Audio-lyrics-generator-index.html', error='No audio file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('Audio-lyrics-generator-index.html', error='No audio file selected')

    # Save the uploaded audio file
    audio_path = 'audio.wav'
    file.save(audio_path)

    # Perform speech recognition
    transcription = transcribe_audio(audio_path)

    text = transcription['text']

    # Delete the temporary audio file
    os.remove(audio_path)

    return render_template('Audio-lyrics-generator-index.html', transcription=text)

def transcribe_audio(audio_path):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Perform speech recognition using the Whisper model
    result = model.transcribe(audio_path)

    return result



if __name__ == '__main__':
    # Create the images directory if it doesn't exist
    if not os.path.exists('static/BackgroundRemoverImage'):
        os.makedirs('static/BackgroundRemoverImage')

    app.run(debug=True)
