from flask import Flask, request, render_template
import requests
import os

app = Flask(__name__)

# Enter your Remove.bg API key here
API_KEY = 'MWHQ3Eav9VpWBorUz2CXw7LC'
# OpenAI_API = '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'

# =======================================================================================================
# Route to display the Website Landing page
# ------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


# =======================================================================================================
# Routing the BackgrondRemover Feature
# -------------------------------------
@app.route('/BackgroundFeature')
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
        headers={'X-Api-Key': API_KEY}
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

if __name__ == '__main__':
    # Create the images directory if it doesn't exist
    if not os.path.exists('static/BackgroundRemoverImage'):
        os.makedirs('static/BackgroundRemoverImage')

    app.run(debug=True)
