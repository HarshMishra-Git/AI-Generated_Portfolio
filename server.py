import os
import json
import base64
import re
from io import BytesIO
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
from PIL import Image
from flask_mail import Mail, Message

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini models
generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}

image_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config
)

text_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config
)

app = Flask(__name__, static_folder=".")

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'noreply.portfoliobot@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME', 'noreply.portfoliobot@gmail.com')
mail = Mail(app)

# Serve the static files
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# API for image classification
@app.route('/api/classify-image', methods=['POST'])
def classify_image():
    try:
        data = request.json
        image_data = data.get('imageData')
        
        # Extract the base64 data from the data URL
        base64_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Classify with Gemini
        prompt = """
        You are an expert image classifier. Please identify what's drawn in this image.
        Focus on simple shapes or objects that might be hand-drawn, like: 
        circle, square, triangle, star, heart, arrow, line, smiley face, house, tree, flower, 
        cat, dog, bird, sun, moon, cloud.
        
        Format your response as a JSON object like this:
        {
            "predictions": [
                {"label": "shape_name", "confidence": 85},
                {"label": "alternative_shape", "confidence": 10},
                {"label": "another_possibility", "confidence": 5}
            ]
        }
        
        Ensure confidence values add up to 100 and represent percentages. Include 3-4 possibilities.
        Respond ONLY with the JSON object.
        """
        
        response = image_model.generate_content([prompt, image])
        result_text = response.text
        
        # Extract JSON from the response (handling potential markdown code blocks)
        json_pattern = r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```|(\{.*\})'
        match = re.search(json_pattern, result_text, re.DOTALL)
        
        if match:
            json_str = match.group(1) or match.group(2) or match.group(3)
            result = json.loads(json_str)
        else:
            # If no JSON found, create a fallback response
            result = {
                "predictions": [
                    {"label": "Undefined Shape", "confidence": 70},
                    {"label": "Could be a Drawing", "confidence": 20},
                    {"label": "Not Recognized", "confidence": 10}
                ]
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API for sentiment analysis
@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        text = data.get('text')
        
        prompt = f"""
        Analyze the sentiment of the following text: "{text}"
        
        Format your response as a JSON object with the following structure:
        {{
            "label": "Positive/Negative/Neutral",
            "emoji": "😃/😞/😐",
            "color": "#55FF55/#FF5555/#AAAAAA",
            "confidence": 85,
            "distribution": {{
                "positive": 70,
                "neutral": 20,
                "negative": 10
            }},
            "keyPhrases": ["important phrase 1", "important phrase 2", "important phrase 3"]
        }}
        
        The distribution values should add up to 100.
        The key phrases should be 2-4 important words or phrases from the text.
        Respond ONLY with the JSON object.
        """
        
        response = text_model.generate_content(prompt)
        result_text = response.text
        
        # Extract JSON from the response (handling potential markdown code blocks)
        json_pattern = r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```|(\{.*\})'
        match = re.search(json_pattern, result_text, re.DOTALL)
        
        if match:
            json_str = match.group(1) or match.group(2) or match.group(3)
            result = json.loads(json_str)
        else:
            # If no JSON found, create a fallback response
            result = {
                "label": "Neutral",
                "emoji": "😐",
                "color": "#AAAAAA",
                "confidence": 50,
                "distribution": {
                    "positive": 30,
                    "neutral": 40,
                    "negative": 30
                },
                "keyPhrases": ["Unable to analyze sentiment"]
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API for sending contact emails
@app.route('/api/send-email', methods=['POST'])
def send_email():
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message_content = data.get('message')
        
        if not all([name, email, subject, message_content]):
            return jsonify({"error": "All fields are required"}), 400
        
        # Create message to send to Harsh
        recipient_email = "harsh.mishra2022@glbajajgroup.org"
        
        msg = Message(
            subject=f"Portfolio Contact: {subject}",
            recipients=[recipient_email],
            html=f"""
            <h3>New message from your portfolio website</h3>
            <p><strong>From:</strong> {name} ({email})</p>
            <p><strong>Subject:</strong> {subject}</p>
            <p><strong>Message:</strong></p>
            <p>{message_content}</p>
            """
        )
        
        # Add reply-to header so Harsh can reply directly to the sender
        msg.extra_headers = {"Reply-To": email}
        
        # Also send a confirmation email to the sender
        confirmation_msg = Message(
            subject="Thank you for contacting Harsh Mishra",
            recipients=[email],
            html=f"""
            <h3>Thank you for your message!</h3>
            <p>Hello {name},</p>
            <p>I've received your message and will get back to you as soon as possible.</p>
            <p>Here's a copy of your message:</p>
            <p><strong>Subject:</strong> {subject}</p>
            <p><strong>Message:</strong></p>
            <p>{message_content}</p>
            <p>Best regards,<br>Harsh Mishra<br>AI/ML Engineer</p>
            """
        )
        
        # Send both emails
        mail.send(msg)
        mail.send(confirmation_msg)
        
        return jsonify({"success": True, "message": "Your message has been sent!"})
        
    except Exception as e:
        print(f"Email error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)