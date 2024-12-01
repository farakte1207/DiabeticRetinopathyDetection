from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define Flask app
app = Flask(__name__)

# Load the trained model
class DR_CNN(torch.nn.Module):
    def __init__(self):
        super(DR_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 512)
        self.fc2 = torch.nn.Linear(512, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DR_CNN()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# Define class names
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    if file:
        # Load the image
        image = Image.open(file).convert('RGB')
        
        # Apply transformations
        image = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
        
        return render_template('result.html', prediction=predicted_class)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
