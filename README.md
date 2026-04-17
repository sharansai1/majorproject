# Pancreatic Tumor Detection App

A Streamlit web application for detecting and localizing pancreatic tumors in CT scan slices using a Hybrid GAT U-Net model.

## 🚀 Quick Deploy to Streamlit Cloud

### Step 1: Push to GitHub

Create a new GitHub repository and push this folder:

```bash
# Initialize git repo
git init
git add .
git commit -m "Initial commit"

# Add your GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/pancreas-tumor-detection.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Click "New App"
3. Select your GitHub repository
4. Select `app.py` as the main file
5. Click "Deploy" 🎉

You'll get a public URL like: `https://your-app-name.streamlit.app`

## 📁 Project Structure

```
majorprojectcopy/
├── app.py                 # Streamlit application (entry point)
├── model.py               # HybridGATUNet model architecture
├── best_gat_unet.pth      # Trained model weights (~31 MB)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🧠 Model Details

- **Architecture**: Hybrid GAT U-Net with Attention Gates
- **Input**: 256×256 grayscale CT scan slices
- **Output**: Tumor probability map (segmentation mask)
- **Dataset**: Medical Segmentation Decathlon Task07 Pancreas

## 📦 Requirements

All dependencies are listed in `requirements.txt`:
- streamlit
- torch
- torchvision
- numpy
- pillow
- opencv-python
- albumentations
- matplotlib
- groq (for AI diagnosis feature)

## 🖼️ Usage

1. Upload a CT scan slice image (PNG/JPG)
2. Adjust the detection threshold if needed (default: 0.5)
3. View the tumor detection results with:
   - Input CT scan
   - Probability heatmap
   - Segmentation overlay with bounding box
4. Click "Get AI Diagnosis" for an expert analysis by Llama AI (optional)
5. Download the analysis report

## 🔐 Optional: Enable AI Diagnosis

The app includes an optional AI diagnosis feature powered by Groq's Llama vision model:

1. Get a free API key at [console.groq.com/keys](https://console.groq.com/keys)
2. In your Streamlit Cloud dashboard:
   - Go to **Settings** → **Secrets**
   - Add: `GROQ_API_KEY = "your-api-key-here"`
3. Restart the app — the AI diagnosis button will now work!

**Note:** The app works perfectly without the AI feature. This is just an optional enhancement.

## ⚠️ Limitations

- This is a demo application for educational purposes
- The model is trained on a specific dataset and may not generalize to all CT scanners
- Always consult a qualified radiologist for medical diagnosis

## 📝 Notes

- Model file size: ~31 MB (well within Streamlit Cloud's limits)
- Cold start time: ~10-30 seconds (due to model loading)
- For GPU acceleration, the app automatically uses CUDA if available
- **No `.env` file needed** — use Streamlit Secrets for API keys instead
- Never commit API keys to GitHub!
