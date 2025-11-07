# Back4app Deployment Guide for Gold Price Predictor

This guide will help you deploy your Gold Price Predictor backend to Back4app for free, without requiring any credit card information.

## Prerequisites

- GitHub account
- Your code pushed to a GitHub repository
- Back4app account (free, no credit card required)

## Step-by-Step Deployment

### 1. Sign Up for Back4app

1. Go to [back4app.com](https://www.back4app.com)
2. Click "Sign Up" or "Get Started for Free"
3. Sign up using your GitHub account (recommended)
4. **No credit card information required** for the free tier

### 2. Prepare Your Repository

Your repository should have:
- The `backend/` folder with your FastAPI application
- The `Dockerfile` we created in the backend folder
- Your data files in `backend/data/raw/` (if you have historical data)

### 3. Create New Container App

1. Log in to your Back4app dashboard
2. Click "NEW APP" or "Create New App"
3. Choose "Containers" (not "Back4app")
4. Select "Deploy from GitHub"
5. Connect your GitHub account if prompted
6. Choose your repository
7. Configure the deployment:
   - **Branch:** main (or your preferred branch)
   - **Dockerfile Path:** `backend/Dockerfile`
   - **App Name:** `gold-predictor-api`
   - **Port:** 8000

### 4. Environment Variables (Optional)

If needed, you can add environment variables in the Back4app dashboard:
- Go to your app settings
- Find "Environment Variables"
- Add any required variables

### 5. Deploy

1. Click "Deploy" or "Create App"
2. Back4app will:
   - Build your Docker image
   - Run the preprocessing and training (as specified in Dockerfile)
   - Start your FastAPI application
3. Wait for the deployment to complete (usually 2-5 minutes)

### 6. Get Your API URL

Once deployed, Back4app will provide you with a public URL like:
`https://gold-predictor-api-xxxxx.back4app.io`

### 7. Test Your Deployment

Test your endpoints:
```bash
# Health check
curl https://gold-predictor-api-xxxxx.back4app.io/

# Get latest prediction
curl https://gold-predictor-api-xxxxx.back4app.io/predict/latest
```

## Free Tier Limits

- **RAM:** 256MB
- **Monthly Transfer:** 100GB
- **Active Hours:** 600 hours/month
- **No credit card required**

## Troubleshooting

### Build Issues
- Check the deployment logs in Back4app dashboard
- Ensure your Dockerfile is in the correct location
- Verify all dependencies are listed in `requirements.txt`

### Data Issues
- The preprocessing and training run during build time
- If you have large datasets, consider uploading them separately
- Check that your data files are committed to the repository

### Memory Issues
- Free tier has 256MB RAM limit
- If your app exceeds this, consider optimizing your model size
- You may need to use a simpler model for the free tier

## Next Steps

After successful backend deployment:
1. Deploy your React frontend to Vercel (also free, no card required)
2. Set the `REACT_APP_API_BASE_URL` environment variable to your Back4app URL
3. Your full application will be live!

## Support

- Back4app Documentation: https://www.back4app.com/docs
- Community Forum: https://help.back4app.com/
- Your deployment logs are available in the Back4app dashboard

---

**Remember:** Both Back4app and Vercel offer free tiers without credit card requirements, making this a completely free deployment solution!