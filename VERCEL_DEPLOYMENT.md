# Vercel Deployment Guide for PicoTuri-EditJudge

This guide explains how to deploy PicoTuri-EditJudge to Vercel with both frontend and backend.

## Prerequisites

- Vercel account (https://vercel.com)
- GitHub account with the repository pushed
- Node.js 18+ installed locally
- Python 3.12+ installed locally

## Project Structure

```
PicoTuri-EditJudge/
├── api/                    # Python backend (Vercel Functions)
│   ├── index.py           # Main Flask app
│   └── requirements.txt    # Python dependencies
├── dashboard/             # Next.js frontend
│   ├── package.json
│   ├── next.config.js
│   └── src/
├── vercel.json           # Vercel configuration
└── README.md
```

## Step 1: Prepare the Project

### 1.1 Update dashboard/package.json

Ensure your `package.json` has the correct build scripts:

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.5.0",
    "tailwindcss": "^3.3.0"
  }
}
```

### 1.2 Create .vercelignore

```bash
cat > .vercelignore << 'EOF'
.git
.gitignore
README.md
node_modules
.next
.venv
__pycache__
*.pyc
.pytest_cache
tests/
notebooks/
EOF
```

### 1.3 Create .env.local for development

```bash
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:5000
EOF
```

## Step 2: Deploy to Vercel

### Option A: Using Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
vercel --prod
```

### Option B: Using GitHub Integration

1. Go to https://vercel.com/new
2. Import your GitHub repository
3. Select "Other" as the framework
4. Configure:
   - **Root Directory**: Leave empty
   - **Build Command**: `npm run build`
   - **Output Directory**: `dashboard/.next`
5. Add Environment Variables:
   - `PYTHON_VERSION`: `3.12`
   - `NODE_VERSION`: `18.x`
   - `NEXT_PUBLIC_API_URL`: Your Vercel deployment URL
6. Click "Deploy"

## Step 3: Configure Environment Variables

In Vercel Dashboard:

1. Go to Settings → Environment Variables
2. Add the following variables:

```
PYTHON_VERSION=3.12
NODE_VERSION=18.x
PYTHONUNBUFFERED=1
PYTHONPATH=/var/task
NEXT_PUBLIC_API_URL=https://your-deployment.vercel.app
```

## Step 4: API Endpoints

Once deployed, your API will be available at:

```
https://your-deployment.vercel.app/api/
```

### Available Endpoints

#### Health Check
```bash
GET /api/health
```

#### Score Image Quality
```bash
POST /api/score-quality
Content-Type: application/json

{
  "original_image": "base64_encoded_image",
  "edited_image": "base64_encoded_image",
  "instruction": "enhance the lighting"
}
```

#### Embed Text
```bash
POST /api/embed-text
Content-Type: application/json

{
  "texts": ["text1", "text2", ...]
}
```

#### Model Status
```bash
GET /api/models/status
```

#### API Info
```bash
GET /api/info
```

## Step 5: Frontend Integration

Update your frontend to use the API:

```javascript
// pages/api/quality-score.js or similar
const API_URL = process.env.NEXT_PUBLIC_API_URL;

export async function scoreQuality(originalImage, editedImage, instruction) {
  const response = await fetch(`${API_URL}/api/score-quality`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      original_image: originalImage,
      edited_image: editedImage,
      instruction: instruction,
    }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to score quality');
  }
  
  return response.json();
}
```

## Step 6: Monitoring and Logs

### View Logs
```bash
vercel logs
```

### View Deployments
```bash
vercel list
```

### View Project Settings
```bash
vercel env list
```

## Troubleshooting

### Python Dependencies Not Installing
- Ensure `api/requirements.txt` exists
- Check Python version compatibility
- Verify all packages are available on PyPI

### Frontend Build Fails
- Check `dashboard/package.json` exists
- Verify Node.js version is 18+
- Ensure all dependencies are listed

### API Timeout
- Increase function timeout in `vercel.json`
- Optimize model loading (lazy loading)
- Consider using smaller models

### CORS Issues
- Ensure `flask-cors` is installed
- Check `vercel.json` routes configuration
- Verify environment variables are set

## Performance Optimization

### 1. Model Caching
Models are cached in memory after first load to improve performance.

### 2. Lazy Loading
Models are loaded only when first requested.

### 3. Batch Processing
Support batch requests to reduce overhead:

```bash
POST /api/embed-text
{
  "texts": ["text1", "text2", "text3", ...]
}
```

## Cost Optimization

- **Free Tier**: 100 GB-hours/month
- **Pro Tier**: $20/month for additional resources
- **Enterprise**: Custom pricing

### Tips
- Use serverless functions efficiently
- Optimize model sizes
- Cache results when possible
- Monitor usage in Vercel Dashboard

## Continuous Deployment

Once connected to GitHub, deployments happen automatically on:
- Push to main branch
- Pull request creation
- Manual redeploy from Vercel Dashboard

## Custom Domain

1. Go to Vercel Dashboard → Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed
4. Wait for DNS propagation (up to 48 hours)

## Security

- Store sensitive keys in environment variables
- Use HTTPS (automatic with Vercel)
- Implement API authentication if needed
- Monitor logs for suspicious activity

## Support

- Vercel Docs: https://vercel.com/docs
- GitHub Issues: https://github.com/mangeshraut712/PicoTuri-EditJudge/issues
- Discord Community: [Join our Discord]

## Next Steps

1. ✅ Deploy to Vercel
2. ✅ Test API endpoints
3. ✅ Connect frontend to backend
4. ✅ Set up monitoring
5. ✅ Configure custom domain
6. ✅ Enable analytics

Happy deploying! 🚀
