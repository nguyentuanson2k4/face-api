# Face Recognition Service - Docker & Render Deployment Guide

## Local Docker Testing

### 1. Build Docker image locally
```bash
docker build -t face-recognition-service:latest .
```

### 2. Run container locally
```bash
docker run -p 8000:8000 face-recognition-service:latest
```

Check health endpoint:
```bash
curl http://localhost:8000/health
```

## Deployment to Render

### Prerequisites
- Render account (https://render.com)
- GitHub repository with your code
- Docker Hub account (optional, for private images)

### Step 1: Push Code to GitHub

```bash
git init
git add .
git commit -m "Initial commit with Docker support"
git remote add origin https://github.com/your-username/face-recognition-service.git
git push -u origin main
```

### Step 2: Connect GitHub to Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → Select **"Web Service"**
3. Click **"Connect a repository"**
4. Authorize GitHub and select your repository
5. Fill in the service details:
   - **Name**: `face-recognition-service`
   - **Region**: Singapore (or your preferred region)
   - **Branch**: `main`
   - **Runtime**: Docker
   - **Port**: `8000`

### Step 3: Configure Environment (if needed)

In Render dashboard, go to **Environment** tab and add:
```
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### Step 4: Deploy

1. Click **"Create Web Service"**
2. Render will automatically build and deploy your Docker container
3. Your service will be available at: `https://face-recognition-service.onrender.com`

### Step 5: Verify Deployment

```bash
curl https://face-recognition-service.onrender.com/health
```

Expected response:
```json
{"status": "ok", "model": "insightface/buffalo_l"}
```

## Important Notes

### ⚠️ Model Download on First Run
The InsightFace model (~281MB) will download on first startup. This may take 1-2 minutes during the initial deployment. Check logs in Render dashboard to monitor progress.

### 🔒 Private Resources
If using private resources, configure credentials via environment variables:
- For image storage: Add authentication tokens/credentials
- For database: Pass connection strings as environment variables

### 📊 Monitoring
In Render dashboard, you can:
- View deployment logs
- Monitor CPU/Memory usage
- Set up alerts
- Configure auto-scaling (on paid plans)

### ⚙️ Advanced Configuration

**Use render.yaml for Infrastructure as Code:**
1. Commit `render.yaml` to repository
2. Go to Render Dashboard → **Settings** → **Connect Blueprint**
3. Select your repository and branch
4. Render will automatically deploy using the `render.yaml` config

### 🔄 Auto-Redeploy on Push

By default, Render deploys automatically when you push to your selected branch.

### 📝 Common Issues & Solutions

**Issue**: "Model download timeout"
- **Solution**: Free tier may timeout. Consider switching to a paid plan or pre-building the image.

**Issue**: "Port 8000 not accessible"
- **Solution**: Ensure `EXPOSE 8000` in Dockerfile and Render port is set to 8000

**Issue**: "Memory exceeded"
- **Solution**: The insightface models are memory-intensive. Use at least a 512MB RAM plan on Render.

## Testing Endpoints

Once deployed, test with:

```bash
BASE_URL=https://face-recognition-service.onrender.com

# Health check
curl $BASE_URL/health

# Upload and detect faces
curl -X POST -F "file=@photo.jpg" $BASE_URL/detect-faces

# Extract embedding from single face
curl -X POST -F "file=@face.jpg" $BASE_URL/extract-embedding
```

## Alternative: Docker Compose (Local Development)

```yaml
version: '3.8'
services:
  face-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ~/.insightface:/root/.insightface  # Cache models
```

Run with:
```bash
docker-compose up
```

## Costs on Render

- **Free tier**: Limited, auto-pauses after 15 mins inactivity
- **Standard**: $7-12/month for persistent deployment
- **Pro**: Recommended for production with model caching

For this face recognition service, **Standard or Pro tier recommended** due to model size and memory requirements.
