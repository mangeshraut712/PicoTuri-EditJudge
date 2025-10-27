# Vercel Quick Start - 5 Minutes to Deploy

## 1. Prerequisites (1 min)

- [ ] GitHub account with repo pushed
- [ ] Vercel account (free at vercel.com)
- [ ] Node.js 18+ installed

## 2. Connect GitHub to Vercel (2 min)

1. Go to https://vercel.com/new
2. Click "Import Git Repository"
3. Select your GitHub repository
4. Click "Import"

## 3. Configure Build Settings (1 min)

When prompted, use these settings:

| Setting | Value |
|---------|-------|
| Framework | Other |
| Root Directory | `.` (leave empty) |
| Build Command | `npm run build` |
| Output Directory | `dashboard/.next` |
| Install Command | `npm install` |

## 4. Add Environment Variables (1 min)

Click "Environment Variables" and add:

```
PYTHON_VERSION=3.12
NODE_VERSION=18.x
NEXT_PUBLIC_API_URL=https://<your-deployment>.vercel.app
```

## 5. Deploy! (Click Deploy)

That's it! Vercel will:
- ✅ Install dependencies
- ✅ Build frontend
- ✅ Deploy backend API
- ✅ Set up serverless functions
- ✅ Configure routing

## Your URLs

After deployment, you'll have:

- **Frontend**: `https://<your-deployment>.vercel.app`
- **API**: `https://<your-deployment>.vercel.app/api`

## Test Your Deployment

```bash
# Health check
curl https://<your-deployment>.vercel.app/api/health

# API info
curl https://<your-deployment>.vercel.app/api/info
```

## Next Steps

1. Visit your frontend URL
2. Test image quality scoring
3. Monitor logs in Vercel Dashboard
4. Add custom domain (optional)
5. Set up GitHub auto-deployments (automatic)

## Troubleshooting

**Build fails?**
- Check `dashboard/package.json` exists
- Ensure `api/requirements.txt` exists
- View logs in Vercel Dashboard

**API not working?**
- Check environment variables
- View function logs
- Verify Python version is 3.12+

**Slow performance?**
- Models are cached after first load
- Subsequent requests are faster
- Check Vercel Analytics

## Support

- 📖 Full guide: See `VERCEL_DEPLOYMENT.md`
- 🐛 Issues: GitHub Issues
- 💬 Questions: Vercel Community

---

**Deployed successfully? Share your URL!** 🎉
