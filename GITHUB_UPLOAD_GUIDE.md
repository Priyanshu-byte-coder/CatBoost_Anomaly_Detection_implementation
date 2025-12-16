# GitHub Upload Guide

This guide will help you upload this repository to GitHub.

## Prerequisites

1. **Git installed** on your system
   - Download from: https://git-scm.com/downloads
   - Verify installation: `git --version`

2. **GitHub account**
   - Create one at: https://github.com/signup

## Step-by-Step Upload Instructions

### Option 1: Using GitHub Desktop (Easiest)

1. **Download GitHub Desktop**
   - Visit: https://desktop.github.com/
   - Install and sign in with your GitHub account

2. **Add Repository**
   - Click "File" → "Add local repository"
   - Browse to: `C:\Users\Priyanshu\OneDrive\Desktop\papers\apscon\implementation_anuja`
   - Click "Add Repository"

3. **Create Repository on GitHub**
   - Click "Publish repository" button
   - Choose a repository name (e.g., `robotic-arm-fault-detection`)
   - Add description: "Self-supervised learning for robotic arm fault detection using CASPER dataset"
   - Uncheck "Keep this code private" if you want it public
   - Click "Publish repository"

4. **Done!** Your code is now on GitHub.

---

### Option 2: Using Command Line (Git Bash/PowerShell)

#### Step 1: Initialize Git Repository

Open PowerShell or Git Bash in your project folder and run:

```bash
cd "C:\Users\Priyanshu\OneDrive\Desktop\papers\apscon\implementation_anuja"
git init
```

#### Step 2: Configure Git (First time only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

#### Step 3: Add Files to Git

```bash
git add .
git status  # Verify files are staged (data/ should NOT appear due to .gitignore)
```

**Important:** Verify that the `data/` folder is NOT listed. If it appears, check your `.gitignore` file.

#### Step 4: Create First Commit

```bash
git commit -m "Initial commit: Robotic arm fault detection implementation"
```

#### Step 5: Create Repository on GitHub

1. Go to: https://github.com/new
2. Repository name: `robotic-arm-fault-detection` (or your choice)
3. Description: "Self-supervised learning for robotic arm fault detection using CASPER dataset"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

#### Step 6: Link Local Repository to GitHub

Copy the commands from GitHub (they'll look like this):

```bash
git remote add origin https://github.com/YOUR_USERNAME/robotic-arm-fault-detection.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

#### Step 7: Push Your Code

```bash
git push -u origin main
```

Enter your GitHub credentials when prompted.

---

### Option 3: Using GitHub Web Interface (Upload Files)

**Note:** This method has file size and count limitations. Use Option 1 or 2 for better results.

1. Go to: https://github.com/new
2. Create a new repository
3. Click "uploading an existing file"
4. Drag and drop files (excluding the `data/` folder)
5. Commit changes

---

## Verify Upload

After uploading, verify:

1. ✅ `data/` folder is **NOT** uploaded (too large)
2. ✅ `README.md` displays correctly with dataset download instructions
3. ✅ All result files in `output2/` and `multi_run_analysis_output/` are present
4. ✅ `requirements.txt` and `.gitignore` are present
5. ✅ LaTeX tables in `latex_tables/` are uploaded

## Repository Size Check

Your repository should be approximately:
- **Without data:** ~15-20 MB (manageable)
- **With data:** ~5.6 GB (too large for GitHub)

GitHub has a 100 MB file size limit and recommends repositories under 1 GB.

## Troubleshooting

### Problem: "File too large" error

**Solution:** Ensure `.gitignore` includes `data/` and `*.csv`. Remove large files:

```bash
git rm --cached -r data/
git commit -m "Remove large data files"
git push
```

### Problem: Authentication failed

**Solution:** Use a Personal Access Token instead of password:
1. Go to: https://github.com/settings/tokens
2. Generate new token (classic)
3. Select scopes: `repo`
4. Use token as password when pushing

### Problem: Files not uploading

**Solution:** Check `.gitignore` isn't excluding needed files:

```bash
git status --ignored
```

## Next Steps After Upload

1. **Add topics** to your repository (Settings → Topics):
   - `machine-learning`
   - `robotics`
   - `fault-detection`
   - `self-supervised-learning`
   - `pytorch`
   - `anomaly-detection`

2. **Enable GitHub Pages** (optional) for documentation

3. **Add a license** (Settings → Add license)
   - Recommended: MIT License or Apache 2.0

4. **Create releases** for major versions

5. **Add badges** to README (optional):
   - Python version
   - License
   - Dataset link

## Useful Git Commands

```bash
# Check status
git status

# View commit history
git log --oneline

# Update repository after changes
git add .
git commit -m "Description of changes"
git push

# Clone repository elsewhere
git clone https://github.com/YOUR_USERNAME/robotic-arm-fault-detection.git
```

## Support

If you encounter issues:
- GitHub Docs: https://docs.github.com/
- Git Documentation: https://git-scm.com/doc
- GitHub Community: https://github.community/
