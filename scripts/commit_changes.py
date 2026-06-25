import subprocess
import os

repo_dir = r"c:\Users\PMIHIR\Desktop\FaceID\FaceID_Recognition"

def run_git(args):
    result = subprocess.run(['git'] + args, cwd=repo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running git {' '.join(args)}:\n{result.stderr}")
    else:
        print(f"Success: git {' '.join(args)}")

# 1. Neumorphic UI
run_git(['add', 'src/web', 'src/web_interface'])
run_git(['commit', '-m', 'feat(ui): Implement Neumorphic UI design system'])

# 2. Training & Evaluation
run_git(['add', 'src/training', 'src/evaluation'])
run_git(['commit', '-m', 'feat(ml): Add ArcFace fine-tuning pipeline and biometric evaluation framework'])

# 3. Liveness & Quality
run_git(['add', 'src/liveness', 'src/quality', 'notebooks'])
run_git(['commit', '-m', 'feat(ml): Add liveness detection, face quality assessment, and experiment notebooks'])

# 4. Docs
run_git(['add', 'docs', '*.md'])
run_git(['commit', '-m', 'docs: Consolidate markdown reports and update guides'])

# 5. Architecture Reorganization & Core Fixes
run_git(['add', '.'])
run_git(['commit', '-m', 'refactor: Reorganize project architecture, implement multi-face support, and fix core security vulnerabilities'])
