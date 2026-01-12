# Project Submission for Overleaf

## Files Included

1. **cs577_project.tex** - Main LaTeX report file
2. **neurips_2024.sty** - NeurIPS 2024 style file
3. **result/** - Directory for generated plots (PDFs will be generated when notebook is run)

## Setup Instructions

1. **Generate Plots First**: 
   - Run the notebook `VIT_main_fashionmnist.ipynb` to generate:
     - `result/training_curves.pdf`
     - `result/attention_visualization.pdf`
   - These PDFs are required for the LaTeX to compile

2. **Upload to Overleaf**:
   - Upload `cs577_project.tex` and `neurips_2024.sty` to your Overleaf project
   - Upload the `result/` folder with the generated PDFs
   - Make sure the folder structure is maintained: `result/training_curves.pdf` and `result/attention_visualization.pdf`

3. **Compile**:
   - In Overleaf, set the main document to `cs577_project.tex`
   - Compile the document (should work on first try if all files are present)

## Note

The LaTeX file is configured for anonymous submission (no `[final]` flag). The report includes all required sections:
1. Background
2. Model Architecture
3. Model Training
4. MWE-CPU
5. Discussion

