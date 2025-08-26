#  Table Cell Extraction 

This project extracts **individual table cells** from a scanned or digital image of a table   
It uses **image processing techniques** such as thresholding, morphological operations, and line detection to detect the grid structure and then **slices out each cell as a separate image**.

---

## Features
- Detects **horizontal and vertical lines** in a table image.
- Identifies **cell boundaries** based on intersections of detected lines.
- Extracts each cell into a separate image (`cell_rXX_cYY.png`).
- Creates a **visualization image** with bounding boxes and row-column labels.
- Saves all extracted cells in the `extracted_cells/` folder.

---

##  Dependencies
Install the following packages:
```bash
pip install opencv-python numpy
