Fairness Income Dataset  

Overview  
This repository contains a small-scale dataset for educational use in fairness and bias analysis.  
It is inspired by the UCI Adult dataset and includes demographic attributes and income labels for binary classification tasks.  

Note  
This dataset is synthetic and not representative of any real population. It is intended for teaching and demonstration purposes only.  

Files  
- fairness_income.csv — the dataset  
- example.py — sample code for fairness metric calculation  
- README.md — project documentation  

Dataset Description  
- Rows: approximately 200  
- Columns: age, workclass, educationnum, hoursperweek, capitalgain, capital_loss, sex, race, income  

How to Run  

1. Clone or download the project folder.  
2. (Optional) Create a virtual environment:  
   - python -m venv venv  
   - macOS/Linux: source venv/bin/activate  
   - Windows: venv\Scripts\activate  
3. Install dependencies:  
   pip install pandas scikit-learn matplotlib seaborn numpy  
4. Ensure the CSV file (fairness_income.csv) is in the same folder as example.py.  
5. Run the analysis:  
   python example.py  

Expected Output  
- A logistic regression model will be trained  
- Accuracy, ROC AUC, and confusion matrix will be printed  
- Group-wise fairness metrics (by sex and race) will be computed  
- Bar plots comparing performance across groups will be displayed  