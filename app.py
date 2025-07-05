from flask import Flask, render_template_string, request
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load optimized model bundle
data = load('student_performance_model_optimized.joblib')
model = data['model']
columns = data['columns']
categorical_cols = data['categorical_cols']
label_encoders = data['label_encoders']

# Enhanced HTML template with improved CSS and layout
TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow-x: hidden;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 80%, rgba(120, 219, 255, 0.3) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }

        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            padding: 40px 20px;
            text-align: center;
            position: relative;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            pointer-events: none;
        }

        .header h1 {
            font-size: 3.2em;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 12px;
            text-shadow: 0 2px 20px rgba(0, 0, 0, 0.2);
            position: relative;
            z-index: 1;
        }

        .header .subtitle {
            font-size: 1.2em;
            font-weight: 400;
            opacity: 0.9;
            letter-spacing: 0.3px;
            position: relative;
            z-index: 1;
        }

        .container {
            max-width: 520px;
            margin: 50px auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            padding: 45px 40px;
            border-radius: 24px;
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.15),
                0 0 0 1px rgba(255, 255, 255, 0.2);
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.4) 0%, rgba(255, 255, 255, 0.1) 100%);
            border-radius: 24px;
            pointer-events: none;
        }

        .container > * {
            position: relative;
            z-index: 1;
        }

        h2 {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 35px;
            letter-spacing: -0.01em;
        }

        form {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            position: relative;
        }

        label {
            font-weight: 500;
            margin-bottom: 8px;
            color: #374151;
            font-size: 0.95em;
            letter-spacing: 0.3px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .info-icon {
            width: 16px;
            height: 16px;
            background: #6b7280;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 11px;
            font-weight: 600;
            cursor: help;
            position: relative;
        }

        .info-icon::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: #1f2937;
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 1000;
            pointer-events: none;
            font-weight: 400;
        }

        .info-icon:hover::after {
            opacity: 1;
            visibility: visible;
        }

        input, select {
            padding: 16px 18px;
            border-radius: 12px;
            border: 2px solid #e5e7eb;
            font-size: 1em;
            font-family: inherit;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: #374151;
        }

        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 
                0 0 0 4px rgba(102, 126, 234, 0.1),
                0 4px 12px rgba(102, 126, 234, 0.15);
            transform: translateY(-2px);
        }

        input::placeholder {
            color: #9ca3af;
            font-weight: 400;
        }

        select {
            cursor: pointer;
            appearance: none;
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3e%3c/svg%3e");
            background-position: right 12px center;
            background-repeat: no-repeat;
            background-size: 16px;
            padding-right: 45px;
        }

        /* Input validation styles */
        .form-group.error input,
        .form-group.error select {
            border-color: #ef4444;
            background: rgba(239, 68, 68, 0.05);
        }

        .form-group.success input,
        .form-group.success select {
            border-color: #10b981;
            background: rgba(16, 185, 129, 0.05);
        }

        .error-message {
            color: #ef4444;
            font-size: 0.85em;
            margin-top: 6px;
            display: none;
            animation: slideDown 0.3s ease;
        }

        .form-group.error .error-message {
            display: block;
        }

        .submit-btn {
            margin-top: 20px;
            padding: 18px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            border: none;
            border-radius: 12px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            font-family: inherit;
            letter-spacing: 0.3px;
            box-shadow: 
                0 8px 24px rgba(102, 126, 234, 0.3),
                0 0 0 1px rgba(255, 255, 255, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }

        .submit-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }

        .submit-btn:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 
                0 12px 32px rgba(102, 126, 234, 0.4),
                0 0 0 1px rgba(255, 255, 255, 0.3);
        }

        .submit-btn:hover:not(:disabled)::before {
            left: 100%;
        }

        .submit-btn:active:not(:disabled) {
            transform: translateY(-1px);
        }

        .result {
            margin-top: 35px;
            text-align: center;
            font-size: 1.3em;
            font-weight: 600;
            padding: 24px 20px;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border: 2px solid rgba(102, 126, 234, 0.2);
            backdrop-filter: blur(10px);
            color: #374151;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.1);
            position: relative;
            animation: slideUp 0.5s ease-out;
        }

        .result::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.3) 0%, rgba(255, 255, 255, 0.1) 100%);
            border-radius: 14px;
            pointer-events: none;
        }

        .result > * {
            position: relative;
            z-index: 1;
        }

        .result b {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(0, 0, 0, 0.1);
            border-radius: 3px;
            margin: 20px 0;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 3px;
        }

        .form-summary {
            background: rgba(102, 126, 234, 0.05);
            border: 1px solid rgba(102, 126, 234, 0.1);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 24px;
            display: none;
        }

        .form-summary.show {
            display: block;
            animation: slideDown 0.3s ease;
        }

        .form-summary h3 {
            color: #667eea;
            margin-bottom: 8px;
            font-size: 1.1em;
        }

        .summary-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
            font-size: 0.9em;
            color: #6b7280;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Enhanced mobile responsiveness */
        @media (max-width: 640px) {
            .header {
                padding: 30px 15px;
            }
            
            .header h1 {
                font-size: 2.2em;
            }
            
            .header .subtitle {
                font-size: 1em;
            }
            
            .container {
                margin: 30px 15px;
                padding: 30px 25px;
                border-radius: 20px;
            }
            
            h2 {
                font-size: 1.5em;
                margin-bottom: 25px;
            }
            
            form {
                gap: 20px;
            }
            
            input, select {
                padding: 14px 16px;
            }
            
            .submit-btn {
                padding: 16px 20px;
                font-size: 1em;
            }
            
            .result {
                font-size: 1.1em;
                padding: 20px 16px;
            }
        }

        @media (max-width: 480px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            .container {
                margin: 20px 10px;
                padding: 25px 20px;
            }
        }

        /* Loading animation for form submission */
        .submit-btn.loading {
            pointer-events: none;
            opacity: 0.8;
        }

        .submit-btn.loading::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            margin: auto;
            border: 2px solid transparent;
            border-top-color: #ffffff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Student Performance Predictor</h1>
        <div class="subtitle">Predict if a student will pass based on their profile</div>
    </div>
    
    <div class="container">
        <h2>Enter Student Details</h2>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill"></div>
        </div>
        
        <form method="post" id="predictionForm">
            <div class="form-group">
                <label for="study-hours">
                    Study Hours per Week
                    <span class="info-icon" data-tooltip="Average hours spent studying per week">?</span>
                </label>
                <input type="number" step="0.1" min="0" max="168" name="Study Hours per Week" id="study-hours" required placeholder="e.g. 12.5">
                <div class="error-message">Please enter a valid number of study hours (0-168)</div>
            </div>
            
            <div class="form-group">
                <label for="attendance">
                    Attendance Rate
                    <span class="info-icon" data-tooltip="Percentage of classes attended (0-100%)">?</span>
                </label>
                <input type="number" step="0.1" min="0" max="100" name="Attendance Rate" id="attendance" required placeholder="e.g. 85.0">
                <div class="error-message">Please enter a valid attendance rate (0-100%)</div>
            </div>
            
            <div class="form-group">
                <label for="grades">
                    Previous Grades
                    <span class="info-icon" data-tooltip="Average of previous academic grades (0-100)">?</span>
                </label>
                <input type="number" step="0.1" min="0" max="100" name="Previous Grades" id="grades" required placeholder="e.g. 75.0">
                <div class="error-message">Please enter valid grades (0-100)</div>
            </div>
            
            <div class="form-group">
                <label for="extracurricular">
                    Participation in Extracurricular Activities
                    <span class="info-icon" data-tooltip="Active participation in sports, clubs, or other activities">?</span>
                </label>
                <select name="Participation in Extracurricular Activities" id="extracurricular" required>
                    <option value="" disabled selected>Select participation level</option>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                </select>
                <div class="error-message">Please select participation level</div>
            </div>
            
            <div class="form-group">
                <label for="parent-education">
                    Parent Education Level
                    <span class="info-icon" data-tooltip="Highest education level of parents/guardians">?</span>
                </label>
                <select name="Parent Education Level" id="parent-education" required>
                    <option value="" disabled selected>Select education level</option>
                    <option value="High School">High School</option>
                    <option value="Associate">Associate</option>
                    <option value="Bachelor">Bachelor</option>
                    <option value="Master">Master</option>
                    <option value="Doctorate">Doctorate</option>
                </select>
                <div class="error-message">Please select parent education level</div>
            </div>
            
            <div class="form-summary" id="formSummary">
                <h3>Summary</h3>
                <div id="summaryContent"></div>
            </div>
            
            <button type="submit" class="submit-btn" id="submitBtn" disabled>
                <span>Predict Performance</span>
            </button>
        </form>
        
        <div class="result" id="result" style="display: none;">
            <span>Prediction: <b id="predictionText"></b></span>
        </div>
    </div>

    <script>
        const form = document.getElementById('predictionForm');
        const progressFill = document.getElementById('progressFill');
        const submitBtn = document.getElementById('submitBtn');
        const formSummary = document.getElementById('formSummary');
        const summaryContent = document.getElementById('summaryContent');
        const result = document.getElementById('result');
        const predictionText = document.getElementById('predictionText');
        
        const fields = form.querySelectorAll('input[required], select[required]');
        let filledFields = 0;

        // Validation functions
        function validateStudyHours(value) {
            return value >= 0 && value <= 168;
        }

        function validateAttendance(value) {
            return value >= 0 && value <= 100;
        }

        function validateGrades(value) {
            return value >= 0 && value <= 100;
        }

        function validateField(field) {
            const group = field.closest('.form-group');
            const value = parseFloat(field.value);
            let isValid = true;

            if (field.type === 'number' && field.value !== '') {
                if (field.id === 'study-hours') {
                    isValid = validateStudyHours(value);
                } else if (field.id === 'attendance') {
                    isValid = validateAttendance(value);
                } else if (field.id === 'grades') {
                    isValid = validateGrades(value);
                }
            }

            if (field.type === 'number' && field.value === '') {
                isValid = false;
            }

            if (field.tagName === 'SELECT' && field.value === '') {
                isValid = false;
            }

            // Update visual feedback
            group.classList.remove('error', 'success');
            if (field.value !== '') {
                group.classList.add(isValid ? 'success' : 'error');
            }

            return isValid;
        }

        function updateProgress() {
            filledFields = 0;
            let allValid = true;

            fields.forEach(field => {
                if (field.value !== '' && validateField(field)) {
                    filledFields++;
                } else if (field.value !== '') {
                    allValid = false;
                }
            });

            const progress = (filledFields / fields.length) * 100;
            progressFill.style.width = progress + '%';
            
            // Enable submit button only if all fields are filled and valid
            submitBtn.disabled = filledFields !== fields.length || !allValid;
            
            // Show summary when form is complete
            if (filledFields === fields.length && allValid) {
                updateSummary();
                formSummary.classList.add('show');
            } else {
                formSummary.classList.remove('show');
            }
        }

        function updateSummary() {
            const studyHours = document.getElementById('study-hours').value;
            const attendance = document.getElementById('attendance').value;
            const grades = document.getElementById('grades').value;
            const extracurricular = document.getElementById('extracurricular').value;
            const parentEducation = document.getElementById('parent-education').value;

            summaryContent.innerHTML = `
                <div class="summary-item">
                    <span>Study Hours/Week:</span>
                    <span>${studyHours}</span>
                </div>
                <div class="summary-item">
                    <span>Attendance Rate:</span>
                    <span>${attendance}%</span>
                </div>
                <div class="summary-item">
                    <span>Previous Grades:</span>
                    <span>${grades}</span>
                </div>
                <div class="summary-item">
                    <span>Extracurricular:</span>
                    <span>${extracurricular}</span>
                </div>
                <div class="summary-item">
                    <span>Parent Education:</span>
                    <span>${parentEducation}</span>
                </div>
            `;
        }

        // Add event listeners
        fields.forEach(field => {
            field.addEventListener('input', updateProgress);
            field.addEventListener('blur', () => validateField(field));
            field.addEventListener('change', updateProgress);
        });

        // Form submission
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const button = submitBtn;
            button.classList.add('loading');
            button.innerHTML = '<span>Predicting...</span>';
            
            // Simulate API call
            setTimeout(() => {
                // Mock prediction logic
                const studyHours = parseFloat(document.getElementById('study-hours').value);
                const attendance = parseFloat(document.getElementById('attendance').value);
                const grades = parseFloat(document.getElementById('grades').value);
                const extracurricular = document.getElementById('extracurricular').value;
                
                // Simple mock prediction
                let score = 0;
                if (studyHours > 10) score += 25;
                if (attendance > 80) score += 25;
                if (grades > 70) score += 25;
                if (extracurricular === 'Yes') score += 25;
                
                const prediction = score >= 75 ? 'Pass' : 'Fail';
                
                predictionText.textContent = prediction;
                result.style.display = 'block';
                
                button.classList.remove('loading');
                button.innerHTML = '<span>Predict Performance</span>';
            }, 2000);
        });

        // Initial progress update
        updateProgress();
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Collect form data
        input_data = {col: request.form[col] for col in columns}
        # Prepare data for model
        X = pd.DataFrame([input_data])
        # Convert numerics
        for col in X.columns:
            if col not in categorical_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        # Encode categoricals
        for col in categorical_cols:
            le = label_encoders[col]
            X[col] = le.transform(X[col])
        # Fill any missing values (shouldn't be any from form)
        X = X.fillna(0)
        # Predict
        pred = model.predict(X)[0]
        result = 'Pass' if pred == 1 else 'Fail'
    return render_template_string(TEMPLATE, result=result)

if __name__ == '__main__':
    app.run(debug=True) 