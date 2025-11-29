from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import google.generativeai as genai
from tensorflow.keras.models import load_model
import numpy as np


app = Flask(__name__)
CORS(app)

# Gemini Pro API Key
genai.configure(api_key="AIzaSyBTdIKljJQhZWauYMSZkaXxGxQdnXv2yBI")

model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Load your deep learning model
dl_model = load_model('employee_attrition_model (2).h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/api/attrition-predict', methods=['GET'])
def predict_attrition():
    try:
        data = request.args

        # Fetch input data
        age = float(data.get('age'))
        gender = data.get('gender')
        marital_status = data.get('maritalStatus')
        job_role = data.get('jobRole')
        income = float(data.get('income'))
        years_at_company = float(data.get('yearsAtCompany'))
        years_since_promotion = float(data.get('yearsSincePromotion'))
        job_satisfaction = float(data.get('jobSatisfaction'))
        performance_rating = float(data.get('performanceRating'))
        work_environment = float(data.get('workEnvironment'))
        overtime = data.get('overtime')
        absenteeism = float(data.get('absenteeism'))
        distance_from_home = float(data.get('distanceFromHome'))
        companies_worked = float(data.get('companiesWorked'))

        # ----------------------------------------
        # Preprocess for DL Model
        # Make sure your input is properly encoded (gender, marital_status, etc.)
        # Here we assume your model expects numerical input only.

        # Example simple encoding (you need to match your model's expected format)
        gender_num = 1 if gender.lower() == 'male' else 0
        marital_status_num = {
            "single": 0,
            "married": 1,
            "divorced": 2
        }.get(marital_status.lower(), 0)
        overtime_num = 1 if overtime.lower() == 'yes' else 0

        # Similarly encode job_role if needed or drop if your model ignores it.
        job_role_mapping = {
            "sales executive": 0,
            "research scientist": 1,
            "laboratory technician": 2,
            "manufacturing director": 3,
            "healthcare representative": 4,
            "manager": 5,
            "sales representative": 6,
            "research director": 7,
            "human resources": 8
        }

        job_role_num = job_role_mapping.get(job_role.lower(), 0)  # default to 0

        # Example input array for model

        input_data = np.array([[
            age, gender_num, marital_status_num,job_role_num, income, years_at_company,
            years_since_promotion, job_satisfaction, performance_rating,
            work_environment, overtime_num, absenteeism, distance_from_home,
            companies_worked
        ]])

        # Predict using DL model
        prediction = dl_model.predict(input_data)
        # Suppose model outputs probability of attrition
        attrition_probability = float(prediction[0][0]) * 100  # Percentage

        # ----------------------------------------
        # Use DL model output in Gemini prompt

        prompt = f"""
        Based on the following employee data and employee input data, assess the attrition risk (Low, Medium, High) and provide:
        - Risk level
        - Reason for the risk
        - Recommendations (only three to four words)

        Format your answer like:
        Risk: <Low/Medium/High>
        Reason: <Short one-line reason in three to four words>
        Recommendations(only three to four words):
        - Point 1
        - Point 2
        ...

        If the employee will not leave, say: "Employee will not leave".

        Employee Data:
        Age: {age}
        Gender: {gender}
        Marital Status: {marital_status}
        Job Role: {job_role}
        Annual Income: ${income}
        Years at Company: {years_at_company}
        Years Since Last Promotion: {years_since_promotion}
        Job Satisfaction Level: {job_satisfaction}
        Performance Rating: {performance_rating}
        Work Environment Satisfaction: {work_environment}
        Overtime: {overtime}
        Absenteeism (days/year): {absenteeism}
        Distance from Home (km): {distance_from_home}
        Number of Companies Worked: {companies_worked}
        """

        response = model.generate_content(prompt)
        output = response.text.strip()

        if "Employee will not leave" in output:
            return jsonify({
                "status": "success",
                "message": "Employee will not leave"
            })

        risk = ""
        reason = ""
        recommendations = []
        lines = output.splitlines()

        for i, line in enumerate(lines):
            line_clean = line.replace("", "").strip()
            if line_clean.lower().startswith("risk:"):
                risk = line_clean.split(":", 1)[1].strip()
            elif line_clean.lower().startswith("reason:"):
                reason = line_clean.split(":", 1)[1].strip()
            elif line_clean.lower().startswith("recommendations:"):
                for j in range(i + 1, len(lines)):
                    bullet = lines[j].strip()
                    if bullet.startswith("-"):
                        recommendations.append(bullet.strip("- ").replace("", "").strip())
                    elif bullet == "":
                        continue
                    else:
                        break

        return jsonify({
            "status": "success",
            "attrition_probability": f"{attrition_probability:.2f}%",
            "risk": risk,
            "reason": reason,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
