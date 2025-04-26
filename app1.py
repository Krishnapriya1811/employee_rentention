from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Gemini Pro API Key
genai.configure(api_key="AIzaSyBATLWCzIEAfdJRsQuZqbBXf2pJDWMx310")

model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

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

        age = data.get('age')
        gender = data.get('gender')
        marital_status = data.get('maritalStatus')
        job_role = data.get('jobRole')
        income = data.get('income')
        years_at_company = data.get('yearsAtCompany')
        years_since_promotion = data.get('yearsSincePromotion')
        job_satisfaction = data.get('jobSatisfaction')
        performance_rating = data.get('performanceRating')
        work_environment = data.get('workEnvironment')
        overtime = data.get('overtime')
        absenteeism = data.get('absenteeism')
        distance_from_home = data.get('distanceFromHome')
        companies_worked = data.get('companiesWorked')

        prompt = f"""
        Based on the following employee data, assess the attrition risk (Low, Medium, High) and provide:
        - Risk level
        - Reason for the risk
        - Recommendations in points to retain them

        Format your answer like:
        Risk: <Low/Medium/High>
        Reason: <Short one-line reason in three to four words>
        Recommendations(only three to four words):
        - Point 1
        - Point 2
        ...

        If the employee will not leave, say: "Employee will not leave".

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

        # If employee is fine
        if "Employee will not leave" in output:
            return jsonify({
                "status": "success",
                "message": "Employee will not leave"
            })

        # Initialize
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
                # Collect all bullet points after this
                for j in range(i + 1, len(lines)):
                    bullet = lines[j].strip()
                    if bullet.startswith("-"):
                        recommendations.append(bullet.strip("- ").replace("", "").strip())
                    elif bullet == "":
                        continue
                    else:
                        break
        print(risk,reason,recommendations)

        return jsonify({
            "status": "success",
            "risk": risk,
            "reason": reason,
            "recommendations": recommendations
        })


    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)