from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and column order
model = joblib.load(open("ckd_model.pkl", "rb"))
columns = joblib.load(open("columns.pkl", "rb"))


def calculate_egfr(age, serum_creatinine, sex):
    """
    Estimate eGFR using CKD-EPI 2021 creatinine equation.
    Serum creatinine must be in mg/dL.
    """

    if sex.lower() == "female":
        k = 0.7
        alpha = -0.241
        sex_factor = 1.012
    else:
        k = 0.9
        alpha = -0.302
        sex_factor = 1.0

    scr_k = serum_creatinine / k

    egfr = (
        142
        * (min(scr_k, 1) ** alpha)
        * (max(scr_k, 1) ** -1.200)
        * (0.9938 ** age)
        * sex_factor
    )

    return round(egfr, 2)


def assess_ckd_and_stage(prediction_code, egfr, al, htn, dm):
    """
    Determine CKD classification and staging based on ML model prediction,
    eGFR calculation, and key clinical markers (albuminuria, hypertension, diabetes).
    """
    # A patient has CKD if:
    # 1. Model predicted CKD (prediction_code == 0)
    # 2. Or eGFR < 60 mL/min (chronically reduced eGFR is clinically CKD Stage 3-5)
    # 3. Or urine albumin >= 1 (albuminuria indicates kidney damage)
    is_ckd = (prediction_code == 0) or (egfr < 60) or (al >= 1)

    if not is_ckd:
        status = "Normal"
        result = "No CKD Detected"
        stage = "Normal"
        if egfr >= 90:
            stage_description = (
                f"Kidney function is within the normal healthy range (eGFR: {egfr} mL/min/1.73m²) "
                "with no clinical signs of chronic kidney disease or kidney damage."
            )
        else:
            stage_description = (
                f"Kidney function is within expected limits (eGFR: {egfr} mL/min/1.73m²) "
                "with no clinical markers of kidney damage detected (normal for age)."
            )
    else:
        status = "CKD Detected"
        result = "CKD Detected"

        if egfr >= 90:
            stage = "Stage 1"
            stage_description = (
                "eGFR remains within a normal range (90 mL/min or more), "
                "but other test results suggest signs of kidney damage (such as albuminuria or hypertension)."
            )
        elif egfr >= 60:
            stage = "Stage 2"
            stage_description = (
                "Slightly reduced kidney function (60 to 89 mL/min) "
                "with other tests suggesting kidney damage."
            )
        elif egfr >= 30:
            stage = "Stage 3"
            stage_description = (
                "Moderately reduced kidney function (30 to 59 mL/min)."
            )
        elif egfr >= 15:
            stage = "Stage 4"
            stage_description = (
                "Severely reduced kidney function (15 to 29 mL/min)."
            )
        else:
            stage = "Stage 5"
            stage_description = (
                "Very severe or end-stage kidney failure (Less than 15 mL/min)."
            )

    return status, result, stage, stage_description


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        bp = float(request.form["bp"])
        sg = float(request.form["sg"])
        al = float(request.form["al"])
        su = float(request.form["su"])
        rbc = float(request.form["rbc"])
        pc = float(request.form["pc"])
        pcc = float(request.form["pcc"])
        ba = float(request.form["ba"])
        bgr = float(request.form["bgr"])
        bu = float(request.form["bu"])
        sc = float(request.form["sc"])
        sod = float(request.form["sod"])
        pot = float(request.form["pot"])
        hemo = float(request.form["hemo"])
        pcv = float(request.form["pcv"])
        wc = float(request.form["wc"])
        rc = float(request.form["rc"])
        htn = float(request.form["htn"])
        dm = float(request.form["dm"])
        cad = float(request.form["cad"])
        appet = float(request.form["appet"])
        pe = float(request.form["pe"])
        ane = float(request.form["ane"])

        # Extra input only for eGFR calculation
        sex = request.form["sex"]

        input_data = pd.DataFrame({
            "age": [age],
            "bp": [bp],
            "sg": [sg],
            "al": [al],
            "su": [su],
            "rbc": [rbc],
            "pc": [pc],
            "pcc": [pcc],
            "ba": [ba],
            "bgr": [bgr],
            "bu": [bu],
            "sc": [sc],
            "sod": [sod],
            "pot": [pot],
            "hemo": [hemo],
            "pcv": [pcv],
            "wc": [wc],
            "rc": [rc],
            "htn": [htn],
            "dm": [dm],
            "cad": [cad],
            "appet": [appet],
            "pe": [pe],
            "ane": [ane]
        })

        # Make sure input column order follows training column order
        input_data = input_data[columns]

        prediction = model.predict(input_data)

        # Always calculate eGFR using CKD-EPI 2021 equation
        egfr = calculate_egfr(age, sc, sex)

        # Assess clinical classification, status, and precise CKD stage
        status, result, stage, stage_description = assess_ckd_and_stage(
            prediction[0], egfr, al, htn, dm
        )

        patient_summary = {
            "Age": f"{int(age)} yrs",
            "Sex": sex.capitalize(),
            "Blood Pressure": f"{bp} mmHg",
            "Serum Creatinine": f"{sc} mg/dL",
            "Blood Glucose": f"{bgr} mg/dL",
            "Blood Urea": f"{bu} mg/dL",
            "Hemoglobin": f"{hemo} g/dL",
            "Urine Albumin": f"{int(al)}/5",
            "Hypertension": "Yes" if htn == 1 else "No",
            "Diabetes Mellitus": "Yes" if dm == 1 else "No",
        }

        return render_template(
            "result.html",
            status=status,
            result=result,
            egfr=egfr,
            stage=stage,
            stage_description=stage_description,
            patient_summary=patient_summary
        )

    except Exception as e:
        return render_template(
            "result.html",
            result="Error",
            egfr="-",
            stage="-",
            stage_description=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)