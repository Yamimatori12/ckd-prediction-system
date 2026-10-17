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


def get_ckd_stage(egfr):
    """
    Estimate CKD stage based on eGFR value.
    """

    if egfr >= 90:
        return "Stage 1", "Normal or high kidney function, but kidney damage may be present."
    elif egfr >= 60:
        return "Stage 2", "Mild decrease in kidney function."
    elif egfr >= 45:
        return "Stage 3a", "Mild to moderate decrease in kidney function."
    elif egfr >= 30:
        return "Stage 3b", "Moderate to severe decrease in kidney function."
    elif egfr >= 15:
        return "Stage 4", "Severe decrease in kidney function."
    else:
        return "Stage 5", "Kidney failure stage."


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

        if prediction[0] == 0:
            result = "CKD Detected"

            egfr = calculate_egfr(age, sc, sex)
            stage, stage_description = get_ckd_stage(egfr)

        else:
            result = "No CKD Detected"
            egfr = "-"
            stage = "Not Applicable"
            stage_description = "CKD stage estimation is only shown when CKD is detected."

        return render_template(
            "result.html",
            result=result,
            egfr=egfr,
            stage=stage,
            stage_description=stage_description
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