from flask import Flask, render_template, request, redirect, session, send_file
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import time
import sqlite3
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.secret_key = "secret123"

if not os.path.exists("static"):
    os.makedirs("static")

latest_report_data = {}

def init_db():
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        stocks TEXT,
        amount REAL,
        risk TEXT,
        portfolio_return REAL,
        portfolio_risk REAL,
        sharpe REAL,
        score REAL
    )
    """)
    conn.commit()
    conn.close()

init_db()

def stock_logo(stock):
    domain_map = {
        "AAPL": "apple.com",
        "MSFT": "microsoft.com",
        "GOOGL": "google.com",
        "GOOG": "google.com",
        "AMZN": "amazon.com",
        "META": "meta.com",
        "TSLA": "tesla.com",
        "NFLX": "netflix.com",
        "NVDA": "nvidia.com",
        "RELIANCE.NS": "ril.com",
        "TCS.NS": "tcs.com",
        "INFY.NS": "infosys.com",
        "HDFCBANK.NS": "hdfcbank.com",
        "ICICIBANK.NS": "icicibank.com",
        "SBIN.NS": "sbi.co.in"
    }
    return f"https://logo.clearbit.com/{domain_map.get(stock, 'moneycontrol.com')}"

def create_stock_trend_chart(stock, close_prices):
    chart_path = f"static/trend_{stock.replace('.', '_')}.png"

    plt.figure(figsize=(5, 3))
    plt.plot(close_prices.tail(60))

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    plt.xticks(rotation=30)
    plt.title(f"{stock} Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    return f"trend_{stock.replace('.', '_')}.png"

@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("portfolio.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("portfolio.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect("/")
        else:
            return render_template("error.html", message="Invalid login")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

@app.route("/optimize", methods=["POST"])
def optimize():
    global latest_report_data

    if "user" not in session:
        return redirect("/login")

    stocks = request.form["stocks"]
    amount = float(request.form["amount"])
    risk_level = request.form["risk"]
    username = session["user"]

    stock_list = list(dict.fromkeys([s.strip().upper() for s in stocks.split(",") if s.strip()]))

    try:
        data = yf.download(stock_list, period="1y", auto_adjust=True, progress=False)["Close"]

        if isinstance(data, pd.Series):
            data = data.to_frame()

        data = data.dropna(axis=1, how="all")
        stock_list = list(data.columns)

        if len(stock_list) < 2:
            return render_template("error.html", message="Please enter at least 2 valid stock symbols.")

        daily_returns = data.pct_change().dropna()

        annual_returns = daily_returns.mean() * 252
        covariance_matrix = daily_returns.cov() * 252
        annual_risk = daily_returns.std() * (252 ** 0.5)

        equal_weights = np.array([1 / len(stock_list)] * len(stock_list))

        portfolio_return = np.dot(equal_weights, annual_returns.values)
        portfolio_risk = np.sqrt(np.dot(equal_weights.T, np.dot(covariance_matrix.values, equal_weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk != 0 else 0

        if risk_level == "low":
            score_data = 1 / annual_risk
        elif risk_level == "medium":
            score_data = annual_returns / annual_risk
        else:
            score_data = annual_returns

        score_data = score_data.replace([np.inf, -np.inf], 0).fillna(0)

        if score_data.sum() == 0:
            optimized_weights = equal_weights
        else:
            optimized_weights = (score_data / score_data.sum()).values

        plt.figure(figsize=(4, 4))
        plt.pie(optimized_weights, labels=stock_list, autopct="%1.1f%%", startangle=140)
        plt.title("Optimized Portfolio Allocation")
        plt.savefig("static/allocation_chart.png", bbox_inches="tight")
        plt.close()

        frontier_results = np.zeros((3, 100))

        for i in range(100):
            w = np.random.random(len(stock_list))
            w /= np.sum(w)

            sim_return = np.dot(w, annual_returns.values)
            sim_risk = np.sqrt(np.dot(w.T, np.dot(covariance_matrix.values, w)))

            frontier_results[0, i] = sim_risk
            frontier_results[1, i] = sim_return
            frontier_results[2, i] = sim_return / sim_risk if sim_risk != 0 else 0

        plt.figure(figsize=(5, 3))
        plt.scatter(frontier_results[0], frontier_results[1], c=frontier_results[2], cmap="viridis")
        plt.colorbar(label="Sharpe Ratio")
        plt.xlabel("Risk")
        plt.ylabel("Return")
        plt.title("Efficient Frontier")
        plt.tight_layout()
        plt.savefig("static/efficient_frontier.png", bbox_inches="tight")
        plt.close()

        norm_return = min(max(portfolio_return, 0), 0.3) / 0.3
        norm_risk = 1 - min(portfolio_risk, 0.4) / 0.4
        diversification = min(len(stock_list), 10) / 10

        portfolio_score = (0.5 * norm_return + 0.3 * norm_risk + 0.2 * diversification) * 100
        portfolio_score = round(portfolio_score, 2)

        if portfolio_score >= 75:
            insight = "Your portfolio looks strong with good return potential and controlled risk."
        elif portfolio_score >= 50:
            insight = "Your portfolio is balanced, but diversification and return can still improve."
        else:
            insight = "Your portfolio needs improvement. Consider adding stable and diversified stocks."

        if portfolio_risk > 0.30:
            risk_insight = "Risk level is high, so the portfolio may face larger price fluctuations."
        elif portfolio_risk > 0.18:
            risk_insight = "Risk level is moderate and suitable for balanced investors."
        else:
            risk_insight = "Risk level is low and suitable for conservative investors."

        if len(stock_list) < 4:
            diversification_insight = "Diversification is low. Adding more stocks can reduce risk."
        else:
            diversification_insight = "Diversification is good because multiple stocks are included."

        result = []

        for i, stock in enumerate(stock_list):
            stock_return = annual_returns[stock]
            stock_risk = annual_risk[stock]

            if stock_return > 0.15 and stock_risk < 0.25:
                recommendation = "BUY"
            elif stock_return > 0.05:
                recommendation = "HOLD"
            else:
                recommendation = "SELL"

            last_price = data[stock].iloc[-1]
            predictions = [round(last_price * (1 + 0.01 * j), 2) for j in range(1, 6)]

            result.append({
                "stock": stock,
                "logo": stock_logo(stock),
                "expected_return": round(annual_returns[stock] * 100, 2),
                "risk": round(annual_risk[stock] * 100, 2),
                "weight": round(optimized_weights[i] * 100, 2),
                "optimized_allocation": round(amount * optimized_weights[i], 2),
                "recommendation": recommendation,
                "trend_chart": create_stock_trend_chart(stock, data[stock]),
                "predictions": predictions
            })

        metrics = {
            "portfolio_return": round(portfolio_return * 100, 2),
            "portfolio_risk": round(portfolio_risk * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }

        conn = sqlite3.connect("portfolio.db")
        c = conn.cursor()

        c.execute("""
        INSERT INTO portfolio_history
        (username, stocks, amount, risk, portfolio_return, portfolio_risk, sharpe, score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username,
            ",".join(stock_list),
            amount,
            risk_level,
            metrics["portfolio_return"],
            metrics["portfolio_risk"],
            metrics["sharpe_ratio"],
            portfolio_score
        ))

        conn.commit()
        conn.close()

        latest_report_data = {
            "username": username,
            "amount": amount,
            "risk_level": risk_level,
            "score": portfolio_score,
            "metrics": metrics,
            "insight": insight,
            "risk_insight": risk_insight,
            "diversification_insight": diversification_insight,
            "result": result
        }

        return render_template(
            "result.html",
            result=result,
            score=portfolio_score,
            metrics=metrics,
            insight=insight,
            risk_insight=risk_insight,
            diversification_insight=diversification_insight,
            sector_data=[],
            time=time.time()
        )

    except Exception as e:
        return render_template("error.html", message=str(e))

@app.route("/download_report")
def download_report():
    if "user" not in session:
        return redirect("/login")

    if not latest_report_data:
        return render_template("error.html", message="Please optimize a portfolio first, then download the report.")

    pdf_path = "static/portfolio_report.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("StockVision AI - Portfolio Optimization Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"User: {latest_report_data['username']}", styles["Normal"]))
    elements.append(Paragraph(f"Generated On: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}", styles["Normal"]))
    elements.append(Paragraph(f"Total Investment: {latest_report_data['amount']}", styles["Normal"]))
    elements.append(Paragraph(f"Selected Risk Level: {latest_report_data['risk_level']}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Portfolio Score", styles["Heading2"]))
    elements.append(Paragraph(f"{latest_report_data['score']} / 100", styles["Heading1"]))
    elements.append(Paragraph(latest_report_data["insight"], styles["Normal"]))
    elements.append(Paragraph(latest_report_data["risk_insight"], styles["Normal"]))
    elements.append(Paragraph(latest_report_data["diversification_insight"], styles["Normal"]))
    elements.append(Spacer(1, 12))

    metrics = latest_report_data["metrics"]

    summary_table = Table([
        ["Expected Return (%)", "Portfolio Risk (%)", "Sharpe Ratio"],
        [metrics["portfolio_return"], metrics["portfolio_risk"], metrics["sharpe_ratio"]]
    ])

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))

    elements.append(Paragraph("Portfolio Summary", styles["Heading2"]))
    elements.append(summary_table)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Optimized Allocation Chart", styles["Heading2"]))
    elements.append(Image("static/allocation_chart.png", width=260, height=260))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Efficient Frontier Chart", styles["Heading2"]))
    elements.append(Image("static/efficient_frontier.png", width=360, height=220))
    elements.append(Spacer(1, 12))

    table_data = [["Stock", "Return %", "Risk %", "Weight %", "Amount", "AI Suggestion"]]

    for item in latest_report_data["result"]:
        table_data.append([
            item["stock"],
            item["expected_return"],
            item["risk"],
            item["weight"],
            item["optimized_allocation"],
            item["recommendation"]
        ])

    stock_table = Table(table_data, repeatRows=1)

    stock_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ec4899")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8)
    ]))

    elements.append(Paragraph("AI Stock Recommendation Table", styles["Heading2"]))
    elements.append(stock_table)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("5-Day Future Price Prediction", styles["Heading2"]))

    prediction_data = [["Stock", "Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]]

    for item in latest_report_data["result"]:
        prediction_data.append([item["stock"]] + item["predictions"])

    prediction_table = Table(prediction_data, repeatRows=1)

    prediction_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14b8a6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8)
    ]))

    elements.append(prediction_table)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("Note", styles["Heading2"]))
    elements.append(Paragraph(
        "This report is generated using historical stock data and rule-based analytical logic. "
        "It is intended for educational and project demonstration purposes.",
        styles["Normal"]
    ))

    doc.build(elements)

    return send_file(pdf_path, as_attachment=True)

@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()

    c.execute("""
    SELECT stocks, amount, risk, portfolio_return, portfolio_risk, sharpe
    FROM portfolio_history
    WHERE username=?
    ORDER BY id DESC
    """, (session["user"],))

    data = c.fetchall()
    conn.close()

    return render_template("history.html", data=data)

if __name__ == "__main__":
    app.run(debug=True, port=5001)