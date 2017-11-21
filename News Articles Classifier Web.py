from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        return render_template("index.html", article = request.form["news"])
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run()