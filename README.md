# Falling Knife Trading Agent

A CLI tool inspired by Michael Burry's approach to scaling into falling assets only after enough volume turnover suggests the shareholder base has rotated — helping you avoid catching a falling knife too early.

## Usage

Install dependencies with `pip install -r requirements.txt`, then run `python knife_agent.py --ticker SYMBOL [--type TYPE] [--initial_price PRICE]`. `--ticker` is required (e.g. `PLTR`, `SPY`); `--type` sets turnover thresholds (`Old Guard`, `Standard`, `High-Growth`, default `Standard`); `--initial_price` triggers a DCA buy signal if you're down 20%+ from that entry. The tool prints a colored dashboard (price, volume turnover, fundamentals, sentiment) and a final verdict: **BUY SIGNAL**, **WARNING**, or **HOLD/WAIT**.

## For Example
python knife_agent.py --ticker PLTR --initial_price 168.45 --type "High-Growth"
