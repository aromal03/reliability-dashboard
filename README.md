# AI Agent Reliability Framework — Dashboard

## Run locally (takes 2 minutes)

```bash
# Step 1: install dependencies
pip install streamlit plotly pandas numpy pgmpy networkx matplotlib Pillow

# Step 2: run the app
streamlit run app.py

# App opens at http://localhost:8501
```

## Deploy publicly — FREE (takes 10 minutes)

### Option A: Streamlit Cloud (easiest, recommended)

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click "New app" → select your repo → select app.py
5. Click "Deploy" — you get a public URL like:
   https://yourname-reliability-app-xxxx.streamlit.app

### Option B: Run in Colab with ngrok

Paste this in a Colab cell:

```python
!pip install streamlit pyngrok plotly pandas numpy -q

import subprocess, threading
from pyngrok import ngrok

# Write app.py to disk (paste full app.py content here)
with open('app.py', 'w') as f:
    f.write(open('app.py').read())  # or paste directly

# Start streamlit in background
def run():
    subprocess.run(['streamlit', 'run', 'app.py',
                    '--server.port', '8501',
                    '--server.headless', 'true'])

t = threading.Thread(target=run)
t.start()

import time; time.sleep(3)

# Create public URL
ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # get free at ngrok.com
public_url = ngrok.connect(8501)
print(f"\nDashboard URL: {public_url}")
print("Share this with your professor!")
```

## Pages in the app

1. **Overview** — project summary, pipeline diagram, key metrics
2. **Live Pipeline Demo** — type any question, watch 5 components run
3. **Fault Tree Analysis** — OR gate calculations, independence audit
4. **Event Tree Analysis** — safety barriers, unsafe output probabilities
5. **Bayesian Network** — CPT table, evidence injection, live updating
6. **Sensitivity Analysis** — tornado diagram, interactive sliders
7. **V1 vs V2 Comparison** — model upgrade results with CIs
8. **Apply to Any Agent** — generic calculator for new AI systems
