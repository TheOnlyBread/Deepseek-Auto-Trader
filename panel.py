# panel.py
# Render Web Service entrypoint: exposes a concrete ASGI app for uvicorn
from deepseek_binance_autotrader import build_panel_app

app = build_panel_app()
