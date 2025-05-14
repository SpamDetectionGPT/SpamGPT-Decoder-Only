import asyncio
import os
from dotenv import load_dotenv
import aiohttp

load_dotenv()

async def send_inference(text):
    """Send inference results to Discord via webhook"""
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    
    if not webhook_url:
        print("No Discord webhook URL found. Skipping Discord notification.")
        return
    
    async with aiohttp.ClientSession() as session:
        webhook = Webhook.from_url(webhook_url, session=session)
        await webhook.send(f"Model Inference:\n```\n{text}\n```")