import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
TOKEN = os.getenv('EVALBOT_KEY')
CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID', 0))

# Don't create bot at module level
bot = None
is_running = False

async def get_bot():
    global bot, is_running
    if bot is None:
        intents = discord.Intents.default()
        intents.message_content = True
        bot = commands.Bot(command_prefix='!', intents=intents)
        
        @bot.event
        async def on_ready():
            print(f'Logged in as {bot.user}!')
    
    return bot

async def send_inference(text):
    """Send inference results without starting the bot"""
    if not TOKEN:
        print("No Discord token found. Skipping Discord notification.")
        return
    
    # Use a separate Discord client just for sending
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    
    @client.event
    async def on_ready():
        channel = client.get_channel(CHANNEL_ID)
        if channel:
            await channel.send(f"Model Inference:\n```\n{text}\n```")
        await client.close()
    
    try:
        await client.start(TOKEN)
    except Exception as e:
        print(f"Discord error: {e}")

# Only run the bot if this file is executed directly
if __name__ == "__main__":
    if TOKEN:
        intents = discord.Intents.default()
        intents.message_content = True
        bot = commands.Bot(command_prefix='!', intents=intents)
        # Add your commands here
        bot.run(TOKEN)
    else:
        print("No Discord token found!")