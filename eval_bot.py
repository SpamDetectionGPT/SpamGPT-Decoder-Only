import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('EVALBOT_KEY')


intents = discord.Intents.default()
intents.message_content = True  # Needed to read message content

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')

@bot.command()
async def perms(ctx):
    perms = ctx.channel.permissions_for(ctx.guild.me)
    await ctx.send(f"My permissions here:\n{perms}")

@bot.command()
async def loss(ctx):
    await ctx.send('Loss: 1.3872638')

bot.run(TOKEN)
