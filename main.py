import discord
from discord.ext import commands
import processing
import os


intents = discord.Intents.all()
intents.typing = False
intents.presences = False

client = commands.Bot(command_prefix='?', intents=intents)


@client.command()
async def makemap(ctx):
    # Ignore messages sent by the bot itself
    if ctx.author == client.user:
        return

    # Check if the message has an attachment
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Download the image
                image_path = f'./temp/{attachment.filename}'
                await attachment.save(image_path)

                # Create a folder
                file_name = os.path.splitext(os.path.basename(attachment.filename))[0]
                os.makedirs(f'./temp/{file_name}')

                # Process the image to create a map
                processing.transform(image_path)

                # Respond with the map file and preview
                await ctx.channel.send(f"Here's your map:")
                await ctx.channel.send(file=discord.File(f'./temp/{file_name}/{file_name}_map.txt'))
                await ctx.channel.send(file=discord.File(f'./temp/{file_name}/{file_name}_preview.jpg'))
    else:
        await ctx.channel.send("Attach image please")


# Run the bot
client.run('MTE3MzU2OTU3NDY1MDc4NTg4Mw.GAet2l.URjgNvAMTt2RBhyQykiqamS2mjs6zooeK-x2hM')
