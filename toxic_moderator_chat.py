import asyncio
import telegram


async def main():
    bot = telegram.Bot("TOKEN")
    async with bot:
        print(await bot.get_me())
        updates = (await bot.get_updates())[0]
        print(updates)


if __name__ == '__main__':
    asyncio.run(main())
