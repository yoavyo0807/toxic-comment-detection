import asyncio
import telegram


async def main():
    bot = telegram.Bot("6983089788:AAEjUsaehsFtqWrCrfhCT42gksqcKCFJwt0")
    async with bot:
        print(await bot.get_me())
        updates = (await bot.get_updates())[0]
        print(updates)


if __name__ == '__main__':
    asyncio.run(main())