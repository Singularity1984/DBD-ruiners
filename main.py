import asyncio
import logging
from game_controller import GameController

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    """Главная функция"""
    game = GameController()
    await game.run()

if __name__ == "__main__":
    asyncio.run(main())