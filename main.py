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

#добавь еще одного "игрока" - поле, которое должно мешать и выжившим и маньяку, расставляя перед ними маленькие стенки (с откатом по времени достаточным чтобы не ломать игру(через некоторое время стенки должны пропадать))
#маньяк почему-то когда убивает выжившего начинает ходить только около него, так быть не должно