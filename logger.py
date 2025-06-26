from loguru import logger

logger.add("app.log", rotation="1 MB", retention="7 days", level="DEBUG")