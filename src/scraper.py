"""Minimal scraper module providing scrape_all function.

This module provides a placeholder implementation of the scrape_all function
to prevent import errors while the full scraping service is being implemented.
"""

import logging

logger = logging.getLogger(__name__)


async def scrape_all() -> dict[str, int]:
    """Placeholder implementation of scrape_all function.

    This is a temporary implementation that returns empty stats to prevent
    import errors. The full scraping functionality should be implemented
    using the IScrapingService interface defined in
    interfaces/scraping_service_interface.py.

    Returns:
        dict[str, int]: Synchronization statistics with empty values:
            - 'inserted': Number of new jobs added (0)
            - 'updated': Number of jobs updated (0)
            - 'skipped': Number of jobs skipped (0)
    """
    logger.info("scrape_all called - using placeholder implementation")
    logger.warning(
        "TODO: Implement full scraping service using IScrapingService interface"
    )

    return {"inserted": 0, "updated": 0, "skipped": 0}


def scrape_all_sync() -> dict[str, int]:
    """Synchronous wrapper for scrape_all function.

    Provides a synchronous interface for cases where async/await cannot be used.

    Returns:
        dict[str, int]: Synchronization statistics from scrape_all().
    """
    import asyncio

    return asyncio.run(scrape_all())
