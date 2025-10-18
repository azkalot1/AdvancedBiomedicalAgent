#!/usr/bin/env python3
"""
Async database configuration for database.
Provides asyncpg connection management for async database operations.
"""

from typing import Any

import asyncpg
import numpy as np

from .config import DatabaseConfig


def encode_vector(value: np.ndarray) -> str:
    """Formats a numpy array into the string format required by pgvector."""
    if value is None:
        return None
    return "[" + ",".join(map(str, value)) + "]"


def decode_vector(value: str) -> np.ndarray:
    """Formats a pgvector string into a numpy array."""
    if value is None:
        return None
    return np.array(value[1:-1].split(','), dtype=np.float32)


class AsyncDatabaseConfig:
    def __init__(self, config: DatabaseConfig, pool_size: int = 10):
        self.config = config
        self.pool_size = pool_size
        self._pool: asyncpg.Pool | None = None

    # FIX: Create an initializer for new connections
    async def _init_connection(self, conn: asyncpg.Connection):
        """Initializes a new connection, setting up the vector extension and codec."""
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # FIX: Use the custom encoder and decoder
        await conn.set_type_codec('vector', encoder=encode_vector, decoder=decode_vector, schema='public', format='text')

    async def execute_command(self, query: str, *args) -> None:
        """Execute a command that does not return rows (e.g., UPDATE, CREATE INDEX)."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(query, *args)

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=60,
                # FIX: Pass the initializer to the pool creation
                init=self._init_connection,
            )
        return self._pool

    async def execute_query(self, query: str, *args) -> list:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # FIX: The codec setup is no longer needed here
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def execute_one(self, query: str, *args) -> dict[str, Any] | None:
        """Execute a query and return a single result as a dictionary."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def close_pool(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()


async def get_async_connection(config: DatabaseConfig) -> AsyncDatabaseConfig:
    global _async_db_config_instance
    if _async_db_config_instance is None:
        _async_db_config_instance = AsyncDatabaseConfig(config)
    return _async_db_config_instance


_async_db_config_instance: AsyncDatabaseConfig | None = None
