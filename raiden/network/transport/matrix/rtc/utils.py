import asyncio
from asyncio import Future

import gevent
import structlog
from gevent import Greenlet
from gevent.timeout import Timeout

from raiden.exceptions import RaidenUnrecoverableError
from raiden.network.transport.matrix.rtc import aiogevent
from raiden.network.transport.matrix.rtc.aiogevent import yield_future
from raiden.utils.typing import Any, Callable, Coroutine, Generator, Optional, Union

ASYNCIO_LOOP_RUNNING_TIMEOUT = 10

log = structlog.get_logger(__name__)


def setup_asyncio_event_loop() -> Generator:
    asyncio.set_event_loop_policy(aiogevent.EventLoopPolicy())
    asyncio_greenlet = gevent.spawn(asyncio.get_event_loop().run_forever)
    gevent.sleep(0.05)
    if not asyncio.get_event_loop().is_running():
        log.debug("Asyncio loop not running yet. Waiting.")
        with Timeout(ASYNCIO_LOOP_RUNNING_TIMEOUT, RaidenUnrecoverableError):
            while not asyncio.get_event_loop().is_running():
                gevent.sleep(0.05)

    return asyncio_greenlet


def spawn_coroutine(
    coroutine: Union[Coroutine, Future],
    callback: Optional[Callable[[Any, Any], None]],
    **kwargs: Any,
) -> Greenlet:
    """Spawns a greenlet which runs a coroutine inside waiting for the result"""

    wrapped_coroutine = gevent.spawn(
        wait_for_future, asyncio.ensure_future(coroutine), callback, **kwargs
    )
    wrapped_coroutine.name = str(coroutine)
    return wrapped_coroutine


def wait_for_future(future: Future, callback: Callable, **kwargs: Any) -> None:
    """yield future and call callback with the result"""
    result = yield_future(future)
    if callback is not None:
        callback(result, **kwargs)


def create_task_callback(
    callback: Callable[..., None],
    *args: Any,
    **kwargs: Any,
) -> Callable:
    if asyncio.iscoroutine(callback):

        def _coroutine_callback(result: Future) -> None:

            asyncio.create_task(callback(result.result(), *args, **kwargs))  # type: ignore

        return _coroutine_callback

    else:

        def _greenlet_callback(result: Any) -> None:
            wrap_callback(callback, result.result(), *args, **kwargs)

        return _greenlet_callback


def wrap_callback(callback: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    callback_greenlet = gevent.Greenlet(callback, *args, **kwargs)
    callback_greenlet.start()
