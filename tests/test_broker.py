import queue

from utils.broker import AlertBroker, ConnectionLimiter


def test_broker_broadcasts_to_every_subscriber():
    broker = AlertBroker(queue_size=2, history_size=2)
    first, _ = broker.subscribe()
    second, _ = broker.subscribe()
    broker.publish({"id": 1})
    assert first.get_nowait() == {"id": 1}
    assert second.get_nowait() == {"id": 1}
    assert broker.subscriber_count == 2
    broker.unsubscribe(first)
    assert broker.subscriber_count == 1


def test_broker_bounds_slow_subscriber_queue_and_history():
    broker = AlertBroker(queue_size=2, history_size=2)
    subscriber, _ = broker.subscribe()
    for alert_id in range(4):
        broker.publish({"id": alert_id})
    assert subscriber.get_nowait() == {"id": 2}
    assert subscriber.get_nowait() == {"id": 3}
    try:
        subscriber.get_nowait()
        raise AssertionError("queue should be empty")
    except queue.Empty:
        pass

    _, history = broker.subscribe()
    assert history == [{"id": 2}, {"id": 3}]


def test_connection_limiter_never_exceeds_or_underflows():
    limiter = ConnectionLimiter(2)
    assert limiter.acquire()
    assert limiter.acquire()
    assert not limiter.acquire()
    assert limiter.active == 2
    limiter.release()
    limiter.release()
    limiter.release()
    assert limiter.active == 0
