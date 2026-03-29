from acquisition.ring_buffer import RingBuffer


def test_ring_buffer_basic():
    rb = RingBuffer(maxlen=3)
    rb.append(1.0, 10.0)
    rb.append(2.0, 20.0)
    rb.append(3.0, 30.0)
    rb.append(4.0, 40.0)
    ts, xs = rb.to_lists()
    assert ts == [2.0, 3.0, 4.0]
    assert xs == [20.0, 30.0, 40.0]
