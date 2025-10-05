// Source-based slice around line 247
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_concurrent()>

    assertThat(ArbitraryInstances.get(com.google.common.base.Optional.class)).isAbsent();
    ArbitraryInstances.get(Stopwatch.class).start();
    assertNotNull(ArbitraryInstances.get(Ticker.class));
    assertFreshInstanceReturned(Random.class);
    assertEquals(
        ArbitraryInstances.get(Random.class).nextInt(),
        ArbitraryInstances.get(Random.class).nextInt());
  }

  public void testGet_concurrent() {
    assertTrue(ArbitraryInstances.get(BlockingDeque.class).isEmpty());
    assertTrue(ArbitraryInstances.get(BlockingQueue.class).isEmpty());
    assertTrue(ArbitraryInstances.get(DelayQueue.class).isEmpty());
    assertTrue(ArbitraryInstances.get(SynchronousQueue.class).isEmpty());
    assertTrue(ArbitraryInstances.get(PriorityBlockingQueue.class).isEmpty());
    assertTrue(ArbitraryInstances.get(ConcurrentMap.class).isEmpty());
    assertTrue(ArbitraryInstances.get(ConcurrentNavigableMap.class).isEmpty());
    ArbitraryInstances.get(Executor.class).execute(ArbitraryInstances.get(Runnable.class));
    assertNotNull(ArbitraryInstances.get(ThreadFactory.class));
    assertFreshInstanceReturned(
