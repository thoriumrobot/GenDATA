// Source-based slice around line 232
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_misc()>

        LinkedList.class,
        Deque.class,
        Queue.class,
        PriorityQueue.class,
        BitSet.class,
        TreeSet.class,
        TreeMap.class);
  }

  public void testGet_misc() {
    assertNotNull(ArbitraryInstances.get(CharMatcher.class));
    assertNotNull(ArbitraryInstances.get(Currency.class).getCurrencyCode());
    assertNotNull(ArbitraryInstances.get(Locale.class));
    assertNotNull(ArbitraryInstances.get(Joiner.class).join(ImmutableList.of("a")));
    assertNotNull(ArbitraryInstances.get(Splitter.class).split("a,b"));
    assertThat(ArbitraryInstances.get(com.google.common.base.Optional.class)).isAbsent();
    ArbitraryInstances.get(Stopwatch.class).start();
    assertNotNull(ArbitraryInstances.get(Ticker.class));
    assertFreshInstanceReturned(Random.class);
    assertEquals(
