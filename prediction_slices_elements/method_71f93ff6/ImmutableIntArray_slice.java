// Source-based slice around line 421
// Method: <com.google.common.primitives.ImmutableIntArray: void forEach(IntConsumer)>

  public boolean contains(int target) {
    return indexOf(target) >= 0;
  }

  /**
   * Invokes {@code consumer} for each value contained in this array, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
  public void forEach(IntConsumer consumer) {
    checkNotNull(consumer);
    for (int i = start; i < end; i++) {
      consumer.accept(array[i]);
    }
  }

  /**
   * Returns a stream over the values in this array, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
