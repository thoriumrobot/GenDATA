// Source-based slice around line 423
// Method: <com.google.common.primitives.ImmutableLongArray: void forEach(LongConsumer)>

  public boolean contains(long target) {
    return indexOf(target) >= 0;
  }

  /**
   * Invokes {@code consumer} for each value contained in this array, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
  public void forEach(LongConsumer consumer) {
    checkNotNull(consumer);
    for (int i = start; i < end; i++) {
      consumer.accept(array[i]);
    }
  }

  /**
   * Returns a stream over the values in this array, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
