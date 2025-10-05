// Source-based slice around line 426
// Method: <com.google.common.primitives.ImmutableDoubleArray: void forEach(DoubleConsumer)>

  public boolean contains(double target) {
    return indexOf(target) >= 0;
  }

  /**
   * Invokes {@code consumer} for each value contained in this array, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
  public void forEach(DoubleConsumer consumer) {
    checkNotNull(consumer);
    for (int i = start; i < end; i++) {
      consumer.accept(array[i]);
    }
  }

  /**
   * Returns a stream over the values in this array, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
