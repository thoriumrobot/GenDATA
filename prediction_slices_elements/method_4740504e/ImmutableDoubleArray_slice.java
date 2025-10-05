// Source-based slice around line 417
// Method: <com.google.common.primitives.ImmutableDoubleArray: boolean contains(double)>

      }
    }
    return -1;
  }

  /**
   * Returns {@code true} if {@code target} is present at any index in this array. Values are
   * compared as if by {@link Double#equals}. Equivalent to {@code asList().contains(target)}.
   */
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
