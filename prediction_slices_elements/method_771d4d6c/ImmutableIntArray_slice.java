// Source-based slice around line 412
// Method: <com.google.common.primitives.ImmutableIntArray: boolean contains(int)>

      }
    }
    return -1;
  }

  /**
   * Returns {@code true} if {@code target} is present at any index in this array. Equivalent to
   * {@code asList().contains(target)}.
   */
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
