// Source-based slice around line 758
// Method: <com.google.common.collect.ImmutableRangeSet: boolean isPartialView()>

    }
  }

  /**
   * Returns {@code true} if this immutable range set's implementation contains references to
   * user-created objects that aren't accessible via this range set's methods. This is generally
   * used to determine whether {@code copyOf} implementations should make an explicit copy to avoid
   * memory leaks.
   */
  boolean isPartialView() {
    return ranges.isPartialView();
  }

  /** Returns a new builder for an immutable range set. */
  public static <C extends Comparable<?>> Builder<C> builder() {
    return new Builder<>();
  }

  /**
   * A builder for immutable range sets.
