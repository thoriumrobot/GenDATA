// Source-based slice around line 465
// Method: <com.google.common.primitives.ImmutableDoubleArray: Spliterator spliterator()>

    return startIndex == endIndex
        ? EMPTY
        : new ImmutableDoubleArray(array, start + startIndex, start + endIndex);
  }

  /*
   * We declare this as package-private, rather than private, to avoid generating a synthetic
   * accessor method (under -target 8) that would lack the Android flavor's @IgnoreJRERequirement.
   */
  Spliterator.OfDouble spliterator() {
    return Spliterators.spliterator(array, start, end, Spliterator.IMMUTABLE | Spliterator.ORDERED);
  }

  /**
   * Returns an immutable <i>view</i> of this array's values as a {@code List}; note that {@code
   * double} values are boxed into {@link Double} instances on demand, which can be very expensive.
   * The returned list should be used once and discarded. For any usages beyond that, pass the
   * returned list to {@link com.google.common.collect.ImmutableList#copyOf(Collection)
   * ImmutableList.copyOf} and use that list instead.
   */
