// Source-based slice around line 476
// Method: <com.google.common.primitives.ImmutableDoubleArray: List asList()>

  }

  /**
   * Returns an immutable <i>view</i> of this array's values as a {@code List}; note that {@code
   * double} values are boxed into {@link Double} instances on demand, which can be very expensive.
   * The returned list should be used once and discarded. For any usages beyond that, pass the
   * returned list to {@link com.google.common.collect.ImmutableList#copyOf(Collection)
   * ImmutableList.copyOf} and use that list instead.
   */
  public List<Double> asList() {
    /*
     * Typically we cache this kind of thing, but much repeated use of this view is a performance
     * anti-pattern anyway. If we cache, then everyone pays a price in memory footprint even if
     * they never use this method.
     */
    return new AsList(this);
  }

  private static final class AsList extends AbstractList<Double>
      implements RandomAccess, Serializable {
