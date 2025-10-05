// Source-based slice around line 229
// Method: <com.google.common.collect.Range: Range upTo(C,BoundType)>

    return create(Cut.belowAll(), Cut.aboveValue(endpoint));
  }

  /**
   * Returns a range with no lower bound up to the given endpoint, which may be either inclusive
   * (closed) or exclusive (open).
   *
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> upTo(C endpoint, BoundType boundType) {
    switch (boundType) {
      case OPEN:
        return lessThan(endpoint);
      case CLOSED:
        return atMost(endpoint);
    }
    throw new AssertionError();
  }

  /**
