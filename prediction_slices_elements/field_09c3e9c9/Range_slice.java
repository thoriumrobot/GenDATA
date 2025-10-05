// Source-based slice around line 273
// Method: com.google.common.collect.Range.ALL

    switch (boundType) {
      case OPEN:
        return greaterThan(endpoint);
      case CLOSED:
        return atLeast(endpoint);
    }
    throw new AssertionError();
  }

  private static final Range<Comparable> ALL = new Range<>(Cut.belowAll(), Cut.aboveAll());

  /**
   * Returns a range that contains every value of type {@code C}.
   *
   * @since 14.0
   */
  @SuppressWarnings("unchecked")
  public static <C extends Comparable<?>> Range<C> all() {
    return (Range) ALL;
  }
