// Source-based slice around line 44
// Method: <com.google.common.collect.ReverseNaturalOrdering: Ordering reverse()>

    checkNotNull(left); // right null is caught later
    if (left == right) {
      return 0;
    }

    return ((Comparable<Object>) right).compareTo(left);
  }

  @Override
  public <S extends Comparable<?>> Ordering<S> reverse() {
    return Ordering.natural();
  }

  // Override the min/max methods to "hoist" delegation outside loops

  @Override
  public <E extends Comparable<?>> E min(E a, E b) {
    return NaturalOrdering.INSTANCE.max(a, b);
  }

