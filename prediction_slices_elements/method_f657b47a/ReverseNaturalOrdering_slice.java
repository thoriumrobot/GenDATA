// Source-based slice around line 91
// Method: <com.google.common.collect.ReverseNaturalOrdering: Object readResolve()>

    return NaturalOrdering.INSTANCE.min(iterator);
  }

  @Override
  public <E extends Comparable<?>> E max(Iterable<E> iterable) {
    return NaturalOrdering.INSTANCE.min(iterable);
  }

  // preserving singleton-ness gives equals()/hashCode() for free
  private Object readResolve() {
    return INSTANCE;
  }

  @Override
  public String toString() {
    return "Ordering.natural().reverse()";
  }

  private ReverseNaturalOrdering() {}

