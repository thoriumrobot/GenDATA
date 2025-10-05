// Source-based slice around line 86
// Method: <com.google.common.collect.ReverseNaturalOrdering: E max(Iterable)>

    return NaturalOrdering.INSTANCE.min(a, b, c, rest);
  }

  @Override
  public <E extends Comparable<?>> E max(Iterator<E> iterator) {
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
