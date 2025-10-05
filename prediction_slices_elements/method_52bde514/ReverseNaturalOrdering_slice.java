// Source-based slice around line 81
// Method: <com.google.common.collect.ReverseNaturalOrdering: E max(Iterator)>

    return NaturalOrdering.INSTANCE.min(a, b);
  }

  @Override
  public <E extends Comparable<?>> E max(E a, E b, E c, E... rest) {
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
