// Source-based slice around line 61
// Method: <com.google.common.collect.ReverseNaturalOrdering: E min(Iterator)>

    return NaturalOrdering.INSTANCE.max(a, b);
  }

  @Override
  public <E extends Comparable<?>> E min(E a, E b, E c, E... rest) {
    return NaturalOrdering.INSTANCE.max(a, b, c, rest);
  }

  @Override
  public <E extends Comparable<?>> E min(Iterator<E> iterator) {
    return NaturalOrdering.INSTANCE.max(iterator);
  }

  @Override
  public <E extends Comparable<?>> E min(Iterable<E> iterable) {
    return NaturalOrdering.INSTANCE.max(iterable);
  }

  @Override
  public <E extends Comparable<?>> E max(E a, E b) {
