// Source-based slice around line 71
// Method: <com.google.common.collect.ReverseNaturalOrdering: E max(E,E)>

    return NaturalOrdering.INSTANCE.max(iterator);
  }

  @Override
  public <E extends Comparable<?>> E min(Iterable<E> iterable) {
    return NaturalOrdering.INSTANCE.max(iterable);
  }

  @Override
  public <E extends Comparable<?>> E max(E a, E b) {
    return NaturalOrdering.INSTANCE.min(a, b);
  }

  @Override
  public <E extends Comparable<?>> E max(E a, E b, E c, E... rest) {
    return NaturalOrdering.INSTANCE.min(a, b, c, rest);
  }

  @Override
  public <E extends Comparable<?>> E max(Iterator<E> iterator) {
