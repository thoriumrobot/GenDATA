// Source-based slice around line 56
// Method: <com.google.common.collect.ReverseNaturalOrdering: E min(E,E,E,E)>


  // Override the min/max methods to "hoist" delegation outside loops

  @Override
  public <E extends Comparable<?>> E min(E a, E b) {
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
