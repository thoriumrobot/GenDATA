// Source-based slice around line 80
// Method: <com.google.common.collect.JdkBackedImmutableMultiset: Entry getEntry(int)>

  @LazyInit private transient @Nullable ImmutableSet<E> elementSet;

  @Override
  public ImmutableSet<E> elementSet() {
    ImmutableSet<E> result = elementSet;
    return (result == null) ? elementSet = new ElementSet<>(entries, this) : result;
  }

  @Override
  Entry<E> getEntry(int index) {
    return entries.get(index);
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  public int size() {
