// Source-based slice around line 71
// Method: com.google.common.collect.JdkBackedImmutableMultiset.elementSet

    this.entries = entries;
    this.size = size;
  }

  @Override
  public int count(@Nullable Object element) {
    return delegateMap.getOrDefault(element, 0);
  }

  @LazyInit private transient @Nullable ImmutableSet<E> elementSet;

  @Override
  public ImmutableSet<E> elementSet() {
    ImmutableSet<E> result = elementSet;
    return (result == null) ? elementSet = new ElementSet<>(entries, this) : result;
  }

  @Override
  Entry<E> getEntry(int index) {
    return entries.get(index);
