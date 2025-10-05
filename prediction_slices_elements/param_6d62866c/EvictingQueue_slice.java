// Source-based slice around line 119
// Method: <com.google.common.collect.EvictingQueue: boolean addAll(Collection)>

    if (size() == maxSize) {
      delegate.remove();
    }
    delegate.add(e);
    return true;
  }

  @Override
  @CanIgnoreReturnValue
  public boolean addAll(Collection<? extends E> collection) {
    int size = collection.size();
    if (size >= maxSize) {
      clear();
      return Iterables.addAll(this, Iterables.skip(collection, size - maxSize));
    }
    return standardAddAll(collection);
  }

  @Override
  @J2ktIncompatible // Incompatible return type change. Use inherited implementation
