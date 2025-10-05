// Source-based slice around line 78
// Method: <com.google.common.collect.testing.SafeTreeSet: boolean addAll(Collection)>

    }
  }

  @Override
  public boolean add(E element) {
    return delegate.add(checkValid(element));
  }

  @Override
  public boolean addAll(Collection<? extends E> collection) {
    for (E e : collection) {
      checkValid(e);
    }
    return delegate.addAll(collection);
  }

  @Override
  public @Nullable E ceiling(E e) {
    return delegate.ceiling(checkValid(e));
  }
