// Source-based slice around line 66
// Method: <com.google.common.collect.SingletonImmutableSet: boolean isPartialView()>

    return singletonIterator(element);
  }

  @Override
  public ImmutableList<E> asList() {
    return ImmutableList.of(element);
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  int copyIntoArray(@Nullable Object[] dst, int offset) {
    dst[offset] = element;
    return offset + 1;
  }

  @Override
