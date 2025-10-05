// Source-based slice around line 85
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: Entry lastEntry()>

    }
  }

  @Override
  public @Nullable Entry<E> firstEntry() {
    return isEmpty() ? null : getEntry(0);
  }

  @Override
  public @Nullable Entry<E> lastEntry() {
    return isEmpty() ? null : getEntry(length - 1);
  }

  @Override
  public int count(@Nullable Object element) {
    int index = elementSet.indexOf(element);
    return (index >= 0) ? getCount(index) : 0;
  }

  @Override
