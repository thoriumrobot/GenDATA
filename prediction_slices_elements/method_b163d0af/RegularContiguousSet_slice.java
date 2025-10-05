// Source-based slice around line 79
// Method: <com.google.common.collect.RegularContiguousSet: int indexOf(Object)>

  }

  @Override
  ContiguousSet<C> tailSetImpl(C fromElement, boolean inclusive) {
    return intersectionInCurrentDomain(Range.downTo(fromElement, BoundType.forBoolean(inclusive)));
  }

  @GwtIncompatible // not used by GWT emulation
  @Override
  int indexOf(@Nullable Object target) {
    if (!contains(target)) {
      return -1;
    }
    // The cast is safe because of the contains check—at least for any reasonable Comparable class.
    @SuppressWarnings("unchecked")
    // requireNonNull is safe because of the contains check.
    C c = (C) requireNonNull(target);
    return (int) domain.distance(first(), c);
  }

