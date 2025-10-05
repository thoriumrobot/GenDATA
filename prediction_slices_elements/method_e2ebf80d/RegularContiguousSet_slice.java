// Source-based slice around line 73
// Method: <com.google.common.collect.RegularContiguousSet: ContiguousSet tailSetImpl(C,boolean)>

      return new EmptyContiguousSet<>(domain);
    }
    return intersectionInCurrentDomain(
        Range.range(
            fromElement, BoundType.forBoolean(fromInclusive),
            toElement, BoundType.forBoolean(toInclusive)));
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
