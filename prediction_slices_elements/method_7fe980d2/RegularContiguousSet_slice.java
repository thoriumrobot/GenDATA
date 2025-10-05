// Source-based slice around line 60
// Method: <com.google.common.collect.RegularContiguousSet: ContiguousSet subSetImpl(C,boolean,C,boolean)>

  }

  @Override
  ContiguousSet<C> headSetImpl(C toElement, boolean inclusive) {
    return intersectionInCurrentDomain(Range.upTo(toElement, BoundType.forBoolean(inclusive)));
  }

  @Override
  @SuppressWarnings("unchecked") // TODO(cpovirk): Use a shared unsafeCompare method.
  ContiguousSet<C> subSetImpl(
      C fromElement, boolean fromInclusive, C toElement, boolean toInclusive) {
    if (fromElement.compareTo(toElement) == 0 && !fromInclusive && !toInclusive) {
      // Range would reject our attempt to create (x, x).
      return new EmptyContiguousSet<>(domain);
    }
    return intersectionInCurrentDomain(
        Range.range(
            fromElement, BoundType.forBoolean(fromInclusive),
            toElement, BoundType.forBoolean(toInclusive)));
  }
