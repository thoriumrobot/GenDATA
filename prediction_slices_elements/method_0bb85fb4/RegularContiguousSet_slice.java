// Source-based slice around line 104
// Method: <com.google.common.collect.RegularContiguousSet: UnmodifiableIterator descendingIterator()>

      @Override
      protected @Nullable C computeNext(C previous) {
        return equalsOrThrow(previous, last) ? null : domain.next(previous);
      }
    };
  }

  @GwtIncompatible // NavigableSet
  @Override
  public UnmodifiableIterator<C> descendingIterator() {
    return new AbstractSequentialIterator<C>(last()) {
      final C first = first();

      @Override
      protected @Nullable C computeNext(C previous) {
        return equalsOrThrow(previous, first) ? null : domain.previous(previous);
      }
    };
  }

