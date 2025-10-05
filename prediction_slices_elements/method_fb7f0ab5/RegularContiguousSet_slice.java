// Source-based slice around line 91
// Method: <com.google.common.collect.RegularContiguousSet: UnmodifiableIterator iterator()>

    }
    // The cast is safe because of the contains check—at least for any reasonable Comparable class.
    @SuppressWarnings("unchecked")
    // requireNonNull is safe because of the contains check.
    C c = (C) requireNonNull(target);
    return (int) domain.distance(first(), c);
  }

  @Override
  public UnmodifiableIterator<C> iterator() {
    return new AbstractSequentialIterator<C>(first()) {
      final C last = last();

      @Override
      protected @Nullable C computeNext(C previous) {
        return equalsOrThrow(previous, last) ? null : domain.next(previous);
      }
    };
  }

