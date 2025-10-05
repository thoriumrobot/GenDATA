// Source-based slice around line 405
// Method: com.google.common.collect.ImmutableRangeSet.lazyComplement

    } else if (ranges.isEmpty()) {
      return all();
    } else if (ranges.size() == 1 && ranges.get(0).equals(Range.all())) {
      return of();
    } else {
      return lazyComplement();
    }
  }

  @LazyInit @RetainedWith private transient @Nullable ImmutableRangeSet<C> lazyComplement;

  private ImmutableRangeSet<C> lazyComplement() {
    ImmutableRangeSet<C> result = lazyComplement;
    return result == null
        ? lazyComplement =
            new ImmutableRangeSet<>(new ComplementRanges<>(ranges), /* complement= */ this)
        : result;
  }

  /**
