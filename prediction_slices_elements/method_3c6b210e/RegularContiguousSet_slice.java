// Source-based slice around line 212
// Method: <com.google.common.collect.RegularContiguousSet: Range range()>

      C lowerEndpoint = Ordering.<C>natural().max(this.first(), other.first());
      C upperEndpoint = Ordering.<C>natural().min(this.last(), other.last());
      return (lowerEndpoint.compareTo(upperEndpoint) <= 0)
          ? ContiguousSet.create(Range.closed(lowerEndpoint, upperEndpoint), domain)
          : new EmptyContiguousSet<C>(domain);
    }
  }

  @Override
  public Range<C> range() {
    return range(CLOSED, CLOSED);
  }

  @Override
  public Range<C> range(BoundType lowerBoundType, BoundType upperBoundType) {
    return Range.create(
        range.lowerBound.withLowerBoundType(lowerBoundType, domain),
        range.upperBound.withUpperBoundType(upperBoundType, domain));
  }

