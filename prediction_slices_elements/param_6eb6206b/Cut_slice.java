// Source-based slice around line 50
// Method: <com.google.common.collect.Cut: Cut withLowerBoundType(BoundType,DiscreteDomain)>

    this.endpoint = endpoint;
  }

  abstract boolean isLessThan(C value);

  abstract BoundType typeAsLowerBound();

  abstract BoundType typeAsUpperBound();

  abstract Cut<C> withLowerBoundType(BoundType boundType, DiscreteDomain<C> domain);

  abstract Cut<C> withUpperBoundType(BoundType boundType, DiscreteDomain<C> domain);

  abstract void describeAsLowerBound(StringBuilder sb);

  abstract void describeAsUpperBound(StringBuilder sb);

  abstract @Nullable C leastValueAbove(DiscreteDomain<C> domain);

  abstract @Nullable C greatestValueBelow(DiscreteDomain<C> domain);
