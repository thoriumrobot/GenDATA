// Source-based slice around line 56
// Method: <com.google.common.collect.Cut: void describeAsUpperBound(StringBuilder)>


  abstract BoundType typeAsUpperBound();

  abstract Cut<C> withLowerBoundType(BoundType boundType, DiscreteDomain<C> domain);

  abstract Cut<C> withUpperBoundType(BoundType boundType, DiscreteDomain<C> domain);

  abstract void describeAsLowerBound(StringBuilder sb);

  abstract void describeAsUpperBound(StringBuilder sb);

  abstract @Nullable C leastValueAbove(DiscreteDomain<C> domain);

  abstract @Nullable C greatestValueBelow(DiscreteDomain<C> domain);

  /*
   * The canonical form is a BelowValue cut whenever possible, otherwise ABOVE_ALL, or
   * (only in the case of types that are unbounded below) BELOW_ALL.
   */
  Cut<C> canonical(DiscreteDomain<C> domain) {
