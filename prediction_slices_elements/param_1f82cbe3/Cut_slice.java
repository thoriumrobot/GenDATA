// Source-based slice around line 60
// Method: <com.google.common.collect.Cut: C greatestValueBelow(DiscreteDomain)>


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
    return this;
  }

  // note: overridden by {BELOW,ABOVE}_ALL
