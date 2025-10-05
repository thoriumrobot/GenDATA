// Source-based slice around line 66
// Method: <com.google.common.collect.Cut: Cut canonical(DiscreteDomain)>


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
  @Override
  public int compareTo(Cut<C> that) {
    if (that == belowAll()) {
      return 1;
    }
    if (that == aboveAll()) {
