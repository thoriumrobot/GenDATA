// Source-based slice around line 648
// Method: <com.google.common.collect.Range: Range canonical(DiscreteDomain)>

   * canonical forms:
   *
   * <ul>
   *   <li>[start..end)
   *   <li>[start..+∞)
   *   <li>(-∞..end) (only if type {@code C} is unbounded below)
   *   <li>(-∞..+∞) (only if type {@code C} is unbounded below)
   * </ul>
   */
  public Range<C> canonical(DiscreteDomain<C> domain) {
    checkNotNull(domain);
    Cut<C> lower = lowerBound.canonical(domain);
    Cut<C> upper = upperBound.canonical(domain);
    return (lower == lowerBound && upper == upperBound) ? this : create(lower, upper);
  }

  /**
   * Returns {@code true} if {@code object} is a range having the same endpoints and bound types as
   * this range. Note that discrete ranges such as {@code (1..4)} and {@code [2..3]} are <b>not</b>
   * equal to one another, despite the fact that they each contain precisely the same set of values.
