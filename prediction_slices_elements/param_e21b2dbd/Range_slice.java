// Source-based slice around line 485
// Method: <com.google.common.collect.Range: boolean encloses(Range)>

   *
   * <p>Note that if {@code a.encloses(b)}, then {@code b.contains(v)} implies {@code
   * a.contains(v)}, but as the last two examples illustrate, the converse is not always true.
   *
   * <p>Being reflexive, antisymmetric and transitive, the {@code encloses} relation defines a
   * <i>partial order</i> over ranges. There exists a unique {@linkplain Range#all maximal} range
   * according to this relation, and also numerous {@linkplain #isEmpty minimal} ranges. Enclosure
   * also implies {@linkplain #isConnected connectedness}.
   */
  public boolean encloses(Range<C> other) {
    return lowerBound.compareTo(other.lowerBound) <= 0
        && upperBound.compareTo(other.upperBound) >= 0;
  }

  /**
   * Returns {@code true} if there exists a (possibly empty) range which is {@linkplain #encloses
   * enclosed} by both this range and {@code other}.
   *
   * <p>For example,
   *
