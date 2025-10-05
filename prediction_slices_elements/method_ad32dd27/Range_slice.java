// Source-based slice around line 673
// Method: <com.google.common.collect.Range: int hashCode()>

    if (object instanceof Range) {
      Range<?> other = (Range<?>) object;
      return lowerBound.equals(other.lowerBound) && upperBound.equals(other.upperBound);
    }
    return false;
  }

  /** Returns a hash code for this range. */
  @Override
  public int hashCode() {
    return lowerBound.hashCode() * 31 + upperBound.hashCode();
  }

  /**
   * Returns a string representation of this range, such as {@code "[3..5)"} (other examples are
   * listed in the class documentation).
   */
  @Override
  public String toString() {
    return toString(lowerBound, upperBound);
