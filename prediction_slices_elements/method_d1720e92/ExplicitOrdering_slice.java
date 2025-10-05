// Source-based slice around line 62
// Method: <com.google.common.collect.ExplicitOrdering: int hashCode()>

  public boolean equals(@Nullable Object object) {
    if (object instanceof ExplicitOrdering) {
      ExplicitOrdering<?> that = (ExplicitOrdering<?>) object;
      return this.rankMap.equals(that.rankMap);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return rankMap.hashCode();
  }

  @Override
  public String toString() {
    return "Ordering.explicit(" + rankMap.keySet() + ")";
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
