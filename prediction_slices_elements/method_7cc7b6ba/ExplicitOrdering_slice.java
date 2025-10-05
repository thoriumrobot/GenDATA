// Source-based slice around line 67
// Method: <com.google.common.collect.ExplicitOrdering: String toString()>

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
