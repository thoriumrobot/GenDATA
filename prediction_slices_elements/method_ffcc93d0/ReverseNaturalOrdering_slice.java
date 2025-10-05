// Source-based slice around line 96
// Method: <com.google.common.collect.ReverseNaturalOrdering: String toString()>

    return NaturalOrdering.INSTANCE.min(iterable);
  }

  // preserving singleton-ness gives equals()/hashCode() for free
  private Object readResolve() {
    return INSTANCE;
  }

  @Override
  public String toString() {
    return "Ordering.natural().reverse()";
  }

  private ReverseNaturalOrdering() {}

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
