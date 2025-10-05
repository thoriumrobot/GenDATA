// Source-based slice around line 40
// Method: <com.google.common.collect.UsingToStringOrdering: String toString()>

    return left.toString().compareTo(right.toString());
  }

  // preserve singleton-ness, so equals() and hashCode() work correctly
  private Object readResolve() {
    return INSTANCE;
  }

  @Override
  public String toString() {
    return "Ordering.usingToString()";
  }

  private UsingToStringOrdering() {}

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
