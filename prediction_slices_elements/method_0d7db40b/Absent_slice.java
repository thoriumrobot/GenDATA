// Source-based slice around line 92
// Method: <com.google.common.base.Absent: String toString()>

    return this == obj;
  }

  @Override
  public int hashCode() {
    return 0x79a31aac;
  }

  @Override
  public String toString() {
    return "Optional.absent()";
  }

  private Object readResolve() {
    return INSTANCE;
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
