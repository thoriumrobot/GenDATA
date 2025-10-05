// Source-based slice around line 100
// Method: com.google.common.base.Absent.serialVersionUID

  @Override
  public String toString() {
    return "Optional.absent()";
  }

  private Object readResolve() {
    return INSTANCE;
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
