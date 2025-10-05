// Source-based slice around line 91
// Method: <com.google.common.base.Present: int hashCode()>

  public boolean equals(@Nullable Object obj) {
    if (obj instanceof Present) {
      Present<?> other = (Present<?>) obj;
      return reference.equals(other.reference);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return 0x598df91c + reference.hashCode();
  }

  @Override
  public String toString() {
    return "Optional.of(" + reference + ")";
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
