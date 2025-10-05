// Source-based slice around line 390
// Method: <com.google.common.collect.Cut: Cut aboveValue(C)>


    @Override
    public String toString() {
      return "\\" + endpoint + "/";
    }

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
  }

  static <C extends Comparable> Cut<C> aboveValue(C endpoint) {
    return new AboveValue<>(endpoint);
  }

  private static final class AboveValue<C extends Comparable> extends Cut<C> {
    AboveValue(C endpoint) {
      super(checkNotNull(endpoint));
    }

    @Override
    boolean isLessThan(C value) {
