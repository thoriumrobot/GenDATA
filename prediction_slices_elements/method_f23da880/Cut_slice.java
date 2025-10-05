// Source-based slice around line 309
// Method: <com.google.common.collect.Cut: Cut belowValue(C)>

    }

    private Object readResolve() {
      return INSTANCE;
    }

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
  }

  static <C extends Comparable> Cut<C> belowValue(C endpoint) {
    return new BelowValue<>(endpoint);
  }

  private static final class BelowValue<C extends Comparable> extends Cut<C> {
    BelowValue(C endpoint) {
      super(checkNotNull(endpoint));
    }

    @Override
    boolean isLessThan(C value) {
