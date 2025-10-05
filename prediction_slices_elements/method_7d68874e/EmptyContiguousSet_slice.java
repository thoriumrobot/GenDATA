// Source-based slice around line 144
// Method: <com.google.common.collect.EmptyContiguousSet: int hashCode()>

  }

  @GwtIncompatible // not used in GWT
  @Override
  boolean isHashCodeFast() {
    return true;
  }

  @Override
  public int hashCode() {
    return 0;
  }

  @GwtIncompatible
  @J2ktIncompatible
  private static final class SerializedForm<C extends Comparable> implements Serializable {
    private final DiscreteDomain<C> domain;

    private SerializedForm(DiscreteDomain<C> domain) {
      this.domain = domain;
