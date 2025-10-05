// Source-based slice around line 238
// Method: <com.google.common.collect.RegularContiguousSet: int hashCode()>

      if (this.domain.equals(that.domain)) {
        return this.first().equals(that.first()) && this.last().equals(that.last());
      }
    }
    return super.equals(object);
  }

  // copied to make sure not to use the GWT-emulated version
  @Override
  public int hashCode() {
    return Sets.hashCodeImpl(this);
  }

  @GwtIncompatible
  @J2ktIncompatible
  private static final class SerializedForm<C extends Comparable> implements Serializable {
    final Range<C> range;
    final DiscreteDomain<C> domain;

    private SerializedForm(Range<C> range, DiscreteDomain<C> domain) {
