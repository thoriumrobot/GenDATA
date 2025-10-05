// Source-based slice around line 93
// Method: <com.google.common.collect.AbstractRangeSet: int hashCode()>

      return true;
    } else if (obj instanceof RangeSet) {
      RangeSet<?> other = (RangeSet<?>) obj;
      return this.asRanges().equals(other.asRanges());
    }
    return false;
  }

  @Override
  public final int hashCode() {
    return asRanges().hashCode();
  }

  @Override
  public final String toString() {
    return asRanges().toString();
  }
}
