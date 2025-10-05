// Source-based slice around line 588
// Method: <com.google.common.primitives.ImmutableIntArray: int hashCode()>

      if (this.get(i) != that.get(i)) {
        return false;
      }
    }
    return true;
  }

  /** Returns an unspecified hash code for the contents of this immutable array. */
  @Override
  public int hashCode() {
    int hash = 1;
    for (int i = start; i < end; i++) {
      hash *= 31;
      hash += Integer.hashCode(array[i]);
    }
    return hash;
  }

  /**
   * Returns a string representation of this array in the same form as {@link
