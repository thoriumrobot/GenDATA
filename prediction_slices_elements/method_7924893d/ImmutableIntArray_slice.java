// Source-based slice around line 567
// Method: <com.google.common.primitives.ImmutableIntArray: boolean equals(Object)>

      return parent.toString();
    }
  }

  /**
   * Returns {@code true} if {@code object} is an {@code ImmutableIntArray} containing the same
   * values as this one, in the same order.
   */
  @Override
  public boolean equals(@Nullable Object object) {
    if (object == this) {
      return true;
    }
    if (!(object instanceof ImmutableIntArray)) {
      return false;
    }
    ImmutableIntArray that = (ImmutableIntArray) object;
    if (this.length() != that.length()) {
      return false;
    }
