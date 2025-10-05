// Source-based slice around line 569
// Method: <com.google.common.primitives.ImmutableLongArray: boolean equals(Object)>

      return parent.toString();
    }
  }

  /**
   * Returns {@code true} if {@code object} is an {@code ImmutableLongArray} containing the same
   * values as this one, in the same order.
   */
  @Override
  public boolean equals(@Nullable Object object) {
    if (object == this) {
      return true;
    }
    if (!(object instanceof ImmutableLongArray)) {
      return false;
    }
    ImmutableLongArray that = (ImmutableLongArray) object;
    if (this.length() != that.length()) {
      return false;
    }
