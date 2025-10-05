// Source-based slice around line 572
// Method: <com.google.common.primitives.ImmutableDoubleArray: boolean equals(Object)>

      return parent.toString();
    }
  }

  /**
   * Returns {@code true} if {@code object} is an {@code ImmutableDoubleArray} containing the same
   * values as this one, in the same order. Values are compared as if by {@link Double#equals}.
   */
  @Override
  public boolean equals(@Nullable Object object) {
    if (object == this) {
      return true;
    }
    if (!(object instanceof ImmutableDoubleArray)) {
      return false;
    }
    ImmutableDoubleArray that = (ImmutableDoubleArray) object;
    if (this.length() != that.length()) {
      return false;
    }
