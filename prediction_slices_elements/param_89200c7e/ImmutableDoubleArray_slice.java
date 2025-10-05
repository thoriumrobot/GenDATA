// Source-based slice around line 454
// Method: <com.google.common.primitives.ImmutableDoubleArray: ImmutableDoubleArray subArray(int,int)>

  }

  /**
   * Returns a new immutable array containing the values in the specified range.
   *
   * <p><b>Performance note:</b> The returned array has the same full memory footprint as this one
   * does (no actual copying is performed). To reduce memory usage, use {@code subArray(start,
   * end).trimmed()}.
   */
  public ImmutableDoubleArray subArray(int startIndex, int endIndex) {
    Preconditions.checkPositionIndexes(startIndex, endIndex, length());
    return startIndex == endIndex
        ? EMPTY
        : new ImmutableDoubleArray(array, start + startIndex, start + endIndex);
  }

  /*
   * We declare this as package-private, rather than private, to avoid generating a synthetic
   * accessor method (under -target 8) that would lack the Android flavor's @IgnoreJRERequirement.
   */
