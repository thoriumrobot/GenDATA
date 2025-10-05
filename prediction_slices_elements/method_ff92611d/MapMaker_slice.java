// Source-based slice around line 295
// Method: <com.google.common.collect.MapMaker: String toString()>

    }
    return MapMakerInternalMap.create(this);
  }

  /**
   * Returns a string representation for this MapMaker instance. The exact form of the returned
   * string is not specified.
   */
  @Override
  public String toString() {
    MoreObjects.ToStringHelper s = MoreObjects.toStringHelper(this);
    if (initialCapacity != UNSET_INT) {
      s.add("initialCapacity", initialCapacity);
    }
    if (concurrencyLevel != UNSET_INT) {
      s.add("concurrencyLevel", concurrencyLevel);
    }
    if (keyStrength != null) {
      s.add("keyStrength", Ascii.toLowerCase(keyStrength.toString()));
    }
