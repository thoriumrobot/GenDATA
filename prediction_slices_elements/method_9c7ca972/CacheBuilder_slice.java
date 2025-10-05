// Source-based slice around line 1077
// Method: <com.google.common.cache.CacheBuilder: String toString()>

      }
    }
  }

  /**
   * Returns a string representation for this CacheBuilder instance. The exact form of the returned
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
    if (maximumSize != UNSET_INT) {
      s.add("maximumSize", maximumSize);
    }
