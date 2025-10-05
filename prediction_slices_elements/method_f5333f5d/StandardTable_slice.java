// Source-based slice around line 198
// Method: <com.google.common.collect.StandardTable: boolean removeMapping(Object,Object,Object)>

    return output;
  }

  private boolean containsMapping(
      @Nullable Object rowKey, @Nullable Object columnKey, @Nullable Object value) {
    return value != null && value.equals(get(rowKey, columnKey));
  }

  /** Remove a row key / column key / value mapping, if present. */
  private boolean removeMapping(
      @Nullable Object rowKey, @Nullable Object columnKey, @Nullable Object value) {
    if (containsMapping(rowKey, columnKey, value)) {
      remove(rowKey, columnKey);
      return true;
    }
    return false;
  }

  // Views

