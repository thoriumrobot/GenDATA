// Source-based slice around line 122
// Method: <com.google.common.collect.RegularImmutableSet: int hashCode()>

    return (table.length == 0) ? ImmutableList.of() : new RegularImmutableAsList<>(this, elements);
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  boolean isHashCodeFast() {
    return true;
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
