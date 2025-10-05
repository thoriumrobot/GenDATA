// Source-based slice around line 118
// Method: <com.google.common.collect.ImmutableMapValues: Object writeReplace()>

    checkNotNull(action);
    map.forEach((k, v) -> action.accept(v));
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
  @J2ktIncompatible
  @GwtIncompatible
    Object writeReplace() {
    return super.writeReplace();
  }

  @GwtIncompatible
  @J2ktIncompatible
  /*
   * The mainline copy of ImmutableMapValues doesn't produce this serialized form anymore, though
   * the backport does. For now, we're keeping the class declaration in *both* flavors so that both
   * flavors can read old data or data from the other flavor. However, we strongly discourage
   * relying on this, as we have made incompatible changes to serialized forms in the past and
