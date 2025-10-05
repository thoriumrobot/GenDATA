// Source-based slice around line 108
// Method: <com.google.common.collect.ImmutableMapValues: void forEach(Consumer)>

      @GwtIncompatible
            Object writeReplace() {
        return super.writeReplace();
      }
    };
  }

  @GwtIncompatible
  @Override
  public void forEach(Consumer<? super V> action) {
    checkNotNull(action);
    map.forEach((k, v) -> action.accept(v));
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
  @J2ktIncompatible
  @GwtIncompatible
    Object writeReplace() {
