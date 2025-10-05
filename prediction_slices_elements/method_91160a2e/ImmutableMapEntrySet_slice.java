// Source-based slice around line 114
// Method: <com.google.common.collect.ImmutableMapEntrySet: boolean isPartialView()>

    if (object instanceof Entry) {
      Entry<?, ?> entry = (Entry<?, ?>) object;
      V value = map().get(entry.getKey());
      return value != null && value.equals(entry.getValue());
    }
    return false;
  }

  @Override
  boolean isPartialView() {
    return map().isPartialView();
  }

  @Override
  @GwtIncompatible // not used in GWT
  boolean isHashCodeFast() {
    return map().isHashCodeFast();
  }

  @Override
