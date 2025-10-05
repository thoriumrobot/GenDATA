// Source-based slice around line 69
// Method: <com.google.common.collect.ImmutableMapKeySet: void forEach(Consumer)>

    return map.containsKey(object);
  }

  @Override
  K get(int index) {
    return map.entrySet().asList().get(index).getKey();
  }

  @Override
  public void forEach(Consumer<? super K> action) {
    checkNotNull(action);
    map.forEach((k, v) -> action.accept(k));
  }

  @Override
  boolean isPartialView() {
    return true;
  }

  // redeclare to help optimizers with b/310253115
