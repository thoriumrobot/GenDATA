// Source-based slice around line 52
// Method: <com.google.common.collect.testing.ReserializedSafeTreeMapMapInterfaceTest: Integer getValueNotInPopulatedMap()>

    return SerializableTester.reserialize(map);
  }

  @Override
  protected String getKeyNotInPopulatedMap() {
    return "minus one";
  }

  @Override
  protected Integer getValueNotInPopulatedMap() {
    return -1;
  }
}
