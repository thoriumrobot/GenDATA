// Source-based slice around line 32
// Method: <com.google.common.collect.testing.ReserializedSafeTreeMapMapInterfaceTest: SortedMap makePopulatedMap()>


@GwtIncompatible // SerializableTester
public class ReserializedSafeTreeMapMapInterfaceTest
    extends SortedMapInterfaceTest<String, Integer> {
  public ReserializedSafeTreeMapMapInterfaceTest() {
    super(false, true, true, true, true);
  }

  @Override
  protected SortedMap<String, Integer> makePopulatedMap() {
    NavigableMap<String, Integer> map = new SafeTreeMap<>();
    map.put("one", 1);
    map.put("two", 2);
    map.put("three", 3);
    return SerializableTester.reserialize(map);
  }

  @Override
  protected SortedMap<String, Integer> makeEmptyMap() throws UnsupportedOperationException {
    NavigableMap<String, Integer> map = new SafeTreeMap<>();
