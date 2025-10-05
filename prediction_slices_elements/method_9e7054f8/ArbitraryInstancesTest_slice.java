// Source-based slice around line 144
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_primitives()>


/**
 * Unit test for {@link ArbitraryInstances}.
 *
 * @author Ben Yu
 */
@NullUnmarked
public class ArbitraryInstancesTest extends TestCase {

  public void testGet_primitives() {
    assertNull(ArbitraryInstances.get(void.class));
    assertNull(ArbitraryInstances.get(Void.class));
    assertEquals(Boolean.FALSE, ArbitraryInstances.get(boolean.class));
    assertEquals(Boolean.FALSE, ArbitraryInstances.get(Boolean.class));
    assertEquals(Character.valueOf('\0'), ArbitraryInstances.get(char.class));
    assertEquals(Character.valueOf('\0'), ArbitraryInstances.get(Character.class));
    assertEquals(Byte.valueOf((byte) 0), ArbitraryInstances.get(byte.class));
    assertEquals(Byte.valueOf((byte) 0), ArbitraryInstances.get(Byte.class));
    assertEquals(Short.valueOf((short) 0), ArbitraryInstances.get(short.class));
    assertEquals(Short.valueOf((short) 0), ArbitraryInstances.get(Short.class));
