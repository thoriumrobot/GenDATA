package org.checkerframework.specimin;

import java.io.IOException;
import org.junit.Test;

/**
 * This test checks the case where there are two wildcard imports: unsolved classes are placed in
 * the first one (this is an arbitrary choice, but Specimin has no way to do better without access
 * to the classpath).
 */
public class WildcardImport2Test {
  @Test
  public void runTest() throws IOException {
    SpeciminTestExecutor.runTestWithoutJarPaths(
        "wildcardimport2",
        new String[] {"com/example/Simple.java"},
        new String[] {"com.example.Simple#bar()"});
  }
}
