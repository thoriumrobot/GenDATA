package org.checkerframework.specimin;

import java.io.IOException;
import org.junit.Test;

/**
 * This test checks that if there are two targets that reference different classes with the same
 * simple name (in different packages), Specimin correctly creates two different synthetic files.
 */
public class TwoTargetsThatUseSameSimpleNameTest {
  @Test
  public void runTest() throws IOException {
    SpeciminTestExecutor.runTestWithoutJarPaths(
        "twotargetsthatusesamesimplename",
        new String[] {"com/example/Foo.java", "com/example/Bar.java"},
        new String[] {"com.example.Foo#test(Node)", "com.example.Bar#test(Node)"});
  }
}
