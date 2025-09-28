package org.checkerframework.specimin;

import java.io.IOException;
import org.junit.Test;

/** This test checks that enums with fields don't cause a crash */
public class EnumWithFieldTest {
  @Test
  public void runTest() throws IOException {
    SpeciminTestExecutor.runTestWithoutJarPaths(
        "enumwithfield",
        new String[] {"com/example/Simple.java"},
        new String[] {"com.example.Simple#bar()"});
  }
}
