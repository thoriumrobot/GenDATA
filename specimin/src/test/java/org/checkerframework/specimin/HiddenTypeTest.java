package org.checkerframework.specimin;

import java.io.IOException;
import org.junit.Test;

/**
 * This test checks that if the targetted method in a Java file is called by an unsolved method,
 * meaning that the source file of the caller method is not in the root directory, Specimin can
 * create appropriate synthetic files so that the output is compilable
 */
public class HiddenTypeTest {
  @Test
  public void runTest() throws IOException {
    SpeciminTestExecutor.runTestWithoutJarPaths(
        "hiddenType",
        new String[] {"com/example/Simple.java"},
        new String[] {"com.example.Simple#isVoidType(MethodDeclaration)"});
  }
}
