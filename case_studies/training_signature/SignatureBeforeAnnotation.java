// Signature - BEFORE adding annotations (has warnings)
// This file shows the state BEFORE the model adds annotations
// Running the checker on this file produces warnings

import org.checkerframework.checker.signature.qual.BinaryName;
import org.checkerframework.checker.signature.qual.FullyQualifiedName;

public class SignatureBeforeAnnotation {
    
    // Entry points (annotated) - these stay annotated
    public Class<?> loadClass(@BinaryName String className) throws ClassNotFoundException {
        return Class.forName(className);
    }
    
    public void registerType(@FullyQualifiedName String typeName) {
        System.out.println(typeName);
    }
    
    // UNANNOTATED - will cause warning
    @BinaryName
    String className1 = "java.lang.String";
    
    // UNANNOTATED return type - will cause warning
    public String getTypeName() { 
        return "com.example.MyClass"; 
    }
    
    public void test() throws ClassNotFoundException {
        loadClass(className1);       // WARNING: incompatible argument
        registerType(getTypeName()); // WARNING: incompatible argument
    }
}
